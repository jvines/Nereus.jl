# sample_nuts checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any iteration and continued with `resume =
# true` is bit-identical to one that ran straight through, so a run that has not
# mixed can simply be extended. With n_warmup = 200 Stan's schedule is: init
# buffer 1-75 (step size only), metric windows 76-100 and 101-150 (Welford
# accumulators, the metric updated and both adaptors reset at 100 and 150), term
# buffer 151-200, and the step size finalised at 200. The stops below cross all
# of it -- 40, 90, 100, 150, 160, 200 -- plus the first and a later sampling
# iteration (201, 230) and a finished run extended past its own end. Chains stop
# at different iterations in one run, as a kill leaves them; the test hook
# `_NUTS_STOP_AFTER` is what stops them, since no `n_samples` ends a run inside
# the warm-up.
using Test
using Nereus
using Random
using Logging
using Statistics: median

# An eccentric orbit with Mo = 0 at the reference epoch (the median time): its
# posterior straddles Mo's 0/2π seam, so the warm start moves Mo's window, which
# a resume must then put back. The pre-search is long enough (1200 steps) to
# find the mode: it moved the window for every seed tried, where 200 steps moved
# it for some and not others.
const _NR_N = 60
const _NR_T, _NR_RV = let r = MersenneTwister(7)
    t = sort!(100 .* rand(r, _NR_N))
    t, [Nereus.rv_keplerian(ti, 4.23, 40.0, 0.4, 1.0, 0.0, median(t)) for ti in t] .+
       1.5 .* randn(r, _NR_N)
end

# A fresh target per run: the warm start moves circular windows on the target.
_nr_target(rv = _NR_RV) = build_target(
    planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _NR_T, rv = rv, rv_err = fill(1.5, _NR_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))

const _NR_KW = (n_warmup = 200, n_chains = 4, warm_temps = 4, warm_walkers = 20,
                warm_steps = 1200, warm_burnin = 600, progress = false)

# `seed` seeds the `rng` passed in; a resumed run passes a different one, which
# must come out as the uninterrupted run left its own.
function _nr_run(n_samples; stop = Int[], seed = 3, rv = _NR_RV, kw...)
    tg = _nr_target(rv)
    rng = MersenneTwister(seed)
    Nereus._NUTS_STOP_AFTER[] = stop
    try
        # Errors only: the small warm-start pre-search warns about its evidence.
        ch = with_logger(ConsoleLogger(stderr, Logging.Error)) do
            sample_nuts(tg; _NR_KW..., n_samples, rng, kw...)
        end
        return (; ch, rng, windows = Nereus.circular_windows(tg.params))
    finally
        Nereus._NUTS_STOP_AFTER[] = Int[]
    end
end

const _NR_INFO = (:n_divergent, :step_size, :mean_tree_depth, :max_tree_depth, :mean_accept)
_nr_same(a, b) = a.ch.value.data == b.ch.value.data &&
                 all(f -> isequal(getproperty(a.ch.info, f), getproperty(b.ch.info, f)), _NR_INFO) &&
                 a.rng == b.rng && a.windows == b.windows

@testset "sample_nuts resume" begin
    full = _nr_run(150)
    # The warm start moved a circular window, so a resume has one to put back.
    @test full.windows != Nereus.circular_windows(_nr_target().params)

    path = joinpath(mktempdir(), "nuts_state.jls")
    # Killed in the init buffer, a metric window, at a window split and at the
    # last warm-up iteration; resumed, and killed again in the term buffer, at
    # the second split, at the first sampling iteration and later in sampling.
    @test_throws Nereus._NutsStopped _nr_run(150; checkpoint = path, stop = [40, 90, 100, 200])
    @test isfile(path) && !isfile(path * ".tmp")
    @test_throws Nereus._NutsStopped _nr_run(150; checkpoint = path, resume = true, seed = 99,
                                             stop = [160, 150, 201, 230])
    mid = _nr_run(100; checkpoint = path, resume = true, seed = 99)   # a finished run ...
    ext = _nr_run(150; checkpoint = path, resume = true, seed = 99)   # ... extended

    # Identical to the uninterrupted run, bit for bit: draws, every diagnostic
    # in chains.info, the caller's rng afterwards and the windows.
    @test ext.ch.value.data == full.ch.value.data
    for f in _NR_INFO
        @test isequal(getproperty(ext.ch.info, f), getproperty(full.ch.info, f))
    end
    @test isequal(nuts_diagnostics(ext.ch), nuts_diagnostics(full.ch))
    @test ext.rng == full.rng
    @test ext.windows == full.windows
    # The finished 100-sample run kept the first 100 draws of each chain.
    @test mid.ch.value.data == full.ch.value.data[1:100, :, :]

    # Resuming at the checkpoint's own iteration runs nothing and returns the same.
    @test _nr_same(_nr_run(150; checkpoint = path, resume = true, seed = 99), full)

    # Refusals: no checkpoint, a different run, a total shorter than the saved.
    @test_throws ArgumentError _nr_run(150; resume = true)
    @test_throws ArgumentError _nr_run(150; checkpoint = path * ".none", resume = true)
    err = try _nr_run(200; checkpoint = path, resume = true, target_accept = 0.9); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("target_accept", err.msg)
    err = try _nr_run(200; checkpoint = path, resume = true, n_warmup = 300); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("n_warmup", err.msg)
    err = try _nr_run(200; checkpoint = path, resume = true, rv = _NR_RV .+ 1.0); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("data", err.msg)
    @test_throws ArgumentError _nr_run(100; checkpoint = path, resume = true)
end

@testset "sample_nuts resume, one chain on the caller's rng" begin
    # A single chain runs on `rng` itself, which the resume restores in place.
    full = _nr_run(60; n_chains = 1)
    path = joinpath(mktempdir(), "nuts_state.jls")
    @test_throws Nereus._NutsStopped _nr_run(60; n_chains = 1, checkpoint = path, stop = [120])
    @test _nr_same(_nr_run(60; n_chains = 1, checkpoint = path, resume = true, seed = 99), full)
end
