# sample_nuts checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any iteration and continued with `resume =
# true` is bit-identical to one that ran straight through, so a run that has not
# mixed can simply be extended. With n_warmup = 200 Stan's schedule is: init
# buffer 1-75 (step size only), metric windows 76-100 and 101-150 (Welford
# accumulators, the metric updated and both adaptors reset at 100 and 150), term
# buffer 151-200, and the step size finalised at 200. The chains re-cut the
# circular seams together after iterations 75, 100 and 150 (`_hmc_recut!`), and
# none passes one of those before all have reached it: a chain that stops
# before one leaves the others saved at it, the re-cut pending. The stops below
# cross all of it -- 40, 75 (pending), 150, 160 -- plus the first and a later
# sampling iteration (201, 230) and a finished run extended past its own end.
# Chains stop at different iterations in one run, as a kill leaves them; the
# test hook `_NUTS_STOP_AFTER` is what stops them, since no `n_samples` ends a
# run inside the warm-up. The last testset kills a real run, in a child
# process.
using Test
using Nereus
using Random
using Logging
using Serialization

include(joinpath(@__DIR__, "fixtures", "nuts_resume_target.jl"))

# A run killed while one of its chains is still queued for a thread. Chain tasks
# never yield, so with more chains than threads a chain can wait until another
# has finished; every checkpoint written meanwhile must still hold it, or a kill
# in that time leaves none. One default thread and two chains make that wait
# certain: the child runs them one after the other, writes a checkpoint after
# every iteration, and SIGKILLs itself as soon as the first is on disk. Started
# here so that its start-up and compilation overlap the tests below; the last
# testset checks what it left.
const _NR_QUEUED_N = 20_000         # samples per chain: the kill comes long before
const _NR_QUEUED = let dir = mktempdir()
    path = joinpath(dir, "nuts_state.jls")
    script = joinpath(dir, "queued.jl")
    write(script, """
        using Nereus, Random, Logging
        include($(repr(joinpath(@__DIR__, "fixtures", "nuts_resume_target.jl"))))
        # From the interactive thread: the chains hold the one default thread.
        Threads.@spawn :interactive begin
            while !isfile($(repr(path)))
                sleep(0.001)
            end
            ccall(:kill, Cint, (Cint, Cint), getpid(), 9)
        end
        with_logger(ConsoleLogger(stderr, Logging.Error)) do
            sample_nuts(_nr_target(); $(_NR_KW)..., n_chains = 2, warm_start = false,
                        n_samples = $(_NR_QUEUED_N), rng = MersenneTwister(3),
                        checkpoint = $(repr(path)), checkpoint_interval = 0.0)
        end
        """)
    log = joinpath(dir, "child.log")
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=1,1
           --project=$(dirname(@__DIR__)) $script`
    (; path, log, proc = run(pipeline(cmd; stdout = devnull, stderr = log); wait = false))
end

# `rng` is the generator passed in, a fresh MersenneTwister(3) for every run
# unless given. A resumed run gets a fresh one too, which must come out as the
# uninterrupted run left its own.
function _nr_run(n_samples; stop = Int[], rng = MersenneTwister(3), rv = _NR_RV, kw...)
    tg = _nr_target(rv)
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

# The error a call throws, or `nothing`.
_nr_err(f) = try f(); nothing catch e; e end

@testset "sample_nuts resume" begin
    full = _nr_run(150)
    # The warm start moved a circular window, so a resume has one to put back.
    @test full.windows != Nereus.circular_windows(_nr_target().params)

    path = joinpath(mktempdir(), "nuts_state.jls")
    # Killed in the init buffer: the other three, held at the re-cut after 75,
    # are saved there with it pending. Resumed, and killed again at the second
    # window end (after its re-cut), in the term buffer, at the first sampling
    # iteration and later in sampling.
    @test_throws Nereus._NutsStopped _nr_run(150; checkpoint = path, stop = [40, 90, 100, 200])
    @test isfile(path) && !isfile(path * ".tmp")
    @test [s.step for s in open(deserialize, path).state.chains] == [40, 75, 75, 75]
    @test_throws Nereus._NutsStopped _nr_run(150; checkpoint = path, resume = true,
                                             stop = [160, 150, 201, 230])
    mid = _nr_run(100; checkpoint = path, resume = true)   # a finished run ...
    ext = _nr_run(150; checkpoint = path, resume = true)   # ... extended

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
    @test _nr_same(_nr_run(150; checkpoint = path, resume = true), full)

    # Refusals: no checkpoint, a different run, a total shorter than the saved.
    @test_throws ArgumentError _nr_run(150; resume = true)
    @test_throws ArgumentError _nr_run(150; checkpoint = path * ".none", resume = true)
    err = _nr_err(() -> _nr_run(200; checkpoint = path, resume = true, target_accept = 0.9))
    @test err isa ArgumentError && occursin("target_accept", err.msg)
    err = _nr_err(() -> _nr_run(200; checkpoint = path, resume = true, n_warmup = 300))
    @test err isa ArgumentError && occursin("n_warmup", err.msg)
    err = _nr_err(() -> _nr_run(200; checkpoint = path, resume = true, rv = _NR_RV .+ 1.0))
    @test err isa ArgumentError && occursin("data", err.msg)
    @test_throws ArgumentError _nr_run(100; checkpoint = path, resume = true)
    # A generator that did not start where the run's did is another run: another
    # seed, or another kind of generator.
    for rng in (MersenneTwister(4), Xoshiro(3))
        err = _nr_err(() -> _nr_run(200; checkpoint = path, resume = true, rng))
        @test err isa ArgumentError && occursin("rng", err.msg)
    end
end

@testset "sample_nuts resume, one chain on the caller's rng" begin
    # A single chain runs on `rng` itself, which the resume restores in place.
    full = _nr_run(60; n_chains = 1)
    path = joinpath(mktempdir(), "nuts_state.jls")
    @test_throws Nereus._NutsStopped _nr_run(60; n_chains = 1, checkpoint = path, stop = [120])
    @test _nr_same(_nr_run(60; n_chains = 1, checkpoint = path, resume = true), full)
end

@testset "sample_nuts re-cuts a seam during the adaptation, and resumes across it" begin
    # A circular orbit with Mo = 0 at the reference epoch, a posterior of
    # 0.0° ± 2.7° on the seam of Mo's U(0, 2π). `init` puts Mo half a turn away,
    # at π, and centres the window on it, which leaves the seam where it is.
    # The chains reach the posterior within the init buffer, each held on one
    # side of the seam by the logit chart; the re-cut after iteration 75 moves
    # it off them. Before the re-cut existed every chain kept 7-26 % of its
    # draws within 0.5° of a wall, at Mo = 2.0° ± 1.5°.
    err = 10.0
    t, rv = let r = MersenneTwister(7)
        t = sort!(100 .* rand(r, _NR_N))
        t, [Nereus.rv_keplerian(ti, 4.23, 40.0, 0.0, 0.0, 0.0, median(t)) for ti in t] .+
           err .* randn(r, _NR_N)
    end
    target() = build_target(
        planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                        sesinw = FixedPrior(0.0), secosw = FixedPrior(0.0),
                        Mo = UniformPrior(0.0, 2pi)),),
        rv = (SIM = (data = (t = t, rv = rv, rv_err = fill(err, _NR_N)),
                     sigma = LogUniformPrior(0.5, 10.0)),))
    L = target().params.layout
    start = Dict("P_k1" => 4.23, "K_k1" => 40.0, "Mo_k1" => float(π))
    init = map(L.unfrozen_names, L.unfrozen_priors) do nm, ps
        lo, hi = Nereus.bounds(ps)
        get(start, nm, isfinite(lo) && isfinite(hi) ? (lo + hi) / 2 : 0.0)
    end
    function circ_run(n_samples; stop = Int[], kw...)
        tg = target()
        rng = MersenneTwister(3)
        Nereus._NUTS_STOP_AFTER[] = stop
        try
            ch = with_logger(ConsoleLogger(stderr, Logging.Error)) do
                sample_nuts(tg; _NR_KW..., n_samples, rng, init, warm_start = false, kw...)
            end
            return (; ch, rng, windows = Nereus.circular_windows(tg.params))
        finally
            Nereus._NUTS_STOP_AFTER[] = Int[]
        end
    end

    full = circ_run(150)
    lo, hi = full.windows["Mo_k1"]
    @test lo != 0.0                                   # moved during the adaptation
    for c in 1:4                                      # no chain is at a wall ...
        Mo = vec(full.ch[:, :Mo_k1, c].data)
        @test all(x -> x - lo > deg2rad(0.5) && hi - x > deg2rad(0.5), Mo)
        @test abs(angle(sum(cis, Mo))) < deg2rad(1.0)  # ... nor pulled off the mode by one
    end

    # A resume across the re-cuts is bit-identical. Chain 1 stops at 40: the
    # re-cut after 75 cannot be taken without it, and the other three are saved
    # at 75 with it pending, in the windows they had. Resumed, chain 1 stops at
    # 101, after the re-cuts at 75 (the move) and 100; the others are saved at
    # the window end 150 with its re-cut pending, in the moved window (which
    # that re-cut may relabel by a turn: the window keeps the draws' median in
    # the one written, and here the median sits at its edge). Then the run
    # finishes.
    path = joinpath(mktempdir(), "nuts_state.jls")
    chains_at(p) = open(deserialize, p).state.chains
    @test_throws Nereus._NutsStopped circ_run(150; checkpoint = path, stop = [40])
    st = chains_at(path)
    @test [s.step for s in st] == [40, 75, 75, 75]
    @test all(s -> s.circ.recut_at == 0, st)
    @test open(deserialize, path).state.windows["Mo_k1"] == (0.0, 2pi)
    @test_throws Nereus._NutsStopped circ_run(150; checkpoint = path, resume = true,
                                              stop = [101])
    st = chains_at(path)
    @test [s.step for s in st] == [101, 150, 150, 150]
    @test all(s -> s.circ.recut_at == 100, st)
    w = open(deserialize, path).state.windows["Mo_k1"]
    @test w != (0.0, 2pi)
    @test rem2pi(w[1] - lo, RoundNearest) ≈ 0 atol = 1e-12
    @test _nr_same(circ_run(150; checkpoint = path, resume = true), full)
end

@testset "a resume puts back both bounds of a window" begin
    # Each re-cut keeps the span the last one left (hi - lo), so a window re-cut
    # more than once spans 2π only to within an ulp. A resume that put back `lo`
    # alone made hi = lo + 2π, and the resumed run's Mo, mapped into a window an
    # ulp wider, differed from the uninterrupted run's by 4e-16 (the unseeded
    # check below, in about one run in seven).
    tg = _nr_target(); p = tg.params
    i = findfirst(==("Mo_k1"), p.layout.unfrozen_names)
    lo, hi = 1.1, nextfloat(1.1 + 2pi)
    Nereus.set_circular_window!(p, i, lo; hi, transforms = (tg.transform,))
    saved = Nereus.circular_windows(p)
    @test saved["Mo_k1"] == (lo, hi)
    held(t) = (Nereus.bounds(t.params.layout.unfrozen_priors[i]),
               (t.params.layout.packed_priors.lowers[i], t.params.layout.packed_priors.uppers[i]),
               (t.transform.lowers[i], t.transform.uppers[i]))
    t2 = _nr_target()
    Nereus._restore_windows!(t2.params, saved, (t2.transform,))
    @test held(t2) == ((lo, hi), (lo, hi), (lo, hi))
    # The shared restore (several chains, once): `lo` already in place is not
    # enough to skip a window whose `hi` is not.
    t3 = _nr_target()
    Nereus.set_circular_window!(t3.params, i, lo; transforms = (t3.transform,))
    @test Nereus.bounds(t3.params.layout.unfrozen_priors[i]) == (lo, 1.1 + 2pi) != (lo, hi)
    @test Nereus._restore_shared_windows!(t3.params, saved, (t3.transform,))
    @test held(t3) == ((lo, hi), (lo, hi), (lo, hi))
    @test !Nereus._restore_shared_windows!(t3.params, saved, (t3.transform,))
end

@testset "sample_nuts seed: honoured, and part of the checkpoint's fingerprint" begin
    # run_job passes the job's `seed`. sample_nuts used to swallow it, so its
    # NUTS runs were unseeded. Short runs, without the warm start: the RNG is
    # what is under test.
    kw = (rng = nothing, n_chains = 2, warm_start = false)
    path = joinpath(mktempdir(), "nuts_state.jls")
    a = _nr_run(30; kw..., seed = 7, checkpoint = path)
    @test _nr_same(_nr_run(30; kw..., seed = 7), a)
    @test _nr_run(30; kw..., seed = 8).ch.value.data != a.ch.value.data
    err = _nr_err(() -> _nr_run(30; seed = 7))           # with the helper's `rng`
    @test err isa ArgumentError && occursin("not both", err.msg)

    # A resume with another seed is refused; the same seed continues the run.
    err = _nr_err(() -> _nr_run(60; kw..., seed = 8, checkpoint = path, resume = true))
    @test err isa ArgumentError && occursin("seed", err.msg)
    @test _nr_same(_nr_run(60; kw..., seed = 7, checkpoint = path, resume = true),
                   _nr_run(60; kw..., seed = 7))

    # An unseeded run (the default task-local stream) can still be continued.
    upath = joinpath(mktempdir(), "nuts_state.jls")
    u = _nr_run(30; kw..., checkpoint = upath)
    @test _nr_run(40; kw..., checkpoint = upath, resume = true).ch.value.data[1:30, :, :] ==
          u.ch.value.data

    # Through run_job's dispatch: the job's seed reaches the sampler.
    function job(seed)
        tg = _nr_target()
        kwj = Dict{String,Any}("n_samples" => 30, "n_warmup" => 200, "n_chains" => 2,
                               "warm_start" => false, "progress" => false)
        cfg = Dict{String,Any}("sampler" => Dict{String,Any}("name" => "nuts", "kwargs" => kwj),
                               "output_dir" => mktempdir())
        res = with_logger(ConsoleLogger(stderr, Logging.Error)) do
            Nereus._dispatch_sampler(cfg, tg, tg.data, seed)
        end
        return res.chains.value.data
    end
    @test job(7) == job(7)
    @test job(7) != job(8)
end

@testset "sample_nuts checkpoint while a chain waits for a thread" begin
    q = _NR_QUEUED
    wait(q.proc)
    killed = q.proc.termsignal == 9
    killed || @info "queued-chain child process" exitcode = q.proc.exitcode log = read(q.log, String)
    @test killed
    @test isfile(q.path)
    chains = open(deserialize, q.path).state.chains
    started = [haskey(s, :z) for s in chains]
    steps = [s.step for s in chains]
    # The chain that ran had saved mid-run; the one queued behind it is in the
    # checkpoint as its start point. Before the fix the first checkpoint came
    # only once the queued chain had started, with the other one finished.
    @test count(started) == 1
    @test 0 < maximum(steps) < _NR_KW.n_warmup + _NR_QUEUED_N
    @test steps[.!started] == [0]
    # It continues bit for bit, the queued chain starting as it would have.
    n = max(10, maximum(steps) - _NR_KW.n_warmup)
    kw = (n_chains = 2, warm_start = false)
    @test _nr_same(_nr_run(n; kw..., checkpoint = q.path, resume = true), _nr_run(n; kw...))
end
