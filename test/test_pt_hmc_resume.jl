# pt_hmc checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any checkpoint and continued with
# `resume = true` is bit-identical to one that ran straight through, so a run
# that has not mixed can simply be extended. pt_hmc's state between sweeps is
# set up in two phases before the kept sweeps start: the pilot (its own warm-up
# and sweeps, which re-grid the ladder) and the final warm-up (step size and
# mass matrix per temperature). Their lengths do not depend on `n_sweeps`, so a
# shorter total cannot stop inside them; a run is stopped there the way a kill
# would, through the checkpoint hook, right after the pilot's warm-up, inside the
# pilot, at its last sweep (the ladder re-grid follows) and right after the
# final warm-up. A finished run is then extended twice, past its own end.
#
# The eccentricity is fixed (e = 0.02, ω = 45°) and the signal phased so that Mo
# sits on the 0/2π seam: the warm start's pre-search then moves Mo's window (to
# [π, 3π) with this data), and that window is part of what has to come back.
# Not e = 0: the gradient is NaN there and NUTS freezes in warm-up, which
# would leave nothing between sweeps for a resume to get wrong.
using Test
using Nereus
using Random

const _PH_N = 40
const _PH_T, _PH_RV = let rng = MersenneTwister(7)
    t = sort!(60 .* rand(rng, _PH_N))
    t, 40.0 .* sin.(2π .* t ./ 4.23 .+ 5.618) .+ 1.5 .* randn(rng, _PH_N)
end

# A fresh target per run: the warm start moves circular windows on the target.
_ph_target(rv = _PH_RV) = build_target(
    planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                    sesinw = 0.1, secosw = 0.1, Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _PH_T, rv = rv, rv_err = fill(1.5, _PH_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))

const _PH_KW = (n_temps = 4, n_walkers_per_temp = 2, n_warmup = 60, n_pilot = 12,
                seed = 11, progress = false)

_ph_run(n_sweeps; kw...) = sample_pt_hmc(_ph_target(); _PH_KW..., n_sweeps, kw...)
# (draw, parameter, chain) of the cold chain, :lp included; two walkers a sweep.
_ph_draws(r) = r[1].value.data

# Every output: the draws, the headline log-evidence and the whole report.
function _ph_same(a, b)
    _ph_draws(a) == _ph_draws(b) || return false
    isequal(a[2], b[2]) || return false
    return all(isequal(getfield(a[3], f), getfield(b[3], f))
               for f in fieldnames(typeof(a[3])))
end

struct _PHStop <: Exception end

# Run until the checkpoint of (`phase`, `sweep`) is written, then stop as if
# killed. Checkpoints every sweep, so any sweep can be the last one written.
function _ph_stop_at(phase, sweep, n_sweeps; kw...)
    Nereus._PT_HMC_CHECKPOINT_HOOK[] =
        (p, s) -> (p === phase && s == sweep) && throw(_PHStop())
    try
        _ph_run(n_sweeps; checkpoint_interval = 0.0, kw...)
        error("the run was not stopped at $phase sweep $sweep")
    catch e
        e isa _PHStop || rethrow()
    finally
        Nereus._PT_HMC_CHECKPOINT_HOOK[] = nothing
    end
end

@testset "pt_hmc resume" begin
    tg = _ph_target()
    full = sample_pt_hmc(tg; _PH_KW..., n_sweeps = 30)
    # The window moved, so the resumes below have to put it back.
    @test Nereus.circular_windows(tg.params)["Mo_k1"][1] != 0.0
    dir = mktempdir()

    # A finished run, extended past its end twice.
    path = joinpath(dir, "pt_hmc_state.jls")
    short = _ph_run(10; checkpoint = path)
    @test isfile(path) && !isfile(path * ".tmp")
    # NUTS moves (a frozen kernel would leave nothing for a resume to get wrong).
    @test all(>(1e-6), open(Nereus.deserialize, path).state.core.ε)
    @test _ph_draws(short) == _ph_draws(full)[1:20, :, :]
    mid = _ph_run(22; checkpoint = path, resume = true)
    ext = _ph_run(30; checkpoint = path, resume = true)
    @test _ph_same(ext, full)
    @test _ph_draws(mid) == _ph_draws(full)[1:44, :, :]

    # Resuming at the checkpoint's own sweep runs nothing and returns the same.
    @test _ph_same(_ph_run(30; checkpoint = path, resume = true), full)

    # Stopped as if killed: after the pilot's warm-up, inside the pilot, at its
    # last sweep, after the final warm-up, and inside the kept sweeps.
    for (phase, sweep) in ((:pilot, 0), (:pilot, 5), (:pilot, 12), (:final, 0),
                           (:final, 7))
        p = joinpath(dir, "stop_$(phase)_$(sweep).jls")
        _ph_stop_at(phase, sweep, 30; checkpoint = p)
        @test _ph_same(_ph_run(30; checkpoint = p, resume = true), full)
    end

    # A fixed ladder has no pilot: one phase only.
    fixed_kw = (betas = [0.0, 0.1, 0.4, 1.0],)
    fixed = _ph_run(20; fixed_kw...)
    pf = joinpath(dir, "fixed.jls")
    _ph_run(8; checkpoint = pf, fixed_kw...)
    @test _ph_same(_ph_run(20; checkpoint = pf, resume = true, fixed_kw...), fixed)

    # Refusals: no checkpoint, a different run, a total shorter than the saved,
    # and a new total that moves the default pilot length.
    @test_throws ArgumentError _ph_run(30; resume = true)
    @test_throws ArgumentError _ph_run(30; checkpoint = path * ".none", resume = true)
    err = try _ph_run(40; checkpoint = path, resume = true, target_accept = 0.9); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("target_accept", err.msg)
    err = try sample_pt_hmc(_ph_target(_PH_RV .+ 1.0); _PH_KW..., n_sweeps = 40,
                            checkpoint = path, resume = true); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("data", err.msg)
    @test_throws ArgumentError _ph_run(29; checkpoint = path, resume = true)
    err = try _ph_run(40; checkpoint = path, resume = true, n_pilot = nothing); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("n_pilot", err.msg)
    @test_throws ArgumentError _ph_run(40; checkpoint = pf, resume = true)  # ladder
end
