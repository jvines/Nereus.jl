# pt_whitening checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any step and continued with `resume = true` is
# bit-identical to one that ran straight through, so a run that has not mixed
# can simply be extended. The split points below cross everything that carries
# state between steps: the whitening ring buffer part-filled (step 12) and
# wrapped (window 20, step 25), identity swaps giving way to whitened ones (step
# 31), the (μ, σ) refresh cadence (every 7 steps from 31, so step 43 stops
# between refreshes and the saved (μ, σ) lag the ring), the burn-in re-cut of
# the circular Mo (steps 60 and 120: the ring relabelled, the window moved), the
# end of burn-in, and a finished run extended past its own end. A second target,
# with an astrometry-only planet, carries the node-flip RNG streams across a
# split.
using Test
using Nereus
using Nereus: IADData
using Random

# An eccentric planet whose mean anomaly sits 0.02 rad from the 0/2π seam, with
# priors narrow enough that the cold walkers gather there within burn-in, so the
# re-cut has a seam-centred Mo to move (as in test_circular.jl). Measured on
# arm64 under seed 42: it moves at both re-cuts, and the second move's hi comes
# out an ulp off lo + 2π, the case the exact window restore is there for (with
# it off, the windows check below fails).
const _WS_E, _WS_W = 0.3, 1.0
_ws_target(t, rv) = build_target(
    planets = (b = (P = UniformPrior(12.25, 12.35), K = UniformPrior(20.0, 30.0),
                    sesinw = UniformPrior(sqrt(_WS_E) * sin(_WS_W) - 0.1,
                                          sqrt(_WS_E) * sin(_WS_W) + 0.1),
                    secosw = UniformPrior(sqrt(_WS_E) * cos(_WS_W) - 0.1,
                                          sqrt(_WS_E) * cos(_WS_W) + 0.1),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (HARPS = (data = (t = t, rv = rv, rv_err = fill(2.0, length(t))),
                   sigma = LogUniformPrior(0.5, 10.0)),))
const _WS_T, _WS_RV = let rng = MersenneTwister(42), n = 60
    t = sort(55000 .+ 200 .* rand(rng, n))
    th = Theta{Float64}(_ws_target(t, zeros(n)).params)
    for (nm, v) in ("P_k1" => 12.3, "K_k1" => 25.0,
                    "sesinw_k1" => sqrt(_WS_E) * sin(_WS_W),
                    "secosw_k1" => sqrt(_WS_E) * cos(_WS_W), "Mo_k1" => 0.02,
                    "gamma_HARPS" => 0.0, "sigma_HARPS" => 0.01)
        set_param!(th, nm, v)
    end
    model, _ = rv_predictions(th, _ws_target(t, zeros(n)).data)
    t, model .+ 2.0 .* randn(rng, n)
end

const _WS_KW = (n_temps = 4, n_walkers = 20, n_burnin = 120, warmup_swaps = 30,
                whiten_window = 20, whiten_refresh = 7, seed = 42,
                show_progress = false)

# A fresh target per run: the re-cut moves circular windows on the target itself.
_ws_run(n_steps; rv = _WS_RV, kw...) = (tg = _ws_target(_WS_T, rv);
    sample_pt_whitening(tg, tg.data; _WS_KW..., n_steps, kw...))
# (step, parameter, walker); `Array(chains)` would stack the walkers.
_ws_cube(r) = r.chains.value.data

# Every field of the result, bit for bit.
function _ws_same(a, b)
    @test _ws_cube(a) == _ws_cube(b)
    @test names(a.chains) == names(b.chains)
    @test isequal(a.log_evidence, b.log_evidence)
    @test isequal(a.evidence, b.evidence)
    @test a.acceptance_within == b.acceptance_within
    @test a.acceptance_swap == b.acceptance_swap
    @test a.betas == b.betas
    @test a.n_evals == b.n_evals
    @test a.whitening_active_after == b.whitening_active_after
end

@testset "pt_whitening resume" begin
    tg_full = _ws_target(_WS_T, _WS_RV)
    full = sample_pt_whitening(tg_full, tg_full.data; _WS_KW..., n_steps = 190)
    # The run does cross what the splits are placed to cross.
    @test full.whitening_active_after == 31
    @test first(Nereus.circular_windows(tg_full.params)["Mo_k1"]) != 0.0

    path = joinpath(mktempdir(), "pt_whitening_state.jls")
    _ws_run(12; checkpoint = path)                  # ring part-filled
    @test isfile(path) && !isfile(path * ".tmp")
    _ws_run(25; checkpoint = path, resume = true)   # ring wrapped, identity swaps
    _ws_run(43; checkpoint = path, resume = true)   # whitened, between refreshes
    _ws_run(61; checkpoint = path, resume = true)   # just past the first re-cut
    _ws_run(120; checkpoint = path, resume = true)  # end of burn-in, nothing kept
    mid = _ws_run(160; checkpoint = path, resume = true)   # a finished run ...
    tg_ext = _ws_target(_WS_T, _WS_RV)                     # ... extended
    ext = sample_pt_whitening(tg_ext, tg_ext.data; _WS_KW..., n_steps = 190,
                              checkpoint = path, resume = true)

    # Identical to the uninterrupted run, bit for bit, and left in the same
    # chart: run_job and fit_health read the windows the engine sampled in.
    _ws_same(ext, full)
    @test Nereus.circular_windows(tg_ext.params) == Nereus.circular_windows(tg_full.params)
    # The finished 160-step run kept the first 40 post-burn-in steps of it.
    @test _ws_cube(mid) == _ws_cube(full)[1:40, :, :]

    # Resuming at the checkpoint's own step runs nothing and returns the same.
    _ws_same(_ws_run(190; checkpoint = path, resume = true), full)

    # Refusals: no checkpoint, another sampler's, a different run, a total
    # shorter than the saved.
    @test_throws ArgumentError _ws_run(190; resume = true)
    @test_throws ArgumentError _ws_run(190; checkpoint = path * ".none", resume = true)
    other = joinpath(mktempdir(), "pt_emcee_state.jls")
    Nereus.write_checkpoint(other, "pt_emcee", (;), (; step = 1))
    err = try _ws_run(190; checkpoint = other, resume = true); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("pt_emcee", err.msg)
    for (k, v) in ((:n_walkers, 24), (:whiten_window, 30), (:whiten_refresh, 5),
                   (:warmup_swaps, 40))
        err = try _ws_run(200; checkpoint = path, resume = true, k => v); nothing
              catch e; e end
        @test err isa ArgumentError && occursin(String(k), err.msg)
    end
    err = try _ws_run(200; checkpoint = path, resume = true, rv = _WS_RV .+ 1.0); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("data", err.msg)
    @test_throws ArgumentError _ws_run(150; checkpoint = path, resume = true)
end

# An astrometry-only planet, so the node flip (and its per-walker RNG streams)
# is live; the split falls inside burn-in after whitening has started.
const _WS_IAD = let rng = MersenneTwister(11), n = 40
    t   = sort(57000 .+ 1800 .* rand(rng, n))
    psi = 2pi .* rand(rng, n)
    IADData(t = t, abscissa = 0.1 .* randn(rng, n), abscissa_err = fill(0.1, n),
            psi = psi, parallax_factor = sin.(2pi .* (t .- 57000) ./ 365.25 .- psi),
            pm_factor = (t .- sum(t) / n) ./ 365.25)
end
_ws_as_target() = build_target(M_pri = 0.644, iad = _WS_IAD,
    planets = (b = (a = LogUniformPrior(0.3, 4.0), M_sec = LogUniformPrior(0.001, 0.05),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    inc = SinePrior(), Omega = UniformPrior(0.0, 2pi),
                    Mo = UniformPrior(0.0, 2pi)),),
    plx = NormalPrior(13.6, 0.02), M_s = 0.644)

const _WS_AS_KW = (n_temps = 3, n_walkers = 16, n_burnin = 30, warmup_swaps = 10,
                   whiten_window = 10, whiten_refresh = 3, seed = 3,
                   show_progress = false)
_ws_as_run(n_steps; kw...) = (tg = _ws_as_target();
    sample_pt_whitening(tg, tg.data; _WS_AS_KW..., n_steps, kw...))

@testset "pt_whitening resume carries the node-flip streams" begin
    tg = _ws_as_target()
    @test !isempty(Nereus.node_flips(tg.params, tg.data))
    full = _ws_as_run(60)
    path = joinpath(mktempdir(), "pt_whitening_state.jls")
    _ws_as_run(20; checkpoint = path)
    _ws_same(_ws_as_run(60; checkpoint = path, resume = true), full)
end
