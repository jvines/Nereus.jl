# ActivityGP joint-likelihood solver: the quasi-periodic kernel blocks and
# the covariance builders that feed it.

using Nereus, LinearAlgebra, Random, Test
using Random: MersenneTwister

# Transcription of the closed forms in the header of src/noise/activity_gp.jl:
#   f   = -τ²/(2λe²) - sin²(πτ/P)/(2λp²)
#   f'  = -τ/λe² - π/(2Pλp²)·sin(2πτ/P)
#   f'' = -1/λe² - π²/(P²λp²)·cos(2πτ/P)
# and the blocks (k, -f'k, f'k, -(f''+f'²)k), written with the same operations
# as `activity_kernel_blocks` so the two agree bit for bit.
function _qp_blocks_transcribed(τ, amp, P, λe, λp)
    s   = sin(π * τ / P)
    s2  = sin(2π * τ / P)
    c2  = cos(2π * τ / P)
    f   = -τ^2 / (2 * λe^2) - s^2 / (2 * λp^2)
    fp  = -τ / λe^2 - π / (2 * P * λp^2) * s2
    fpp = -1.0 / λe^2 - π^2 / (P^2 * λp^2) * c2
    k   = amp^2 * exp(f)
    return (k, -fp * k, fp * k, -(fpp + fp^2) * k)
end

@testset "AGP kernel blocks and covariance builders keep their values" begin
    rng = MersenneTwister(20261003)
    for _ in 1:200
        τ   = 200 * (rand(rng) - 0.5)
        amp = 0.2 + 3rand(rng); P = 2 + 20rand(rng)
        λe  = 5 + 200rand(rng); λp = 0.2 + 2rand(rng)
        @test Nereus.activity_kernel_blocks(τ, amp, P, λe, λp) ===
              _qp_blocks_transcribed(τ, amp, P, λe, λp)
    end
    # τ = 0: no derivative cross-covariance, Var(Ġ) = amp²·(1/λe² + π²/(P²λp²)).
    kGG, kGd, kdG, kdd = Nereus.activity_kernel_blocks(0.0, 1.3, 8.7, 40.0, 0.6)
    @test kGG == 1.3^2
    @test kGd == 0 && kdG == 0
    @test kdd ≈ 1.3^2 * (1 / 40.0^2 + π^2 / (8.7^2 * 0.6^2)) rtol = 1e-14

    # The flat builder and the block-factored builder evaluate the same
    # expressions per pair. On a channel-major layout their upper triangles
    # agree exactly; below the diagonal the flat builder mirrors the upper
    # entry while the blocked one sums the two derivative terms in the other
    # order, so those entries can differ by an ulp.
    N, C = 23, 4
    t = sort!(60 .* rand(rng, N))
    ca = randn(rng, C); cb = randn(rng, C)
    amp, P, λe, λp = 1.1, 9.3, 35.0, 0.7
    t_flat = repeat(t, C)
    ch = repeat([:rv, :bis, :fwhm, :halpha], inner = N)
    a_flat = repeat(ca, inner = N); b_flat = repeat(cb, inner = N)
    Σd = activity_gp_covariance(t_flat, ch, a_flat, b_flat, amp, P, λe, λp)
    Σb = Nereus.activity_gp_covariance_blocked(t, ca, cb, amp, P, λe, λp)
    @test UpperTriangular(Σd) == UpperTriangular(Σb)
    @test Σd ≈ Σb rtol = 1e-14
    @test issymmetric(Σd)
end

# ---------------------------------------------------------------------
# Shared fixtures for the solver testsets below
# ---------------------------------------------------------------------

# Random inputs of the shape `_activity_gp_joint_ll` hands the low-rank
# solver: N shared epochs, C channel-stacked blocks (RV first).
function _agp_inputs(rng, N, C; amp = 1.0, P = 9.0, λe = 40.0, λp = 0.6)
    epochs = sort!(80 .* rand(rng, N))
    ca = randn(rng, C); cb = 0.5 .* randn(rng, C)
    σ² = 0.05 .+ rand(rng, C * N)
    y = randn(rng, C * N)
    return (epochs, ca, cb, amp, P, λe, λp, y, σ²)
end

# An RV + four-indicator dataset with the five-channel ActivityGP of the
# HD 18599 job (logR'HK has no derivative coupling), and a workspace for it.
function _agp_params(rng; n = 30, use_derivative = true,
                     channels = [:bis, :fwhm, :halpha, :logrhk],
                     extra = (;), floor = false)
    t = sort!(60 .* rand(rng, n))
    inds = Dict(String(c) => randn(rng, n) for c in channels)
    errs = Dict(String(c) => fill(0.3, n) for c in channels)
    data = Data(; t_rv = t, rv = 3 .* randn(rng, n), rv_err = fill(1.0, n),
                  rv_inst = ones(Int, n), indicators = inds, indicator_errs = errs)
    nms = NoiseModel[ActivityGP(; channels, use_derivative, extra...)]
    floor && push!(nms, IndicatorFloor(; channels))
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = InstrumentConfig(rv = ["SIM"]), data = data,
                     M_s = 1.0, noise_models = nms)
    ws = Nereus.PTWorkspace(params, 0, length(nms); n_obs = n)
    return data, params, ws
end

# Draw every unfrozen parameter from its prior (inside its bounds).
function _prior_theta!(theta, params, rng)
    L = params.layout
    for (d, ps) in enumerate(L.unfrozen_priors)
        lo, hi = bounds(ps)
        v = rand(rng, ps.dist)
        k = 0
        while !(lo <= v <= hi) && k < 1000
            v = rand(rng, ps.dist); k += 1
        end
        lo <= v <= hi || (v = lo + (hi - lo) * rand(rng))
        theta.values[L.unfrozen_idx[d]] = v
    end
    return theta
end

@testset "AGP low-rank solver on workspace buffers" begin
    rng = MersenneTwister(11)
    w = Nereus.AGPWorkspace()
    @test w.N == 0
    # Same operations on reused buffers: equal bit for bit, whatever the
    # buffers held before (they are reused across sizes and calls here).
    for (N, C) in ((17, 5), (31, 4), (17, 2), (8, 3), (31, 5))
        for _ in 1:3
            a = _agp_inputs(rng, N, C; λe = 10 + 100rand(rng), λp = 0.3 + rand(rng))
            r_alloc = Nereus.activity_gp_joint_logpdf_lowrank(a...)
            r_ws = Nereus.activity_gp_joint_logpdf_lowrank!(w, a...)
            @test isfinite(r_alloc)
            @test r_ws === r_alloc
            @test w.N == N
        end
    end

    # Through the likelihood: the workspace path equals the allocating path.
    data, params, ws = _agp_params(rng)
    theta = Theta{Float64}(params)
    n_finite = 0
    for _ in 1:60
        _prior_theta!(theta, params, rng)
        a = Nereus.rv_log_likelihood(theta, data, ws)
        b = Nereus.rv_log_likelihood(theta, data)
        @test isequal(a, b)
        n_finite += isfinite(a)
    end
    @test n_finite > 30
    @test ws.agp.N == length(data.t_rv)

    # The four 2N×2N solver matrices are no longer allocated per call.
    _prior_theta!(theta, params, rng)
    Nereus.rv_log_likelihood(theta, data, ws)
    N = length(data.t_rv)
    @test (@allocated Nereus.rv_log_likelihood(theta, data, ws)) < (2N)^2 * 8

    # Solver scratch is not part of a sampler checkpoint.
    @test :agp ∉ keys(Nereus._ws_snapshot(ws))
end

@testset "AGP likelihood setup: cached indices and buffers" begin
    rng = MersenneTwister(23)
    configs = (
        (;),                                                    # 5 channels, low rank
        (; use_derivative = false),
        (; channels = [:bis]),                                  # C = 2, dense
        (; channels = [:bis, :fwhm]),                           # C = 3, dense
        (; extra = (; indicators_only = true)),
        (; channels = [:bis, :fwhm], extra = (; marginalize_indicators = true)),
        (; use_derivative = false, extra = (; latent_kernel = :matern32)),
    )
    for kw in configs
        data, params, ws = _agp_params(rng; kw...)
        # A second dataset with the same layout, evaluated through the same
        # workspace: the cached indices must follow the data they came from.
        data2, _, _ = _agp_params(MersenneTwister(99); kw...)
        theta = Theta{Float64}(params)
        n_finite = 0
        for _ in 1:25
            _prior_theta!(theta, params, rng)
            for d in (data, data2)
                a = Nereus.rv_log_likelihood(theta, d, ws)
                @test isequal(a, Nereus.rv_log_likelihood(theta, d))
                n_finite += isfinite(a)
            end
        end
        @test n_finite > 25
        @test length(ws.agp.index) == 2
    end

    # The five-channel setup no longer builds strings or per-call vectors:
    # what is left of the call is a few small objects outside the AGP code.
    data, params, ws = _agp_params(rng)
    theta = Theta{Float64}(params)
    _prior_theta!(theta, params, rng)
    Nereus.rv_log_likelihood(theta, data, ws)
    @test (@allocated Nereus.rv_log_likelihood(theta, data, ws)) < 2048

    # The same validation as before, on first use of a workspace: a dataset
    # without one of the channels is refused, not scored.
    _, params_b, _ = _agp_params(rng)
    t = sort!(60 .* rand(rng, 30))
    no_logrhk = Data(; t_rv = t, rv = randn(rng, 30), rv_err = ones(30),
                   rv_inst = ones(Int, 30),
                   indicators = Dict(c => randn(rng, 30) for c in ("bis", "fwhm", "halpha")),
                   indicator_errs = Dict(c => fill(0.3, 30) for c in ("bis", "fwhm", "halpha")))
    ws_b = Nereus.PTWorkspace(params_b, 0, 1; n_obs = 30)
    theta_b = Theta{Float64}(params_b)
    _prior_theta!(theta_b, params_b, rng)
    @test_throws ArgumentError Nereus.rv_log_likelihood(theta_b, no_logrhk, ws_b)
    @test_throws ArgumentError Nereus.rv_log_likelihood(theta_b, no_logrhk)
end
