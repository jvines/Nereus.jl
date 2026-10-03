# ActivityGP joint-likelihood solver: the quasi-periodic kernel blocks and
# the covariance builders that feed it.

using Nereus, LinearAlgebra, Random, Test, ForwardDiff
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

# ---------------------------------------------------------------------
# The whitened low-rank solver against the exact dense likelihood
# ---------------------------------------------------------------------

# The dense (C·N)² Gaussian log-density with the kernel, the covariance and
# its Cholesky evaluated in BigFloat from the same Float64 inputs.
function _agp_exact_dense(epochs, ca, cb, amp, P, λe, λp, y, σ²)
    setprecision(BigFloat, 128) do
        Σ = Nereus.activity_gp_covariance_blocked(epochs, big.(ca), big.(cb),
                big(amp), big(P), big(λe), big(λp))
        for i in eachindex(σ²); Σ[i, i] += big(σ²[i]); end
        F = cholesky!(Symmetric(Σ))
        yb = big.(y)
        -(dot(yb, F \ yb) + logdet(F) + length(y) * log(2 * big(π))) / 2
    end
end

# The same density through a Float64 dense Cholesky (generic in the eltype,
# so it also carries dual numbers).
function _agp_dense(epochs, ca, cb, amp, P, λe, λp, y, σ²)
    Σ = Nereus.activity_gp_covariance_blocked(epochs, ca, cb, amp, P, λe, λp)
    for i in eachindex(σ²); Σ[i, i] += σ²[i]; end
    F = cholesky!(Symmetric(Σ))
    return -(dot(y, F \ y) + logdet(F) + length(y) * log(2π)) / 2
end

@testset "AGP whitened solver: exact against the dense likelihood" begin
    rng = MersenneTwister(3)
    N, C = 24, 5
    epochs = sort!(80 .* rand(rng, N))
    σ = [1.0, 0.3, 0.5, 0.2, 0.4]
    σ² = repeat(σ .^ 2, inner = N)
    # Data drawn from the model itself; the fifth channel has no Ġ coupling,
    # as logR'HK in the HD 18599 fit.
    ca0 = [3.0, 0.8, -1.2, 0.6, 0.9]; cb0 = [0.8, -0.3, 0.5, 0.2, 0.0]
    Σ0 = Nereus.activity_gp_covariance_blocked(epochs, ca0, cb0, 1.0, 9.0, 30.0, 0.6)
    y = cholesky(Symmetric(Σ0 + Diagonal(σ²))).L * randn(rng, C * N)
    solve(a) = Nereus.activity_gp_joint_logpdf_lowrank(a...)
    err(a) = Float64(abs(solve(a) - _agp_exact_dense(a...)))

    # Near the truth (the posterior).
    for _ in 1:20
        a = (epochs, ca0 .* (1 .+ 0.01 .* randn(rng, C)), cb0 .* (1 .+ 0.01 .* randn(rng, C)),
             1.0, 9.0 + 0.05randn(rng), 30 * (1 + 0.02randn(rng)), 0.6 * (1 + 0.02randn(rng)),
             y, σ²)
        @test err(a) <= 1e-10
    end
    # Across a broad prior: couplings from 0.1 to 30 times the noise of their
    # channel, coherence 5-500 d, λp 0.2-5, P 3-30 d, extra jitter. Long λe
    # with large λp is where the previous solver's 1e-10·max|K| jitter on K_g
    # moved log L by up to several nats.
    worst = 0.0
    for _ in 1:30
        ca = randn(rng, C) .* σ .* 10 .^ (3rand(rng, C) .- 1)
        cb = randn(rng, C) .* σ .* 10 .^ (3rand(rng, C) .- 1); cb[5] = 0
        a = (epochs, ca, cb, 1.0, 3 + 27rand(rng), 5 * 100^rand(rng), 0.2 * 25^rand(rng),
             y, σ² .* (1 .+ rand(rng, C * N)))
        e = err(a)
        worst = max(worst, e)
        @test e <= 1e-10
    end
    @info "AGP whitened solver: worst |Δ| against BigFloat dense on prior draws" worst

    # Rank-deficient 2×2 blocks are exact, not approximated: no Ġ coupling
    # anywhere (use_derivative = false), Ġ couplings proportional to the G
    # couplings, and no G coupling at all (bGG = 0 at every epoch).
    for (ca, cb) in ((ca0, zeros(C)), (ca0, 0.7 .* ca0), (zeros(C), cb0 .+ 0.1))
        a = (epochs, ca, cb, 1.0, 9.0, 30.0, 0.6, y, σ²)
        @test isfinite(solve(a))
        @test err(a) <= 1e-10
    end

    # Two and three channels (now also on the low-rank path).
    for Cs in (2, 3)
        ys = y[1:(Cs * N)]; s2 = σ²[1:(Cs * N)]
        a = (epochs, ca0[1:Cs], cb0[1:Cs], 1.0, 9.0, 30.0, 0.6, ys, s2)
        @test err(a) <= 1e-10
    end

    # Workspace and allocating methods agree bit for bit; the workspace call
    # allocates nothing that scales with N.
    w = Nereus.AGPWorkspace()
    a = (epochs, ca0, cb0, 1.0, 9.0, 30.0, 0.6, y, σ²)
    @test Nereus.activity_gp_joint_logpdf_lowrank!(w, a...) === solve(a)
    @test (@allocated Nereus.activity_gp_joint_logpdf_lowrank!(w, a...)) < 512

    # Dual numbers take the generic path: the gradient matches the gradient
    # of the dense likelihood.
    f_lr(x) = Nereus.activity_gp_joint_logpdf_lowrank(epochs, x[1:5], x[6:10],
                  x[11], x[12], x[13], x[14], y, σ²)
    f_dn(x) = _agp_dense(epochs, x[1:5], x[6:10], x[11], x[12], x[13], x[14], y, σ²)
    x0 = vcat(ca0, cb0, 1.0, 9.0, 30.0, 0.6)
    g_lr = ForwardDiff.gradient(f_lr, x0)
    g_dn = ForwardDiff.gradient(f_dn, x0)
    @test all(isfinite, g_lr)
    @test g_lr ≈ g_dn rtol = 1e-8
    @test f_lr(x0) ≈ f_dn(x0) atol = 1e-10

    # And through the model: a gradient of the log-density of an ActivityGP
    # target agrees with central differences.
    data, params, _ = _agp_params(MersenneTwister(5))
    target = NereusTarget(params, data)
    th = Theta{Float64}(params)
    _prior_theta!(th, params, MersenneTwister(6))
    yv = Nereus.transform_forward([th.values[i] for i in params.layout.unfrozen_idx],
                                  target.transform)
    lp(v) = Nereus.LogDensityProblems.logdensity(target, v)
    @test isfinite(lp(yv))
    g = ForwardDiff.gradient(lp, yv)
    h = 1e-6
    for d in 1:length(yv)
        e = zeros(length(yv)); e[d] = h
        fd = (lp(yv .+ e) - lp(yv .- e)) / 2h
        @test g[d] ≈ fd rtol = 1e-4 atol = 1e-5
    end
end

@testset "AGP whitened solver: NaN inputs give NaN" begin
    # A NaN coupling used to be read as "no information" by the zero tests of
    # the 2×2 factor (NaN > 0 is false), and the solver returned a finite
    # log-likelihood. The dense likelihood returns NaN; so does the solver.
    rng = MersenneTwister(41)
    w = Nereus.AGPWorkspace()
    for (N, C) in ((12, 5), (9, 2), (10, 3))
        epochs, ca, cb, amp, P, λe, λp, y, σ² = _agp_inputs(rng, N, C)
        solve(ca, cb, y, σ²) = (Nereus.activity_gp_joint_logpdf_lowrank(epochs, ca, cb,
                                    amp, P, λe, λp, y, σ²),
                                Nereus.activity_gp_joint_logpdf_lowrank!(w, epochs, ca, cb,
                                    amp, P, λe, λp, y, σ²))
        @test all(isfinite, solve(ca, cb, y, σ²))
        for c in 1:C
            a = copy(ca); a[c] = NaN
            @test all(isnan, solve(a, cb, y, σ²))
            b = copy(cb); b[c] = NaN
            @test all(isnan, solve(ca, b, y, σ²))
        end
        @test all(isnan, solve(fill(NaN, C), cb, y, σ²))
        @test all(isnan, solve(ca, fill(NaN, C), y, σ²))
        # With no G coupling at all, a NaN Ġ coupling still propagates.
        b = copy(cb); b[1] = NaN
        @test all(isnan, solve(zeros(C), b, y, σ²))
        for k in (1, N + 2, C * N)
            s = copy(σ²); s[k] = NaN
            @test all(isnan, solve(ca, cb, y, s))
            r = copy(y); r[k] = NaN
            @test all(isnan, solve(ca, cb, r, σ²))
        end
    end

    # Through the likelihood, with and without a workspace: NaN in any one
    # coupling or channel jitter of the five-channel model.
    data, params, ws = _agp_params(MersenneTwister(43))
    theta = Theta{Float64}(params)
    L = params.layout
    _prior_theta!(theta, params, MersenneTwister(44))
    @test isfinite(Nereus.rv_log_likelihood(theta, data, ws))
    nan_names = filter(n -> occursin(r"^(Vc|Vr|Bc|Br|Fc|Fr|Hc|Hr|Lc)$", n) ||
                            startswith(n, "gp_act_jit_"), L.unfrozen_names)
    @test length(nan_names) == 13
    for n in nan_names
        th = deepcopy(theta)
        th.values[L.name_to_idx[n]] = NaN
        @test isnan(Nereus.rv_log_likelihood(th, data, ws))
        @test isnan(Nereus.rv_log_likelihood(th, data))
    end
end

@testset "AGP whitened solver: derivatives at and near singular blocks" begin
    # R_j is a square root of the 2×2 block B_j, so it is not differentiable
    # where B_j is singular: with every G coupling (or every Ġ coupling) at
    # exactly 0 the dual-number path used to drop those couplings' derivatives
    # (an error of 11.7 against a gradient of 24.5), and near such points, or
    # when every coupling is tiny, the derivatives through R lost their
    # accuracy. Dual inputs there now take the dense likelihood.
    rng = MersenneTwister(1)
    N, C = 20, 5
    epochs = sort!(60 .* rand(rng, N))
    σ² = 0.1 .+ rand(rng, C * N); y = randn(rng, C * N)
    hyp = (1.0, 9.0, 30.0, 0.6)
    lr(a, b, yy, s) = Nereus.activity_gp_joint_logpdf_lowrank(epochs, a, b, hyp..., yy, s)
    dn(a, b, yy, s) = _agp_dense(epochs, a, b, hyp..., yy, s)
    a0 = [1.0, 0.5, -0.7, 0.3, 0.8]; b0 = [0.4, -0.2, 0.3, 0.1, 0.0]
    nz = [0.3, -0.5, 0.2, 0.9, -0.4]
    split5(x) = (x[1:C], x[C+1:2C])
    function grad_err(split, x, yy = y, s = σ²)
        g1 = ForwardDiff.gradient(v -> lr(split(v)..., yy, s), x)
        g2 = ForwardDiff.gradient(v -> dn(split(v)..., yy, s), x)
        return maximum(abs.(g1 .- g2)) / max(maximum(abs.(g2)), floatmin()), g2
    end

    # Exactly singular blocks.
    for (what, x) in (("every G coupling 0", vcat(zeros(C), b0)),
                      ("every Ġ coupling 0", vcat(a0, zeros(C))),
                      ("Ġ couplings ∝ G couplings", vcat(a0, 0.5 .* a0)),
                      ("every coupling 0", zeros(2C)))
        e, g = grad_err(split5, x)
        @test e <= 1e-10
        # The couplings that vanish do have a derivative (except at all 0).
        what == "every coupling 0" || @test maximum(abs, g) > 1
    end
    # Two channels, the second without a Ġ coupling (as logR'HK): singular
    # exactly when its G coupling is 0.
    two(x) = ([x[1], x[2]], [x[3], zero(x[3])])
    y2 = y[1:2N]; s2 = σ²[1:2N]
    @test grad_err(two, [1.0, 0.0, 0.4], y2, s2)[1] <= 1e-10

    # Approaching them, and every coupling tiny: no cliff in the accuracy.
    nod(x) = (x, zero(x))       # use_derivative = false: 1×1 blocks
    for k in (2, 4, 6, 8, 10, 12, 16, 30, 100, 200)
        ϵ = 10.0^-k
        @test grad_err(split5, vcat(ϵ .* a0, b0))[1] <= 1e-8
        @test grad_err(split5, vcat(a0, ϵ .* b0))[1] <= 1e-8
        @test grad_err(split5, vcat(a0, 0.7 .* a0 .+ ϵ .* nz))[1] <= 1e-8
        @test grad_err(two, [1.0, ϵ, 0.4], y2, s2)[1] <= 1e-8
        # Every coupling tiny. Below ~1e-150 the Float64 dense gradient
        # underflows (ε² does), so the reference there is in BigFloat.
        if k <= 100
            @test grad_err(split5, ϵ .* vcat(a0, b0 .+ 0.1))[1] <= 1e-8
            @test grad_err(nod, ϵ .* a0)[1] <= 1e-8
        else
            for (sp, x) in ((split5, ϵ .* vcat(a0, b0 .+ 0.1)), (nod, ϵ .* a0))
                g1 = ForwardDiff.gradient(v -> lr(sp(v)..., y, σ²), x)
                gb = setprecision(BigFloat, 256) do
                    ForwardDiff.gradient(v -> _agp_dense(big.(epochs), sp(v)..., big.(hyp)...,
                                                         big.(y), big.(σ²)), big.(x))
                end
                @test maximum(abs.(g1 .- gb)) <= 1e-8 * maximum(abs.(gb))
            end
        end
    end

    # Second derivatives (nested duals) at a singular block and a generic one.
    n3 = 8; C3 = 3
    ep3 = epochs[1:n3]; y3 = y[1:(C3 * n3)]; s3 = σ²[1:(C3 * n3)]
    split3(x) = (x[1:C3], x[C3+1:2C3])
    f3l(x) = Nereus.activity_gp_joint_logpdf_lowrank(ep3, split3(x)..., hyp..., y3, s3)
    f3d(x) = _agp_dense(ep3, split3(x)..., hyp..., y3, s3)
    for x in (vcat(a0[1:3], zeros(3)), vcat(a0[1:3], b0[1:3]))
        H1 = ForwardDiff.hessian(f3l, x); H2 = ForwardDiff.hessian(f3d, x)
        @test H1 ≈ H2 rtol = 1e-9
    end

    # Which inputs take the dense route. Constant zeros (no coupling at all)
    # do not; couplings that are parameters at 0 do.
    D(v, k, n) = ForwardDiff.Dual{:t}(v, ntuple(i -> Float64(i == k), n))
    function smooth(a, b)
        n = length(a) + length(b)
        ad = [D(a[c], c, n) for c in eachindex(a)]
        bd = [b[c] === nothing ? D(0.0, 0, n) : D(b[c], length(a) + c, n) for c in eachindex(b)]
        Nv = 4; s = fill(0.5, length(a) * Nv)
        gG = [sum(ad[c]^2 / s[(c - 1) * Nv + j] for c in eachindex(a)) for j in 1:Nv]
        gd = [sum(ad[c] * bd[c] / s[(c - 1) * Nv + j] for c in eachindex(a)) for j in 1:Nv]
        dd = [sum(bd[c]^2 / s[(c - 1) * Nv + j] for c in eachindex(a)) for j in 1:Nv]
        return Nereus._agp_blocks_smooth(gG, gd, dd, ad, bd, 1.0, 0.1)
    end
    @test smooth(a0, b0)
    @test smooth(a0, fill(nothing, C))                # use_derivative = false
    @test !smooth(zeros(C), b0)
    @test !smooth(a0, zeros(C))
    @test !smooth(a0, 0.5 .* a0)
    @test !smooth(1e-9 .* a0, fill(nothing, C))       # SNR ~ 1e-18
    @test !smooth([NaN; a0[2:end]], b0)
    # Float64 inputs never take it: values are exact at singular blocks.
    for (a, b) in ((zeros(C), b0), (a0, zeros(C)), (a0, 0.5 .* a0))
        @test lr(a, b, y, σ²) ≈ _agp_exact_dense(epochs, a, b, hyp..., y, σ²) atol = 1e-10
    end

    # Through the model: the gradient of an ActivityGP target with every G
    # coupling, or every Ġ coupling, exactly 0 agrees with central
    # differences.
    data, params, _ = _agp_params(MersenneTwister(5))
    target = NereusTarget(params, data)
    L = params.layout
    lp(v) = Nereus.LogDensityProblems.logdensity(target, v)
    for zeroed in (["Vc", "Bc", "Fc", "Hc", "Lc"], ["Vr", "Br", "Fr", "Hr"])
        th = Theta{Float64}(params)
        _prior_theta!(th, params, MersenneTwister(6))
        for n in zeroed; th.values[L.name_to_idx[n]] = 0.0; end
        yv = Nereus.transform_forward([th.values[i] for i in L.unfrozen_idx],
                                      target.transform)
        @test isfinite(lp(yv))
        g = ForwardDiff.gradient(lp, yv)
        h = 1e-6
        for (d, n) in enumerate(L.unfrozen_names)
            e = zeros(length(yv)); e[d] = h
            fd = (lp(yv .+ e) - lp(yv .- e)) / 2h
            @test g[d] ≈ fd rtol = 1e-4 atol = 1e-5
        end
    end
end
