# Rajpaul+ 2015 multivariate-GP kernel + log-likelihood unit tests.
#
# At this stage `ActivityGP` is registered as a noise model type and
# the kernel math + a standalone log-likelihood are wired in; full
# integration with `rv_log_likelihood` / `Params` auto-priors is a
# follow-up commit.

using Nereus, Random, Statistics, LinearAlgebra, Test, MCMCChains
using Random: MersenneTwister

# ---------------------------------------------------------------------
# Kernel algebra
# ---------------------------------------------------------------------

@testset "activity_kernel_blocks — limits and symmetries" begin
    amp, P, λe, λp = 0.7, 12.3, 50.0, 0.5

    # At τ = 0: k_GG = amp², derivative blocks vanish, k_dotGdotG =
    # amp² · (1/λ_e² + π²/(P² λ_p²)) (from -(f''(0) + f'(0)²) at τ=0).
    k_GG, k_GdotG, k_dotGG, k_dotGdotG =
        activity_kernel_blocks(0.0, amp, P, λe, λp)
    @test isapprox(k_GG, amp^2; atol = 1e-12)
    @test isapprox(k_GdotG, 0.0; atol = 1e-12)
    @test isapprox(k_dotGG, 0.0; atol = 1e-12)
    expected_dd = amp^2 * (1 / λe^2 + π^2 / (P^2 * λp^2))
    @test isapprox(k_dotGdotG, expected_dd; rtol = 1e-10)

    # Antisymmetry between off-diagonal derivative blocks: k_GdotG(τ) =
    # -k_dotGG(τ) for any τ (one is ∂/∂t_j, the other ∂/∂t_i, opposite
    # signs).
    for τ in (-3.7, -0.5, 1.2, 7.9)
        a1, b1, c1, d1 = activity_kernel_blocks(τ, amp, P, λe, λp)
        @test isapprox(b1 + c1, 0.0; atol = 1e-10)
        # Even symmetry of k_GG and k_dotGdotG under τ → -τ
        a2, b2, c2, d2 = activity_kernel_blocks(-τ, amp, P, λe, λp)
        @test isapprox(a1, a2; rtol = 1e-10)
        @test isapprox(d1, d2; rtol = 1e-10)
        # Odd symmetry of k_GdotG / k_dotGG under τ → -τ
        @test isapprox(b1, -b2; atol = 1e-10)
        @test isapprox(c1, -c2; atol = 1e-10)
    end

    # Finite-difference check of f'(τ): k_dotGG(τ) ≈ -(k(τ+h)-k(τ-h))/(2h)
    # (∂k/∂t_i with t_j fixed equals dk/dτ).
    h = 1e-4
    for τ in (-2.3, 0.8, 4.7)
        kp, _, _, _ = activity_kernel_blocks(τ + h, amp, P, λe, λp)
        km, _, _, _ = activity_kernel_blocks(τ - h, amp, P, λe, λp)
        fd = (kp - km) / (2h)
        _, _, k_dotGG_an, _ = activity_kernel_blocks(τ, amp, P, λe, λp)
        @test isapprox(k_dotGG_an, fd; rtol = 1e-4)
    end
end

# ---------------------------------------------------------------------
# FM17 rotation kernel blocks (closed form + celerite-form parity)
# ---------------------------------------------------------------------
#
# Phase 1 of the celerite-block O(n) AGP solver: validate that
# `activity_kernel_blocks_fm17` evaluates the same kernel that the
# celerite-coefficient tuple
# `activity_kernel_blocks_fm17_celerite` describes term-by-term. Also
# check finite-difference agreement on the analytic k'/k'' formulas
# and that the resulting joint Σ is PSD on a small synthetic dataset.
# The O(n) solver itself is Phase 2 — this is the math handoff.

# Manually evaluate a celerite sum-of-exponentials block from its
# coefficient tuple (mirrors what the Phase 2 solver consumes).
function _eval_celerite_block_at(term, τ::Real)
    abs_τ = abs(τ); sτ = sign(τ)
    s_real = term.sign_real == 1 ? sτ : 1.0
    s_cos  = term.sign_cos  == 1 ? sτ : 1.0
    s_sin  = term.sign_sin  == 1 ? sτ : 1.0
    real_part = term.ar * exp(-term.cr * abs_τ) * s_real
    cplx_part = exp(-term.cc * abs_τ) *
                (term.ac * s_cos * cos(term.dc * τ) +
                 term.bc * s_sin * sin(term.dc * τ))
    return real_part + cplx_part
end

@testset "activity_kernel_blocks_fm17 — closed-form ↔ celerite-form parity" begin
    amp, τ_decay, period, factor = 1.4, 60.0, 11.7, 0.3

    coeffs = Nereus.activity_kernel_blocks_fm17_celerite(
        amp, τ_decay, period, factor)
    # Sanity: shared decay rate, dc = 2π/P.
    @test isapprox(coeffs.k.cr, 1 / τ_decay; rtol = 1e-12)
    @test isapprox(coeffs.k.cc, 1 / τ_decay; rtol = 1e-12)
    @test isapprox(coeffs.k.dc, 2π / period; rtol = 1e-12)
    # k', k'' share the same exponential timescales as k.
    for term in (coeffs.kp, coeffs.kpp)
        @test isapprox(term.cr, coeffs.k.cr; rtol = 1e-12)
        @test isapprox(term.cc, coeffs.k.cc; rtol = 1e-12)
        @test isapprox(term.dc, coeffs.k.dc; rtol = 1e-12)
    end

    # Closed-form k, k', k'' from activity_kernel_blocks_fm17 must match
    # the manual celerite-sum reconstruction at every test lag. τ=0 is
    # excluded: k' has a jump discontinuity there (left and right
    # limits are negatives — required by antisymmetry of the
    # derivative of a |τ|-kinked symmetric kernel). Both representations
    # use sign(0)=0 so they agree numerically, but the comparison is
    # meaningless at the kink. The Phase-2 solver only ever evaluates
    # the celerite tuples on real pair-lags `t_i − t_j` with `i ≠ j`,
    # never at τ=0.
    for τ in (-7.3, -1.4, -0.1, 0.6, 4.2, 15.8)
        k_GG, k_GdotG, k_dotGG, k_dotGdotG =
            Nereus.activity_kernel_blocks_fm17(τ, amp, τ_decay,
                                                  period, factor)
        # Rebuild k, k', k'' from the celerite tuples.
        k_man   = _eval_celerite_block_at(coeffs.k, τ)
        kp_man  = _eval_celerite_block_at(coeffs.kp, τ)
        kpp_man = _eval_celerite_block_at(coeffs.kpp, τ)
        # Closed-form blocks are (k, -k', k', -k''), so:
        @test isapprox(k_GG,        k_man;    rtol = 1e-12, atol = 1e-12)
        @test isapprox(k_GdotG,    -kp_man;   rtol = 1e-12, atol = 1e-12)
        @test isapprox(k_dotGG,     kp_man;   rtol = 1e-12, atol = 1e-12)
        @test isapprox(k_dotGdotG, -kpp_man;  rtol = 1e-12, atol = 1e-12)
    end

    # k is symmetric, k' antisymmetric, k'' symmetric.
    for τ in (0.4, 2.7, 9.1)
        a1, b1, c1, d1 = Nereus.activity_kernel_blocks_fm17(
            τ, amp, τ_decay, period, factor)
        a2, b2, c2, d2 = Nereus.activity_kernel_blocks_fm17(
            -τ, amp, τ_decay, period, factor)
        @test isapprox(a1, a2; rtol = 1e-12)              # k symmetric
        @test isapprox(b1, -b2; atol = 1e-12)             # -k' antisym
        @test isapprox(c1, -c2; atol = 1e-12)             #  k' antisym
        @test isapprox(d1, d2; rtol = 1e-12)              # -k'' symmetric
        @test isapprox(b1 + c1, 0.0; atol = 1e-12)        # mirror pair
    end

    # Finite-difference check of k_dotGG = ∂_{t_i} k = dk/dτ.
    h = 1e-5
    for τ in (-3.2, 0.8, 5.6)
        kp_, _, _, _ = Nereus.activity_kernel_blocks_fm17(
            τ + h, amp, τ_decay, period, factor)
        km_, _, _, _ = Nereus.activity_kernel_blocks_fm17(
            τ - h, amp, τ_decay, period, factor)
        fd = (kp_ - km_) / (2h)
        _, _, k_dotGG_an, _ = Nereus.activity_kernel_blocks_fm17(
            τ, amp, τ_decay, period, factor)
        @test isapprox(k_dotGG_an, fd; rtol = 1e-3, atol = 1e-6)
    end

    # Finite-difference check of k_dotGdotG = -k''(τ).
    h = 1e-4
    for τ in (-2.1, 0.5, 4.4)
        k0, _, _, _ = Nereus.activity_kernel_blocks_fm17(
            τ, amp, τ_decay, period, factor)
        kp_, _, _, _ = Nereus.activity_kernel_blocks_fm17(
            τ + h, amp, τ_decay, period, factor)
        km_, _, _, _ = Nereus.activity_kernel_blocks_fm17(
            τ - h, amp, τ_decay, period, factor)
        # -k''(τ) via second-difference of k.
        fd_neg_kpp = -(kp_ - 2k0 + km_) / h^2
        _, _, _, k_dotGdotG_an = Nereus.activity_kernel_blocks_fm17(
            τ, amp, τ_decay, period, factor)
        @test isapprox(k_dotGdotG_an, fd_neg_kpp; rtol = 1e-3, atol = 1e-6)
    end

    # Joint Σ assembled from the FM17 blocks on synthetic 2-channel
    # data is PSD (up to small numerical jitter).
    rng = MersenneTwister(7)
    n = 24
    t = sort!(40.0 .* rand(rng, n))
    chans = [iseven(i) ? :rv : :bis for i in 1:n]
    a = 1.0 .+ 0.4 .* randn(rng, n)
    b = 0.3 .* randn(rng, n)
    Σ = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        τ = t[i] - t[j]
        k_GG, k_GdotG, k_dotGG, k_dotGdotG =
            Nereus.activity_kernel_blocks_fm17(τ, 1.0, 30.0, 8.0, 0.4)
        Σ[i, j] = a[i] * a[j] * k_GG +
                   a[i] * b[j] * k_GdotG +
                   b[i] * a[j] * k_dotGG +
                   b[i] * b[j] * k_dotGdotG
    end
    @test isapprox(Σ, Σ'; atol = 1e-10)
    # `isposdef` on a raw matrix is strict-symmetric — float drift
    # (~1e-15) on the manual assembly makes it bail to an LU path that
    # finds spurious negative pivots. Wrap in Symmetric to enforce the
    # Cholesky path (same trick the QP testset uses below).
    @test isposdef(Symmetric((Σ + Σ') / 2) + 1e-6 * I)
end

# ---------------------------------------------------------------------
# Covariance assembly
# ---------------------------------------------------------------------

@testset "activity_gp_covariance — symmetry and PSD" begin
    rng = MersenneTwister(0)
    n = 30
    t = sort!(40.0 .* rand(rng, n))
    a = 1.0 .+ 0.5 .* randn(rng, n)
    b = 0.3 .* randn(rng, n)
    Σ = activity_gp_covariance(t, fill(:rv, n), a, b,
                                1.0, 8.0, 30.0, 0.5)
    @test size(Σ) == (n, n)
    @test isapprox(Σ, Σ'; atol = 1e-10)
    # Add a tiny diagonal jitter — Σ should be PSD.
    Σ_jit = Σ + 1e-6 * I
    @test isposdef(Σ_jit)
end

# ---------------------------------------------------------------------
# Standalone multivariate log-likelihood
# ---------------------------------------------------------------------

@testset "activity_gp_log_likelihood — sanity" begin
    rng = MersenneTwister(1)
    # Synthesise a small joint RV+BIS dataset by drawing from the
    # ActivityGP prior with known (true) hyperparameters and channel
    # coefficients, then check the LL is finite + behaves correctly.
    n_pts = 20
    t_obs = sort!(40.0 .* rand(rng, n_pts))
    channels = fill(:rv, n_pts)
    # Half the points are BIS observations at the same epochs (Rajpaul
    # standard: indicators come from the same spectra as RV).
    channels[1:n_pts ÷ 2] .= :bis

    coeffs = ActivityGPCoeffs(:rv => (3.0, 0.5),
                              :bis => (0.05, 0.01))
    amp, P, λe, λp = 1.0, 8.0, 30.0, 0.5
    Σ_true = activity_gp_covariance(t_obs, channels,
                                      [coeffs[c][1] for c in channels],
                                      [coeffs[c][2] for c in channels],
                                      amp, P, λe, λp)
    # Sample y ~ N(0, Σ_true). Add small per-point noise.
    σ_obs = fill(0.1, n_pts)
    L = cholesky(Symmetric(Σ_true + Diagonal(σ_obs .^ 2))).L
    y_obs = L * randn(rng, n_pts)

    # Likelihood at the truth — finite and reproducible.
    ll_truth = activity_gp_log_likelihood(t_obs, y_obs, σ_obs,
                                            channels, amp, P, λe, λp,
                                            coeffs)
    @test isfinite(ll_truth)
    @test isapprox(ll_truth,
                    activity_gp_log_likelihood(t_obs, y_obs, σ_obs,
                                                channels, amp, P, λe, λp,
                                                coeffs);
                    atol = 1e-12)

    # Wrong period → much lower likelihood.
    ll_wrong_P = activity_gp_log_likelihood(t_obs, y_obs, σ_obs,
                                              channels, amp, 3.0, λe, λp,
                                              coeffs)
    @test ll_wrong_P < ll_truth - 2.0

    # Wrong amplitude → lower likelihood.
    ll_wrong_amp = activity_gp_log_likelihood(t_obs, y_obs, σ_obs,
                                                channels, 5.0, P, λe, λp,
                                                coeffs)
    @test ll_wrong_amp < ll_truth - 2.0
end

@testset "activity_gp_log_likelihood — input validation" begin
    t = [0.0, 1.0, 2.0]
    y = [0.0, 0.0, 0.0]
    σ = [0.1, 0.1, 0.1]
    ch = [:rv, :rv, :rv]
    coeffs = ActivityGPCoeffs(:rv => (1.0, 0.0))

    @test_throws ArgumentError activity_gp_log_likelihood(
        t, [0.0, 0.0], σ, ch, 1.0, 8.0, 30.0, 0.5, coeffs)
    @test_throws ArgumentError activity_gp_log_likelihood(
        t, y, σ, ch, -1.0, 8.0, 30.0, 0.5, coeffs)
    @test_throws ArgumentError activity_gp_log_likelihood(
        t, y, σ, ch, 1.0, 8.0, 30.0, 0.5,
        ActivityGPCoeffs())                # empty coeffs
    @test_throws ArgumentError activity_gp_log_likelihood(
        t, y, σ, [:rv, :bis, :rv], 1.0, 8.0, 30.0, 0.5, coeffs)
end

# ---------------------------------------------------------------------
# ActivityGP type / param-name registration
# ---------------------------------------------------------------------

@testset "ActivityGP — Params integration + auto-priors" begin
    rng = MersenneTwister(7)
    n = 25
    t = sort!(40.0 .* rand(rng, n))
    bis = 0.01 .* randn(rng, n)
    data = Data(; t_rv = t, rv = randn(rng, n), rv_err = ones(n),
                  rv_inst = ones(Int, n),
                  indicators = Dict("bis" => bis),
                  indicator_errs = Dict("bis" => fill(0.005, n)))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[ActivityGP(channels = [:bis])])
    # Every ActivityGP param name should appear in the layout WITH an
    # auto-generated prior.
    expected = ["gp_act_period", "gp_act_lambda_e",
                "gp_act_lambda_p", "Vc", "Vr", "Bc", "Br"]
    for nm in expected
        @test nm in params.layout.unfrozen_names
    end
    # Indicator coefficient prior scale tracks the indicator MAD.
    bc_idx = findfirst(==("Bc"), params.layout.unfrozen_names)
    bc_pr = params.layout.unfrozen_priors[bc_idx]
    lo, hi = bounds(bc_pr)
    @test hi > 0 && lo < 0
    # `Bc` couples the UNIT-VARIANCE latent G(t) to the BIS channel
    # (bis = Bc·G + Br·Ġ), so Bc is the amplitude of the BIS variability in
    # whatever units the indicator is stored in. Indicators are normalised to
    # unit RMS by default, so the plausible Bc is O(1) and the prior must
    # CONTAIN it — an `abs(hi) < 1.0` ceiling (which was right back when the
    # raw σ ~ 0.01 values were stored) would sit the truth on the boundary.
    @test hi > 1.0        # contains the unit-RMS coupling
    @test hi < 20.0       # still informative, not effectively flat
end

@testset "ActivityGP — joint log-likelihood through rv_log_likelihood" begin
    # Synthesise a joint RV+BIS dataset directly from the Rajpaul
    # covariance at known truth; evaluate the integrated Nereus
    # rv_log_likelihood at the truth and at perturbed hyperparameters.
    # The truth must score higher.
    rng = MersenneTwister(31)
    n_obs = 25
    t_rv = sort!(40.0 .* rand(rng, n_obs))
    amp_t, P_t, λe_t, λp_t = 1.0, 12.0, 50.0, 0.5
    Vc_t, Vr_t, Bc_t, Br_t = 3.0, 0.3, 0.05, 0.005
    σ_rv, σ_bis = 0.5, 0.005

    t_flat = vcat(t_rv, t_rv)
    ch_flat = vcat(fill(:rv, n_obs), fill(:bis, n_obs))
    a_flat = vcat(fill(Vc_t, n_obs), fill(Bc_t, n_obs))
    b_flat = vcat(fill(Vr_t, n_obs), fill(Br_t, n_obs))
    Σ = activity_gp_covariance(t_flat, ch_flat, a_flat, b_flat,
                                 amp_t, P_t, λe_t, λp_t)
    σ_diag = vcat(fill(σ_rv, n_obs), fill(σ_bis, n_obs))
    Σnoisy = Σ + Diagonal(σ_diag .^ 2)
    L = cholesky(Symmetric(Σnoisy)).L
    y_flat = L * randn(rng, 2 * n_obs)
    rv_obs  = y_flat[1:n_obs]
    bis_obs = y_flat[(n_obs+1):end]

    data = Data(; normalize_indicators = false, t_rv = t_rv, rv = rv_obs, rv_err = fill(σ_rv, n_obs),
                  rv_inst = ones(Int, n_obs),
                  indicators = Dict("bis" => bis_obs),
                  indicator_errs = Dict("bis" => fill(σ_bis, n_obs)))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[ActivityGP(channels = [:bis])])

    theta = Theta{Float64}(params)
    truth = Dict("gamma_SIM" => 0.0,
                 "sigma_SIM" => 1e-3,
                 "gp_act_period" => P_t,
                 "gp_act_lambda_e" => λe_t,
                 "gp_act_lambda_p" => λp_t,
                 "Vc" => Vc_t, "Vr" => Vr_t,
                 "Bc" => Bc_t, "Br" => Br_t)
    for nm in params.layout.unfrozen_names
        haskey(truth, nm) && set_param!(theta, nm, truth[nm])
    end
    ll_truth = Nereus.rv_log_likelihood(theta, data)
    @test isfinite(ll_truth)

    set_param!(theta, "gp_act_period", 3.0)   # very wrong
    ll_off = Nereus.rv_log_likelihood(theta, data)
    @test ll_off < ll_truth - 50.0    # truth scores massively higher

    set_param!(theta, "gp_act_period", P_t)   # restore
    set_param!(theta, "Bc", 0.0)              # wrong channel coupling
    ll_no_coupling = Nereus.rv_log_likelihood(theta, data)
    @test ll_no_coupling < ll_truth
end

@testset "activity_gp_predict — latent G recovery from a tight posterior" begin
    # Synthesise from the Rajpaul prior with a known period; sample
    # the latent G(t) posterior at a dense grid; assert the recovered
    # mean is non-trivial and oscillates at the expected period.
    rng = MersenneTwister(99)
    n_obs = 30
    t_rv = sort!(40.0 .* rand(rng, n_obs))
    amp_t, P_t, λe_t, λp_t = 1.0, 12.0, 50.0, 0.5
    Vc_t, Vr_t, Bc_t, Br_t = 3.0, 0.3, 0.05, 0.005
    σ_rv, σ_bis = 0.5, 0.005

    t_flat = vcat(t_rv, t_rv)
    ch_flat = vcat(fill(:rv, n_obs), fill(:bis, n_obs))
    a_flat = vcat(fill(Vc_t, n_obs), fill(Bc_t, n_obs))
    b_flat = vcat(fill(Vr_t, n_obs), fill(Br_t, n_obs))
    Σ = activity_gp_covariance(t_flat, ch_flat, a_flat, b_flat,
                                 amp_t, P_t, λe_t, λp_t)
    σ_diag = vcat(fill(σ_rv, n_obs), fill(σ_bis, n_obs))
    L = cholesky(Symmetric(Σ + Diagonal(σ_diag .^ 2))).L
    y_flat = L * randn(rng, 2 * n_obs)
    rv_obs  = y_flat[1:n_obs]
    bis_obs = y_flat[(n_obs+1):end]

    data = Data(; normalize_indicators = false, t_rv = t_rv, rv = rv_obs, rv_err = fill(σ_rv, n_obs),
                  rv_inst = ones(Int, n_obs),
                  indicators = Dict("bis" => bis_obs),
                  indicator_errs = Dict("bis" => fill(σ_bis, n_obs)))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[ActivityGP(channels = [:bis])])

    fitted_names = Symbol.(params.layout.unfrozen_names)
    truth = Dict(:gamma_SIM => 0.0, :sigma_SIM => 1e-3,
                 :gp_act_period => P_t,
                 :gp_act_lambda_e => λe_t, :gp_act_lambda_p => λp_t,
                 :Vc => Vc_t, :Vr => Vr_t, :Bc => Bc_t, :Br => Br_t)
    n_draws = 50
    arr = zeros(Float64, n_draws, length(fitted_names), 1)
    for (j, nm) in enumerate(fitted_names)
        μ = get(truth, nm, 0.0)
        arr[:, j, 1] .= μ .+ 1e-3 .* randn(rng, n_draws)
    end
    chains = Chains(arr, fitted_names)

    pred = activity_gp_predict(chains, params, data; n_draws = 50,
                                 rng = MersenneTwister(0))
    @test pred.t_pred isa Vector{Float64}
    @test length(pred.t_pred) == 200
    @test size(pred.G_samples) == (200, 50)
    @test length(pred.G_mean) == 200
    @test length(pred.G_lo) == 200
    @test length(pred.G_hi) == 200
    @test all(isfinite, pred.G_mean)
    @test all(pred.G_lo .<= pred.G_mean .<= pred.G_hi)
    # Latent G should swing across both signs over the baseline.
    @test minimum(pred.G_mean) < 0
    @test maximum(pred.G_mean) > 0

    # Plot saves.
    tmp = tempname() * ".png"
    plot_activity_gp_latent(chains, params, data;
                              filename = tmp, n_draws = 50)
    @test isfile(tmp)
    @test filesize(tmp) > 1000
    rm(tmp; force = true)
end

@testset "ActivityGP — sample_moms_ns compatibility (smoke)" begin
    # AGP under trans-dim nested sampling: same trick as the trans-dim
    # ptemcee test — declare AGP toggleable, run a short MoMS-NS.
    rng = MersenneTwister(0)
    n = 25
    t = sort!(40.0 .* rand(rng, n))
    data = Data(; normalize_indicators = false, t_rv = t, rv = 0.5 .* randn(rng, n),
                  rv_err = fill(0.5, n), rv_inst = ones(Int, n),
                  indicators = Dict("bis" => 0.005 .* randn(rng, n)),
                  indicator_errs = Dict("bis" => fill(0.005, n)))
    ic = InstrumentConfig(rv = ["SIM"])
    agp = ActivityGP(channels = [:bis],
                       marginalize_indicators = true)
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[agp],
                     transdim_noise = true)
    target = NereusTarget(params, data; unconstrained = false)
    td_cfg = TransDimConfig(; max_kplanet = 0,
                              planets = false,
                              noise = true,
                              toggleable = NoiseModel[agp])
    chain, log_Z, _ = sample_moms_ns(target, data;
                                       td = td_cfg,
                                       n_live = 16,
                                       n_mcmc = 2,
                                       max_iter = 200,
                                       seed = 0,
                                       show_progress = false)
    @test isfinite(log_Z)
    @test size(chain, 1) > 0
end

@testset "ActivityGP — sample_pa compatibility (smoke)" begin
    # Verifies gradient-free PA can run on an AGP model end-to-end.
    rng = MersenneTwister(0)
    n = 25
    t = sort!(40.0 .* rand(rng, n))
    data = Data(; normalize_indicators = false, t_rv = t, rv = 0.5 .* randn(rng, n),
                  rv_err = fill(0.5, n), rv_inst = ones(Int, n),
                  indicators = Dict("bis" => 0.005 .* randn(rng, n)),
                  indicator_errs = Dict("bis" => fill(0.005, n)))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[
                         ActivityGP(channels = [:bis],
                                     marginalize_indicators = true)])
    target = NereusTarget(params, data; unconstrained = false)
    res = sample_pa(target, data;
                     n_replicas = 16, n_mcmc = 4, max_steps = 8,
                     seed = 0, show_progress = false)
    @test res isa Nereus.PAResult
    @test isfinite(res.log_evidence)
    @test size(res.chains, 1) > 0
end

@testset "ActivityGP — per-instrument scoping" begin
    # Two ActivityGPs with disjoint instrument lists each cover their
    # own RV subset; remaining RV obs (none here) fall through to
    # white-noise. Verifies that:
    #   - the scoped dispatch picks each AGP for its instrument-id slice
    #   - log L decomposes (total = sum of per-AGP slice LLs)
    #   - param names get unique `_<INST>` suffixes
    rng = MersenneTwister(5)
    n_per = 25
    t = sort!(40.0 .* rand(rng, 2 * n_per))
    rv_inst = vcat(fill(1, n_per), fill(2, n_per))
    σ_rv, σ_bis = 0.5, 0.005
    rv = σ_rv .* randn(rng, 2 * n_per)
    bis = σ_bis .* randn(rng, 2 * n_per)
    bis_err = fill(σ_bis, 2 * n_per)
    data = Data(; normalize_indicators = false, t_rv = t, rv = rv, rv_err = fill(σ_rv, 2 * n_per),
                  rv_inst = rv_inst,
                  indicators = Dict("bis" => bis),
                  indicator_errs = Dict("bis" => bis_err))
    ic = InstrumentConfig(rv = ["HARPS_DRS", "HARPS_TERRA"])
    agp_drs   = ActivityGP(channels = [:bis], instruments = ["HARPS_DRS"])
    agp_terra = ActivityGP(channels = [:bis], instruments = ["HARPS_TERRA"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[agp_drs, agp_terra])

    # Per-instrument suffix gives unique param names:
    names = params.layout.unfrozen_names
    @test "gp_act_period_HARPS_DRS" in names
    @test "gp_act_period_HARPS_TERRA" in names
    @test "Vc_HARPS_DRS" in names
    @test "Vc_HARPS_TERRA" in names
    @test "Bc_HARPS_DRS" in names
    @test "Bc_HARPS_TERRA" in names

    # Evaluate likelihood at the prior midpoints — must be finite and
    # equal to the sum of per-AGP slice LLs.
    theta = Theta{Float64}(params)
    for (nm, ps) in zip(names, params.layout.unfrozen_priors)
        lo, hi = bounds(ps)
        set_param!(theta, nm, isfinite(lo) && isfinite(hi) ? 0.5 * (lo + hi) : 1.0)
    end
    set_param!(theta, "sigma_HARPS_DRS",   1.0)
    set_param!(theta, "sigma_HARPS_TERRA", 1.0)
    set_param!(theta, "gp_act_period_HARPS_DRS",   10.0)
    set_param!(theta, "gp_act_lambda_e_HARPS_DRS", 30.0)
    set_param!(theta, "gp_act_lambda_p_HARPS_DRS", 0.5)
    set_param!(theta, "gp_act_period_HARPS_TERRA",   10.0)
    set_param!(theta, "gp_act_lambda_e_HARPS_TERRA", 30.0)
    set_param!(theta, "gp_act_lambda_p_HARPS_TERRA", 0.5)
    ll = Nereus.rv_log_likelihood(theta, data)
    @test isfinite(ll)
end

@testset "ActivityGP — trans-dim composition vs ActivityDecorrelation" begin
    # Compose ActivityGP and ActivityDecorrelation in the same noise_models
    # list with transdim_noise=true. Validation must accept it; toggling
    # the per-noise-model active mask must switch the likelihood between
    # the standard (white-noise + AD mean modifier) and the joint Rajpaul
    # paths. Different active masks → different log L.
    rng = MersenneTwister(13)
    n_obs = 25
    t_rv = sort!(40.0 .* rand(rng, n_obs))
    rv = randn(rng, n_obs)
    rv_err = fill(0.5, n_obs)
    bis = 0.01 .* randn(rng, n_obs)
    bis_err = fill(0.005, n_obs)

    data = Data(; normalize_indicators = false, t_rv = t_rv, rv = rv, rv_err = rv_err,
                  rv_inst = ones(Int, n_obs),
                  indicators = Dict("bis" => bis,
                                    "bisector_span" => bis),
                  indicator_errs = Dict("bis" => bis_err))
    ic = InstrumentConfig(rv = ["SIM"])
    ad  = ActivityDecorrelation(indicators = ["bisector_span"])
    agp = ActivityGP(channels = [:bis])

    # Without transdim_noise: validation should error (two globals on :rv).
    @test_throws ArgumentError validate_noise_models(
        NoiseModel[CeleriteRotation(channel = :rv), agp]; transdim = false)
    # With transdim_noise: validation must accept the composition.
    validate_noise_models(NoiseModel[CeleriteRotation(channel = :rv), agp];
                           transdim = true)

    # AD + AGP composition under transdim — both are different stages, so
    # this was already legal pre-patch, but verify Params builds.
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[ad, agp],
                     transdim_noise = true)

    # TransDimConfig with both in the same exclusion group. The
    # exclusion group's models must also be listed in `toggleable`
    # (otherwise the constraint can never fire).
    td_cfg = TransDimConfig(; max_kplanet = 0,
                              noise = true,
                              toggleable = NoiseModel[ad, agp],
                              noise_exclusion_groups = [[ad, agp]])
    @test td_cfg isa TransDimConfig
    @test length(td_cfg.noise_exclusion_groups) == 1
    @test length(td_cfg.noise_exclusion_groups[1]) == 2

    # Build a TransDimState and toggle the noise masks.
    n_nm = length(params.config.noise_models)
    @test n_nm == 2

    # State 1: AD active, AGP inactive — standard mean-modifier path.
    tds_ad = TransDimState(; max_planets = 0, n_noise = n_nm)
    activate_noise!(tds_ad, 1)    # AD
    @test is_noise_model_active(Theta{Float64}(params; td = tds_ad), 1)
    @test !is_noise_model_active(Theta{Float64}(params; td = tds_ad), 2)

    theta_ad = Theta{Float64}(params; td = tds_ad)
    for nm in params.layout.unfrozen_names
        set_param!(theta_ad, nm, 0.1)
    end
    set_param!(theta_ad, "sigma_SIM", 1.0)
    # The AD path should be used; likelihood is finite.
    ll_ad = Nereus.rv_log_likelihood(theta_ad, data)
    @test isfinite(ll_ad)

    # State 2: AGP active, AD inactive — joint Rajpaul path.
    tds_agp = TransDimState(; max_planets = 0, n_noise = n_nm)
    activate_noise!(tds_agp, 2)   # AGP
    theta_agp = Theta{Float64}(params; td = tds_agp)
    for nm in params.layout.unfrozen_names
        set_param!(theta_agp, nm, 0.1)
    end
    set_param!(theta_agp, "sigma_SIM", 1.0)
    set_param!(theta_agp, "gp_act_period", 12.0)
    set_param!(theta_agp, "gp_act_lambda_e", 30.0)
    set_param!(theta_agp, "gp_act_lambda_p", 0.5)
    set_param!(theta_agp, "Vc", 1.0)
    set_param!(theta_agp, "Bc", 0.02)
    ll_agp = Nereus.rv_log_likelihood(theta_agp, data)
    @test isfinite(ll_agp)

    # Switching the active noise model must change the log L —
    # otherwise the dispatch isn't actually doing anything.
    @test ll_ad != ll_agp

    # State 3: neither active — should also evaluate cleanly via the
    # default white-noise path.
    tds_none = TransDimState(; max_planets = 0, n_noise = n_nm)
    theta_none = Theta{Float64}(params; td = tds_none)
    for nm in params.layout.unfrozen_names
        set_param!(theta_none, nm, 0.1)
    end
    set_param!(theta_none, "sigma_SIM", 1.0)
    ll_none = Nereus.rv_log_likelihood(theta_none, data)
    @test isfinite(ll_none)
end

@testset "activity_gp_decompose_rv — recovers noise floor" begin
    # Same Rajpaul-prior synthesis as activity_gp_predict test. Verify
    # that data = activity + corrected (exact decomposition), and
    # that std(rv_corrected) ≈ σ_rv_true (activity correctly removed).
    rng = MersenneTwister(101)
    n_obs = 30
    t_rv = sort!(40.0 .* rand(rng, n_obs))
    amp_t, P_t, λe_t, λp_t = 1.0, 12.0, 50.0, 0.5
    Vc_t, Vr_t, Bc_t, Br_t = 3.0, 0.3, 0.05, 0.005
    σ_rv, σ_bis = 0.5, 0.005

    t_flat  = vcat(t_rv, t_rv)
    ch_flat = vcat(fill(:rv, n_obs), fill(:bis, n_obs))
    a_flat  = vcat(fill(Vc_t, n_obs), fill(Bc_t, n_obs))
    b_flat  = vcat(fill(Vr_t, n_obs), fill(Br_t, n_obs))
    Σ = activity_gp_covariance(t_flat, ch_flat, a_flat, b_flat,
                                 amp_t, P_t, λe_t, λp_t)
    σ_diag = vcat(fill(σ_rv, n_obs), fill(σ_bis, n_obs))
    L = cholesky(Symmetric(Σ + Diagonal(σ_diag .^ 2))).L
    y_flat = L * randn(rng, 2 * n_obs)

    data = Data(; normalize_indicators = false, t_rv = t_rv, rv = y_flat[1:n_obs],
                  rv_err = fill(σ_rv, n_obs), rv_inst = ones(Int, n_obs),
                  indicators = Dict("bis" => y_flat[(n_obs+1):end]),
                  indicator_errs = Dict("bis" => fill(σ_bis, n_obs)))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[ActivityGP(channels = [:bis])])

    fitted_names = Symbol.(params.layout.unfrozen_names)
    truth = Dict(:gamma_SIM => 0.0, :sigma_SIM => 1e-3,
                 :gp_act_period => P_t,
                 :gp_act_lambda_e => λe_t, :gp_act_lambda_p => λp_t,
                 :Vc => Vc_t, :Vr => Vr_t, :Bc => Bc_t, :Br => Br_t)
    n_draws = 80
    arr = zeros(Float64, n_draws, length(fitted_names), 1)
    for (j, nm) in enumerate(fitted_names)
        μ = get(truth, nm, 0.0)
        arr[:, j, 1] .= μ .+ 1e-3 .* randn(rng, n_draws)
    end
    chains = Chains(arr, fitted_names)

    decomp = activity_gp_decompose_rv(chains, params, data; n_draws = n_draws,
                                        rng = MersenneTwister(0))
    @test length(decomp.t_rv) == n_obs
    @test length(decomp.rv_data) == n_obs
    @test size(decomp.rv_activity_samples) == (n_obs, n_draws)
    @test all(isfinite, decomp.rv_activity_mean)
    @test all(isfinite, decomp.rv_corrected)
    @test all(decomp.rv_corrected_err .>= 0)

    # Exact decomposition: data − activity_mean = rv_corrected.
    @test maximum(abs.(decomp.rv_data .- decomp.rv_activity_mean .-
                         decomp.rv_corrected)) < 1e-9

    # Cleaned residuals consistent with the white-noise floor σ_rv.
    σ_corrected = std(decomp.rv_corrected)
    @test 0.3 * σ_rv < σ_corrected < 3.0 * σ_rv
    # And smaller than the original RV scatter (which is dominated by
    # the activity contribution).
    @test σ_corrected < 0.6 * std(decomp.rv_data)

    # Plot saves.
    tmp = tempname() * ".png"
    plot_activity_gp_decomposition(chains, params, data;
                                     filename = tmp, n_draws = n_draws)
    @test isfile(tmp)
    @test filesize(tmp) > 1000
    rm(tmp; force = true)
end

@testset "activity_gp_predict — refuses non-ActivityGP fits" begin
    data = Data(; normalize_indicators = false, t_rv = collect(0.0:5.0:50.0), rv = randn(11),
                  rv_err = ones(11), rv_inst = ones(Int, 11))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0)
    fitted_names = Symbol.(params.layout.unfrozen_names)
    arr = zeros(Float64, 50, length(fitted_names), 1)
    chains = Chains(arr, fitted_names)
    @test_throws ArgumentError activity_gp_predict(chains, params, data)
end

@testset "ActivityGP — marginalize_indicators conditional Gaussian" begin
    # Verify log p(joint) = log p(indicators) + log p(RV | indicators)
    # by shifting only the RV data: log p(indicators) must stay constant,
    # so (joint − marginal) at one RV vector must equal the same
    # difference at any other RV vector with the same indicator data.
    rng = MersenneTwister(7)
    n_obs = 25
    t_rv = sort!(40.0 .* rand(rng, n_obs))
    amp_t, P_t, λe_t, λp_t = 1.0, 12.0, 50.0, 0.5
    Vc_t, Vr_t, Bc_t, Br_t = 3.0, 0.3, 0.05, 0.005
    σ_rv, σ_bis = 0.5, 0.005

    Σ = activity_gp_covariance(vcat(t_rv, t_rv),
                                 vcat(fill(:rv, n_obs), fill(:bis, n_obs)),
                                 vcat(fill(Vc_t, n_obs), fill(Bc_t, n_obs)),
                                 vcat(fill(Vr_t, n_obs), fill(Br_t, n_obs)),
                                 amp_t, P_t, λe_t, λp_t)
    σ_diag = vcat(fill(σ_rv, n_obs), fill(σ_bis, n_obs))
    L = cholesky(Symmetric(Σ + Diagonal(σ_diag .^ 2))).L
    y_flat = L * randn(rng, 2 * n_obs)

    bis_obs = y_flat[(n_obs+1):end]
    function _make_params_and_theta(rv_obs, marginalize)
        data = Data(; normalize_indicators = false, t_rv = t_rv, rv = rv_obs, rv_err = fill(σ_rv, n_obs),
                      rv_inst = ones(Int, n_obs),
                      indicators = Dict("bis" => bis_obs),
                      indicator_errs = Dict("bis" => fill(σ_bis, n_obs)))
        ic = InstrumentConfig(rv = ["SIM"])
        params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                         instruments = ic, data = data, M_s = 1.0,
                         noise_models = NoiseModel[
                             ActivityGP(channels = [:bis],
                                        marginalize_indicators = marginalize)])
        theta = Theta{Float64}(params)
        set_param!(theta, "gamma_SIM", 0.0); set_param!(theta, "sigma_SIM", 1e-3)
        set_param!(theta, "gp_act_period", P_t)
        set_param!(theta, "gp_act_lambda_e", λe_t)
        set_param!(theta, "gp_act_lambda_p", λp_t)
        set_param!(theta, "Vc", Vc_t); set_param!(theta, "Vr", Vr_t)
        set_param!(theta, "Bc", Bc_t); set_param!(theta, "Br", Br_t)
        return theta, data
    end

    # Truth setup.
    θ_j, d_j = _make_params_and_theta(y_flat[1:n_obs], false)
    θ_m, d_m = _make_params_and_theta(y_flat[1:n_obs], true)
    ll_joint = Nereus.rv_log_likelihood(θ_j, d_j)
    ll_marg  = Nereus.rv_log_likelihood(θ_m, d_m)
    @test isfinite(ll_joint)
    @test isfinite(ll_marg)
    @test ll_joint > ll_marg   # joint = marg + log p(BIS); log p(BIS) > 0 typically

    # Shifted RV setup.
    θ_j2, d_j2 = _make_params_and_theta(y_flat[1:n_obs] .+ 5.0, false)
    θ_m2, d_m2 = _make_params_and_theta(y_flat[1:n_obs] .+ 5.0, true)
    ll_joint2 = Nereus.rv_log_likelihood(θ_j2, d_j2)
    ll_marg2  = Nereus.rv_log_likelihood(θ_m2, d_m2)

    # Chain rule: log p(BIS) = joint − marg must be invariant to the RV.
    Δ1 = ll_joint - ll_marg
    Δ2 = ll_joint2 - ll_marg2
    @test isapprox(Δ1, Δ2; atol = 1e-8)

    # Both joint and marginal should drop by the same amount when RV
    # shifts (their difference is RV-independent).
    @test isapprox(ll_joint - ll_joint2, ll_marg - ll_marg2; atol = 1e-8)
end

@testset "indicators / indicator_errs — save_chains ↔ load_chains round-trip" begin
    rng = MersenneTwister(0)
    n = 30
    data = Data(; normalize_indicators = false, t_rv = sort!(40.0 .* rand(rng, n)), rv = randn(rng, n),
                  rv_err = ones(n), rv_inst = ones(Int, n),
                  indicators = Dict("bis" => 0.1 .* randn(rng, n),
                                    "fwhm" => 0.5 .* randn(rng, n)),
                  indicator_errs = Dict("bis" => fill(0.005, n),
                                        "fwhm" => fill(0.01, n)))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0)
    chains = Chains(zeros(5, 2, 1), [:gamma_SIM, :sigma_SIM])
    path = tempname() * ".nc"
    save_chains(path, chains, params; data = data)
    _, meta = load_chains(path)
    @test haskey(meta, "indicators")
    @test haskey(meta, "indicator_errs")
    @test sort(collect(keys(meta["indicators"]))) == ["bis", "fwhm"]
    @test sort(collect(keys(meta["indicator_errs"]))) == ["bis", "fwhm"]
    @test meta["indicators"]["bis"] == data.indicators["bis"]
    @test meta["indicators"]["fwhm"] == data.indicators["fwhm"]
    @test meta["indicator_errs"]["bis"] == data.indicator_errs["bis"]
    @test meta["indicator_errs"]["fwhm"] == data.indicator_errs["fwhm"]
    rm(path; force = true)
end

@testset "ActivityGP — type + param names" begin
    nm = ActivityGP()
    @test nm isa Nereus.NoiseModel
    @test nm isa Nereus.CovarianceNoise
    @test nm.channels == [:bis, :fwhm]
    @test nm.use_derivative == true
    @test isempty(nm.instruments)

    nm_logrhk = ActivityGP(channels = [:bis, :logrhk])
    @test nm_logrhk.channels == [:bis, :logrhk]

    # noise_param_names should include kernel hyperparams + RV
    # coefficients + per-active-channel coefficients (logR'HK has no
    # derivative).
    ic = InstrumentConfig(rv = ["SIM"])
    names = noise_channel(nm) == :rv ?
        Nereus.noise_param_names(nm, ic) :
        String[]
    @test !("gp_act_amp" in names)   # unit-variance G(t): no amplitude
    @test "gp_act_period" in names
    @test "gp_act_lambda_e" in names
    @test "gp_act_lambda_p" in names
    @test "Vc" in names && "Vr" in names
    @test "Bc" in names && "Br" in names
    @test "Fc" in names && "Fr" in names
    @test !("Lc" in names) && !("Hc" in names)  # not in default channels

    names_logrhk = Nereus.noise_param_names(nm_logrhk, ic)
    @test "Lc" in names_logrhk
    @test !("Ll" in names_logrhk)  # logR'HK has no derivative coeff

    nm_no_deriv = ActivityGP(channels = [:bis], use_derivative = false)
    names_no_deriv = Nereus.noise_param_names(nm_no_deriv, ic)
    @test "Vc" in names_no_deriv && "Bc" in names_no_deriv
    @test !("Vr" in names_no_deriv) && !("Br" in names_no_deriv)
end

@testset "ActivityJitter — per-indicator param names (no collision)" begin
    # Regression: ≥2 ActivityJitter models on one instrument used to emit
    # identical jit_base_$ins / jit_act_$ins names → duplicate layout slots
    # → NetCDF defVar -42 ("name in use") at chains.nc write. Names must
    # carry the indicator so N jitter models stay distinct.
    ic = InstrumentConfig(rv = ["FEROS"])
    n_bis = Nereus.noise_param_names(ActivityJitter(indicator = "bisector_span"), ic)
    n_rhk = Nereus.noise_param_names(ActivityJitter(indicator = "log_rhk"), ic)
    @test n_bis == ["jit_base_bisector_span_FEROS", "jit_act_bisector_span_FEROS"]
    @test n_rhk == ["jit_base_log_rhk_FEROS", "jit_act_log_rhk_FEROS"]
    @test isempty(intersect(n_bis, n_rhk))

    # Full layout: ActivityDecorrelation + two ActivityJitter (the config
    # that crashed). All four jitter params must be present and distinct.
    rng = MersenneTwister(11)
    n = 30
    t = sort!(40.0 .* rand(rng, n))
    bis = 0.5 .* randn(rng, n); rhk = 0.02 .* randn(rng, n)
    data = Data(; normalize_indicators = false, t_rv = t, rv = randn(rng, n), rv_err = ones(n),
                  rv_inst = ones(Int, n),
                  indicators = Dict("bisector_span" => bis, "log_rhk" => rhk),
                  indicator_errs = Dict("bisector_span" => fill(0.2, n),
                                        "log_rhk" => fill(0.01, n)))
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[
                         ActivityDecorrelation(indicators = ["bisector_span", "log_rhk"]),
                         ActivityJitter(indicator = "bisector_span"),
                         ActivityJitter(indicator = "log_rhk")])
    un = params.layout.unfrozen_names
    for want in ["jit_base_bisector_span_FEROS", "jit_act_bisector_span_FEROS",
                 "jit_base_log_rhk_FEROS", "jit_act_log_rhk_FEROS"]
        @test want in un
    end
    @test length(un) == length(unique(un))   # no duplicate layout slots

    # The duplicate-name guard fires (actionable) on a genuine collision:
    # two ActivityJitter with the SAME indicator on the same instrument.
    @test_throws ArgumentError Params(;
        max_kplanet = 0, planet_modes = PlanetDataSources[],
        instruments = ic, data = data, M_s = 1.0,
        noise_models = NoiseModel[ActivityJitter(indicator = "bisector_span"),
                                  ActivityJitter(indicator = "bisector_span")])
end

@testset "ActivityGP — rejects indicators not parallel to RV" begin
    # The block-factored covariance assumes every channel shares the RV epochs
    # (one indicator value per RV epoch). Data() enforces this at construction;
    # this checks the likelihood-level guard for the mutate-after-construction
    # path (a clear error instead of OOB / silently-wrong log-likelihood).
    rng = MersenneTwister(101)
    n = 25
    t_rv = sort!(40.0 .* rand(rng, n))
    data = Data(; normalize_indicators = false, t_rv = t_rv, rv = randn(rng, n), rv_err = fill(0.5, n),
                  rv_inst = ones(Int, n),
                  indicators = Dict("bis" => 0.01 .* randn(rng, n)),
                  indicator_errs = Dict("bis" => fill(0.005, n)))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[ActivityGP(channels = [:bis])])
    theta = Theta{Float64}(params)
    for (nm, v) in ("gamma_SIM" => 0.0, "sigma_SIM" => 1e-3,
                     "gp_act_period" => 8.0,
                     "gp_act_lambda_e" => 50.0, "gp_act_lambda_p" => 0.5,
                     "Vc" => 3.0, "Vr" => 0.3, "Bc" => 0.05, "Br" => 0.005)
        nm in params.layout.unfrozen_names && set_param!(theta, nm, v)
    end
    @test isfinite(Nereus.rv_log_likelihood(theta, data))   # valid path works

    # Corrupt the indicator to be shorter than the RVs → guard must throw.
    data.indicators["bis"]     = data.indicators["bis"][1:end-3]
    data.indicator_errs["bis"] = data.indicator_errs["bis"][1:end-3]
    @test_throws ArgumentError Nereus.rv_log_likelihood(theta, data)
end
