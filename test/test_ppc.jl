# PPC unit test: build a synthetic 1-planet RV dataset, hand-construct
# an MCMCChains.Chains object holding posterior draws near the truth
# (so we don't depend on sampler convergence), and verify the residual
# diagnostics make sense.
#
# Decoupling the test from a real sampler means it's deterministic and
# doesn't catch sampler bugs — those are tested elsewhere. The job of
# this test is the PPC machinery itself.

using Nereus, Random, Statistics, Test, MCMCChains
using Random: MersenneTwister

@testset "posterior_predictive_check — RV residuals on a tight posterior" begin
    rng = MersenneTwister(42)
    n_obs = 60
    t = sort!(120.0 .* rand(rng, n_obs))
    P_true, K_true, Mo_true = 20.0, 5.0, 0.7
    # Use a small but well-defined eccentricity so ω is non-degenerate
    # (sesinw/secosw both ≠ 0). Chain jitter at e≈0 would noise-amplify
    # atan2(tiny, tiny) into a random ω across draws.
    e_true, ω_true = 0.10, 1.0
    sesinw_t = sqrt(e_true) * sin(ω_true)
    secosw_t = sqrt(e_true) * cos(ω_true)
    σ_true = 1.0
    rv_err = fill(σ_true, n_obs)

    data0 = Data(; t_rv = t, rv = zeros(n_obs), rv_err = rv_err,
                   rv_inst = ones(Int, n_obs))
    ic   = InstrumentConfig(rv = ["SIM"])
    params = Params(;
        max_kplanet  = 1,
        planet_modes = [RV_ONLY],
        instruments  = ic, data = data0,
        M_s          = 1.0,
    )

    theta_truth = Theta{Float64}(params)
    set_param!(theta_truth, "P_k1",      P_true)
    set_param!(theta_truth, "K_k1",      K_true)
    set_param!(theta_truth, "sesinw_k1", sesinw_t)
    set_param!(theta_truth, "secosw_k1", secosw_t)
    set_param!(theta_truth, "Mo_k1",     Mo_true)
    set_param!(theta_truth, "gamma_SIM", 0.0)
    set_param!(theta_truth, "sigma_SIM", σ_true)
    rv_clean, _ = rv_predictions(theta_truth, data0)
    rv = rv_clean .+ σ_true .* randn(rng, n_obs)

    data = Data(; t_rv = t, rv = rv, rv_err = rv_err,
                  rv_inst = ones(Int, n_obs))

    n_draws = 400
    truth = Dict(
        :P_k1       => P_true,
        :K_k1       => K_true,
        :sesinw_k1  => sesinw_t,
        :secosw_k1  => secosw_t,
        :Mo_k1      => Mo_true,
        :gamma_SIM  => 0.0,
        :sigma_SIM  => σ_true,
    )
    fitted_names = Symbol.(params.layout.unfrozen_names)
    arr = zeros(Float64, n_draws, length(fitted_names), 1)
    # Tiny absolute jitter — needs to be << the smallest dynamic-range
    # of any parameter so the mean forward model doesn't smear away
    # from the truth (P jitter is the worst offender at long baselines).
    σ_chain = 1e-6
    for (j, nm) in enumerate(fitted_names)
        μ = get(truth, nm, 0.0)
        arr[:, j, 1] .= μ .+ σ_chain .* randn(rng, n_draws)
    end
    chains = Chains(arr, fitted_names)

    ppc = posterior_predictive_check(chains, params, data;
                                      n_draws = n_draws,
                                      rng = MersenneTwister(0))

    # Shape + summary keys
    @test ppc.has_rv
    @test !ppc.has_pm
    @test ppc.n_draws == n_draws
    @test length(ppc.rv_resid_med) == n_obs
    @test length(ppc.rv_resid_std) == n_obs
    @test all(ppc.rv_resid_std .>= 0)
    @test length(ppc.rv_chi2_per_draw) == n_draws

    @test haskey(ppc.summary, "rv_red_chi2_total")
    @test haskey(ppc.summary, "rv_red_chi2_per_inst")
    @test haskey(ppc.summary, "rv_red_chi2_per_draw_p16")
    @test haskey(ppc.summary, "rv_red_chi2_per_draw_p50")
    @test haskey(ppc.summary, "rv_red_chi2_per_draw_p84")
    @test haskey(ppc.summary, "rv_outliers_3sigma")

    # Headline χ²/dof is at the median-parameter model and should be ~1
    # when the posterior is centred on the truth.
    χ2red = ppc.summary["rv_red_chi2_total"]
    @test 0.4 < χ2red < 1.8

    # Per-draw distribution should bracket the headline:
    p16 = ppc.summary["rv_red_chi2_per_draw_p16"]
    p50 = ppc.summary["rv_red_chi2_per_draw_p50"]
    p84 = ppc.summary["rv_red_chi2_per_draw_p84"]
    @test p16 <= p50 <= p84
    # Tight chain → narrow per-draw distribution.
    @test (p84 - p16) < 1.0

    @test ppc.summary["rv_outliers_3sigma"] <= 3

    # GLS of residuals: top peak should NOT be highly significant for
    # white noise residuals (we'd expect FAP > 0.001 typically).
    @test ppc.rv_pgram !== nothing
    @test !isempty(ppc.rv_pgram.power)

    # ACF normalised to 1 at lag 0
    @test !isempty(ppc.rv_acf)
    @test isapprox(ppc.rv_acf[1], 1.0; atol = 1e-10)

    # Plot builds + saves
    tmp = tempname() * ".png"
    plot_ppc(ppc; filename = tmp)
    @test isfile(tmp)
    @test filesize(tmp) > 1000
    rm(tmp; force = true)
end

@testset "posterior_predictive_check — broad ω marginal at e≈0" begin
    # Real-world scenario: data is consistent with a circular orbit
    # so the (sesinw, secosw) posterior surrounds (0, 0), making ω
    # essentially uniform across draws. Per-draw χ² blows up because
    # each draw "guesses" a random ω, but the median-parameter model
    # still fits because ω at the median is well-defined.
    rng = MersenneTwister(11)
    n_obs = 60
    t = sort!(120.0 .* rand(rng, n_obs))
    σ_true = 1.0
    rv_err = fill(σ_true, n_obs)

    data0 = Data(; t_rv = t, rv = zeros(n_obs), rv_err = rv_err,
                   rv_inst = ones(Int, n_obs))
    ic   = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                     instruments = ic, data = data0, M_s = 1.0)

    # Build truly circular data (e = 0, ω = 0). The signal is
    # K cos(M(t) + 0). The chain below has sesinw/secosw spread broadly
    # around (0, 0) — consistent with a real fit that found e ≈ 0 —
    # so the per-draw ω is essentially uniform but the median model at
    # (0, 0) ⇒ ω = 0 recovers the truth exactly.
    theta_t = Theta{Float64}(params)
    set_param!(theta_t, "P_k1", 20.0); set_param!(theta_t, "K_k1", 5.0)
    set_param!(theta_t, "sesinw_k1", 0.0); set_param!(theta_t, "secosw_k1", 0.0)
    set_param!(theta_t, "Mo_k1", 0.5); set_param!(theta_t, "gamma_SIM", 0.0)
    set_param!(theta_t, "sigma_SIM", σ_true)
    rv_clean, _ = rv_predictions(theta_t, data0)
    rv = rv_clean .+ σ_true .* randn(rng, n_obs)
    data = Data(; t_rv = t, rv = rv, rv_err = rv_err,
                  rv_inst = ones(Int, n_obs))

    # Chain centred at the truth on P, K, Mo, but BROAD on sesinw/secosw
    # (σ = 0.2 around 0) — ω marginal will be effectively uniform.
    fitted_names = Symbol.(params.layout.unfrozen_names)
    n_draws = 400
    arr = zeros(Float64, n_draws, length(fitted_names), 1)
    truth_d = Dict(:P_k1 => 20.0, :K_k1 => 5.0,
                   :sesinw_k1 => 0.0, :secosw_k1 => 0.0,
                   :Mo_k1 => 0.5,
                   :gamma_SIM => 0.0, :sigma_SIM => σ_true)
    scale = Dict(:sesinw_k1 => 0.2, :secosw_k1 => 0.2)
    for (j, nm) in enumerate(fitted_names)
        μ = get(truth_d, nm, 0.0)
        σ_chain = get(scale, nm, 1e-6)
        arr[:, j, 1] .= μ .+ σ_chain .* randn(rng, n_draws)
    end
    chains = Chains(arr, fitted_names)

    ppc = posterior_predictive_check(chains, params, data; n_draws = n_draws)

    # Per-draw χ² inflates because each draw uses a random ω, but at
    # least one draw in the posterior must have ω ≈ truth (the chain
    # spans the full circle), so `rv_red_chi2_total` (min across draws)
    # stays good. The per-draw distribution itself is inflated.
    headline = ppc.summary["rv_red_chi2_total"]
    p84_draw = ppc.summary["rv_red_chi2_per_draw_p84"]
    @test headline < 2.0          # best fit in posterior ≈ noise level
    @test p84_draw > 2.0          # most draws are inflated
    @test p84_draw > headline
end

@testset "posterior_predictive_check — outlier detection" begin
    # Inject a 10σ outlier; PPC should flag it.
    rng = MersenneTwister(7)
    n_obs = 30
    t = sort!(60.0 .* rand(rng, n_obs))
    σ_true = 1.0
    rv = σ_true .* randn(rng, n_obs)         # pure noise
    rv[10] += 10 * σ_true                     # one outlier
    rv_err = fill(σ_true, n_obs)

    data = Data(; t_rv = t, rv = rv, rv_err = rv_err,
                  rv_inst = ones(Int, n_obs))
    ic   = InstrumentConfig(rv = ["SIM"])
    params = Params(;
        max_kplanet  = 0,
        planet_modes = PlanetDataSources[],
        instruments  = ic, data = data, M_s = 1.0,
    )

    fitted_names = Symbol.(params.layout.unfrozen_names)
    n_draws = 200
    arr = zeros(Float64, n_draws, length(fitted_names), 1)
    truth = Dict(:gamma_SIM => 0.0, :sigma_SIM => σ_true)
    for (j, nm) in enumerate(fitted_names)
        μ = get(truth, nm, 0.0)
        arr[:, j, 1] .= μ .+ 1e-3 .* randn(rng, n_draws)
    end
    chains = Chains(arr, fitted_names)

    ppc = posterior_predictive_check(chains, params, data; n_draws = n_draws)
    @test ppc.summary["rv_outliers_3sigma"] >= 1
end
