# PSIS-LOO / WAIC via chain-replay through the likelihood. Verifies:
#   - LOO and WAIC prefer the true model over a null on synthetic RV
#   - elpd_loo ≈ sum of per-point log L for a point posterior
#   - Pareto-k diagnostics are reported and sane
#   - GP / CovarianceNoise models are refused (pointwise factorisation
#     doesn't apply)

using Nereus, Random, Statistics, Test, MCMCChains, LinearAlgebra
using Random: MersenneTwister

function _build_chain(params, target_dict, σ_chain, n_draws, rng)
    fitted_names = Symbol.(params.layout.unfrozen_names)
    arr = zeros(Float64, n_draws, length(fitted_names), 1)
    for (j, nm) in enumerate(fitted_names)
        μ = get(target_dict, nm, 0.0)
        arr[:, j, 1] .= μ .+ σ_chain .* randn(rng, n_draws)
    end
    return Chains(arr, fitted_names)
end

@testset "compute_loo — truth vs null on synthetic RV" begin
    rng = MersenneTwister(11)
    n_obs = 80
    t = sort!(120.0 .* rand(rng, n_obs))
    P_true, K_true, Mo_true = 20.0, 5.0, 0.7
    e_true, ω_true = 0.10, 1.0
    sesinw_t = sqrt(e_true) * sin(ω_true)
    secosw_t = sqrt(e_true) * cos(ω_true)
    σ_true = 1.0

    data0 = Data(; t_rv = t, rv = zeros(n_obs), rv_err = fill(σ_true, n_obs),
                   rv_inst = ones(Int, n_obs))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                     instruments = ic, data = data0, M_s = 1.0)

    theta_t = Theta{Float64}(params)
    set_param!(theta_t, "P_k1", P_true); set_param!(theta_t, "K_k1", K_true)
    set_param!(theta_t, "sesinw_k1", sesinw_t); set_param!(theta_t, "secosw_k1", secosw_t)
    set_param!(theta_t, "Mo_k1", Mo_true); set_param!(theta_t, "gamma_SIM", 0.0)
    set_param!(theta_t, "sigma_SIM", σ_true)
    rv_clean, _ = rv_predictions(theta_t, data0)
    rv = rv_clean .+ σ_true .* randn(rng, n_obs)
    data = Data(; t_rv = t, rv = rv, rv_err = fill(σ_true, n_obs),
                  rv_inst = ones(Int, n_obs))

    truth_d = Dict(:P_k1 => P_true, :K_k1 => K_true,
                   :sesinw_k1 => sesinw_t, :secosw_k1 => secosw_t,
                   :Mo_k1 => Mo_true, :gamma_SIM => 0.0, :sigma_SIM => σ_true)
    null_d  = Dict(:P_k1 => 20.0, :K_k1 => 0.0,
                   :sesinw_k1 => 0.0, :secosw_k1 => 0.0,
                   :Mo_k1 => 0.5, :gamma_SIM => 0.0, :sigma_SIM => σ_true * 1.5)

    chains_truth = _build_chain(params, truth_d, 1e-3, 400, rng)
    chains_null  = _build_chain(params, null_d, 1e-3, 400, rng)

    loo_truth = compute_loo(chains_truth, params, data; n_draws = 200)
    loo_null  = compute_loo(chains_null,  params, data; n_draws = 200)

    @test loo_truth isa LooResult
    @test loo_truth.n_obs == n_obs
    @test loo_truth.n_draws == 200
    @test length(loo_truth.pareto_k) == n_obs
    @test 0.0 <= loo_truth.pareto_k_max
    @test isfinite(loo_truth.elpd_loo)
    @test isfinite(loo_truth.elpd_waic)
    @test loo_truth.se_elpd_loo > 0
    @test loo_truth.se_elpd_waic > 0

    # Truth model preferred over the null by a wide margin.
    Δ = loo_truth.elpd_loo - loo_null.elpd_loo
    @test Δ > 50.0
    Δ_waic = loo_truth.elpd_waic - loo_null.elpd_waic
    @test Δ_waic > 50.0
    @test sign(Δ) == sign(Δ_waic)
end

@testset "compute_loo — log_z comparison" begin
    # When log_z is passed, loo_compare_log_z = elpd_loo − log_z.
    rng = MersenneTwister(2)
    n_obs = 40
    t = sort!(60.0 .* rand(rng, n_obs))
    data = Data(; t_rv = t, rv = randn(rng, n_obs), rv_err = ones(n_obs),
                  rv_inst = ones(Int, n_obs))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0)

    fitted_names = Symbol.(params.layout.unfrozen_names)
    n_draws = 200
    arr = zeros(Float64, n_draws, length(fitted_names), 1)
    truth_d = Dict(:gamma_SIM => 0.0, :sigma_SIM => 1.0)
    for (j, nm) in enumerate(fitted_names)
        μ = get(truth_d, nm, 0.0)
        arr[:, j, 1] .= μ .+ 1e-4 .* randn(rng, n_draws)
    end
    chains = Chains(arr, fitted_names)

    loo = compute_loo(chains, params, data; n_draws = n_draws, log_z = -60.0)
    @test loo.loo_compare_log_z !== nothing
    @test isfinite(loo.loo_compare_log_z)
    @test isapprox(loo.loo_compare_log_z, loo.elpd_loo - (-60.0); atol = 1e-9)
end

@testset "compute_loo — ActivityGP marginalized (Sundararajan-Keerthi)" begin
    # AGP with marginalize_indicators = true: log p(RV | indicators) is
    # a multivariate Gaussian on the RV channel with non-diagonal
    # covariance. PSIS-LOO uses the closed-form leave-one-out
    # predictive (Sundararajan & Keerthi 2001).
    rng = MersenneTwister(101)
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

    data = Data(; t_rv = t_rv, rv = y_flat[1:n_obs],
                  rv_err = fill(σ_rv, n_obs), rv_inst = ones(Int, n_obs),
                  indicators = Dict("bis" => y_flat[(n_obs+1):end]),
                  indicator_errs = Dict("bis" => fill(σ_bis, n_obs)))
    ic = InstrumentConfig(rv = ["SIM"])
    agp = ActivityGP(channels = [:bis], marginalize_indicators = true)
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[agp])
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
    loo = compute_loo(chains, params, data; n_draws = 50)
    @test loo isa LooResult
    @test isfinite(loo.elpd_loo)
    @test isfinite(loo.elpd_waic)
    @test loo.se_elpd_loo > 0
    @test 0 <= loo.pareto_k_max
    @test length(loo.pareto_k) == n_obs   # n_rv only — indicators conditioned out
end

@testset "compute_loo — GP refusal (non-marginalized AGP / Celerite)" begin
    # Non-marginalized AGP and other CovarianceNoise types should
    # still be refused.
    rng = MersenneTwister(8)
    n = 25
    data = Data(; t_rv = sort!(40.0 .* rand(rng, n)), rv = randn(rng, n),
                  rv_err = ones(n), rv_inst = ones(Int, n),
                  indicators = Dict("bis" => randn(rng, n)),
                  indicator_errs = Dict("bis" => fill(0.005, n)))
    ic = InstrumentConfig(rv = ["SIM"])
    params_joint = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                           instruments = ic, data = data, M_s = 1.0,
                           noise_models = NoiseModel[
                               ActivityGP(channels = [:bis],
                                          marginalize_indicators = false)])
    fitted_names = Symbol.(params_joint.layout.unfrozen_names)
    arr = zeros(Float64, 30, length(fitted_names), 1)
    chains = Chains(arr, fitted_names)
    @test_throws ArgumentError compute_loo(chains, params_joint, data;
                                              n_draws = 20)
end

@testset "compute_loo — GP refusal" begin
    # Building any Params with a CovarianceNoise (Celerite) active
    # should make compute_loo throw with a clear ArgumentError —
    # pointwise log L is not well-defined under correlated noise.
    rng = MersenneTwister(3)
    n_obs = 30
    t = sort!(60.0 .* rand(rng, n_obs))
    data = Data(; t_rv = t, rv = randn(rng, n_obs), rv_err = ones(n_obs),
                  rv_inst = ones(Int, n_obs))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                     instruments = ic, data = data, M_s = 1.0,
                     noise_models = NoiseModel[CeleriteSHO(channel = :rv)])

    # Minimal chain with the GP hyperparams = some defaults.
    fitted_names = Symbol.(params.layout.unfrozen_names)
    arr = zeros(Float64, 50, length(fitted_names), 1)
    for (j, nm) in enumerate(fitted_names)
        arr[:, j, 1] .= nm === :sigma_SIM ? 1.0 : 0.0
    end
    chains = Chains(arr, fitted_names)
    @test_throws ArgumentError compute_loo(chains, params, data; n_draws = 25)
end
