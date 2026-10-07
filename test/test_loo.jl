# PSIS-LOO / WAIC via chain-replay through the likelihood. Verifies:
#   - LOO and WAIC prefer the true model over a null on synthetic RV
#   - elpd_loo ≈ sum of per-point log L for a point posterior
#   - Pareto-k diagnostics are reported and sane
#   - under correlated noise (GPs, HarmonicBlock, NightlyOffset, MA, the γ
#     marginalization, the marginalized ActivityGP) the pointwise term is the
#     exact leave-one-out predictive, not the density of the point alone
#   - models LOO cannot score exactly are refused

using Nereus, Random, Statistics, Test, MCMCChains, LinearAlgebra
using Random: MersenneTwister
using Nereus: CeleriteRotation, MaternGP, HarmonicBlock, NightlyOffset, StudentT, MAModel,
              NoiseModel

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

# =====================================================================
# Correlated noise: exact leave-one-out
# =====================================================================
#
# The oracle at fixed θ: log p(r_i | r_{-i}) = log L(r) − log L(r_{-i}), both
# scored by the package's own channel likelihood (celerite, Woodbury, the
# nightly closed form, ...). Exact for any Gaussian process-like model, whose
# marginal on the other points is the same model on those points.
function _ratio_oracle(th, r, v, t, inst, ch)
    n = length(r)
    full = Nereus._eval_channel_likelihood(th, r, v, t, inst, ch, 2π)
    return [full - Nereus._eval_channel_likelihood(th, r[k], v[k], t[k], inst[k], ch, 2π)
            for k in (deleteat!(collect(1:n), i) for i in 1:n)]
end

# log p(r_i | r_{-i}) for r ~ N(0, Σ) by explicit conditioning (Schur complement).
function _schur_oracle(r, Σ)
    n = length(r)
    out = zeros(n)
    for i in 1:n
        k = deleteat!(collect(1:n), i)
        S = Σ[k, k]; c = Σ[k, i]
        μ = dot(c, S \ r[k]); s2 = Σ[i, i] - dot(c, S \ c)
        out[i] = -0.5 * (log(2π * s2) + (r[i] - μ)^2 / s2)
    end
    return out
end

# A chain of draws scattered tightly about `th`: LOO's elpd is then the sum of
# the pointwise predictive densities at `th`.
function _chain_at(params, th, σ, n_draws, rng)
    names = Symbol.(params.layout.unfrozen_names)
    arr = zeros(n_draws, length(names), 1)
    for (j, idx) in enumerate(params.layout.unfrozen_idx)
        arr[:, j, 1] .= th.values[idx] .+ σ .* randn(rng, n_draws)
    end
    return Chains(arr, names)
end

function _theta_with(params, vals)
    th = Theta{Float64}(params)
    for (k, v) in vals
        haskey(params.layout.name_to_idx, k) && set_param!(th, k, v)
    end
    return th
end

# Two RV instruments observed in nights (several points within hours, nights
# days apart), stored instrument by instrument -- not in time order.
function _night_epochs(rng; nights = (9, 7), per_night = 3)
    t = Float64[]; inst = Int[]
    for (m, nn) in enumerate(nights)
        for g in 1:nn
            night0 = 5.0 * g + 1.7 * m + 2 * rand(rng)
            for _ in 1:per_night
                push!(t, night0 + 0.1 * rand(rng)); push!(inst, m)
            end
        end
    end
    for m in 1:2                               # each instrument in time order
        ix = findall(==(m), inst); t[ix] = sort(t[ix])
    end
    return t, inst
end

const _LOO_HYP = Dict(
    "gamma_A" => 0.4, "gamma_B" => -0.3, "sigma_A" => 0.6, "sigma_B" => 0.9,
    "gp_log_S0" => 0.8, "gp_log_Q" => 0.6, "gp_log_omega0" => -0.7,
    "gp_log_S0_A" => 0.5, "gp_log_Q_A" => 1.0, "gp_log_omega0_A" => -0.5,
    "gp_sigma_B" => 1.5, "gp_period_B" => 11.0, "gp_Q0_B" => 1.0, "gp_dQ_B" => 0.5,
    "gp_f_B" => 0.5, "matern_sigma" => 1.8, "matern_rho" => 4.0,
    "harm_period" => 13.0, "harm_amp_A" => 2.0, "harm_amp_B" => 1.4,
    "night_sigma_A" => 1.5, "night_sigma_B" => 0.8, "studentt_nu" => 3.5)

@testset "compute_loo — exact under correlated RV noise: $label" for (label, noise) in (
        ("celerite SHO", [CeleriteSHO()]),
        ("per-instrument GPs", [CeleriteSHO(instruments = ["A"]),
                                CeleriteRotation(instruments = ["B"])]),
        ("Matérn GP", [MaternGP()]),
        ("HarmonicBlock", [HarmonicBlock(nharm = 3)]),
        ("NightlyOffset", [NightlyOffset(gap = 0.5)]),
        ("SHO + HarmonicBlock", [CeleriteSHO(), HarmonicBlock(nharm = 2)]),
        ("Student-t", [StudentT()]))
    rng = MersenneTwister(4711)
    t, inst = _night_epochs(rng)
    n = length(t)
    rv = 2.5 .* sin.(2π .* t ./ 13) .+ 1.2 .* cos.(4π .* t ./ 13) .+
         [0.9, -0.6][inst] .+ 0.7 .* randn(rng, n)
    data = Data(; t_rv = t, rv = rv, rv_err = fill(0.5, n), rv_inst = inst)
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                    instruments = InstrumentConfig(rv = ["A", "B"]), data = data,
                    M_s = 1.0, noise_models = NoiseModel[noise...])
    th = _theta_with(params, _LOO_HYP)
    preds, vars = rv_predictions(th, data)
    r = data.rv .- preds
    oracle = _ratio_oracle(th, r, vars, t, inst, :rv)
    @test all(isfinite, oracle)

    chains = _chain_at(params, th, 1e-7, 200, MersenneTwister(1))
    loo = compute_loo(chains, params, data; n_draws = 100)
    @test loo.n_obs == n
    @test loo.elpd_loo ≈ sum(oracle) atol = 1e-3
    @test loo.elpd_waic ≈ sum(oracle) atol = 1e-3
    if label != "Student-t"
        # Scoring each point alone with its own variance -- what LOO did for
        # every non-CovarianceNoise model -- is far off.
        indep = sum(@. -0.5 * (log(2π * vars) + r^2 / vars))
        @test abs(loo.elpd_loo - indep) > 5
    end

    # Point by point, at θ.
    @test Nereus._pointwise_loo_rv(th, data, 10_000) ≈ oracle rtol = 1e-7
end

@testset "compute_loo — MA residuals are a correlated Gaussian" begin
    # ε = B r, B unit lower triangular (ε_i = r_i − ω e^{−Δt/β} r_{i−1}), and
    # ε ~ N(0, diag(v)), so r ~ N(0, B⁻¹ diag(v) B⁻ᵀ): MA's points are coupled
    # in both directions, and the sequential factors N(ε_i; 0, v_i) are not
    # their leave-one-out densities.
    rng = MersenneTwister(77)
    n = 50
    t = sort!(80 .* rand(rng, n))
    data = Data(; t_rv = t, rv = randn(rng, n), rv_err = fill(0.5, n),
                rv_inst = ones(Int, n))
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                    instruments = InstrumentConfig(rv = ["SIM"]), data = data, M_s = 1.0,
                    noise_models = NoiseModel[MAModel(order = 1)])
    ω, β = 0.7, 3.0
    th = _theta_with(params, Dict("sigma_SIM" => 0.4, "gamma_SIM" => 0.1,
                                  "ma_omega_1" => ω, "ma_beta_1" => β))
    preds, vars = rv_predictions(th, data)
    r = data.rv .- preds
    B = Matrix(1.0I, n, n)
    for i in 2:n
        B[i, i - 1] = -ω * exp(-(t[i] - t[i - 1]) / β)
    end
    Bi = inv(B)
    oracle = _schur_oracle(r, Bi * Diagonal(vars) * Bi')
    loo = compute_loo(_chain_at(params, th, 1e-7, 200, rng), params, data; n_draws = 100)
    @test loo.elpd_loo ≈ sum(oracle) atol = 1e-3
    @test Nereus._pointwise_loo_rv(th, data, 10_000) ≈ oracle rtol = 1e-7
end

@testset "compute_loo — γ marginalized analytically" begin
    rng = MersenneTwister(91)
    t, inst = _night_epochs(rng)
    n = length(t)
    data = Data(; t_rv = t, rv = [5.0, -3.0][inst] .+ randn(rng, n),
                rv_err = fill(0.7, n), rv_inst = inst)
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                    instruments = InstrumentConfig(rv = ["A", "B"]), data = data, M_s = 1.0,
                    parametrization = Nereus.ParametrizationConfig(marginalize_gamma = true))
    th = _theta_with(params, Dict("sigma_A" => 0.5, "sigma_B" => 0.8))
    preds, vars = rv_predictions(th, data)
    d = data.rv .- preds
    slot = params.layout.systemic.rv_gamma
    full = Nereus._rv_ll_gamma_marginalized(d, vars, inst, slot, n, 2π)
    oracle = [full - Nereus._rv_ll_gamma_marginalized(d[k], vars[k], inst[k], slot, n - 1, 2π)
              for k in (deleteat!(collect(1:n), i) for i in 1:n)]
    loo = compute_loo(_chain_at(params, th, 1e-5, 200, rng), params, data; n_draws = 100)
    @test loo.elpd_loo ≈ sum(oracle) atol = 1e-2
    @test Nereus._pointwise_loo_rv(th, data, 10_000) ≈ oracle rtol = 1e-9
end

@testset "compute_loo — marginalized ActivityGP against the joint" begin
    # log p(y_Ri | y_R,−i, y_I): RV point i given every other RV point AND the
    # indicators, by conditioning the joint Rajpaul covariance directly.
    rng = MersenneTwister(101)
    n = 25
    t_rv = sort!(40.0 .* rand(rng, n))
    data = Data(; t_rv = t_rv, rv = 3 .* randn(rng, n), rv_err = fill(0.5, n),
                rv_inst = ones(Int, n),
                indicators = Dict("bis" => 0.05 .* randn(rng, n)),
                indicator_errs = Dict("bis" => fill(0.005, n)))
    agp = ActivityGP(channels = [:bis], marginalize_indicators = true)
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                    instruments = InstrumentConfig(rv = ["SIM"]), data = data, M_s = 1.0,
                    noise_models = NoiseModel[agp])
    P, λe, λp, Vc, Vr, Bc, Br = 12.0, 50.0, 0.5, 3.0, 0.3, 0.05, 0.005
    th = _theta_with(params, Dict("sigma_SIM" => 0.3, "gp_act_period" => P,
                                  "gp_act_lambda_e" => λe, "gp_act_lambda_p" => λp,
                                  "Vc" => Vc, "Vr" => Vr, "Bc" => Bc, "Br" => Br,
                                  "gp_act_jit_bis" => 0.002))
    inv_sdG = 1 / sqrt(1 / λe^2 + π^2 / (P^2 * λp^2))
    preds, vars = rv_predictions(th, data)
    jit = haskey(params.layout.name_to_idx, "gp_act_jit_bis") ? 0.002 : 0.0
    Σ = activity_gp_covariance(vcat(t_rv, t_rv), vcat(fill(:rv, n), fill(:bis, n)),
                               vcat(fill(Vc, n), fill(Bc, n)),
                               vcat(fill(Vr * inv_sdG, n), fill(Br * inv_sdG, n)),
                               1.0, P, λe, λp)
    # (Data standardizes the indicators and their errors; read them back.)
    Σ += Diagonal(vcat(vars, data.indicator_errs["bis"] .^ 2 .+ jit^2))
    y = vcat(data.rv .- preds, data.indicators["bis"])
    oracle = _schur_oracle(y, Σ)[1:n]
    @test Nereus._pointwise_loo_rv(th, data, 10_000) ≈ oracle rtol = 1e-6
end

@testset "compute_loo — photometry HarmonicBlock (external comb)" begin
    rng = MersenneTwister(12)
    n = 300
    t = sort!(15 .* rand(rng, n))
    flux = 1 .+ 8e-4 .* sin.(2π .* 1.3 .* t) .+ 5e-4 .* cos.(2π .* 2.9 .* t) .+
           3e-4 .* randn(rng, n)
    data = Data(; t_phot = t, flux = flux, flux_err = fill(3e-4, n),
                phot_inst = ones(Int, n))
    params = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                    instruments = InstrumentConfig(String[], ["TESS"]), data = data,
                    M_s = 1.0, R_s = 1.0,
                    noise_models = NoiseModel[HarmonicBlock(channel = :phot,
                                                            freqs = [1.3, 2.9])])
    th = _theta_with(params, Dict("harm_amp_TESS_phot" => 1e-3))
    preds, vars = Nereus.phot_predictions(th, data)
    r = data.flux .- preds
    oracle = _ratio_oracle(th, r, vars, t, data.phot_inst, :phot)
    loo = compute_loo(_chain_at(params, th, 1e-6, 200, rng), params, data; n_draws = 100)
    @test loo.elpd_loo ≈ sum(oracle) atol = 5e-2
    @test abs(loo.elpd_loo - sum(@. -0.5 * (log(2π * vars) + r^2 / vars))) > 5
    @test Nereus._pointwise_loo_phot(th, data, 10_000) ≈ oracle rtol = 1e-7
end

# A GP type compute_loo has never heard of: it must be refused, not scored as
# independent points (the routing reads the covariance, not a type list).
struct _LooTestGP <: Nereus.CovarianceNoise
    channel::Symbol
    instruments::Vector{String}
end
Nereus.noise_param_names(::_LooTestGP, ::InstrumentConfig; data = nothing) = String[]
Nereus._default_noise_priors!(dic, ::_LooTestGP, args...; kwargs...) = dic
function Nereus.gp_log_likelihood(r::AbstractVector{T}, v::AbstractVector{T},
                                  t::AbstractVector{Float64}, ::Theta{T},
                                  ::_LooTestGP) where {T}
    C = [exp(-abs(a - b) / 5.0) for a in t, b in t] + Diagonal(v)
    F = cholesky(Symmetric(C))
    return -0.5 * (dot(r, F \ r) + logdet(F) + length(r) * log(2π))
end

@testset "compute_loo — refusals" begin
    rng = MersenneTwister(3)
    n = 30
    t = sort!(60.0 .* rand(rng, n))
    data = Data(; t_rv = t, rv = randn(rng, n), rv_err = ones(n),
                  rv_inst = ones(Int, n))
    ic = InstrumentConfig(rv = ["SIM"])
    mk(noise) = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                        instruments = ic, data = data, M_s = 1.0,
                        noise_models = NoiseModel[noise...])

    p = mk([_LooTestGP(:rv, String[])])
    th = _theta_with(p, Dict("sigma_SIM" => 0.5))
    @test isfinite(Nereus._rv_log_likelihood_core(th, data))
    @test_throws ArgumentError compute_loo(_chain_at(p, th, 1e-6, 50, rng), p, data;
                                           n_draws = 20)

    # Dense solves above the size limit are refused, not run.
    p = mk([CeleriteSHO()])
    th = _theta_with(p, Dict("sigma_SIM" => 0.5, "gp_log_S0" => 0.0))
    @test_throws ArgumentError compute_loo(_chain_at(p, th, 1e-6, 50, rng), p, data;
                                           n_draws = 20, max_dense_obs = n - 1)

    # A covariance that does not reproduce the likelihood is refused.
    preds, vars = rv_predictions(th, data)
    ref = Nereus._rv_log_likelihood_core(th, data)
    r = data.rv .- preds
    @test Nereus._channel_loo(th, r, vars, t, data.rv_inst, :rv, ref, 10_000) isa Vector
    @test_throws ArgumentError Nereus._channel_loo(th, r, vars, t, data.rv_inst, :rv,
                                                   ref + 0.5, 10_000)
end
