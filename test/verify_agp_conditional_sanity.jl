#!/usr/bin/env julia
# Free-lunch discriminator for the ActivityGP conditional likelihood.
#
# Generate synthetic data where the TRUTH is the simple model:
#   RV = Keplerian + C·I(t) + white noise,
# with I(t) a smooth quasi-periodic indicator (so the activity proxy is
# genuinely informative, as on a real active star). Fit AD-only and
# AGP-only (marginalize_indicators = true) and compare logZ + K.
#
# PASS: AD wins or ties (|ΔlogZ| ≲ 20), both recover K_true.
# FAIL: AGP claims a large evidence advantage (≫ 20 nats) on data that
#       contain NO Rajpaul structure — the conditional path is buying
#       likelihood it hasn't earned, and every real-data AGP evidence
#       is suspect.

using Nereus
using MCMCChains
using Statistics: median, quantile, mean, std
using Printf
using Random
using LinearAlgebra: Symmetric, cholesky, I

rng = MersenneTwister(424242)

# ---- truth ----------------------------------------------------------
n      = 90
t      = sort(rand(rng, n)) .* 120.0
P_true = 5.234; K_true = 10.0
P_rot  = 9.1
σ_white = 2.0
C_true  = 8.0          # m/s per unit of normalized indicator

# Smooth QP indicator: draw from a unit QP GP (λe=40, λp=0.6), then
# normalize to unit RMS — a clean activity proxy.
Σ_I = [exp(-(t[i]-t[j])^2 / (2*40.0^2) -
            sin(π*(t[i]-t[j])/P_rot)^2 / (2*0.6^2)) for i in 1:n, j in 1:n]
LI = cholesky(Symmetric(Σ_I + 1e-8 * Matrix{Float64}(I, n, n))).L
ind = LI * randn(rng, n)
ind ./= std(ind)

rv = K_true .* sin.(2π .* t ./ P_true) .+ C_true .* ind .+ σ_white .* randn(rng, n)
ind_err = fill(0.05, n)   # indicator measured ~20x better than its RMS

data = Data(; t_rv = t, rv = rv, rv_err = fill(σ_white, n),
              rv_inst = ones(Int, n),
              indicators = Dict("act_AD" => ind, "bis" => ind),
              indicator_errs = Dict("bis" => ind_err))
ic = InstrumentConfig(rv = ["SIM"])

function base_priors()
    Dict{String, PriorSpec}(
        "P_k1"      => LogUniformPrior(3.0, 9.0),
        "K_k1"      => UniformPrior(0.0, 40.0),
        "sesinw_k1" => UniformPrior(-1.0, 1.0),
        "secosw_k1" => UniformPrior(-1.0, 1.0),
        "Mo_k1"     => UniformPrior(0.0, 2π),
        "gamma_SIM" => UniformPrior(-30.0, 30.0),
        "sigma_SIM" => LogUniformPrior(0.1, 20.0))
end

function run_case(label, noise_model; extra = Dict{String, PriorSpec}())
    priors = base_priors(); merge!(priors, extra)
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
        instruments = ic, data = data, M_s = 1.0,
        parametrization = ParametrizationConfig(time = :Mo),
        priors = priors, noise_models = [noise_model])
    target = NereusTarget(params, data)
    t0 = time()
    res = sample_ptemcee(target, data; n_temps = 10, n_walkers = 96,
                          n_steps = 5000, n_burnin = 2000, seed = 5,
                          init_strategy = :prior, show_progress = false)
    K = vec(Array(res.chains[:, :K_k1, :]))
    @printf("%s: %.0f s  logZ = %.2f  K = %.2f [%.2f, %.2f] (true %.1f)\n",
            label, time() - t0, res.log_evidence,
            median(K), quantile(K, 0.16), quantile(K, 0.84), K_true)
    return res.log_evidence, median(K)
end

ad  = ActivityDecorrelation(indicators = ["act_AD"])
agp = ActivityGP(channels = [:bis], marginalize_indicators = true)

lz_ad, K_ad   = run_case("AD-only  (truth)", ad)
lz_agp, K_agp = run_case("AGP-only        ", agp;
    extra = Dict{String, PriorSpec}(
        "gp_act_period" => NormalPrior(P_rot, 1.5, 4.0, 18.0)))

Δ = lz_ad - lz_agp
@printf("\nΔlogZ (AD − AGP) = %.2f  on AD-TRUTH data\n", Δ)
if Δ > -20
    println("✅ no free lunch: AGP does not overclaim on non-Rajpaul data")
else
    @printf("❌ FREE LUNCH: AGP claims +%.0f nats on data with NO Rajpaul structure\n", -Δ)
end
