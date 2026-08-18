#!/usr/bin/env julia
# Toy validation of the trans-dim NOISE-toggle machinery: does the
# occupancy reproduce the evidence ratio?
#
# THE GATE (set after the HD 18599 230-nat post-mortem): before any
# trans-dim noise occupancy may be reported as P(model | data), it must
# reproduce a Z-ratio computed independently. Here the independent
# reference is two SEPARATE fixed-config PT runs (white-only vs
# white+MA(1)), whose TI+/SS+/H+ estimators are validated to <0.01 nats
# on analytic targets. Data are PURE WHITE, so the MA(1) model is
# spurious and Occam-disfavored by a computable margin ΔlogZ.
#
# Pass: logit(P(MA active)) ≈ (logZ_MA − logZ_white) + log(p_incl/(1−p_incl))
#       within a few × the MC error. Fail: the noise machinery is not
#       sampling P(M|D) and NO occupancy is trustworthy.

using Nereus
using MCMCChains
using Statistics: mean, median, quantile, std
using Printf
using Random

rng = MersenneTwister(20260613)
n   = 60
t   = sort(rand(rng, n)) .* 90.0
P_true, K_true, σ_true = 12.0, 8.0, 3.0
rv  = K_true .* sin.(2π .* t ./ P_true) .+ σ_true .* randn(rng, n)   # PURE WHITE
data = Data(t_rv = t, rv = rv, rv_err = fill(σ_true, n), rv_inst = ones(Int, n))

function build(noise_models)
    priors = Dict{String, PriorSpec}(
        "n_p"       => FixedPrior(1.0),
        "P_k1"      => LogUniformPrior(8.0, 18.0),
        "K_k1"      => UniformPrior(0.0, 25.0),
        "sesinw_k1" => UniformPrior(-0.6, 0.6),
        "secosw_k1" => UniformPrior(-0.6, 0.6),
        "Mo_k1"     => UniformPrior(0.0, 2π),
        "gamma_I1"  => UniformPrior(-15.0, 15.0),
        "sigma_I1"  => LogUniformPrior(0.5, 12.0))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
        instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
        parametrization = ParametrizationConfig(time = :Mo),
        priors = priors, noise_models = noise_models)
    return params, NereusTarget(params, data)
end

ma = MAModel(order = 1)

# ---- Reference: two separate fixed-config PT evidences ---------------
pA, tA = build(NoiseModel[])
pB, tB = build(NoiseModel[ma])
zA = Float64[]; zB = Float64[]
for sd in (1, 2, 3)
    rA = sample_ptemcee(tA, data; n_temps = 12, n_walkers = 60,
                         n_steps = 8000, n_burnin = 3000, seed = sd,
                         init_strategy = :prior, show_progress = false)
    rB = sample_ptemcee(tB, data; n_temps = 12, n_walkers = 60,
                         n_steps = 8000, n_burnin = 3000, seed = sd,
                         init_strategy = :prior, show_progress = false)
    push!(zA, rA.log_evidence); push!(zB, rB.log_evidence)
    @printf("seed %d:  logZ_white = %.3f   logZ_MA = %.3f   ΔlogZ = %+.3f\n",
            sd, rA.log_evidence, rB.log_evidence, rB.log_evidence - rA.log_evidence)
end
Δz   = mean(zB) .- mean(zA)
Δz_e = sqrt(std(zA)^2 + std(zB)^2) / sqrt(3)
@printf("\nReference ΔlogZ (MA − white) = %.3f ± %.3f (3-seed)\n", Δz, Δz_e)

# ---- Trans-dim: toggle MA on/off, planet fixed -----------------------
pT, tT = build(NoiseModel[ma])
td = TransDimConfig(; max_kplanet = 1, planets = false, noise = true,
                      toggleable = NoiseModel[ma])
incl = 0.5
pMA = Float64[]
for sd in (1, 2, 3)
    res = sample_transdim_ptemcee(tT, data; td = td, n_temps = 12,
                                   n_walkers = 80, n_steps = 12000,
                                   n_burnin = 4000, n_birth_tries = 8,
                                   n_birth_refine = 10, seed = sd,
                                   show_progress = false)
    a = vec(Array(res.chains[:noise_active_1])) .> 0.5
    p = mean(a)
    push!(pMA, p)
    nacc = sum(res.noise_td_accepted)
    @printf("td seed %d:  P(MA active) = %.3f   (noise toggles accepted = %d)\n",
            sd, p, nacc)
end
p̄  = mean(pMA)
# observed log-odds vs predicted
logit(p) = log(p / (1 - p))
pred_logit = Δz + log(incl / (1 - incl))          # = Δz since incl=0.5
obs_logit  = logit(clamp(p̄, 1e-4, 1 - 1e-4))
@printf("\nP(MA active) = %.3f (3-seed mean, spread %.3f)\n", p̄,
        maximum(pMA) - minimum(pMA))
@printf("observed logit P(MA)   = %+.3f\n", obs_logit)
@printf("predicted (ΔlogZ + logit prior) = %+.3f ± %.3f\n", pred_logit, Δz_e)
disc = abs(obs_logit - pred_logit)
@printf("discrepancy = %.3f nats\n", disc)
if disc < 3 * max(Δz_e, 0.3) + 0.7
    println("\n✅ TOY NOISE MODEL SELECTION VALIDATED — occupancy reproduces the Z-ratio")
else
    println("\n❌ FAIL — occupancy does NOT match the evidence ratio; ",
            "noise-toggle occupancy is not P(M|D)")
end
