#!/usr/bin/env julia
# Localise the ~2.5-nat TI(PT/HMC) − NS evidence offset. Both integrate the SAME
# likelihood, so a near-constant offset must live in the PRIOR measure:
#   TI logZ = ∫ exp(log_prior(θ)) L dθ  → equals truth ONLY if ∫exp(log_prior)dθ=1
#   NS  logZ = ∫ L dπ_NS , π_NS = ∏ prior_dists[j]  (normalised by construction)
# If c ≡ log ∫ exp(log_prior) dθ ≠ 0, then TI = NS + c. Estimate c by importance
# sampling against π_NS:  ∫exp(log_prior)dθ = E_{π_NS}[exp(log_prior − log π_NS)].
# Also confirm log_prior == Σ logpdf(prior_dists) pointwise (else the two priors
# are genuinely different measures — e.g. a joint e<1 truncation).

using Nereus, Random, Printf, Statistics
using Distributions: logpdf
using LogExpFunctions: logsumexp

priors = Dict{String,PriorSpec}(
    "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
    "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
    "Mo_k1" => UniformPrior(0.0, 2π),
    "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))
rng = MersenneTwister(1); n = 30
t = sort(rand(rng, n) .* 60.0)
data = Data(t_rv = t, rv = randn(rng, n), rv_err = fill(1.2, n), rv_inst = ones(Int, n))
params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
    parametrization = ParametrizationConfig(time = :Mo), priors = priors)
target = NereusTarget(params, data; unconstrained = true)
layout = params.layout
dists = [layout.unfrozen_priors[j].dist for j in 1:length(layout.unfrozen_idx)]
names = layout.unfrozen_names
@printf("params: %s\n", join(string.(names), ", "))

lp_internal(x) = (th = Nereus.Theta{Float64}(params); Nereus.set_unfrozen!(th, x); Nereus.log_prior(th))

# pointwise: log_prior vs Σ logpdf(prior_dists)
rng2 = MersenneTwister(7); N = 50_000
diffs = Float64[]; finite_int = Float64[]
for _ in 1:N
    x = [rand(rng2, d) for d in dists]
    lp  = lp_internal(x)
    lpd = sum(logpdf(dists[j], x[j]) for j in eachindex(dists))
    push!(diffs, lp - lpd)
    push!(finite_int, isfinite(lp) ? (lp - lpd) : -Inf)
end
fin = filter(isfinite, diffs)
frac_rej = count(!isfinite, diffs) / N    # θ that log_prior rejects but π_NS allows (e<1 etc.)
@printf("\nlog_prior − Σlogpdf(dist):  mean=%.4f  std=%.4f  (over %d finite)\n",
        mean(fin), std(fin), length(fin))
@printf("fraction θ∼π_NS rejected by log_prior (=-Inf): %.4f\n", frac_rej)
# c = log ∫ exp(log_prior) dθ  = log E_{π_NS}[ exp(log_prior − log π_NS) ]
c = logsumexp(finite_int) - log(N)
@printf("c = log ∫exp(log_prior)dθ = %.4f   ⇒  TI should sit  c=%.3f  vs NS\n", c, c)
@printf("   (if c≈0 prior is normalised → offset is sampling/estimator, not prior)\n")

# likelihood parity: ws-aware (NS) vs plain (TI) at one θ
let x = [rand(rng2, d) for d in dists]
    th = Nereus.Theta{Float64}(params); Nereus.set_unfrozen!(th, x)
    ll_plain = Nereus.rv_log_likelihood(th, data)
    wb = Nereus.PTWorkspace(params, params.config.max_kplanet,
                             length(params.config.noise_models);
                             n_obs = length(data.t_rv), n_phot = length(data.t_phot))
    ll_ws = Nereus.rv_log_likelihood(th, data, wb)
    @printf("\nlikelihood parity: plain=%.6f  ws=%.6f  Δ=%.2e\n", ll_plain, ll_ws, ll_plain - ll_ws)
end
