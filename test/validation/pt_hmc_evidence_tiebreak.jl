#!/usr/bin/env julia
# Tie-breaker: PT-HMC TI+ (-93.48) vs in-house stretch-PT TI+ (-97.12) disagree
# by 3.65 nats on the synthetic 1-planet RV. Both use the SAME TI+ estimator, so
# the gap is sampler+ladder resolution. PT-HMC's adaptive ladder passes the
# analytic gate to 0.01 nats → suspect the stretch-PT is under-resolved (coarse
# 12-temp fixed ladder → TI biased LOW near β=1). Test with:
#   (a) nested sampling — INDEPENDENT algorithm, no β-ladder/TI → ground-truth-ish
#   (b) denser stretch-PT (24 temps, 14 rounds) — does its TI climb toward -93.5?
# If both → -93.5, PT-HMC is right and the gate's reference was under-resolved.

using Nereus, MCMCChains, Statistics, Printf, Random

rng = MersenneTwister(11); n = 45
t   = sort(rand(rng, n) .* 60.0)
P_true, K_true = 7.3, 9.0
rv  = K_true .* sin.(2π .* t ./ P_true .+ 0.5) .+ 1.2 .* randn(rng, n)
data = Data(t_rv = t, rv = rv, rv_err = fill(1.2, n), rv_inst = ones(Int, n))
priors = Dict{String,PriorSpec}(
    "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
    "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
    "Mo_k1" => UniformPrior(0.0, 2π),
    "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))
params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
    parametrization = ParametrizationConfig(time = :Mo), priors = priors)
mk() = NereusTarget(params, data; unconstrained = true)

@printf("=== evidence tie-breaker (synthetic 1-planet RV) ===\n")
@printf("reference: PT-HMC TI+ = -93.48 ± 1.50 ; stretch-PT(12) TI+ = -97.12\n\n")

# (a) nested sampling — independent of any β-ladder / TI
t0 = time()
_, logz_ns = sample_nested(mk(), data; dlogz = 0.1)
@printf("[NS ]  log Z = %8.3f      (%.0fs)  — independent algorithm\n", logz_ns, time() - t0)

# (b) denser stretch-PT — if TI was under-resolved it should climb toward PT-HMC
for nc in (12, 24, 36)
    t1 = time()
    _, lz = sample_pt(mk(); n_rounds = 14, n_chains = nc, seed = 1, show_report = false)
    @printf("[PT ]  log Z = %8.3f      (%.0fs)  — stretch-PT, %d temps\n", lz, time() - t1, nc)
end

@printf("\nIf NS and dense-PT both ≈ -93.5 → PT-HMC vindicated; the <2-nat gate's\n")
@printf("stretch-PT(12) reference was under-resolved. If they sit at -97 → PT-HMC biased high.\n")
