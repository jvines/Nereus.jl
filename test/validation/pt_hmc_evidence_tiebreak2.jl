#!/usr/bin/env julia
# Round 2: NS (-98) vs TI-family (climbing to ~-92) split by ~6 nats. Decisive
# checks:
#  (1) does NESTED recover the posterior (P=7.3, K=9.0)? If its multi-ellipsoid
#      bound mis-handles the RV period geometry, its logZ is the artifact
#      (cf. blind 51 Peg: ellipsoidal NS alias-trapped, PT recovers).
#  (2) is NS logZ n_live-stable (1500 → 4000)?
#  (3) where does the TI family PLATEAU (PT-HMC adaptive 24 temps; stretch 48)?

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

med(ch, s) = median(vec(Array(ch[s])))

@printf("=== evidence tie-breaker round 2 (truth P=%.2f K=%.1f) ===\n\n", P_true, K_true)

# (1)+(2) nested posterior recovery + n_live stability
for nl in (1500, 4000)
    t0 = time()
    ch, lz = sample_nested(mk(), data; n_live = nl, dlogz = 0.05)
    @printf("[NS  n_live=%4d]  logZ=%8.3f   P=%.3f  K=%.2f   (%.0fs)\n",
            nl, lz, med(ch, :P_k1), med(ch, :K_k1), time() - t0)
end

# (3) TI family plateau
println()
for nt in (20, 24)
    t0 = time()
    ch, lz, _ = Nereus.sample_pt_hmc(mk(); n_temps = nt, n_sweeps = 2000,
                                      n_warmup = 400, n_walkers_per_temp = 2,
                                      seed = 1, progress = false)
    @printf("[PT-HMC %2d temps]  logZ=%8.3f   P=%.3f  K=%.2f   (%.0fs)\n",
            nt, lz, med(ch, :P_k1), med(ch, :K_k1), time() - t0)
end
for nc in (48,)
    t0 = time()
    _, lz = sample_pt(mk(); n_rounds = 14, n_chains = nc, seed = 1, show_report = false)
    @printf("[stretch %2d temps] logZ=%8.3f                       (%.0fs)\n", nc, lz, time() - t0)
end
