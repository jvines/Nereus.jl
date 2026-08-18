#!/usr/bin/env julia
# MLFriends' ACTUAL green-lit purpose: the period-alias TRAP (NS confidently
# converging to the WRONG period alias — the blind-51-Peg failure), NOT the
# curved-ridge evidence bias. This is a DISCOVERY/posterior question, tractable
# in low-d. Few points + sparse sampling → comparable alias peaks → ellipsoidal
# NS can latch onto a wrong one. Does MLFriends (balls on live points,
# non-shrinking bootstrap radius) recover the TRUE period where MultiEllipsoid
# traps? Fit (P, K, M0), e=0, wide period prior.

using Nereus, MCMCChains, Statistics, Printf, Random

P_true, K_true = 4.2308, 55.0
rng = MersenneTwister(7); n = 14
# sparse, irregular-ish nightly sampling over a short baseline → strong aliases
t = sort(vcat(collect(0.0:1.0:6.0), collect(13.0:1.0:19.0)) .+ 0.1 .* randn(rng, n))
rv = K_true .* sin.(2π .* t ./ P_true .+ 0.7) .+ 2.0 .* randn(rng, n)
data = Data(t_rv = t, rv = rv, rv_err = fill(2.0, n), rv_inst = ones(Int, n))
pr = Dict{String,PriorSpec}(
    "P_k1" => LogUniformPrior(1.5, 30.0), "K_k1" => UniformPrior(0.0, 150.0),
    "Mo_k1" => UniformPrior(0.0, 2π),
    "sesinw_k1" => FixedPrior(0.0), "secosw_k1" => FixedPrior(0.0),
    "gamma_I1" => FixedPrior(0.0), "sigma_I1" => FixedPrior(2.0))
params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
    parametrization = ParametrizationConfig(time = :Mo), priors = pr)
mk() = NereusTarget(params, data; unconstrained = true)
@printf("=== period-alias trap (true P=%.4f d, %d pts, wide prior 1.5–30 d) ===\n", P_true, n)

for (bd, pp) in ((:multi, :rslice), (:mlfriends, :rslice), (:mlfriends, :unif))
    t0 = time()
    ch, lz = sample_nested(mk(), data; n_live = 1000, dlogz = 0.1, bounds = bd, proposal = pp, seed = 1)
    P = vec(Array(ch[:P_k1]))
    hit = abs(median(P) - P_true) < 0.05
    @printf("[%-9s+%-6s] P=%.4f [%.4f,%.4f]  K=%.1f  logZ=%.2f  %s  (%.0fs)\n",
            String(bd), String(pp), median(P), quantile(P,0.16), quantile(P,0.84),
            median(vec(Array(ch[:K_k1]))), lz, hit ? "✓TRUE" : "✗ALIAS", time()-t0)
end
@printf("\nIf multi ✗ALIAS but mlfriends ✓TRUE → MLFriends fixes the alias trap (its purpose).\n")
