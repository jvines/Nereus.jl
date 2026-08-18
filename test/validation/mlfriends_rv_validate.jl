#!/usr/bin/env julia
# THE GOAL: does Bounds.MLFriends remove the ellipsoidal-bound logZ bias on a
# curved RV posterior? Target = n=60 eccentric 1-planet RV (the case where
# multi-ellipsoid NS plateaus at ~-122 regardless of slices/enlarge), with an
# independent reference: TI(PT-HMC) = -115.69 and direct Laplace = -115.56.
# Pass if :mlfriends recovers ~-115.6 (within ~2 nats) while :multi sits ~-122.

using Nereus, MCMCChains, Statistics, Printf, Random

function kepler_rv(t, P, K, e, ω; t0 = 0.0)
    M = 2π .* (t .- t0) ./ P; E = copy(M)
    for _ in 1:60; E .-= (E .- e .* sin.(E) .- M) ./ (1 .- e .* cos.(E)); end
    ν = 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
    return K .* (cos.(ν .+ ω) .+ e * cos(ω))
end
rng = MersenneTwister(11); n = 60
t  = sort(rand(rng, n) .* 80.0)
rv = kepler_rv(t, 7.3, 9.0, 0.3, 0.8) .+ 1.2 .* randn(rng, n)
data = Data(t_rv = t, rv = rv, rv_err = fill(1.2, n), rv_inst = ones(Int, n))
pr = Dict{String,PriorSpec}(
    "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
    "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
    "Mo_k1" => UniformPrior(0.0, 2π),
    "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))
params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
    parametrization = ParametrizationConfig(time = :Mo), priors = pr)
mk() = NereusTarget(params, data; unconstrained = true)

REF = -115.6   # TI(PT-HMC) -115.69 ; Laplace -115.56
NLIVE = 1000   # MLFriends now pairs with :rslice (friends-scale slice, tractable).
@printf("=== MLFriends vs MultiEllipsoid on curved RV (reference logZ = %.1f) ===\n", REF)
med(ch, s) = median(vec(Array(ch[s])))

nlz(b) = begin
    t0 = time()
    ch, lz = sample_nested(mk(), data; n_live = NLIVE, dlogz = 0.1, bounds = b, seed = 1)
    @printf("[bounds=%-10s] logZ=%8.3f  Δ(ref)=%+6.2f   P=%.3f K=%.2f  (%.0fs)\n",
            String(b), lz, lz - REF, med(ch, :P_k1), med(ch, :K_k1), time() - t0)
    lz
end

lz_ml  = nlz(:mlfriends)
lz_mul = nlz(:multi)

@printf("\nMLFriends recovers reference: %s (|Δ|=%.2f < 2)\n",
        abs(lz_ml - REF) < 2.0 ? "YES" : "NO", abs(lz_ml - REF))
@printf("MultiEllipsoid still biased:  %s (|Δ|=%.2f)\n",
        abs(lz_mul - REF) > 3.0 ? "YES" : "NO", abs(lz_mul - REF))
ok = abs(lz_ml - REF) < 2.0 && (lz_ml - lz_mul) > 2.0
println(ok ? "\n✅ MLFriends FIXES the curved-RV NS evidence bias" :
             "\n❌ MLFriends did not close the gap — investigate")
