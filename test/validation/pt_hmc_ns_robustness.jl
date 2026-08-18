#!/usr/bin/env julia
# Is NS the ~2.5-nat-low one? Hypothesis: multi-ellipsoid+rslice under-mixes the
# curved e–ω–Mo ridge of the RV posterior → live points not uniform on {L>L*} →
# logZ biased LOW. Test: strengthen decorrelation (more slices/walks) and vary
# bound/proposal. If NS climbs toward the TI value (~-94.6 eccentric) it was
# under-mixing → TI is right. If NS is stable across settings, the gap is real.

using Nereus, MCMCChains, Statistics, Printf, Random

function kepler_rv(t, P, K, e, ω; t0 = 0.0)
    M = 2π .* (t .- t0) ./ P; E = copy(M)
    for _ in 1:60; E .-= (E .- e .* sin.(E) .- M) ./ (1 .- e .* cos.(E)); end
    ν = 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
    return K .* (cos.(ν .+ ω) .+ e * cos(ω))
end
rng = MersenneTwister(11); n = 45
t  = sort(rand(rng, n) .* 60.0)
rv = kepler_rv(t, 7.3, 9.0, 0.3, 0.8) .+ 1.2 .* randn(rng, n)
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

@printf("=== NS robustness on ECCENTRIC RV (TI reference ≈ -94.6) ===\n\n")
configs = [
    (:multi,  :rslice, 5,  25, "default"),
    (:multi,  :rslice, 15, 50, "strong rslice"),
    (:multi,  :slice,  15, 50, "slice"),
    (:multi,  :rwalk,  5,  100,"rwalk x100"),
    (:single, :rslice, 15, 50, "single-ellipsoid"),
]
for (b, p, sl, nw, tag) in configs
    t0 = time()
    ch, lz = sample_nested(mk(), data; n_live = 1500, dlogz = 0.05,
                           bounds = b, proposal = p, slices = sl, n_walks = nw)
    @printf("[%-17s b=%-7s p=%-7s]  logZ=%8.3f  P=%.3f K=%.2f  (%.0fs)\n",
            tag, b, p, lz, median(vec(Array(ch[:P_k1]))), median(vec(Array(ch[:K_k1]))), time() - t0)
end
@printf("\nclimb toward -94.6 ⇒ NS was under-mixing (TI right); flat ⇒ real gap\n")
