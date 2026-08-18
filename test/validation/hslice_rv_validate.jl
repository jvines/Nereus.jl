#!/usr/bin/env julia
# Does the reflective/Hamiltonian slice (:hslice) — which follows the ridge via
# gradient reflections — fix the curved-RV NS evidence bias that :rslice can't?
# Target: n=60 eccentric 1-planet RV. Reference (TI/Laplace) = -115.6;
# :rslice plateaus ~-121. Pass if :hslice recovers ~-115.6.

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
REF = -115.6; NLIVE = 600
@printf("=== :hslice vs :rslice on curved RV (reference logZ = %.1f, n_live=%d) ===\n", REF, NLIVE)
med(ch, s) = median(vec(Array(ch[s])))

for (pp, kw) in (("rslice", (proposal=:rslice,)),
                 ("hslice", (proposal=:hslice, slices=4)))
    t0 = time()
    ch, lz = sample_nested(mk(), data; n_live = NLIVE, dlogz = 0.15, bounds = :multi, seed = 1, kw...)
    @printf("[multi + %-7s] logZ=%8.3f  Δ(ref)=%+6.2f   P=%.3f K=%.2f  (%.0fs)\n",
            pp, lz, lz - REF, med(ch, :P_k1), med(ch, :K_k1), time() - t0)
end
@printf("\nhslice within ~2 of ref ⇒ reflective slice follows the ridge → fixes the bias\n")
