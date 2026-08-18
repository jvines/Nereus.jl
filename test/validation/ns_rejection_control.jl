#!/usr/bin/env julia
# Control I should have run before MLFriends: isolate BOUND vs PROPOSAL on the
# curved n=60 RV target (TI/Laplace reference = -115.6). MLFriends pairs with
# Rejection, so compare ellipsoid bounds ALSO under Rejection (:unif):
#   - :multi + :unif   (rejection from multi-ellipsoid)
#   - :single + :unif
#   - :multi + :rslice (the earlier -122 config, for reference)
# If ellipsoid+rejection ≈ -115.6, the bias was the rslice PROPOSAL, not the
# bound → MLFriends unnecessary. If ≈ -122, the ellipsoid BOUND under-covers →
# MLFriends justified.

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
REF = -115.6
@printf("=== bound vs proposal control (reference logZ = %.1f) ===\n", REF)

for (bd, pp) in ((:multi, :unif), (:single, :unif), (:multi, :rslice))
    t0 = time()
    _, lz = sample_nested(mk(), data; n_live = 1000, dlogz = 0.1, bounds = bd, proposal = pp, seed = 1)
    @printf("[%-7s + %-7s] logZ=%8.3f  Δ(ref)=%+6.2f   (%.0fs)\n",
            String(bd), String(pp), lz, lz - REF, time() - t0)
end
