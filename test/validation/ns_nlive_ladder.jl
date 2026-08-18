#!/usr/bin/env julia
# DECISIVE paper-gating experiment: is the curved-RV NS evidence bias a genuine
# proposal/exploration limit (persists at high n_live) or just under-resourcing
# (vanishes with more live points → KNOWN result, no paper)?
# Target: n=60 eccentric 1-planet RV. Truth (TI/Laplace) = -115.6.
# multi+rslice at increasing n_live; if logZ stays ~-121 → real; if → -115.6 → artifact.

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
@printf("=== NS n_live ladder, multi+rslice (truth %.1f) ===\n", REF)
for nl in (1000, 4000, 8000, 16000)
    t0 = time()
    _, lz = sample_nested(mk(), data; n_live = nl, dlogz = 0.05, bounds = :multi, proposal = :rslice, seed = 1)
    @printf("[n_live=%5d] logZ=%8.3f  Δ(truth)=%+6.2f   (%.0fs)\n", nl, lz, lz - REF, time() - t0)
end
@printf("\nbias persists at n_live=16000 ⇒ genuine proposal/exploration limit (paper viable);\nlogZ→%.1f ⇒ under-resourcing artifact (KNOWN result, no paper)\n", REF)
