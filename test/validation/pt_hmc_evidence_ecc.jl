#!/usr/bin/env julia
# Decisive: NS(-98.4) vs TI-family(-92.5) split by ~6 nats on the CIRCULAR
# injection, both internally converged + same posterior. Prime suspect: the e=0
# sesinw/secosw origin degeneracy (Mo/ω unidentifiable → posterior cusp → NS and
# TI integrate the degenerate volume differently). Test: add an independent
# LAPLACE anchor (sample_map.log_evidence_laplace; reliable only when the
# posterior is near-Gaussian) and re-run on an ECCENTRIC injection.
#   - circular: expect Laplace garbage/NaN (degenerate Hessian) + NS≠TI
#   - eccentric: expect Laplace finite + NS≈TI≈Laplace  → confirms e=0 artifact

using Nereus, MCMCChains, Statistics, Printf, Random

base_priors() = Dict{String,PriorSpec}(
    "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
    "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
    "Mo_k1" => UniformPrior(0.0, 2π),
    "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))

function kepler_rv(t, P, K, e, ω; t0 = 0.0)
    M = 2π .* (t .- t0) ./ P
    E = copy(M)
    for _ in 1:60
        E .-= (E .- e .* sin.(E) .- M) ./ (1 .- e .* cos.(E))
    end
    ν = 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
    return K .* (cos.(ν .+ ω) .+ e * cos(ω))
end

function build(e)
    rng = MersenneTwister(11); n = 45
    t  = sort(rand(rng, n) .* 60.0)
    rv = kepler_rv(t, 7.3, 9.0, e, 0.8) .+ 1.2 .* randn(rng, n)
    data = Data(t_rv = t, rv = rv, rv_err = fill(1.2, n), rv_inst = ones(Int, n))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
        instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
        parametrization = ParametrizationConfig(time = :Mo), priors = base_priors())
    return params, data
end

med(ch, s) = median(vec(Array(ch[s])))
emed(ch) = median(sqrt.(vec(Array(ch[:sesinw_k1])).^2 .+ vec(Array(ch[:secosw_k1])).^2))

for (tag, e) in (("CIRCULAR e=0.0", 0.0), ("ECCENTRIC e=0.3", 0.3))
    params, data = build(e)
    mk() = NereusTarget(params, data; unconstrained = true)
    @printf("\n========== %s ==========\n", tag)

    chn, lzn = sample_nested(mk(), data; n_live = 1500, dlogz = 0.05)
    @printf("[NS      ]  logZ=%8.3f   P=%.3f K=%.2f e=%.2f\n",
            lzn, med(chn,:P_k1), med(chn,:K_k1), emed(chn))

    chh, lzh, _ = Nereus.sample_pt_hmc(mk(); n_temps = 20, n_sweeps = 2000,
                                        n_warmup = 400, n_walkers_per_temp = 2,
                                        seed = 1, progress = false)
    @printf("[PT-HMC  ]  logZ=%8.3f   P=%.3f K=%.2f e=%.2f\n",
            lzh, med(chh,:P_k1), med(chh,:K_k1), emed(chh))

    _, lzs = sample_pt(mk(); n_rounds = 14, n_chains = 24, seed = 1, show_report = false)
    @printf("[stretch ]  logZ=%8.3f\n", lzs)

    r = sample_map(mk(); n_starts = 48, seed = 7)
    @printf("[Laplace ]  logZ=%8.3f   (converged=%s)\n", r.log_evidence_laplace, r.converged)
end
