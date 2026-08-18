#!/usr/bin/env julia
# Empirically pin the dimension-scaled rslice schedule for sample_nested.
# Usage:  julia ns_slice_sweep.jl <mode> <arg1> <arg2> <nplanets>
#   mode=ns : arg1=slices, arg2=seed     → NS logZ at that slice count
#   mode=ti : arg1=seed    (arg2 unused) → PT-HMC TI+ reference logZ
# nplanets ∈ {1,2} sets the dimension (1→7-D, 2→~11-D).

using Nereus, MCMCChains, Statistics, Printf, Random

function kepler_rv(t, P, K, e, ω; t0 = 0.0)
    M = 2π .* (t .- t0) ./ P; E = copy(M)
    for _ in 1:60; E .-= (E .- e .* sin.(E) .- M) ./ (1 .- e .* cos.(E)); end
    ν = 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
    return K .* (cos.(ν .+ ω) .+ e * cos(ω))
end

function build(nplanets)
    rng = MersenneTwister(11); n = 60
    t  = sort(rand(rng, n) .* 80.0)
    rv = kepler_rv(t, 7.3, 9.0, 0.3, 0.8) .+ 1.2 .* randn(rng, n)
    pr = Dict{String,PriorSpec}(
        "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
        "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
        "Mo_k1" => UniformPrior(0.0, 2π),
        "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))
    if nplanets == 2
        rv .+= kepler_rv(t, 15.1, 6.0, 0.2, 2.1)
        merge!(pr, Dict{String,PriorSpec}(
            "P_k2" => LogUniformPrior(2.0, 30.0), "K_k2" => UniformPrior(0.0, 30.0),
            "sesinw_k2" => UniformPrior(-0.9, 0.9), "secosw_k2" => UniformPrior(-0.9, 0.9),
            "Mo_k2" => UniformPrior(0.0, 2π)))
    end
    data = Data(t_rv = t, rv = rv, rv_err = fill(1.2, n), rv_inst = ones(Int, n))
    params = Params(; max_kplanet = nplanets, planet_modes = fill(RV_ONLY, nplanets),
        instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
        parametrization = ParametrizationConfig(time = :Mo), priors = pr)
    return params, data
end

mode = ARGS[1]
nplanets = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
params, data = build(nplanets)
ndim = length(params.layout.unfrozen_idx)
mk() = NereusTarget(params, data; unconstrained = true)

if mode == "ti"
    seed = parse(Int, ARGS[2])
    _, lz, rep = Nereus.sample_pt_hmc(mk(); n_temps = 22, n_sweeps = 2200,
                                       n_warmup = 450, n_walkers_per_temp = 2,
                                       seed = seed, progress = false)
    @printf("[TI  ndim=%2d seed=%d] logZ=%8.3f ± %.3f\n", ndim, seed, lz, rep.ti_plus[2])
else
    slices = parse(Int, ARGS[2]); seed = parse(Int, ARGS[3])
    enl = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : 1.25
    t0 = time()
    _, lz = sample_nested(mk(), data; n_live = 1500, dlogz = 0.05,
                          proposal = :rslice, slices = slices, enlarge = enl, seed = seed)
    @printf("[NS  ndim=%2d slices=%2d enl=%.2f seed=%d] logZ=%8.3f  (%.0fs)\n",
            ndim, slices, enl, seed, lz, time() - t0)
end
