#!/usr/bin/env julia
# Smoke: confirm sample_nested_ins / sample_nested_dynamic RUN end-to-end with
# the new dimension-scaled default (n_walks = nothing → resolved), returning a
# finite logZ. (Correctness of their logZ is the same bound-bias story as
# sample_nested; this only checks the nothing→resolution plumbing doesn't break.)
using Nereus, Random, Printf

rng = MersenneTwister(5); n = 35
t  = sort(rand(rng, n) .* 50.0)
rv = 8.0 .* sin.(2π .* t ./ 6.5 .+ 0.4) .+ 1.0 .* randn(rng, n)
data = Data(t_rv = t, rv = rv, rv_err = fill(1.0, n), rv_inst = ones(Int, n))
pr = Dict{String,PriorSpec}(
    "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
    "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
    "Mo_k1" => UniformPrior(0.0, 2π),
    "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))
params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
    parametrization = ParametrizationConfig(time = :Mo), priors = pr)
mk() = NereusTarget(params, data; unconstrained = true)
ndim = length(params.layout.unfrozen_idx)
@printf("ndim=%d → expect n_walks resolves to max(25, 2·ndim)=%d\n", ndim, max(25, 2*ndim))

ri = sample_nested_ins(mk(), data; n_live = 200, dlogz = 0.5, max_iter = 20_000, seed = 1)
@printf("[INS    ] log_z_ins=%.2f  log_z_ns=%.2f  %s\n",
        ri.log_z_ins, ri.log_z_ns, isfinite(ri.log_z_ins) ? "OK" : "FAIL")

rd = sample_nested_dynamic(mk(), data; n_live_init = 120, n_live_batch = 200,
                           dlogz_init = 2.0, dlogz_batch = 0.8, max_iter = 20_000, seed = 1)
@printf("[Dynamic] log_z=%.2f  %s\n", rd.log_z, isfinite(rd.log_z) ? "OK" : "FAIL")

println(isfinite(ri.log_z_ins) && isfinite(rd.log_z) ?
        "\n✅ INS + dynamic run with dimension-scaled default" : "\n❌ FAIL")
