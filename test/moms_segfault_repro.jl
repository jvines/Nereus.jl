#!/usr/bin/env julia
# Minimal reproduction of the sample_moms segfault on WASP-47 (native cadence).
# Smoke crashed with signal 11 at warmup iter 1. Run this with `julia -t 1`
# (single thread) AND n_chains=1 to distinguish a threading race (would NOT
# crash single-threaded) from an initialization out-of-bounds (WOULD).

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
println("threads = ", Threads.nthreads())

lc_raw = load_tess_lc(joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"))
perm = sortperm(lc_raw.t)
flat = (t = lc_raw.t[perm], flux = lc_raw.flux[perm], flux_err = lc_raw.flux_err[perm])

rv_raw_full = readdlm(joinpath(DATADIR, "WASP47_RVs_combined.csv"), ',', Any, '\n'; header=true)
rv_raw = rv_raw_full[1]; hdr = vec(rv_raw_full[2])
col(name) = findfirst(==(name), hdr)
flag_col = col("flag")
keep = trues(size(rv_raw, 1))
for i in 1:size(rv_raw, 1)
    f = rv_raw[i, flag_col]; f === missing && continue
    String(strip(string(f))) in ("transit","transit_night","anomalous") && (keep[i] = false)
end
rv_raw = rv_raw[keep, :]
bjd = Float64.(rv_raw[:, col("bjd")]); rv = Float64.(rv_raw[:, col("rv")])
rv_err = Float64.(rv_raw[:, col("rv_err")]); inst_str = String.(rv_raw[:, col("instrument")])
inst_names = sort(unique(inst_str)); i2i = Dict(n=>i for (i,n) in enumerate(inst_names))
rv_inst = [i2i[s] for s in inst_str]

data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst,
            t_phot=flat.t, flux=flat.flux, flux_err=flat.flux_err,
            phot_inst=ones(Int, length(flat.t)))
ic = InstrumentConfig(rv = inst_names, pm = ["TESS"])
priors = Dict{String, PriorSpec}()
for k in 1:3
    priors["P_k$k"]=LogUniformPrior(0.5,30.0); priors["K_k$k"]=UniformPrior(0.0,200.0)
    priors["sesinw_k$k"]=UniformPrior(-0.7,0.7); priors["secosw_k$k"]=UniformPrior(-0.7,0.7)
    priors["Tc_k$k"]=UniformPrior(flat.t[1], flat.t[1]+30.0)
    priors["b_k$k"]=UniformPrior(0.0,1.0); priors["rr_k$k"]=UniformPrior(0.005,0.20)
end
priors["P_k4"]=LogUniformPrior(100.0,1500.0); priors["K_k4"]=UniformPrior(0.0,200.0)
priors["sesinw_k4"]=UniformPrior(-0.7,0.7); priors["secosw_k4"]=UniformPrior(-0.7,0.7)
for name in inst_names
    rv_i = rv[rv_inst .== i2i[name]]; isempty(rv_i) && continue
    c = median(rv_i); sp = max(maximum(rv_i)-minimum(rv_i), 1.0)
    priors["gamma_$name"]=UniformPrior(c-3*sp, c+3*sp); priors["sigma_$name"]=ModJeffreysPrior(0.1,50.0)
end
priors["offset_TESS"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); priors["jitter_TESS"]=LogUniformPrior(1e-5,5e-3)
priors["q1_TESS"]=UniformPrior(0.0,1.0); priors["q2_TESS"]=UniformPrior(0.0,1.0)
priors["rho_s"]=NormalPrior(1.00,0.30,0.1,5.0)
params = Params(; max_kplanet=4, planet_modes=[RVPM,RVPM,RVPM,RV_ONLY], instruments=ic,
    data=data, M_s=1.04, R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true),
    priors=priors, trend_order=0, stability=:amd,
    external_priors=[ExternalPrior(:ecc, NormalPrior(0.0,0.3), true)])
target = NereusTarget(params, data; unconstrained=true)
td = TransDimConfig(max_kplanet=4, planets=true, noise=false,
    birth_strategies=[PriorBirth(), InformedBirth()], birth_weights=[0.3,0.7], transdim_fraction=0.3)

const NCH = parse(Int, get(ENV, "MOMS_NCHAINS", "1"))
const MARK = joinpath(@__DIR__, "moms_segfault_markers.txt")
mark(s) = (open(MARK, "a") do io; println(io, s); end; println(s); flush(stdout))

mark(">>> threads=$(Threads.nthreads()) n_chains=$NCH : target built, calling sample_moms ...")
const NW = parse(Int, get(ENV, "MOMS_NW", "40"))
const NS = parse(Int, get(ENV, "MOMS_NS", "40"))
const WM = Symbol(get(ENV, "MOMS_WM", "rwm"))
chains, n_evals, _ = sample_moms(target, target.data; td=td, n_samples=NS, n_warmup=NW,
    seed=42, n_chains=NCH, init_scale=1.0, within_model=WM, informed_birth_fraction=0.5)
mark(">>> SURVIVED: threads=$(Threads.nthreads()) n_chains=$NCH n_evals=$n_evals")
