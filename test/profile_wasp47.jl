#!/usr/bin/env julia
# Profile WHERE the ~130ms/eval goes on the native-cadence WASP-47 trans-dim
# likelihood. Builds the same target, runs a short rjmcmc under the statistical
# profiler, prints the hot frames in Nereus source (transit? GP? RV? alloc?).
# Also times a single full logdensity eval and an RV-only variant for a number.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std
using Printf
using Profile

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))

function build(; with_phot=true, with_gp=true)
    lc = load_tess_lc(joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"))
    p = sortperm(lc.t); flat = (t=lc.t[p], flux=lc.flux[p], flux_err=lc.flux_err[p])
    rr = readdlm(joinpath(DATADIR, "WASP47_RVs_combined.csv"), ',', Any, '\n'; header=true)
    raw = rr[1]; hdr = vec(rr[2]); c(n)=findfirst(==(n),hdr)
    fcol=c("flag"); keep=trues(size(raw,1))
    for i in 1:size(raw,1); f=raw[i,fcol]; f===missing && continue
        String(strip(string(f))) in ("transit","transit_night","anomalous") && (keep[i]=false); end
    raw=raw[keep,:]
    bjd=Float64.(raw[:,c("bjd")]); rv=Float64.(raw[:,c("rv")]); rve=Float64.(raw[:,c("rv_err")])
    istr=String.(raw[:,c("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
    rinst=[i2i[s] for s in istr]
    data = with_phot ?
        Data(; t_rv=bjd, rv=rv, rv_err=rve, rv_inst=rinst, t_phot=flat.t, flux=flat.flux,
             flux_err=flat.flux_err, phot_inst=ones(Int,length(flat.t))) :
        Data(; t_rv=bjd, rv=rv, rv_err=rve, rv_inst=rinst)
    ic = with_phot ? InstrumentConfig(rv=inames, pm=["TESS"]) : InstrumentConfig(rv=inames)
    pri = Dict{String,PriorSpec}()
    for k in 1:3
        pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["K_k$k"]=UniformPrior(0.0,200.0)
        pri["sesinw_k$k"]=UniformPrior(-0.7,0.7); pri["secosw_k$k"]=UniformPrior(-0.7,0.7)
        if with_phot
            pri["Tc_k$k"]=UniformPrior(flat.t[1],flat.t[1]+30.0)
            pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
        end
    end
    pri["P_k4"]=LogUniformPrior(100.0,1500.0); pri["K_k4"]=UniformPrior(0.0,200.0)
    pri["sesinw_k4"]=UniformPrior(-0.7,0.7); pri["secosw_k4"]=UniformPrior(-0.7,0.7)
    for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
        ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
        pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
    if with_phot
        pri["offset_TESS"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_TESS"]=LogUniformPrior(1e-5,5e-3)
        pri["q1_TESS"]=UniformPrior(0.0,1.0); pri["q2_TESS"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(1.0,0.3,0.1,5.0)
    end
    modes = with_phot ? [RVPM,RVPM,RVPM,RV_ONLY] : [RV_ONLY,RV_ONLY,RV_ONLY,RV_ONLY]
    par = with_phot ? ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true) : ParametrizationConfig()
    params = Params(; max_kplanet=4, planet_modes=modes, instruments=ic, data=data, M_s=1.04, R_s=1.137,
        parametrization=par, priors=pri, trend_order=0, stability=:amd,
        external_priors=[ExternalPrior(:ecc, NormalPrior(0.0,0.3), true)])
    target = NereusTarget(params, data; unconstrained=true)
    td = TransDimConfig(max_kplanet=4, planets=true, noise=false,
        birth_strategies=[PriorBirth(),InformedBirth()], birth_weights=[0.3,0.7], transdim_fraction=0.3)
    return target, td, length(flat.t), length(bjd)
end

target, td, nphot, nrv = build(; with_phot=true, with_gp=true)
@printf("native s42: %d phot, %d RV\n", nphot, nrv)

println("compiling rjmcmc (tiny) ...")
sample_rjmcmc(target, target.data; td=td, n_samples=2, n_warmup=3, seed=1, n_chains=1)

println("profiling rjmcmc 25 warmup iters, single chain ...")
Profile.clear(); Profile.init(n=20_000_000, delay=0.0005)
@profile sample_rjmcmc(target, target.data; td=td, n_samples=2, n_warmup=25, seed=1, n_chains=1)

open(joinpath(@__DIR__, "wasp47_profile_flat.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize=>(100000,400)); format=:flat,
                  sortedby=:count, mincount=10)
end
open(joinpath(@__DIR__, "wasp47_profile_tree.txt"), "w") do io
    Profile.print(IOContext(io, :displaysize=>(100000,400)); format=:tree,
                  mincount=20, maxdepth=30)
end
println("wrote wasp47_profile_flat.txt and wasp47_profile_tree.txt")
println("Done.")
