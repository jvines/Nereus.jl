#!/usr/bin/env julia
# WASP-47 b end-to-end recovery AFTER the TESS BTJD->BJD time fix.
# Corrected LCs load directly (all full BJD). Tight P prior + Tc seed
# emulate a real BLS/TLS pre-search (the realistic transit setup — blind
# LogUniform(0.5,30) can't cross the long-baseline ephemeris needle).
# Joint 1-planet RVPM fit. Success = K~140, rr~0.094, deep aligned transit.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, quantile
using Printf
using MCMCChains

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const B = (P=4.1591287, K=140.84, rr=0.094)

pm_names=["K2","TESS42","TESS92"]
files=["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
pt_t=Float64[]; pt_f=Float64[]; pt_e=Float64[]; pt_i=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(pt_t,lc.t); append!(pt_f,lc.flux)
    append!(pt_e,lc.flux_err); append!(pt_i,fill(ix,length(lc.t))); end
perm=sortperm(pt_t); pt_t=pt_t[perm]; pt_f=pt_f[perm]; pt_e=pt_e[perm]; pt_i=pt_i[perm]
@printf("baseline = %.1f d (%.0f periods)  — all full BJD\n", pt_t[end]-pt_t[1], (pt_t[end]-pt_t[1])/B.P)

rr=readdlm(joinpath(DATADIR,"WASP47_RVs_combined.csv"),',',Any,'\n';header=true)
raw=rr[1]; hdr=vec(rr[2]); c(n)=findfirst(==(n),hdr); fcol=c("flag"); keep=trues(size(raw,1))
for i in 1:size(raw,1); f=raw[i,fcol]; f===missing&&continue
    String(strip(string(f))) in ("transit","transit_night","anomalous")&&(keep[i]=false); end
raw=raw[keep,:]; bjd=Float64.(raw[:,c("bjd")]); rv=Float64.(raw[:,c("rv")]); rve=Float64.(raw[:,c("rv_err")])
istr=String.(raw[:,c("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
rinst=[i2i[s] for s in istr]
data=Data(;t_rv=bjd,rv=rv,rv_err=rve,rv_inst=rinst,t_phot=pt_t,flux=pt_f,flux_err=pt_e,phot_inst=pt_i)
ic=InstrumentConfig(rv=inames,pm=pm_names)

pri=Dict{String,PriorSpec}()
pri["n_p"]=FixedPrior(1.0)
pri["P_k1"]=UniformPrior(4.15905,4.15922)            # tight: from a pre-search
pri["K_k1"]=UniformPrior(0.0,250.0)
pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
pri["Tc_k1"]=UniformPrior(2456978.3,2456979.3)       # seed near a real transit
pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
    ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
for n in pm_names
    pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
pri["rho_s"]=NormalPrior(0.707,0.2,0.1,5.0)
params=Params(;max_kplanet=1,planet_modes=[RVPM],instruments=ic,data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,
    trend_order=0,stability=:none,external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])
target=NereusTarget(params,data;unconstrained=true)

println("== WASP-47 b joint recovery (time-fixed, seeded) ==")
t0=time()
res=sample_ptemcee(target,data;n_temps=8,n_walkers=100,n_steps=1200,n_burnin=700,seed=42,show_progress=false)
secs=time()-t0
ch=res.chains; gv(s)=vec(Array(ch[:,s,:]))
Kq=quantile(gv(:K_k1),[0.16,0.5,0.84]); rrq=quantile(gv(:rr_k1),[0.16,0.5,0.84])
@printf("\n%.0fs\n", secs)
@printf("  P  = %.6f d        (truth %.6f)\n", median(gv(:P_k1)), B.P)
@printf("  K  = %.1f [%.1f,%.1f] m/s   (truth %.1f)\n", Kq[2],Kq[1],Kq[3], B.K)
@printf("  rr = %.4f [%.4f,%.4f]      (truth ~%.3f)\n", rrq[2],rrq[1],rrq[3], B.rr)
@printf("  b  = %.3f   Tc = %.4f\n", median(gv(:b_k1)), median(gv(:Tc_k1)))
ok = abs(Kq[2]-B.K)/B.K < 0.15 && abs(rrq[2]-B.rr)/B.rr < 0.25
println(ok ? "\n>> RECOVERED: time fix + tight seed brings b back at the published K and depth." :
             "\n>> still off — inspect (period/phase lock or budget).")
