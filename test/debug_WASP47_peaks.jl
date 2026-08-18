#!/usr/bin/env julia
# Instrument the informed-birth peak machinery on time-corrected WASP-47:
# what does _bls_peaks_filtered return (refined P + dur), and how does the
# JointInformedBirth merge classify b (source + σ)?

using Nereus
using DelimitedFiles: readdlm
using Statistics: median
using Printf, Random

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const P_B = 4.1591287

pm_names=["K2","TESS42","TESS92"]
files=["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
pt_t=Float64[]; pt_f=Float64[]; pt_e=Float64[]; pt_i=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(pt_t,lc.t); append!(pt_f,lc.flux)
    append!(pt_e,lc.flux_err); append!(pt_i,fill(ix,length(lc.t))); end
perm=sortperm(pt_t); pt_t=pt_t[perm]; pt_f=pt_f[perm]; pt_e=pt_e[perm]; pt_i=pt_i[perm]
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
pri["P_k1"]=LogUniformPrior(0.5,30.0); pri["K_k1"]=UniformPrior(0.0,250.0)
pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
pri["Tc_k1"]=UniformPrior(pt_t[1],pt_t[1]+30.0); pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue; ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
for n in pm_names
    pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
pri["rho_s"]=NormalPrior(0.707,0.2,0.1,5.0)
params=Params(;max_kplanet=4,planet_modes=[RVPM,RVPM,RVPM,RV_ONLY],instruments=ic,data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,trend_order=0,stability=:amd)
target=NereusTarget(params,data;unconstrained=true)
td=TransDimState(max_planets=params.config.max_kplanet); th=Theta(params; td=td)

# direct BLS peaks (refined)
cache=Nereus.BLSCache()
P_ref, w, dep, t0, dur = Nereus._bls_peaks_filtered(th, data, 0.5, 30.0, cache)
println("=== _bls_peaks_filtered (refined) ===")
for i in eachindex(P_ref)
    @printf("  P=%.6f  w=%.3f  depth=%.5f  t0=%.4f  dur=%.4f d   |ΔP/P vs b|=%.2e\n",
            P_ref[i], w[i], dep[i], t0[i], dur[i], abs(P_ref[i]-P_B)/P_B)
end
baseline=pt_t[end]-pt_t[1]
@printf("\nbaseline=%.1f d  -> for a b peak: dur/baseline=%.2e (σ would be clamp to [1e-5,0.05])\n",
        baseline, (isempty(dur) ? 0.0 : dur[argmin(abs.(P_ref.-P_B))]/baseline))

# LS (RV) peaks for comparison
resid=Nereus._compute_rv_residuals(th, data)
lsP, lsw, lsK, lsMo = Nereus._find_peaks(data.t_rv, resid, 0.5, 30.0; t_ref=data.t_ref, σ=data.rv_err)
println("\n=== LS (RV) peaks ===")
for i in eachindex(lsP)
    @printf("  P=%.5f  w=%.3f  K=%.1f   |ΔP/P vs b|=%.2e\n", lsP[i], lsw[i], lsK[i], abs(lsP[i]-P_B)/P_B)
end
