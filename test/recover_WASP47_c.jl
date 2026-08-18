#!/usr/bin/env julia
# WASP-47 c-recovery diagnostic. The 4-slot blind run found only b. c
# (K=31.6, P=588 d, non-transiting) only fits an RV_ONLY slot with
# P∈[100,1500], but `first_inactive_planet` fills slots in order, so c's
# slot 4 is unreachable until the weak d/e fill the RVPM slots 2-3 — the
# 4-slot run NEVER tried a b+c model. This uses a 2-slot [RVPM(b),
# RV_ONLY(c)] model so c is reachable in slot 2, with STANDARD jitter, to
# test whether the blocker was the slot ordering (c appears) or jitter
# absorption (only b appears -> then tighten jitter).
#
#   ENV JIT_MAX (default 50)  — ModJeffreys jitter upper bound, to retest
#   with a tighter cap if standard jitter absorbs c.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, quantile
using Printf, MCMCChains

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const LIT = [(name="b", P=4.1591287, K=140.84), (name="c", P=588.4, K=31.6)]
JIT_MAX = parse(Float64, get(ENV, "JIT_MAX", "50.0"))

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
# slot 1 = b (RVPM, blind)
pri["P_k1"]=LogUniformPrior(0.5,30.0); pri["K_k1"]=UniformPrior(0.0,200.0)
pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
pri["Tc_k1"]=UniformPrior(pt_t[1],pt_t[1]+30.0); pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
# slot 2 = c (RV_ONLY, long period)
pri["P_k2"]=LogUniformPrior(100.0,1500.0); pri["K_k2"]=UniformPrior(0.0,200.0)
pri["sesinw_k2"]=UniformPrior(-0.7,0.7); pri["secosw_k2"]=UniformPrior(-0.7,0.7)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue; ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,JIT_MAX); end
for n in pm_names
    pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
pri["rho_s"]=NormalPrior(1.0,0.3,0.1,5.0)
params=Params(;max_kplanet=2,planet_modes=[RVPM,RV_ONLY],instruments=ic,data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,trend_order=0,
    stability=:amd,external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])
target=NereusTarget(params,data;unconstrained=true)
td=TransDimConfig(max_kplanet=2,planets=true,noise=false,
    birth_strategies=[PriorBirth(),InformedBirth()],birth_weights=[0.3,0.7],transdim_fraction=0.3)

ev(k,d)=parse(Int,get(ENV,k,string(d)))
NT=ev("TD_NT",10); NW=ev("TD_NW",100); NS=ev("TD_NS",4000); NB=ev("TD_NB",2000)
println("== WASP-47 b+c 2-slot recovery (jitter_max=$JIT_MAX) $NT×$NW×$NS+$NB ==")
t0=time()
res=sample_transdim_ptemcee(target,data;td=td,n_temps=NT,n_walkers=NW,n_steps=NS,n_burnin=NB,
                            informed_birth_fraction=0.7,seed=42,show_progress=true)
secs=time()-t0
ch=res.chains
probs=model_probabilities(ch;max_kplanet=2); np_post=[get(probs,k,0.0) for k in 0:2]; modal=argmax(np_post)-1
@printf("\n%.0fs  modalNp=%d  n_p posterior: 0:%.2f 1:%.2f 2:%.2f\n", secs, modal, np_post...)
# per-active-slot medians at the modal model
np_vec=vec(Array(ch[:,:n_planets,:])); mask=np_vec.==modal
has_bits = Symbol("planet_active_1") in names(ch)
for k in 1:2
    sym=Symbol("P_k$k"); sym in names(ch) || continue
    act = has_bits ? vec(Array(ch[:,Symbol("planet_active_$k"),:]))[mask] .> 0.5 : trues(count(mask))
    Pk=vec(Array(ch[:,sym,:]))[mask][act]; Kk=vec(Array(ch[:,Symbol("K_k$k"),:]))[mask][act]
    isempty(Pk) && continue
    @printf("  slot %d: P=%.4f  K=%.1f  (n=%d active)\n", k, median(Pk), median(Kk), length(Pk))
end
found=String[]; for lit in LIT
    for k in 1:2
        sym=Symbol("P_k$k"); sym in names(ch) || continue
        act = has_bits ? vec(Array(ch[:,Symbol("planet_active_$k"),:]))[mask].>0.5 : trues(count(mask))
        Pk=vec(Array(ch[:,sym,:]))[mask][act]; isempty(Pk) && continue
        abs(median(Pk)-lit.P)/lit.P<0.05 && !(lit.name in found) && push!(found,lit.name)
    end
end
@printf("\nfound: [%s]   (target b+c)\n", join(found,","))
println("b" in found && "c" in found ? ">> b+c RECOVERED with jitter_max=$JIT_MAX" :
        "c" in found ? ">> c found" :
        "b" in found ? ">> only b — c still missing (try tighter JIT_MAX)" : ">> neither — inspect")
