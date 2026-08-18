#!/usr/bin/env julia
# JOINT K2 + RV blind trans-dim recovery (the real test — not PM-only).
# 4 slots: [RVPM, RVPM, RVPM, RV_ONLY(c)]. Truth (Becker+2015/Bryant+2022):
#   e P=0.789593 K=4.55 (transit+RV)   b P=4.1591 K=140.84
#   d P=9.0307  K=4.96                 c P=588.4  K=31.6 (RV only)
# Run with `julia -t auto`. Env: NT NW NS NB NR NTRY, TD_DEBUG_BIRTHS=1.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, quantile
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const LIT = [(name="e",P=0.789593),(name="b",P=4.1591287),(name="d",P=9.030672),(name="c",P=588.4)]

# K2 photometry (flattened — same standard detrend as the PM-only loop)
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
pt_t=lc.t; pt_f=lc.flux./runmed(lc.flux,151); pt_e=lc.flux_err; pt_i=ones(Int,length(pt_t))

# RVs (combined, flag-filtered like the menu)
rr=readdlm(joinpath(DATADIR,"WASP47_RVs_combined.csv"),',',Any,'\n';header=true)
raw=rr[1]; hdr=vec(rr[2]); c(n)=findfirst(==(n),hdr); fcol=c("flag"); keep=trues(size(raw,1))
for i in 1:size(raw,1); fl=raw[i,fcol]; fl===missing&&continue
    String(strip(string(fl))) in ("transit","transit_night","anomalous")&&(keep[i]=false); end
raw=raw[keep,:]; bjd=Float64.(raw[:,c("bjd")]); rv=Float64.(raw[:,c("rv")]); rve=Float64.(raw[:,c("rv_err")])
istr=String.(raw[:,c("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
rinst=[i2i[s] for s in istr]
@printf("K2: %d cadences  |  RV: %d points, %d instruments\n", length(pt_t), length(bjd), length(inames))

data=Data(;t_rv=bjd,rv=rv,rv_err=rve,rv_inst=rinst,t_phot=pt_t,flux=pt_f,flux_err=pt_e,phot_inst=pt_i)
ic=InstrumentConfig(rv=inames,pm=["K2"]); pri=Dict{String,PriorSpec}()
for k in 1:3
    pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["K_k$k"]=UniformPrior(0.0,200.0)
    pri["sesinw_k$k"]=UniformPrior(-0.7,0.7); pri["secosw_k$k"]=UniformPrior(-0.7,0.7)
    pri["Tc_k$k"]=UniformPrior(pt_t[1],pt_t[1]+30.0); pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
end
pri["P_k4"]=LogUniformPrior(100.0,1500.0); pri["K_k4"]=UniformPrior(0.0,200.0)
pri["sesinw_k4"]=UniformPrior(-0.7,0.7); pri["secosw_k4"]=UniformPrior(-0.7,0.7)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue; ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,15.0); end
pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0)
pri["rho_s"]=NormalPrior(0.71,0.3,0.1,5.0)
params=Params(;max_kplanet=4,planet_modes=[RVPM,RVPM,RVPM,RV_ONLY],instruments=ic,data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,trend_order=0,
    stability=:none,external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])
target=NereusTarget(params,data;unconstrained=true)
td=TransDimConfig(max_kplanet=4,planets=true,noise=false,
    birth_strategies=[PriorBirth(),InformedBirth()],birth_weights=[0.3,0.7],transdim_fraction=0.3)

ev(k,d)=parse(Int,get(ENV,k,string(d)))
NT=ev("NT",10); NW=ev("NW",100); NS=ev("NS",3500); NB=ev("NB",2000); NR=ev("NR",12); NTRY=ev("NTRY",12)
@printf("JOINT K2+RV trans-dim: %d temps × %d walkers × %d+%d  NR=%d NTRY=%d\n", NT,NW,NS,NB,NR,NTRY)
t0=time()
res=sample_transdim_ptemcee(target,data;td=td,n_temps=NT,n_walkers=NW,n_steps=NS,n_burnin=NB,
    informed_birth_fraction=0.7,n_birth_refine=NR,n_birth_tries=NTRY,seed=42,show_progress=true)
secs=time()-t0

probs=model_probabilities(res.chains;max_kplanet=4)
np_post=[get(probs,k,0.0) for k in 0:4]; modal=argmax(np_post)-1
@printf("\n== %.0fs  modal Np=%d  logZ=%.1f ==\n", secs, modal, res.log_evidence)
@printf("Np posterior: %s\n", join([@sprintf("%d:%.2f",k,np_post[k+1]) for k in 0:4],"  "))
nv=vec(Array(res.chains[:,:n_planets,:])); mask=nv.==modal
found=String[]; recs=String[]
if modal>=1 && any(mask)
    cn=names(res.chains); hasb=Symbol("planet_active_1") in cn
    Pc=[vec(Array(res.chains[:,Symbol("P_k$k"),:]))[mask] for k in 1:4]
    Kc=[vec(Array(res.chains[:,Symbol("K_k$k"),:]))[mask] for k in 1:4]
    Ac= hasb ? [vec(Array(res.chains[:,Symbol("planet_active_$k"),:]))[mask] for k in 1:4] : nothing
    Ps=[Float64[] for _ in 1:modal]; Ks=[Float64[] for _ in 1:modal]
    for s in 1:count(mask)
        act = hasb ? [k for k in 1:4 if Ac[k][s]>0.5] : collect(1:4)
        length(act)>=modal || continue
        Pv=[Pc[k][s] for k in act]; Kv=[Kc[k][s] for k in act]; o=sortperm(Pv)
        for j in 1:modal; push!(Ps[j],Pv[o[j]]); push!(Ks[j],Kv[o[j]]); end
    end
    for j in 1:modal
        isempty(Ps[j]) && continue
        Pm=median(Ps[j]); Km=median(Ks[j]); push!(recs,@sprintf("P=%.4f/K=%.1f",Pm,Km))
        for l in LIT; abs(Pm-l.P)/l.P<0.02 && !(l.name in found) && push!(found,l.name); end
    end
end
@printf("recovered: %s\n", join(recs,"  "))
@printf("FOUND: [%s] %d/4\n", join(found,","), length(found))

# autopsy on e/d/c (if TD_DEBUG_BIRTHS=1)
if !isempty(Nereus._TD_DEBUG_EVENTS)
    evts=Nereus._TD_DEBUG_EVENTS
    for (nm,Pl) in [("e",0.789593),("d",9.030672),("c",588.4)]
        sel=[x for x in evts if abs(x[3]-Pl)/Pl<0.01]
        isempty(sel) && (println("$nm: no events"); continue)
        cold(cc)=count(x->x[4]==cc && x[2]<=2, sel)
        dll=[x[5] for x in sel if x[4]==4 && x[2]<=2 && isfinite(x[5])]
        qs = isempty(dll) ? (NaN,NaN) : (quantile(dll,0.9), maximum(dll))
        @printf("%s: cold sel=%d born=%d died=%d | cand n=%d q90=%.0f max=%.0f\n",
                nm, cold(3), cold(1), cold(2), length(dll), qs[1], qs[2])
    end
end

# ---- plots + chains ----
OUTP = joinpath(@__DIR__, "plots_WASP47_transdim"); mkpath(OUTP)
try save_chains(joinpath(OUTP,"chains.nc"), res.chains, params; data=data) catch e; println("save_chains: ",sprint(showerror,e)) end
for k in 1:4
    try plot_pm_phasefold(res.chains, params, data; planet=k, output=OUTP)
    catch e; println("pm_phasefold K$k: ", sprint(showerror,e)) end
    try plot_rv_phasefold(res.chains, params, data; planet=k, output=OUTP)
    catch e; println("rv_phasefold K$k: ", sprint(showerror,e)) end
end
println("plots in $OUTP")
try plot_posteriors_lp(res.chains, params; output=OUTP)
catch e; println("posteriors_lp: ", sprint(showerror,e)) end
