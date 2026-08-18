#!/usr/bin/env julia
# JOINT K2 + HARPS blind trans-dim recovery on K2-138 — second real-target
# system after WASP-47. Six planets in a near-3:2 resonant chain
# (Christiansen+2018, Lopez+2019): adjacent period ratios 1.513-1.544 sit
# INSIDE the 3% harmonic-dedup tolerance of 3/2, so this deliberately
# stresses the family/harmonic dedup machinery; there is also NO dominant
# hot Jupiter (all depths 260-1300 ppm) — the opposite SNR regime from
# WASP-47. Truth (Lopez+2019): P = 2.35322, 3.55987, 5.40478, 8.26144,
# 12.7576, 41.966 d; K = 1.61, 2.62, 2.80, 2.61, 1.30, ~0 (g undetected
# in RV). M*=0.93, R*=0.86 (rho*≈1.46 rho_sun).
# Run with `julia -t auto`. Env: NT NW NS NB NR NTRY, TD_DEBUG_BIRTHS=1.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, quantile
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "K2138"))
const LIT = [(name="b",P=2.35322),(name="c",P=3.55987),(name="d",P=5.40478),
             (name="e",P=8.26144),(name="f",P=12.7576),(name="g",P=41.966)]

lc = load_tess_lc(joinpath(DATADIR, "K2-138_k2_c12_everest_lc.csv"))
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
pt_t=lc.t; pt_f=lc.flux./runmed(lc.flux,151); pt_e=lc.flux_err; pt_i=ones(Int,length(pt_t))

rr=readdlm(joinpath(DATADIR,"K2138_RVs_combined.csv"),',',Any,'\n';header=true)
raw=rr[1]; hdr=vec(rr[2]); cl(n)=findfirst(==(n),hdr)
bjd=Float64.(raw[:,cl("bjd")]); rv=Float64.(raw[:,cl("rv")]); rve=Float64.(raw[:,cl("rv_err")])
rinst=ones(Int,length(bjd))
@printf("K2 C12: %d cadences (%.1f d)  |  HARPS: %d RVs (%.0f d span)\n",
        length(pt_t), pt_t[end]-pt_t[1], length(bjd), maximum(bjd)-minimum(bjd))

data=Data(;t_rv=bjd,rv=rv,rv_err=rve,rv_inst=rinst,t_phot=pt_t,flux=pt_f,flux_err=pt_e,phot_inst=pt_i)
ic=InstrumentConfig(rv=["HARPS"],pm=["K2"]); pri=Dict{String,PriorSpec}()
for k in 1:8
    pri["P_k$k"]=LogUniformPrior(1.5,60.0); pri["K_k$k"]=UniformPrior(0.0,20.0)
    pri["sesinw_k$k"]=UniformPrior(-0.7,0.7); pri["secosw_k$k"]=UniformPrior(-0.7,0.7)
    pri["Tc_k$k"]=UniformPrior(pt_t[1],pt_t[1]+60.0); pri["b_k$k"]=UniformPrior(0.0,2.0)   # b>1+rr = non-transiting allowed: junk RV signals must not be forced to fake transits
    pri["rr_k$k"]=UniformPrior(0.005,0.10)
end
ce=median(rv); sp=max(maximum(rv)-minimum(rv),1.0)
pri["gamma_HARPS"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_HARPS"]=ModJeffreysPrior(0.1,15.0)
pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0)
pri["rho_s"]=NormalPrior(1.46,0.4,0.2,5.0)
# GP RV activity (always-on): the white-noise run absorbed the stellar
# rotation as Keplerians at 25.3 d (P_rot) and 52.6 d (harmonic). Two-SHO
# rotation kernel anchored on P_rot ≈ 25 d; with rotation in the noise the
# modal planet COUNT should drop toward the honest 6 and b gets room.
rv_gp = CeleriteRotation(channel = :rv)
pri["gp_sigma"]  = LogUniformPrior(0.1, 20.0)
pri["gp_period"] = NormalPrior(25.0, 3.0, 15.0, 40.0)
# Q0 must reach COHERENT regimes: the HARPS span is ~14 rotations, and a
# phase-stable spot signal needs Q ≳ πN ≈ 40 to stay coherent that long.
# The HD18599-copied LogU(1,10) capped the kernel at ~3 coherent rotations,
# so the GP structurally COULD NOT absorb the coherent core of the rotation
# signal — a Keplerian at P_rot was then the genuinely-preferred description.
pri["gp_Q0"]     = LogUniformPrior(1.0, 300.0)
pri["gp_dQ"]     = UniformPrior(0.0, 5.0)
pri["gp_f"]      = UniformPrior(0.05, 0.5)
params=Params(;max_kplanet=8,planet_modes=[RVPM,RVPM,RVPM,RVPM,RVPM,RVPM,RVPM,RVPM],instruments=ic,data=data,
    M_s=0.93,R_s=0.86,parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),
    priors=pri,trend_order=0,stability=:none,noise_models=[rv_gp],transdim_noise=false,
    external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])
target=NereusTarget(params,data;unconstrained=true)
td=TransDimConfig(max_kplanet=8,planets=true,noise=false,
    birth_strategies=[PriorBirth(),InformedBirth()],birth_weights=[0.3,0.7],transdim_fraction=0.3)

ev(k,d)=parse(Int,get(ENV,k,string(d)))
NT=ev("NT",10); NW=ev("NW",100); NS=ev("NS",12000); NB=ev("NB",9000); NR=ev("NR",12); NTRY=ev("NTRY",12)
@printf("K2-138 trans-dim: %d temps × %d walkers × %d+%d  NR=%d NTRY=%d\n", NT,NW,NS,NB,NR,NTRY)
t0=time()
res=sample_transdim_ptemcee(target,data;td=td,n_temps=NT,n_walkers=NW,n_steps=NS,n_burnin=NB,
    informed_birth_fraction=0.7,n_birth_refine=NR,n_birth_tries=NTRY,seed=42,show_progress=true)
secs=time()-t0

probs=model_probabilities(res.chains;max_kplanet=8)
np_post=[get(probs,k,0.0) for k in 0:8]; modal=argmax(np_post)-1
@printf("\n== %.0fs  modal Np=%d  logZ=%.1f ==\n", secs, modal, res.log_evidence)
@printf("Np posterior: %s\n", join([@sprintf("%d:%.2f",k,np_post[k+1]) for k in 0:8],"  "))
nv=vec(Array(res.chains[:,:n_planets,:])); mask=nv.==modal
found=String[]; recs=String[]
if modal>=1 && any(mask)
    cn=names(res.chains); hasb=Symbol("planet_active_1") in cn
    Pc=[vec(Array(res.chains[:,Symbol("P_k$k"),:]))[mask] for k in 1:8]
    Kc=[vec(Array(res.chains[:,Symbol("K_k$k"),:]))[mask] for k in 1:8]
    Ac= hasb ? [vec(Array(res.chains[:,Symbol("planet_active_$k"),:]))[mask] for k in 1:8] : nothing
    Ps=[Float64[] for _ in 1:modal]; Ks=[Float64[] for _ in 1:modal]
    for s in 1:count(mask)
        act = hasb ? [k for k in 1:8 if Ac[k][s]>0.5] : collect(1:6)
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
@printf("FOUND: [%s] %d/6 (8 slots)\n", join(found,","), length(found))

if !isempty(Nereus._TD_DEBUG_EVENTS)
    evts=Nereus._TD_DEBUG_EVENTS
    INTERLOPERS = [(name="P_rot?",P=26.0),(name="36d?",P=36.3),(name="2P_rot?",P=52.6)]
    for l in vcat(collect(LIT), INTERLOPERS)
        sel=[x for x in evts if abs(x[3]-l.P)/l.P<0.03]
        isempty(sel) && (println("$(l.name): no events"); continue)
        cold(cc)=count(x->x[4]==cc && x[2]<=2, sel)
        dll=[x[5] for x in sel if x[4]==4 && x[2]<=2 && isfinite(x[5])]
        qs = isempty(dll) ? (NaN,NaN) : (quantile(dll,0.9), maximum(dll))
        @printf("%s (P=%.3f): cold sel=%d born=%d died=%d | cand n=%d q90=%.0f max=%.0f\n",
                l.name, l.P, cold(3), cold(1), cold(2), length(dll), qs[1], qs[2])
    end
end

# GP hyperparameter posteriors — did the activity model actually engage?
for g in ("gp_sigma","gp_period","gp_Q0","gp_dQ","gp_f")
    sym=Symbol(g)
    if sym in names(res.chains)
        v=vec(Array(res.chains[:,sym,:]))
        @printf("%-10s med=%.3f  q16=%.3f q84=%.3f\n", g, median(v), quantile(v,0.16), quantile(v,0.84))
    end
end

# ---- plots + chains ----
OUTP = joinpath(@__DIR__, "plots_K2138_transdim"); mkpath(OUTP)
try save_chains(joinpath(OUTP,"chains.nc"), res.chains, params; data=data) catch e; println("save_chains: ",sprint(showerror,e)) end
for k in 1:8
    try plot_pm_phasefold(res.chains, params, data; planet=k, output=OUTP)
    catch e; println("pm_phasefold K$k: ", sprint(showerror,e)) end
    try plot_rv_phasefold(res.chains, params, data; planet=k, output=OUTP)
    catch e; println("rv_phasefold K$k: ", sprint(showerror,e)) end
end
println("plots in $OUTP")
try plot_posteriors_lp(res.chains, params; output=OUTP)
catch e; println("posteriors_lp: ", sprint(showerror,e)) end
