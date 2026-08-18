#!/usr/bin/env julia
# Fast dev loop for the trans-dim birth fix: PM-only K2 (where d/e live).
# Blind trans-dim — can it recover b/d/e (k=3)? Truth: b=4.159, d=9.031, e=0.790.
# Run with `julia -t auto`.

using Nereus
using Statistics: median, std
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
flat=f./runmed(f,151)

data=Data(; t_phot=t, flux=flat, flux_err=e, phot_inst=ones(Int,length(t)))
pri=Dict{String,PriorSpec}()
for k in 1:3
    pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["Tc_k$k"]=UniformPrior(t[1],t[1]+30.0)
    pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
end
pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0)
pri["rho_s"]=NormalPrior(0.71,0.3,0.1,5.0)
params=Params(;max_kplanet=3,planet_modes=[PM_ONLY,PM_ONLY,PM_ONLY],instruments=InstrumentConfig(pm=["K2"]),
    data=data,M_s=1.04,R_s=1.137,parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),
    priors=pri,stability=:none)
target=NereusTarget(params,data;unconstrained=true)
td=TransDimConfig(max_kplanet=3,planets=true,noise=false,
    birth_strategies=[PriorBirth(),InformedBirth()],birth_weights=[0.3,0.7],transdim_fraction=0.3)

ev(k,d)=parse(Int,get(ENV,k,string(d)))
NT=ev("NT",8); NW=ev("NW",80); NS=ev("NS",2500); NB=ev("NB",1200); NR=ev("NR",8); NTRY=ev("NTRY",12)
@printf("PM-only K2 trans-dim: %d temps × %d walkers × %d+%d steps  n_birth_refine=%d n_birth_tries=%d\n", NT,NW,NS,NB,NR,NTRY)
t0=time()
res=sample_transdim_ptemcee(target,data;td=td,n_temps=NT,n_walkers=NW,n_steps=NS,n_burnin=NB,
    informed_birth_fraction=0.7,n_birth_refine=NR,n_birth_tries=NTRY,seed=42,show_progress=true)
secs=time()-t0

probs=model_probabilities(res.chains;max_kplanet=3)
np_post=[get(probs,k,0.0) for k in 0:3]; modal=argmax(np_post)-1
@printf("\n== %.0fs  modal Np=%d ==\n", secs, modal)
@printf("Np posterior: %s\n", join([@sprintf("%d:%.2f",k,np_post[k+1]) for k in 0:3],"  "))
# recovered periods at modal Np (active slots)
nv=vec(Array(res.chains[:,:n_planets,:])); mask=nv.==modal
LIT=[("e",0.789593),("b",4.1591287),("d",9.030672)]
found=String[]
if modal>=1 && any(mask)
    cn=names(res.chains); hasb=Symbol("planet_active_1") in cn
    Pc=[vec(Array(res.chains[:,Symbol("P_k$k"),:]))[mask] for k in 1:3]
    Ac= hasb ? [vec(Array(res.chains[:,Symbol("planet_active_$k"),:]))[mask] for k in 1:3] : nothing
    Ps=[Float64[] for _ in 1:modal]
    for s in 1:count(mask)
        act = hasb ? [k for k in 1:3 if Ac[k][s]>0.5] : collect(1:3)
        length(act)>=modal || continue
        Pv=[Pc[k][s] for k in act]; o=sortperm(Pv)
        for j in 1:modal; push!(Ps[j],Pv[o[j]]); end
    end
    rec=[median(p) for p in Ps if !isempty(p)]
    for (nm,Pl) in LIT, Pm in rec; abs(Pm-Pl)/Pl<0.02 && !(nm in found) && push!(found,nm); end
    @printf("recovered periods: %s\n", join([@sprintf("%.4f",p) for p in rec]," "))
end
@printf("FOUND: %s  (%d/3)  %s\n", join(found,","), length(found),
        all(x->x in found,["b","d","e"]) ? "ALL THREE ✓" : "missing "*join(setdiff(["b","d","e"],found),","))

# Birth/death autopsy (TD_DEBUG_BIRTHS=1): is e never-born or born-and-dies?
if !isempty(Nereus._TD_DEBUG_EVENTS)
    evts=Nereus._TD_DEBUG_EVENTS
    using Statistics: quantile
    for (nm,Pl) in [("e",0.789593),("d",9.030672)]
        sel=[x for x in evts if abs(x[3]-Pl)/Pl<0.01]
        sl(c,burn)=count(x->x[4]==c && (burn ? x[1]<=NB : x[1]>NB), sel)
        cold(c)=count(x->x[4]==c && x[2]<=2, sel)
        @printf("%s events: select=%d  births(burn)=%d births(post)=%d  deaths(burn)=%d deaths(post)=%d  | cold(t<=2): sel=%d born=%d died=%d\n",
                nm, count(x->x[4]==3,sel), sl(1,true), sl(1,false), sl(2,true), sl(2,false),
                cold(3), cold(1), cold(2))
        # candidate-quality distribution at cold temps (code 4 = every multi-try candidate)
        dll=[x[5] for x in sel if x[4]==4 && x[2]<=2 && isfinite(x[5])]
        if !isempty(dll)
            q=quantile(dll,[0.1,0.5,0.9])
            @printf("   cold %s-candidate ΔlogL: n=%d  q10=%.0f med=%.0f q90=%.0f  max=%.0f  frac>+50=%.4f\n",
                    nm, length(dll), q[1], q[2], q[3], maximum(dll), count(>(50.0),dll)/length(dll))
        end
    end
    nb_all=count(x->x[4]==1, evts); nd_all=count(x->x[4]==2, evts)
    @printf("ALL periods: births=%d deaths=%d selections=%d candidates=%d\n",
            nb_all, nd_all, count(x->x[4]==3,evts), count(x->x[4]==4,evts))
    # candidate anatomy at e's period: rr (code 5) and Tc phase offset (code 6)
    Pe=0.789593; Tce_true=2456977.4029   # empirical e Tc from the detrend work
    rrs=[x[5] for x in evts if x[4]==5 && abs(x[3]-Pe)/Pe<0.01]
    tcs=[x[5] for x in evts if x[4]==6 && abs(x[3]-Pe)/Pe<0.01]
    if !isempty(rrs)
        q=quantile(rrs,[0.1,0.5,0.9])
        @printf("e-candidate rr: n=%d q10=%.4f med=%.4f q90=%.4f  (true 0.0122)\n", length(rrs), q...)
    end
    if !isempty(tcs)
        dph=[abs(mod(tc-Tce_true+Pe/2, Pe)-Pe/2)/Pe for tc in tcs]
        q=quantile(dph,[0.1,0.5,0.9])
        @printf("e-candidate |Tc offset|/P: n=%d q10=%.3f med=%.3f q90=%.3f  (in-transit < ~0.035)\n", length(tcs), q...)
    end
    # live BLS cache contents: what do the e-entries actually carry?
    @printf("live BLS caches: %d entries\n", length(Nereus._BLS_CACHES))
    let shown=0
        for (k,c) in Nereus._BLS_CACHES
            idx=findall(p->abs(p-Pe)/Pe<0.01, c.periods)
            isempty(idx) && continue
            for i in idx
                @printf("  cache %s: e-peak P=%.6f w=%.3f depth=%.0fppm t0=%.4f dur=%.4f\n",
                        string(k,base=16)[1:6], c.periods[i], c.weights[i], c.depths[i]*1e6, c.t0s[i], c.durs[i])
            end
            shown+=1; shown>=10 && break
        end
    end
end

# Is d/e even FOUND in the Np>=2 samples (found-but-losing vs not-found)?
cn=names(res.chains); hasb=Symbol("planet_active_1") in cn
nv2=vec(Array(res.chains[:,:n_planets,:]))
Pall=[vec(Array(res.chains[:,Symbol("P_k$k"),:])) for k in 1:3]
Aall= hasb ? [vec(Array(res.chains[:,Symbol("planet_active_$k"),:])) for k in 1:3] : nothing
actP=Float64[]
for s in 1:length(nv2)
    nv2[s]>=2 || continue
    for k in 1:3; (hasb ? Aall[k][s]>0.5 : true) && push!(actP, Pall[k][s]); end
end
println("active-slot periods among Np>=2 samples (is d/e found, just losing?):")
for (nm,Pl) in [("e",0.789593),("b",4.1591287),("d",9.030672)]
    fr = isempty(actP) ? 0.0 : count(p->abs(p-Pl)/Pl<0.02, actP)/length(actP)
    @printf("   %s P=%.3f : %.1f%%\n", nm, Pl, 100*fr)
end
