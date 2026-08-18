#!/usr/bin/env julia
# WASP-47 joint-landscape diagnostic. RV likelihood, RV blind search, and
# the BLS seed ALL find b correctly — so the failure is the trans-dim
# birth/death navigation (sits at Np=0, or births onto the 2×b=8.32d
# harmonic the BLS also returns). This localizes likelihood-vs-sampler:
#
#  (A) Fixed-dim 1-planet JOINT (RVPM) fit, period seeded near b. If the
#      joint likelihood recovers K=140 + an aligned transit, the joint
#      likelihood is FINE and the whole problem is trans-dim navigation.
#  (B) Direct logL at the recovered b solution vs the SAME solution with
#      P doubled to 8.32d (2×b) vs the planet switched off. Confirms the
#      likelihood correctly penalizes the harmonic + strongly favors b
#      over Np=0 — i.e. the landscape the sampler SHOULD be climbing.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, quantile
using Printf
using MCMCChains

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const B = (P=4.1591287, K=140.84)

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
pri["n_p"]=FixedPrior(1.0)
pri["P_k1"]=UniformPrior(4.10,4.22)                 # seed b (tight)
pri["K_k1"]=UniformPrior(0.0,250.0)
pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
pri["Tc_k1"]=UniformPrior(pt_t[1],pt_t[1]+B.P)      # one period — find the phase
pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
    ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
for n in pm_names
    pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
pri["rho_s"]=NormalPrior(1.0,0.3,0.1,5.0)

params=Params(;max_kplanet=1,planet_modes=[RVPM],instruments=ic,data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,
    trend_order=0,stability=:none,external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])
target=NereusTarget(params,data;unconstrained=true)

println("== (A) WASP-47 b fixed-dim JOINT (RVPM) fit, period seeded ==")
t0=time()
res=sample_ptemcee(target,data;n_temps=10,n_walkers=120,n_steps=4000,
                   n_burnin=2000,seed=42,show_progress=false)
secs=time()-t0
ch=res.chains
gv(s)=vec(Array(ch[:,s,:]))
Pm=median(gv(:P_k1)); Kq=quantile(gv(:K_k1),[0.16,0.5,0.84])
rrm=median(gv(:rr_k1)); bm=median(gv(:b_k1)); Tcm=median(gv(:Tc_k1))
@printf("\n%.0fs   P=%.4f d   K=%.1f [%.1f,%.1f] m/s   rr=%.4f  b=%.3f  Tc=%.4f\n",
        secs, Pm, Kq[2],Kq[1],Kq[3], rrm, bm, Tcm)
@printf("   (truth: P=%.4f  K=%.1f)\n", B.P, B.K)

# ---- (B) logL landscape: b  vs  2×b  vs  planet OFF -------------------
# Build a Theta at the posterior-median solution, then evaluate logL with
# P=P_b, P=2·P_b (harmonic), and the planet's amplitude zeroed (Np=0 proxy).
θ = Theta(params)
set_param!(θ,"P_k1",Pm); set_param!(θ,"K_k1",Kq[2])
set_param!(θ,"sesinw_k1",median(gv(:sesinw_k1))); set_param!(θ,"secosw_k1",median(gv(:secosw_k1)))
set_param!(θ,"Tc_k1",Tcm); set_param!(θ,"b_k1",bm); set_param!(θ,"rr_k1",rrm)
set_param!(θ,"rho_s",median(gv(:rho_s)))
for n in pm_names
    set_param!(θ,"offset_$n",median(gv(Symbol("offset_$n")))); set_param!(θ,"jitter_$n",median(gv(Symbol("jitter_$n"))))
    set_param!(θ,"q1_$n",median(gv(Symbol("q1_$n")))); set_param!(θ,"q2_$n",median(gv(Symbol("q2_$n")))); end
for n in inames
    set_param!(θ,"gamma_$n",median(gv(Symbol("gamma_$n")))); set_param!(θ,"sigma_$n",median(gv(Symbol("sigma_$n")))); end

Lb_rv  = rv_log_likelihood(θ,data);  Lb_ph  = transit_log_likelihood(θ,data)
set_param!(θ,"P_k1",2*Pm)            # 2×b harmonic
L2_rv  = rv_log_likelihood(θ,data);  L2_ph  = transit_log_likelihood(θ,data)
set_param!(θ,"P_k1",Pm); set_param!(θ,"rr_k1",1e-4); set_param!(θ,"K_k1",0.0)  # planet ~OFF
L0_rv  = rv_log_likelihood(θ,data);  L0_ph  = transit_log_likelihood(θ,data)

println("\n== (B) logL landscape (RV + photometry) ==")
lab_b = @sprintf("b (P=%.3f)", Pm); lab_2 = @sprintf("2×b (P=%.3f)", 2Pm)
@printf("  %-14s  logL_rv=%12.1f   logL_phot=%14.1f   total=%14.1f\n", lab_b, Lb_rv, Lb_ph, Lb_rv+Lb_ph)
@printf("  %-14s  logL_rv=%12.1f   logL_phot=%14.1f   total=%14.1f\n", lab_2, L2_rv, L2_ph, L2_rv+L2_ph)
@printf("  %-14s  logL_rv=%12.1f   logL_phot=%14.1f   total=%14.1f\n", "planet OFF", L0_rv, L0_ph, L0_rv+L0_ph)
@printf("\n  Δlog L(b − 2×b)   = %+.1f   (should be >>0: harmonic must lose)\n", (Lb_rv+Lb_ph)-(L2_rv+L2_ph))
@printf("  Δlog L(b − OFF)   = %+.1f   (should be >>0: dropping b is absurd)\n", (Lb_rv+Lb_ph)-(L0_rv+L0_ph))
