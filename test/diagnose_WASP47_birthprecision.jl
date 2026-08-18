#!/usr/bin/env julia
# Proposal-level test of the refined informed-birth fix (no MCMC). Drives
# JointInformedBirth directly on the time-corrected WASP-47 photometry
# from a k=0 state with BLIND priors (P~LogU(0.5,30)), and measures the
# proposed-period precision. Before the fix: σ_P=0.05 (5%) smears every
# proposal ~1700x wider than b's ~1e-5 transit basin, so ~0 births align.
# After: BLS peak refined + tight basin-matched σ -> births should cluster
# transit-precise on b (and its real harmonics), enabling acceptance.

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
pri["P_k1"]=LogUniformPrior(0.5,30.0); pri["K_k1"]=UniformPrior(0.0,250.0)     # BLIND
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

# k=0 Theta
td=TransDimState(max_planets=params.config.max_kplanet)
th=Theta(params; td=td)
blk=params.layout.planet_blocks[1]

rng=MersenneTwister(123)
N=3000; Ps=Float64[]; Tcs=Float64[]
for _ in 1:N
    nt, lq = Nereus.propose_planet_birth(th, rng, Nereus.JointInformedBirth(); data=data)
    isfinite(lq) || continue
    push!(Ps, nt.values[blk.P]); push!(Tcs, nt.values[blk.t])
end
@printf("\n%d births sampled\n", length(Ps))

near(P,frac)=count(p->abs(p-P)/P<frac, Ps)
@printf("proposed-period precision vs b (4.1591287):\n")
for fr in (1e-5,3e-5,1e-4,3e-4,1e-3,1e-2)
    @printf("  within %7.0e : %5.1f%%  (n=%d)\n", fr, 100*near(P_B,fr)/length(Ps), near(P_B,fr))
end
@printf("\nclustering around real harmonics/aliases (within 0.1%%):\n")
for (nm,P0) in (("b",P_B),("2×b",2P_B),("b/2",P_B/2),("d?",9.0307),("e?",0.789593))
    @printf("  %-4s P=%8.4f : %5.1f%%\n", nm, P0, 100*near(P0,1e-3)/length(Ps))
end
# Tc precision for the births that hit b's period
bmask=[abs(p-P_B)/P_B<1e-4 for p in Ps]
if any(bmask)
    tcb=Tcs[bmask]; phase=[mod(t-2456978.81, P_B)/P_B for t in tcb]
    @printf("\nfor on-b births: Tc phase spread (mod P, ref real transit) min=%.4f max=%.4f\n",
            minimum(phase), maximum(phase))
end
println("\n>> goal: a meaningful %% within ~1e-4 of b (was ~0 with σ_P=0.05)")
