#!/usr/bin/env julia
# HD 18599 b — FIXED CeleriteRotation RVPM (reproduce Desidera; isolate whether
# the trans-dim MENU, not the rotation prior, is what inflates K). No menu, no
# trans-dim, no ErrorScale/AGP/floor — just the anchored rotation GP.
using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf
using LinearAlgebra: BLAS; BLAS.set_num_threads(1)
const REPO=abspath(joinpath(@__DIR__,"..","..")); const RVF=joinpath(@__DIR__,"data","hd18599.csv")
const LCF=joinpath(REPO,"data","HD18599","HD18599_cleaned_lc.csv")
const P_REF,T0_REF,DUR,M_S,R_S,P_ROT=4.1374685534602405,2458354.5857470357,0.067,0.807,0.798,8.74
const PAPER=Set(["HARPS_PRE","HARPS_POST","FEROS"])
raw=readdlm(RVF,',',Any,'\n';header=true); dm=raw[1]; keep=Int[]; istr=String[]
for i in 1:size(dm,1); ins=strip(String(dm[i,16])); prov=strip(String(dm[i,17]))
  (ins=="HARPS_POST"&&prov=="ESO_PHASE3")&&continue; ins in PAPER||continue; push!(keep,i); push!(istr,ins); end
dm=dm[keep,:]; bjd=Float64.(dm[:,1]); rv=Float64.(dm[:,2]); rverr=Float64.(dm[:,3])
inm=sort!(unique(istr)); imap=Dict(n=>i for (i,n) in enumerate(inm)); rvi=[imap[s] for s in istr]
let kc=trues(length(bjd)); for id in unique(rvi); idx=findall(==(id),rvi); m=median(rv[idx]); thr=5*max(1.4826*median(abs.(rv[idx].-m)),1e-6)
  for k in idx; abs(rv[k]-m)>thr&&(kc[k]=false); end; end
  global bjd=bjd[kc];global rv=rv[kc];global rverr=rverr[kc];global rvi=rvi[kc]; end
lc=load_tess_lc(LCF); half=2*DUR; nlc=length(lc.t); nper=ceil(Int,(lc.t[end]-lc.t[1])/P_REF)+2
intr=falses(nlc); for k in -1:(nper+1); tc=T0_REF+k*P_REF; @inbounds for i in 1:nlc; abs(lc.t[i]-tc)<half&&(intr[i]=true); end; end
oot=falses(nlc); let c=0; for i in 1:nlc; intr[i]&&continue; c+=1; oot[i]=(c%200==0); end; end
kp=intr .| oot; t_phot=lc.t[kp]; flux=lc.flux[kp]; flux_err=lc.flux_err[kp]
@printf("RV %d / %d inst | LC %d in-transit\n", length(bjd), length(inm), count(intr))
data=Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rvi, t_phot=t_phot, flux=flux, flux_err=flux_err, phot_inst=ones(Int,length(t_phot)))
ic=InstrumentConfig(rv=inm, pm=["TESS"]); rvmax=maximum(abs,rv)
pri=Dict{String,PriorSpec}(
  "P_k1"=>UniformPrior(P_REF-0.01,P_REF+0.01), "K_k1"=>UniformPrior(0.0,50.0),
  "sesinw_k1"=>UniformPrior(-1.0,1.0), "secosw_k1"=>UniformPrior(-1.0,1.0),
  "Tc_k1"=>NormalPrior(T0_REF,0.05,T0_REF-0.5,T0_REF+0.5),
  "b_k1"=>UniformPrior(0.0,1.0), "rr_k1"=>NormalPrior(0.031,0.005,0.005,0.10),
  "rho_s"=>NormalPrior(2.241,0.479,0.1,10.0),
  "offset_TESS"=>NormalPrior(0.0,1e-3,-5e-3,5e-3), "jitter_TESS"=>LogUniformPrior(1e-5,5e-3),
  "q1_TESS"=>UniformPrior(0.0,1.0), "q2_TESS"=>UniformPrior(0.0,1.0),
  "gp_period"=>NormalPrior(P_ROT,1.0,4.0,16.0))
for n in inm; pri["gamma_$n"]=UniformPrior(-3rvmax,3rvmax); end
params=Params(; max_kplanet=1, planet_modes=[RVPM], instruments=ic, data=data, M_s=M_S, R_s=R_S,
  parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true),
  priors=pri, noise_models=NoiseModel[CeleriteRotation()])   # FIXED noise, no trans-dim
tgt=NereusTarget(params, data; unconstrained=false)
@printf("Free params: %d (fixed CeleriteRotation, RVPM)\n", n_unfrozen(params))
NT=parse(Int,get(ENV,"NT","10")); NW=parse(Int,get(ENV,"NW","100")); NS=parse(Int,get(ENV,"NS","4000")); NB=parse(Int,get(ENV,"NB","2000"))
t0=time()
res=sample_ptemcee(tgt, data; n_temps=NT, n_walkers=NW, n_steps=NS, n_burnin=NB, init_strategy=:map_scatter, seed=42)
K=vec(Array(res.chains[:K_k1])); gp=vec(Array(res.chains[:gp_period])); gs=vec(Array(res.chains[:gp_sigma]))
@printf("done %.1f min\nPLANET b: K=%.2f  16/50/84=%.2f/%.2f/%.2f  frac(8<K<14)=%.2f  | gp_period=%.2f gp_sigma=%.1f\n",
  (time()-t0)/60, median(K), quantile(K,0.16), quantile(K,0.5), quantile(K,0.84), mean(8 .<K.<14), median(gp), median(gs))
