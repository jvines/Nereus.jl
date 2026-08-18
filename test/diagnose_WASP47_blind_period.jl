#!/usr/bin/env julia
# WASP-47 period-search diagnostic. The γ test proved the RV data contain
# b's K=140 cleanly WHEN the period is known. So the menu's K~56 failure
# is a PERIOD-SEARCH / alias problem, not RV attribution. This isolates
# it: RV-ONLY, 1 planet, BLIND broad period prior (the menu's
# LogUniform(0.5,30)) — does the RV alone land on b (4.159d) or get
# trapped on an alias?  Dumps the top period-posterior modes so we can see
# which aliases compete with b.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, quantile
using Printf
using MCMCChains

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const B = (P=4.1591287, K=140.84)

rr=readdlm(joinpath(DATADIR,"WASP47_RVs_combined.csv"),',',Any,'\n';header=true)
raw=rr[1]; hdr=vec(rr[2]); c(n)=findfirst(==(n),hdr); fcol=c("flag"); keep=trues(size(raw,1))
for i in 1:size(raw,1); f=raw[i,fcol]; f===missing&&continue
    String(strip(string(f))) in ("transit","transit_night","anomalous")&&(keep[i]=false); end
raw=raw[keep,:]; bjd=Float64.(raw[:,c("bjd")]); rv=Float64.(raw[:,c("rv")]); rve=Float64.(raw[:,c("rv_err")])
istr=String.(raw[:,c("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
rinst=[i2i[s] for s in istr]
data=Data(;t_rv=bjd,rv=rv,rv_err=rve,rv_inst=rinst)
ic=InstrumentConfig(rv=inames)

pri=Dict{String,PriorSpec}()
pri["n_p"]=FixedPrior(1.0)
pri["P_k1"]=LogUniformPrior(0.5,30.0)            # BLIND — the menu's prior
pri["K_k1"]=UniformPrior(0.0,250.0)
pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
pri["Mo_k1"]=UniformPrior(-2π,2π)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
    ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end

params=Params(;max_kplanet=1,planet_modes=[RV_ONLY],instruments=ic,data=data,
    M_s=1.04,R_s=1.137,priors=pri,trend_order=0,stability=:none,
    parametrization=ParametrizationConfig(time=:Mo),
    external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])
target=NereusTarget(params,data;unconstrained=true)

println("== WASP-47 b BLIND RV-only period search (P~LogU(0.5,30)) ==")
t0=time()
res=sample_ptemcee(target,data;n_temps=12,n_walkers=120,n_steps=6000,
                   n_burnin=3000,seed=42,show_progress=false)
secs=time()-t0
ch=res.chains
Pc=vec(Array(ch[:,:P_k1,:])); Kc=vec(Array(ch[:,:K_k1,:]))
@printf("\n%.0fs   P_med=%.4f d   K_med=%.1f m/s   (truth P=%.4f K=%.1f)\n",
        secs, median(Pc), median(Kc), B.P, B.K)

# Top period modes: histogram in log-P, report peaks with their K.
lp=log10.(Pc); edges=range(log10(0.5),log10(30.0);length=121)
hcnt=zeros(Int,length(edges)-1)
for x in lp; b=searchsortedlast(edges,x); (1<=b<=length(hcnt))&&(hcnt[b]+=1); end
order=sortperm(hcnt;rev=true)
println("\nTop period-posterior modes (log-P histogram peaks):")
shown=0
for bi in order
    shown>=6 && break
    hcnt[bi]==0 && continue
    Plo=10^edges[bi]; Phi=10^edges[bi+1]; Pmid=sqrt(Plo*Phi)
    msk=(Pc.>=Plo).&(Pc.<Phi); frac=count(msk)/length(Pc)
    frac<0.01 && continue
    isb=abs(Pmid-B.P)/B.P<0.03 ? "  <== b" : ""
    @printf("  P~%7.3f d  (%.0f%%)  K_med=%6.1f m/s%s\n", Pmid, 100frac, median(Kc[msk]), isb)
    global shown+=1
end
onb=count(p->abs(p-B.P)/B.P<0.03, Pc)/length(Pc)
@printf("\nposterior mass within 3%% of b's period: %.0f%%\n", 100onb)
println(onb>0.5 ? ">> RV alone FINDS b — blind period search is fine; failure is joint/trans-dim machinery." :
        ">> RV alone TRAPPED on alias(es) — photometry must break the degeneracy in the joint fit.")
