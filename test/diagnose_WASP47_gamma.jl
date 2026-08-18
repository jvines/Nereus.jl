#!/usr/bin/env julia
# WASP-47 K-attribution diagnostic: does the broad multi-instrument γ
# fragment b's K=140 signal?  Fixed-dim, 1-planet (b), RV-ONLY, tight
# period prior so BOTH runs land squarely on b — then compare recovered K
# with marginalize_gamma OFF (6 free per-instrument γ) vs ON (γ
# analytically integrated, flat prior). RV-only strips photometry +
# trans-dim confounds; this isolates the γ effect on K alone.
#
#   K -> ~140 with marginalize_gamma  => failure was γ–K fragmentation
#   K stays ~56                       => γ exonerated; look elsewhere
#
# Also prints per-instrument surviving-point counts after the flag cut
# (the "is precise HIRES being thrown away by the transit-night cut?" Q).

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, quantile
using Printf
using MCMCChains

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const B = (P=4.1591287, K=140.84)   # Bryant+2022 b

# ---- load combined RV, same flag filter as the menu ------------------
rr=readdlm(joinpath(DATADIR,"WASP47_RVs_combined.csv"),',',Any,'\n';header=true)
raw=rr[1]; hdr=vec(rr[2]); c(n)=findfirst(==(n),hdr); fcol=c("flag"); keep=trues(size(raw,1))
for i in 1:size(raw,1); f=raw[i,fcol]; f===missing&&continue
    String(strip(string(f))) in ("transit","transit_night","anomalous")&&(keep[i]=false); end

# per-instrument survival report (before vs after the flag cut)
istr_all=String.(raw[:,c("instrument")])
println("--- per-instrument point counts (kept / total after flag cut) ---")
for n in sort(unique(istr_all))
    tot=count(==(n),istr_all); kpt=count(i->istr_all[i]==n && keep[i],1:length(istr_all))
    @printf("  %-12s %3d / %3d\n", n, kpt, tot)
end

raw=raw[keep,:]; bjd=Float64.(raw[:,c("bjd")]); rv=Float64.(raw[:,c("rv")]); rve=Float64.(raw[:,c("rv_err")])
istr=String.(raw[:,c("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
rinst=[i2i[s] for s in istr]
data=Data(;t_rv=bjd,rv=rv,rv_err=rve,rv_inst=rinst)
ic=InstrumentConfig(rv=inames)
println("\nkept $(length(bjd)) RV points across $(length(inames)) instruments: ", join(inames,", "))
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
    @printf("  %-12s n=%-3d scatter=%6.1f m/s  med_err=%5.1f m/s\n",
            n, length(ri), std(ri), median(rve[rinst.==i2i[n]])); end

# ---- shared priors: 1 planet (b), tight period so both runs find b ----
function build_priors()
    pri=Dict{String,PriorSpec}()
    pri["n_p"]=FixedPrior(1.0)
    pri["P_k1"]=UniformPrior(4.0,4.32)          # tight around b=4.159
    pri["K_k1"]=UniformPrior(0.0,250.0)         # generous; truth 140.8
    pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
    pri["Mo_k1"]=UniformPrior(-2π,2π)
    for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
        ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
        pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp)     # ignored when marg on
        pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
    return pri
end

function build_params(marg::Bool)
    Params(;max_kplanet=1,planet_modes=[RV_ONLY],instruments=ic,data=data,
        M_s=1.04,R_s=1.137,priors=build_priors(),trend_order=0,stability=:none,
        parametrization=ParametrizationConfig(time=:Mo,marginalize_gamma=marg),
        external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])
end

function run_case(label, marg)
    params=build_params(marg)
    target=NereusTarget(params,data;unconstrained=true)
    ndim=length(params.layout.unfrozen_idx)
    t0=time()
    res=sample_ptemcee(target,data;n_temps=8,n_walkers=60,n_steps=3000,
                       n_burnin=1500,seed=42,show_progress=false)
    secs=time()-t0
    ch=res.chains
    Kc=vec(Array(ch[:,:K_k1,:])); Pc=vec(Array(ch[:,:P_k1,:]))
    Kq=quantile(Kc,[0.16,0.5,0.84]); Pm=median(Pc)
    @printf("\n[%s] marg=%-5s ndim=%d  %.0fs\n", label, marg, ndim, secs)
    @printf("    P = %.4f d   K = %.1f  [%.1f, %.1f] m/s   (truth K=%.1f)\n",
            Pm, Kq[2], Kq[1], Kq[3], B.K)
    for n in inames
        s=Symbol("sigma_$n"); s in names(ch) || continue
        @printf("    jitter_%-10s = %5.1f m/s\n", n, median(vec(Array(ch[:,s,:]))))
    end
    return Kq[2]
end

println("\n==================  WASP-47 b  K-attribution  ==================")
Koff = run_case("free-γ ", false)
Kon  = run_case("marg-γ ", true)
@printf("\nΔ: free-γ K=%.1f  →  marg-γ K=%.1f   (truth %.1f)\n", Koff, Kon, B.K)
println(Kon > 110 && Koff < 90 ? ">> γ-fragmentation CONFIRMED: marginalizing γ recovers b." :
        Kon < 90 ? ">> γ exonerated: K stays low even with γ marginalized — look elsewhere." :
        ">> ambiguous; inspect chains.")
