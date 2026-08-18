#!/usr/bin/env julia
# Profile likelihood in K at e=0.2 (Vines value) on the REAL HD 18599 RV.
# For the white+AD model the RV likelihood is exactly Gaussian with a LINEAR
# mean  μ = γ_inst + Keplerian(K,e,ω,P,Tc) + Σ_k C_k I_k(t), so at fixed
# (K,e,ω,jitter) the (γ,C) optimum is weighted least squares — closed form.
# Profile = max over ω and per-instrument jitter (iterated). This is Nereus's
# white+AD likelihood exactly, with NO sampler. It localizes which K the data
# PREFER under each activity model — the clean likelihood-vs-sampling test.

using DelimitedFiles, Statistics, Printf, LinearAlgebra

const RV_FILE = joinpath(@__DIR__, "..", "data", "hd18599.csv")
const P_REF, T0_REF = 4.1374685534602405, 2.4583545857470357e6
const E_FIX = 0.2
const PAPER_INST = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

raw = readdlm(RV_FILE, ',', Any, '\n'; header=true); dm = raw[1]
keep=Int[]; inst_str=String[]
for i in 1:size(dm,1)
    ins=strip(String(dm[i,16])); prov=strip(String(dm[i,17]))
    (ins=="HARPS_POST" && prov=="ESO_PHASE3") && continue
    ins in PAPER_INST || continue; push!(keep,i); push!(inst_str,ins)
end
dm=dm[keep,:]; bjd=Float64.(dm[:,1]); rv=Float64.(dm[:,2]); rverr=Float64.(dm[:,3])
inst_names=sort!(unique(inst_str)); imap=Dict(n=>i for (i,n) in enumerate(inst_names))
rv_inst=[imap[s] for s in inst_str]
let kc=trues(length(bjd))
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); m=median(rv[idx]); thr=5*max(1.4826*median(abs.(rv[idx].-m)),1e-6)
        for k in idx; abs(rv[k]-m)>thr && (kc[k]=false); end
    end
    global bjd=bjd[kc]; global rv=rv[kc]; global rverr=rverr[kc]; global rv_inst=rv_inst[kc]; global dm=dm[kc,:]
end
n=length(bjd); nI=length(inst_names)
col=Dict(:bis=>4,:fwhm=>6,:halpha=>10,:logrhk=>12)
function load_adfmt(cv)
    v=Float64[let x=dm[i,cv]; (x==="" || (x isa AbstractString && strip(x)=="")) ? 0.0 : Float64(x) end for i in 1:size(dm,1)]
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); fin=filter(k->isfinite(v[k]),idx); isempty(fin)&&continue
        μ=mean(v[fin]); for k in fin; v[k]-=μ; end
        vmax=maximum(abs,v[fin]); vmax==0&&continue
        rvv=rv[idx]; rmax=maximum(abs,rvv.-mean(rvv)); rmax==0&&continue
        for k in fin; v[k]=v[k]/vmax*rmax; end
    end
    v
end
chs=[:bis,:fwhm,:halpha,:logrhk]; Ind=Dict(ch=>load_adfmt(col[ch]) for ch in chs)

function kepler_rv(K,e,w)
    out=zeros(n); f_tc=π/2-w
    E_tc=2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2)); M_tc=E_tc-e*sin(E_tc)
    @inbounds for i in 1:n
        M=2π*(bjd[i]-T0_REF)/P_REF + M_tc
        E=M; for _ in 1:50; E-=(E-e*sin(E)-M)/(1-e*cos(E)); end
        f=2*atan(sqrt((1+e)/(1-e))*tan(E/2)); out[i]=K*(cos(f+w)+e*cos(w))
    end
    out
end

# design matrix for an AD config (list of indicator symbols): per-inst intercept + per (ind,inst) col
function design(cfg)
    cols=Vector{Vector{Float64}}()
    for ii in 1:nI; push!(cols, Float64[rv_inst[k]==ii ? 1.0 : 0.0 for k in 1:n]); end
    for ch in cfg, ii in 1:nI
        push!(cols, Float64[rv_inst[k]==ii ? Ind[ch][k] : 0.0 for k in 1:n])
    end
    hcat(cols...)
end

function profile_ll_K(K, X)
    best=-Inf
    for w in range(0, 2π, length=13)[1:12]
        kep=kepler_rv(K, E_FIX, w); y=rv .- kep
        jit=fill(median(rverr), nI)               # per-instrument jitter, iterated
        ll=-Inf
        for _ in 1:8
            W=Float64[1.0/(rverr[k]^2 + jit[rv_inst[k]]^2) for k in 1:n]
            XtW=X .* W; A=X'*XtW; A[diagind(A)].+=1e-6
            β=A \ (X'*(W.*y)); resid=y .- X*β
            ll=-0.5*sum(@. resid^2*W - log(W) + log(2π))
            for ii in 1:nI
                idx=findall(==(ii),rv_inst); isempty(idx)&&continue
                v=mean(resid[idx].^2 .- rverr[idx].^2); jit[ii]=sqrt(max(v,1e-6))
            end
        end
        best=max(best, ll)
    end
    best
end

configs=[("white",Symbol[]), ("BIS-only",[:bis]), ("12-coeff",chs)]
Ks=collect(0.0:0.5:18.0)
@printf("Real HD18599 RV-only profile logL(K) at e=%.2f (%d RV)\n\n", E_FIX, n)
for (lab,cfg) in configs
    X=design(cfg)
    lls=[profile_ll_K(K,X) for K in Ks]
    kbest=Ks[argmax(lls)]; llmax=maximum(lls)
    ll6=lls[findfirst(==(6.0),Ks)]; ll11=lls[findfirst(==(11.0),Ks)]; ll0=lls[1]
    @printf("%-9s  peak K=%4.1f   ΔlogL[K=11 − K=6]=%+6.2f   ΔlogL[K=6 − K=0]=%+6.2f   ΔlogL[K=11 − K=0]=%+6.2f\n",
            lab, kbest, ll11-ll6, ll6-ll0, ll11-ll0)
end
