#!/usr/bin/env julia
# Why does joint RVPM give K=6.23 while RV-only gives K=9.52 under the SAME
# 12-coeff ActivityDecorrelation? Hypothesis: the TESS transit's photoeccentric
# constraint pins e, and the K-e degeneracy drags K down. FIXED-DIM (no
# trans-dim entrenchment), 12-coeff AD always on, multi-seed. Report K, e,
# and corr(K,e) for RV-only vs joint.

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

const RV_FILE = joinpath(@__DIR__, "..", "data", "hd18599.csv")
const LC_FILE = joinpath(@__DIR__, "..", "..", "..", "data", "HD18599", "HD18599_cleaned_lc.csv")
const P_REF, T0_REF, M_S, R_S = 4.1374685534602405, 2.4583545857470357e6, 0.807, 0.798
const RHO_S = M_S / R_S^3
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
adkey=Dict(:bis=>"bisector_span",:fwhm=>"fwhm_AD",:halpha=>"halpha_AD",:logrhk=>"log_rhk_AD")
chs=[:bis,:fwhm,:halpha,:logrhk]
indicators=Dict{String,Vector{Float64}}(); ierrs=Dict{String,Vector{Float64}}()
for ch in chs; indicators[adkey[ch]]=load_adfmt(col[ch]); ierrs[adkey[ch]]=fill(1.0,length(bjd)); end
ad() = ActivityDecorrelation(indicators=[adkey[c] for c in chs])

base_pri() = begin
    rvmax=maximum(abs,rv)
    p=Dict{String,PriorSpec}(
        "n_p"=>FixedPrior(1.0),
        "P_k1"=>NormalPrior(P_REF,1e-3,P_REF-0.01,P_REF+0.01),
        "Tc_k1"=>NormalPrior(T0_REF,0.05,T0_REF-0.3,T0_REF+0.3),
        "K_k1"=>UniformPrior(0.0,50.0),
        "sesinw_k1"=>UniformPrior(-0.7,0.7), "secosw_k1"=>UniformPrior(-0.7,0.7))
    for nm in inst_names; p["gamma_$nm"]=UniformPrior(-3rvmax,3rvmax); p["sigma_$nm"]=LogUniformPrior(1e-3,30.0); end
    p
end

function summarize(res, tag)
    K=vec(Array(res.chains[:,:K_k1,:]))
    se=vec(Array(res.chains[:,:sesinw_k1,:])); co=vec(Array(res.chains[:,:secosw_k1,:]))
    e=se.^2 .+ co.^2
    r = cor(K, e)
    @printf("  [%s] K=%5.2f [%.2f,%.2f]   e=%.3f [%.3f,%.3f]   corr(K,e)=%+.2f\n",
            tag, median(K),quantile(K,0.16),quantile(K,0.84),
            median(e),quantile(e,0.16),quantile(e,0.84), r)
    flush(stdout)
    (median(K), median(e))
end

# ---------- RV-ONLY ----------
println("=== RV-ONLY, 12-coeff AD ===")
data_rv=Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst, indicators=indicators, indicator_errs=ierrs)
for sd in 1:3
    pr=base_pri()
    par=Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=inst_names),
        data=data_rv, parametrization=ParametrizationConfig(time=:Tc), priors=pr,
        noise_models=NoiseModel[ad()], stability=:none)
    res=sample_ptemcee(NereusTarget(par,data_rv), data_rv; n_temps=12, n_walkers=64,
        n_steps=9000, n_burnin=4000, seed=sd, init_strategy=:prior, show_progress=false)
    summarize(res, "rv s$sd")
end

# ---------- JOINT RVPM (with TESS transit) ----------
println("\n=== JOINT RVPM (+TESS transit), 12-coeff AD ===")
lc = readdlm(LC_FILE, ',', Float64; comments=true, comment_char='#', header=true)[1]
pt_t=lc[:,1]; pt_f=lc[:,2]; pt_e=lc[:,3]; pt_i=ones(Int,length(pt_t))
@printf("  TESS: %d cadences\n", length(pt_t))
data_j=Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
              t_phot=pt_t, flux=pt_f, flux_err=pt_e, phot_inst=pt_i,
              indicators=indicators, indicator_errs=ierrs)
ic=InstrumentConfig(rv=inst_names, pm=["TESS"])
for sd in 1:2
    pr=base_pri()
    pr["b_k1"]=UniformPrior(0.0,1.0); pr["rr_k1"]=UniformPrior(0.005,0.10)
    pr["rho_s"]=NormalPrior(RHO_S,0.4,0.3,4.0)
    pr["q1_TESS"]=UniformPrior(0.0,1.0); pr["q2_TESS"]=UniformPrior(0.0,1.0)
    pr["offset_TESS"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pr["jitter_TESS"]=LogUniformPrior(1e-5,5e-3)
    par=Params(; max_kplanet=1, planet_modes=[RVPM], instruments=ic, data=data_j, M_s=M_S, R_s=R_S,
        parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true), priors=pr,
        noise_models=NoiseModel[ad()], stability=:none)
    res=sample_ptemcee(NereusTarget(par,data_j), data_j; n_temps=12, n_walkers=48,
        n_steps=6000, n_burnin=2500, seed=sd, init_strategy=:prior, show_progress=false)
    summarize(res, "joint s$sd")
end
