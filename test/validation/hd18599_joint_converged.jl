#!/usr/bin/env julia
# HD 18599 — converged joint RVPM fit, 12-coeff AD, to MAP THE BIMODAL e
# posterior and report the marginal K + detection significance. Many walkers
# + 14 temps + prior init so both e-basins (low-e≈0.15 non-detection,
# high-e≈0.32 detection) are populated and PT swaps weight them. Two seeds for
# robustness; chains saved for plotting.
#
# Run: julia --project=. -t 10 Nereus.jl/test/validation/hd18599_joint_converged.jl

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

const RV_FILE = joinpath(@__DIR__, "..", "data", "hd18599.csv")
const LC_FILE = joinpath(@__DIR__, "..", "..", "..", "data", "HD18599", "HD18599_cleaned_lc.csv")
const P_REF, T0_REF, M_S, R_S = 4.1374685534602405, 2.4583545857470357e6, 0.807, 0.798
const RHO_S = M_S / R_S^3
const PAPER_INST = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])
const OUTDIR = joinpath(@__DIR__, "..", "..", "results", "HD18599_joint_converged")

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

lc = readdlm(LC_FILE, ',', Float64; comments=true, comment_char='#', header=true)[1]
pt_t=lc[:,1]; pt_f=lc[:,2]; pt_e=lc[:,3]; pt_i=ones(Int,length(pt_t))
data=Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
            t_phot=pt_t, flux=pt_f, flux_err=pt_e, phot_inst=pt_i,
            indicators=indicators, indicator_errs=ierrs)
ic=InstrumentConfig(rv=inst_names, pm=["TESS"])
@printf("%d RV (%s), %d TESS cadences\n", length(bjd), join(inst_names,","), length(pt_t))

rvmax=maximum(abs,rv)
pri=Dict{String,PriorSpec}(
    "n_p"=>FixedPrior(1.0),
    "P_k1"=>NormalPrior(P_REF,1e-3,P_REF-0.01,P_REF+0.01),
    "Tc_k1"=>NormalPrior(T0_REF,0.05,T0_REF-0.3,T0_REF+0.3),
    "K_k1"=>UniformPrior(0.0,50.0),
    "sesinw_k1"=>UniformPrior(-0.7,0.7), "secosw_k1"=>UniformPrior(-0.7,0.7),
    "b_k1"=>UniformPrior(0.0,1.0), "rr_k1"=>UniformPrior(0.005,0.10),
    "rho_s"=>NormalPrior(RHO_S,0.4,0.3,4.0),
    "q1_TESS"=>UniformPrior(0.0,1.0), "q2_TESS"=>UniformPrior(0.0,1.0),
    "offset_TESS"=>NormalPrior(0.0,1e-3,-5e-3,5e-3), "jitter_TESS"=>LogUniformPrior(1e-5,5e-3))
for nm in inst_names; pri["gamma_$nm"]=UniformPrior(-3rvmax,3rvmax); pri["sigma_$nm"]=LogUniformPrior(1e-3,30.0); end
params=Params(; max_kplanet=1, planet_modes=[RVPM], instruments=ic, data=data, M_s=M_S, R_s=R_S,
    parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true),
    priors=pri, noise_models=NoiseModel[ActivityDecorrelation(indicators=[adkey[c] for c in chs])],
    stability=:none)

NT=parse(Int,get(ENV,"N_TEMPS","16")); NW=parse(Int,get(ENV,"N_WALKERS","200"))
NS=parse(Int,get(ENV,"N_STEPS","25000")); NB=parse(Int,get(ENV,"N_BURNIN","6000"))   # NS = max-steps CEILING
SEED=parse(Int,get(ENV,"SEED","1"))
@printf("converged joint (run-until-converged): %d temps × %d walkers, ceiling %d, burn %d, seed %d\n", NT,NW,NS,NB,SEED)
mkpath(OUTDIR)
t0=time()
# Run-until-converged: stop once the science params (planet block + rho_s,
# auto-detected; nuisances non-gating) clear R-hat<1.01 AND tail-ESS>1000 for
# 3 consecutive checks. Live R-hat/ESS (mean/worst) print in the progress bar.
res=sample_ptemcee(NereusTarget(params,data), data; n_temps=NT, n_walkers=NW,
    n_steps=NS, n_burnin=NB, seed=SEED, init_strategy=:prior, show_progress=true,
    convergence_stop=true, rhat_threshold=1.01, tail_ess_threshold=1000,
    n_converged_checks=3, min_steps=10000, diag_every=1000)
# per-walker cold-chain series (iterations × walkers) for the mode-crossing check
Kmat=Array(res.chains[:,:K_k1,:]); semat=Array(res.chains[:,:sesinw_k1,:]); comat=Array(res.chains[:,:secosw_k1,:])
emat=semat.^2 .+ comat.^2
allK=vec(Kmat); alle=vec(emat)
@printf("seed %d done %.1f min: K=%.2f [%.2f,%.2f] e=%.3f logZ=%.1f\n",
        SEED,(time()-t0)/60, median(allK),quantile(allK,0.16),quantile(allK,0.84),median(alle),res.log_evidence)
save_chains(joinpath(OUTDIR,"chains_s1.nc"), res.chains, params; data=data)
# MODE-CROSSING DIAGNOSTIC: a real posterior weight requires walkers to
# transition across the e-barrier, not just start on either side. Count
# cold-chain walkers whose e series spans the 0.22 boundary.
nw_chain=size(emat,2)
crossers=count(w -> minimum(@view emat[:,w])<0.22 && maximum(@view emat[:,w])>0.22, 1:nw_chain)
@printf("mode-crossing: %d/%d cold-chain walkers cross e=0.22 (need >>0 for a real bimodal weight)\n",
        crossers, nw_chain)
flush(stdout)

# ---- CONVERGENCE CRITERIA: ESS + R-hat (walkers as chains) ----
print_ess_rhat(res.chains)
function ess_rhat_of(name, mat)   # mat: iterations × walkers
    c = MCMCChains.Chains(reshape(mat, size(mat,1), 1, size(mat,2)), [Symbol(name)])
    t = MCMCChains.ess_rhat(c)
    (Float64(t[1,:ess]), Float64(t[1,:rhat]))
end
essK,rhatK = ess_rhat_of("K", Kmat); esse,rhate = ess_rhat_of("e", emat)
conv = rhatK<1.01 && rhate<1.01 && essK>400 && esse>400 && crossers>0
@printf("\nCONVERGENCE:  K  ESS=%.0f Rhat=%.4f  |  e  ESS=%.0f Rhat=%.4f  |  e-crossers=%d/%d\n",
        essK,rhatK, esse,rhate, crossers,nw_chain)
@printf("VERDICT: %s  (need Rhat<1.01, ESS>400, crossers>0)\n",
        conv ? "CONVERGED — marginal trustworthy" : "NOT CONVERGED — extend steps/temps")

# ---- marginal: detection significance + bimodal e weights ----
q(p)=quantile(allK,p)
@printf("\n===== MARGINAL (single converged PT run, %d samples) =====\n", length(allK))
@printf("K = %.2f  68%%[%.2f,%.2f]  95%%[%.2f,%.2f]  99.7%%[%.2f,%.2f]\n",
        median(allK), q(0.16),q(0.84), q(0.025),q(0.975), q(0.0015),q(0.9985))
@printf("K lower bounds: 1σ=%.2f  2σ=%.2f  3σ=%.2f   → %s at 3σ\n",
        q(0.16),q(0.025),q(0.0015), q(0.0015)>0 ? "DETECTION" : "non-detection")
lo = alle .< 0.22
@printf("e bimodality: low-e(<0.22) frac=%.2f  K|low=%.2f ;  high-e(≥0.22) frac=%.2f  K|high=%.2f\n",
        mean(lo), median(allK[lo]), mean(.!lo), median(allK[.!lo]))
writedlm(joinpath(OUTDIR,"K_e_samples.csv"), hcat(allK,alle), ',')
println("saved chains + K_e_samples.csv to ", OUTDIR)
