#!/usr/bin/env julia
# Real-data K-by-activity-model on HD 18599 (RV only, P/Tc transit-pinned).
# Mirror of the injection test, but on the ACTUAL RV. If the 12-coeff AD is
# an unbiased K estimator (injection proved it is) yet returns K≈6 here while
# a GP-rotation model returns K≈11, then K=6 is MODEL-MISMATCH (linear
# indicator decorrelation cannot represent the QP rotation activity), not
# over-absorption — and the two-paper K≈11 is the honest value.
#
# Configs: white | BIS-only AD | 4-indicator AD (12 coef) | CeleriteRotation GP.

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random

const RV_FILE = joinpath(@__DIR__, "..", "data", "hd18599.csv")
const P_REF, T0_REF, P_ROT = 4.1374685534602405, 2.4583545857470357e6, 8.74
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
# AD-format indicators: mean-subtract per instrument, scale to RV range (EXACT joint-run prep)
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
data=Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst, indicators=indicators, indicator_errs=ierrs)
n=length(bjd)
@printf("Real HD18599: %d RV (%s), RV rms=%.2f m/s\n", n, join(inst_names,","), std(rv))

function fitK(nmods, label)
    rvmax=maximum(abs,rv)
    pri=Dict{String,PriorSpec}(
        "n_p"=>FixedPrior(1.0),
        "P_k1"=>NormalPrior(P_REF,1e-3,P_REF-0.01,P_REF+0.01),
        "Tc_k1"=>NormalPrior(T0_REF,0.05,T0_REF-0.3,T0_REF+0.3),
        "K_k1"=>UniformPrior(0.0,50.0),
        "sesinw_k1"=>UniformPrior(-0.7,0.7), "secosw_k1"=>UniformPrior(-0.7,0.7))
    for nm in inst_names; pri["gamma_$nm"]=UniformPrior(-3rvmax,3rvmax); pri["sigma_$nm"]=LogUniformPrior(1e-3,30.0); end
    any(m -> m isa CeleriteRotation, nmods) && (pri["gp_period"]=NormalPrior(P_ROT,1.0,4.0,16.0))
    params=Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=inst_names),
        data=data, parametrization=ParametrizationConfig(time=:Tc), priors=pri,
        noise_models=nmods, stability=:none)
    tgt=NereusTarget(params, data)
    Ks=Float64[]; los=Float64[]; his=Float64[]
    for sd in 1:3
        res=sample_ptemcee(tgt, data; n_temps=12, n_walkers=60, n_steps=9000,
            n_burnin=4000, seed=sd, init_strategy=:prior, show_progress=false)
        K=vec(Array(res.chains[:,:K_k1,:]))
        push!(Ks,median(K)); push!(los,quantile(K,0.16)); push!(his,quantile(K,0.84))
    end
    @printf("%-26s  K=%5.2f  [%.2f, %.2f]   logZ=%.1f\n", label, mean(Ks), mean(los), mean(his),
            try; r=sample_ptemcee(tgt,data;n_temps=12,n_walkers=60,n_steps=9000,n_burnin=4000,seed=99,init_strategy=:prior,show_progress=false); r.log_evidence; catch; NaN; end)
end

@printf("\n%-26s  %-22s\n", "activity model", "K (med [16,84]) — published K≈11")
fitK(NoiseModel[], "white (no activity)")
fitK(NoiseModel[ActivityDecorrelation(indicators=["bisector_span"])], "BIS-only AD (3 coef)")
fitK(NoiseModel[ActivityDecorrelation(indicators=[adkey[c] for c in chs])], "4-indicator AD (12 coef)")
fitK(NoiseModel[CeleriteRotation(channel=:rv)], "CeleriteRotation GP")
