#!/usr/bin/env julia
# HD 18599 — FULL NOISE SUITE evidence ranking (the "flexibility pitch").
#
# Every noise model that scores p(RV) — treats indicators (if used) as
# fixed mean regressors, or doesn't use them — on the SAME RV data, so
# their fixed-dim PT log-evidences are directly comparable. Ranked by
# logZ (SIGN-CORRECTED, evidence.jl e4249ce) with K + detection per model.
#
# NOT here: ActivityGP scores the JOINT p(RV, indicators) — different
# data — comparable only via the chain-rule conditional
# (xcheck_HD18599_AD_vs_AGP.jl; disfavored, AD beats AGP ~73-148 nats).
#
# Run: julia --project=. -t 10 Nereus.jl/test/validation/hd18599_noise_suite.jl

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random

const RV_FILE = joinpath(@__DIR__, "..", "data", "hd18599.csv")
const P_REF, M_S, R_S, P_ROT = 4.1374685534602405, 0.807, 0.798, 8.74
const PAPER_INST = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

raw = readdlm(RV_FILE, ',', Any, '\n'; header=true); dm = raw[1]
keep = Int[]; inst_str = String[]
for i in 1:size(dm,1)
    ins = strip(String(dm[i,16])); prov = strip(String(dm[i,17]))
    (ins=="HARPS_POST" && prov=="ESO_PHASE3") && continue
    ins in PAPER_INST || continue
    push!(keep, i); push!(inst_str, ins)
end
dm = dm[keep,:]
bjd=Float64.(dm[:,1]); rv=Float64.(dm[:,2]); rverr=Float64.(dm[:,3])
inst_names = sort!(unique(inst_str))
imap = Dict(n=>i for (i,n) in enumerate(inst_names)); rv_inst=[imap[s] for s in inst_str]
# 5σ-MAD per-instrument outlier clip (the FEROS +453 m/s point)
let kc = trues(length(bjd))
    for id in unique(rv_inst)
        idx = findall(==(id), rv_inst); m = median(rv[idx])
        thr = 5*max(1.4826*median(abs.(rv[idx].-m)), 1e-6)
        for k in idx; abs(rv[k]-m)>thr && (kc[k]=false); end
    end
    global bjd=bjd[kc]; global rv=rv[kc]; global rverr=rverr[kc]
    global rv_inst=rv_inst[kc]; global dm=dm[kc,:]
end

# Indicators (AD-format: EMPEROR-normalized fixed regressors)
col = Dict(:bis=>(4,5), :fwhm=>(6,7), :halpha=>(10,11), :logrhk=>(12,13))
function load_ind(ch)
    cv,_ = col[ch]
    v = Float64[let x=dm[i,cv]; (x==="" || (x isa AbstractString && strip(x)=="")) ? NaN : Float64(x) end for i in 1:size(dm,1)]
    for i in eachindex(v); isfinite(v[i]) || (v[i]=0.0); end
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); fin=filter(k->isfinite(v[k]),idx); isempty(fin)&&continue
        μ=mean(v[fin]); for k in fin; v[k]-=μ; end
        vmax=maximum(abs,v[fin]); vmax==0&&continue
        rvv=rv[idx]; rmax=maximum(abs,rvv.-mean(rvv)); rmax==0&&continue
        for k in fin; v[k]=v[k]/vmax*rmax; end
    end
    v
end
indicators = Dict("bisector_span"=>load_ind(:bis), "fwhm_AD"=>load_ind(:fwhm),
                  "halpha_AD"=>load_ind(:halpha), "log_rhk_AD"=>load_ind(:logrhk))
data = Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst, indicators=indicators)
ic = InstrumentConfig(rv=inst_names)
@printf("Loaded %d RVs, %d instruments (clipped)\n", length(bjd), length(inst_names))

rvmax = maximum(abs, rv)
function base_priors()
    p = Dict{String,PriorSpec}(
        "P_k1"=>NormalPrior(P_REF,1e-3,P_REF-0.01,P_REF+0.01),
        "K_k1"=>UniformPrior(0.0,50.0), "sesinw_k1"=>UniformPrior(-1.0,1.0),
        "secosw_k1"=>UniformPrior(-1.0,1.0), "Mo_k1"=>UniformPrior(0.0,2π))
    for n in inst_names; p["gamma_$n"]=UniformPrior(-3rvmax,3rvmax); end
    p
end

ad  = ActivityDecorrelation(indicators=["bisector_span","fwhm_AD","halpha_AD","log_rhk_AD"])
suite = [
    ("white",         NoiseModel[],                          Dict{String,PriorSpec}()),
    ("AD-4indicator", NoiseModel[ad],                        Dict{String,PriorSpec}()),
    ("CeleriteRotation", NoiseModel[CeleriteRotation(channel=:rv)],
        Dict{String,PriorSpec}("gp_period"=>NormalPrior(P_ROT,1.0,4.0,16.0))),
    ("CeleriteSHO",   NoiseModel[CeleriteSHO(channel=:rv)],   Dict{String,PriorSpec}()),
    ("CeleriteFM17",  NoiseModel[CeleriteRotationFM17(channel=:rv)],
        Dict{String,PriorSpec}("gp_log_period"=>NormalPrior(log(P_ROT),0.15,log(4.0),log(16.0)))),
    ("MA(1)",         NoiseModel[MAModel(order=1)],           Dict{String,PriorSpec}()),
    ("AR(1)",         NoiseModel[ARModel(order=1)],           Dict{String,PriorSpec}()),
    ("ActivityJitter",NoiseModel[ActivityJitter(indicator="log_rhk_AD")], Dict{String,PriorSpec}()),
]

NT=parse(Int,get(ENV,"N_TEMPS","14")); NW=parse(Int,get(ENV,"N_WALKERS","120"))
NS=parse(Int,get(ENV,"N_STEPS","12000")); NB=parse(Int,get(ENV,"N_BURNIN","5000"))
results = Tuple{String,Float64,Float64,Float64,Float64,Float64,Float64}[]
for (label, nm, extra) in suite
    pri = base_priors(); merge!(pri, extra)
    local params, target
    try
        params = Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic,
            data=data, M_s=M_S, R_s=R_S,
            parametrization=ParametrizationConfig(time=:Mo),
            priors=pri, noise_models=nm)
        target = NereusTarget(params, data)
    catch e
        @printf("%-16s BUILD FAILED: %s\n", label, sprint(showerror,e)[1:min(end,120)]); continue
    end
    zs=Float64[]; Ks=Float64[]
    for sd in (3,17)
        try
            res = sample_ptemcee(target, data; n_temps=NT, n_walkers=NW, n_steps=NS,
                n_burnin=NB, seed=sd, init_strategy=:prior, show_progress=false)
            push!(zs, res.log_evidence)
            append!(Ks, vec(Array(res.chains[:,:K_k1,:])))
        catch e
            @printf("%-16s seed %d FAILED: %s\n", label, sd, sprint(showerror,e)[1:min(end,100)])
        end
    end
    isempty(zs) && continue
    z=mean(zs); zspread=length(zs)>1 ? abs(zs[1]-zs[2]) : 0.0
    Kmed=median(Ks); Klo=quantile(Ks,0.0015); pK1=mean(Ks.<1)
    push!(results, (label, z, zspread, Kmed, quantile(Ks,0.16), Klo, pK1))
    @printf("%-16s logZ=%.2f (spread %.2f)  K=%.2f [1σlo %.2f, 3σlo %.2f]  P(K<1)=%.3f\n",
            label, z, zspread, Kmed, quantile(Ks,0.16), Klo, pK1)
end

println("\n" * "="^72)
println("RANKED BY logZ (higher = favored; ΔlogZ vs best):")
sort!(results, by=r->-r[2])
zbest = isempty(results) ? 0.0 : results[1][2]
for r in results
    @printf("  %-16s logZ=%.2f  ΔlogZ=%+.1f  K=%.2f m/s  P(K<1)=%.3f%s\n",
            r[1], r[2], r[2]-zbest, r[4], r[7], r[3]>2 ? "  ⚠seed-unstable" : "")
end
println("\nAGP (Rajpaul, chain-rule conditional): DISFAVORED, AD beats it ~73-148 nats")
