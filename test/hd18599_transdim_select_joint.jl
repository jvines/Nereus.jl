#!/usr/bin/env julia
# HD 18599 — JOINT RV + TESS-photometry TRANS-DIM noise-model SELECTION.
# Planet (TOI-179 b) constrained by transits + RV; the activity noise
# models toggle on the RV side. One trans-dim chain → P(M | RV, phot,
# indicators). Photometry = the cleaned (detrended) all-sector TESS LC.
#
# Run: julia --project=. -t 10 Nereus.jl/test/hd18599_transdim_select_joint.jl

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

const RV_FILE = joinpath(@__DIR__, "data", "hd18599.csv")
const LC_FILE = joinpath(@__DIR__, "..", "..", "data", "HD18599", "HD18599_cleaned_lc.csv")
const P_REF, T0_REF, M_S, R_S, P_ROT = 4.1374685534602405, 2.4583545857470357e6, 0.807, 0.798, 8.74
const RHO_S = M_S / R_S^3                          # ≈ 1.59 ρ_sun
const PAPER_INST = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

# ---- RV + indicators (same prep + 5σ-MAD clip as the RV-only script) ----
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
col=Dict(:bis=>(4,5),:fwhm=>(6,7),:halpha=>(10,11),:logrhk=>(12,13))
function load_raw(ch)
    cv,ce=col[ch]
    v=Float64[let x=dm[i,cv]; (x==="" || (x isa AbstractString && strip(x)=="")) ? NaN : Float64(x) end for i in 1:size(dm,1)]
    e=Float64[let x=dm[i,ce]; (x==="" || (x isa AbstractString && strip(x)=="")) ? NaN : Float64(x) end for i in 1:size(dm,1)]
    vs=copy(v); es=copy(e)
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); fin=filter(k->isfinite(vs[k]),idx)
        μ=isempty(fin) ? 0.0 : median(vs[fin]); mad=isempty(fin) ? 1.0 : max(1.4826*median(abs.(vs[fin].-μ)),1e-12)
        fe=filter(k->isfinite(es[k])&&es[k]>0,idx); mes=isempty(fe) ? 0.1 : median(es[fe])/mad
        for k in idx
            if isfinite(vs[k]); vs[k]=(vs[k]-μ)/mad; es[k]=(isfinite(es[k])&&es[k]>0) ? es[k]/mad : mes
            else; vs[k]=0.0; es[k]=1e3 end
        end
    end
    vs, es
end
function load_adfmt(ch)
    cv,_=col[ch]
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
chs=[:bis,:fwhm,:halpha,:logrhk]
indicators=Dict{String,Vector{Float64}}(); ierrs=Dict{String,Vector{Float64}}()
for ch in chs; vs,es=load_raw(ch); indicators[String(ch)]=vs; ierrs[String(ch)]=es; end
adkey=Dict(:bis=>"bisector_span",:fwhm=>"fwhm_AD",:halpha=>"halpha_AD",:logrhk=>"log_rhk_AD")
for ch in chs; indicators[adkey[ch]]=load_adfmt(ch); end

# ---- TESS cleaned LC (detrended; flux ~1) ----
lc = readdlm(LC_FILE, ',', Float64; comments=true, comment_char='#', header=true)[1]
pt_t = lc[:,1]; pt_f = lc[:,2]; pt_e = lc[:,3]
pt_i = ones(Int, length(pt_t))
@printf("Loaded %d RVs (%d inst), %d indicator chans, %d TESS cadences (%.0f d span)\n",
        length(bjd), length(inst_names), length(chs), length(pt_t), pt_t[end]-pt_t[1])

data = Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
              t_phot=pt_t, flux=pt_f, flux_err=pt_e, phot_inst=pt_i,
              indicators=indicators, indicator_errs=ierrs)
ic = InstrumentConfig(rv=inst_names, pm=["TESS"])

rvmax=maximum(abs,rv)
pri=Dict{String,PriorSpec}(
    "P_k1"=>NormalPrior(P_REF, 1e-3, P_REF-0.01, P_REF+0.01),
    "Tc_k1"=>NormalPrior(T0_REF, 0.05, T0_REF-0.3, T0_REF+0.3),
    "K_k1"=>UniformPrior(0.0, 50.0),
    "sesinw_k1"=>UniformPrior(-0.7,0.7), "secosw_k1"=>UniformPrior(-0.7,0.7),
    "b_k1"=>UniformPrior(0.0,1.0), "rr_k1"=>UniformPrior(0.005,0.10),
    "rho_s"=>NormalPrior(RHO_S, 0.4, 0.3, 4.0),
    "q1_TESS"=>UniformPrior(0.0,1.0), "q2_TESS"=>UniformPrior(0.0,1.0),
    "offset_TESS"=>NormalPrior(0.0,1e-3,-5e-3,5e-3), "jitter_TESS"=>LogUniformPrior(1e-5,5e-3))
for n in inst_names; pri["gamma_$n"]=UniformPrior(-3rvmax,3rvmax); end
pri["gp_period"]=NormalPrior(P_ROT,1.0,4.0,16.0)
pri["gp_act_period"]=NormalPrior(P_ROT,1.0,4.0,16.0)

ad  = ActivityDecorrelation(indicators=["bisector_span","fwhm_AD","halpha_AD","log_rhk_AD"])
agp = ActivityGP(channels=chs)
rot = CeleriteRotation(channel=:rv)
ma  = MAModel(order=1); ar = ARModel(order=1)
flr = IndicatorFloor(channels=chs)
toggle = NoiseModel[ad, agp, rot, ma, ar]
params=Params(; max_kplanet=1, planet_modes=[RVPM], instruments=ic, data=data,
    M_s=M_S, R_s=R_S, parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true),
    priors=pri, noise_models=[ad,agp,rot,ma,ar,flr], transdim_noise=true, stability=:none)
target=NereusTarget(params, data; unconstrained=false)
@printf("Free params: %d  | RVPM joint, toggling %d noise models + IndicatorFloor\n",
        n_unfrozen(params), length(toggle))

td=TransDimConfig(; max_kplanet=1, planets=false, noise=true,
    toggleable=toggle, noise_exclusion_groups=[toggle])

N_TEMPS=parse(Int,get(ENV,"N_TEMPS","12")); N_WALKERS=parse(Int,get(ENV,"N_WALKERS","128"))
N_STEPS=parse(Int,get(ENV,"N_STEPS","20000")); N_BURNIN=parse(Int,get(ENV,"N_BURNIN","8000"))
N_BIRTH_TRIES=parse(Int,get(ENV,"N_BIRTH_TRIES","12")); N_BIRTH_REFINE=parse(Int,get(ENV,"N_BIRTH_REFINE","18"))
@printf("JOINT trans-dim noise select: %d temps × %d walkers × %d+%d\n",N_TEMPS,N_WALKERS,N_STEPS,N_BURNIN)
t0=time()
res=sample_transdim_ptemcee(target, data; td=td, n_temps=N_TEMPS, n_walkers=N_WALKERS, n_steps=N_STEPS,
    n_burnin=N_BURNIN, n_birth_tries=N_BIRTH_TRIES, n_birth_refine=N_BIRTH_REFINE, seed=42, show_progress=true)
@printf("done in %.1f min  logZ=%.2f\n", (time()-t0)/60, res.log_evidence)

ch=res.chains; cn=names(ch,:parameters); nm=params.config.noise_models
labels=["AD","AGP","CeleriteRotation","MA(1)","AR(1)"]; occ=Float64[]
for m in toggle
    local nmi=findfirst(==(m),nm); local acol=Symbol("noise_active_$nmi")
    push!(occ, acol in cn ? mean(vec(Array(ch[acol])).>0.5) : NaN)
end
@printf("\nMODEL POSTERIOR P(M | RV, phot, indicators):\n")
for (l,p) in zip(labels,occ); @printf("  P(%-16s) = %.3f\n", l, p); end
@printf("  P(%-16s) = %.3f\n", "white/none", 1-sum(occ))
K=vec(Array(ch[:,:K_k1,:]))
@printf("K_k1 = %.2f [%.2f, %.2f] m/s   (separate-runs RV-only: AD wins, K~8)\n",
        median(K), quantile(K,0.16), quantile(K,0.84))
mkpath(joinpath(@__DIR__,"..","results","HD18599_transdim_joint"))
save_chains(joinpath(@__DIR__,"..","results","HD18599_transdim_joint","chains.nc"), ch, params; data=data)
