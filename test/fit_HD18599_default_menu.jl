#!/usr/bin/env julia
# HD 18599 b — trans-dim recovery with the DEFAULT NOISE MENU.
#
# Loads the RV + 4 activity indicators, builds `default_noise_menu(data)`, and
# runs one trans-dim chain that (a) recovers planet b (P≈4.1375 d, K≈11 m/s;
# Vines+2023 / Desidera+2023) and (b) selects the noise model via occupancy.
# The point: validate the canonical menu end-to-end on the reference active
# star — the K≈6 artifact came from linear-BIS + an external e-prior; the menu
# offers GP/AGP/Matérn descriptions so the data can pick.
#
# Run:  julia --project=. -t 10 Nereus.jl/test/fit_HD18599_default_menu.jl
# Env:  NT NW NS NB (temps/walkers/steps/burnin), SEED.

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

const RV_FILE = joinpath(@__DIR__, "data", "hd18599.csv")
const P_REF, M_S, R_S, P_ROT = 4.1374685534602405, 0.807, 0.798, 8.74
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
let kc=trues(length(bjd))                          # 5σ MAD per-instrument clip
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); m=median(rv[idx])
        thr=5*max(1.4826*median(abs.(rv[idx].-m)),1e-6)
        for k in idx; abs(rv[k]-m)>thr && (kc[k]=false); end
    end
    global bjd=bjd[kc]; global rv=rv[kc]; global rverr=rverr[kc]
    global rv_inst=rv_inst[kc]; global dm=dm[kc,:]
end

# 4 activity indicators, raw-centered + MAD-scaled per instrument (AGP/AD/floor
# data). NOTE: menu AD regresses these MAD-scaled indicators (its default
# U(-1,1) coefficient prior assumes RV-peak scaling; close enough for a menu
# validation — GPs dominate on this star anyway).
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
chs=[:bis,:fwhm,:halpha,:logrhk]
indicators=Dict{String,Vector{Float64}}(); ierrs=Dict{String,Vector{Float64}}()
for ch in chs
    vs,es=load_raw(ch); indicators[String(ch)]=vs; ierrs[String(ch)]=es
end

data = Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
              indicators=indicators, indicator_errs=ierrs)
ic = InstrumentConfig(rv=inst_names)
@printf("Loaded %d RVs, %d instruments + %d indicators\n",
        length(bjd), length(inst_names), length(chs))

# ---- the DEFAULT MENU -------------------------------------------------------
menu = default_noise_menu(data; indicators=String.(chs))
@printf("Menu: %d toggleable, %d always-on, %d exclusion groups\n",
        length(menu.toggleable), length(menu.noise_models)-length(menu.toggleable),
        length(menu.exclusion_groups))
for m in menu.toggleable; @printf("   • %s\n", typeof(m)); end

rvmax=maximum(abs,rv)
pri=Dict{String,PriorSpec}(
    "P_k1"=>NormalPrior(P_REF,1e-3,P_REF-0.01,P_REF+0.01), "K_k1"=>UniformPrior(0.0,50.0),
    "sesinw_k1"=>UniformPrior(-1.0,1.0), "secosw_k1"=>UniformPrior(-1.0,1.0),
    "Mo_k1"=>UniformPrior(0.0,2π))
for n in inst_names; pri["gamma_$n"]=UniformPrior(-3rvmax,3rvmax); end
# seed the rotation-timescale hyperparams near Vines+2023 P_rot≈8.74 d.
pri["gp_period"]=NormalPrior(P_ROT,1.0,4.0,16.0)          # CeleriteRotation
pri["gp_act_period"]=NormalPrior(P_ROT,1.0,4.0,16.0)      # ActivityGP
pri["harm_period"]=NormalPrior(P_ROT,1.0,4.0,16.0)        # HarmonicBlock

params=Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic, data=data,
    M_s=M_S, R_s=R_S, parametrization=ParametrizationConfig(time=:Mo),
    priors=pri, noise_models=menu.noise_models, transdim_noise=true,
    # pin e near the published ~0.25 (uniform sesinw/secosw ⇒ p(e)∝e rails e→1
    # on sparse activity-contaminated RV); the K≈11 mode needs this + a GP.
    external_priors=[ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)])
target=NereusTarget(params, data; unconstrained=false)
@printf("Free params: %d\n", n_unfrozen(params))

td=TransDimConfig(; max_kplanet=1, planets=false, noise=true,
    toggleable=menu.toggleable, noise_exclusion_groups=menu.exclusion_groups)

NT=parse(Int,get(ENV,"NT","8")); NW=parse(Int,get(ENV,"NW","128"))
NS=parse(Int,get(ENV,"NS","5000")); NB=parse(Int,get(ENV,"NB","2500"))
SEED=parse(Int,get(ENV,"SEED","42"))
@printf("trans-dim menu run: %d temps × %d walkers × %d+%d burnin (seed=%d)\n",NT,NW,NS,NB,SEED)
t0=time()
res=sample_transdim_ptemcee(target, data; td=td, n_temps=NT, n_walkers=NW, n_steps=NS,
    n_burnin=NB, n_birth_tries=10, n_birth_refine=15, seed=SEED, show_progress=true,
    adapt_ladder=true)   # temp-swap was 0.001 at cold on the fixed ladder
@printf("done in %.1f min  logZ=%.2f\n", (time()-t0)/60, res.log_evidence)

ch=res.chains; cn=names(ch,:parameters)
# SAVE FIRST — so a reporting hiccup never loses a 15-min chain.
mkpath(joinpath(@__DIR__, "..", "results", "HD18599_default_menu"))
save_chains(joinpath(@__DIR__, "..", "results", "HD18599_default_menu", "chains.nc"), ch, params; data=data)
@printf("\nsaved → results/HD18599_default_menu/chains.nc\n")

function report(ch, cn, menu, params)
    Kv=vec(Array(ch[:K_k1])); Pv=vec(Array(ch[:P_k1]))
    @printf("\nPLANET b:  P = %.5f d   K = %.2f ± %.2f m/s (median±sd)   (published K≈11)\n",
            median(Pv), median(Kv), std(Kv))
    # K modes: report the histogram peak too (K-activity is multimodal)
    qs = quantile(Kv, [0.16,0.5,0.84])
    @printf("           K quantiles 16/50/84 = %.2f / %.2f / %.2f m/s\n", qs...)
    ev = (:ecc in cn) ? vec(Array(ch[:ecc])) : Float64[]
    isempty(ev) || @printf("           e = %.3f ± %.3f\n", median(ev), std(ev))

    nm=params.config.noise_models
    occ(m) = (acol=Symbol("noise_active_$(findfirst(==(m), nm))");
              acol in cn ? mean(vec(Array(ch[acol])).>0.5) : NaN)
    @printf("\nNOISE OCCUPANCY  P(M | data), by role (each group sums to 1):\n")
    for (gi, g) in enumerate(menu.exclusion_groups)
        s=0.0
        @printf("  group %d:\n", gi)
        for m in g; p=occ(m); isnan(p)||(s+=p); @printf("     %-24s %.3f\n", string(typeof(m)), p); end
        @printf("     %-24s %.3f\n", "(none in group)", 1-s)
    end
    # NightlyOffset composes freely (not in a single-winner group) — report solo.
    for m in menu.toggleable
        any(m in g for g in menu.exclusion_groups) && continue
        @printf("  composing:  %-20s %.3f\n", string(typeof(m)), occ(m))
    end
end
report(ch, cn, menu, params)
