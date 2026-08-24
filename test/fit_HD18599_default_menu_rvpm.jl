#!/usr/bin/env julia
# HD 18599 b — RVPM recovery with the DEFAULT NOISE MENU.
#
# Joint RV + TESS transit + default_noise_menu, trans-dim noise selection. The
# transit PINS P/Tc/geometry from the DATA (not a prior peg) and constrains e via
# the stellar density, so K can be recovered against the flexible noise (unlike
# the RV-only run, where P was prior-pegged and the noise ate K). Native-cadence
# LC windowed to b's transits (never binned).
#
# Run: julia --project=. -t 8 Nereus.jl/test/fit_HD18599_default_menu_rvpm.jl
# Env: N_TEMPS N_WALKERS N_STEPS N_BURNIN SEED.

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const RV_FILE   = joinpath(@__DIR__, "data", "hd18599.csv")
# The cleaned TESS light curve is an 11 MB product that does not ship with the
# package. HD18599_LC selects it; the old default resolved to <repo>/../../data,
# which after the rename lands outside any checkout.
const LC_FILE   = get(ENV, "HD18599_LC",
                      joinpath(REPO_ROOT, "data", "HD18599", "HD18599_cleaned_lc.csv"))
const P_REF, T0_REF, DUR = 4.1374685534602405, 2458354.5857470357, 0.067
const M_S, R_S, P_ROT = 0.807, 0.798, 8.74
const PAPER = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

# ---- RV + 4 indicators (raw-centered, MAD-scaled) ---------------------------
raw = readdlm(RV_FILE, ',', Any, '\n'; header=true); dm = raw[1]
keep=Int[]; istr=String[]
for i in 1:size(dm,1)
    ins=strip(String(dm[i,16])); prov=strip(String(dm[i,17]))
    (ins=="HARPS_POST" && prov=="ESO_PHASE3") && continue
    ins in PAPER || continue; push!(keep,i); push!(istr,ins)
end
dm=dm[keep,:]; bjd=Float64.(dm[:,1]); rv=Float64.(dm[:,2]); rverr=Float64.(dm[:,3])
inm=sort!(unique(istr)); imap=Dict(n=>i for (i,n) in enumerate(inm)); rv_inst=[imap[s] for s in istr]
let kc=trues(length(bjd))
    for id in unique(rv_inst); idx=findall(==(id),rv_inst); m=median(rv[idx]); thr=5*max(1.4826*median(abs.(rv[idx].-m)),1e-6)
        for k in idx; abs(rv[k]-m)>thr && (kc[k]=false); end; end
    global bjd=bjd[kc]; global rv=rv[kc]; global rverr=rverr[kc]; global rv_inst=rv_inst[kc]; global dm=dm[kc,:]
end
col=Dict(:bis=>(4,5),:fwhm=>(6,7),:halpha=>(10,11),:logrhk=>(12,13))
# RAW indicators, per-instrument centered only (missing → 0 = the centered mean,
# missing err → large). NO manual scaling — the scale-adaptive AD prior and the
# MAD-scaled AGP/floor priors handle the units automatically inside Nereus.
function lr(ch); cv,ce=col[ch]
    v=Float64[let x=dm[i,cv]; (x==""||(x isa AbstractString&&strip(x)=="")) ? NaN : Float64(x) end for i in 1:size(dm,1)]
    e=Float64[let x=dm[i,ce]; (x==""||(x isa AbstractString&&strip(x)=="")) ? NaN : Float64(x) end for i in 1:size(dm,1)]
    for id in unique(rv_inst); idx=findall(==(id),rv_inst); fin=filter(k->isfinite(v[k]),idx)
        isempty(fin) && (for k in idx; v[k]=0.0; e[k]=1e3; end; continue)
        μ=median(v[fin]); em=(fe=filter(k->isfinite(e[k])&&e[k]>0,idx); isempty(fe) ? 0.1 : median(e[fe]))
        for k in idx
            v[k]=isfinite(v[k]) ? v[k]-μ : 0.0
            e[k]=(isfinite(e[k])&&e[k]>0) ? e[k] : em
        end
    end; v,e; end
# INDICATORS env (comma-separated) selects which activity channels the menu
# uses (default: all four). Not every indicator correlates with the RVs on
# every target; a non-correlating channel is leak surface (BIS at P_rot/2
# absorbed ~4 m/s of planet-phase power here). Mirrors run_job's
# `noise_menu.indicators`.
chs=Symbol.(split(get(ENV,"INDICATORS","bis,fwhm,halpha,logrhk"),","))
indicators=Dict{String,Vector{Float64}}(); ierrs=Dict{String,Vector{Float64}}()
for ch in chs; vs,es=lr(ch); indicators[String(ch)]=vs; ierrs[String(ch)]=es; end

# ---- TESS LC: native cadence, windowed to b's transits (NEVER binned) -------
lc = load_tess_lc(LC_FILE)
half = 2 * DUR; nlc = length(lc.t)
nper = ceil(Int, (lc.t[end]-lc.t[1]) / P_REF) + 2
intr = falses(nlc)
for k in -1:(nper+1); tc=T0_REF+k*P_REF
    @inbounds for i in 1:nlc; abs(lc.t[i]-tc)<half && (intr[i]=true); end
end
oot=falses(nlc); let c=0; for i in 1:nlc; intr[i]&&continue; c+=1; oot[i]=(c%200==0); end; end
kp=intr .| oot
t_phot=lc.t[kp]; flux=lc.flux[kp]; flux_err=lc.flux_err[kp]; phot_inst=ones(Int,length(t_phot))
@printf("RV: %d pts / %d inst + 4 indicators | LC: %d → %d native pts (%d in-transit)\n",
        length(bjd), length(inm), nlc, length(t_phot), count(intr))

data = Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
              indicators=indicators, indicator_errs=ierrs,
              t_phot=t_phot, flux=flux, flux_err=flux_err, phot_inst=phot_inst)
ic = InstrumentConfig(rv=inm, pm=["TESS"])
menu = default_noise_menu(data; indicators=String.(chs),
    include_activity_gp = get(ENV,"INCLUDE_ACTIVITY_GP","1") != "0")
@printf("Menu: %d toggleable (+%d always-on)\n", length(menu.toggleable),
        length(menu.noise_models)-length(menu.toggleable))

rvmax=maximum(abs,rv)
pri=Dict{String,PriorSpec}(
    "P_k1"=>UniformPrior(P_REF-0.01,P_REF+0.01),   # let the TRANSIT constrain P, not a σ=1min peg
    "K_k1"=>UniformPrior(0.0,50.0),
    "sesinw_k1"=>UniformPrior(-1.0,1.0), "secosw_k1"=>UniformPrior(-1.0,1.0),
    "Tc_k1"=>NormalPrior(T0_REF,0.05,T0_REF-0.5,T0_REF+0.5),
    "b_k1"=>UniformPrior(0.0,1.0), "rr_k1"=>NormalPrior(0.031,0.005,0.005,0.10),
    "rho_s"=>NormalPrior(2.241,0.479,0.1,10.0),
    "offset_TESS"=>NormalPrior(0.0,1e-3,-5e-3,5e-3), "jitter_TESS"=>LogUniformPrior(1e-5,5e-3),
    "q1_TESS"=>UniformPrior(0.0,1.0), "q2_TESS"=>UniformPrior(0.0,1.0))
for n in inm; pri["gamma_$n"]=UniformPrior(-3rvmax,3rvmax); end
# Rotation period IS the (measured) stellar rotation period — a physical constant,
# not a free parameter. Anchor it; the GP can't be anything else.
pri["gp_period"]=NormalPrior(P_ROT,1.0,4.0,16.0)
# gp_act_period exists only when ActivityGP is in the menu
any(m->m isa ActivityGP, menu.toggleable) && (pri["gp_act_period"]=NormalPrior(P_ROT,1.0,4.0,16.0))

# EPRIOR=vines applies Jose's published eccentricity prior (Vines+2023):
# external N(0, 0.3^2) on the DERIVED e, per planet, on top of the flat
# sesinw/secosw parametrization.
extp = get(ENV,"ECC_PRIOR","") == "vines" ?
    [ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)] : ExternalPrior[]
isempty(extp) || println("external e prior: N(0, 0.3^2) on ecc (Vines+2023)")
params=Params(; max_kplanet=1, planet_modes=[RVPM], instruments=ic, data=data,
    M_s=M_S, R_s=R_S, parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true),
    priors=pri, noise_models=menu.noise_models, transdim_noise=true,
    external_priors=extp)
target=NereusTarget(params, data; unconstrained=false)
@printf("Free params: %d\n", n_unfrozen(params))
td=TransDimConfig(; max_kplanet=1, planets=false, noise=true,
    toggleable=menu.toggleable, noise_exclusion_groups=menu.exclusion_groups)

N_TEMPS=parse(Int,get(ENV,"N_TEMPS","12")); N_WALKERS=parse(Int,get(ENV,"N_WALKERS","100"))
N_STEPS=parse(Int,get(ENV,"N_STEPS","4000")); N_BURNIN=parse(Int,get(ENV,"N_BURNIN","2000")); SEED=parse(Int,get(ENV,"SEED","42"))
# COLD-DENSE explicit ladder.
#
# SIGMA_LOGL_RV is the measured standard deviation of the RV log-likelihood at
# beta=1 — NOT the joint one, because untemper_transit below holds the transit
# term outside the tempered target. Parallel-tempering swap acceptance between
# adjacent rungs goes as exp(-delta_beta * sigma_logL), so the cold-end spacing
# is set to 1/sigma and the ladder is built outward from there. Spacing beta
# uniformly instead is what collapses swap acceptance on a joint fit.
# The default geometric ladder puts beta_2 at 0.43 — with sigma(ll_RV)~20 the
# cold gap needs Delta-beta ~ 1/sigma ~ 0.05 (beta_2 ~ 0.95), and the Vousden
# adaptation provably cannot move it that far during burn-in (cumulative
# log-movement ceiling ~0.2x over 2000 steps). Seed the density explicitly:
# Delta-beta grows geometrically (r) from 1/sigma at the cold end down to
# beta_min; Vousden then only makes small interior corrections.
BETAS = haskey(ENV,"SIGMA_LOGL_RV") ? let σll=parse(Float64,ENV["SIGMA_LOGL_RV"]), d1=1/σll, r=1.10, βmin=1e-3
    d=Float64[]; s=0.0; k=0
    while s < 1-βmin; push!(d, d1*r^k); s += d[end]; k += 1; end
    b=vcat(1.0, 1.0 .- cumsum(d)); b[end]=βmin
    @printf("cold-dense ladder: %d rungs, beta_2=%.3f (sigma_ll=%.0f)\n", length(b), b[2], σll)
    b
end : nothing
BETAS !== nothing && (N_TEMPS=length(BETAS))
# MEASURE_SIGMA_LOGL_RV=<chains.nc>: report the number the cold-dense ladder
# needs, from a pilot, instead of guessing it.
#
# It must be the RV log-likelihood ALONE. The saved `lp` is the full log
# posterior and is dominated by the ~25k-point transit term (mean ~8.6e4 on this
# target), so its spread says nothing about the tempered problem — with
# untemper_transit the transit sits outside the tempered target entirely.
# Reported robustly (IQR/1.349) as well as by std, because a pilot chain still
# carries burn-in draws and the two differ by ~8x when it does.
if haskey(ENV, "MEASURE_SIGMA_LOGL_RV")
  let
    chp = ENV["MEASURE_SIGMA_LOGL_RV"]
    chM, _ = Nereus.load_chains(chp)
    cnM = names(chM, :parameters)
    nm  = params.config.noise_models
    lls = Float64[]; lls_tr = Float64[]
    ndraw = size(chM, 1) * size(chM, 3)
    draw_stride = max(1, ndraw ÷ 4000)          # cap the cost; 4k draws is plenty
    k = 0
    for c in 1:size(chM, 3), i in 1:size(chM, 1)
        k += 1; k % draw_stride == 0 || continue
        tds = TransDimState(; max_planets=1, n_noise=length(nm))
        tds.planet_active[1] = true; tds.n_planets_active = 1
        for j in 1:length(nm)
            col = Symbol("noise_active_$j")
            (col in cnM && Array(chM[col])[i, 1, c] > 0.5) && (tds.noise_active[j] = true)
        end
        th = Theta{Float64}(params; td=tds)
        for pn in params.layout.unfrozen_names
            Symbol(pn) in cnM && set_param!(th, pn, Array(chM[Symbol(pn)])[i, 1, c])
        end
        v  = try Nereus.rv_log_likelihood(th, data)      catch; NaN end
        vt = try Nereus.transit_log_likelihood(th, data) catch; NaN end
        (isfinite(v) && isfinite(vt)) && (push!(lls, v); push!(lls_tr, vt))
    end
    if isempty(lls)
        println("MEASURE_SIGMA_LOGL_RV: no finite draws"); exit(1)
    end
    sd  = std(lls)
    iqr = (quantile(lls, 0.75) - quantile(lls, 0.25)) / 1.349
    # Decomposed exactly, not by subtracting off `lp`. What sets swap acceptance
    # is the SPREAD of each term, not its magnitude: the transit mean is huge and
    # irrelevant, its sigma is what would cost rungs if it were tempered.
    tot  = lls .+ lls_tr
    rob(x) = (quantile(x, 0.75) - quantile(x, 0.25)) / 1.349
    @printf("  sigma(logL_transit) robust = %.2f   (mean %.1f)\n", rob(lls_tr), mean(lls_tr))
    @printf("  sigma(logL_total)   robust = %.2f\n", rob(tot))
    @printf("  untempering saves a factor %.2f in cold-end rung density\n", rob(tot)/iqr)
    @printf("sigma(logL_RV) over %d draws:  std=%.2f   robust(IQR/1.349)=%.2f\n",
            length(lls), sd, iqr)
    @printf("  -> SIGMA_LOGL_RV=%.1f   (robust value; cold-end delta_beta = 1/sigma = %.2e)\n",
            iqr, 1/iqr)
  end
  exit(0)
end

@printf("RVPM menu run: %d temps × %d walkers × %d+%d (seed=%d)\n",N_TEMPS,N_WALKERS,N_STEPS,N_BURNIN,SEED)
t0=time()
res=sample_transdim_ptemcee(target, data; td=td, n_temps=N_TEMPS, n_walkers=N_WALKERS, n_steps=N_STEPS,
    betas=BETAS,
    n_burnin=N_BURNIN, n_birth_tries=10, n_birth_refine=15, seed=SEED, show_progress=true, adapt_ladder=true,
    # Transit held in the UNTEMPERED reference: the 16k-point transit term dominates
    # σ(logL) (hence Δβ·σ, hence swap acceptance) but is identical across every
    # noise-menu member, so tempering it costs swaps and buys nothing. Valid here
    # because planets=false ⇒ L_transit is model-invariant. β=1 target unchanged.
    untemper_transit = parse(Bool, get(ENV, "UNTEMPER_TRANSIT", "true")))
@printf("done in %.1f min  logZ=%.2f\n", (time()-t0)/60, res.log_evidence)

ch=res.chains; cn=names(ch,:parameters)
outd=get(ENV,"OUTPUT_DIR","results/HD18599_menu_rvpm_bounded"); mkpath(outd)
save_chains(joinpath(outd,"chains.nc"), ch, params; data=data)
@printf("saved → %s/chains.nc\n", outd)

Kv=vec(Array(ch[:K_k1])); Pv=vec(Array(ch[:P_k1]))
@printf("\nPLANET b:  P=%.5f d  K=%.2f  16/50/84=%.2f/%.2f/%.2f m/s  frac(8<K<14)=%.2f  (published K≈11)\n",
        median(Pv), median(Kv), quantile(Kv,0.16), quantile(Kv,0.5), quantile(Kv,0.84), mean(8 .< Kv .< 14))
(:ecc in cn) && @printf("           e = %.3f ± %.3f\n", median(vec(Array(ch[:ecc]))), std(vec(Array(ch[:ecc]))))
# index-based: toggleable[i] is noise_models[i] (menu = vcat(toggleable, always_on)),
# so its column is noise_active_$i — no struct-equality lookup (Vector-field structs
# fall back to === and break findfirst(==)).
@printf("\nNOISE OCCUPANCY:\n")
for (i,m) in enumerate(menu.toggleable)
    acol=Symbol("noise_active_$i")
    p = acol in cn ? mean(vec(Array(ch[acol])).>0.5) : NaN
    tag = m isa ActivityDecorrelation ? (m.derivative ? "(FF′)" : "(lin)") : ""
    @printf("   P(%-16s%-6s)=%.3f\n", string(nameof(typeof(m))), tag, p)
end
# lean: only the (cheap) split occupancy plot. NO rv_phasefold (60k draws → the
# slow tail that looked like a hang). Diagnose K via bf-cluster on the saved chain.
pd=joinpath(outd,"plots"); mkpath(pd)
lbl=[string(nameof(typeof(m)))*(m isa ActivityDecorrelation ? (m.derivative ? "(FF′)" : "(lin)") : "") for m in menu.toggleable]
try; Nereus.plot_transdim_occupancy(ch, params; td=td, noise_labels=lbl, split=true,
        output=joinpath(pd,"occupancy.png")); println("✓ occupancy (split)"); catch e; println("occ FAIL: ",e); end
