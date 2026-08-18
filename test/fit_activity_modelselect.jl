# Real-target trans-dim planet ↔ activity-model selection (#48).
# Positive control: CoRoT-7 (b 0.854 d + c 3.69 d + activity).
# Disputed nulls:   HD 41248, Kapteyn — claimed planets later argued as
#                   rotation/activity; the sampler should adjudicate.
#
# Activity-model menu (mutually exclusive via noise_exclusion_groups):
#   AD  = ActivityDecorrelation (linear BIS/FWHM regression)
#   AGP = ActivityGP (joint Rajpaul GP on RV+BIS+FWHM)
#   AR  = ARModel(1), MA = MAModel(1)
# IndicatorFloor is ALWAYS ON (not toggleable) so non-AGP states score
# the SAME data (RV + indicator block) as AGP → occupancy is a valid P(M|D).
# Planets toggle freely (recovery on CoRoT-7, adjudication on the disputed).
# Default calibrated config (n_birth_tries=1/n_birth_refine=0 by sampler
# default) + adequate ladder. Smoke phase first (finite/distinct logL in
# every mode) before the sampler.
#
#   TARGET=corot7|hd41248|kapteyn julia --project=. -t8 test/fit_activity_modelselect.jl

using Nereus, DelimitedFiles, Printf, MCMCChains, Random
using Statistics: median, mean, std, quantile
import Dates, Profile

const TARGET = get(ENV, "TARGET", "corot7")
# Per-target config: file, M_s, R_s, P_rot (lit.), max_kplanet, P range, seeds.
# lc = TESS cleaned LC (transit) or nothing; ntrans = # leading RVPM (transiting)
# slots — CoRoT-7b transits (0.854 d), so slot 1 is RVPM and the BLS-informed
# birth pins it; CoRoT-7 c/d and the disputed targets don't transit (RV_ONLY).
const CFG = Dict(
    "corot7"   => (dir="CoRoT-7", file="corot7_rv.csv",  M_s=0.91, R_s=0.82,
                   P_rot=23.0, prot_lo=15.0, prot_hi=35.0, maxk=2,
                   P_lo=0.5, P_hi=50.0, seeds=[0.8535, 3.69],
                   lc="corot7_cleaned_lc.csv", ntrans=1),
    "hd41248"  => (dir="HD41248", file="hd41248_rv.csv", M_s=0.94, R_s=0.90,
                   P_rot=25.0, prot_lo=15.0, prot_hi=40.0, maxk=2,
                   P_lo=1.0, P_hi=400.0, seeds=[18.36, 25.6],
                   lc=nothing, ntrans=0),
    "kapteyn"  => (dir="Kapteyn", file="kapteyn_rv.csv", M_s=0.28, R_s=0.29,
                   P_rot=125.0, prot_lo=80.0, prot_hi=160.0, maxk=2,
                   P_lo=2.0, P_hi=400.0, seeds=[48.6, 121.5],
                   lc=nothing, ntrans=0),
)
haskey(CFG, TARGET) || error("unknown TARGET=$TARGET")
c = CFG[TARGET]
# AGP is ~100× slower/eval (dense joint GP). NO_AGP=1 drops it (+ its floor) →
# fast trans-dim over AD/AR/MA; AGP then compared via separate fixed-dim evidence.
const NO_AGP = get(ENV,"NO_AGP","0")=="1"
# AGP_ONLY=1 — Haywood route: ActivityGP as the (always-on) activity model, RV-only,
# planets trans-dim; the GP lifts the planets above rotation. NOPHOT=1 — drop the LC.
const AGP_ONLY = get(ENV,"AGP_ONLY","0")=="1"
const NOPHOT   = haskey(ENV,"NOPHOT")
# NOARMA=1 — drop the generic red-noise AR/MA from the menu (they structurally
# absorb weak planets), leaving only the physical activity models AD/AGP/GP-Rot.
const NOARMA   = get(ENV,"NOARMA","0")=="1"
const RV_FILE = joinpath(@__DIR__, "..", "..", "data", c.dir, c.file)

println("="^70)
println("Trans-dim planet↔activity model selection — $TARGET")
println("="^70)

# ---- load RV + indicators -------------------------------------------
# cols: bjd,rv,rv_err,instrument,bis,bis_err,fwhm,fwhm_err,s_index,s_index_err
raw = readdlm(RV_FILE, ',', Any, '\n'; header=true)
M = raw[1]
tonum(x) = (x === "" || (x isa AbstractString && strip(x)=="")) ? NaN : Float64(x)
bjd = Float64.(M[:,1]); rv = Float64.(M[:,2]); rv_err = Float64.(M[:,3])
inst_str = strip.(String.(M[:,4]))
inst_names = sort!(unique(inst_str)); inst_map = Dict(n=>i for (i,n) in enumerate(inst_names))
rv_inst = [inst_map[s] for s in inst_str]
bis = tonum.(M[:,5]); bis_e = tonum.(M[:,6]); fwhm = tonum.(M[:,7]); fwhm_e = tonum.(M[:,8])

# ---- per-instrument 5σ MAD clip (basic data hygiene) ----------------
let keep = trues(length(bjd))
    for id in unique(rv_inst)
        idx = findall(==(id), rv_inst); med = median(rv[idx])
        thr = 5*max(1.4826*median(abs.(rv[idx].-med)), 1e-6)
        for k in idx; abs(rv[k]-med)>thr && (keep[k]=false); end
    end
    nclip = count(.!keep)
    nclip>0 && @printf("clipped %d outliers\n", nclip)
    global bjd=bjd[keep]; global rv=rv[keep]; global rv_err=rv_err[keep]
    global rv_inst=rv_inst[keep]; global bis=bis[keep]; global bis_e=bis_e[keep]
    global fwhm=fwhm[keep]; global fwhm_e=fwhm_e[keep]
end
@printf("Loaded %d RVs / %d instruments: %s\n", length(bjd), length(inst_names), join(inst_names,", "))

# ---- indicator double-load: AGP-format (robust-standardized, with errs)
#      + AD-format (EMPEROR per-instrument normalized to RV range) -----
function agp_format(v, e)
    vs=copy(v); es=copy(e)
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); fin=filter(k->isfinite(vs[k]),idx)
        μ=isempty(fin) ? 0.0 : median(vs[fin])
        mad=isempty(fin) ? 1.0 : max(1.4826*median(abs.(vs[fin].-μ)),1e-12)
        fe=filter(k->isfinite(es[k])&&es[k]>0,idx)
        med_e=isempty(fe) ? 0.1 : median(es[fe])/mad
        for k in idx
            if isfinite(vs[k]); vs[k]=(vs[k]-μ)/mad
                es[k]=(isfinite(es[k])&&es[k]>0) ? es[k]/mad : med_e
            else; vs[k]=0.0; es[k]=1e3; end
        end
    end
    vs, es
end
function ad_format(v)
    w=copy(v); for i in eachindex(w); isfinite(w[i])||(w[i]=0.0); end
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); fin=filter(k->isfinite(v[k]),idx); isempty(fin)&&continue
        μ=mean(@view w[fin]); for k in fin; w[k]-=μ; end
        wmax=maximum(abs,@view w[fin]); wmax==0 && continue
        rvv=@view rv[idx]; rmax=maximum(abs, rvv.-mean(rvv)); rmax==0 && continue
        for k in fin; w[k]=w[k]/wmax*rmax; end
    end
    w
end
bis_c, bis_es = agp_format(bis, bis_e); fwhm_c, fwhm_es = agp_format(fwhm, fwhm_e)
indicators = Dict{String,Vector{Float64}}(
    "bis_AD"=>ad_format(bis), "fwhm_AD"=>ad_format(fwhm),
    "bis"=>bis_c, "fwhm"=>fwhm_c)
indicator_errs = Dict{String,Vector{Float64}}("bis"=>bis_es, "fwhm"=>fwhm_es)
# transit photometry (e.g. CoRoT-7b at 0.854 d) — pins the period/ephemeris the
# RV alone can't dig out from under the rotation; the BLS-informed birth uses it.
tph=Float64[]; fph=Float64[]; feph=Float64[]
const NT_TRANS = (c.lc !== nothing && !NOPHOT) ? c.ntrans : 0   # # of RVPM (transiting) slots
if c.lc !== nothing && !NOPHOT
    lc = load_tess_lc(joinpath(@__DIR__,"..","..","data",c.dir,c.lc))
    perm = sortperm(lc.t)
    tph=lc.t[perm]; fph=lc.flux[perm]; feph=lc.flux_err[perm]
    # PHOT_DAYS: keep only the first N days of photometry (one TESS sector is
    # plenty of transits to pin a short-period planet; the full multi-sector LC
    # is 200k+ points and makes the transit likelihood the per-eval bottleneck).
    if haskey(ENV,"PHOT_DAYS")
        m = tph .<= minimum(tph) + parse(Float64,ENV["PHOT_DAYS"])
        tph=tph[m]; fph=fph[m]; feph=feph[m]
    end
    @printf("Loaded %d photometry points from %s\n", length(tph), c.lc)
end
data = Data(; t_rv=bjd, rv=rv, rv_err=rv_err, rv_inst=rv_inst,
            indicators=indicators, indicator_errs=indicator_errs,
            t_phot=tph, flux=fph, flux_err=feph)
ic = c.lc === nothing ? InstrumentConfig(rv=inst_names) :
                        InstrumentConfig(rv=inst_names, pm=["TESS"])

# ---- activity-model menu + always-on indicator floor ----------------
ad    = ActivityDecorrelation(indicators=["bis_AD","fwhm_AD"])
agp   = ActivityGP(channels=[:bis,:fwhm], marginalize_indicators=false)
gprot = CeleriteRotation(channel=:rv)            # QP GP rotation on the RV
ar    = ARModel(order=1); ma = MAModel(order=1)
flr   = IndicatorFloor(channels=[:bis,:fwhm])

rv_max = maximum(abs, rv .- median(rv))
priors = Dict{String,PriorSpec}()
for k in 1:c.maxk
    priors["P_k$k"]=LogUniformPrior(c.P_lo, c.P_hi)
    priors["K_k$k"]=UniformPrior(0.0, 50.0)
    priors["sesinw_k$k"]=UniformPrior(-0.9,0.9); priors["secosw_k$k"]=UniformPrior(-0.9,0.9)
    priors["Mo_k$k"]=UniformPrior(0.0,2π)
end
for k in 1:NT_TRANS   # transit geometry for the RVPM (transiting) slots
    priors["b_k$k"]=UniformPrior(0.0,1.0); priors["rr_k$k"]=UniformPrior(0.005,0.10)
end
# SEED=1 — "with priors" mode: tight period priors at the known ephemerides
# (Jose's realistic workflow). Bypasses the BLS blind-search, which misses the
# 0.854-d CoRoT-7b transit in the 4000-ppm TESS noise.
if get(ENV,"SEED","0")=="1"
    for (k,Ps) in enumerate(c.seeds)
        k <= c.maxk || break
        priors["P_k$k"]=NormalPrior(Ps, 0.005*Ps, 0.7*Ps, 1.3*Ps)
    end
    @printf("SEED on: tight P priors at %s\n", string(c.seeds))
end
# BRACKET=1 — FLAT uniform period prior ±BRACKET_FRAC around each known period.
# Confines the planet search to the planet's neighbourhood (so it's found, and
# the rotation/aliases can't steal the slot) WITHOUT feeding the period back: the
# posterior period inside the flat window is purely data-driven. Purpose: isolate
# the NOISE-MODEL selection by removing planet-finding as a confound.
if get(ENV,"BRACKET","0")=="1"
    f = parse(Float64, get(ENV,"BRACKET_FRAC","0.15"))
    for (k,Ps) in enumerate(c.seeds)
        k <= c.maxk || break
        priors["P_k$k"]=UniformPrior(Ps*(1-f), Ps*(1+f))
    end
    @printf("BRACKET on: flat P_k uniform ±%.0f%% around %s\n", 100f, string(c.seeds))
end
for n in inst_names
    priors["gamma_$n"]=UniformPrior(median(rv[rv_inst.==inst_map[n]])-5rv_max,
                                    median(rv[rv_inst.==inst_map[n]])+5rv_max)
end
if !NO_AGP   # AGP present (full menu or AGP-only): rotation-period prior + λ_p floor
    priors["gp_act_period"]=NormalPrior(c.P_rot, 0.3*c.P_rot, c.prot_lo, c.prot_hi)
    priors["gp_act_lambda_p"]=UniformPrior(0.25, 2.0)   # λ_p floor ≥0.25 (degeneracy memo)
end
if !NO_AGP && !AGP_ONLY   # GP-Rot present (full menu): rotation-period + coherence priors
    priors["gp_period"]=NormalPrior(c.P_rot, 0.3*c.P_rot, c.prot_lo, c.prot_hi)
    # Coherence floor: Q_sec = 0.5+Q0 ≳ 5.5 keeps the SHO power NARROW-BAND around
    # the rotation frequency. Low Q → broad spectrum with high-freq power → the GP
    # interpolates sub-rotation planet signals. Flooring Q stops the devouring.
    priors["gp_Q0"]=LogUniformPrior(5.0, 500.0)
end

if AGP_ONLY
    nmods = NoiseModel[agp]; toggle = NoiseModel[]; excl = Vector{NoiseModel}[]
    menu = "AGP-only (always on, Haywood route)"
elseif NO_AGP
    nmods = NoiseModel[ad,ar,ma]; toggle = NoiseModel[ad,ar,ma]; excl=[[ad,ar,ma]]
    menu = "AD/AR/MA (no AGP)"
else
    # AD / AGP / GP-Rot / ARMA, mutually exclusive — but AR+MA COEXIST (=ARMA).
    # Two overlapping groups give: AD,AGP,GPRot pairwise-exclusive AND each
    # exclusive with AR and with MA, while AR and MA (sharing no group) coexist.
    # (CovarianceNoise typing already makes AGP⊥GPRot and GP⊥AR/MA; these groups
    # add AD's exclusivity without forcing AR⊥MA.)
    if NOARMA   # only the physical activity models, mutually exclusive
        nmods  = NoiseModel[ad,agp,gprot,flr]
        toggle = NoiseModel[ad,agp,gprot]
        excl   = [[ad,agp,gprot]]
        menu   = "AD/AGP/GPRot (no ARMA, excl.) + floor(on)"
    else
        nmods  = NoiseModel[ad,agp,gprot,ar,ma,flr]
        toggle = NoiseModel[ad,agp,gprot,ar,ma]
        excl   = [[ad,agp,gprot,ar],[ad,agp,gprot,ma]]
        menu   = "AD/AGP/GPRot/ARMA (excl., AR+MA coexist) + floor(on)"
    end
end
pmodes = PlanetDataSources[fill(RVPM, NT_TRANS); fill(RV_ONLY, c.maxk - NT_TRANS)]
params = Params(; max_kplanet=c.maxk, planet_modes=pmodes,
    instruments=ic, data=data, M_s=c.M_s, R_s=c.R_s,
    parametrization=ParametrizationConfig(time=:Mo), priors=priors,
    noise_models=nmods, transdim_noise=true)
@printf("Params: %d unfrozen; menu = %s; planets 0..%d\n", n_unfrozen(params), menu, c.maxk)
for (nm,ps) in zip(params.layout.unfrozen_names, params.layout.unfrozen_priors)
    occursin("period", lowercase(nm)) && @printf("  PRIOR %-16s %s  bounds=%s\n", nm, typeof(ps).name.name, string(round.(bounds(ps),digits=2)))
end

td = TransDimConfig(; max_kplanet=c.maxk, planets=true, noise=!isempty(toggle),
    toggleable=toggle, noise_exclusion_groups=excl)

# ---- SMOKE: finite + distinct logL in each activity mode -------------
function build_theta(mask)
    tds = TransDimState(; max_planets=c.maxk, n_noise=length(params.config.noise_models))
    tds.planet_active[1]=true; tds.n_planets_active=1
    for (i,on) in enumerate(mask); on && activate_noise!(tds,i); end
    th = Theta{Float64}(params; td=tds)
    for (nm,ps) in zip(params.layout.unfrozen_names, params.layout.unfrozen_priors)
        lo,hi=bounds(ps); set_param!(th,nm,0.5*(lo+hi))
    end
    set_param!(th,"P_k1",c.seeds[1]); set_param!(th,"K_k1",5.0); set_param!(th,"Mo_k1",1.0)
    for n in inst_names; set_param!(th,"gamma_$n",median(rv[rv_inst.==inst_map[n]])); set_param!(th,"sigma_$n",3.0); end
    th
end
function smoke(mask)
    th = build_theta(mask)
    rvll = try Nereus.rv_log_likelihood(th, data) catch; NaN end
    full = try rvll + Nereus.transit_log_likelihood(th, data) catch; NaN end
    return (rvll, full)
end
# noise_models order = [ad,agp,ar,ma,flr]; floor always on (non-toggleable).
# FULL = rv + transit(=indicator-floor block). For a VALID occupancy, the
# non-AGP states' FULL logL must include the floor'd indicator block, putting
# them on the same data scale as AGP (both ≈ RV + 2 indicator channels).
@printf("\nSMOKE logL  (rv | full=rv+floor/indicator block):\n")
# Full-menu smoke (6 noise slots incl. floor). Only for the full 6-slot menu.
if !NO_AGP && !AGP_ONLY && !NOARMA
    # noise order [ad,agp,gprot,ar,ma,flr]; floor (index 6) ON in every mode.
    for (lbl,mask) in (("AD",[1,0,0,0,0,1]),("AGP",[0,1,0,0,0,1]),("GPRot",[0,0,1,0,0,1]),
                       ("ARMA",[0,0,0,1,1,1]),("white",[0,0,0,0,0,1]),("AGP(noflr)",[0,1,0,0,0,0]))
        r,f = smoke(Bool.(mask)); @printf("  %-6s rv = %11.3f   full = %11.3f\n", lbl, r, f)
    end
end

# ---- VERIFY: load a saved chain and diagnose the fit (no sampling) ---
# Is the recovered fit physically correct, or is the rotation GP devouring
# the planet? Two crux diagnostics per planet-count state:
#   (1) gp_period posterior — does the rotation GP sit at the true P_rot, or
#       has it locked onto a planet alias (0.854 / 3.68 d)?
#   (2) residual GLS — in the Np=0 (GP-only) state, does coherent planet power
#       SURVIVE at the known periods (GP not absorbing → coin-flip is the
#       sampler under-committing; transit would nail it) or is it whitened
#       away (GP genuinely absorbs → RV-only honestly can't separate them)?
if haskey(ENV,"VERIFY")
    vpath = ENV["VERIFY"]
    @printf("\nVERIFY: loading chain %s\n", vpath)
    ch, _ = Nereus.load_chains(vpath)
    np = vec(Array(ch[:n_planets]))
    cnames = Set(names(ch,:parameters))
    @printf("P(Np=k):"); for k in 0:c.maxk; @printf(" k=%d:%.3f", k, mean(np.==k)); end; println()
    vlbl(nm) = nm isa ActivityDecorrelation ? "AD" : nm isa ActivityGP ? "AGP" :
               nm isa ARModel ? "AR" : nm isa MAModel ? "MA" : string(nameof(typeof(nm)))
    for (i,nm) in enumerate(toggle)
        col=Symbol("noise_active_$i")
        col in cnames && @printf("  P(%s)=%.3f\n", vlbl(nm), mean(vec(Array(ch[col])).>0.5))
    end
    @printf("\nGP hyperparameters (true P_rot ≈ %.1f d):\n", c.P_rot)
    for gpn in ("gp_period","gp_Q0","gp_sigma","gp_act_period","gp_act_lambda_p")
        s=Symbol(gpn)
        s in cnames && (v=vec(Array(ch[s])); @printf("  %-16s med=%.4f  [%.4f,%.4f]\n",
            gpn, median(v), quantile(v,.16), quantile(v,.84)))
    end
    function theta_at_np(tnp)
        idx = findall(np .== tnp); isempty(idx) && return nothing
        tds = Nereus.winning_td_state(ch, params, idx, tnp)
        th  = Theta{Float64}(params; td=tds)
        Nereus.set_theta_best_lp!(th, ch, params, idx)
        (th, idx)
    end
    powat(pg,Pt) = (i=argmin(abs.(pg.periods .- Pt)); pg.power[i])
    for tnp in 0:c.maxk
        r = theta_at_np(tnp)
        r === nothing && (@printf("\n[Np=%d] no samples\n",tnp); continue)
        th, idx = r
        preds,_ = Nereus.rv_predictions(th, data); resid = data.rv .- preds
        gp = try Nereus.channel_gp_mean_at(th, resid, data.rv_err.^2, data.t_rv,
                 data.t_rv, data.rv_inst, :rv; data=data)
             catch e; @warn "GP mean failed" exception=e; zeros(length(resid)) end
        cleaned = resid .- gp
        @printf("\n[Np=%d, n=%d]  rms(mean-only resid)=%.3f  rms(GP-cleaned)=%.3f m/s\n",
            tnp, length(idx), std(resid), std(cleaned))
        if tnp >= 1
            for k in 1:tnp
                @printf("   planet k=%d: P=%.4f  K=%.3f\n", k,
                    Nereus.planet_P(th,k), Nereus.planet_K(th,k))
            end
        end
        for (lbl,y) in (("mean-only",resid),("GP-cleaned",cleaned))
            pg = Nereus.gls_periodogram(data.t_rv, y, data.rv_err; period_min=0.5, period_max=50.0)
            f1 = pg.fap_thresholds[findfirst(==(0.01), pg.fap_levels)]
            @printf("  GLS %-9s peaks:", lbl)
            for pk in pg.peaks[1:min(4,length(pg.peaks))]
                @printf(" %.4fd(z=%.2f,fap=%.1e)", pk.period, pk.power, pk.fap); end
            println()
            for Pt in c.seeds[1:min(c.maxk,length(c.seeds))]
                pw=powat(pg,Pt)
                @printf("     @%.4fd → z=%.3f  (1%% thr=%.3f) %s\n", Pt, pw, f1,
                    pw>f1 ? "ABOVE FAP" : "below")
            end
        end
    end
    # ---------- verification plots ----------
    rdir = dirname(vpath)
    pdf  = haskey(ENV,"PDF")   # PDF=1 → also emit .pdf alongside every .png
    @printf("\nplots → %s/  (pdf=%s)\n", rdir, pdf)
    # rebuild a Chains restricted to a draw-mask (keeps every column incl.
    # n_planets / noise_active_* / lp) so plot_rv_*'s modal-Np conditioning
    # lands on the planet-bearing state we want to inspect.
    function subset_chain(mask)
        alln = names(ch)
        M = reduce(hcat, [vec(Array(ch[n]))[mask] for n in alln])
        MCMCChains.Chains(M, alln)
    end
    # (0) trans-dim occupancy: what the model prefers across dimensions —
    #     P(Nₚ|D), marginal noise-model P(active|D), and the Nₚ trace.
    try
        Nereus.plot_transdim_occupancy(ch, params; td=td, save_pdf=pdf,
            output=joinpath(rdir,"transdim_occupancy.png"))
        @printf("  ✓ transdim_occupancy.png\n")
    catch e; @warn "transdim occupancy plot failed" exception=e end
    # (1) GP-devours-planet proof: residual GLS in the Np=0 (GP-only) winning
    #     state — mean model only vs GP mean subtracted. Planet peaks at
    #     0.854/3.69 d sit above FAP in the first and vanish in the second.
    try
        th,_ = theta_at_np(0)
        preds,_ = Nereus.rv_predictions(th, data); resid = data.rv .- preds
        gp = try Nereus.channel_gp_mean_at(th, resid, data.rv_err.^2, data.t_rv,
                 data.t_rv, data.rv_inst, :rv; data=data) catch; zeros(length(resid)) end
        cleaned = resid .- gp
        pg_raw = Nereus.gls_periodogram(data.t_rv, resid,   data.rv_err; period_min=0.5, period_max=50.0)
        pg_cln = Nereus.gls_periodogram(data.t_rv, cleaned, data.rv_err; period_min=0.5, period_max=50.0)
        Nereus.plot_periodogram(pg_raw; filename=joinpath(rdir,"gls_raw.png"), save_pdf=pdf,
            channel_label="RV resid · GP-only state · mean model only", max_peak_labels=6)
        Nereus.plot_periodogram(pg_cln; filename=joinpath(rdir,"gls_gpcleaned.png"), save_pdf=pdf,
            channel_label="RV resid · GP-only state · GP mean subtracted", max_peak_labels=6)
        @printf("  ✓ gls_raw.png  gls_gpcleaned.png\n")
    catch e; @warn "residual GLS plots failed" exception=e end
    # plot helper: render to a FLAT png in rdir (the plot fns nest output/models/
    # — capture the returned fig and save it ourselves to avoid the folder sprawl).
    saveflat(fig, name) = (Nereus._save_plot(joinpath(rdir, name), fig; px_per_unit=3, save_pdf=pdf);
                           @printf("  ✓ %s\n", name))
    # (2) b fold (P≈0.854 d) — _raw shows the coherent Keplerian under rotation;
    #     _gp shows it after GP subtraction. Sub-chain = planet-bearing (modal_np≥1).
    if any(np .>= 1)
        s1 = subset_chain(np .>= 1)
        for (tag,sg) in (("raw",false),("gp",true))
            try saveflat(Nereus.plot_rv_phasefold(s1, params, data; planet=1, subtract_gp=sg),
                         "fold_b_$tag.png")
            catch e; @warn "fold b ($tag) failed" exception=e end
        end
    end
    # (3) c fold (P≈3.69 d) — needs the Np=2 state.
    if any(np .== 2)
        s2 = subset_chain(np .== 2)
        for (tag,sg) in (("raw",false),("gp",true))
            try saveflat(Nereus.plot_rv_phasefold(s2, params, data; planet=2, subtract_gp=sg),
                         "fold_c_$tag.png")
            catch e; @warn "fold c ($tag) failed" exception=e end
        end
    end
    # (4) RV timeseries in the planet+GP state (shows the near-interpolator GP).
    try saveflat(Nereus.plot_rv_timeseries(subset_chain(np .>= 1), params, data), "timeseries.png")
    catch e; @warn "timeseries plot failed" exception=e end
    exit(0)
end

# ---- sampler --------------------------------------------------------
const STAMP = Dates.format(Dates.now(),"yyyymmdd_HHMMSS")
const OUT = joinpath(@__DIR__,"..","results","$(TARGET)_modelselect_$STAMP"); mkpath(OUT)
NT=parse(Int,get(ENV,"NTEMPS","20")); NW=parse(Int,get(ENV,"NWALKERS","128"))
NS=parse(Int,get(ENV,"NSTEPS","30000")); NB=parse(Int,get(ENV,"NBURNIN","8000")); TH=parse(Int,get(ENV,"NTHIN","8"))
# Recovery aids (multi-try births + newborn refinement). Default OFF (calibrated)
# per the truncation-fix finding; turn ON (e.g. 8/15) for needle recovery like
# CoRoT-7b. They trade some null calibration for detection sensitivity.
NTRIES=parse(Int,get(ENV,"NTRIES","1")); NREFINE=parse(Int,get(ENV,"NREFINE","0"))
# Informed births (BLS-on-photometry + LS-on-RV, cached, cold-side) are the
# planet-FINDER. Default ON — without them births are random and nothing is found.
INFORMED=parse(Float64,get(ENV,"INFORMED_FRAC","0.5"))
if get(ENV,"SMOKE_ONLY","0")=="1"; println("\nSMOKE_ONLY — stop before sampler."); exit(0); end
if get(ENV,"PROFILE","0")=="1"
    import Profile
    tgt = NereusTarget(params,data)
    @printf("PROFILE: warmup…\n")
    sample_transdim_ptemcee(tgt, data; td=td, n_temps=4, n_walkers=16, n_steps=5,
        n_burnin=0, n_birth_tries=NTRIES, n_birth_refine=NREFINE, seed=1, show_progress=false)
    Profile.clear(); Profile.init(n=10^7, delay=0.0003)
    @printf("PROFILE: profiling 60 steps…\n")
    Profile.@profile sample_transdim_ptemcee(tgt, data; td=td, n_temps=8, n_walkers=32,
        n_steps=60, n_burnin=0, n_birth_tries=NTRIES, n_birth_refine=NREFINE, seed=2, show_progress=false)
    Profile.print(format=:flat, sortedby=:count, mincount=30, maxdepth=40)
    exit(0)
end
@printf("\nsample_transdim_ptemcee: %d temps × %d walkers × %d steps + %d burnin (tries=%d refine=%d) → %s\n", NT,NW,NS,NB,NTRIES,NREFINE,OUT)
t0=time()
res = sample_transdim_ptemcee(NereusTarget(params,data), data; td=td,
    n_temps=NT, n_walkers=NW, n_steps=NS, n_burnin=NB, thin=TH,
    n_birth_tries=NTRIES, n_birth_refine=NREFINE, informed_birth_fraction=INFORMED,
    seed=42, show_progress=true)
@printf("done in %.1f min   logZ=%.2f\n", (time()-t0)/60, res.log_evidence)
ch=res.chains
np=vec(Array(ch[:n_planets]))
@printf("\nP(Np=k):"); for k in 0:c.maxk; @printf(" k=%d:%.3f", k, mean(np.==k)); end; println()
lblof(nm) = nm isa ActivityDecorrelation ? "AD" : nm isa ActivityGP ? "AGP" :
            nm isa ARModel ? "AR" : nm isa MAModel ? "MA" : string(nameof(typeof(nm)))
for (i,nm) in enumerate(toggle)   # noise_active_$i indexes nmods; toggle matches its order
    col=Symbol("noise_active_$i")
    col in names(ch,:parameters) && @printf("  P(%s)=%.3f\n", lblof(nm), mean(vec(Array(ch[col])).>0.5))
end
for k in 1:c.maxk
    pc=Symbol("P_k$k"); kc=Symbol("K_k$k")
    if pc in names(ch,:parameters)
        P=vec(Array(ch[pc])); K=vec(Array(ch[kc])); sel=np.>=k
        any(sel) && @printf("  k=%d: P=%.4f [%.4f,%.4f]  K=%.2f [%.2f,%.2f]\n", k,
            median(P[sel]),quantile(P[sel],.16),quantile(P[sel],.84),
            median(K[sel]),quantile(K[sel],.16),quantile(K[sel],.84))
    end
end
save_chains(joinpath(OUT,"chains.nc"), ch, params; data=data)
@printf("\nartifacts → %s\n", OUT)
