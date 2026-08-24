#!/usr/bin/env julia
# HD 18599 — TRANS-DIMENSIONAL noise-model SELECTION in ONE chain.
# sample_transdim_ptemcee toggles among the activity/noise models and
# returns the model posterior P(M | RV, indicators). ActivityGP is
# included on an equal footing via IndicatorFloor (always-on white model
# for the indicator channels), so every model state scores the SAME data
# — AGP-active: p(RV,y_I|GP); AGP-inactive: p(RV|model)·p(y_I|floor).
#
# This is the deliverable: NOT separate fixed-dim runs — ONE trans-dim
# chain doing the selection. Models toggled (mutually exclusive):
#   AD, ActivityGP, CeleriteRotation, MA(1), AR(1)
# (a tractable core menu; extend via $MODELS). Planet fixed (planets=false).
#
# Run: julia --project=. -t 10 Nereus.jl/test/hd18599_transdim_noise_select.jl

using Nereus, MCMCChains, DelimitedFiles, Statistics, Printf, Random
using LinearAlgebra: BLAS
# Outer Julia threading + multithreaded OpenBLAS = oversubscription
# (10 Julia threads × 12 BLAS = 120 workers on 12 cores) → threads block
# on BLAS internal locks, ~30% CPU, 2.65× SLOWER (measured). The
# covariance matrices here are tiny (≤456²) so BLAS threading never
# helped anyway. One BLAS thread per Julia thread.
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
let kc=trues(length(bjd))
    for id in unique(rv_inst)
        idx=findall(==(id),rv_inst); m=median(rv[idx])
        thr=5*max(1.4826*median(abs.(rv[idx].-m)),1e-6)
        for k in idx; abs(rv[k]-m)>thr && (kc[k]=false); end
    end
    global bjd=bjd[kc]; global rv=rv[kc]; global rverr=rverr[kc]
    global rv_inst=rv_inst[kc]; global dm=dm[kc,:]
end
col=Dict(:bis=>(4,5),:fwhm=>(6,7),:halpha=>(10,11),:logrhk=>(12,13))
# raw centered (AGP/floor data) + EMPEROR-normalized (AD regressors)
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
for ch in chs
    vs,es = load_raw(ch); indicators[String(ch)]=vs; ierrs[String(ch)]=es
end
adkey=Dict(:bis=>"bisector_span",:fwhm=>"fwhm_AD",:halpha=>"halpha_AD",:logrhk=>"log_rhk_AD")
for ch in chs; indicators[adkey[ch]]=load_adfmt(ch); end
# RVPM=1: model planet b as RV+transit. b transits (P,Tc,dur from ephemeris.txt);
# the transit pins P/Tc/geometry and constrains e/ω, which RV-only leaves free.
# Window the LC to ±PAD·dur around b's transits — native-cadence masking (never
# binning), to keep the O(N_phot) transit likelihood tractable on the 228k-pt LC.
const USE_PHOT = haskey(ENV, "RVPM")
if USE_PHOT
    Tc_b = 2458354.5857470357; dur_b = 0.07408203703703704
    # The cleaned TESS light curve is a 11 MB product that does not live in the
    # package. HD18599_LC points at it; the default is a sibling data/ directory
    # next to the repo. The old path resolved to <repo>/../../data, which after
    # the rename landed outside any checkout and failed only once the RVPM
    # branch was actually taken.
    lcpath = get(ENV, "HD18599_LC",
                 normpath(joinpath(@__DIR__, "..", "..", "data", "HD18599",
                                   "HD18599_cleaned_lc.csv")))
    isfile(lcpath) || error("HD 18599 light curve not found at $lcpath — " *
                            "set HD18599_LC to the cleaned TESS LC csv")
    lc = load_tess_lc(lcpath)
    pad = parse(Float64, get(ENV, "PAD", "3.0"))
    w = window_to_transits(lc.t, lc.flux, lc.flux_err, [P_REF], [Tc_b], [dur_b]; pad=pad)
    @printf("RVPM: windowed LC %d → %d phot pts (±%.1f·dur around b)\n",
            length(lc.t), length(w.t), pad)
    global data = Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
                       indicators=indicators, indicator_errs=ierrs,
                       t_phot=w.t, flux=w.flux, flux_err=w.flux_err)
    global ic = InstrumentConfig(rv=inst_names, pm=["TESS"])
else
    global data = Data(; t_rv=bjd, rv=rv, rv_err=rverr, rv_inst=rv_inst,
                       indicators=indicators, indicator_errs=ierrs)
    global ic = InstrumentConfig(rv=inst_names)
end
@printf("Loaded %d RVs, %d instruments + 4 indicator channels  (RVPM=%s)\n",
        length(bjd), length(inst_names), USE_PHOT)

rvmax=maximum(abs,rv)
pri=Dict{String,PriorSpec}(
    "P_k1"=>NormalPrior(P_REF,1e-3,P_REF-0.01,P_REF+0.01), "K_k1"=>UniformPrior(0.0,50.0),
    "sesinw_k1"=>UniformPrior(-1.0,1.0), "secosw_k1"=>UniformPrior(-1.0,1.0), "Mo_k1"=>UniformPrior(0.0,2π))
if USE_PHOT   # transit geometry for the RVPM planet
    pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.10)
end
for n in inst_names; pri["gamma_$n"]=UniformPrior(-3rvmax,3rvmax); end
pri["gp_period"]=NormalPrior(P_ROT,1.0,4.0,16.0)            # CeleriteRotation
pri["gp_act_period"]=NormalPrior(P_ROT,1.0,4.0,16.0)        # AGP

ad  = ActivityDecorrelation(indicators=["bisector_span","fwhm_AD","halpha_AD","log_rhk_AD"])
agp = ActivityGP(channels=chs)                              # joint mode
rot = CeleriteRotation(channel=:rv)
ma  = MAModel(order=1); ar = ARModel(order=1)
flr = IndicatorFloor(channels=chs, kernel=:qp)              # always-on QP-GP indicator floor
                                                            # (fair AD↔AGP: white floor strawman
                                                            #  handed AGP ~82 nats of indicator
                                                            #  correlation for free → spurious 15%)
toggle = NoiseModel[ad, agp, rot, ma, ar]
params=Params(; max_kplanet=1, planet_modes=(USE_PHOT ? [RVPM] : [RV_ONLY]),
    instruments=ic, data=data,
    M_s=M_S, R_s=R_S, parametrization=ParametrizationConfig(time=:Mo),
    priors=pri, noise_models=[ad, agp, rot, ma, ar, flr], transdim_noise=true,
    # Eccentricity prior: uniform sesinw/secosw gives p(e)∝e, which rails e→1
    # for a sparse, activity-contaminated RV (was 40% of draws at e>0.5, max
    # 1.0 — wrecking the phase-fold band). Half-Normal(0,0.3) on the derived e
    # (codebase convention) pins it near the published ~0.25.
    external_priors=[ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)])
target=NereusTarget(params, data; unconstrained=false)
@printf("Free params: %d  | toggling %d models + always-on IndicatorFloor\n",
        n_unfrozen(params), length(toggle))

# mutually-exclusive selection: at most ONE RV-noise model active (none = white)
td=TransDimConfig(; max_kplanet=1, planets=false, noise=true,
    toggleable=toggle, noise_exclusion_groups=[toggle])

# CHECK_BOUNDS=<chains.nc>: reconstruct an out-of-bounds draw's θ and recompute
# its log_prior — tells desync (stored lp ≠ θ's actual prior) from bound-gap.
if haskey(ENV,"CHECK_BOUNDS")
    chp=ENV["CHECK_BOUNDS"]; chC,_=Nereus.load_chains(chp); cn=names(chC,:parameters)
    ss=vec(Array(chC[:sesinw_k1])); lpc=vec(Array(chC[:lp]))
    i=findfirst(x->abs(x)>1, ss); i===nothing && (println("no OOB draw"); exit(0))
    npv = :n_planets in cn ? Int(vec(Array(chC[:n_planets]))[i]) : 1
    tds=TransDimState(; max_planets=1, n_noise=length(params.config.noise_models))
    for k in 1:npv; tds.planet_active[k]=true; end; tds.n_planets_active=npv
    for j in 1:length(params.config.noise_models)
        c=Symbol("noise_active_$j"); (c in cn && vec(Array(chC[c]))[i]>0.5) && (tds.noise_active[j]=true)
    end
    th=Theta{Float64}(params; td=tds)
    for nm in params.layout.unfrozen_names
        Symbol(nm) in cn && set_param!(th, nm, vec(Array(chC[Symbol(nm)]))[i])
    end
    e,_ = Nereus.planet_e_w(th,1)
    @printf("OOB draw %d: stored sesinw_k1=%.3f  stored lp=%.3f  n_planets=%d  derived e=%.4f\n",
            i, ss[i], lpc[i], npv, e)
    @printf("  recomputed log_prior(θ) = %.4f   (−Inf ⇒ stored (θ,lp) DESYNCED)\n", Nereus.log_prior(th))
    exit(0)
end

# PLOT_FROM=<chains.nc>: emit the FULL organized plot tree (same layout as
# run_job): plots/{models,transdim,posteriors/{raw,parameters,histograms},traces}.
# FOLD_ONLY iterates just the fold. CREDMASS tunes the fold band pool.
if haskey(ENV,"PLOT_FROM")
    chp = ENV["PLOT_FROM"]; pdir = joinpath(dirname(chp), "plots"); mkpath(pdir)
    chP,_ = Nereus.load_chains(chp)
    fold_only = haskey(ENV,"FOLD_ONLY")
    cm = parse(Float64, get(ENV,"CREDMASS","0.85"))
    sp = haskey(ENV,"SAVE_PDF")
    # plots/models/RV_phasefold_P1  (GP-subtracted default; named by the fn)
    try
        fig = Nereus.plot_rv_phasefold(chP, params, data; planet=1, credmass=cm)
        mkpath(joinpath(pdir,"models"))
        Nereus._save_plot(joinpath(pdir,"models","RV_phasefold_P1.png"), fig; px_per_unit=3, save_pdf=sp)
        @printf("✓ models/RV_phasefold_P1.png (credmass=%.2f)\n", cm)
    catch e; @warn "fold failed" exception=e end
    # plots/models/rv_timeseries.png
    try
        Nereus.plot_rv_timeseries(chP, params, data; output=pdir, save_pdf=sp)
        @printf("✓ models/rv_timeseries.png\n")
    catch e; @warn "timeseries failed" exception=e end
    if !fold_only
        # plots/transdim/occupancy.png  (2×2)
        Nereus.plot_transdim_occupancy(chP;
            noise_labels=["AD","AGP","GP-Rot","MA(1)","AR(1)"], save_pdf=sp,
            output=joinpath(pdir,"transdim","occupancy.png"))
        @printf("✓ transdim/occupancy.png\n")
        # ALL params (no planet filter) → posteriors/{raw,parameters,histograms} + traces/
        Nereus.plot_posteriors_raw(chP, params; output=pdir, save_pdf=sp)
        Nereus.plot_posteriors_parameters(chP, params; output=pdir, save_pdf=sp)
        Nereus.plot_posteriors_histograms(chP, params; output=pdir, save_pdf=sp)
        Nereus.plot_traces_grouped(chP, params; output=pdir, save_pdf=sp, n_walkers=160)
        @printf("✓ posteriors/{raw,parameters,histograms} (all params) + traces/\n")
    end
    @printf("PLOT_FROM done → %s\n", pdir); exit(0)
end

# SCITAB=<chains.nc>: build the model-conditioned, unit-tagged fitted science
# table and dump it as JSON (chunk 1 of the output-package build).
if haskey(ENV,"SCITAB")
    chP,_ = Nereus.load_chains(ENV["SCITAB"])
    fit_e, fit_c = Nereus.science_fitted(chP, params)
    der_e, der_c = Nereus.science_derived(chP, params; T_eff=5060.0)  # HD18599 K2 dwarf
    @printf("\n--- FITTED (%d params, conditioned on %s, Np=%d) ---\n",
            length(fit_e), join(fit_c["active_noise_models"],","), fit_c["n_planets"])
    for (k,v) in fit_e
        @printf("  %-22s = %12.5g  -%.4g / +%.4g  [%.4g, %.4g] %s\n",
                k, v["value"], v["err_lo"], v["err_hi"], v["ci3"][1], v["ci3"][2], v["unit"])
    end
    @printf("\n--- DERIVED (%d params, stellar unc propagated) ---\n", length(der_e))
    for (k,v) in der_e
        @printf("  %-18s = %12.5g  -%.4g / +%.4g  [%.4g, %.4g] %s\n",
                k, v["value"], v["err_lo"], v["err_hi"], v["ci3"][1], v["ci3"][2], v["unit"])
    end
    @printf("\nstellar assumed: %s\n", der_c["stellar"])
    # chunk 3: write all formats + model selection
    tdir = joinpath(dirname(ENV["SCITAB"]), "tables")
    fp = Nereus.write_science_table(tdir, "fitted", fit_e, fit_c; title="HD 18599 fitted parameters")
    dp = Nereus.write_science_table(tdir, "derived", der_e, der_c; title="HD 18599 derived parameters")
    ms = Nereus.science_model_selection(chP, params)
    @printf("\nwrote fitted: %s\n", join(values(fp), " "))
    @printf("wrote derived: %s\n", join(values(dp), " "))
    @printf("model_selection: %s\n", ms)
    exit(0)
end

# SCIOUT=<chains.nc>: assemble the full return-JSON contract (chunk 4) + write
# tables, then dump the JSON top-level structure.
if haskey(ENV,"SCIOUT")
    import JSON3
    chp = ENV["SCIOUT"]
    chP,_ = Nereus.load_chains(chp)
    summ = Nereus.science_summary(dirname(chp), chP, params, data;
                                   n_walkers=160, T_eff=5060.0)
    open(joinpath(dirname(chp), "summary.json"), "w") do io
        JSON3.pretty(io, JSON3.write(summ; allow_inf=true))
    end
    @printf("=== return-JSON contract: top-level keys ===\n")
    for k in keys(summ); @printf("  %s\n", k); end
    @printf("\nrun_info.convergence: %s\n", summ["run_info"]["convergence"])
    @printf("model_selection.noise_models: %s\n", summ["model_selection"]["noise_models"])
    @printf("tables.fitted: %s\n", summ["tables"]["fitted"])
    @printf("derived params: %s\n", sort(collect(keys(summ["derived"]["parameters"]))))
    @printf("\nwrote summary.json → %s\n", joinpath(dirname(chp),"summary.json"))
    exit(0)
end

# POSTSCI=<chains.nc>: render the three posterior-plot variants (raw/parameters/
# histograms) for planet 1 (chunk 5).
if haskey(ENV,"POSTSCI")
    chp = ENV["POSTSCI"]; od = joinpath(dirname(chp), "plots")
    chP,_ = Nereus.load_chains(chp)
    Nereus.plot_posteriors_raw(chP, params; output=od, planet=1)
    Nereus.plot_posteriors_parameters(chP, params; output=od, planet=1)
    Nereus.plot_posteriors_histograms(chP, params; output=od, planet=1)
    Nereus.plot_traces_grouped(chP, params; output=od)
    @printf("rendered posteriors + traces → %s\n", od)
    exit(0)
end

# CONV_FROM=<chains.nc>: re-run the (active-conditional) convergence gate on a
# saved chain, passing `params` so per-component params are assessed conditioned
# on their active mask (inactive-slot junk no longer poisons R̂/ESS).
if haskey(ENV,"CONV_FROM")
    chP,_ = Nereus.load_chains(ENV["CONV_FROM"])
    nweff = parse(Int, get(ENV,"NWEFF","160"))
    Nereus.convergence_report(chP, nweff; model_params=params)
    exit(0)
end

NT=parse(Int,get(ENV,"NT","12")); NW=parse(Int,get(ENV,"NW","128"))
NS=parse(Int,get(ENV,"NS","20000")); NB=parse(Int,get(ENV,"NB","8000"))
NTRY=parse(Int,get(ENV,"NTRY","10")); NREF=parse(Int,get(ENV,"NREF","15"))
@printf("trans-dim noise select: %d temps × %d walkers × %d+%d (NTRY=%d NREF=%d)\n",NT,NW,NS,NB,NTRY,NREF)
t0=time()
# ADAPTIVE LADDER, on by default. A fixed geometric ladder from beta_min to 1
# is adequate for the RV-only likelihood (114 points) and fails outright for
# RVPM: adding ~25k photometric points makes the log-likelihood range two
# orders of magnitude larger, adjacent rungs stop overlapping, and temp-swap
# acceptance collapses to ~0.001 at the cold end. With the ladder frozen the
# cold chains never communicate, R-hat runs to 4.5, and the model posterior
# inverts to a model the separate-runs check says loses by 38 nats.
# ADAPT=0 restores the old fixed ladder; BETAMIN tunes the cold end.
ADAPT = get(ENV, "ADAPT", "1") != "0"
BETAMIN = parse(Float64, get(ENV, "BETAMIN", "1e-4"))
@printf("ladder: %s, beta_min=%.1e\n", ADAPT ? "ADAPTIVE" : "fixed geometric", BETAMIN)
res=sample_transdim_ptemcee(target, data; td=td, n_temps=NT, n_walkers=NW, n_steps=NS,
    n_burnin=NB, n_birth_tries=NTRY, n_birth_refine=NREF, seed=42, show_progress=true,
    adapt_ladder=ADAPT, beta_min=BETAMIN)
@printf("done in %.1f min  logZ=%.2f\n", (time()-t0)/60, res.log_evidence)
@printf("noise toggles accepted/proposed per temp (cold→hot): %s\n",
        join(string.(res.noise_td_accepted, "/", res.noise_td_proposed), " "))

ch=res.chains; cn=names(ch,:parameters)
labels=["AD","AGP","CeleriteRotation","MA(1)","AR(1)"]
@printf("\nMODEL POSTERIOR P(M | RV, indicators)  (one trans-dim chain):\n")
nm=params.config.noise_models
occ=Float64[]
for (j,m) in enumerate(toggle)
    local nmi = findfirst(==(m), nm)
    local acol = Symbol("noise_active_$nmi")
    local p = acol in cn ? mean(vec(Array(ch[acol])).>0.5) : NaN
    push!(occ, p)
end
pnone = 1 - sum(occ)
for (l,p) in zip(labels,occ); @printf("  P(%-16s) = %.3f\n", l, p); end
@printf("  P(%-16s) = %.3f\n", "white/none", pnone)
@printf("\nseparate-runs CHECK (model selection, sign-fixed): AD wins, +38.5 over CeleriteRotation\n")
# OUTDIR names the run. The default keeps the historical RV-only name; the
# RVPM artifact is a different measurement on a different parameter space and
# must not land on top of it.
let outdir = get(ENV, "OUTDIR",
                 joinpath(@__DIR__, "..", "results",
                          USE_PHOT ? "HD18599_RVPM_transdim_select"
                                   : "HD18599_transdim_select"))
    mkpath(outdir)
    save_chains(joinpath(outdir, "chains.nc"), ch, params; data=data)
    @printf("chains → %s\n", joinpath(outdir, "chains.nc"))
end
