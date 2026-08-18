#!/usr/bin/env julia
# WASP-47 — trans-dim RV+photometry recovery menu (real-target menu, transit leg).
#
# Same proven model/prior/trans-dim setup as compare_WASP47_joint_search.jl,
# but the TESS LC is kept at NATIVE cadence (no 5-min binning — honors the
# never-bin-the-LC rule, and preserves the USP planet e's short ingress/egress).
# Runs the four trans-dim samplers and scores recovery of the published
# 4-planet system (Bryant & Bayliss 2022). Each sampler's verdict is appended
# to RESULTS as it finishes, so a slow/hung sampler doesn't cost earlier ones.
#
# Published (post-hoc only, NEVER plugged into priors — blind search):
#   e: P=0.789593 d  K=4.55    b: P=4.1591287 d K=140.84
#   d: P=9.030672 d  K=4.96    c: P=588.4 d     K=31.6 (RV-only, P>baseline)

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, mean, quantile
using Printf
using MCMCChains
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const RESULTS   = joinpath(@__DIR__, "menu_WASP47_results.txt")
logline(s) = (open(RESULTS, "a") do io; println(io, s); end; println(s); flush(stdout))

const LIT = [(name="e", P=0.789593, K=4.55), (name="b", P=4.1591287, K=140.84),
             (name="d", P=9.030672, K=4.96), (name="c", P=588.4, K=31.6)]

println("="^70); println("WASP-47 — trans-dim RV+phot recovery menu (native cadence)")
println("="^70)

# ---- TESS LC at NATIVE cadence (no binning) --------------------------
lc_raw = load_tess_lc(joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"))
perm = sortperm(lc_raw.t)
flat = (t = lc_raw.t[perm], flux = lc_raw.flux[perm], flux_err = lc_raw.flux_err[perm])
@printf("TESS s42 NATIVE: %d pts, σ ≈ %.2e, %.1f d span\n",
        length(flat.t), std(flat.flux), flat.t[end] - flat.t[1])

# ---- Multi-instrument RVs (mirror Bryant 2022 exclusions) ------------
rv_raw_full = readdlm(joinpath(DATADIR, "WASP47_RVs_combined.csv"), ',', Any, '\n'; header=true)
rv_raw = rv_raw_full[1]; hdr = vec(rv_raw_full[2])
col(name) = findfirst(==(name), hdr)
flag_col = col("flag")
keep_mask = trues(size(rv_raw, 1))
for i in 1:size(rv_raw, 1)
    f = rv_raw[i, flag_col]; f === missing && continue
    String(strip(string(f))) in ("transit","transit_night","anomalous") && (keep_mask[i] = false)
end
rv_raw = rv_raw[keep_mask, :]
bjd_rv = Float64.(rv_raw[:, col("bjd")]); rv = Float64.(rv_raw[:, col("rv")])
rv_err = Float64.(rv_raw[:, col("rv_err")]); rv_inst_str = String.(rv_raw[:, col("instrument")])
inst_names = sort(unique(rv_inst_str))
inst_to_idx = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_to_idx[s] for s in rv_inst_str]
@printf("Loaded %d RVs across %d instruments (%.1f d span)\n",
        length(bjd_rv), length(inst_names), maximum(bjd_rv) - minimum(bjd_rv))

# ---- Build Data + Params (identical to proven shootout) --------------
data = Data(; t_rv=bjd_rv, rv=rv, rv_err=rv_err, rv_inst=rv_inst,
            t_phot=flat.t, flux=flat.flux, flux_err=flat.flux_err,
            phot_inst=ones(Int, length(flat.t)))
ic = InstrumentConfig(rv = inst_names, pm = ["TESS"])
priors = Dict{String, PriorSpec}()
for k in 1:3
    priors["P_k$k"]=LogUniformPrior(0.5,30.0); priors["K_k$k"]=UniformPrior(0.0,200.0)
    priors["sesinw_k$k"]=UniformPrior(-0.7,0.7); priors["secosw_k$k"]=UniformPrior(-0.7,0.7)
    priors["Tc_k$k"]=UniformPrior(flat.t[1], flat.t[1]+30.0)
    priors["b_k$k"]=UniformPrior(0.0,1.0); priors["rr_k$k"]=UniformPrior(0.005,0.20)
end
priors["P_k4"]=LogUniformPrior(100.0,1500.0); priors["K_k4"]=UniformPrior(0.0,200.0)
priors["sesinw_k4"]=UniformPrior(-0.7,0.7); priors["secosw_k4"]=UniformPrior(-0.7,0.7)
for name in inst_names
    rv_i = rv[rv_inst .== inst_to_idx[name]]; isempty(rv_i) && continue
    centre = median(rv_i); span = max(maximum(rv_i)-minimum(rv_i), 1.0)
    priors["gamma_$name"]=UniformPrior(centre-3*span, centre+3*span)
    priors["sigma_$name"]=ModJeffreysPrior(0.1, 50.0)
end
priors["offset_TESS"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); priors["jitter_TESS"]=LogUniformPrior(1e-5,5e-3)
priors["q1_TESS"]=UniformPrior(0.0,1.0); priors["q2_TESS"]=UniformPrior(0.0,1.0)
priors["rho_s"]=NormalPrior(1.00,0.30,0.1,5.0)
parametrization = ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true)
params = Params(; max_kplanet=4, planet_modes=[RVPM,RVPM,RVPM,RV_ONLY],
    instruments=ic, data=data, M_s=1.04, R_s=1.137, parametrization=parametrization,
    priors=priors, trend_order=0, stability=:amd,
    external_priors=[ExternalPrior(:ecc, NormalPrior(0.0,0.3), true)])
@printf("Free params: %d\n", n_unfrozen(params))
target = NereusTarget(params, data; unconstrained=true)

td = TransDimConfig(max_kplanet=4, planets=true, noise=false,
    birth_strategies=[PriorBirth(), InformedBirth()], birth_weights=[0.3,0.7],
    transdim_fraction=0.3)

const SEED = 42
# Budgets (env-overridable for smoke testing). Defaults = full shootout.
ev(k, d) = parse(Int, get(ENV, k, string(d)))
const RJ_S = ev("W47_RJ_S", 30000);  const RJ_W = ev("W47_RJ_W", 8000)
const PT_R = ev("W47_PT_R", 10);     const PT_C = ev("W47_PT_C", 16)
const MO_S = ev("W47_MO_S", 8000);   const MO_W = ev("W47_MO_W", 2000)
const NS_L = ev("W47_NS_L", 400);    const NS_M = ev("W47_NS_M", 30)

# Recovery scorer: at modal N_p, which published planets match (period within 2%)?
function score(name, chains, secs; log_Z=NaN)
    probs = model_probabilities(chains; max_kplanet=4)
    np_post = [get(probs, k, 0.0) for k in 0:4]
    modal_np = argmax(np_post) - 1
    np_vec = vec(Array(chains[:, :n_planets, :])); mask = np_vec .== modal_np
    rec = Tuple{Float64,Float64}[]
    if any(mask) && modal_np >= 1
        Pc = [vec(Array(chains[:, Symbol("P_k$k"), :]))[mask] for k in 1:4]
        Kc = [vec(Array(chains[:, Symbol("K_k$k"), :]))[mask] for k in 1:4]
        Ps = [Float64[] for _ in 1:modal_np]; Ks = [Float64[] for _ in 1:modal_np]
        for s in 1:count(mask)
            Pv = [Pc[k][s] for k in 1:4]; Kv = [Kc[k][s] for k in 1:4]
            o = sortperm(Pv)
            for j in 1:modal_np; push!(Ps[j], Pv[o[j]]); push!(Ks[j], Kv[o[j]]); end
        end
        for j in 1:modal_np; push!(rec, (median(Ps[j]), median(Ks[j]))); end
    end
    found = String[]
    for lit in LIT, (Pm,_) in rec
        abs(Pm-lit.P)/lit.P < 0.02 && !(lit.name in found) && push!(found, lit.name)
    end
    ess = try Float64(MCMCChains.ess_rhat(chains[:, [:n_planets], :])[1, :ess]) catch; NaN end
    recstr = join([@sprintf("P=%.3f/K=%.1f", Pm, Km) for (Pm,Km) in rec], " ")
    zstr = isnan(log_Z) ? "" : @sprintf(" logZ=%.1f", log_Z)
    logline(@sprintf("%-8s %5.0fs modalNp=%d found=[%s] %d/4 ESS(Np)=%.0f%s | %s",
                     name, secs, modal_np, join(found, ","), length(found), ess, zstr, recstr))
end

logline("# WASP-47 trans-dim menu (native s42)  lit: e=0.79 b=4.16 d=9.03 c=588")

# ---- 1. rjmcmc -------------------------------------------------------
println("\n--- rjmcmc ---"); t0=time()
try
    chains, _ = sample_rjmcmc(target, target.data; td=td, n_samples=RJ_S,
                              n_warmup=RJ_W, seed=SEED, n_chains=4,
                              within_model=:rwm)
    score("rjmcmc", chains, time()-t0)
catch e; logline(@sprintf("rjmcmc   %5.0fs FAILED: %s", time()-t0, sprint(showerror,e))); end

# ---- 2. moms ---------------------------------------------------------
println("\n--- moms ---"); t0=time()
try
    chains, _, _ = sample_moms(target, target.data; td=td, n_samples=MO_S, n_warmup=MO_W,
        seed=SEED, n_chains=4, init_scale=1.0, within_model=:rwm, informed_birth_fraction=0.5)
    score("moms", chains, time()-t0)
catch e; logline(@sprintf("moms     %5.0fs FAILED: %s", time()-t0, sprint(showerror,e))); end

# ---- 4. moms_ns ------------------------------------------------------
println("\n--- moms_ns ---"); t0=time()
try
    chains, log_Z, _ = sample_moms_ns(target, target.data; td=td, n_live=NS_L, dlogz=0.5,
        n_mcmc=NS_M, seed=SEED, informed_birth_fraction=0.5, batch_size=8)
    score("moms_ns", chains, time()-t0; log_Z=log_Z)
catch e; logline(@sprintf("moms_ns  %5.0fs FAILED: %s", time()-t0, sprint(showerror,e))); end

# ---- 4. pt (trans-dim, Pathfinder-warm) — LAST: Pigeons-doubling = the multi-day pole
println("\n--- pt ---"); t0=time()
try
    chains, _, _, _ = sample_pt_warm(target; n_pathfinder_runs=16, n_pathfinder_draws=200,
        td=td, n_rounds=PT_R, n_chains=PT_C, seed=SEED, show_report=false,
        within_model=:rwm, early_stop_thresh=0.001, early_stop_min_rounds=12)
    score("pt", chains, time()-t0)
catch e; logline(@sprintf("pt       %5.0fs FAILED: %s", time()-t0, sprint(showerror,e))); end

println("\nDone. Verdicts in $RESULTS")
