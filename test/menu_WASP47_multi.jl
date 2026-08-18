#!/usr/bin/env julia
# WASP-47 trans-dim RV+photometry recovery — MULTI-SECTOR.
# Adds K2 C3 (80-day discovery baseline, ~19 transits of b / ~9 of d / ~100 of
# e) + TESS s92 to the existing TESS s42, as THREE separate phot instruments
# (own offset/jitter/LD). All native cadence (no binning). The K2 baseline is
# the key addition for the small transiting planets e (0.79d) and d (9.03d).
#
# rjmcmc runs :rwm (Fix A) with a big budget — it's the proven, now-fast
# recoverer. moms + pt(10 rounds) too. moms_ns SKIPPED (too slow at ~155k pts).
# Scored vs Bryant & Bayliss 2022.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, mean, quantile
using Printf
using MCMCChains

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const RESULTS   = joinpath(@__DIR__, "menu_WASP47_multi_results.txt")
logline(s) = (open(RESULTS, "a") do io; println(io, s); end; println(s); flush(stdout))
const LIT = [(name="e", P=0.789593, K=4.55), (name="b", P=4.1591287, K=140.84),
             (name="d", P=9.030672, K=4.96), (name="c", P=588.4, K=31.6)]

ev(k, d) = parse(Int, get(ENV, k, string(d)))
const RJ_S = ev("W47_RJ_S", 80000);  const RJ_W = ev("W47_RJ_W", 20000)
const PT_R = ev("W47_PT_R", 10);     const PT_C = ev("W47_PT_C", 16)
const MO_S = ev("W47_MO_S", 12000);  const MO_W = ev("W47_MO_W", 3000)

println("="^70); println("WASP-47 multi-sector trans-dim recovery (K2 + TESS s42 + s92)")

# ---- Photometry: 3 instruments, native cadence ----------------------
pm_names = ["K2", "TESS42", "TESS92"]
files = ["WASP-47_k2_c03_everest_lc.csv", "WASP-47_tess_s42_lc.csv", "WASP-47_tess_s92_lc.csv"]
pt_t = Float64[]; pt_f = Float64[]; pt_e = Float64[]; pt_i = Int[]
for (ix, fn) in enumerate(files)
    lc = load_tess_lc(joinpath(DATADIR, fn))
    append!(pt_t, lc.t); append!(pt_f, lc.flux); append!(pt_e, lc.flux_err)
    append!(pt_i, fill(ix, length(lc.t)))
    @printf("  %-7s %6d pts  %.1f d baseline\n", pm_names[ix], length(lc.t), lc.t[end]-lc.t[1])
end
perm = sortperm(pt_t)
pt_t = pt_t[perm]; pt_f = pt_f[perm]; pt_e = pt_e[perm]; pt_i = pt_i[perm]
@printf("TOTAL phot: %d pts across %d instruments\n", length(pt_t), length(pm_names))

# ---- RVs (mirror Bryant 2022 exclusions) ----------------------------
rr = readdlm(joinpath(DATADIR, "WASP47_RVs_combined.csv"), ',', Any, '\n'; header=true)
raw = rr[1]; hdr = vec(rr[2]); c(n)=findfirst(==(n),hdr)
fcol=c("flag"); keep=trues(size(raw,1))
for i in 1:size(raw,1); f=raw[i,fcol]; f===missing && continue
    String(strip(string(f))) in ("transit","transit_night","anomalous") && (keep[i]=false); end
raw=raw[keep,:]
bjd=Float64.(raw[:,c("bjd")]); rv=Float64.(raw[:,c("rv")]); rve=Float64.(raw[:,c("rv_err")])
istr=String.(raw[:,c("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
rinst=[i2i[s] for s in istr]
@printf("Loaded %d RVs across %d instruments\n", length(bjd), length(inames))

# ---- Data + Params --------------------------------------------------
data = Data(; t_rv=bjd, rv=rv, rv_err=rve, rv_inst=rinst,
            t_phot=pt_t, flux=pt_f, flux_err=pt_e, phot_inst=pt_i)
ic = InstrumentConfig(rv=inames, pm=pm_names)
pri = Dict{String,PriorSpec}()
for k in 1:3
    pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["K_k$k"]=UniformPrior(0.0,200.0)
    pri["sesinw_k$k"]=UniformPrior(-0.7,0.7); pri["secosw_k$k"]=UniformPrior(-0.7,0.7)
    pri["Tc_k$k"]=UniformPrior(pt_t[1], pt_t[1]+30.0)
    pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
end
pri["P_k4"]=LogUniformPrior(100.0,1500.0); pri["K_k4"]=UniformPrior(0.0,200.0)
pri["sesinw_k4"]=UniformPrior(-0.7,0.7); pri["secosw_k4"]=UniformPrior(-0.7,0.7)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
    ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
# Per phot-instrument systematics + limb darkening
for n in pm_names
    pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0)
end
pri["rho_s"]=NormalPrior(1.0,0.3,0.1,5.0)
params = Params(; max_kplanet=4, planet_modes=[RVPM,RVPM,RVPM,RV_ONLY], instruments=ic, data=data,
    M_s=1.04, R_s=1.137, parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true),
    priors=pri, trend_order=0, stability=:amd, external_priors=[ExternalPrior(:ecc, NormalPrior(0.0,0.3), true)])
@printf("Free params: %d\n", n_unfrozen(params))
target = NereusTarget(params, data; unconstrained=true)
td = TransDimConfig(max_kplanet=4, planets=true, noise=false,
    birth_strategies=[PriorBirth(),InformedBirth()], birth_weights=[0.3,0.7], transdim_fraction=0.3)
const SEED = 42

function score(name, chains, secs; log_Z=NaN)
    probs = model_probabilities(chains; max_kplanet=4)
    np_post = [get(probs, k, 0.0) for k in 0:4]; modal_np = argmax(np_post) - 1
    np_vec = vec(Array(chains[:, :n_planets, :])); mask = np_vec .== modal_np
    rec = Tuple{Float64,Float64}[]
    if any(mask) && modal_np >= 1
        Pc = [vec(Array(chains[:, Symbol("P_k$k"), :]))[mask] for k in 1:4]
        Kc = [vec(Array(chains[:, Symbol("K_k$k"), :]))[mask] for k in 1:4]
        Ps = [Float64[] for _ in 1:modal_np]; Ks = [Float64[] for _ in 1:modal_np]
        for s in 1:count(mask)
            Pv=[Pc[k][s] for k in 1:4]; Kv=[Kc[k][s] for k in 1:4]; o=sortperm(Pv)
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

logline("# WASP-47 multi-sector (K2+s42+s92)  lit: e=0.79 b=4.16 d=9.03 c=588")

println("\n--- rjmcmc :rwm (big budget) ---"); t0=time()
try
    chains, _ = sample_rjmcmc(target, target.data; td=td, n_samples=RJ_S, n_warmup=RJ_W,
                              seed=SEED, n_chains=4, within_model=:rwm)
    score("rjmcmc", chains, time()-t0)
catch e; logline(@sprintf("rjmcmc   %5.0fs FAILED: %s", time()-t0, sprint(showerror,e))); end

println("\n--- moms ---"); t0=time()
try
    chains, _, _ = sample_moms(target, target.data; td=td, n_samples=MO_S, n_warmup=MO_W,
        seed=SEED, n_chains=4, init_scale=1.0, within_model=:rwm, informed_birth_fraction=0.5)
    score("moms", chains, time()-t0)
catch e; logline(@sprintf("moms     %5.0fs FAILED: %s", time()-t0, sprint(showerror,e))); end

println("\n--- pt (10 rounds) ---"); t0=time()
try
    chains, _, _, _ = sample_pt_warm(target; td=td, n_rounds=PT_R, n_chains=PT_C, seed=SEED,
        show_report=false, within_model=:rwm, init_strategy=:prior,
        early_stop_thresh=0.001, early_stop_min_rounds=8)
    score("pt", chains, time()-t0)
catch e; logline(@sprintf("pt       %5.0fs FAILED: %s", time()-t0, sprint(showerror,e))); end

println("\nDone. Verdicts in $RESULTS")
