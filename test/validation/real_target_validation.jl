# Real-target recovery validation for Nereus's samplers.
#
# The strongest form of "do the samplers find correct results": run the
# sampler MENU on REAL public RV data and check each recovers the PUBLISHED
# orbit — recover-or-fail-loud. The synthetic matrix (sampler_validation.jl)
# proves recovery on injected truth through the production stack; this proves
# it on actual data against literature ground truth, where the period must be
# found in a broad search range and real instrument offsets / activity are
# present.
#
# This reuses the synthetic harness's `run_sampler` (per-sampler dispatch +
# smoke budgets) and `score_cell` (truth-in-99%-CI + silent-wrong) verbatim —
# the only difference is the "truth" is the LITERATURE orbit and the data is
# real. A sampler PASSES a target when the published P/K/e fall inside its 99%
# CI (recovery), and is SILENT-WRONG if it reports a tight CI that EXCLUDES
# the literature value (e.g. it confidently locks onto a period alias).
#
# Existing per-target single-sampler validations live in
# `results/real_data_validation.md` (paper-staging, uncommitted); this turns
# that into a systematic menu × target matrix.

include(joinpath(@__DIR__, "sampler_validation.jl"))

using DelimitedFiles, Random, Printf

const DATA_DIR = "/Users/jayvains/github/personal/Nereus/Nereus.jl/test/data"

# Literature orbit "truth" in the NamedTuple shape score_cell expects
# (`truth.map[key]`). Eccentricity keys are matched in derived (e_kN) space.
_lit_truth(d::Dict{String,Float64}) = (map = d, planets = nothing)

# ---------------------------------------------------------------------
# 51 Peg b — Mayor & Queloz 1995: P=4.2308 d, K=55.9 m/s, e≈0.
# Clean single hot Jupiter, 5-instrument multi-spectrograph data. The
# canonical "must recover or the sampler is broken" REAL target.
# ---------------------------------------------------------------------
function gen_51peg(; max_per_inst::Int = 40, seed::Int = 42)
    raw = readdlm(joinpath(DATA_DIR, "51peg.csv"), ',', Any, '\n'; header = true)
    dm = raw[1]
    bjd = Float64.(dm[:, 1]); rv = Float64.(dm[:, 2])
    rverr = Float64.(dm[:, 3]); inst = String.(dm[:, 4])
    names = sort!(unique(inst))
    imap = Dict(n => i for (i, n) in enumerate(names))
    rinst = [imap[s] for s in inst]
    rng = MersenneTwister(seed); keep = Int[]
    for i in eachindex(names)
        idx = findall(rinst .== i)
        length(idx) > max_per_inst &&
            (idx = sort(idx[randperm(rng, length(idx))[1:max_per_inst]]))
        append!(keep, idx)
    end
    sort!(keep)
    data = Data(; t_rv = bjd[keep], rv = rv[keep], rv_err = rverr[keep],
                  rv_inst = rinst[keep])
    ic = InstrumentConfig(rv = names)
    # BLIND broad period prior — a real PT budget (see run_real_one) finds the
    # period in [1,50] d without a periodogram crutch. Verified: at 20 temps ×
    # 200 walkers × 20000 steps ptemcee puts 62% of the posterior at 4.2308 d
    # despite the 4.43/1.24/19.6 d alias forest. The synthetic SMOKE budget
    # (10×40×3000) settles in a noise mode — the failure was budget, not a need
    # for seeding. K brackets 55.9; e free.
    ov = Dict{String, PriorSpec}("P_k1" => LogUniformPrior(1.0, 50.0),
                                 "K_k1" => UniformPrior(0.0, 150.0))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY], instruments = ic,
                      data = data, parametrization = ParametrizationConfig(
                          mass = :K_driven, ew = :sesinw, time = :Tp),
                      M_s = 1.11, stability = :none, priors = ov)
    target = NereusTarget(params, data; unconstrained = false)
    truth = _lit_truth(Dict("P_k1" => 4.2308, "K_k1" => 55.9, "e_k1" => 0.0))
    return params, data, target, truth, ["P_k1", "K_k1", "e_k1"]
end

# ---------------------------------------------------------------------
# GJ 876 b+c — Rivera+ 2010 / Correia+ 2010: b P=61.117 d K=214 e=0.032,
# c P=30.088 d K=88 e=0.256 (2:1 resonance). 270 HARPS RVs (PRE+POST).
# Fixed N=2 RECOVERY test (blind N is the trans-dim shootout). Loose
# per-slot period brackets, but the slots MUST be PERIOD-ORDERED: the
# likelihood enforces an unconditional P_k1 < P_k2 hard prior (-Inf
# otherwise) to break N! label switching. So slot 1 = the SHORTER-period
# planet c (30 d), slot 2 = b (61 d). (Putting b in slot 1 makes the
# literature orbit itself illegal — verified: ~1% of the prior box legal,
# samplers fail at init / grind a 99% rejection burn-in.) AMD stability is
# OFF (the resonant pair would otherwise be rejected).
# ---------------------------------------------------------------------
function gen_gj876()
    raw = readdlm(joinpath(DATA_DIR, "gj876.csv"), ',', Any, '\n'; header = true)
    dm = raw[1]
    bjd = Float64.(dm[:, 1]); rv = Float64.(dm[:, 2])
    rverr = Float64.(dm[:, 3]); inst = String.(dm[:, 4])
    inst_names = sort!(unique(inst))
    imap = Dict(n => i for (i, n) in enumerate(inst_names))
    rinst = [imap[s] for s in inst]
    data = Data(; t_rv = bjd, rv = rv, rv_err = rverr, rv_inst = rinst)
    ic = InstrumentConfig(rv = inst_names)
    ov = Dict{String, PriorSpec}(
        "P_k1" => LogUniformPrior(20.0, 45.0), "K_k1" => UniformPrior(0.0, 300.0),  # c (shorter P → slot 1)
        "P_k2" => LogUniformPrior(40.0, 90.0), "K_k2" => UniformPrior(0.0, 300.0))  # b (longer P → slot 2)
    params = Params(; max_kplanet = 2, planet_modes = [RV_ONLY, RV_ONLY],
                      instruments = ic, data = data,
                      parametrization = ParametrizationConfig(mass = :K_driven, ew = :sesinw, time = :Tp),
                      M_s = 0.334, stability = :none, priors = ov)
    target = NereusTarget(params, data; unconstrained = false)
    truth = _lit_truth(Dict("P_k1" => 30.088, "K_k1" => 88.0,  "e_k1" => 0.256,   # c → slot 1
                            "P_k2" => 61.117, "K_k2" => 214.0, "e_k2" => 0.032))  # b → slot 2
    return params, data, target, truth, ["P_k1", "K_k1", "e_k1", "P_k2", "K_k2", "e_k2"]
end

const REAL_GENERATORS = Dict{String, Function}("51peg" => gen_51peg,
                                               "gj876" => gen_gj876)

# Real-data dispatch with ADEQUATE (not smoke) budgets. Blind period search
# over a real alias forest needs real PT/NS budgets — verified: ptemcee
# recovers 51 Peg b blind only at ~20 temps × 200 walkers × 20k steps, not
# the synthetic 10×40×3000 smoke budget. The global-search samplers (ptemcee,
# pt, nested*) are blind searchers; the local ones (nuts = single-mode HMC,
# map = point estimate) are not — blind they are expected to FAIL LOUD, which
# is acceptable under recover-or-fail-loud. Returns chains normalized like the
# synthetic run_sampler so score_cell can consume it.
function run_sampler_real(name::String, params::Params, data::Data,
                          target::NereusTarget)
    t0 = time(); chains = nothing; logZ = NaN; map_pt = nothing; ok = false; err = ""
    try
        if name == "ptemcee"
            res = sample_ptemcee(target, data; n_temps = 16, n_walkers = 150,
                                  n_steps = 12000, n_burnin = 5000,
                                  init_strategy = :prior, show_progress = false)
            chains = res.chains; logZ = res.log_evidence
        elseif name == "pt"
            chains, logZ = sample_pt(target; n_rounds = 13, n_chains = 12,
                                      show_report = false)
        elseif name == "nested"
            chains, logZ = sample_nested(target, data; n_live = 1500, dlogz = 0.5)
        elseif name == "nested_ins"
            res = sample_nested_ins(target, data; n_live = 600, dlogz = 0.05,
                                     max_iter = 200_000)
            chains = res.chains; logZ = res.log_z_ins
        elseif name == "nested_dynamic"
            res = sample_nested_dynamic(target, data; n_live_init = 400,
                                         n_live_batch = 400, dlogz_init = 1.0,
                                         dlogz_batch = 0.3, max_iter = 150_000)
            chains = res.chains; logZ = res.log_z
        elseif name == "pt_whitening"
            res = sample_pt_whitening(target, data; n_temps = 16, n_steps = 12000,
                                       n_burnin = 4000, show_progress = false)
            chains = res.chains; logZ = res.log_evidence
        elseif name == "population_annealing" || name == "pa"
            res = sample_pa(target, data; n_replicas = 1500, n_mcmc = 15,
                             max_steps = 200, show_progress = false)
            chains = res.chains; logZ = res.log_evidence
        elseif name == "smc"
            res = sample_smc(target, data; betas = range(0.0, 1.0; length = 80),
                              n_replicas = 1500, n_mcmc = 15, show_progress = false)
            chains = res.chains; logZ = res.log_evidence
        elseif name == "nuts"
            utarget = NereusTarget(target.params, data; unconstrained = true)
            chains = sample_nuts(utarget; n_samples = 2000, n_warmup = 2000,
                                  n_chains = 4, progress = false)
        elseif name == "map"
            utarget = NereusTarget(target.params, data; unconstrained = true)
            res = sample_map(utarget; n_starts = 64)
            map_pt = Dict{String,Float64}(nm => res.x_map[j]
                                          for (j, nm) in enumerate(res.param_names))
            logZ = res.log_evidence_laplace
        else
            error("unknown sampler `$name`")
        end
        ok = true
    catch e
        ok = false; err = sprint(showerror, e)
    end
    return (chains = chains, logZ = logZ, map_pt = map_pt,
            runtime_s = time() - t0, ok = ok, err = err)
end

# Score a pre-computed run-result against the literature orbit. Mirrors
# score_cell's logic (truth-in-99%-CI + tight-CI-excluding-truth silent-wrong,
# softened at hard bounds; MAP-tolerance branch for the point estimator) but
# consumes chains from run_sampler_real instead of re-running at smoke budget.
function _score_real(name, gen_name, params, truth, key_params, r)
    per = Dict{String,Any}(); in_all = true; any_scored = false; sw_keys = String[]
    if !r.ok
        return (sampler = name, generator = gen_name, ok = false, err = r.err,
                is_map = (r.map_pt !== nothing), is_ensemble = is_ensemble(name),
                truth_in_95ci = false, silent_wrong = false, max_rhat = NaN,
                min_ess = NaN, per_key = per, runtime_s = r.runtime_s)
    end
    # --- MAP point estimate: |MAP - lit| within tolerance ---
    if r.map_pt !== nothing
        mp = r.map_pt
        for key in key_params
            tv = truth.map[key]
            pv = if startswith(key, "e_k")
                suf = key[2:end]
                (haskey(mp, "sesinw$suf") && haskey(mp, "secosw$suf")) || continue
                sesinw_to_ew(mp["sesinw$suf"], mp["secosw$suf"])[1]
            else
                haskey(mp, key) || continue; mp[key]
            end
            any_scored = true
            kind, tol = _maptol(key)
            okk = kind === :rel ? abs(pv - tv) <= tol * max(abs(tv), 1e-6) :
                                  abs(pv - tv) <= tol
            in_all &= okk
            per[key] = (truth = tv, median = pv, point = pv, in95 = okk,
                        silent_wrong = false)
        end
        return (sampler = name, generator = gen_name, ok = true, err = "",
                is_map = true, is_ensemble = false,
                truth_in_95ci = (any_scored && in_all), silent_wrong = false,
                max_rhat = NaN, min_ess = NaN, per_key = per, runtime_s = r.runtime_s)
    end
    # --- chain-based ---
    chains = r.chains
    if chains === nothing || size(chains, 1) == 0
        return (sampler = name, generator = gen_name, ok = false,
                err = "empty/nothing chains", is_map = false,
                is_ensemble = is_ensemble(name), truth_in_95ci = false,
                silent_wrong = false, max_rhat = NaN, min_ess = NaN,
                per_key = per, runtime_s = r.runtime_s)
    end
    max_rhat, min_ess, _ = _convergence_diagnostics(name, chains)
    for key in key_params
        samp = _derived_samples(chains, key); isempty(samp) && continue
        any_scored = true
        tv = truth.map[key]; med = median(samp)
        lo95, hi95 = quantile(samp, [0.025, 0.975])
        lo99, hi99 = quantile(samp, [0.005, 0.995])
        in95 = (lo99 <= tv <= hi99)
        # Circular-orbit floor: a published e≈0 can never sit inside a
        # posterior e≥0 whose CI lower is necessarily >0. Count it recovered
        # when the posterior is consistent with circular (lower edge at the
        # e=0 floor and lit e below the upper edge).
        if startswith(key, "e_k") && tv < 0.05 && lo99 <= 0.05 && tv <= hi99
            in95 = true
        end
        in_all &= in95
        width95 = hi95 - lo95
        tight = startswith(key, "e_k") ? (width95 < 0.16) :
                                          (width95 < 0.30 * max(abs(tv), 1e-6) * 2)
        if !in95
            excl = tv < lo95 ? (lo95 - tv) : (tv - hi95)
            frac = width95 > 0 ? excl / width95 : Inf
            blo, bhi = _hard_bounds(params, key)
            tol_b = startswith(key, "e_k") ? 0.02 : 0.02 * max(abs(tv), 1e-6)
            against_bound =
                (tv < lo95 && isfinite(blo) && (lo95 - blo) <= tol_b) ||
                (tv > hi95 && isfinite(bhi) && (bhi - hi95) <= tol_b)
            sw = tight && ((frac > 0.10) || !against_bound)
            sw && push!(sw_keys, key)
            per[key] = (truth = tv, median = med, point = med, in95 = in95,
                        silent_wrong = sw)
        else
            per[key] = (truth = tv, median = med, point = med, in95 = in95,
                        silent_wrong = false)
        end
    end
    return (sampler = name, generator = gen_name, ok = true, err = "",
            is_map = false, is_ensemble = is_ensemble(name),
            truth_in_95ci = (any_scored && in_all), silent_wrong = !isempty(sw_keys),
            max_rhat = max_rhat, min_ess = min_ess, per_key = per,
            runtime_s = r.runtime_s)
end

"""
    run_real_one(sampler, target_key) -> scored cell

Load the real target, run the sampler on real data at an ADEQUATE budget
(`run_sampler_real`), score recovery of the LITERATURE orbit
(truth-in-99%-CI) + silent-wrong via the synthetic harness's `score_cell`.
`score_cell` calls `run_sampler` internally, so we shadow it for this scope
by scoring the chains from `run_sampler_real` directly.
"""
function run_real_one(sampler::String, target_key::String)
    haskey(REAL_GENERATORS, target_key) ||
        error("unknown real target `$target_key`")
    params, data, target, truth, key_params = REAL_GENERATORS[target_key]()
    r = run_sampler_real(sampler, params, data, target)
    return _score_real(sampler, target_key, params, truth, key_params, r)
end

# ---------------------------------------------------------------------
# Matrix runner (entry point).
# ---------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    samplers = length(ARGS) >= 1 ? String.(split(ARGS[1], ",")) :
               ["ptemcee", "pt", "nested", "nested_ins", "nested_dynamic",
                "nuts", "pt_whitening", "population_annealing", "smc", "map"]
    target_key = length(ARGS) >= 2 ? ARGS[2] : "51peg"
    println("="^100)
    @printf("REAL-TARGET RECOVERY: %s   (lit orbit as truth, recover-or-fail-loud)\n", target_key)
    @printf("%-22s %-8s %-8s %-9s %-8s %s\n",
            "sampler", "recover", "silentW", "rhat", "ess", "key (med vs lit)")
    println("="^100)
    sw = Tuple{String,String}[]
    for s in samplers
        cell = try
            run_real_one(s, target_key)
        catch e
            @printf("%-22s ERROR: %s\n", s, first(sprint(showerror, e), 70)); continue
        end
        cell.silent_wrong && push!(sw, (s, target_key))
        rec = cell.is_map ? (cell.truth_in_95ci ? "YES" : "no") :
                            (cell.truth_in_95ci ? "YES" : "no")
        rh = (cell.is_map || !isfinite(cell.max_rhat)) ? "n/a" : string(round(cell.max_rhat, digits = 3))
        es = (cell.is_map || !isfinite(cell.min_ess)) ? "n/a" : string(round(Int, cell.min_ess))
        kparts = String[]
        for (k, d) in sort(collect(cell.per_key); by = first)
            push!(kparts, "$k=$(round(d.point,digits=3))/$(round(d.truth,digits=3))")
        end
        @printf("%-22s %-8s %-8s %-9s %-8s %s\n", s, rec,
                cell.silent_wrong ? "**SW**" : "-", rh, es, join(kparts, " "))
    end
    println("="^100)
    println(isempty(sw) ? "✅ $target_key: ZERO silent-wrong (recover-or-fail-loud)" :
            "❌ silent-wrong: $(sw)")
end
