#!/usr/bin/env julia
# WASP-47 — JOINT (TESS LC + HARPS-N RV) blind-search trans-dim shootout.
#
# Tests all four trans-dim samplers Nereus supports on the same joint
# target with **broad period priors** — the sampler must DISCOVER each
# planet, not have its period plugged in. This is the realistic test
# case for trans-dim model selection on a multi-planet system.
#
# WASP-47 has four known planets:
#   * b — hot Jupiter, P = 4.159 d, K ≈ 140 m/s, transits + RV
#   * c — outer giant,  P = 395 d,   K ≈ 30 m/s,  RV-only (no transit)
#   * d — Neptune,      P = 9.031 d, K ≈ 5 m/s,   transits + RV
#   * e — sub-Earth,    P = 0.789 d, K ≈ 5 m/s,   transits + RV
#
# Trans-dim setup: max_kplanet = 4, planet_modes = [RVPM, RVPM, RVPM,
# RV_ONLY]. Slots 1-3 search short-period transit candidates; slot 4
# searches a long-period RV-only candidate (c). The sampler can
# deactivate any slot if the data doesn't support it.
#
# Reports per sampler:
#   * wall time + likelihood evals
#   * N_p posterior P(N_p = k) for k = 0..4
#   * recovered (P, K, Tc) at the modal N_p, sorted by period
#   * ESS for n_planets (mixing on the trans-dim coordinate)
#   * log Z (MoMS-NS only)
#   * Bayes factors via model_probabilities

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, mean, quantile
using Printf
using MCMCChains
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results",
                            "WASP47_joint_search")
mkpath(OUT_DIR)

# Literature values (Bryant & Bayliss 2022, arxiv 2202.12747) — used
# ONLY for post-hoc comparison, never plugged into the priors. The
# sampler must find these blind.
#   * b: hot Jupiter, transits + RV
#   * c: outer eccentric giant (e ≈ 0.36), RV-only, P > RV baseline
#     (495 d) so the sampler will under-constrain its period
#   * d: warm Neptune, transits + RV
#   * e: ultra-short-period super-Earth, transits + RV
const Pb_LIT, Kb_LIT = 4.1591287, 140.84
const Pc_LIT, Kc_LIT = 588.4,      31.6     # Bryant+ 2022; K from M_p sin i
const Pd_LIT, Kd_LIT = 9.030672,    4.96
const Pe_LIT, Ke_LIT = 0.789593,    4.55

# ---- Load TESS LC (binned to 5 min) ---------------------------------
println("="^70)
println("WASP-47 — joint RV+LC blind-search shootout")
println("="^70)

lc_raw = load_tess_lc(joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"))
flat = let bd = 5.0 / (60 * 24)
    bin_id = floor.(Int, (lc_raw.t .- minimum(lc_raw.t)) ./ bd)
    ub = sort(unique(bin_id))
    tt = Float64[]; ff = Float64[]; ee = Float64[]
    for b in ub
        idxs = findall(==(b), bin_id)
        push!(tt, mean(lc_raw.t[idxs])); push!(ff, mean(lc_raw.flux[idxs]))
        push!(ee, mean(lc_raw.flux_err[idxs]) / sqrt(length(idxs)))
    end
    (t = tt, flux = ff, flux_err = ee)
end
@printf("TESS s42 binned: %d pts, σ ≈ %.2e, %.1f d span\n",
        length(flat.t), std(flat.flux), flat.t[end] - flat.t[1])

# ---- Load multi-instrument RVs (HARPS-N + ESPRESSO + HIRES + CORALIE
# +PFS, sorted by BJD). Skip rows flagged as transit / transit_night /
# anomalous so we mirror Bryant 2022 / Sinukoff 2017 exclusions and
# avoid the Rossiter–McLaughlin distortion of WASP-47b transits.
rv_raw_full = readdlm(joinpath(DATADIR, "WASP47_RVs_combined.csv"),
                       ',', Any, '\n'; header = true)
rv_raw  = rv_raw_full[1]
hdr     = vec(rv_raw_full[2])
col(name) = findfirst(==(name), hdr)

flag_col = col("flag")
keep_mask = trues(size(rv_raw, 1))
for i in 1:size(rv_raw, 1)
    f = rv_raw[i, flag_col]
    f === missing && continue
    s = String(strip(string(f)))
    if s in ("transit", "transit_night", "anomalous")
        keep_mask[i] = false
    end
end
rv_raw = rv_raw[keep_mask, :]

bjd_rv = Float64.(rv_raw[:, col("bjd")])
rv     = Float64.(rv_raw[:, col("rv")])
rv_err = Float64.(rv_raw[:, col("rv_err")])
rv_inst_str = String.(rv_raw[:, col("instrument")])

# Per-instrument index, ordered for deterministic γ/σ assignment.
inst_names = sort(unique(rv_inst_str))
inst_to_idx = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst = [inst_to_idx[s] for s in rv_inst_str]
rv_max  = maximum(abs, rv .- median(rv))

@printf("Loaded %d RVs across %d instruments (BJD %.1f → %.1f, %.1f d span):\n",
        length(bjd_rv), length(inst_names),
        minimum(bjd_rv), maximum(bjd_rv),
        maximum(bjd_rv) - minimum(bjd_rv))
for (i, name) in enumerate(inst_names)
    n = count(==(i), rv_inst)
    σ = std(rv[rv_inst .== i])
    @printf("  [%d] %-13s  n=%3d  σ_RV=%6.2f m/s\n", i, name, n, σ)
end

# ---- Build Data + Params ---------------------------------------------
data = Data(;
    t_rv = bjd_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    t_phot = flat.t, flux = flat.flux, flux_err = flat.flux_err,
    phot_inst = ones(Int, length(flat.t)),
)
ic = InstrumentConfig(rv = inst_names, pm = ["TESS"])

# **Blind-search priors**: slots 1-3 cover short-period transit
# candidates (e at 0.79d, b at 4.16d, d at 9.03d); slot 4 covers the
# long-period RV-only regime (c at 588d). Slot asymmetry biases the
# search toward this decomposition by prior, which is necessary at
# moderate live-point budgets — flat priors across all slots are
# theoretically cleaner but PriorBirth's hit rate scales as 1/log-span
# so widening hurts random-search at fixed budget.
priors = Dict{String, PriorSpec}()
for k in 1:3
    priors["P_k$k"]      = LogUniformPrior(0.5, 30.0)
    priors["K_k$k"]      = UniformPrior(0.0, 200.0)
    priors["sesinw_k$k"] = UniformPrior(-0.7, 0.7)
    priors["secosw_k$k"] = UniformPrior(-0.7, 0.7)
    priors["Tc_k$k"]     = UniformPrior(flat.t[1], flat.t[1] + 30.0)
    priors["b_k$k"]      = UniformPrior(0.0, 1.0)
    priors["rr_k$k"]     = UniformPrior(0.005, 0.20)
end
priors["P_k4"]      = LogUniformPrior(100.0, 1500.0)
priors["K_k4"]      = UniformPrior(0.0, 200.0)
priors["sesinw_k4"] = UniformPrior(-0.7, 0.7)
priors["secosw_k4"] = UniformPrior(-0.7, 0.7)

# Per-RV-instrument systematics. Every instrument gets its own γ
# (zero-point) and σ (jitter) since each spectrograph has independent
# absolute calibration and red noise. The CORALIE pre/post upgrade
# split (Nov 2014) is treated as two distinct instruments — Bryant
# 2022 + Neveu-VanMalle 2016 both fit them with separate γ values.
for name in inst_names
    rv_i = rv[rv_inst .== inst_to_idx[name]]
    if isempty(rv_i)
        continue
    end
    centre = median(rv_i)
    span   = max(maximum(rv_i) - minimum(rv_i), 1.0)
    priors["gamma_$name"] = UniformPrior(centre - 3 * span,
                                            centre + 3 * span)
    # σ_RV jitter — log-uniform over plausible bounds. ESPRESSO is
    # capable of ~0.5 m/s so the lower bound is small; CORALIE/PFS
    # have ~10–20 m/s native scatter so 50 m/s is a safe upper.
    priors["sigma_$name"] = ModJeffreysPrior(0.1, 50.0)
end

priors["offset_TESS"]   = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
priors["jitter_TESS"]   = LogUniformPrior(1e-5, 5e-3)
priors["q1_TESS"]       = UniformPrior(0.0, 1.0)
priors["q2_TESS"]       = UniformPrior(0.0, 1.0)

# Stellar density — Bonomo+ 2017 give M_s = 1.04, R_s = 1.137; ρ ≈ 1.0
# with a generous prior so the transit fit drives it.
priors["rho_s"] = NormalPrior(1.00, 0.30, 0.1, 5.0)

parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)

params = Params(;
    max_kplanet     = 4,
    planet_modes    = [RVPM, RVPM, RVPM, RV_ONLY],
    instruments     = ic,
    data            = data,
    M_s             = 1.04,
    R_s             = 1.137,
    parametrization = parametrization,
    priors          = priors,
    trend_order     = 0,
    stability       = :amd,
    external_priors = [ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)],
)
@printf("Free params: %d (max_kplanet = %d, modes = [RVPM, RVPM, RVPM, RV_ONLY])\n",
        n_unfrozen(params), params.config.max_kplanet)

target = NereusTarget(params, data; unconstrained = true)

# ---- Trans-dim config -----------------------------------------------
# Planet birth/death enabled. No noise toggling — keep the search
# clean (just K > 0 vs K = 0 selection per slot). InformedBirth
# proposes periods from a Lomb-Scargle periodogram of the residuals,
# critical for blind search on broad log-uniform priors where
# PriorBirth alone has near-zero acceptance. Mix with PriorBirth to
# preserve dimension-jump validity.
td = TransDimConfig(
    max_kplanet       = 4,
    planets           = true,
    noise             = false,
    birth_strategies  = [PriorBirth(), InformedBirth()],
    birth_weights     = [0.3, 0.7],
    transdim_fraction = 0.3,
)

# ---- Budgets (overridable) -------------------------------------------
const SEED      = parse(Int,   get(ENV, "WASP47_SEED",      "42"))
const N_WARMUP  = parse(Int,   get(ENV, "WASP47_WARMUP",    "8000"))
const N_SAMPLES = parse(Int,   get(ENV, "WASP47_SAMPLES",   "30000"))
# MoMS has the highest per-iteration cost on broad-prior blind search
# (its M-H proposal is a Gaussian RW around `off_values`, which on
# log-uniform priors covering 1.5 decades is mostly into low-density
# regions; acceptance rate is low). Run it with a smaller budget by
# default so the shootout doesn't take days.
const N_WARMUP_MOMS  = parse(Int, get(ENV, "WASP47_MOMS_WARMUP",  "2000"))
const N_SAMPLES_MOMS = parse(Int, get(ENV, "WASP47_MOMS_SAMPLES", "8000"))
const PT_ROUNDS = parse(Int,   get(ENV, "WASP47_PT_ROUNDS", "14"))
const PT_CHAINS = parse(Int,   get(ENV, "WASP47_PT_CHAINS", "16"))

@printf("\nBudgets:\n")
@printf("  RJMCMC: %d warmup + %d samples\n", N_WARMUP, N_SAMPLES)
@printf("  MoMS:   %d warmup + %d samples\n", N_WARMUP_MOMS, N_SAMPLES_MOMS)
@printf("  PT:     %d rounds × %d chains  (seed=%d)\n",
        PT_ROUNDS, PT_CHAINS, SEED)

# ---- Result struct + summarizer -------------------------------------
struct ShootoutResult
    name::String
    chains::MCMCChains.Chains
    n_evals::Int
    wall_time::Float64
    np_posterior::Vector{Float64}    # P(N_p = k) for k = 0..4
    modal_np::Int
    recovered::Vector{Tuple{Float64, Float64, Float64, Float64}}
                                       # (P_med, P_std, K_med, K_std), sorted by P
    ess_np::Float64
    log_Z::Float64                    # NaN unless sampler computes it
end

"""
Sort each posterior sample's active planets by period, then return the
median (P, K) at the modal N_p with planets relabeled by period rank.
This is the right way to compare blind-search chains across samplers —
slot identity is arbitrary, period rank is not.
"""
function summarize(name::String, chains, n_evals::Int, wall_time::Float64;
                    log_Z::Float64 = NaN)
    probs = model_probabilities(chains; max_kplanet = 4)
    np_post = [get(probs, k, 0.0) for k in 0:4]
    modal_np = argmax(np_post) - 1     # k=0..4

    rec = Tuple{Float64, Float64, Float64, Float64}[]
    np_vec = vec(Array(chains[:, :n_planets, :]))
    mask = np_vec .== modal_np
    if any(mask) && modal_np >= 1
        n_kept = count(mask)
        P_cols = [vec(Array(chains[:, Symbol("P_k$k"), :]))[mask] for k in 1:4]
        K_cols = [vec(Array(chains[:, Symbol("K_k$k"), :]))[mask] for k in 1:4]
        # For each posterior sample, take the modal_np shortest periods
        # (period-rank labelling). Active slots are determined by k <= n_p
        # under the slot-1-first convention used by trans-dim moves;
        # however, with a broad-prior search the data picks ANY slot, so
        # we sort by period and take the lowest modal_np values per sample.
        P_sorted = Vector{Vector{Float64}}(undef, modal_np)
        K_sorted = Vector{Vector{Float64}}(undef, modal_np)
        for j in 1:modal_np
            P_sorted[j] = Float64[]
            K_sorted[j] = Float64[]
        end
        for s in 1:n_kept
            Ps = [P_cols[k][s] for k in 1:4]
            Ks = [K_cols[k][s] for k in 1:4]
            order = sortperm(Ps)
            for j in 1:modal_np
                push!(P_sorted[j], Ps[order[j]])
                push!(K_sorted[j], Ks[order[j]])
            end
        end
        for j in 1:modal_np
            push!(rec, (median(P_sorted[j]), std(P_sorted[j]),
                         median(K_sorted[j]), std(K_sorted[j])))
        end
    end

    ess_val = try
        ess_tbl = MCMCChains.ess_rhat(chains[:, [:n_planets], :])
        Float64(ess_tbl[1, :ess])
    catch
        NaN
    end
    return ShootoutResult(name, chains, n_evals, wall_time,
                            np_post, modal_np, rec, ess_val, log_Z)
end

results = ShootoutResult[]

# ---- 1. sample_rjmcmc ------------------------------------------------
println("\n" * "="^70)
println("[1/4] sample_rjmcmc")
println("="^70)
let
    t0 = time()
    chains, n_evals = sample_rjmcmc(target, target.data;
        td        = td,
        n_samples = N_SAMPLES,
        n_warmup  = N_WARMUP,
        seed      = SEED,
        n_chains  = 4,   # 4 independent chains in parallel via Threads.@spawn
    )
    dt = time() - t0
    push!(results, summarize("RJMCMC", chains, n_evals, dt))
    @printf("Done in %.1f min, %d evals\n", dt / 60, n_evals)
end

# ---- 2. sample_pt ----------------------------------------------------
println("\n" * "="^70)
println("[2/4] sample_pt (trans-dim, Pathfinder warmup)")
println("="^70)
let
    t0 = time()
    chains, _, n_evals, _ = sample_pt_warm(target;
        n_pathfinder_runs    = PT_CHAINS,
        n_pathfinder_draws   = max(2 * PT_CHAINS, 200),
        td                   = td,
        n_rounds             = PT_ROUNDS,
        n_chains             = PT_CHAINS,
        seed                 = SEED,
        show_report          = true,
        within_model         = :rwm,
        early_stop_thresh    = 0.001,
        early_stop_min_rounds = 12,
    )
    dt = time() - t0
    push!(results, summarize("PT", chains, n_evals, dt))
    @printf("Done in %.1f min, %d evals\n", dt / 60, n_evals)
end

# ---- 3. sample_moms --------------------------------------------------
println("\n" * "="^70)
println("[3/4] sample_moms")
println("="^70)
let
    t0 = time()
    chains, n_evals, _ = sample_moms(target, target.data;
        td        = td,
        n_samples = N_SAMPLES_MOMS,
        n_warmup  = N_WARMUP_MOMS,
        seed      = SEED,
        n_chains  = 4,   # 4 independent chains in parallel
        init_scale = 1.0,
        within_model = :rwm,
        informed_birth_fraction = 0.5,
    )
    dt = time() - t0
    push!(results, summarize("MoMS", chains, n_evals, dt))
    @printf("Done in %.1f min, %d evals\n", dt / 60, n_evals)
end

# ---- 4. sample_moms_ns -----------------------------------------------
println("\n" * "="^70)
println("[4/4] sample_moms_ns")
println("="^70)
let
    n_live = parse(Int, get(ENV, "WASP47_NS_LIVE", "400"))
    dlogz  = parse(Float64, get(ENV, "WASP47_NS_DLOGZ", "0.5"))
    n_mcmc = parse(Int, get(ENV, "WASP47_NS_MCMC", "30"))
    t0 = time()
    chains, log_Z, _ = sample_moms_ns(target, target.data;
        td             = td,
        n_live         = n_live,
        dlogz          = dlogz,
        n_mcmc         = n_mcmc,
        seed           = SEED,
        informed_birth_fraction = 0.5,
        # Batched live-point replacement: kill batch_size worst points
        # per iter and evolve them concurrently via Threads.@spawn.
        # n_live=400 / batch_size=8 gives 8× wall-clock speedup with
        # negligible bias (batch_size << n_live keeps Skilling's NS
        # recursion accurate). Standard dynesty-style batching.
        batch_size = 8,
    )
    dt = time() - t0
    # MoMS-NS does not return a likelihood-eval counter. Approximate
    # from n_iter × n_mcmc + initial n_live so the comparison column
    # has a meaningful value alongside the other samplers.
    n_iter_ns = size(chains, 1) - n_live
    n_evals_ns = max(n_iter_ns, 0) * n_mcmc + n_live
    push!(results, summarize("MoMS-NS", chains, n_evals_ns, dt; log_Z = log_Z))
    @printf("Done in %.1f min, ~%d evals, log Z = %+.2f\n",
            dt / 60, n_evals_ns, log_Z)
end

# ---- Comparison report ----------------------------------------------
println("\n" * "="^70)
println("WASP-47 joint blind-search shootout — summary")
println("="^70)

@printf("\n%-10s %12s %14s %10s %12s %10s\n",
        "Sampler", "wall (min)", "log-L evals", "ESS(N_p)", "modal N_p", "log Z")
@printf("%-10s %12s %14s %10s %12s %10s\n",
        "-"^10, "-"^12, "-"^14, "-"^10, "-"^12, "-"^10)
for r in results
    logz_str = isnan(r.log_Z) ? "  --" : @sprintf("%+.2f", r.log_Z)
    @printf("%-10s %12.1f %14d %10.0f %12d %10s\n",
            r.name, r.wall_time / 60, r.n_evals, r.ess_np,
            r.modal_np, logz_str)
end

println("\nN_p posterior P(N_p = k):")
@printf("%-10s %10s %10s %10s %10s %10s\n",
        "Sampler", "k=0", "k=1", "k=2", "k=3", "k=4")
@printf("%-10s %10s %10s %10s %10s %10s\n",
        "-"^10, "-"^10, "-"^10, "-"^10, "-"^10, "-"^10)
for r in results
    @printf("%-10s %10.4f %10.4f %10.4f %10.4f %10.4f\n",
            r.name, r.np_posterior...)
end

println("\nRecovered planets at modal N_p (sorted by period):")
println("  Literature:")
@printf("    e:  P = %7.4f d  K = %6.2f m/s\n", Pe_LIT, Ke_LIT)
@printf("    b:  P = %7.4f d  K = %6.2f m/s\n", Pb_LIT, Kb_LIT)
@printf("    d:  P = %7.4f d  K = %6.2f m/s\n", Pd_LIT, Kd_LIT)
@printf("    c:  P = %7.4f d  K = %6.2f m/s\n", Pc_LIT, Kc_LIT)

for r in results
    println("  $(r.name) (modal N_p = $(r.modal_np)):")
    if isempty(r.recovered)
        println("    (no samples at modal N_p — sampler degenerate)")
    else
        for (j, (Pm, Ps, Km, Ks)) in enumerate(r.recovered)
            @printf("    rank %d:  P = %7.4f ± %6.4f d   K = %6.2f ± %5.2f m/s\n",
                    j, Pm, Ps, Km, Ks)
        end
    end
end

println("\nBayes factors BF(N_p = k vs N_p = 0), uniform model prior:")
@printf("%-10s %12s %12s %12s %12s\n",
        "Sampler", "BF(1 vs 0)", "BF(2 vs 0)", "BF(3 vs 0)", "BF(4 vs 0)")
@printf("%-10s %12s %12s %12s %12s\n",
        "-"^10, "-"^12, "-"^12, "-"^12, "-"^12)
for r in results
    bfs = try
        bayes_factors(r.chains; reference = 0)
    catch
        nothing
    end
    fmt(b) = (b !== nothing && isfinite(b) && b > 0) ?
              @sprintf("%10.2e", b) : "  --"
    if bfs === nothing
        @printf("%-10s %12s %12s %12s %12s   (reference unsampled)\n",
                r.name, "  --", "  --", "  --", "  --")
    else
        @printf("%-10s %12s %12s %12s %12s\n",
                r.name,
                fmt(get(bfs, 1, 0.0)),
                fmt(get(bfs, 2, 0.0)),
                fmt(get(bfs, 3, 0.0)),
                fmt(get(bfs, 4, 0.0)))
    end
end

# Save chains
for r in results
    path = joinpath(OUT_DIR, "$(lowercase(replace(r.name, "-" => "_")))_chains.nc")
    save_chains(path, r.chains, target.params; data = target.data)
end
@printf("\nChains saved under %s/\n", OUT_DIR)
println("="^70)
