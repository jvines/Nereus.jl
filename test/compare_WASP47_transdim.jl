#!/usr/bin/env julia
# WASP-47 — trans-dim sampler shootout.
#
# Runs the four trans-dim samplers Nereus supports on the same target,
# same data, same TransDimConfig:
#   1. sample_rjmcmc — standalone RJMCMC (Green 1995) with PriorBirth
#   2. sample_pt     — parallel-tempering trans-dim
#   3. sample_moms   — Mixtures of Mutually Singular distributions
#                       (van den Bergh+ 2026, arXiv:2604.27791)
#   4. sample_moms_ns — MoMS-flavored nested sampling (Nereus native);
#                       gives a calibrated log-Z that the others do not
#
# Reports for each sampler:
#   * wall time and likelihood evals
#   * N_p posterior probabilities
#   * recovered (P, Tc) at N_p = 3
#   * ESS for n_planets (mixing quality on the trans-dim coordinate)
#   * log Z (only MoMS-NS computes this natively)
#
# Budgets are chosen so each sampler does roughly the same wall-time work;
# environment variables let the caller override.
#
# Data: WASP-47 TESS s42 (binned to 5 min). Three known transiting planets
# at 4.16, 9.03, 0.79 d (b, d, e). Trans-dim target: max_kplanet = 3,
# each slot priored on one literature period.

using Nereus
using Statistics: median, std, mean, quantile
using Printf
using MCMCChains
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results",
                            "WASP47_transdim_shootout")

const Pb_LIT, Pd_LIT, Pe_LIT     = 4.1591287, 9.030672, 0.789593
const TCb_LIT, TCd_LIT, TCe_LIT  = 2461.834, 2458.373, 2461.201

# ---- Load + bin LC ---------------------------------------------------
println("="^70)
println("WASP-47 — trans-dim sampler comparison")
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
@printf("Binned PDCSAP: %d pts, σ ≈ %.6f\n", length(flat.t), std(flat.flux))

# ---- Build target (single shared definition for all samplers) --------
target = build_target(
    M_s = 1.04, R_s = 1.137,
    planets = (
        b = (
            P  = NormalPrior(Pb_LIT, 0.005, 4.10, 4.22),
            Tc = NormalPrior(TCb_LIT, 0.01, TCb_LIT - 0.1, TCb_LIT + 0.1),
            sesinw = UniformPrior(-0.15, 0.15),
            secosw = UniformPrior(-0.15, 0.15),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.05, 0.18),
        ),
        d = (
            P  = NormalPrior(Pd_LIT, 0.01, 8.95, 9.10),
            Tc = NormalPrior(TCd_LIT, 0.02, TCd_LIT - 0.2, TCd_LIT + 0.2),
            sesinw = UniformPrior(-0.15, 0.15),
            secosw = UniformPrior(-0.15, 0.15),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.005, 0.05),
        ),
        e = (
            P  = NormalPrior(Pe_LIT, 0.001, 0.78, 0.80),
            Tc = NormalPrior(TCe_LIT, 0.01, TCe_LIT - 0.05, TCe_LIT + 0.05),
            sesinw = UniformPrior(-0.05, 0.05),
            secosw = UniformPrior(-0.05, 0.05),
            b  = UniformPrior(0.0, 0.95),
            rr = UniformPrior(0.005, 0.025),
        ),
    ),
    phot = (TESS = (
        data    = flat,
        jitter  = LogUniformPrior(1e-5, 1e-3),
        offset  = NormalPrior(0.0, 1e-3, -0.01, 0.01),
        q1      = UniformPrior(0.0, 1.0),
        q2      = UniformPrior(0.0, 1.0),
    ),),
)
@printf("Free params: %d (max_kplanet = %d)\n",
        n_unfrozen(target.params), target.params.config.max_kplanet)

td = TransDimConfig(
    max_kplanet       = 3,
    planets           = true,
    noise             = false,
    birth_strategies  = [PriorBirth()],
    birth_weights     = [1.0],
    transdim_fraction = 0.3,
)

# ---- Budgets (overridable via env vars) ------------------------------
const SEED      = parse(Int,   get(ENV, "WASP47_SEED",      "42"))
const N_WARMUP  = parse(Int,   get(ENV, "WASP47_WARMUP",    "5000"))
const N_SAMPLES = parse(Int,   get(ENV, "WASP47_SAMPLES",   "15000"))
const PT_ROUNDS = parse(Int,   get(ENV, "WASP47_PT_ROUNDS", "10"))
const PT_CHAINS = parse(Int,   get(ENV, "WASP47_PT_CHAINS", "10"))

@printf("\nBudgets: RJMCMC/MoMS = %d warmup + %d samples ; PT = %d rounds × %d chains (seed=%d)\n",
        N_WARMUP, N_SAMPLES, PT_ROUNDS, PT_CHAINS, SEED)

# ---- Helper: extract per-sampler stats from a chain -----------------
struct ShootoutResult
    name::String
    chains::MCMCChains.Chains
    n_evals::Int
    wall_time::Float64
    np_posterior::Vector{Float64}    # P(N_p = k) for k = 0..3
    recovered::Dict{Symbol, Tuple{Float64, Float64}}  # period quantiles at N_p=3
    ess_np::Float64
    log_Z::Float64                    # NaN unless sampler computes it natively
end

function summarize(name, chains, n_evals, wall_time; log_Z::Float64 = NaN)
    # Use the model_probabilities helper from Nereus so the column
    # always covers k=0..3 even when the sampler never visited some k.
    probs = model_probabilities(chains; max_kplanet=3)
    np_post = [get(probs, k, 0.0) for k in 0:3]

    np_vec = vec(Array(chains[:, :n_planets, :]))
    mask3 = np_vec .== 3
    rec = Dict{Symbol, Tuple{Float64, Float64}}()
    if any(mask3)
        for k in 1:3
            sym = Symbol("P_k$k")
            sym in names(chains, :parameters) || continue
            P_v = vec(Array(chains[:, sym, :]))[mask3]
            isempty(P_v) && continue
            rec[sym] = (median(P_v), std(P_v))
        end
    end
    ess_val = try
        ess_tbl = MCMCChains.ess_rhat(chains[:, [:n_planets], :])
        Float64(ess_tbl[1, :ess])
    catch
        NaN
    end
    return ShootoutResult(name, chains, n_evals, wall_time,
                           np_post, rec, ess_val, log_Z)
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
    )
    dt = time() - t0
    push!(results, summarize("RJMCMC", chains, n_evals, dt))
    @printf("Done in %.1f min, %d evals\n", dt / 60, n_evals)
end

# ---- 2. sample_pt ----------------------------------------------------
println("\n" * "="^70)
println("[2/4] sample_pt (trans-dim)")
println("="^70)
let
    t0 = time()
    chains, _, n_evals = sample_pt(target;
        td          = td,
        n_rounds    = PT_ROUNDS,
        n_chains    = PT_CHAINS,
        seed        = SEED,
        show_report = false,
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
        n_samples = N_SAMPLES,
        n_warmup  = N_WARMUP,
        seed      = SEED,
        init_scale = 1.0,
    )
    dt = time() - t0
    push!(results, summarize("MoMS", chains, n_evals, dt))
    @printf("Done in %.1f min, %d evals\n", dt / 60, n_evals)
end

# ---- 4. sample_moms_ns ------------------------------------------------
# MoMS-NS computes a calibrated log-Z natively, supporting Bayes factor
# comparison between dimensionalities. Other samplers in this shootout
# only return N_p posterior frequencies. Wall time depends on dlogz +
# n_live; tuned to roughly match the others.
println("\n" * "="^70)
println("[4/4] sample_moms_ns (nested sampling)")
println("="^70)
let
    n_live = parse(Int, get(ENV, "WASP47_NS_LIVE", "300"))
    dlogz  = parse(Float64, get(ENV, "WASP47_NS_DLOGZ", "0.5"))
    n_mcmc = parse(Int, get(ENV, "WASP47_NS_MCMC", "25"))
    t0 = time()
    chains, log_Z, _ = sample_moms_ns(target, target.data;
        td             = td,
        n_live         = n_live,
        dlogz          = dlogz,
        n_mcmc         = n_mcmc,
        seed           = SEED,
        show_progress  = true,
        progress_every = 200,
    )
    dt = time() - t0
    # Pseudo-eval count: NS evaluates one log-L per MCMC step; we don't
    # have the exact total without instrumenting — report iter count.
    push!(results, summarize("MoMS-NS", chains, length(chains.value), dt;
                              log_Z = log_Z))
    @printf("Done in %.1f min, log Z = %.3f\n", dt / 60, log_Z)
end

# ---- Comparison report ------------------------------------------------
mkpath(OUT_DIR)
println("\n" * "="^70)
println("WASP-47 trans-dim sampler comparison")
println("="^70)
@printf("\n%-10s %10s %14s %10s %10s\n",
        "Sampler", "wall (min)", "log-L evals", "ESS(N_p)", "log Z")
@printf("%-10s %10s %14s %10s %10s\n", "-"^10, "-"^10, "-"^14, "-"^10, "-"^10)
for r in results
    logz_str = isnan(r.log_Z) ? "  --" : @sprintf("%+.2f", r.log_Z)
    @printf("%-10s %10.1f %14d %10.0f %10s\n",
            r.name, r.wall_time / 60, r.n_evals, r.ess_np, logz_str)
end

println("\nN_p posterior P(N_p = k):")
@printf("%-10s %10s %10s %10s %10s\n", "Sampler", "k=0", "k=1", "k=2", "k=3")
@printf("%-10s %10s %10s %10s %10s\n", "-"^10, "-"^10, "-"^10, "-"^10, "-"^10)
for r in results
    @printf("%-10s %10.3f %10.3f %10.3f %10.3f\n",
            r.name, r.np_posterior...)
end

println("\nRecovered periods at N_p = 3 (median ± stddev):")
for r in results
    println("  $(r.name):")
    for k in 1:3
        sym = Symbol("P_k$k")
        if haskey(r.recovered, sym)
            μ, σ = r.recovered[sym]
            @printf("    P_k%d = %.5f ± %.5f d\n", k, μ, σ)
        end
    end
end

println("\nBayes factors BF(N_p = k vs N_p = 0), uniform model prior:")
@printf("%-10s %12s %12s %12s\n", "Sampler", "BF(1 vs 0)", "BF(2 vs 0)", "BF(3 vs 0)")
@printf("%-10s %12s %12s %12s\n", "-"^10, "-"^12, "-"^12, "-"^12)
for r in results
    bfs = try
        bayes_factors(r.chains; reference=0)
    catch e
        # If P(N_p=0) is zero or k=0 isn't observed, skip with note.
        nothing
    end
    if bfs === nothing
        @printf("%-10s %12s %12s %12s   (N_p=0 unsampled)\n",
                r.name, "  --", "  --", "  --")
    else
        function fmt(b)
            isfinite(b) && b > 0 ? @sprintf("%10.2e", b) : "  --"
        end
        @printf("%-10s %12s %12s %12s\n",
                r.name, fmt(get(bfs, 1, 0.0)),
                       fmt(get(bfs, 2, 0.0)),
                       fmt(get(bfs, 3, 0.0)))
    end
end

# Save chains for offline analysis
for r in results
    save_chains(joinpath(OUT_DIR, "$(lowercase(r.name))_chains.nc"),
                 r.chains, target.params; data = target.data)
end
@printf("\nChains saved under %s/\n", OUT_DIR)
println("="^70)
