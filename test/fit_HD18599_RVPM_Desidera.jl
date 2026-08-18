#!/usr/bin/env julia
# HD 18599 — fixed-config replication of DESIDERA+ 2023 (A&A,
# arxiv:2210.07933, "TOI-179: a young system with a transiting compact
# Neptune-mass planet ...").
#
# Methodology differences from fit_HD18599_RVPM_Desidera.jl (Vines/Nereus
# default), per Desidera Table 3:
#   - **GP RV activity** (CeleriteRotation, the two-SHO QPO kernel)
#     instead of linear BIS decorrelation.
#   - **No external eccentricity prior** N(0, 0.3²). Only the box
#     U(-1, 1) on √e·cos(ω), √e·sin(ω).
#   - **γ̇ ∈ U(0, 0.1)** (positive only, narrow) instead of U(-1, 1).
#   - No BIS indicator wiring.
#
# Desidera headline numbers to reproduce:
#   K_b   = 11.3 +3.3 / -3.6 m/s  (3.3σ detection)
#   m_b   = 24.1 +7.1 / -7.7 M_⊕
#   e_b   = 0.34 +0.07 / -0.09   (3.8σ from zero)
#   P_b   = 4.1374354 d
#
# Hypothesis being tested: Vines+ 2023 and our HD18599 Run 2 land at
# K~11 / e~0 and K~6 / e~0 because the N(0, 0.3²) external e prior
# tips the K-e joint posterior toward the circular mode. Without that
# prior, the K~11.3 / e~0.34 mode Desidera reports should be recoverable.

using Nereus
using DelimitedFiles
using Statistics: median, mean, std
using Printf
using MCMCChains
using Random

const REPO_ROOT  = abspath(joinpath(@__DIR__, "..", ".."))
const RV_FILE    = joinpath(@__DIR__, "data", "hd18599.csv")
const LC_FILE    = joinpath(REPO_ROOT, "data", "HD18599",
                             "HD18599_cleaned_lc.csv")
const OUT_DIR    = joinpath(REPO_ROOT, "Nereus.jl", "results",
                             "HD18599_RVPM_Desidera")

const P_REF       = 4.1374685534602405
const T0_REF      = 2.4583545857470357e6
const DURATION_D  = 0.067
const M_S, R_S    = 0.807, 0.798
const T_EFF       = 5290.0
const J_MAG       = 8.064
const K_MAG       = 7.629
const A_B         = 0.3

const PAPER_INSTRUMENTS = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

println("="^70)
println("HD 18599 — Desidera+ 2023 replication (GP RV activity, no ext e prior)")
println("="^70)

# ---- RV load -----------------------------------------------------------
raw      = readdlm(RV_FILE, ',', Any, '\n'; header = true)
data_mat = raw[1]
inst_raw = String.(data_mat[:, 16])
prov_raw = String.(data_mat[:, 17])
keep     = Int[]
inst_str = String[]
for i in 1:size(data_mat, 1)
    ins  = strip(inst_raw[i])
    prov = strip(prov_raw[i])
    if ins == "HARPS_POST" && prov == "ESO_PHASE3"
        continue
    end
    ins in PAPER_INSTRUMENTS || continue
    push!(keep, i)
    push!(inst_str, ins)
end
data_mat = data_mat[keep, :]
bjd      = Float64.(data_mat[:, 1])
rv       = Float64.(data_mat[:, 2])
rv_err   = Float64.(data_mat[:, 3])
inst_names = sort!(unique(inst_str))
inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst    = [inst_map[s] for s in inst_str]

function mad_outlier_mask(values::Vector{Float64}, inst::Vector{Int};
                            k::Float64 = 5.0)
    keep = trues(length(values))
    for ins_id in unique(inst)
        idx = findall(==(ins_id), inst)
        v_i = view(values, idx)
        med = median(v_i)
        mad = median(abs.(v_i .- med))
        mad == 0 && continue
        sigma = 1.4826 * mad
        for k_ in idx
            if abs(values[k_] - med) > k * sigma
                keep[k_] = false
            end
        end
    end
    return keep
end
keep_rv = mad_outlier_mask(rv, rv_inst; k = 5.0)
n_drop  = count(.!keep_rv)
n_drop > 0 && @printf("Dropped %d / %d RVs as 5σ MAD outliers\n", n_drop, length(rv))
data_mat = data_mat[keep_rv, :]
bjd      = bjd[keep_rv]
rv       = rv[keep_rv]
rv_err   = rv_err[keep_rv]
rv_inst  = rv_inst[keep_rv]
inst_str = inst_str[keep_rv]
@printf("Loaded %d RVs across %d instruments (HARPS_PRE, HARPS_POST, FEROS)\n",
         length(bjd), length(inst_names))

# ---- BIS per-instrument normalization (EMPEROR convention, exact) ----
# astroEMPEROR `emperors_library.py:read_rv_data` (lines 251-260):
#   1. Compute per-instrument RV mean: μ_rv = mean(RV_inst)
#   2. Mean-subtract per instrument: RV_inst -= μ_rv  (and store μ_rv as γ)
#   3. process_activities() on indicators of this instrument:
#         a. replace -999 (missing) with column median
#         b. mean-subtract each indicator column
#         c. divide by max(|processed|) → peak ±1
#   4. Rescale: acts *= max(|RV_inst_mean_subtracted|)
#                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                            crucial: it's the MEAN-SUBTRACTED RV peak,
#                            not the raw RV peak, because γ has already
#                            been absorbed at this point.
# This puts the activity indicator on the same scale as the RV residual
# variability. Using raw RV (γ-dominated) inflates BIS by ~γ/σ_RV and
# lets the BIS regression eat the Keplerian signal.
function per_instrument_normalize!(v::Vector{Float64},
                                    rv_vec::Vector{Float64},
                                    inst::Vector{Int})
    for ins_id in unique(inst)
        idx = findall(==(ins_id), inst)
        finite_idx = filter(k -> isfinite(v[k]), idx)
        isempty(finite_idx) && continue
        # Step 2 above — center indicator
        μ = mean(@view v[finite_idx])
        for k in finite_idx
            v[k] -= μ
        end
        v_max = maximum(abs, @view v[finite_idx])
        v_max == 0 && continue
        # Step 4 — rescale to MEAN-SUBTRACTED RV peak (EMPEROR convention)
        rv_inst = @view rv_vec[idx]
        rv_mu = mean(rv_inst)
        rv_max_ins = maximum(abs, rv_inst .- rv_mu)
        rv_max_ins == 0 && continue
        for k in finite_idx
            v[k] = v[k] / v_max * rv_max_ins
        end
    end
    return v
end
ind_bis = Float64[
    let v = data_mat[i, 4]
        v === "" || (v isa AbstractString && strip(v) == "") ? NaN : Float64(v)
    end
    for i in 1:size(data_mat, 1)
]
per_instrument_normalize!(ind_bis, rv, rv_inst)

# ---- LC load + bin + transit-windowed subsample -----------------------
LC_BIN_MIN = 5.0
OOT_STRIDE = 200
lc_raw = load_tess_lc(LC_FILE)
lc = let bd = LC_BIN_MIN / (60 * 24)
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
half_window = 2 * DURATION_D
n_lc        = length(lc.t)
n_per_orbit = ceil(Int, (lc.t[end] - lc.t[1]) / P_REF) + 2
in_transit  = falses(n_lc)
for k in -1:(n_per_orbit + 1)
    tc = T0_REF + k * P_REF
    @inbounds for i in 1:n_lc
        if abs(lc.t[i] - tc) < half_window
            in_transit[i] = true
        end
    end
end
oot_keep = falses(n_lc)
let oot_count = 0
    for i in 1:n_lc
        in_transit[i] && continue
        oot_count += 1
        oot_keep[i] = (oot_count % OOT_STRIDE == 0)
    end
end
keep_lc = in_transit .| oot_keep
t_phot   = lc.t[keep_lc]
flux     = lc.flux[keep_lc]
flux_err = lc.flux_err[keep_lc]
phot_inst = ones(Int, length(t_phot))

# ---- Build joint Data + Params ---------------------------------------
# No BIS indicator wiring — GP handles activity instead.
data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    t_phot = t_phot, flux = flux, flux_err = flux_err, phot_inst = phot_inst,
)
ic = InstrumentConfig(rv = inst_names, pm = ["TESS"])

# Desidera GP RV activity: quasi-periodic kernel (their h, λ, w, θ).
# Nereus's `CeleriteRotation` is a two-SHO QPO approximation —
# closest available match. Single global GP applied to all RV inst.
rv_gp = CeleriteRotation(channel = :rv)

rv_max = maximum(abs, rv)
priors = Dict{String, PriorSpec}()
priors["P_k1"]      = NormalPrior(P_REF, 1e-3, P_REF - 0.01, P_REF + 0.01)
priors["K_k1"]      = UniformPrior(0.0, 50.0)
# Desidera Table 3: √e·cos(ω), √e·sin(ω) on U(-1, 1) — full box, no
# external eccentricity prior. The high-e mode (e ≈ 0.34) lives near
# √e·cos(ω) ≈ 0.56, which our previous U(-0.7, 0.7) box reached but
# the external N(0, 0.3²) prior penalized.
priors["sesinw_k1"] = UniformPrior(-1.0, 1.0)
priors["secosw_k1"] = UniformPrior(-1.0, 1.0)
priors["Tc_k1"]     = NormalPrior(T0_REF, 0.05, T0_REF - 0.5, T0_REF + 0.5)
priors["b_k1"]      = UniformPrior(0.0, 1.0)
priors["rr_k1"]     = NormalPrior(0.031, 0.005, 0.005, 0.10)
for name in inst_names
    priors["gamma_$name"] = UniformPrior(-3 * rv_max, 3 * rv_max)
end
# Desidera tightens γ̇ to U(0, 0.1) m/s/d (positive-only).
priors["dvdt"]      = UniformPrior(0.0, 0.1)

# GP rv hyperparams: anchor period prior on Vines+ 2023's P_rot ≈ 8.74 d.
# σ (amplitude) ranges 1 mm/s to 5x rv_max for headroom.
# Q0 is the primary mode's quality factor — high Q ⇒ narrowband rotation.
# Tightened to keep the celerite Cholesky positive-definite during
# Pathfinder LBFGS exploration. Loose bounds give PosDefException.
priors["gp_sigma"]  = LogUniformPrior(0.1, rv_max)
priors["gp_period"] = NormalPrior(8.74, 0.5, 6.0, 12.0)
priors["gp_Q0"]     = LogUniformPrior(1.0, 10.0)
priors["gp_dQ"]     = UniformPrior(0.0, 3.0)
priors["gp_f"]      = UniformPrior(0.05, 0.5)

priors["offset_TESS"]  = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
priors["jitter_TESS"]  = LogUniformPrior(1e-5, 5e-3)
priors["q1_TESS"]      = UniformPrior(0.0, 1.0)
priors["q2_TESS"]      = UniformPrior(0.0, 1.0)
parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
priors["rho_s"] = NormalPrior(2.241, 0.479, 0.1, 10.0)

params = Params(;
    max_kplanet     = 1,                # FIXED N_p = 1
    planet_modes    = [RVPM],
    instruments     = ic,
    data            = data,
    M_s             = M_S,
    R_s             = R_S,
    parametrization = parametrization,
    priors          = priors,
    noise_models    = [rv_gp],          # GP RV activity (no linear BIS, no ARMA)
    transdim_noise  = false,            # FIXED noise
    trend_order     = 1,
    # Desidera uses NO external eccentricity prior — only the U(-1,1)
    # box on (sesinw, secosw). Removing N(0, 0.3²) lets the high-e
    # mode (e ≈ 0.34) be data-driven.
)
target = NereusTarget(params, data; unconstrained = true)
@printf("\nModel: N_p=1 fixed, activities only (paper Run 2), %d unfrozen params\n",
         n_unfrozen(params))

# ---- PT sampler -------------------------------------------------------
n_rounds = parse(Int, get(ENV, "HD18599_ROUNDS", "16"))
n_chains = parse(Int, get(ENV, "HD18599_CHAINS", "16"))
seed     = parse(Int, get(ENV, "HD18599_SEED",   "777"))
@printf("Starting PT: %d rounds × %d chains (seed=%d)\n", n_rounds, n_chains, seed)

# Fixed-dim run — sample_pt now routes through the optimised in-house
# PT (no Pigeons), so we just pass td=nothing (the default) and get the
# zero-alloc threaded RWM hot path automatically. Reddemcee TI+/SS+/H+
# evidence is reported in the @info block of the sampler.
#
# HD18599_REPLOT_ONLY=1 skips sampling and just loads chains.nc to
# regenerate plots/summaries with the current code.
replot_only = parse(Bool, get(ENV, "HD18599_REPLOT_ONLY", "false"))

if replot_only
    chain_path = joinpath(OUT_DIR, "chains.nc")
    isfile(chain_path) || error(
        "REPLOT_ONLY=1 but no chains.nc at $chain_path — run the fit at " *
        "least once first.")
    @info "Loading saved chains from $chain_path (sampler skipped)"
    chains, _ = load_chains(chain_path)
    log_ev = -1.0
else
    # `HD18599_SAMPLER=ptemcee` switches from the default Pigeons-style
    # PT (single walker per temperature) to the Vousden+ 2016 ensemble
    # PT used in the original paper. The single-walker cold chain
    # exhibits mode-trapping for HD 18599's K-activity-ecc degeneracy
    # (K~6 vs paper K=11); the ensemble within each temperature is the
    # algorithmic fix.
    sampler_choice = get(ENV, "HD18599_SAMPLER", "pt")
    t0 = time()
    if sampler_choice == "ptemcee"
        n_walkers = parse(Int, get(ENV, "HD18599_N_WALKERS", "100"))
        n_steps   = parse(Int, get(ENV, "HD18599_N_STEPS",   "1500"))
        n_burnin  = parse(Int, get(ENV, "HD18599_N_BURNIN",  "500"))
        n_temps   = parse(Int, get(ENV, "HD18599_N_TEMPS",   "5"))
        init_strat = Symbol(get(ENV, "HD18599_PTEMCEE_INIT", "map_scatter"))
        @printf("Running ptemcee: %d temps × %d walkers × %d steps (burnin %d, init=%s)\n",
                n_temps, n_walkers, n_steps, n_burnin, init_strat)
        res = sample_ptemcee(target, data;
            n_temps       = n_temps,
            n_walkers     = n_walkers,
            n_steps       = n_steps,
            n_burnin      = n_burnin,
            init_strategy = init_strat,
            seed          = seed,
        )
        chains = res.chains
        log_ev = res.log_evidence
        @printf("ptemcee done in %.1f min (log Z = %.2f, n_evals = %d)\n",
                 (time() - t0) / 60, log_ev, res.n_evals)
        @printf("  acceptance_within (per β) = %s\n",
                 join(map(a -> @sprintf("%.2f", a), res.acceptance_within), ", "))
        @printf("  acceptance_swap   (k,k+1) = %s\n",
                 join(map(a -> @sprintf("%.2f", a), res.acceptance_swap), ", "))
        ev = res.evidence
        @printf("  Evidence stack (TI / TI+ / SS+ / H+):\n")
        @printf("    TI  (trapezoidal) = %12.4f\n", ev.ti[1])
        @printf("    TI+ (PCHIP)       = %12.4f ± %.4f\n", ev.ti_plus[1], ev.ti_plus[2])
        @printf("    SS+ (geom-bridge) = %12.4f ± %.4f\n", ev.ss_plus[1], ev.ss_plus[2])
        @printf("    H+  (β* = %.4f)  = %12.4f ± %.4f\n",
                 ev.hybrid_beta_star, ev.hybrid[1], ev.hybrid[2])
        spread = maximum([ev.ti[1], ev.ti_plus[1], ev.ss_plus[1], ev.hybrid[1]]) -
                 minimum([ev.ti[1], ev.ti_plus[1], ev.ss_plus[1], ev.hybrid[1]])
        @printf("    Spread across estimators = %.4f nats\n", spread)
    elseif sampler_choice == "whitening"
        # NF-coupled PT with diagonal-Gaussian whitening swap proposals.
        n_temps  = parse(Int, get(ENV, "HD18599_N_TEMPS",  "8"))
        n_steps  = parse(Int, get(ENV, "HD18599_N_STEPS",  "2000"))
        n_burnin = parse(Int, get(ENV, "HD18599_N_BURNIN", "500"))
        @printf("Running pt_whitening: %d temps × %d steps (burnin %d)\n",
                n_temps, n_steps, n_burnin)
        res = sample_pt_whitening(target, data;
            n_temps  = n_temps, n_steps = n_steps, n_burnin = n_burnin,
            seed     = seed)
        chains = res.chains
        log_ev = res.log_evidence
        @printf("pt_whitening done in %.1f min (log Z = %.2f, n_evals = %d)\n",
                 (time() - t0) / 60, log_ev, res.n_evals)
        @printf("  acceptance_within = %s\n",
                 join(map(a -> @sprintf("%.2f", a), res.acceptance_within), ", "))
        @printf("  acceptance_swap   = %s\n",
                 join(map(a -> @sprintf("%.2f", a), res.acceptance_swap),   ", "))
    elseif sampler_choice == "ins"
        n_live   = parse(Int, get(ENV, "HD18599_N_LIVE",   "200"))
        dlogz    = parse(Float64, get(ENV, "HD18599_DLOGZ", "1.0"))
        max_iter = parse(Int, get(ENV, "HD18599_MAX_ITER", "30000"))
        @printf("Running nested_ins: n_live=%d  dlogz=%.2f  max_iter=%d\n",
                n_live, dlogz, max_iter)
        res = sample_nested_ins(target, data;
            n_live = n_live, dlogz = dlogz, max_iter = max_iter, seed = seed)
        chains = res.chains
        log_ev = res.log_z_ins
        @printf("nested_ins done in %.1f min (log Z_NS = %.2f, log Z_INS = %.2f, n_iters = %d, n_evals = %d)\n",
                 (time() - t0) / 60, res.log_z_ns, res.log_z_ins,
                 res.n_iters, res.n_evals)
    elseif sampler_choice == "dyn_ns"
        n_live_init  = parse(Int, get(ENV, "HD18599_N_LIVE_INIT",  "100"))
        n_live_batch = parse(Int, get(ENV, "HD18599_N_LIVE_BATCH", "300"))
        dlogz_init   = parse(Float64, get(ENV, "HD18599_DLOGZ", "1.0"))
        max_iter     = parse(Int, get(ENV, "HD18599_MAX_ITER", "30000"))
        @printf("Running nested_dynamic: n_live_init=%d  n_live_batch=%d  dlogz_init=%.2f\n",
                n_live_init, n_live_batch, dlogz_init)
        res = sample_nested_dynamic(target, data;
            n_live_init = n_live_init, n_live_batch = n_live_batch,
            dlogz_init = dlogz_init, max_iter = max_iter, seed = seed)
        chains = res.chains
        log_ev = res.log_z
        @printf("nested_dynamic done in %.1f min (log Z = %.2f, baseline=%.2f batch=%.2f, n_evals = %d)\n",
                 (time() - t0) / 60, res.log_z, res.log_z_baseline,
                 res.log_z_batch, res.n_evals)
    elseif sampler_choice == "smc"
        # SMC interface (delegates to PA with adaptive ESS scheduling).
        n_replicas = parse(Int, get(ENV, "HD18599_N_REPLICAS", "300"))
        n_mcmc     = parse(Int, get(ENV, "HD18599_N_MCMC",     "10"))
        n_betas    = parse(Int, get(ENV, "HD18599_N_BETAS",    "30"))
        βs = collect(range(0.0, 1.0; length = n_betas))
        @printf("Running sample_smc: %d β-steps × %d replicas × %d mcmc\n",
                n_betas, n_replicas, n_mcmc)
        # `:adaptive_cov` was added to handle ≥20-d like HD 18599 where
        # partner-based stretch can't break clone clusters after
        # resampling. Uses empirical-covariance Gaussian proposals.
        kernel = Symbol(get(ENV, "HD18599_SMC_KERNEL", "adaptive_cov"))
        res = sample_smc(target, data;
            betas      = βs,
            n_replicas = n_replicas,
            n_mcmc     = n_mcmc,
            seed       = seed,
            mutation_kernel = kernel)
        chains = res.chains
        log_ev = res.log_evidence
        @printf("sample_smc done in %.1f min (log Z = %.2f, n_evals = %d)\n",
                 (time() - t0) / 60, log_ev, res.n_evals)
    else
        chains, log_ev, _ = sample_pt_warm(target;
            n_pathfinder_runs    = 16,
            n_pathfinder_draws   = max(2 * n_chains, 200),
            n_rounds             = n_rounds,
            n_chains             = n_chains,
            seed                 = seed,
            show_report          = true,
            within_model         = :rwm,
        )
        @printf("PT done in %.1f min (log Z = %.2f)\n",
                 (time() - t0) / 60, log_ev)
    end
    mkpath(OUT_DIR)
    save_chains(joinpath(OUT_DIR, "chains.nc"), chains, params; data = data)
end

# ---- Diagnostic plots (Nereus plotters) ------------------------------
println("\nGenerating plots ...")
plot_trace(chains, params; output = OUT_DIR)
plot_histograms(chains, params; output = OUT_DIR)
plot_rv_timeseries(chains, params, data; output = OUT_DIR)
plot_rv_phasefold(chains, params, data; planet = 1, output = OUT_DIR)
plot_pm_timeseries(chains, params, data; output = OUT_DIR)
plot_pm_phasefold(chains, params, data; planet = 1, output = OUT_DIR)
# Corner plot — PairPlots can hit world-age inside try/eval on a fresh
# load. Wrap in try/catch so a corner-plot failure doesn't lose the
# other plots.
try
    corner_params = String[
        "K_k1", "sesinw_k1", "secosw_k1",
        "gp_sigma", "gp_period", "gp_Q0", "dvdt",
    ]
    plot_corner(chains, params; params_to_plot = corner_params, output = OUT_DIR)
catch e
    @warn "plot_corner failed (likely PairPlots world-age compat); other plots OK" exception=e
end
@printf("Plots saved to %s\n", OUT_DIR)

# ---- Report -----------------------------------------------------------
K_samples = vec(Array(chains[:, :K_k1, :]))
K_med = median(K_samples)
K_lo, K_hi = quantile(K_samples, 0.16), quantile(K_samples, 0.84)
K_3lo, K_3hi = quantile(K_samples, 0.0027/2), quantile(K_samples, 1 - 0.0027/2)
K_iqr_half = (K_hi - K_lo) / 2
println("\n" * "="^70)
println("  HD 18599 Desidera setup (GP RV activity, no ext e prior) — K posterior")
println("="^70)
# Asymmetric 68% reporting (the only sane way to summarise a skewed
# posterior). The paper Run 2 result is K = 11 ± 3 m/s.
@printf("  K_k1 = %.2f +%.2f -%.2f m/s   (Desidera: 11.3 +3.3/-3.6)\n",
         K_med, K_hi - K_med, K_med - K_lo)
@printf("  68%% CI = [%.2f, %.2f]\n", K_lo, K_hi)
@printf("  3σ CI  = [%.2f, %.2f]\n", K_3lo, K_3hi)

# Convergence judgement: posterior is informative if both the 68%
# half-width is reasonable AND the 3σ tail isn't pegged at the prior
# upper bound (K prior is U(0, 50); 3σ at 49 means the chain is
# wandering the full prior).
K_prior_hi = 50.0
informative = K_iqr_half < 5.0 && K_3hi < 0.9 * K_prior_hi
paper_overlap = K_lo ≤ 14.0 && K_hi ≥ 8.0  # 68% CI overlaps paper [8, 14]
status = if informative && paper_overlap
    "PASS — 68%% CI overlaps paper, posterior informative"
elseif paper_overlap && !informative
    "PARTIAL — 68%% CI overlaps paper but 3σ tail not constrained " *
    "(K_3hi=$(round(K_3hi, digits=1)) ≈ prior bound $(K_prior_hi))"
else
    "FAIL — 68%% CI does not overlap paper [8, 14]"
end
@printf("  %s\n", status)
println("="^70)

# Eccentricity posterior: report e = sesinw² + secosw² to compare to
# Desidera's e_b = 0.34 +0.07/-0.09 (3.8σ from zero).
if :sesinw_k1 in names(chains, :parameters) && :secosw_k1 in names(chains, :parameters)
    sesinw = vec(Array(chains[:, :sesinw_k1, :]))
    secosw = vec(Array(chains[:, :secosw_k1, :]))
    e_samp = sesinw .^ 2 .+ secosw .^ 2
    e_med = median(e_samp); e_lo = quantile(e_samp, 0.16); e_hi = quantile(e_samp, 0.84)
    @printf("  e_k1 = %.3f +%.3f -%.3f   (Desidera: 0.34 +0.07/-0.09)\n",
             e_med, e_hi - e_med, e_med - e_lo)
end

# GP rv hyperparam medians
for sym in (:gp_sigma, :gp_period, :gp_Q0, :gp_dQ, :gp_f, :dvdt)
    if sym in names(chains, :parameters)
        x = vec(Array(chains[:, sym, :]))
        @printf("  %-15s = %+.4f ± %.4f\n", String(sym), median(x), std(x))
    end
end
