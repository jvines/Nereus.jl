#!/usr/bin/env julia
# HD 18599 — fixed-config replication of Vines+ 2023 Run 2.
#
# Direct comparison to paper Table 8 Run 2:
#   - 3 RV instruments (HARPS_PRE, HARPS_POST, FEROS)
#   - Noise = ActivityDecorrelation(BIS) only — NO ARMA
#   - No trans-dim noise (fixed model)
#   - Joint with TESS LC
#
# Paper Run 2 result: K = 11 ± 3 m/s.
#
# The trans-dim run gave a Run-2-conditional K = 5.93 ± 1.55 m/s, but
# only on ~60 samples — the chain concentrates on (MA=1, Act=1) and
# leaves the pure-activities mode statistically thin. This script
# locks the noise config and gets a clean, well-sampled K posterior to
# compare to the paper.

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
                             "HD18599_RVPM_Run2")

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
println("HD 18599 — paper Run 2 replication (activities only, no ARMA)")
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
indicators = Dict("bisector_span" => ind_bis)
data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    indicators = indicators,
    t_phot = t_phot, flux = flux, flux_err = flux_err, phot_inst = phot_inst,
)
ic = InstrumentConfig(rv = inst_names, pm = ["TESS"])

# FIXED noise config: ActivityDecorrelation only (paper Run 2).
act = ActivityDecorrelation(indicators = ["bisector_span"])

rv_max = maximum(abs, rv)
priors = Dict{String, PriorSpec}()
priors["P_k1"]      = NormalPrior(P_REF, 1e-3, P_REF - 0.01, P_REF + 0.01)
priors["K_k1"]      = UniformPrior(0.0, 50.0)
priors["sesinw_k1"] = UniformPrior(-0.7, 0.7)
priors["secosw_k1"] = UniformPrior(-0.7, 0.7)
priors["Tc_k1"]     = NormalPrior(T0_REF, 0.05, T0_REF - 0.5, T0_REF + 0.5)
priors["b_k1"]      = UniformPrior(0.0, 1.0)
priors["rr_k1"]     = NormalPrior(0.031, 0.005, 0.005, 0.10)
for name in inst_names
    priors["gamma_$name"] = UniformPrior(-3 * rv_max, 3 * rv_max)
end
priors["offset_TESS"]  = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
priors["jitter_TESS"]  = LogUniformPrior(1e-5, 5e-3)
priors["q1_TESS"]      = UniformPrior(0.0, 1.0)
priors["q2_TESS"]      = UniformPrior(0.0, 1.0)
parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
priors["rho_s"] = NormalPrior(2.241, 0.479, 0.1, 10.0)
for (i, name) in enumerate(inst_names)
    has_bis = any(j -> rv_inst[j] == i && isfinite(ind_bis[j]),
                   eachindex(ind_bis))
    has_bis || continue
    # Widen from U(-1, 1) to U(-3, 3): Vines+ 2023 Run 2 reported
    # |C| > 1 in raw slope (e.g. C_HARPS_post = -2.1 in Table 9 raw
    # units, and -0.836 in normalized units, well past 1 in the
    # current EMPEROR-rescaled BIS frame). The tight ±1 bound was
    # railing C_HARPS_PRE and making K unidentifiable as the BIS
    # regression couldn't grow large enough to absorb the activity
    # correctly.
    priors["C_bisector_span_$name"] = UniformPrior(-3.0, 3.0)
end

params = Params(;
    max_kplanet     = 1,                # FIXED N_p = 1
    planet_modes    = [RVPM],
    instruments     = ic,
    data            = data,
    M_s             = M_S,
    R_s             = R_S,
    parametrization = parametrization,
    priors          = priors,
    noise_models    = [act],            # activities ONLY — no AR, no MA
    transdim_noise  = false,            # FIXED noise
    trend_order     = 1,
    external_priors = [ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)],
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
    if sampler_choice == "pa"
        n_replicas = parse(Int, get(ENV, "HD18599_N_REPLICAS", "300"))
        n_mcmc     = parse(Int, get(ENV, "HD18599_N_MCMC",     "10"))
        ess_target = parse(Float64, get(ENV, "HD18599_PA_ESS", "0.5"))
        @printf("Running Population Annealing: %d replicas × %d MCMC/step (ESS target %.2f)\n",
                n_replicas, n_mcmc, ess_target)
        res = sample_pa(target, data;
            n_replicas = n_replicas,
            n_mcmc     = n_mcmc,
            ess_target = ess_target,
            seed       = seed,
        )
        chains = res.chains
        log_ev = res.log_evidence
        @printf("PA done in %.1f min (log Z = %.2f, n_steps = %d, mean acc = %.3f, n_evals = %d)\n",
                 (time() - t0) / 60, log_ev, length(res.beta_history) - 1,
                 res.acceptance, res.n_evals)
    elseif sampler_choice == "ptemcee"
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
        "K_k1",
        [n for n in params.layout.unfrozen_names
         if startswith(n, "C_bisector_span")]...,
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
println("  HD 18599 Run 2 (activities only, no ARMA) — K posterior")
println("="^70)
# Asymmetric 68% reporting (the only sane way to summarise a skewed
# posterior). The paper Run 2 result is K = 11 ± 3 m/s.
@printf("  K_k1 = %.2f +%.2f -%.2f m/s   (paper: 11 ± 3)\n",
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

# C posteriors
for name in inst_names
    sym = Symbol("C_bisector_span_$name")
    if sym in names(chains, :parameters)
        c = vec(Array(chains[:, sym, :]))
        @printf("  C_bisector_span_%-12s = %+.4f ± %.4f\n",
                 name, median(c), std(c))
    end
end
