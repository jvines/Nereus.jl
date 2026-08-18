#!/usr/bin/env julia
# HD 18599 — joint RV + transit trans-dimensional fit ("ultimate test").
#
# Combines:
#   - 137 RVs (HARPS_PRE, HARPS_POST, FEROS, ESPRESSO, CORALIE_14)
#   - 4 sectors of cleaned TESS photometry (rotation-detrended via
#     `clean_HD18599.jl` → `data/HD18599/HD18599_cleaned_lc.csv`)
#
# The known planet HD 18599b / TOI-179b (Vines+ 2023):
#   P = 4.1375 d, K = 11 ± 3 m/s, Rp/Rs = 0.0311, Mp = 25.5 ± 4.6 Me
#   M_s = 0.807 M_sun, R_s = 0.798 R_sun (ARIADNE)
#
# Trans-dim setup:
#   slot 1 (RVPM)    — known transiting planet
#   slot 2 (RV_ONLY) — non-transiting candidate, period 1–6 d
#   N_p ∈ {0, 1, 2}, RV noise: AR(1), MA(1),
#   ActivityDecorrelation(BIS, derivative=true) — linear + FF' (Aigrain+
#   2012) in one toggle; chain finds Cdot ≈ 0 if FF' isn't needed —
#   and CeleriteRotation(GP). All toggleable independently → 16 noise
#   configs × 3 N_p = 48 models.
#   explored in a single PT trans-dim run.

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
                             "HD18599_RVPM_transdim")

# Known ephemeris from `clean_HD18599.jl` metadata.
const P_REF       = 4.1374685534602405
const T0_REF      = 2.4583545857470357e6
const DURATION_D  = 0.067   # ~1.6 hr; from Vines+ 2023 light-curve fit
const M_S, R_S    = 0.807, 0.798
const T_EFF       = 5290.0       # ARIADNE (Vines+ 2023, Table 2)
const J_MAG       = 8.064        # 2MASS
const K_MAG       = 7.629        # 2MASS
const A_B         = 0.3          # nominal Bond albedo

# ---- 1. Load RVs ------------------------------------------------------
println("="^70)
println("HD 18599 — joint RV + transit trans-dim fit")
println("="^70)

raw      = readdlm(RV_FILE, ',', Any, '\n'; header = true)
data_mat = raw[1]
inst_raw = String.(data_mat[:, 16])
prov_raw = String.(data_mat[:, 17])
keep     = Int[]
inst_str = String[]
# Restrict to the paper's instrument subset (HARPS_PRE + HARPS_POST +
# FEROS) so the trans-dim posterior is comparable to Vines+ 2023
# Run 1-6 results. CORALIE_14 and ESPRESSO are present in the data
# file but were not used in the paper; including them adds RV residuals
# at phases uncorrelated with the BIS regression and can shift the
# joint K | activities-on posterior.
const PAPER_INSTRUMENTS = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])
for i in 1:size(data_mat, 1)
    ins  = strip(inst_raw[i])
    prov = strip(prov_raw[i])
    # Drop the bad HARPS_POST DRS rows (errors ~6600 m/s, ingestion bug).
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

# Per-instrument 5×MAD outlier rejection on RV. The FEROS data file has
# one ~580 m/s point that the paper excluded but slipped through here;
# leaving it in pegs σ_FEROS at the upper bound and biases the joint
# Keplerian + BIS regression. MAD-based clipping is robust to small N
# and matches what astroEMPEROR / ARIADNE do at ingestion time.
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
if n_drop > 0
    @printf("RV outlier rejection (5σ MAD per instrument): dropped %d / %d points\n",
             n_drop, length(rv))
    for j in eachindex(rv)
        keep_rv[j] && continue
        @printf("  drop  bjd=%.5f  rv=%9.2f  inst=%s\n",
                 bjd[j], rv[j], inst_names[rv_inst[j]])
    end
end
data_mat = data_mat[keep_rv, :]
bjd      = bjd[keep_rv]
rv       = rv[keep_rv]
rv_err   = rv_err[keep_rv]
rv_inst  = rv_inst[keep_rv]
inst_str = inst_str[keep_rv]

@printf("Loaded %d RVs across %d instruments\n", length(bjd), length(inst_names))
for (i, name) in enumerate(inst_names)
    mask = rv_inst .== i
    n    = sum(mask)
    @printf("  [%d] %-13s n=%3d  σ=%.2f m/s  ⟨err⟩=%.2f m/s\n",
            i, name, n, std(rv[mask]), mean(rv_err[mask]))
end

# Activity indicator (BIS) — EMPEROR-style PER-INSTRUMENT normalization
# (astroEMPEROR `process_activities` + per-file `acts *= max(abs(rv))`):
#   1. For each instrument independently:
#        a. mean-subtract that instrument's BIS values (skipping NaNs)
#        b. divide by max(|BIS_inst|) so peak is ±1
#        c. multiply by max(|rv_inst|) so peak matches that instrument's
#           RV scale
# Doing this *globally* across all instruments (the previous behaviour)
# mixes per-instrument BIS scales and lets a single instrument with
# extreme RV values (e.g. a FEROS outlier) drive the rescale factor for
# every other instrument's BIS — biasing the joint regression and
# reducing K_b. Vines+ 2023 normalizes per-instrument by construction
# (data layout is one file per instrument, processed inside that loop).
function per_instrument_normalize!(v::Vector{Float64},
                                    rv_vec::Vector{Float64},
                                    inst::Vector{Int})
    # astroEMPEROR convention: BIS peak rescaled to MEAN-SUBTRACTED RV peak
    # per instrument (emperors_library.py line 252+259). Using raw RV peak
    # inflates BIS by the γ_inst offset and lets the BIS regression
    # absorb the Keplerian signal.
    for ins_id in unique(inst)
        idx = findall(==(ins_id), inst)
        finite_idx = filter(k -> isfinite(v[k]), idx)
        isempty(finite_idx) && continue
        μ = mean(@view v[finite_idx])
        for k in finite_idx
            v[k] -= μ
        end
        v_max = maximum(abs, @view v[finite_idx])
        v_max == 0 && continue
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

# ---- 2. Load + bin + subsample LC ------------------------------------
# 228K cadences at native 2-min TESS short-cadence is overkill for the
# trans-dim PT budget. Two-stage reduction:
#   Stage A — bin to LC_BIN_MIN-minute cadence over the whole LC.
#             At LC_BIN_MIN = 5 min the 1.6 h transit is still resolved
#             to ~20 points, so transit shape is preserved.
#   Stage B — keep ALL binned cadences within ±2× duration of any
#             literature-period transit; downsample the OOT remainder
#             by `OOT_STRIDE` since out-of-transit cadences only anchor
#             the offset/jitter and a few hundred is plenty.
LC_BIN_MIN = parse(Float64, get(ENV, "HD18599_LC_BIN_MIN", "5.0"))
OOT_STRIDE = parse(Int,     get(ENV, "HD18599_OOT_STRIDE", "200"))

println("\nLoading cleaned LC ...")
lc_raw = load_tess_lc(LC_FILE)
@printf("  raw cadences: %d, %.1f d span, σ = %.4e\n",
        length(lc_raw.t), lc_raw.t[end] - lc_raw.t[1], std(lc_raw.flux))

# Stage A — bin to LC_BIN_MIN cadence
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
@printf("  binned to %.1f-min cadence: %d cadences, σ = %.4e\n",
        LC_BIN_MIN, length(lc.t), std(lc.flux))

# Stage B — mask cadences within ±2× duration of any transit
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

# Take all in-transit + every OOT_STRIDE-th OOT
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

@printf("  kept %d cadences (%d in-transit + %d OOT @ 1/%d) — %.1f%% of binned, %.2f%% of raw\n",
        length(t_phot), count(in_transit), count(oot_keep), OOT_STRIDE,
        100 * length(t_phot) / n_lc,
        100 * length(t_phot) / length(lc_raw.t))

# ---- 3. Build joint Data + Params -------------------------------------
indicators = Dict("bisector_span" => ind_bis)
data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    indicators = indicators,
    t_phot = t_phot, flux = flux, flux_err = flux_err, phot_inst = phot_inst,
)
ic = InstrumentConfig(rv = inst_names, pm = ["TESS"])

# Toggleable noise: AR(1), MA(1), ActivityDecorrelation(BIS) on RV.
# Vines+ 2023 Eq. (3)/(4): AR and MA are **per-instrument** — separate
# (φ, α) and (ω, β) per spectrograph, lag chain only within instrument.
# Global ARMA with one parameter set linking cross-instrument residuals
# is physically wrong (different spectrographs don't share noise) and
# also has a much smaller Occam penalty, which was over-favouring the
# MA-on configuration in Nereus's joint trans-dim posterior.
ar  = ARModel(order = 1, per_instrument = true)
ma  = MAModel(order = 1, per_instrument = true)
# Single BIS regression model with FF' enabled (Aigrain, Pont & Zucker
# 2012): pred = C · BIS + Cdot · dBIS/dt per instrument. The linear-only
# limit is recovered automatically by the chain setting Cdot → 0 (Occam
# handles the extra-parameter penalty), so we don't need a separate
# `act_linear` toggle — would just duplicate the C coefficient names
# anyway, since ActivityDecorrelation registers C in the same namespace
# whether or not derivative=true.
act = ActivityDecorrelation(indicators = ["bisector_span"], derivative = true)
# GP RV activity (Desidera-style, two-SHO QPO). The Desidera analysis
# (arxiv:2210.07933) showed GP beats linear BIS decorrelation by Δlog Z
# ≈ 1300 on this data, with K = 11.3 +3.3/-3.6 m/s and e = 0.34 +0.07/-0.09.
# Adding GP to the trans-dim mix lets the chain decide between BIS, GP,
# ARMA, FF', or any combination.
gp_rv = CeleriteRotation(channel = :rv)

rv_max = maximum(abs, rv)
priors = Dict{String, PriorSpec}()

# Planet 1 — known transiting (RVPM). Period tightly constrained from
# the LC; a/R* now derived from the shared `rho_s` parameter (Vines+
# 2023 Table 7: rho_s ~ N(2.241, 0.479²) g/cc, derived from ARIADNE).
priors["P_k1"]      = NormalPrior(P_REF, 1e-3, P_REF - 0.01, P_REF + 0.01)
# Vines+ 2023 Table 7: K ~ U(0, 50). The previous ModJeffreysPrior peaks
# at small K and was biasing the K_b posterior toward zero.
priors["K_k1"]      = UniformPrior(0.0, 50.0)
priors["sesinw_k1"] = UniformPrior(-1.0, 1.0)
priors["secosw_k1"] = UniformPrior(-1.0, 1.0)
priors["Tc_k1"]     = NormalPrior(T0_REF, 0.05, T0_REF - 0.5, T0_REF + 0.5)
priors["b_k1"]      = UniformPrior(0.0, 1.0)
priors["rr_k1"]     = NormalPrior(0.031, 0.005, 0.005, 0.10)

# Planet 2 priors dropped — trans-dim is now noise-models-only with
# N_p fixed at 1 (the LC-anchored transiting planet).

# Per-RV-instrument systematics. γ matches Vines+ 2023 Table 7
# (U(0, 3·max(|rv|)) → here we centre on zero and use ±3·max for
# multi-instrument data where sign matters). σ_RV is the paper's
# N(5, 5²) — now the Nereus default for RV-channel jitter — so no
# explicit override needed here.
for name in inst_names
    priors["gamma_$name"] = UniformPrior(-3 * rv_max, 3 * rv_max)
end

# Photometry instrument: LC continuum already at 1.0, so offset prior
# tight; jitter on top of the cleaned-LC σ ≈ 1.2e-3.
priors["offset_TESS"]  = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
priors["jitter_TESS"]  = LogUniformPrior(1e-5, 5e-3)
priors["q1_TESS"]      = UniformPrior(0.0, 1.0)
priors["q2_TESS"]      = UniformPrior(0.0, 1.0)

# Time anchor: Tc (transit center) — natural for PM data, mapped to Tp
# internally. Eccentricity parametrization stays at the default sesinw.
# `use_rho_s=true` matches Vines+ 2023: a/R* derived from the shared
# stellar density rather than from M_s, R_s via Kepler's third law.
parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
priors["rho_s"] = NormalPrior(2.241, 0.479, 0.1, 10.0)

# Activity coefficient priors — with EMPEROR normalization (BIS scaled
# to max(|RV|)), C is dimensionless and the data should pin |C| < 1.
# The default U(-1, 1) is the right bound; we set it explicitly here
# only because the layout otherwise picks up a copy and so this
# documents intent.
for (i, name) in enumerate(inst_names)
    has_bis = any(j -> rv_inst[j] == i && isfinite(ind_bis[j]),
                   eachindex(ind_bis))
    has_bis || continue
    # Widened to U(-3, 3) — fit_HD18599_RVPM_Run2 showed C_HARPS_PRE
    # rails at -1.18 ± 0.23 with the tighter bound, biasing K downward.
    priors["C_bisector_span_$name"] = UniformPrior(-3.0, 3.0)
end

# GP RV activity hyperparameter priors (per the Desidera replication
# in fit_HD18599_RVPM_Desidera.jl — these are kept numerically stable
# under Pathfinder LBFGS exploration; looser bounds give PosDef errors).
priors["gp_sigma"]  = LogUniformPrior(0.1, rv_max)
# Tight Gaussian on the rotation period, anchored to Vines+ 2023's
# ground-based ASAS-SN/WASP photometry (not TESS — TESS only covers
# ~27 d per sector and can't constrain P_rot ≈ 8.7 d at the percent level).
# Hard bounds [8.0, 9.5] explicitly exclude the P_rot/2 ≈ 4.4 d alias.
priors["gp_period"] = NormalPrior(8.74, 0.1, 8.0, 9.5)
priors["gp_Q0"]     = LogUniformPrior(1.0, 10.0)
priors["gp_dQ"]     = UniformPrior(0.0, 3.0)
priors["gp_f"]      = UniformPrior(0.05, 0.5)
# Tightened γ̇ to U(0, 0.1) — Desidera convention (positive linear
# trend only); the Vines U(-1, 1) is broader but the chain just settles
# on the positive part anyway.
priors["dvdt"]      = UniformPrior(0.0, 0.1)

params = Params(;
    # N_p fixed at 1 — the transit-LC anchored planet is known. Trans-dim
    # here is *only* over the noise/activity model lineup. Running over
    # N_p too produces a trivial P(N_p=1) ≈ 1 result while burning ~⅓ of
    # the wall time sampling impossible N_p=0 and N_p=2 configurations
    # (verified empirically — Run 4 had P(N_p=1)=99% with 6.5-min runtime
    # split across 3 N_p modes).
    max_kplanet     = 1,
    planet_modes    = [RVPM],
    instruments     = ic,
    data            = data,
    M_s             = M_S,
    R_s             = R_S,
    parametrization = parametrization,
    priors          = priors,
    noise_models    = [ar, ma, act, gp_rv],
    transdim_noise  = true,
    trend_order     = 1,
    # External N(0, 0.3²) e prior DROPPED. Ablation
    # (fit_HD18599_RVPM_Run2_noecc.jl) showed it shifts K by only ~0.3
    # m/s and log Z by ~7 units — vs the GP-vs-BIS choice which moves
    # K by ~5 m/s and log Z by ~1300. We let the chain find e itself
    # so the GP+eccentric mode (Desidera) competes fairly with the
    # BIS+circular mode (Vines).
)

# `unconstrained = true` so Pathfinder's L-BFGS gets a target with
# proper gradients across the unconstrained ℝⁿ; the trans-dim PT
# then operates in bounded space (sample_pt_warm transforms inits).
target = NereusTarget(params, data; unconstrained = true)
@printf("\nModel: max_kplanet=1 (fixed), %d unfrozen params\n", n_unfrozen(params))
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- 4. PT trans-dim --------------------------------------------------
td = TransDimConfig(;
    max_kplanet       = 1,
    planets           = false,           # N_p fixed at 1 (no birth/death)
    transdim_fraction = 0.3,
    noise             = true,
    toggleable        = [ar, ma, act, gp_rv],
    # ActivityDecorrelation(BIS) and CeleriteRotation both model the
    # same stellar-rotation modulation in RV — letting them coexist
    # leaks planet signal into the joint fit (HD 18599 run 4: K dragged
    # to 7.58 m/s by 90% of mass having Activity=GP=on). Force the data
    # to pick *one* rotation tracker. AR + MA stay composable (real
    # ARMA = AR + MA), AR/MA with Activity stay composable (per-instr
    # residual correlation is orthogonal to mean-modelling).
    noise_exclusion_groups = [[act, gp_rv]],
)

# Allow CLI override for budget. Defaults: production budget.
n_rounds = parse(Int, get(ENV, "HD18599_ROUNDS", "16"))
n_chains = parse(Int, get(ENV, "HD18599_CHAINS", "16"))
seed     = parse(Int, get(ENV, "HD18599_SEED",   "777"))
@printf("\nStarting PT trans-dim: %d rounds × %d chains (seed=%d)\n",
        n_rounds, n_chains, seed)

# `HD18599_REPLOT_ONLY=1` skips sampling and just loads `chains.nc` from
# OUT_DIR + regenerates plots/summaries against the *current* plotting
# code. Use this when iterating on plot styling — saves the 9-minute
# sampler cost.
replot_only = parse(Bool, get(ENV, "HD18599_REPLOT_ONLY", "false"))

if replot_only
    chain_path = joinpath(OUT_DIR, "chains.nc")
    isfile(chain_path) || error(
        "REPLOT_ONLY=1 but no chains.nc at $chain_path — run sampler at " *
        "least once first.")
    @info "Loading saved chains from $chain_path (sampler skipped)"
    chains, _ = load_chains(chain_path)
    log_ev = -1.0
    pt_evals = 0
else
    t0 = time()
    # Pathfinder warmup → trans-dim PT with RWM within-model kernel +
    # early stopping. The four optimizations together typically deliver
    # 10–50× speedup vs the default `sample_pt` on this kind of joint fit:
    #   * Pathfinder skips the cold-chain planet-discovery phase
    #   * RWM does ~1 likelihood eval per coord vs ~7 for slice sampling
    #   * Early stop terminates rounds 13-15 when the N_p posterior has converged
    #   * Per-iter log-L cost is reduced by transit-likelihood pre-allocation
    chains, log_ev, pt_evals, pf = sample_pt_warm(target;
        n_pathfinder_runs    = 16,
        n_pathfinder_draws   = max(2 * n_chains, 200),
        td                   = td,
        n_rounds             = n_rounds,
        n_chains             = n_chains,
        seed                 = seed,
        show_report          = true,
        within_model         = :rwm,
        # Tighter early-stop so we actually run the requested 16 rounds.
        # At 0.001 the run will burn the last (largest) round even if
        # the N_p posterior is already stable, doubling sample count
        # versus stopping at round 14.
        early_stop_thresh    = 0.001,
        early_stop_min_rounds = 14,
    )
    dt = time() - t0
    @printf("PT-warm done in %.1f min (%d log-likelihood evaluations)\n",
            dt / 60, pt_evals)

    # ---- 5. Save chains + summary + plots --------------------------------
    mkpath(OUT_DIR)
    chain_path = joinpath(OUT_DIR, "chains.nc")
    save_chains(chain_path, chains, params; data = data)
    @printf("Chains saved to %s\n", chain_path)
end

println("\n" * "="^70)
println("  HD 18599 RV+PM trans-dim — summary")
println("="^70)
print_transdim_summary(chains, params; td = td, M_s = M_S, R_s = R_S,
                        T_eff = T_EFF, J_mag = J_MAG, K_mag = K_MAG, Ab = A_B)
save_transdim_summary(chains, params, OUT_DIR;
                       starname = "HD18599",
                       td = td, log_evidence = log_ev,
                       M_s = M_S, R_s = R_S,
                       T_eff = T_EFF, J_mag = J_MAG, K_mag = K_MAG, Ab = A_B)

println("\nGenerating plots ...")
plot_rv_timeseries(chains, params, data; output = OUT_DIR)
for k in 1:params.config.max_kplanet
    plot_rv_phasefold(chains, params, data; planet = k, output = OUT_DIR)
end
plot_pm_timeseries(chains, params, data; output = OUT_DIR)
plot_pm_phasefold(chains, params, data; planet = 1, output = OUT_DIR)
plot_histograms(chains, params; output = OUT_DIR)
@printf("Plots saved to %s\n", OUT_DIR)
println("\n" * "="^70)
