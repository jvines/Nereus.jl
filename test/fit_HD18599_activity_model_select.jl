# HD 18599 — full Rajpaul (multi-indicator) ActivityGP setup, plus
# the analogous multi-indicator ActivityDecorrelation, both wired into
# the same trans-dim composition. Closes the long-standing K-gap
# memo: BIS-decorrelation gives K ≈ 6 m/s (Vines+ 2023), the GP
# RV+indicator solution gives K ≈ 11 m/s (Desidera+ 2023 / canonical
# Rajpaul). This recipe is the canonical setup to reproduce both
# and compare.
#
# Run with:
#
#   julia --project=. -t 8 Nereus.jl/test/fit_HD18599_activity_model_select.jl
#
# The smoke part of this script (data load → params build → likelihood
# evaluation in both AD-only and AGP-only modes) runs unconditionally
# so the recipe is exercised in CI-light fashion. The full
# `sample_transdim_ptemcee` block at the bottom is commented out
# pending the multi-hour production run.
#
# Comparison caveat
# -----------------
# `ActivityDecorrelation` treats indicators as known regressors — its
# log L is `log p(RV | model, indicators)`. `ActivityGP` treats them
# as modeled time series — its log L is the joint
# `log p(RV, BIS, FWHM, Hα, logR'HK | model)`. Direct log-L
# comparison at the same posterior is therefore NOT apples-to-apples:
# AGP's log L absorbs ~4× more data terms. For honest model
# comparison, either condition AGP on the indicators (compute
# `log p(RV | indicators, AGP)` via the conditional-Gaussian formula —
# follow-up commit) or compare derived quantities (recovered K, RV
# PSIS-LOO, BIC) across two independent fixed-noise fits.

using Nereus
using DelimitedFiles
using Statistics: median, mean, std, quantile
using Printf
using MCMCChains
using Random
import Dates

const REPO_ROOT  = abspath(joinpath(@__DIR__, "..", ".."))
const RV_FILE    = joinpath(@__DIR__, "data", "hd18599.csv")

# Stellar constants (Vines+ 2023 / Desidera+ 2023; match the Desidera
# replication script).
const P_REF  = 4.1374685534602405
const T0_REF = 2.4583545857470357e6
const M_S    = 0.807
const R_S    = 0.798
const P_ROT  = 8.74     # Vines+ 2023 rotation period prior centre
const PAPER_INSTRUMENTS = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

println("="^70)
println("HD 18599 — Activity model selection: ActivityDecorrelation vs ActivityGP")
println("="^70)

# ---- RV load ---------------------------------------------------------
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
    push!(keep, i); push!(inst_str, ins)
end
data_mat = data_mat[keep, :]
bjd      = Float64.(data_mat[:, 1])
rv       = Float64.(data_mat[:, 2])
rv_err   = Float64.(data_mat[:, 3])
inst_names = sort!(unique(inst_str))
inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst    = [inst_map[s] for s in inst_str]

# Per-instrument robust outlier clip BEFORE anything touches the RVs:
# |rv − median_inst| > 5·1.4826·MAD_inst. Different reductions differ
# by zero-point (γ's job); within-instrument excursions of hundreds of
# m/s are outliers, full stop (HD 18599 FEROS: one 579 m/s point was
# inflating σ_FEROS to 44 m/s and decoupling γ_FEROS entirely).
let keep_clip = trues(length(bjd))
    for ins_id in unique(rv_inst)
        idx = findall(==(ins_id), rv_inst)
        med = median(rv[idx])
        mad = 1.4826 * median(abs.(rv[idx] .- med))
        thr = 5 * max(mad, 1e-6)
        for k in idx
            if abs(rv[k] - med) > thr
                keep_clip[k] = false
                @printf("clipped %s BJD %.4f: RV−med = %+.1f m/s (>%.0f)\n",
                        inst_names[ins_id], bjd[k], rv[k] - med, thr)
            end
        end
    end
    global bjd     = bjd[keep_clip]
    global rv      = rv[keep_clip]
    global rv_err  = rv_err[keep_clip]
    global rv_inst = rv_inst[keep_clip]
    global data_mat = data_mat[keep_clip, :]
end
@printf("Loaded %d RVs across %d instruments\n", length(bjd), length(inst_names))

# ---- Multi-indicator load: full Rajpaul couples RV with BIS + FWHM +
# Hα + log R'HK simultaneously. Each indicator goes into Data twice:
#   - AD-format (EMPEROR per-instrument normalized) under string key
#     "<name>_AD" → consumed by ActivityDecorrelation
#   - AGP-format (raw, per-instrument mean-subtracted) under canonical
#     channel-name string ("bis", "fwhm", "halpha", "logrhk") →
#     consumed by ActivityGP. Uncertainties land in indicator_errs.
#
# Column layout in hd18599.csv:
#   4/5 = bisector_span + err
#   6/7 = fwhm + err
#  10/11 = halpha + err
#  12/13 = log_rhk + err
const _CSV_COL_VAL_ERR = Dict(
    :bis    => (4, 5),
    :fwhm   => (6, 7),
    :halpha => (10, 11),
    :logrhk => (12, 13),
)

function _load_indicator(channel::Symbol)
    col_v, col_e = _CSV_COL_VAL_ERR[channel]
    raw_v = Float64[
        let v = data_mat[i, col_v]
            v === "" || (v isa AbstractString && strip(v) == "") ?
                NaN : Float64(v)
        end
        for i in 1:size(data_mat, 1)
    ]
    raw_e = Float64[
        let v = data_mat[i, col_e]
            v === "" || (v isa AbstractString && strip(v) == "") ?
                NaN : Float64(v)
        end
        for i in 1:size(data_mat, 1)
    ]
    # Per-instrument ROBUST STANDARDIZATION (median + 1.4826·MAD), not
    # just centering: the AGP stacks all instruments' series into ONE
    # channel with a SINGLE coupling pair, so per-instrument SCALE
    # differences force the coupling into contortions (review finding,
    # 2026-06-12). Errors are scaled by the same per-instrument factor.
    # Imputed (missing) values get err = 1e3 in standardized units —
    # NEVER a fabricated high-precision point: in the GP conditional a
    # fake mean-value point with a median error actively steers μ_cond.
    v_std = copy(raw_v)
    e_std = copy(raw_e)
    for ins_id in unique(rv_inst)
        idx_ins = findall(==(ins_id), rv_inst)
        fin = filter(k -> isfinite(v_std[k]), idx_ins)
        μ   = isempty(fin) ? 0.0 : median(v_std[fin])
        mad = isempty(fin) ? 1.0 :
              max(1.4826 * median(abs.(v_std[fin] .- μ)), 1e-12)
        fin_e = filter(k -> isfinite(e_std[k]) && e_std[k] > 0, idx_ins)
        med_e_scaled = isempty(fin_e) ? 0.1 : median(e_std[fin_e]) / mad
        for k in idx_ins
            if isfinite(v_std[k])
                v_std[k] = (v_std[k] - μ) / mad
                e_std[k] = (isfinite(e_std[k]) && e_std[k] > 0) ?
                           e_std[k] / mad : med_e_scaled
            else
                v_std[k] = 0.0
                e_std[k] = 1e3
            end
        end
    end
    return (raw = raw_v, centered = v_std, err = e_std)
end

ind_bis     = _load_indicator(:bis)
ind_fwhm    = _load_indicator(:fwhm)
ind_halpha  = _load_indicator(:halpha)
ind_logrhk  = _load_indicator(:logrhk)
@printf("Indicators loaded:\n")
@printf("  BIS     range [%.3g, %.3g]  median σ = %.3g\n",
        extrema(ind_bis.centered)..., median(ind_bis.err))
@printf("  FWHM    range [%.3g, %.3g]  median σ = %.3g\n",
        extrema(ind_fwhm.centered)..., median(ind_fwhm.err))
@printf("  Hα      range [%.3g, %.3g]  median σ = %.3g\n",
        extrema(ind_halpha.centered)..., median(ind_halpha.err))
@printf("  logR'HK range [%.3g, %.3g]  median σ = %.3g\n",
        extrema(ind_logrhk.centered)..., median(ind_logrhk.err))

# AD-format indicators: EMPEROR per-instrument normalize (rescaled to
# RV peak so the linear regressor coefficient has natural prior scale).
function _emperor_normalize!(v::Vector{Float64}, rv_vec::Vector{Float64},
                              inst::Vector{Int})
    for ins_id in unique(inst)
        idx = findall(==(ins_id), inst)
        finite = filter(k -> isfinite(v[k]), idx)
        isempty(finite) && continue
        μ = mean(@view v[finite])
        for k in finite; v[k] -= μ; end
        v_max = maximum(abs, @view v[finite])
        v_max == 0 && continue
        rv_view = @view rv_vec[idx]
        rv_max_ins = maximum(abs, rv_view .- mean(rv_view))
        rv_max_ins == 0 && continue
        for k in finite; v[k] = v[k] / v_max * rv_max_ins; end
    end
    return v
end

# Build the 4 AD-format regressors (one per indicator).
function _ad_format(raw_v::Vector{Float64})
    v = copy(raw_v)
    for i in eachindex(v); isfinite(v[i]) || (v[i] = 0.0); end
    _emperor_normalize!(v, rv, rv_inst)
    return v
end
ind_bis_AD     = _ad_format(ind_bis.raw)
ind_fwhm_AD    = _ad_format(ind_fwhm.raw)
ind_halpha_AD  = _ad_format(ind_halpha.raw)
ind_logrhk_AD  = _ad_format(ind_logrhk.raw)

# ---- Build joint Data ------------------------------------------------
# Each indicator stored twice: AD key (EMPEROR-normalized) + AGP key
# (raw, mean-subtracted) with uncertainties on the AGP key only.
indicators_dict = Dict{String, Vector{Float64}}(
    "bisector_span" => ind_bis_AD,
    "fwhm_AD"       => ind_fwhm_AD,
    "halpha_AD"     => ind_halpha_AD,
    "log_rhk_AD"    => ind_logrhk_AD,
    "bis"           => ind_bis.centered,
    "fwhm"          => ind_fwhm.centered,
    "halpha"        => ind_halpha.centered,
    "logrhk"        => ind_logrhk.centered,
)
indicator_errs_dict = Dict{String, Vector{Float64}}(
    "bis"    => ind_bis.err,
    "fwhm"   => ind_fwhm.err,
    "halpha" => ind_halpha.err,
    "logrhk" => ind_logrhk.err,
)
data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    indicators = indicators_dict,
    indicator_errs = indicator_errs_dict,
)
ic = InstrumentConfig(rv = inst_names)

# ---- Noise models — like-for-like multi-indicator activity ----------
# AD: linear regressor on 4 indicators (the most that the Nereus
# `ActivityDecorrelation` machinery handles in one model).
# AGP: full Rajpaul multivariate-GP coupling RV with the same 4
# indicators through a single latent G(t).
ad  = ActivityDecorrelation(indicators = ["bisector_span", "fwhm_AD",
                                            "halpha_AD", "log_rhk_AD"])
# JOINT likelihood (marginalize_indicators is a diagnostic, NOT a model-selection
# objective — it drops log p(y_I|θ) and buys conditional sharpness for
# free; see the ActivityGP docstring). NOTE the consequence: in the
# trans-dim run below the AD↔AGP occupancy is NOT a model-selection
# number (the two states score different data); the honest AD-vs-AGP
# comparison is the chain-rule evidence in
# test/xcheck_HD18599_AD_vs_AGP.jl.
agp = ActivityGP(channels = [:bis, :fwhm, :halpha, :logrhk],
                  marginalize_indicators = false)

# Priors specific to HD 18599: anchor the Rajpaul rotation period on
# Vines+ 2023's P_rot ≈ 8.74 d so the GP kernel has the right
# regularization. Other ActivityGP params use auto-priors.
priors = Dict{String, PriorSpec}()
rv_max = maximum(abs, rv)
priors["P_k1"]      = NormalPrior(P_REF, 1e-3, P_REF - 0.01, P_REF + 0.01)
priors["K_k1"]      = UniformPrior(0.0, 50.0)
priors["sesinw_k1"] = UniformPrior(-1.0, 1.0)
priors["secosw_k1"] = UniformPrior(-1.0, 1.0)
priors["Mo_k1"]     = UniformPrior(0.0, 2π)
for name in inst_names
    priors["gamma_$name"] = UniformPrior(-3 * rv_max, 3 * rv_max)
end
priors["gp_act_period"] = NormalPrior(P_ROT, 1.0, 4.0, 16.0)

params = Params(;
    max_kplanet     = 1, planet_modes = [RV_ONLY],
    instruments     = ic, data = data,
    M_s             = M_S, R_s = R_S,
    parametrization = ParametrizationConfig(time = :Mo),
    priors          = priors,
    noise_models    = [ad, agp],
    transdim_noise  = true,
)
@printf("\nParams built: %d unfrozen params (AD + AGP both registered)\n",
         n_unfrozen(params))

# ---- TransDimConfig with the AD/AGP exclusion group ------------------
td_cfg = TransDimConfig(;
    max_kplanet            = 1,
    planets                = false,    # fix K=1 — comparing activity models, not planet count
    noise                  = true,
    toggleable             = NoiseModel[ad, agp],
    noise_exclusion_groups = [[ad, agp]],
)
@printf("Exclusion group: [%s, %s]\n",
         typeof(ad).name.name, typeof(agp).name.name)

# ---- Smoke evaluation: build a Theta in each mode + compute log L ----
# This proves the trans-dim dispatch is live on real HD 18599 data
# without committing to a multi-hour sampler run.
function _smoke_eval(active_mask::Vector{Bool})
    tds = TransDimState(; max_planets = 1,
                          n_noise = length(params.config.noise_models))
    tds.planet_active[1] = true
    tds.n_planets_active = 1
    for (i, on) in enumerate(active_mask)
        on && activate_noise!(tds, i)
    end
    theta = Theta{Float64}(params; td = tds)

    # Initialise every unfrozen param to its prior median for a clean draw.
    for (nm, ps) in zip(params.layout.unfrozen_names,
                         params.layout.unfrozen_priors)
        lo, hi = bounds(ps)
        set_param!(theta, nm, 0.5 * (lo + hi))
    end
    # Sensible orbital seed.
    set_param!(theta, "P_k1", P_REF)
    set_param!(theta, "K_k1", 6.0)
    set_param!(theta, "Mo_k1", 1.0)
    set_param!(theta, "sesinw_k1", 0.0)
    set_param!(theta, "secosw_k1", 0.0)
    for name in inst_names
        set_param!(theta, "gamma_$name", median(rv[rv_inst .== inst_map[name]]))
        set_param!(theta, "sigma_$name", 3.0)
    end
    # AGP kernel + per-channel coupling seeds (only meaningful when
    # AGP active; those slots are inert under the AD path).
    set_param!(theta, "gp_act_period",    P_ROT)
    set_param!(theta, "gp_act_lambda_e",  30.0)
    set_param!(theta, "gp_act_lambda_p",  0.5)
    set_param!(theta, "Vc", 5.0);  set_param!(theta, "Vr", 0.0)
    set_param!(theta, "Bc", 1.0);  set_param!(theta, "Br", 0.0)
    set_param!(theta, "Fc", 0.05); set_param!(theta, "Fr", 0.0)
    set_param!(theta, "Hc", 0.05); set_param!(theta, "Hr", 0.0)
    set_param!(theta, "Lc", 0.05)
    return Nereus.rv_log_likelihood(theta, data)
end

ll_AD  = _smoke_eval([true, false])
ll_AGP = _smoke_eval([false, true])
ll_none = _smoke_eval([false, false])
@printf("\nLikelihood evaluations at canonical seed:\n")
@printf("  AD active, AGP off : log L = %12.3f\n", ll_AD)
@printf("  AGP active, AD off : log L = %12.3f\n", ll_AGP)
@printf("  both off (white)   : log L = %12.3f\n", ll_none)
@assert isfinite(ll_AD)  "AD path returned non-finite log L"
@assert isfinite(ll_AGP) "AGP path returned non-finite log L"
@assert isfinite(ll_none) "white-noise path returned non-finite log L"
@assert ll_AD != ll_AGP "AD and AGP returned identical log L — dispatch is dead"

println("\n✓ Trans-dim dispatch live on real HD 18599 data — full")
println("  multi-indicator Rajpaul GP (4 channels: BIS + FWHM + Hα +")
println("  logR'HK) and 4-indicator linear decorrelation both wire")
println("  through their respective likelihood paths.")
println("\n  Note: AGP log L absorbs the indicator data terms (4×")
println("  more data than AD sees), so the seed-value magnitudes")
println("  above aren't apples-to-apples. See the header comment.")

# ---- Sampler run -----------------------------------------------------
const _STAMP = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const OUT_DIR = joinpath(@__DIR__, "..", "results",
                          "HD18599_activity_modelselect_" * _STAMP)
mkpath(OUT_DIR)
@printf("\nOutput dir: %s\n", OUT_DIR)

target = NereusTarget(params, data; unconstrained = false)
# Run size via ENV so the production run doesn't need a script fork.
# Defaults = the "proper full run" config: a ladder deep enough for
# noise toggles to accept on the hot side and percolate down via swaps,
# walkers ≥ 2×ndim for Goodman-Weare, and enough steps to SEE occupancy
# stabilize (the 4×64×5000 run was still drifting when it ended).
N_TEMPS   = parse(Int, get(ENV, "NEREUS_NTEMPS",   "12"))
N_WALKERS = parse(Int, get(ENV, "NEREUS_NWALKERS", "128"))
N_STEPS   = parse(Int, get(ENV, "NEREUS_NSTEPS",   "30000"))
N_BURNIN  = parse(Int, get(ENV, "NEREUS_NBURNIN",  "6000"))
N_THIN    = parse(Int, get(ENV, "NEREUS_NTHIN",    "8"))
# Noise-birth assembly machinery (multi-try + pre-accept climb during
# burn-in). The sampler defaults are OFF (n_birth_tries=1, refine=0) —
# without these a noise birth is a single blind prior draw over the
# model's full hyperparam vector and multi-param activity models never
# get a fair birth (the 230-nat occupancy-vs-ΔlogZ failure).
N_TRIES   = parse(Int, get(ENV, "NEREUS_NTRIES",   "12"))
N_REFINE  = parse(Int, get(ENV, "NEREUS_NREFINE",  "25"))
@printf("Starting sample_transdim_ptemcee — %d temps × %d walkers × %d steps + %d burnin...\n",
        N_TEMPS, N_WALKERS, N_STEPS, N_BURNIN)
t_start = time()
res = sample_transdim_ptemcee(target, data;
                                td        = td_cfg,
                                n_temps   = N_TEMPS,
                                n_walkers = N_WALKERS,
                                n_steps   = N_STEPS,
                                n_burnin  = N_BURNIN,
                                thin      = N_THIN,
                                n_birth_tries  = N_TRIES,
                                n_birth_refine = N_REFINE,
                                seed      = 42,
                                show_progress = true)
@printf("Sampler done in %.1f min\n", (time() - t_start) / 60)
chains = res.chains
@printf("logZ = %.2f\n", res.log_evidence)
@printf("noise toggles accepted/proposed per temp (cold→hot): %s\n",
        join(string.(res.noise_td_accepted, "/", res.noise_td_proposed), "  "))

# Posterior fraction of draws with each activity model active. (Columns
# are noise_active_<i> — the old :noise_1 check matched nothing, so this
# block silently never printed.)
if :noise_active_1 in names(chains, :parameters)
    a_AD  = vec(Array(chains[:noise_active_1])) .> 0.5
    a_AGP = vec(Array(chains[:noise_active_2])) .> 0.5
    @printf("\nPosterior model probabilities (cold chain):\n")
    @printf("  P(AD active)  = %.3f\n", mean(a_AD))
    @printf("  P(AGP active) = %.3f\n", mean(a_AGP))
    @printf("  P(both)  = %.4f  (MUST be 0 — exclusion group)\n",
            mean(a_AD .& a_AGP))
    @printf("  P(none)  = %.3f\n", mean(.!a_AD .& .!a_AGP))
    # Convergence read: config occupancy per quarter of the run (draw
    # order = walker-blocks, but quarters still expose drift). Stable
    # quarters ⇒ usable weights; drifting quarters ⇒ run longer.
    N = length(a_AD)
    @printf("  occupancy drift by quarter (AD-only | AGP-only | none):\n")
    for q in 1:4
        r = (1 + (q-1)*N÷4):(q*N÷4)
        @printf("    Q%d: %.3f | %.3f | %.3f\n", q,
                mean(a_AD[r] .& .!a_AGP[r]),
                mean(a_AGP[r] .& .!a_AD[r]),
                mean(.!a_AD[r] .& .!a_AGP[r]))
    end
end

# Save chains.
save_chains(joinpath(OUT_DIR, "chains.nc"), chains, params; data = data)

# Generate plots.
@printf("\nGenerating plots...\n")
try
    plot_activity_gp_latent(chains, params, data;
                              filename = joinpath(OUT_DIR, "activity_gp_latent.png"))
    @printf("  ✓ activity_gp_latent.png\n")
catch err
    @warn "activity_gp_latent failed" exception = err
end
try
    plot_activity_gp_decomposition(chains, params, data;
                                     filename = joinpath(OUT_DIR, "activity_gp_decomposition.png"))
    @printf("  ✓ activity_gp_decomposition.png\n")
catch err
    @warn "activity_gp_decomposition failed" exception = err
end
try
    plot_rv_timeseries(chains, params, data; output = OUT_DIR)
    @printf("  ✓ rv_timeseries\n")
catch err
    @warn "rv_timeseries failed" exception = err
end
try
    ppc = posterior_predictive_check(chains, params, data; n_draws = 300)
    plot_ppc(ppc; filename = joinpath(OUT_DIR, "ppc.png"))
    @printf("  ✓ ppc.png — RV χ²/dof (best) = %.3f\n",
             get(ppc.summary, "rv_red_chi2_total", NaN))
catch err
    @warn "ppc failed" exception = err
end

# Recovered K.
if :K_k1 in names(chains, :parameters)
    K_samples = vec(Array(chains[:K_k1]))
    @printf("\nRecovered K_k1 = %.2f [%.2f, %.2f] m/s (median, 16/84)\n",
             median(K_samples), quantile(K_samples, 0.16), quantile(K_samples, 0.84))
end

@printf("\nAll artifacts in: %s\n", OUT_DIR)

# EMPEROR-style lp panels + folds (appended 2026-06-11)
try plot_posteriors_lp(chains, params; output = OUT_DIR)
catch e; println("posteriors_lp: ", sprint(showerror, e)) end
try plot_pm_phasefold(chains, params, data; planet = 1, output = OUT_DIR)
catch e; println("pm_phasefold: ", sprint(showerror, e)) end
try plot_rv_phasefold(chains, params, data; planet = 1, output = OUT_DIR)
catch e; println("rv_phasefold: ", sprint(showerror, e)) end
