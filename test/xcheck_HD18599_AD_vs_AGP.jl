#!/usr/bin/env julia
# Chain-rule evidence cross-check: ActivityDecorrelation vs ActivityGP
# on HD 18599 — the validation-gate instrument for any AD-vs-AGP claim.
#
# The honest AGP score for comparison against AD (whose likelihood is
# log p(y_R | indicators-as-regressors)) is the CONDITIONAL evidence
#
#   log Z_cond(AGP) = log Z(joint run) − log Z_gp(y_I)
#   log Z_gp(y_I)   = log Z(indicators_only run) − log Z(white run)
#
# (the indicators_only run factorizes as Z_gp(y_I)·Z_white(y_R) since
# its GP block and its white-RV block share no parameters). Using the
# conditional LIKELIHOOD (marginalize_indicators) as the objective
# instead is provably a free lunch — see the ActivityGP docstring.
#
# Run:  julia --project=. -t 8 Nereus.jl/test/xcheck_HD18599_AD_vs_AGP.jl

using Nereus
using DelimitedFiles
using Statistics: median, mean, std, quantile
using Printf
using MCMCChains

const RV_FILE = joinpath(@__DIR__, "data", "hd18599.csv")
const P_REF  = 4.1374685534602405
const M_S    = 0.807
const R_S    = 0.798
const P_ROT  = 8.74
const PAPER_INSTRUMENTS = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

# ---- data load: identical prep to fit_HD18599_activity_model_select.jl -------
raw      = readdlm(RV_FILE, ',', Any, '\n'; header = true)
data_mat = raw[1]
inst_raw = String.(data_mat[:, 16])
prov_raw = String.(data_mat[:, 17])
keep     = Int[]
inst_str = String[]
for i in 1:size(data_mat, 1)
    ins  = strip(inst_raw[i])
    prov = strip(prov_raw[i])
    (ins == "HARPS_POST" && prov == "ESO_PHASE3") && continue
    ins in PAPER_INSTRUMENTS || continue
    push!(keep, i); push!(inst_str, String(ins))
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

const _CSV_COL_VAL_ERR = Dict(
    :bis => (4, 5), :fwhm => (6, 7), :halpha => (10, 11), :logrhk => (12, 13))

# Per-instrument robust standardization + no-fabricated-precision
# imputation (mirrors fit_HD18599_activity_model_select.jl).
function _load_indicator(channel::Symbol)
    col_v, col_e = _CSV_COL_VAL_ERR[channel]
    raw_v = Float64[let v = data_mat[i, col_v]
        v === "" || (v isa AbstractString && strip(v) == "") ? NaN : Float64(v)
    end for i in 1:size(data_mat, 1)]
    raw_e = Float64[let v = data_mat[i, col_e]
        v === "" || (v isa AbstractString && strip(v) == "") ? NaN : Float64(v)
    end for i in 1:size(data_mat, 1)]
    v_std = copy(raw_v); e_std = copy(raw_e)
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
    return (raw = raw_v, std = v_std, err = e_std)
end

ind = Dict(c => _load_indicator(c) for c in (:bis, :fwhm, :halpha, :logrhk))

function _emperor_normalize!(v, rv_vec, inst)
    for ins_id in unique(inst)
        idx = findall(==(ins_id), inst)
        finite = filter(k -> isfinite(v[k]), idx)
        isempty(finite) && continue
        μ = mean(@view v[finite]); for k in finite; v[k] -= μ; end
        v_max = maximum(abs, @view v[finite]); v_max == 0 && continue
        rv_view = @view rv_vec[idx]
        rv_max_ins = maximum(abs, rv_view .- mean(rv_view))
        rv_max_ins == 0 && continue
        for k in finite; v[k] = v[k] / v_max * rv_max_ins; end
    end
    return v
end
function _ad_format(raw_v)
    v = copy(raw_v)
    for i in eachindex(v); isfinite(v[i]) || (v[i] = 0.0); end
    _emperor_normalize!(v, rv, rv_inst)
    return v
end

indicators_dict = Dict{String, Vector{Float64}}(
    "bisector_span" => _ad_format(ind[:bis].raw),
    "fwhm_AD"       => _ad_format(ind[:fwhm].raw),
    "halpha_AD"     => _ad_format(ind[:halpha].raw),
    "log_rhk_AD"    => _ad_format(ind[:logrhk].raw),
    "bis"           => ind[:bis].std,
    "fwhm"          => ind[:fwhm].std,
    "halpha"        => ind[:halpha].std,
    "logrhk"        => ind[:logrhk].std,
)
indicator_errs_dict = Dict{String, Vector{Float64}}(
    "bis" => ind[:bis].err, "fwhm" => ind[:fwhm].err,
    "halpha" => ind[:halpha].err, "logrhk" => ind[:logrhk].err)
data = Data(; t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
              indicators = indicators_dict,
              indicator_errs = indicator_errs_dict)
ic = InstrumentConfig(rv = inst_names)

rv_max = maximum(abs, rv)
function planet_priors()
    Dict{String, PriorSpec}(
        "P_k1"      => NormalPrior(P_REF, 1e-3, P_REF - 0.01, P_REF + 0.01),
        "K_k1"      => UniformPrior(0.0, 50.0),
        "sesinw_k1" => UniformPrior(-1.0, 1.0),
        "secosw_k1" => UniformPrior(-1.0, 1.0),
        "Mo_k1"     => UniformPrior(0.0, 2π))
end
function systemic_priors!(priors)
    for name in inst_names
        priors["gamma_$name"] = UniformPrior(-3 * rv_max, 3 * rv_max)
    end
    return priors
end

const AGP_KERNEL_PRIORS = Dict{String, PriorSpec}(
    "gp_act_period" => NormalPrior(P_ROT, 1.0, 4.0, 16.0))

function run_case(label, noise_models; n_planets = 1, seed = 3,
                   extra = Dict{String, PriorSpec}(),
                   n_temps = 10, n_walkers = 96,
                   n_steps = 6000, n_burnin = 2500, report = String[])
    priors = n_planets == 1 ? planet_priors() : Dict{String, PriorSpec}()
    systemic_priors!(priors)
    merge!(priors, extra)
    params = n_planets == 1 ?
        Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                 instruments = ic, data = data, M_s = M_S, R_s = R_S,
                 parametrization = ParametrizationConfig(time = :Mo),
                 priors = priors, noise_models = noise_models) :
        Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                 instruments = ic, data = data, M_s = M_S, R_s = R_S,
                 priors = priors, noise_models = noise_models)
    target = NereusTarget(params, data)
    t0 = time()
    res = sample_ptemcee(target, data; n_temps = n_temps,
                          n_walkers = n_walkers, n_steps = n_steps,
                          n_burnin = n_burnin, seed = seed,
                          init_strategy = :prior, show_progress = false)
    @printf("%-28s seed=%-3d %5.0f s  logZ = %10.2f", label, seed,
            time() - t0, res.log_evidence)
    if n_planets == 1
        K = vec(Array(res.chains[:, :K_k1, :]))
        @printf("  K = %.2f [%.2f, %.2f]", median(K),
                quantile(K, 0.16), quantile(K, 0.84))
    end
    println()
    for nm in report
        try
            v = vec(Array(res.chains[:, Symbol(nm), :]))
            @printf("    %-18s = %.3g [%.3g, %.3g]\n", nm, median(v),
                    quantile(v, 0.16), quantile(v, 0.84))
        catch; end
    end
    return res.log_evidence
end

ad   = ActivityDecorrelation(indicators = ["bisector_span", "fwhm_AD",
                                             "halpha_AD", "log_rhk_AD"])
agp  = ActivityGP(channels = [:bis, :fwhm, :halpha, :logrhk])  # joint
agpI = ActivityGP(channels = [:bis, :fwhm, :halpha, :logrhk],
                   indicators_only = true)

println("="^72)
println("HD 18599 — chain-rule evidence cross-check (AD vs AGP)")
println("="^72)

z_ad      = run_case("AD-only", NoiseModel[ad])
z_joint_a = run_case("AGP joint", NoiseModel[agp]; seed = 3,
                      extra = AGP_KERNEL_PRIORS,
                      report = ["gp_act_lambda_p", "gp_act_period", "Vc",
                                "Vr", "gp_act_jit_bis", "gp_act_jit_fwhm"])
z_joint_b = run_case("AGP joint", NoiseModel[agp]; seed = 17,
                      extra = AGP_KERNEL_PRIORS)
z_indonly = run_case("AGP indicators-only", NoiseModel[agpI];
                      n_planets = 0, extra = AGP_KERNEL_PRIORS,
                      n_walkers = 80, n_steps = 5000, n_burnin = 2000)
z_white   = run_case("white (0 planets)", NoiseModel[];
                      n_planets = 0, n_temps = 8, n_walkers = 64,
                      n_steps = 3000, n_burnin = 1200)

# AD + QP-GP indicator floor: the FAIR joint comparison against AGP-joint.
# Both score p(RV, indicators); AD's indicator block is now a QP-GP (same
# correlation power as AGP) instead of the white strawman, so the indicator
# terms ≈ cancel and the joint Bayes factor should reduce to the conditional
# one (AD wins). Predict logZ(AD+QPfloor) ≈ logZ(AD) + logZ_gp(y_I) ≫ logZ(AGP).
flr_qp = IndicatorFloor(channels = [:bis, :fwhm, :halpha, :logrhk], kernel = :qp)
z_ad_qpfloor = run_case("AD + QP-floor (joint)", NoiseModel[ad, flr_qp])

z_gp_yI  = z_indonly - z_white
z_cond_a = z_joint_a - z_gp_yI
z_cond_b = z_joint_b - z_gp_yI
@printf("\nlogZ_gp(y_I)          = %.2f\n", z_gp_yI)
@printf("logZ_cond(AGP)        = %.2f / %.2f  (seed spread %.2f)\n",
        z_cond_a, z_cond_b, abs(z_cond_a - z_cond_b))
@printf("logZ(AD)              = %.2f\n", z_ad)
@printf("Δ = logZ_cond(AGP) − logZ(AD) = %.2f / %.2f\n",
        z_cond_a - z_ad, z_cond_b - z_ad)
println("\nPositive Δ ⇒ data prefer the Rajpaul AGP over 4-indicator")
println("linear decorrelation, scored on the SAME conditional question")
println("p(y_R | y_I) with hyperparams anchored by the indicators.")

println("\n" * "="^72)
println("FAIR JOINT comparison (QP floor) — what the trans-dim chain sees")
println("="^72)
@printf("logZ(AD + QP-floor, joint) = %.2f\n", z_ad_qpfloor)
@printf("logZ(AGP joint)            = %.2f / %.2f\n", z_joint_a, z_joint_b)
@printf("Δ_joint = logZ(AD+QPfloor) − logZ(AGP) = %+.2f / %+.2f\n",
        z_ad_qpfloor - z_joint_a, z_ad_qpfloor - z_joint_b)
println(z_ad_qpfloor - max(z_joint_a, z_joint_b) > 5 ?
    "✅ AD+QP-floor WINS the fair joint Bayes factor ⇒ trans-dim AGP occupancy → ~0" :
    z_ad_qpfloor - min(z_joint_a, z_joint_b) > -5 ?
    "≈ near-tie — QP floor helps but check whether it captures AGP's indicator power" :
    "❌ AGP still wins the joint — QP floor under-captures the indicator correlation")
