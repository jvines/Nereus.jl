#!/usr/bin/env julia
# Diagnose the HD 18599 K = 6 vs K = 11 discrepancy directly at the
# log-likelihood level. Builds the same target as fit_HD18599_RVPM_Run2,
# takes the posterior median from the saved chain, then evaluates
# log L at:
#   (a) our posterior median K (~6 m/s, plus median for everything else)
#   (b) K = 11 m/s (paper value), everything else at our medians
#   (c) K = 0 m/s (no planet sanity check)
#
# If (b) > (a), the RV data PREFERS K=11 over K=6 at the same nuisance
# params and our chains miss it → sampling problem.
# If (b) < (a), our likelihood implementation says K=11 is WORSE than
# K=6 → the disagreement with the paper is a model/data difference,
# not a sampling artifact.
#
# Only `rv_log_likelihood` depends on K; `transit_log_likelihood` is K-
# invariant. So we can directly compare ΔrvLL between the two K values
# to localize the discrepancy.

using Nereus
using DelimitedFiles
using Statistics: median
using Printf

# ---- Reproduce the exact Run 2 target setup ---------------------------
const REPO_ROOT  = abspath(joinpath(@__DIR__, "..", ".."))
const RV_FILE    = joinpath(@__DIR__, "data", "hd18599.csv")
const LC_FILE    = joinpath(REPO_ROOT, "data", "HD18599",
                             "HD18599_cleaned_lc.csv")
const CHAIN_PATH = joinpath(REPO_ROOT, "Nereus.jl", "results",
                             "HD18599_RVPM_Run2", "chains.nc")
const P_REF      = 4.1374685534602405
const T0_REF     = 2.4583545857470357e6
const DURATION_D = 0.067
const M_S, R_S   = 0.807, 0.798
const PAPER_INSTRUMENTS = Set(["HARPS_PRE", "HARPS_POST", "FEROS"])

println("="^70)
println("HD 18599 — log-likelihood diagnostic at K = 6 / 11 / 0 m/s")
println("="^70)

# RV load + paper-instrument filter + 5σ MAD outlier drop
raw      = readdlm(RV_FILE, ',', Any, '\n'; header = true)
data_mat = raw[1]
inst_raw = String.(data_mat[:, 16])
keep     = Int[]
inst_str = String[]
for i in 1:size(data_mat, 1)
    inst_raw[i] in PAPER_INSTRUMENTS || continue
    push!(keep, i)
    push!(inst_str, inst_raw[i])
end
data_mat = data_mat[keep, :]
bjd     = Float64.(data_mat[:, 1])
rv      = Float64.(data_mat[:, 2])
rv_err  = Float64.(data_mat[:, 3])

# 5σ MAD outlier drop (matches Run 2 script)
mad(x) = median(abs.(x .- median(x)))
keep2 = trues(length(rv))
for ins in unique(inst_str)
    idx = findall(==(ins), inst_str)
    σ = 1.4826 * mad(rv[idx])
    μ = median(rv[idx])
    for k in idx
        abs(rv[k] - μ) > 5σ && (keep2[k] = false)
    end
end
n_drop = count(!, keep2)
println("Dropped $n_drop outlier RVs")
bjd      = bjd[keep2]
rv       = rv[keep2]
rv_err   = rv_err[keep2]
inst_str = inst_str[keep2]
inst_names = sort(unique(inst_str))
inst_id_map = Dict(n => i for (i, n) in enumerate(inst_names))
rv_inst = [inst_id_map[s] for s in inst_str]

# BIS (per-instrument normalized to mean-subtracted RV peak — EMPEROR
# convention, identical to Run 2 script)
ind_bis = Float64[
    let v = data_mat[keep2, 4][i]
        v === "" || (v isa AbstractString && strip(v) == "") ? NaN : Float64(v)
    end
    for i in 1:length(bjd)
]
function per_instrument_normalize!(v::Vector{Float64}, rv_vec::Vector{Float64},
                                    inst::Vector{Int})
    for ins_id in unique(inst)
        idx = findall(==(ins_id), inst)
        finite_idx = filter(k -> isfinite(v[k]), idx)
        isempty(finite_idx) && continue
        μ = sum(v[finite_idx]) / length(finite_idx)
        for k in finite_idx; v[k] -= μ; end
        v_max = maximum(abs, @view v[finite_idx])
        v_max == 0 && continue
        rv_inst_sub = rv_vec[idx] .- (sum(rv_vec[idx]) / length(idx))
        rv_max_ins = maximum(abs, rv_inst_sub)
        rv_max_ins == 0 && continue
        for k in finite_idx; v[k] = v[k] / v_max * rv_max_ins; end
    end
end
per_instrument_normalize!(ind_bis, rv, rv_inst)

# Photometry load (bin + transit-windowed subsample matches Run 2)
LC_BIN_MIN = 5.0
OOT_STRIDE = 200
lc_raw = load_tess_lc(LC_FILE)
lc = let bd = LC_BIN_MIN / (60 * 24)
    bin_id = floor.(Int, (lc_raw.t .- minimum(lc_raw.t)) ./ bd)
    ub = sort(unique(bin_id))
    tt = Float64[]; ff = Float64[]; ee = Float64[]
    for b in ub
        idxs = findall(==(b), bin_id)
        push!(tt, sum(lc_raw.t[idxs]) / length(idxs))
        push!(ff, sum(lc_raw.flux[idxs]) / length(idxs))
        push!(ee, (sum(lc_raw.flux_err[idxs]) / length(idxs)) / sqrt(length(idxs)))
    end
    (t = tt, flux = ff, flux_err = ee)
end
half_window = 2 * DURATION_D
n_lc        = length(lc.t)
n_per_orbit = ceil(Int, (lc.t[end] - lc.t[1]) / P_REF) + 2
in_transit  = falses(n_lc)
for k in -1:(n_per_orbit + 1)
    tc = T0_REF + k * P_REF
    for i in 1:n_lc
        abs(lc.t[i] - tc) < half_window && (in_transit[i] = true)
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
t_phot   = lc.t[keep_lc]; flux = lc.flux[keep_lc]; flux_err = lc.flux_err[keep_lc]
phot_inst = ones(Int, length(t_phot))

# Build Data + Params (matches Run 2 setup exactly)
indicators = Dict("bisector_span" => ind_bis)
data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    indicators = indicators,
    t_phot = t_phot, flux = flux, flux_err = flux_err, phot_inst = phot_inst,
)
ic = InstrumentConfig(rv = inst_names, pm = ["TESS"])
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
    priors["sigma_$name"] = NormalPrior(5.0, 5.0, 0.0, 50.0)
    priors["C_bisector_span_$name"] = UniformPrior(-3.0, 3.0)
end
priors["offset_TESS"]  = NormalPrior(0.0, 1e-3, -5e-3, 5e-3)
priors["jitter_TESS"]  = LogUniformPrior(1e-5, 5e-3)
priors["q1_TESS"]      = UniformPrior(0.0, 1.0)
priors["q2_TESS"]      = UniformPrior(0.0, 1.0)
priors["rho_s"]        = NormalPrior(2.241, 0.479, 0.1, 10.0)

parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
params = Params(;
    max_kplanet     = 1,
    planet_modes    = [RVPM],
    instruments     = ic,
    data            = data,
    M_s             = M_S, R_s = R_S,
    parametrization = parametrization,
    stability       = :none,
    trend_order     = 1,           # dvdt linear trend (paper Table 7)
    noise_models    = NoiseModel[act],
    priors          = priors,
    external_priors = [ExternalPrior(:ecc, NormalPrior(0.0, 0.3), true)],
)
target = NereusTarget(params, data)

# ---- Load the saved chain to extract our posterior median ------------
isfile(CHAIN_PATH) || error("No chain at $CHAIN_PATH — run fit_HD18599_RVPM_Run2 first")
chains, _ = load_chains(CHAIN_PATH)
param_names_chain = String.(names(chains, :parameters))

# Build x_median: posterior median per unfrozen parameter, in bounded space
layout = params.layout
n_unfrozen = length(layout.unfrozen_idx)
x_median = Vector{Float64}(undef, n_unfrozen)
for (j, name) in enumerate(layout.unfrozen_names)
    sym = Symbol(name)
    if sym in names(chains, :parameters)
        x_median[j] = median(vec(Array(chains[:, sym, :])))
    else
        # Fallback: prior mean
        ps = layout.unfrozen_priors[j]
        x_median[j] = (isfinite(ps.lo) && isfinite(ps.hi)) ? (ps.lo + ps.hi) / 2 : 0.0
        @warn "Param $name not in chain; using prior mid $(x_median[j])"
    end
end

# Build a Theta + evaluate log_prior + log_like at the bounded point
function eval_at(x_bounded::Vector{Float64})
    theta = Theta{Float64}(params)
    for (j, idx) in enumerate(layout.unfrozen_idx)
        theta.values[idx] = x_bounded[j]
    end
    lp = log_prior(theta)
    rv_ll = rv_log_likelihood(theta, data)
    tr_ll = transit_log_likelihood(theta, data)
    return (log_prior = lp, rv_ll = rv_ll, transit_ll = tr_ll,
            log_post = lp + rv_ll + tr_ll)
end

# Find K_k1 index
k_idx = findfirst(==("K_k1"), layout.unfrozen_names)
k_idx === nothing && error("K_k1 not in unfrozen layout")
K_our = x_median[k_idx]
@printf("\nOur posterior median K_k1 = %.3f m/s\n", K_our)

# ---- Build three test points: K_our, K=11 (paper), K=0 (no planet)
x_paper = copy(x_median);  x_paper[k_idx] = 11.0
x_zero  = copy(x_median);  x_zero[k_idx]  = 0.0

# Paper Table 9: full posterior medians from emperor's joint fit
# (the OTHER posterior mode at K=11). Where we have it, override.
c_pre_idx  = findfirst(==("C_bisector_span_HARPS_PRE"), layout.unfrozen_names)
c_post_idx = findfirst(==("C_bisector_span_HARPS_POST"), layout.unfrozen_names)
gpre_idx   = findfirst(==("gamma_HARPS_PRE"), layout.unfrozen_names)
gpost_idx  = findfirst(==("gamma_HARPS_POST"), layout.unfrozen_names)
x_paper_mode = copy(x_median)
x_paper_mode[k_idx]    = 11.0
if c_pre_idx  !== nothing; x_paper_mode[c_pre_idx]  = +1.0;  end
if c_post_idx !== nothing; x_paper_mode[c_post_idx] = -2.1;  end
if gpre_idx   !== nothing; x_paper_mode[gpre_idx]   = -2.7;  end
if gpost_idx  !== nothing; x_paper_mode[gpost_idx]  = -68.9; end

results = Dict{String, Any}()
for (label, x) in (("Our  (K=$(round(K_our, digits=2)))",  x_median),
                    ("Paper-K only (K=11, our nuis)",      x_paper),
                    ("Paper full mode (K=11, paper nuis)", x_paper_mode),
                    ("Zero  (K=0.00)",                     x_zero))
    r = eval_at(x)
    results[label] = r
end

println("\n" * "="^70)
println("Log-density components at each test K (all other params at our medians)")
println("="^70)
@printf("%-22s  %12s  %12s  %12s  %12s\n",
        "configuration", "log_prior", "rv_log_L", "transit_log_L", "log_post")
println("-"^80)
for label in ("Our  (K=$(round(K_our, digits=2)))",
              "Paper-K only (K=11, our nuis)",
              "Paper full mode (K=11, paper nuis)",
              "Zero  (K=0.00)")
    r = results[label]
    @printf("%-22s  %12.4f  %12.4f  %12.4f  %12.4f\n",
            label, r.log_prior, r.rv_ll, r.transit_ll, r.log_post)
end

# Diff vs our median
println()
r_our   = results["Our  (K=$(round(K_our, digits=2)))"]
r_paper_k    = results["Paper-K only (K=11, our nuis)"]
r_paper_full = results["Paper full mode (K=11, paper nuis)"]
r_zero  = results["Zero  (K=0.00)"]
@printf("Δlog_post  (paper-K only, our nuis vs ours)         = %+.4f\n",
        r_paper_k.log_post - r_our.log_post)
@printf("Δlog_post  (paper FULL mode, paper nuis vs ours)    = %+.4f\n",
        r_paper_full.log_post - r_our.log_post)
@printf("Δlog_post  (K=0  vs our K)                          = %+.4f\n",
        r_zero.log_post - r_our.log_post)

println()
if abs(r_paper_full.log_post - r_our.log_post) < 2.0
    println("→ Paper's FULL mode (K=11 with paper's nuisance values) is COMPARABLE")
    println("  to our mode at K=6. The posterior is BIMODAL — our chain found one")
    println("  mode (BIS absorbs RV variability), emperor found the other (Keplerian")
    println("  absorbs RV variability). Multimodal Keplerian-activity degeneracy at")
    println("  the rotation/orbit 2:1 alias (P_rot=8.74d, P_orb=4.14d).")
elseif r_paper_full.log_post > r_our.log_post
    println("→ Paper's full mode is HIGHER posterior than ours → our chain missed")
    println("  the dominant mode. Sampler needs warm-start from paper's mode or")
    println("  better mode-finding.")
else
    println("→ Paper's full mode is LOWER posterior than ours by " *
            "$(round(r_our.log_post - r_paper_full.log_post, digits=2)) units →")
    println("  our likelihood implementation truly excludes paper's mode. Real model gap.")
end
