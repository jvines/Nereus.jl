#!/usr/bin/env julia
# ε Eridani (HD 22049, HIP 16537) — Hipparcos IAD + Gaia GOST + RV joint fit.
#
# ε Eri is a young (~600 Myr) K2V at 3.22 pc, with a long-standing
# claimed RV planet (ε Eri b) heavily contested due to extreme
# rotational activity (P_rot ≈ 11 d, log R'_HK ≈ -4.5). Recent joint
# Hipparcos + Gaia + RV analyses (Llop-Sayson+ 2021; Mawet+ 2019;
# Brandt+ 2021) recover the planet at:
#
#   P    ≈ 7.4 yr  (≈ 2700 d)
#   K    ≈ 11 m/s  (heavily activity-dependent)
#   m·sin i ≈ 0.7 M_J  (RV-only)
#   m_true ≈ 0.7-0.8 M_J  (HIP+Gaia: low-i, but less extreme than HD 33636)
#   a    ≈ 3.5 AU
#   e    ≈ 0.07  (Llop-Sayson) or 0.4 (older)
#
# This is the **GOST + IAD + RV** validation test for Nereus. The 35 yr
# RV baseline (1987 - 2022, 2 133 RVs across 10 inst/prov groups in the
# exoautomata DB) is comparable to or longer than the published joint
# fits. ε Eri's activity will exercise the GP+activity machinery hard.
#
# Astrometric channels:
#   - HIP IAD (van Leeuwen 2007): 78 scans → 78 unrejected, t span 1990-1992
#   - Gaia GOST: 152 predicted scans, 2014-09 → 2025-01 (DR3 + DR4 window)
# The HIP→Gaia baseline of ~25 yr × ε Eri's high parallax (310 mas)
# yields strong inclination/Ω constraints if the orbit projects above
# the per-scan precision (~0.07 mas combined Hip+Gaia).

using Nereus
using DelimitedFiles
using Statistics: median, std, quantile
using Printf

println("=" ^ 70)
println("ε Eridani joint HIP IAD + Gaia GOST + RV fit")
println("=" ^ 70)

RV_FILE = joinpath(@__DIR__, "..", "data", "eps_eri.csv")
OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results", "eps_eri_IAD_GOST_RV")

raw = readdlm(RV_FILE, ',', Any, '\n'; header = true)
data_mat = raw[1]
function _to_f(col)
    out = Float64[]
    for v in col
        if v isa AbstractString
            s = strip(v)
            push!(out, isempty(s) ? NaN : parse(Float64, s))
        else
            push!(out, Float64(v))
        end
    end
    return out
end
bjd      = _to_f(data_mat[:, 1])
rv       = _to_f(data_mat[:, 2])
rv_err   = _to_f(data_mat[:, 3])
bisector = _to_f(data_mat[:, 4])
inst_str = String.(data_mat[:, 16])

# Use only the high-precision modern subset (≥1 m/s era). HAMILTON,
# COUDE, 2DCS, ELODIE have σ_RV > 5 m/s; including them inflates chains
# without helping the orbit fit on an active star (activity dominates
# noise budget anyway). HARPS_PRE/POST + APF + HIRES_POST + ESPRESSO
# is the modern Vines/CLS reduction subset.
keep_inst = Set(["HARPS_PRE", "HARPS_POST", "APF", "HIRES_POST", "ESPRESSO"])
keep = findall(s -> s in keep_inst, inst_str)
bjd = bjd[keep]; rv = rv[keep]; rv_err = rv_err[keep]
bisector = bisector[keep]; inst_str = inst_str[keep]

# 5σ MAD per-instrument outlier rejection.
inst_names = sort(unique(inst_str))
inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst    = Int[inst_map[s] for s in inst_str]
keep_o = trues(length(bjd))
for ins_id in unique(rv_inst)
    idx = findall(==(ins_id), rv_inst)
    v   = rv[idx]
    med = median(v); mad = median(abs.(v .- med))
    mad == 0 && continue
    σ = 1.4826 * mad
    for k in idx
        abs(rv[k] - med) > 5 * σ && (keep_o[k] = false)
    end
end
n_drop = count(.!keep_o)
n_drop > 0 && @printf("Dropped %d / %d RVs as 5σ MAD outliers\n", n_drop, length(bjd))
bjd = bjd[keep_o]; rv = rv[keep_o]; rv_err = rv_err[keep_o]
bisector = bisector[keep_o]; rv_inst = rv_inst[keep_o]

@printf("\nLoaded %d RVs across %d instruments: %s\n",
        length(bjd), length(inst_names), join(inst_names, ", "))
@printf("RV baseline: %.1f years\n",
        (maximum(bjd) - minimum(bjd)) / 365.25)

# Astrometry — both HIP and Gaia GOST channels via the cached fetchers
iad  = fetch_hip_iad(16537;  verbose = true)
gost = fetch_gost(53.232665, -9.458250;
                   from = "2014-07-26T00:00:00",
                   to   = "2025-01-15T00:00:00",
                   verbose = true)
@printf("HIP IAD:  %d scans, median σ = %.2f mas\n",
        n_iad(iad), median(iad.abscissa_err))
@printf("Gaia GOST: %d predicted scans\n", n_gost(gost))

# Filter bisector_span finite for activity decorrelation
bis_dict = if any(isfinite, bisector)
    Dict("bisector_span" => bisector)
else
    nothing
end

data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    indicators = bis_dict,
    iad  = iad,
    gost = gost,
)
ic = InstrumentConfig(rv = inst_names)

# ---- 2. Model -------------------------------------------------------
const M_PRI = 0.82       # ε Eri primary mass (Boyajian+ 2012)

# Active K-dwarf — GP RV activity model anchored on the photometric
# rotation period P_rot ≈ 11.2 d.
gp_rv = CeleriteRotation(channel = :rv)

parametrization = ParametrizationConfig(
    ew   = :sesinw,
    time = :Tp,
    geom = :b_rr,
    mass = :M_sec_driven,
)

rv_max = maximum(abs, rv)
priors = Dict{String, PriorSpec}(
    # Planet
    "P_k1"       => LogUniformPrior(1000.0, 5000.0),  # 2.7-13.7 yr
    "M_sec_k1"   => LogUniformPrior(1e-4, 0.05),       # 0.1-50 M_J
    "sesinw_k1"  => UniformPrior(-1.0, 1.0),
    "secosw_k1"  => UniformPrior(-1.0, 1.0),
    "Tp_k1"      => UniformPrior(2_450_000.0, 2_456_000.0),
    "inc_k1"     => SinePrior(),
    "Omega_k1"   => UniformPrior(0.0, 2π),
    # ε Eri parallax — van Leeuwen 2007 IAD header: 310.94 ± 0.17 mas.
    # The HIP catalogue value is unusually precise for ε Eri (very
    # bright, well-observed). Tight prior keeps the orbit scale honest.
    "plx"        => NormalPrior(310.94, 0.17, 305.0, 320.0),
    # Per-instrument
    ("gamma_$inst"  => UniformPrior(-rv_max, rv_max)  for inst in inst_names)...,
    ("sigma_$inst"  => LogUniformPrior(0.1, 100.0)    for inst in inst_names)...,
    # GP rotation — P_rot anchored on Donahue+ 1996 (ε Eri rotation
    # period = 11.2 ± 0.1 d from Ca II HK time series, well outside the
    # planet orbital period)
    "gp_sigma"   => LogUniformPrior(0.1, rv_max),
    "gp_period"  => NormalPrior(11.2, 0.5, 9.0, 14.0),
    "gp_Q0"      => LogUniformPrior(1.0, 30.0),
    "gp_dQ"      => UniformPrior(0.0, 5.0),
    "gp_f"       => UniformPrior(0.05, 0.5),
    "M_pri"      => FixedPrior(M_PRI),
)

params = Params(;
    max_kplanet     = 1,
    planet_modes    = [RVAS],
    instruments     = ic,
    data            = data,
    M_s             = M_PRI,
    parametrization = parametrization,
    priors          = priors,
    noise_models    = [gp_rv],
    trend_order     = 0,
    stability       = :none,
)
target_bd = NereusTarget(params, data; unconstrained = false)  # ptemcee wants bounded

@printf("\nModel: %d unfrozen parameters\n", n_unfrozen(params))
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- 3. Sample ------------------------------------------------------
# sample_ptemcee — same sampler that worked fast on HD 33636 and HD 18599.
# pt_warm was 60× slower here because Pigeons' doubling schedule does 2^14
# iterations per chain and GP+2000 RVs is ~200ms per likelihood eval.
n_temps   = parse(Int, get(ENV, "EPSERI_N_TEMPS",   "12"))
n_walkers = parse(Int, get(ENV, "EPSERI_N_WALKERS", "100"))
n_steps   = parse(Int, get(ENV, "EPSERI_N_STEPS",   "3000"))
n_burnin  = parse(Int, get(ENV, "EPSERI_N_BURNIN",  "1000"))
seed      = parse(Int, get(ENV, "EPSERI_SEED",      "42"))

mkpath(OUT_DIR)
chain_path = joinpath(OUT_DIR, "chains.nc")

if get(ENV, "PLOT_ONLY", "0") == "1" && isfile(chain_path)
    @info "PLOT_ONLY=1: loading saved chains from $chain_path"
    chains, _meta = load_chains(chain_path)
    log_ev = NaN
else
    @printf("\nStarting ptemcee: %d temps × %d walkers × %d steps (burnin %d)\n",
            n_temps, n_walkers, n_steps, n_burnin)
    t0 = time()
    res = sample_ptemcee(target_bd, data;
        n_temps  = n_temps,
        n_walkers = n_walkers,
        n_steps  = n_steps,
        n_burnin = n_burnin,
        seed     = seed,
        thin     = 1,
    )
    chains  = res.chains
    log_ev  = res.log_evidence
    @printf("ptemcee done in %.1f min (log Z = %.2f, n_evals = %d)\n",
            (time() - t0) / 60, log_ev, res.n_evals)
    @printf("  acceptance_within per β: %s\n",
            join(map(a -> @sprintf("%.2f", a), res.acceptance_within), ", "))
    @printf("  acceptance_swap pairs:   %s\n",
            join(map(a -> @sprintf("%.2f", a), res.acceptance_swap), ", "))
    save_chains(chain_path, chains, params; data = data)
end

plot_rv_timeseries(chains, params, data; output = OUT_DIR)
plot_rv_phasefold(chains, params, data; planet = 1, output = OUT_DIR)
plot_rv_astrom_phasefold(chains, params, data; planet_idx = 1, output = OUT_DIR)
plot_orbit_skyplane(chains, params, data; planet_idx = 1, output = OUT_DIR)
plot_iad_residuals(chains, params, data; output = OUT_DIR)
plot_corner(chains, params; output = OUT_DIR,
            params_to_plot = ["P_k1", "M_sec_k1", "sesinw_k1", "secosw_k1",
                              "inc_k1", "Omega_k1", "plx"])

fitted  = summarize_fitted(chains, params)
derived = try
    summarize_derived(chains, params)
catch err
    @warn "summarize_derived failed (M_sec_driven parametrisation lacks K_k1)" exception=(err, catch_backtrace())
    Pair{String, ParamStats}[]
end
print_results(fitted, derived)

println("\n" * "="^70)
println("Comparison vs Llop-Sayson+ 2021 (joint HARPS-N + APF + HIRES + HIP+Gaia)")
println("="^70)
function _med_q(name)
    samp = vec(Array(chains[Symbol(name)]))
    return median(samp), quantile(samp, 0.16), quantile(samp, 0.84)
end
P_med, P_16, P_84 = _med_q("P_k1")
M_med, M_16, M_84 = _med_q("M_sec_k1")
i_med, i_16, i_84 = _med_q("inc_k1")
e_samp = vec(Array(chains[:sesinw_k1])).^2 .+ vec(Array(chains[:secosw_k1])).^2
e_med, e_16, e_84 = median(e_samp), quantile(e_samp, 0.16), quantile(e_samp, 0.84)
P_yr = P_med / 365.25
M_med_MJ = M_med * 1047.57
@printf("  P       = %.0f  +%.0f / -%.0f  d  (%.2f yr)  (LLS+: ~2700 d / 7.4 yr)\n",
        P_med, P_84 - P_med, P_med - P_16, P_yr)
@printf("  M_sec   = %.2f +%.2f / -%.2f  M_J        (LLS+: 0.74 +0.36/-0.34)\n",
        M_med_MJ, (M_84 - M_med) * 1047.57, (M_med - M_16) * 1047.57)
@printf("  e       = %.3f +%.3f / -%.3f            (LLS+: 0.07 +0.06/-0.05)\n",
        e_med, e_84 - e_med, e_med - e_16)
@printf("  inc     = %.1f +%.1f / -%.1f  deg        (LLS+: 78 +9/-12)\n",
        rad2deg(i_med), rad2deg(i_84 - i_med), rad2deg(i_med - i_16))
