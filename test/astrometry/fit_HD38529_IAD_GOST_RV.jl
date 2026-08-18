#!/usr/bin/env julia
# HD 38529 (HIP 27253) — multi-planet IAD + GOST + RV joint fit.
#
# HD 38529 is a G8IV subgiant at 39 pc with two confirmed companions:
#   b: P = 14.31 d,  K = 56 m/s,  m sin i = 0.78 M_J  (Fischer+ 2003)
#   c: P = 2134 d,   K = 170 m/s, m sin i = 13 M_J,   e = 0.36
#
# Planet c's *true* mass has been measured by three independent groups
# with notable tension:
#   Benedict+ 2010 (HST FGS):      m_c = 17.6 +1.5/-1.2 M_J,  i = 48°
#   Sahlmann+ 2011 (HIP IAD only): m_c = 37 +12/-8   M_J,    i = 24°
#   Xuan & Wyatt 2020 (HIP+Gaia PMA): m_c = 23 +3/-2 M_J
#
# Where Nereus's joint HIP IAD + Gaia GOST + RV fit lands is a
# *useful* datum — the spread spans factor-of-2 in mass.
#
# Multi-planet astrometry test: this is the first validation target
# that exercises max_kplanet=2 with one short-period RV-only planet
# and one long-period RVAS companion.

using Nereus
using DelimitedFiles
using Statistics: median, quantile
using Printf

println("=" ^ 70)
println("HD 38529 joint Hipparcos IAD + Gaia GOST + RV fit")
println("=" ^ 70)

RV_FILE = joinpath(@__DIR__, "..", "data", "hd38529.csv")
OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results", "HD38529_IAD_GOST_RV")

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
inst_str = String.(data_mat[:, 16])

# Three instruments: CLS=HIRES (Rosenthal+ 2021), LCES=Lick (Howard
# Lick-Carnegie reduction), HARPS. Keep all three.
inst_names = sort(unique(inst_str))
inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst    = Int[inst_map[s] for s in inst_str]
@printf("Loaded %d RVs across %d instruments: %s\n",
        length(bjd), length(inst_names), join(inst_names, ", "))
@printf("RV baseline: %.1f years\n",
        (maximum(bjd) - minimum(bjd)) / 365.25)

# Astrometry — HIP IAD + Gaia GOST
iad = fetch_hip_iad(27253; verbose = true)
gost = fetch_gost(86.57344854, 1.21168033;
                   from = "2014-07-26T00:00:00",
                   to   = "2025-01-15T00:00:00",
                   verbose = true)
@printf("HIP IAD:   %d scans, median σ = %.2f mas\n",
        n_iad(iad), median(iad.abscissa_err))
@printf("Gaia GOST: %d predicted scans\n", n_gost(gost))

data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    iad  = iad,
    gost = gost,
)
ic = InstrumentConfig(rv = inst_names)

# ---- 2. Model -------------------------------------------------------
# HD 38529 G8IV subgiant: M = 1.48 ± 0.05 M_⊙ (Henry+ 2013; Brewer+ 2016)
const M_PRI = 1.48

parametrization = ParametrizationConfig(
    ew   = :sesinw,
    time = :Tp,
    geom = :b_rr,
    mass = :M_sec_driven,
)

# Two-planet model. Planet b is RV-only (P=14d too short for IAD/GOST
# to resolve). Planet c is RVAS — the BD candidate at P=2134 d.
priors = Dict{String, PriorSpec}(
    # Planet b — short-period (Fischer+ 2003)
    "P_k1"       => LogUniformPrior(10.0, 50.0),
    "M_sec_k1"   => LogUniformPrior(1e-4, 5e-3),         # 0.1 - 5 M_J
    "sesinw_k1"  => UniformPrior(-1.0, 1.0),
    "secosw_k1"  => UniformPrior(-1.0, 1.0),
    "Tp_k1"      => UniformPrior(2_450_000.0, 2_450_050.0),
    # Planet c — long-period BD candidate
    "P_k2"       => LogUniformPrior(1500.0, 3500.0),
    "M_sec_k2"   => LogUniformPrior(0.005, 0.1),          # 5 - 100 M_J
    "sesinw_k2"  => UniformPrior(-1.0, 1.0),
    "secosw_k2"  => UniformPrior(-1.0, 1.0),
    "Tp_k2"      => UniformPrior(2_449_500.0, 2_453_000.0),
    "inc_k2"     => SinePrior(),
    "Omega_k2"   => UniformPrior(0.0, 2π),
    # van Leeuwen 2007 parallax for HIP 27253: 24.34 ± 0.32 mas
    "plx"        => NormalPrior(24.34, 0.32, 15.0, 35.0),
    # Per-instrument γ, σ
    ("gamma_$inst"  => UniformPrior(-500.0, 500.0)   for inst in inst_names)...,
    ("sigma_$inst"  => LogUniformPrior(0.5, 50.0)    for inst in inst_names)...,
    "M_pri"       => FixedPrior(M_PRI),
)

params = Params(;
    max_kplanet     = 2,
    planet_modes    = [RV_ONLY, RVAS],     # b: RV only, c: RV + astrometry
    instruments     = ic,
    data            = data,
    M_s             = M_PRI,
    parametrization = parametrization,
    priors          = priors,
    stability       = :none,
    trend_order     = 0,
)
target_bd = NereusTarget(params, data; unconstrained = false)

@printf("\nModel: %d unfrozen parameters\n", n_unfrozen(params))
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- 3. Sample ------------------------------------------------------
# Literature-seeded init. Use Rosenthal+ 2021 CLS values (most recent
# and most precise), Benedict+ 2010 inclination for planet c, mid-prior
# Ω.
using Random: MersenneTwister
n_temps   = parse(Int, get(ENV, "HD38529_N_TEMPS",   "15"))
n_walkers = parse(Int, get(ENV, "HD38529_N_WALKERS", "100"))
n_steps   = parse(Int, get(ENV, "HD38529_N_STEPS",   "3000"))
n_burnin  = parse(Int, get(ENV, "HD38529_N_BURNIN",  "1000"))
seed      = parse(Int, get(ENV, "HD38529_SEED",      "42"))

# Reference values
const P_b      = 14.3106
const K_b      = 56.31
const Mb_MJ    = 0.853
const eb       = 0.244
const ωb_deg   = 92.7
const Tp_b     = 2_450_005.7

const P_c      = 2135.8
const K_c      = 170.5
const Mc_MJ    = 13.4         # m·sin i — Rosenthal+ 2021
const ec       = 0.360
const ωc_deg   = 17.7
const Tp_c     = 2_450_240.4
const ic_deg   = 48.0         # Benedict+ 2010 inclination

const sesinwb  = sqrt(eb) * sind(ωb_deg)
const secoswb  = sqrt(eb) * cosd(ωb_deg)
const sesinwc  = sqrt(ec) * sind(ωc_deg)
const secoswc  = sqrt(ec) * cosd(ωc_deg)

function _init_value(name::AbstractString)
    name == "P_k1"      && return P_b
    name == "M_sec_k1"  && return Mb_MJ / 1047.57
    name == "sesinw_k1" && return sesinwb
    name == "secosw_k1" && return secoswb
    name == "Tp_k1"     && return Tp_b
    name == "P_k2"      && return P_c
    name == "M_sec_k2"  && return Mc_MJ / (1047.57 * sind(ic_deg))   # true mass
    name == "sesinw_k2" && return sesinwc
    name == "secosw_k2" && return secoswc
    name == "Tp_k2"     && return Tp_c
    name == "inc_k2"    && return deg2rad(ic_deg)
    name == "Omega_k2"  && return π
    name == "plx"       && return 24.34
    startswith(name, "gamma_") && return 0.0
    startswith(name, "sigma_") && return 3.0
    return 0.0
end

init_vec = Float64[_init_value(name)
                    for name in params.layout.unfrozen_names]

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
        init          = init_vec,
        init_strategy = :map_scatter,
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
plot_rv_phasefold(chains, params, data; planet = 2, output = OUT_DIR)
plot_rv_astrom_phasefold(chains, params, data; planet_idx = 2, output = OUT_DIR)
plot_orbit_skyplane(chains, params, data; planet_idx = 2, output = OUT_DIR)
plot_iad_residuals(chains, params, data; output = OUT_DIR)
plot_corner(chains, params; output = OUT_DIR,
            params_to_plot = ["P_k1", "M_sec_k1",
                              "P_k2", "M_sec_k2", "sesinw_k2", "secosw_k2",
                              "inc_k2", "Omega_k2", "plx"])

fitted  = summarize_fitted(chains, params)
derived = try
    summarize_derived(chains, params)
catch err
    @warn "summarize_derived skipped" exception=(err, catch_backtrace())
    Pair{String, ParamStats}[]
end
print_results(fitted, derived)

# ---- 5. Compare against published joint-fit groups ----------------
println("\n" * "="^70)
println("Comparison vs published mass solutions for HD 38529 c")
println("="^70)
function _med_q(name)
    samp = vec(Array(chains[Symbol(name)]))
    return median(samp), quantile(samp, 0.16), quantile(samp, 0.84)
end

for (label, k) in [("b", 1), ("c", 2)]
    println("\n--- Planet $label (k=$k) ---")
    P_med, P_16, P_84 = _med_q("P_k$k")
    M_med, M_16, M_84 = _med_q("M_sec_k$k")
    es_samp = vec(Array(chains[Symbol("sesinw_k$k")])).^2 .+
              vec(Array(chains[Symbol("secosw_k$k")])).^2
    e_med, e_16, e_84 = median(es_samp), quantile(es_samp, 0.16), quantile(es_samp, 0.84)
    Mmj = M_med * 1047.57
    @printf("  P      = %.2f  +%.2f / -%.2f  d\n", P_med, P_84 - P_med, P_med - P_16)
    @printf("  M_sec  = %.2f  +%.2f / -%.2f  M_J\n",
            Mmj, (M_84 - M_med) * 1047.57, (M_med - M_16) * 1047.57)
    @printf("  e      = %.3f +%.3f / -%.3f\n", e_med, e_84 - e_med, e_med - e_16)
end

if "inc_k2" in params.layout.unfrozen_names
    i_med, i_16, i_84 = _med_q("inc_k2")
    @printf("\n  inc_c  = %.1f  +%.1f / -%.1f  deg\n",
            rad2deg(i_med), rad2deg(i_84 - i_med), rad2deg(i_med - i_16))
end

println("\nLiterature mass for planet c (true mass, IAD/PMA-constrained):")
println("  Benedict+ 2010 (HST FGS):           m_c = 17.6 +1.5/-1.2 M_J  (i = 48°)")
println("  Sahlmann+ 2011 (HIP IAD only):      m_c = 37 +12/-8   M_J     (i = 24°)")
println("  Xuan & Wyatt 2020 (HIP+Gaia PMA):   m_c = 23 +3/-2    M_J")
println("  Rosenthal+ 2021 (CLS, RV only):     m_c · sin i = 13.4 ± 0.1 M_J")
