#!/usr/bin/env julia
# HD 33636 b — Hipparcos IAD + RV joint fit (Nereus validation).
#
# Bean+ 2007 (AJ 134, 749) used the original 1997 Hipparcos IAD plus
# Lick/HET RVs (Perrier+ 2003; Cochran+ 2007) to show that the
# RV-detected ~9 M_J planet candidate is actually a ~141 M_J
# low-mass M dwarf seen at very low inclination (i ≈ 4°).
#
# Reference values (Bean+ 2007 Table 4, "joint" solution):
#   P            = 2447.3 ± 1.2 d  (≈ 6.7 yr)
#   K            = 164.0 ± 3.4 m/s
#   e            = 0.4805 ± 0.0042
#   ω            = 169.4 ± 0.7 deg
#   T_p (HJD)    = 2451772.0 ± 2.4
#   m·sin i (RV) = 9.28 ± 0.77 M_J
#   m_true       = 142 ± 11  M_J  =  0.136 ± 0.011  M_sun
#   inclination  = 4.1 +1.4/-1.1  deg  (HIP-constrained)
#   a            = 3.46 ± 0.03 AU
#
# Nereus's fit uses the van Leeuwen 2007 re-reduction IAD (78
# scans → 50 after rejection filtering) plus the exoautomata DB's
# accumulated RV time series (231 RVs across 9 inst/prov groups,
# 1997-2022). The longer-baseline modern RV set is a more
# stringent test than Bean+'s original ~25 RV epochs.

using Nereus
using DelimitedFiles
using Statistics: median, std, quantile
using Printf

println("=" ^ 70)
println("HD 33636 joint Hipparcos IAD + multi-epoch RV fit")
println("=" ^ 70)

# ---- 1. Data ---------------------------------------------------------
# IAD-fidelity diagnostic: re-run with Bean+ 2007's actual HET/HRS RVs
# (parsed from arxiv:0705.1861 tab1.tex — 67 epochs, ~5 yr baseline,
# Bean's published reduction). If Nereus reproduces Bean's i=4.1°,
# then the earlier i=9° result on modern multi-reduction data was an
# RV-contamination artefact. If i still → 9°, there's a residual
# IAD-side issue (since S/N on HD 33636 IAD is 27.8 — should pin i).
RV_FILE = joinpath(@__DIR__, "..", "data", "hd33636_bean.csv")
OUT_DIR = joinpath(@__DIR__, "..", "..", "..", "results", "HD33636_IAD_RV_BeanLocked")

raw = readdlm(RV_FILE, ',', Any, '\n'; header = true)
data_mat = raw[1]
_to_f(col) = Float64[v isa AbstractString ? parse(Float64, v) : Float64(v) for v in col]
bjd      = _to_f(data_mat[:, 1])
rv       = _to_f(data_mat[:, 2])
rv_err   = _to_f(data_mat[:, 3])
inst_str = String.(data_mat[:, 16])

# Drop instruments with <3 observations — γ would absorb the entire
# RV and σ from std() would be undefined / pathological.
inst_counts = Dict{String,Int}()
for s in inst_str
    inst_counts[s] = get(inst_counts, s, 0) + 1
end
keep_mask = Bool[inst_counts[s] >= 3 for s in inst_str]
n_drop_inst = count(.!keep_mask)
n_drop_inst > 0 && @printf("Dropped %d RVs from <3-obs instruments\n", n_drop_inst)
bjd      = bjd[keep_mask]
rv       = rv[keep_mask]
rv_err   = rv_err[keep_mask]
inst_str = inst_str[keep_mask]
inst_names = sort(unique(inst_str))
inst_map   = Dict(name => i for (i, name) in enumerate(inst_names))
rv_inst    = Int[inst_map[s] for s in inst_str]
@printf("Loaded %d RVs across %d instruments: %s\n",
        length(bjd), length(inst_names), join(inst_names, ", "))
@printf("RV baseline: %.1f years\n",
        (maximum(bjd) - minimum(bjd)) / 365.25)

# Hipparcos IAD — cached fetcher (5 KB extraction from 350 MB bundle)
iad = fetch_hip_iad(24205; verbose = true)
@printf("IAD: %d scans (van Leeuwen 2007), median σ = %.2f mas\n",
        n_iad(iad), median(iad.abscissa_err))

data = Data(;
    t_rv = bjd, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
    iad  = iad,
)
ic = InstrumentConfig(rv = inst_names)

# ---- 2. Model -------------------------------------------------------
# HD 33636 stellar mass: 1.02 M_⊙ (Valenti & Fischer 2005, used by Bean+ 2007).
const M_PRI = 1.02

# :M_sec_driven sampling — companion mass M_sec is the natural variable
# when IAD constrains the inclination (and thus m_true) directly.
parametrization = ParametrizationConfig(
    ew   = :sesinw,
    time = :Tp,
    geom = :b_rr,        # unused for RVAS (no transit), but required
    mass = :M_sec_driven,
)

priors = Dict{String, PriorSpec}(
    # IAD-fidelity diagnostic: lock the orbital parameters at Bean+
    # 2007's published joint-fit values via tight Gaussians, leaving
    # only (inc, Ω, M_sec, plx, γ, σ) free. If the IAD likelihood is
    # correctly implemented, inc should pin near Bean's 4.1°. M_sec
    # stays LogUniform-broad so the chain can explore the m·sin(i) →
    # m_true tradeoff freely; it's the IAD that breaks the degeneracy.
    "P_k1"       => NormalPrior(2447.3, 1.2, 2440.0, 2455.0),
    "M_sec_k1"   => LogUniformPrior(0.005, 0.5),
    "sesinw_k1"  => NormalPrior(0.12522, 0.005, -1.0, 1.0),  # = √0.4805 sin169.4°
    "secosw_k1"  => NormalPrior(-0.68186, 0.005, -1.0, 1.0), # = √0.4805 cos169.4°
    "Tp_k1"      => NormalPrior(2_451_772.0, 2.4, 2_451_762.0, 2_451_782.0),
    "inc_k1"     => SinePrior(),                          # geometric prior
    "Omega_k1"   => UniformPrior(0.0, 2π),
    # Parallax anchored on the Hipparcos van Leeuwen 2007 value
    # (35.25 ± 1.02 mas — file header `Plx`/`e_Plx` columns).
    # Without this the IAD likelihood's orbit-scale parameter has
    # no constraint and the chain wanders into the prior box.
    "plx"        => NormalPrior(35.25, 1.02, 5.0, 70.0),
    # Per-instrument: γ wide enough for NEA's absolute-RV reductions
    # (some report ~star's systemic v_r ≈ 5.5 km/s for HD 33636), σ
    # moderate LogUniform
    ("gamma_$inst"  => UniformPrior(-10_000.0, 10_000.0) for inst in inst_names)...,
    ("sigma_$inst"  => LogUniformPrior(0.5, 50.0)        for inst in inst_names)...,
    # Stellar mass fixed (we trust Valenti & Fischer 2005).
    "M_pri"       => FixedPrior(M_PRI),
)

params = Params(;
    max_kplanet     = 1,
    planet_modes    = [RVAS],     # RV + astrometry (IAD provides AS)
    instruments     = ic,
    data            = data,
    M_s             = M_PRI,
    parametrization = parametrization,
    priors          = priors,
    stability       = :none,      # single companion; no stability check
    trend_order     = 0,          # pure period-locked test, no dvdt
)
target = NereusTarget(params, data; unconstrained = true)

@printf("\nModel: %d unfrozen parameters\n", n_unfrozen(params))
println("Unfrozen: ", join(params.layout.unfrozen_names, ", "))

# ---- 3. Sample ------------------------------------------------------
# Seed every PT chain near Bean+ 2007's reported best-fit (P=2447 d,
# M_sec=142 M_J, e=0.48, ω=169°, Tp=2451772, inc=4.1°). Pathfinder
# warmstart fails for this 14-d posterior (Pareto k > 9 in repeated
# tests) and PT alone can't tunnel from the inc≈90° "ignore IAD" local
# mode to the inc≈4° true mode without a hint. Initializing at the
# published mode lets us validate the joint likelihood is correct —
# convergence + tight CIs at Bean+'s values means the IAD likelihood
# is wired right; deviations are sampling/data issues.
using Random: MersenneTwister
n_temps   = parse(Int, get(ENV, "HD33636_N_TEMPS",   "15"))
n_walkers = parse(Int, get(ENV, "HD33636_N_WALKERS", "100"))
n_steps   = parse(Int, get(ENV, "HD33636_N_STEPS",   "3000"))
n_burnin  = parse(Int, get(ENV, "HD33636_N_BURNIN",  "1000"))
seed      = parse(Int, get(ENV, "HD33636_SEED",      "42"))

# Bean+ 2007 reference values for HD 33636 b
const BEAN_P_d    = 2447.3
const BEAN_e      = 0.4805
const BEAN_ω_deg  = 169.4
const BEAN_Tp     = 2_451_772.0
const BEAN_M_J    = 142.0                 # m_true, joint solution
const BEAN_M_sun  = BEAN_M_J / 1047.57    # → M_⊙
const BEAN_inc_deg = 4.1
const BEAN_sesinw = sqrt(BEAN_e) * sind(BEAN_ω_deg)
const BEAN_secosw = sqrt(BEAN_e) * cosd(BEAN_ω_deg)

using Statistics: std
inst_gamma_init = Dict{String, Float64}()
inst_sigma_init = Dict{String, Float64}()
for inst in inst_names
    idx = findall(==(inst), inst_str)
    inst_gamma_init[inst] = isempty(idx) ? 0.0 : median(rv[idx])
    inst_sigma_init[inst] = isempty(idx) ? 5.0 :
        max(min(std(rv[idx]) / 2, 25.0), 1.0)  # ~half RMS, clamped
end

function _bean_init_value(name::AbstractString)
    name == "P_k1"     && return BEAN_P_d
    name == "M_sec_k1" && return BEAN_M_sun
    name == "sesinw_k1" && return BEAN_sesinw
    name == "secosw_k1" && return BEAN_secosw
    name == "Tp_k1"    && return BEAN_Tp
    name == "inc_k1"   && return deg2rad(BEAN_inc_deg)
    name == "Omega_k1" && return π                    # unknown; mid-prior
    name == "plx"      && return 35.25
    if startswith(name, "gamma_")
        inst = name[7:end]
        return get(inst_gamma_init, inst, 0.0)
    end
    if startswith(name, "sigma_")
        inst = name[7:end]
        return get(inst_sigma_init, inst, 5.0)
    end
    return 0.0
end

init_vec = Float64[_bean_init_value(name)
                    for name in params.layout.unfrozen_names]

# Switch to ensemble PT (sample_ptemcee). Per Run 3 on HD 18599, this
# consistently finds the right mode where Pigeons-PT mode-traps. 15
# temperatures × 100 walkers × 3000 steps is the same budget that
# recovered Desidera's K=11 mode on HD 18599.
target_bd = NereusTarget(params, data; unconstrained = false)  # ptemcee wants bounded

@printf("\nStarting ptemcee (Bean+-seeded ensemble): %d temps × %d walkers × %d steps (burnin %d)\n",
        n_temps, n_walkers, n_steps, n_burnin)

t0 = time()
res = sample_ptemcee(target_bd, data;
    n_temps  = n_temps,
    n_walkers = n_walkers,
    n_steps  = n_steps,
    n_burnin = n_burnin,
    init          = init_vec,         # Bean+ values, Gaussian-scattered
    init_strategy = :map_scatter,     # used when init is supplied
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

# ---- 4. Save + summarise -------------------------------------------
mkpath(OUT_DIR)
chain_path = joinpath(OUT_DIR, "chains.nc")
save_chains(chain_path, chains, params; data = data)
println("\nChains saved to $chain_path")

# Plots
plot_rv_timeseries(chains, params, data; output = OUT_DIR)
plot_rv_phasefold(chains, params, data; planet = 1, output = OUT_DIR)

# Posterior summary
fitted  = summarize_fitted(chains, params)
derived = try
    summarize_derived(chains, params)
catch err
    @warn "summarize_derived failed (M_sec_driven parametrisation lacks K_k1)" exception=(err, catch_backtrace())
    Pair{String, ParamStats}[]
end
print_results(fitted, derived)

# Bean+ 2007 comparison
println("\n" * "="^70)
println("Comparison vs Bean+ 2007 Table 4 (joint HIP + RV)")
println("="^70)
import MCMCChains
function _med_q(name)
    samp = vec(Array(chains[Symbol(name)]))
    return median(samp), quantile(samp, 0.16), quantile(samp, 0.84)
end
P_med, P_16, P_84 = _med_q("P_k1")
M_med, M_16, M_84 = _med_q("M_sec_k1")
# With :sesinw parametrisation, sesinw² + secosw² = e directly (no sqrt!)
e_samp = vec(Array(chains[:sesinw_k1])).^2 .+ vec(Array(chains[:secosw_k1])).^2
e_med, e_16, e_84 = median(e_samp), quantile(e_samp, 0.16), quantile(e_samp, 0.84)
i_med, i_16, i_84 = _med_q("inc_k1")
M_med_MJ = M_med * 1047.57
@printf("  P       = %.1f  +%.1f / -%.1f  d        (Bean+: 2447.3 ± 1.2)\n",
        P_med, P_84 - P_med, P_med - P_16)
@printf("  M_sec   = %.1f  +%.1f / -%.1f  M_J     (Bean+: 142 ± 11)\n",
        M_med_MJ, (M_84 - M_med) * 1047.57, (M_med - M_16) * 1047.57)
@printf("  e       = %.3f +%.3f / -%.3f          (Bean+: 0.4805 ± 0.0042)\n",
        e_med, e_84 - e_med, e_med - e_16)
@printf("  inc     = %.1f  +%.1f / -%.1f  deg      (Bean+: 4.1 +1.4/-1.1)\n",
        rad2deg(i_med), rad2deg(i_84 - i_med), rad2deg(i_med - i_16))

# Period-locked diagnostic: did σ drop to the noise floor?
println("\n" * "-"^70)
println("P-locked diagnostic: per-instrument σ vs noise floor")
println("-"^70)
σ_floor_guess = 3.5  # quoted HIRES PRE/POST median internal σ (m/s)
for inst in inst_names
    σ_med, σ_16, σ_84 = _med_q("sigma_$inst")
    flag = σ_med < 8.0  ? "OK noise-floor"  :
           σ_med < 20.0 ? "moderate excess" : "high excess"
    @printf("  σ_%-12s = %5.2f +%.2f / -%.2f  m/s   [%s]\n",
            inst, σ_med, σ_84 - σ_med, σ_med - σ_16, flag)
end
@printf("(Reference HIRES internal precision ≈ %.1f m/s)\n", σ_floor_guess)
println("\nInterpretation:")
println("  All σ near floor (≤8) → σ inflation was downstream of wrong P;")
println("                          Bean's mode is the correct fit and the")
println("                          earlier P≈2186 run failed to sample it.")
println("  σ stay high (≥20)    → data has intrinsic scatter at that level;")
println("                          modern dataset cannot reach Bean's K-amplitude")
println("                          regardless of period — Nereus's P≈2186 is the")
println("                          right answer for this dataset.")
