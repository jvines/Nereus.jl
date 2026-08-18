#!/usr/bin/env julia
# Kepler-9 b/c TTV demo — the canonical 2:1 MMR multi-transit system
# from Holman+ 2010, refined by Hadden+ 2017. TTVs of HOURS are easily
# detected from Kepler's 4-year photometry, so this is a "yes there are
# TTVs and we can see them" demo (unlike WASP-47 where the signal lives
# below TESS/K2 precision).
#
# Both planets transit at high SNR. Fit TTV-A free per-transit offsets
# for both. The chain will recover the well-known TTV envelopes
# directly from the data.

using Nereus
using Statistics: median, std, mean, quantile
using Printf
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "Kepler9")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results", "Kepler9_TTV")
mkpath(OUT_DIR)

println("="^70)
println("Kepler-9 b/c TTV demo — 2:1 MMR, hour-scale TTVs")
println("="^70)

# --- Hadden+ 2017 ephemerides + system ---
const P_b_LIT  = 19.23891           # days (b is inner, P_b < P_c)
const T0_b_LIT = 2455073.4359
const P_c_LIT  = 38.9123
const T0_c_LIT = 2455079.2989
const M_s_LIT  = 1.022
const R_s_LIT  = 0.958
# Transit geometry (Holman 2010 Table 2)
const RR_B_LIT = 0.0768
const B_B_LIT  = 0.34
const RR_C_LIT = 0.0741
const B_C_LIT  = 0.04
# Limb darkening (Kepler bandpass, Claret quadratic for K2V star)
const Q1_K     = 0.49
const Q2_K     = 0.32
# Stellar density from Hadden 2017
const RHO_S_LIT = 1.62

# --- Load Kepler LC ----------------------------------------------------
lc = load_tess_lc(joinpath(DATADIR, "Kepler9_kplr_lc.csv"))
t_full, flux, err = lc.t, lc.flux, lc.flux_err
@printf("Kepler LC: %d points, span %.1f d (%.1f yr)\n",
        length(t_full), t_full[end] - t_full[1],
        (t_full[end] - t_full[1])/365.25)

# --- Transit centers per planet ----------------------------------------
function transits_in(t::AbstractVector, P::Real, T0::Real)
    n_lo = ceil(Int, (minimum(t) + 0.1 - T0) / P)
    n_hi = floor(Int, (maximum(t) - 0.1 - T0) / P)
    return [(n, T0 + n*P) for n in n_lo:n_hi]
end
tcs_b = transits_in(t_full, P_b_LIT, T0_b_LIT)
tcs_c = transits_in(t_full, P_c_LIT, T0_c_LIT)
# Shift T0 to the FIRST observed transit so all cycle indices are ≥ 0.
# This makes the TTV-slot allocation cover every observed transit (the
# slot scheme indexes by cycle number i = round((t - Tc)/P) starting
# at slot i+1 = 1 for the first observed transit).
T0_b_FIT = tcs_b[1][2]
T0_c_FIT = tcs_c[1][2]
tcs_b = transits_in(t_full, P_b_LIT, T0_b_FIT)
tcs_c = transits_in(t_full, P_c_LIT, T0_c_FIT)
@printf("Predicted transits: b → %d (idx 0..%d), c → %d (idx 0..%d)\n",
        length(tcs_b), tcs_b[end][1],
        length(tcs_c), tcs_c[end][1])

# --- Detrend (savgol with transit mask wide enough for ~6 hr T14) -----
mask_b = mask_transits(t_full, [P_b_LIT], [T0_b_FIT]; window = 0.15)
mask_c = mask_transits(t_full, [P_c_LIT], [T0_c_FIT]; window = 0.15)
mask = mask_b .| mask_c
dt = detrend_savgol(t_full, flux, err;
                     window_length = 401, polyorder = 3,
                     transit_mask = mask)
flux_dt = dt.flux_detrended

# --- Subset to ±0.20 d around transits of either planet ---------------
keep = falses(length(t_full))
for (_, Tc) in vcat(tcs_b, tcs_c)
    @. keep |= abs(t_full - Tc) < 0.20
end
# Drop NaN/Inf from the detrended flux — savgol's interpolation across
# Kepler quarter-boundary gaps can introduce a handful of Inf points
# that poison the likelihood (∞ residual² in the Gaussian sum).
keep .&= isfinite.(flux_dt)
keep .&= isfinite.(err)
t_fit   = t_full[keep]
flux_fit = flux_dt[keep]
err_fit  = err[keep]
@printf("Subset: %d points (%.1f%% of full LC)\n",
        length(t_fit), 100*length(t_fit)/length(t_full))

data = Data(; t_phot = t_fit, flux = flux_fit, flux_err = err_fit,
              phot_inst = ones(Int, length(t_fit)))
ic = InstrumentConfig(rv = String[], pm = ["Kepler"])

# --- Priors (orbit pinned to lit, TTV slots free) ---------------------
priors = Dict{String, PriorSpec}()

# Slot 1: Kepler-9 b (inner). Free Tc + small TTV deviations.
priors["P_k1"]      = FixedPrior(P_b_LIT)
priors["Tc_k1"]     = FixedPrior(T0_b_FIT)
priors["sesinw_k1"] = FixedPrior(0.0)
priors["secosw_k1"] = FixedPrior(0.0)
priors["b_k1"]      = FixedPrior(B_B_LIT)
priors["rr_k1"]     = FixedPrior(RR_B_LIT)

# Slot 2: Kepler-9 c (outer)
priors["P_k2"]      = FixedPrior(P_c_LIT)
priors["Tc_k2"]     = FixedPrior(T0_c_FIT)
priors["sesinw_k2"] = FixedPrior(0.0)
priors["secosw_k2"] = FixedPrior(0.0)
priors["b_k2"]      = FixedPrior(B_C_LIT)
priors["rr_k2"]     = FixedPrior(RR_C_LIT)

priors["rho_s"]            = FixedPrior(RHO_S_LIT)
priors["offset_Kepler"]    = NormalPrior(0.0, 5e-4, -3e-3, 3e-3)
priors["jitter_Kepler"]    = LogUniformPrior(1e-5, 5e-3)
priors["q1_Kepler"]        = FixedPrior(Q1_K)
priors["q2_Kepler"]        = FixedPrior(Q2_K)

# TTV slots: per-transit ±60 min hard bound (TTVs are KNOWN to be hours
# in this system; need wide hard bounds). σ = 20 min prior.
n_ttv_b = maximum(x[1] for x in tcs_b) + 1
n_ttv_c = maximum(x[1] for x in tcs_c) + 1
obs_b   = Set(x[1] for x in tcs_b)
obs_c   = Set(x[1] for x in tcs_c)
for i in 1:n_ttv_b
    priors["ttv_k1_t$i"] = (i-1) in obs_b ?
        NormalPrior(0.0, 0.015, -0.05, 0.05) :  # ±72 min hard, σ=22 min
        FixedPrior(0.0)
end
for i in 1:n_ttv_c
    priors["ttv_k2_t$i"] = (i-1) in obs_c ?
        NormalPrior(0.0, 0.015, -0.05, 0.05) :
        FixedPrior(0.0)
end

parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
params = Params(;
    max_kplanet     = 2,
    planet_modes    = [PM_TTV, PM_TTV],
    instruments     = ic,
    data            = data,
    M_s             = M_s_LIT,
    R_s             = R_s_LIT,
    parametrization = parametrization,
    priors          = priors,
    stability       = :none,
    ttv_n_transits  = Dict(1 => n_ttv_b, 2 => n_ttv_c),
)
@printf("Free params: %d  (%d obs transits b, %d obs c)\n",
        n_unfrozen(params), length(obs_b), length(obs_c))

target = NereusTarget(params, data; unconstrained = false)

# --- Sample (tuned for many-transit phot-only fit) ---------------------
const SEED      = parse(Int, get(ENV, "KEPLER9_TTV_SEED",    "42"))
const N_STEPS   = parse(Int, get(ENV, "KEPLER9_TTV_STEPS",   "2000"))
const N_WALKERS = parse(Int, get(ENV, "KEPLER9_TTV_WALKERS", "60"))
const N_TEMPS   = parse(Int, get(ENV, "KEPLER9_TTV_TEMPS",   "4"))
const N_BURNIN  = parse(Int, get(ENV, "KEPLER9_TTV_BURNIN",  "500"))

println("\nsample_ptemcee: $N_TEMPS temps × $N_WALKERS walkers × $N_STEPS steps")
t0 = time()
result = sample_ptemcee(target, data;
    n_temps   = N_TEMPS,
    n_walkers = N_WALKERS,
    n_steps   = N_STEPS,
    n_burnin  = N_BURNIN,
    seed      = SEED,
    show_progress = true,
)
elapsed = time() - t0
@printf("Done in %.1f s  (%.0f evals/sec)\n",
        elapsed, result.n_evals / elapsed)

chains = result.chains
save_chains(joinpath(OUT_DIR, "kepler9_ttv_chains.nc"), chains, params; data = data)

# --- Extract TTV posteriors -------------------------------------------
function _stats(v)
    return (med = median(v), lo = quantile(v, 0.16), hi = quantile(v, 0.84))
end

# Planet b
obs_b_sorted = sort(collect(obs_b))
δt_b_med = Float64[]; δt_b_lo = Float64[]; δt_b_hi = Float64[]
for i in obs_b_sorted
    s = _stats(vec(Array(chains[:, Symbol("ttv_k1_t$(i+1)"), :])))
    push!(δt_b_med, s.med); push!(δt_b_lo, s.lo); push!(δt_b_hi, s.hi)
end

# Planet c
obs_c_sorted = sort(collect(obs_c))
δt_c_med = Float64[]; δt_c_lo = Float64[]; δt_c_hi = Float64[]
for i in obs_c_sorted
    s = _stats(vec(Array(chains[:, Symbol("ttv_k2_t$(i+1)"), :])))
    push!(δt_c_med, s.med); push!(δt_c_lo, s.lo); push!(δt_c_hi, s.hi)
end

@printf("\nKepler-9 b TTV peak-to-peak: %.1f min  (median errorbar %.2f min)\n",
        (maximum(δt_b_med) - minimum(δt_b_med)) * 24*60,
        median(δt_b_hi .- δt_b_lo) / 2 * 24*60)
@printf("Kepler-9 c TTV peak-to-peak: %.1f min  (median errorbar %.2f min)\n",
        (maximum(δt_c_med) - minimum(δt_c_med)) * 24*60,
        median(δt_c_hi .- δt_c_lo) / 2 * 24*60)
@printf("Holman+ 2010 reports b ~20 min, c ~80 min peak-to-peak.\n")

# --- Plot: two panels stacked (b on top, c on bottom) -----------------
using CairoMakie

set_theme!(nereus_theme())
fig = Figure(size = (1100, 700))
for (k, (idx, med, lo, hi, lbl)) in enumerate([
        (obs_b_sorted, δt_b_med, δt_b_lo, δt_b_hi, "Kepler-9 b"),
        (obs_c_sorted, δt_c_med, δt_c_lo, δt_c_hi, "Kepler-9 c")])
    ax = Axis(fig[k, 1];
               xlabel = k == 2 ? "Transit number" : "",
               ylabel = "$(lbl)  O − C [min]")
    xs = collect(idx)
    y_min = 24*60 .* med
    lo_min = 24*60 .* (med .- lo)
    hi_min = 24*60 .* (hi .- med)
    hlines!(ax, [0.0]; color = (:black, 0.5), linestyle = :dash, linewidth = 1)
    errorbars!(ax, xs, y_min, lo_min, hi_min;
                color = :black, linewidth = 0.9, whiskerwidth = 4)
    scatter!(ax, xs, y_min;
              color = :black, marker = :circle, markersize = 7,
              strokewidth = 0)
end
save(joinpath(OUT_DIR, "oc_diagram_kepler9.png"), fig)
println("\nWrote $(OUT_DIR)/oc_diagram_kepler9.png")

println("\n" * "="^70)
println("Demo complete. See $(OUT_DIR)/")
println("="^70)
