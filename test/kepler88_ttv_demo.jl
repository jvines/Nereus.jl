#!/usr/bin/env julia
# Kepler-88 b (KOI-142) TTV demo — the canonical "King of TTVs" system.
# Single transiting planet (b) with ~12-hour peak-to-peak TTVs caused by
# a non-transiting outer perturber (c) near the 2:1 MMR. TTVs are HUGE
# in long-cadence Kepler data (Nesvorný+ 2013, Holczer+ 2016) — there
# is no way to miss them if the pipeline works.
#
# Reference: Nesvorný+ 2013 "KOI-142, The King of Transit Variations"
#            Hadden+ 2017 (system mass refinement)
#            Yoffe+ 2021 (post-Kepler refit)

using Nereus
using Statistics: median, std, mean, quantile
using Printf
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "Kepler88")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results", "Kepler88_TTV")
mkpath(OUT_DIR)

println("="^70)
println("Kepler-88 b (KOI-142) TTV demo — ~12 hr peak-to-peak TTVs")
println("="^70)

# --- Kepler-88 b ephemerides (Holczer+ 2016 / Nesvorný+ 2013) ---
const P_b_LIT  = 10.9541651
const T0_b_LIT = 2454969.20559
const M_s_LIT  = 0.99
const R_s_LIT  = 0.880
const RR_B_LIT = 0.094
const B_B_LIT  = 0.595
const Q1_K     = 0.50
const Q2_K     = 0.30
const RHO_S_LIT = 2.05      # ρ_s = 3M/(4πR³)·M_sun/R_sun³ → g/cm³ for 0.99 / 0.88

# --- Load Kepler LC ----------------------------------------------------
lc = load_tess_lc(joinpath(DATADIR, "Kepler88_kplr_lc.csv"))
t_full, flux, err = lc.t, lc.flux, lc.flux_err
@printf("Kepler LC: %d points, span %.1f d (%.1f yr)\n",
        length(t_full), t_full[end] - t_full[1],
        (t_full[end] - t_full[1])/365.25)

# --- Transit centers ---------------------------------------------------
function transits_in(t::AbstractVector, P::Real, T0::Real)
    n_lo = ceil(Int, (minimum(t) + 0.1 - T0) / P)
    n_hi = floor(Int, (maximum(t) - 0.1 - T0) / P)
    return [(n, T0 + n*P) for n in n_lo:n_hi]
end
tcs_b = transits_in(t_full, P_b_LIT, T0_b_LIT)
# Shift T0 to first observed transit so cycle indices ≥ 0.
T0_b_FIT = tcs_b[1][2]
tcs_b = transits_in(t_full, P_b_LIT, T0_b_FIT)
@printf("Predicted b transits: %d (cycle idx 0..%d)\n",
        length(tcs_b), tcs_b[end][1])

# --- Detrend (savgol, mask around transits ±0.20 d) -------------------
# b's T14 is ~5.5 hr (0.23 d) → mask window ±0.18 d generously covers it.
mask = mask_transits(t_full, [P_b_LIT], [T0_b_FIT]; window = 0.18)
dt = detrend_savgol(t_full, flux, err;
                     window_length = 401, polyorder = 3,
                     transit_mask = mask)
flux_dt = dt.flux_detrended

# --- Subset to ±0.5 d around each predicted transit (wide enough that
# even ±6-hour TTVs fall well inside the window) -----------------------
keep = falses(length(t_full))
for (_, Tc) in tcs_b
    @. keep |= abs(t_full - Tc) < 0.50
end
# Drop savgol-induced Inf/NaN at quarter-gap boundaries.
keep .&= isfinite.(flux_dt)
keep .&= isfinite.(err)
t_fit    = t_full[keep]
flux_fit = flux_dt[keep]
err_fit  = err[keep]
@printf("Subset: %d points (%.1f%% of full LC)\n",
        length(t_fit), 100*length(t_fit)/length(t_full))

data = Data(; t_phot = t_fit, flux = flux_fit, flux_err = err_fit,
              phot_inst = ones(Int, length(t_fit)))
ic = InstrumentConfig(rv = String[], pm = ["Kepler"])

# --- Priors ------------------------------------------------------------
priors = Dict{String, PriorSpec}()

# Slot 1: Kepler-88 b. Orbit pinned (TTVs absorb timing); shape free.
priors["P_k1"]      = FixedPrior(P_b_LIT)
priors["Tc_k1"]     = FixedPrior(T0_b_FIT)
priors["sesinw_k1"] = FixedPrior(0.0)
priors["secosw_k1"] = FixedPrior(0.0)
priors["b_k1"]      = NormalPrior(B_B_LIT, 0.15, 0.0, 0.95)
priors["rr_k1"]     = NormalPrior(RR_B_LIT, 0.01, 0.05, 0.15)

priors["rho_s"]         = NormalPrior(RHO_S_LIT, 0.3, 0.5, 5.0)
priors["offset_Kepler"] = NormalPrior(0.0, 5e-4, -3e-3, 3e-3)
priors["jitter_Kepler"] = LogUniformPrior(1e-5, 5e-3)
priors["q1_Kepler"]     = NormalPrior(Q1_K, 0.15, 0.0, 1.0)
priors["q2_Kepler"]     = NormalPrior(Q2_K, 0.15, 0.0, 1.0)

# TTV slots: hard bound ±0.5 d (huge — covers ±12 hr peak-to-peak with
# slack). σ = 0.25 d so the prior is effectively uniform across the
# bound, not pulling slots to zero like the Kepler-9 σ = 15 min did.
n_ttv_b = maximum(x[1] for x in tcs_b) + 1
obs_b   = Set(x[1] for x in tcs_b)
for i in 1:n_ttv_b
    priors["ttv_k1_t$i"] = (i-1) in obs_b ?
        NormalPrior(0.0, 0.25, -0.5, 0.5) :
        FixedPrior(0.0)
end

parametrization = ParametrizationConfig(time = :Tc, geom = :b_rr,
                                          use_rho_s = true)
params = Params(;
    max_kplanet     = 1,
    planet_modes    = [PM_TTV],
    instruments     = ic,
    data            = data,
    M_s             = M_s_LIT,
    R_s             = R_s_LIT,
    parametrization = parametrization,
    priors          = priors,
    stability       = :none,
    ttv_n_transits  = Dict(1 => n_ttv_b),
)
@printf("Free params: %d  (%d observed transits)\n",
        n_unfrozen(params), length(obs_b))

target = NereusTarget(params, data; unconstrained = false)

# --- Sample ------------------------------------------------------------
const SEED      = parse(Int, get(ENV, "KEPLER88_TTV_SEED",    "42"))
const N_STEPS   = parse(Int, get(ENV, "KEPLER88_TTV_STEPS",   "3000"))
# Goodman-Weare stretch ensemble needs n_walkers ≥ 2(D+1); for D~140
# this means ≥ 282. Use 320 (4× D safety margin) to ensure ergodicity.
const N_WALKERS = parse(Int, get(ENV, "KEPLER88_TTV_WALKERS", "320"))
const N_TEMPS   = parse(Int, get(ENV, "KEPLER88_TTV_TEMPS",   "4"))
const N_BURNIN  = parse(Int, get(ENV, "KEPLER88_TTV_BURNIN",  "1500"))

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
save_chains(joinpath(OUT_DIR, "kepler88_ttv_chains.nc"), chains, params; data = data)

# --- Extract TTV posteriors -------------------------------------------
function _stats(v)
    return (med = median(v), lo = quantile(v, 0.16), hi = quantile(v, 0.84))
end

obs_b_sorted = sort(collect(obs_b))
δt_b_med = Float64[]; δt_b_lo = Float64[]; δt_b_hi = Float64[]
for i in obs_b_sorted
    s = _stats(vec(Array(chains[:, Symbol("ttv_k1_t$(i+1)"), :])))
    push!(δt_b_med, s.med); push!(δt_b_lo, s.lo); push!(δt_b_hi, s.hi)
end

p2p_b   = (maximum(δt_b_med) - minimum(δt_b_med)) * 24*60
err_med = median(δt_b_hi .- δt_b_lo) / 2 * 24*60
@printf("\nKepler-88 b TTV peak-to-peak: %.1f min  (median errorbar %.2f min)\n",
        p2p_b, err_med)
@printf("Nesvorný+ 2013 reports ~720 min peak-to-peak (~12 hr).\n")

# --- Plot: single-panel O−C diagram ----------------------------------
using CairoMakie

set_theme!(nereus_theme())
fig = Figure(size = (1100, 480))
ax = Axis(fig[1, 1];
          xlabel = "Transit number",
          ylabel = "O − C [min]")
xs = collect(obs_b_sorted)
y_min = 24*60 .* δt_b_med
lo_min = 24*60 .* (δt_b_med .- δt_b_lo)
hi_min = 24*60 .* (δt_b_hi .- δt_b_med)
hlines!(ax, [0.0]; color = (:black, 0.5), linestyle = :dash, linewidth = 1)
errorbars!(ax, xs, y_min, lo_min, hi_min;
            color = :black, linewidth = 0.9, whiskerwidth = 4)
scatter!(ax, xs, y_min;
          color = :black, marker = :circle, markersize = 7,
          strokewidth = 0)
save(joinpath(OUT_DIR, "oc_diagram_kepler88.png"), fig)
println("\nWrote $(OUT_DIR)/oc_diagram_kepler88.png")

println("\n" * "="^70)
println("Demo complete. See $(OUT_DIR)/")
println("="^70)
