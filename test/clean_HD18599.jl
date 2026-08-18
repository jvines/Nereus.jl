#!/usr/bin/env julia
# HD 18599 — *just* clean the lightcurve.
#
# Pipeline:
#   1. Load all four TESS sectors of HD 18599 (S02, S03, S29, S30).
#   2. Mask the transits using the known ephemeris (P = 4.1375 d,
#      T0 from earlier data-driven phase-fold).
#   3. Fit a CeleriteRotationFM17 GP per segment to the OOT data.
#   4. Subtract the GP trend → flat LC with transits preserved as
#      ~970 ppm residual dips.
#   5. Save the cleaned LC as CSV (`data/HD18599/HD18599_cleaned_lc.csv`)
#      so downstream fits can ingest it like any other `load_tess_lc` CSV.
#   6. Plot via `plot_detrending`: top panel = raw flux + GP overlay,
#      bottom = cleaned, "//" break marks between the four sectors.
#
# This is the "test the cleaning routine" run, with no transit fit on
# top — just verify the cleaner produces a flat LC that preserves the
# transits and looks right when plotted across the multi-sector span.

using Nereus
using TransitLeastSquares
using Statistics: median, mean, std
using Printf

const DATADIR = joinpath(@__DIR__, "..", "..", "data", "HD18599")
const RESULTS = joinpath(@__DIR__, "..", "..", "results", "HD18599_clean")

println("=" ^ 70)
println("HD 18599 — multi-sector LC cleaning (mask + GP rotation + subtract)")
println("=" ^ 70)

# ---- Load all four sectors --------------------------------------------------
# Keep only per-sector CSVs from `fetch_lcs_from_db.py` (skip the cleaned
# LC this script writes back into the same directory).
sector_files = sort(filter(
    f -> occursin(r"_tess_s\d+_", f) && endswith(f, "_lc.csv"),
    readdir(DATADIR)))
isempty(sector_files) && error(
    "No HD 18599 sector CSVs in $DATADIR — run scripts/fetch_lcs_from_db.py first")

ts, fs, es = Float64[], Float64[], Float64[]
sectors       = Int[]      # per-cadence 1-indexed sector tag
sector_labels = String[]
for (k, fn) in enumerate(sector_files)
    lc = load_tess_lc(joinpath(DATADIR, fn))
    append!(ts, lc.t)
    append!(fs, lc.flux)
    append!(es, lc.flux_err)
    append!(sectors, fill(k, length(lc.t)))
    m = match(r"_tess_(s\d+)_", fn)
    push!(sector_labels, m === nothing ? fn : m.captures[1])
    @printf("  %-50s n=%6d  σ=%.4f  span=%.2f d\n",
            fn, length(lc.t), std(lc.flux), lc.t[end] - lc.t[1])
end
n_sectors = length(sector_labels)
@printf("\nConcat: %d cadences across %.1f d, %d sectors: %s\n",
        length(ts), ts[end] - ts[1], n_sectors,
        join(sector_labels, ", "))

# ---- TLS pre-pass: derive period, first-transit T0, and duration -----------
# Cached to `data/HD18599/ephemeris.txt` (P, T0, duration on three
# lines). The TLS period scan over 228k cadences is ~270s and the
# ephemeris is run-invariant — reload unless CLEAN_HD18599_REFIT_TLS=1.
# `TransitLeastSquares.jl` reports `T0 ∈ [0, P)` — i.e. transit phase
# in absolute time, NOT offset from `ts[1]`. To get the BJD of the
# first transit center in the data, advance by integer periods until
# `T0 + n·P ≥ ts[1]`.
const EPHEM_CACHE = joinpath(DATADIR, "ephemeris.txt")
local _P, _T0, _DUR
if isfile(EPHEM_CACHE) && get(ENV, "CLEAN_HD18599_REFIT_TLS", "0") != "1"
    _P, _T0, _DUR = parse.(Float64, readlines(EPHEM_CACHE))
    @printf("\nLoaded cached ephemeris from %s: P=%.6f d, T0=%.4f, dur=%.4f d\n",
            EPHEM_CACHE, _P, _T0, _DUR)
else
    println("\nRunning notch + TLS pre-pass to derive ephemeris ...")
    t_pre = time()
    notch_pre = detrend_notch(ts, fs, es;
                              window = 1.0,
                              durations = [1.0, 2.0, 3.0, 4.0, 6.0] ./ 24,
                              delta_bic = -1.0)
    tls_res = tls(ts, notch_pre.flux_detrended;
                  flux_err = es,
                  period_min = 1.0, period_max = 10.0,
                  R_star = 0.798, M_star = 0.807,
                  u = [0.5, 0.2],
                  oversampling_factor = 3,
                  verbose = false)
    _P   = tls_res.period
    _T0  = tls_res.T0 +
           ceil((ts[1] - tls_res.T0) / tls_res.period) * tls_res.period
    _DUR = tls_res.duration
    open(EPHEM_CACHE, "w") do io
        println(io, _P); println(io, _T0); println(io, _DUR)
    end
    @printf("  done in %.1f s. P=%.6f d, T0=%.4f BJD, dur=%.4f d (%.2f h), SDE=%.1f\n",
            time() - t_pre, _P, _T0, _DUR, _DUR * 24, tls_res.SDE)
    @printf("  cached to %s\n", EPHEM_CACHE)
end
const P_REF      = _P
const T0_REF     = _T0
const DURATION_D = _DUR
# Mask half-width in phase: 1.5 × half-duration (= 75% of full duration
# on each side of T0). 50% safety margin over the actual transit shape
# without re-introducing the over-wide ±2.5 h window that came from the
# old 0.025 hardcoded constant.
const HALF_WINDOW_PHASE = 1.5 * (DURATION_D / 2) / P_REF

# ---- Build transit mask -----------------------------------------------------
mask = mask_transits(ts, P_REF, T0_REF; window = HALF_WINDOW_PHASE)
@printf("Transit mask: %d / %d cadences (%.2f%%) flagged, half-width %.3f d (%.2f h)\n",
        count(mask), length(mask), 100 * count(mask) / length(mask),
        HALF_WINDOW_PHASE * P_REF, HALF_WINDOW_PHASE * P_REF * 24)

# ---- Detrend with GP rotation kernel + transit mask -------------------------
# CeleriteRotation (two-SHO QPO kernel, astroEMPEROR's `RotationTerm`).
# Switched from `CeleriteRotationFM17` because FM17's complex term lets
# the GP track sub-P_rot features cycle-by-cycle — visible wiggle on the
# rendered trend. Two-SHO with high Q0 + low f forces a narrowband
# oscillator: rotation is fit as a near-rigid sinusoid, not per-cycle.
#
# Params: σ, period, Q0, dQ, f (real-space, NOT log).
# HD 18599: P_rot = 8.74 d (Vines+ 2023), σ ≈ 1% rotation amplitude.
init = [0.01,    # σ rotation amplitude
        8.74,    # period (Vines+ 2023)
        8.0,     # Q0 (high → narrowband rotation)
        0.5,     # dQ (extra Q for secondary mode)
        0.05]    # f (modest secondary-mode amplitude fraction)
kernel = CeleriteRotation()
gp_priors = Dict{String, PriorSpec}(
    # Period tight on Vines+ 2023 8.74 d (breaks P/P/2 multimodality).
    "gp_period_phot" => NormalPrior(8.74, 0.5, 4.0, 20.0),
    # σ in K-dwarf rotation amplitude range.
    "gp_sigma_phot"  => LogUniformPrior(1e-3, 5e-2),
    # High Q0 forces narrow-band rotation (defaults: LogUniform 0.1..10).
    "gp_Q0_phot"     => NormalPrior(8.0, 1.5, 3.0, 10.0),
    # Small dQ keeps both SHO modes ~ equally damped.
    "gp_dQ_phot"     => NormalPrior(0.0, 1.0, 0.0, 3.0),
    # Small f → primary mode at P_rot dominates over P_rot/2 harmonic.
    "gp_f_phot"      => UniformPrior(0.01, 0.20),
)
# `_fit_gp_hyperparams` freezes jitter at 0 by default — without a
# white-noise sink the GP posterior mean is forced to interpolate
# every cadence, producing wiggle on dense LCs. Unfreezing jitter
# per segment lets it absorb per-cadence noise (TESS 2-min flux_err
# ≈ 4e-4) so the GP carries only the rotation envelope.
for k in 1:n_sectors
    gp_priors["jitter_S$k"] = LogUniformPrior(1e-5, 5e-3)
end

println("\nFitting CeleriteRotation (two-SHO QPO) GP jointly across sectors " *
        "(init: P_rot=8.74 d, σ=1%, Q0=8; full 2-min cadence) ...")
t0 = time()
res = detrend_gp(ts, fs, es, kernel;
                  transit_mask = mask,
                  init_params = init,
                  gp_priors = gp_priors,
                  joint_segments = true,
                  sector_id = sectors)
@printf("Done in %.1f s. Segments: %d (gap > 0.5 d split)\n",
        time() - t0, length(res.segments))

trend_full          = res.trend
flux_detrended_full = res.flux_detrended

@printf("\nJoint GP hyperparameters (CeleriteRotation: σ, P_rot, Q0, dQ, f):\n")
for (k, p) in enumerate(res.gp_params)
    label = length(res.gp_params) == 1 ? "joint" : "seg $k"
    @printf("  %s: σ=%.4e  P=%.3f d  Q0=%.2f  dQ=%.2f  f=%.3f\n",
            label, p[1], p[2], p[3], p[4], p[5])
end

@printf("\nFitted per-sector offsets:\n")
for (k, off) in enumerate(res.offsets)
    @printf("  %s: offset = %+.4e\n", sector_labels[k], off)
end

# ---- Save cleaned LC in multiple formats -----------------------------------
joint = res.gp_params[1]
metadata = Dict(
    "star"            => "HD 18599",
    "tic_id"          => "207141131",
    "n_sectors"       => 4,
    "kernel"          => "CeleriteRotation",
    "gp_sigma"        => joint[1],
    "gp_period"       => joint[2],
    "gp_Q0"           => joint[3],
    "gp_dQ"           => joint[4],
    "gp_f"            => joint[5],
    "p_rot_d"         => joint[2],
    "transit_p_d"     => P_REF,
    "transit_t0_bjd"  => T0_REF,
    "mask_phase_half" => HALF_WINDOW_PHASE,
    "raw_sigma"       => std(fs),
    "cleaned_sigma"   => std(flux_detrended_full),
)

mkpath(DATADIR)
for (ext, fmt) in (("csv", :csv), ("nc", :nc), ("fits", :fits))
    out = joinpath(DATADIR, "HD18599_cleaned_lc.$ext")
    save_lightcurve(out, ts, flux_detrended_full, es;
                    metadata = metadata, format = fmt)
    @printf("  wrote %s (%d cadences)\n", out, length(ts))
end

@printf("\nCleaned LC summary:\n")
@printf("  raw σ        : %.4e\n", std(fs))
@printf("  cleaned σ    : %.4e   (%.1f× reduction)\n",
        std(flux_detrended_full), std(fs) / std(flux_detrended_full))
@printf("  cleaned σ OOT: %.4e\n", std(flux_detrended_full[.!mask]))

# Sanity check: phase-fold the cleaned LC at the known period, locate
# the transit dip directly. This measures the ACTUAL preserved depth
# at the transit core (±0.75 h) rather than averaging over the wider
# fit mask (±2.5 h), which dilutes the dip with OOT phase.
let
    nbins = 200
    phase = mod.(ts .- T0_REF, P_REF) ./ P_REF       # 0..1 with T0 at phase 0
    binsum = zeros(nbins); bincnt = zeros(Int, nbins)
    for i in eachindex(ts)
        b = clamp(Int(floor(phase[i] * nbins)) + 1, 1, nbins)
        binsum[b] += flux_detrended_full[i]; bincnt[b] += 1
    end
    binmean = binsum ./ max.(bincnt, 1)
    binmean .-= median(binmean)                       # subtract OOT level
    minb = argmin(binmean)
    @printf("  phase-fold (P=%.4f, %d bins) min: phase=%.3f, depth=%+.4e\n",
            P_REF, nbins, (minb - 0.5) / nbins, binmean[minb])
    @printf("  preserved depth (vs Vines+ 2023 −9.7e-4)\n")
end

# ---- Plot -------------------------------------------------------------------
# Group consecutive sectors in pairs: N sectors ⇒ ceil(N/2) plot files,
# each showing 2 sectors side by side with a "//" break between them.
# Sector boundaries come from the load order (one segment per CSV),
# NOT from gap detection — within-sector momentum-dump gaps stay
# inside their sector pane. Time axis is auto-converted to TJD
# (BJD − 2457000).
println("\nRendering plot (CairoMakie)...")
mkpath(RESULTS)
# 1 sector = 1 segment (from load order, no gap heuristics). Group
# consecutive sectors in pairs ⇒ ceil(N/2) plot files, each pane spans
# one sector, two panes per file separated by `⫽`.
const SECTORS_PER_PLOT = 2
n_groups = cld(n_sectors, SECTORS_PER_PLOT)
for g in 1:n_groups
    lo = (g - 1) * SECTORS_PER_PLOT + 1
    hi = min(g * SECTORS_PER_PLOT, n_sectors)
    in_grp = (sectors .>= lo) .& (sectors .<= hi)
    grp_idx = findall(in_grp)
    grp_label = join(sector_labels[lo:hi], "+")
    plot_detrending(ts[grp_idx], fs[grp_idx], es[grp_idx],
                    (flux_detrended = flux_detrended_full[grp_idx],
                     trend          = trend_full[grp_idx],
                     segments       = nothing,
                     transit_mask   = mask[grp_idx]);
                    output = RESULTS,
                    title  = "HD 18599 — GP rotation cleaning ($grp_label)",
                    sector_id = sectors[grp_idx],
                    name = "detrending_$grp_label",
                    fmt = :both)
end
@printf("Wrote %d plot(s) under %s/detrend/\n", n_groups, RESULTS)

println("\nDone.")
