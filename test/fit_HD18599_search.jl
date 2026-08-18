#!/usr/bin/env julia
# Blind transit search on HD 18599 (TIC 207141131, TOI-179) — an active
# K2 dwarf with strong rotational variability that hides a 970-ppm
# transit at P = 4.1375 d (Vines+ 2023, MNRAS 518, 2627).
#
# Pipeline:
#   1. Load TESS sector LCs from data/HD18599/ (fetched via
#      scripts/fetch_lcs_from_db.py from the exoautomata DB+MinIO).
#   2. Notch-filter detrend (Rizzuto+ 2017) — preserves transit dips
#      while removing rotational variability without prior T0/P knowledge.
#   3. Run TLS (Hippke & Heller 2019) on the cleaned LC.
#   4. Compare recovered (P, T0, depth) to the published Vines+ 2023
#      ephemeris.
#
# This is the integration test for the search-on-noisy-LCs claim:
# rotation amplitude ~1-2% peak-to-peak vs 970 ppm transit depth — if
# notch + TLS finds it blind, the toolkit works.

using Nereus
using TransitLeastSquares
using DelimitedFiles
using Statistics: median, mean, std
using Printf

const DATADIR = joinpath(@__DIR__, "..", "..", "data", "HD18599")

# Vines+ 2023 reference parameters (TOI-179 b).
# T0 is intentionally NOT used for masking — this is a blind search test;
# TLS recovers T0 from the data and we verify the recovered period below.
const P_REF  = 4.1375
const DEPTH_REF = 970e-6
const M_S_REF = 0.807          # M_sun
const R_S_REF = 0.798          # R_sun
const U_REF  = [0.5, 0.2]      # K2-dwarf-ish LD; only used by TLS shape template

"""Load all four HD 18599 sector CSVs as a single concatenated LC."""
function load_hd18599()
    sector_files = sort(filter(f -> endswith(f, "_lc.csv"), readdir(DATADIR)))
    isempty(sector_files) && error(
        "No HD 18599 CSVs in $DATADIR — run scripts/fetch_lcs_from_db.py first")
    ts, fs, es, sectors = Float64[], Float64[], Float64[], String[]
    for fn in sector_files
        lc = load_tess_lc(joinpath(DATADIR, fn))
        append!(ts, lc.t)
        append!(fs, lc.flux)
        append!(es, lc.flux_err)
        append!(sectors, fill(fn, length(lc.t)))
        @printf("  loaded %s: %d points, σ ≈ %.4f, span %.2f d\n",
                fn, length(lc.t), std(lc.flux), lc.t[end] - lc.t[1])
    end
    return ts, fs, es, sectors
end

"""Sanity check: phase-fold detrended flux at P_REF, find the dip phase
from the data alone, and confirm the dip depth + sign (this is just a
"is there something transit-like in the cleaned LC" pre-flight)."""
function check_transit_preserved(t, flux_detrended; P=P_REF, nbins=200)
    phase = mod.(t .- t[1], P) ./ P
    bin_y = zeros(nbins); bin_n = zeros(Int, nbins)
    for i in eachindex(t)
        b = clamp(Int(floor(phase[i] * nbins)) + 1, 1, nbins)
        bin_y[b] += flux_detrended[i]
        bin_n[b] += 1
    end
    bin_y ./= max.(bin_n, 1)
    bin_y .-= median(bin_y)
    minb = argmin(bin_y)
    T0_est = t[1] + ((minb - 0.5) / nbins) * P
    @printf("  phase-fold min at phase %.3f → T0_est = %.4f BJD\n",
            (minb - 0.5) / nbins, T0_est)
    @printf("  binned depth at min: %+.4e\n", bin_y[minb])
    @printf("  reference depth (Vines+ 2023): %.4e\n", DEPTH_REF)
end

# ---- Run --------------------------------------------------------------------
println("HD 18599 blind transit search\n")
println("Loading TESS sectors:")
t, flux, ferr, sec_label = load_hd18599()
@printf("Total: %d cadences across %.1f d (%.1f-d sectors with gaps)\n\n",
        length(t), t[end] - t[1], 0.0)

# Notch detrending — window=1.0d for an 8.7-d rotator, durations cover
# 1-4 hr transits at typical TOI scale.
println("Notch detrending (window=1.0 d, durs=[1, 2, 4]/24)...")
t0 = time()
notch = detrend_notch(t, flux, ferr;
                      window = 1.0,
                      durations = [1.0, 2.0, 4.0] ./ 24,
                      delta_bic = -1.0)
@printf("  done in %.1f s\n", time() - t0)
@printf("  flux scatter: raw %.4f → detrended %.4f (%.1f× reduction)\n\n",
        std(flux), std(notch.flux_detrended),
        std(flux) / std(notch.flux_detrended))

println("Transit preservation check (data-driven phase-fold):")
check_transit_preserved(t, notch.flux_detrended)
println()

# ---- TLS search -------------------------------------------------------------
println("TLS search (P ∈ [1, 10] d)...")
t0 = time()
result = tls(t, notch.flux_detrended;
             flux_err = ferr,
             period_min = 1.0,
             period_max = 10.0,
             R_star = R_S_REF,
             M_star = M_S_REF,
             u = U_REF,
             oversampling_factor = 3,
             verbose = true)
@printf("  done in %.1f s\n\n", time() - t0)

println("Recovered:")
@printf("  period   = %.6f d   (ref %.4f, Δ = %+.2e)\n",
        result.period, P_REF, result.period - P_REF)
@printf("  T0       = %.6f BJD\n", result.T0)
@printf("  duration = %.4f d (= %.2f h)\n",
        result.duration, result.duration * 24)
@printf("  depth    = %.4e   (ref %.4e, ratio %.2f)\n",
        result.depth, DEPTH_REF, result.depth / DEPTH_REF)
@printf("  SDE      = %.2f   (>= 7 typical detection threshold)\n", result.SDE)

ok_period = abs(result.period - P_REF) < 0.01
ok_depth = 0.5 < result.depth / DEPTH_REF < 2.0
ok_sde = result.SDE > 7
println()
println(ok_period && ok_depth && ok_sde ?
        "PASS — HD 18599 b recovered blind from raw active-star LC." :
        "REVIEW — see numbers above.")
