#!/usr/bin/env julia
# Notch-detrend the TESS LCs for the SOUTHERN nulls: HD 41248 (G, "activity,
# no planets"; CVZ, 33 sectors) and Kapteyn's star (M, refuted planets; 7
# sectors). Both have no transiting planet → Notch acts as a local-poly
# systematics/activity detrender (delta_bic flags are spurious, ignore).
# Per-sector-normalized combined CSVs fetched by /tmp/fetch_lcs2.py.
#
# Run: julia --project=. -t 6 Nereus.jl/test/clean_nulls.jl

using Nereus, Statistics, Printf
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))

# (target dir, prefix, notch window [d]) — both slow rotators (25 d / 143 d)
TARGETS = [("HD41248", "hd41248", 1.0), ("Kapteyn", "kapteyn", 1.0)]

for (tdir, pref, win) in TARGETS
    d = joinpath(ROOT, "data", tdir)
    f = joinpath(d, "$(pref)_tess_all_lc.csv")
    isfile(f) || (@warn "missing $f"; continue)
    lc0 = load_tess_lc(f)
    @printf("\n===== %s : %d cadences, raw rms=%.4e, notch window=%.2f d =====\n",
            tdir, length(lc0.t), std(lc0.flux), win)
    lc = LightCurve(lc0.t, lc0.flux, lc0.flux_err)   # segments split by time-gaps inside detrend_notch
    t0 = time()
    cl = clean_lightcurve(lc; method=:notch, window=win)
    raw_sd = std(cl.flux); det_sd = std(cl.flux_detrended)
    @printf("  notch %.0f s | method=%s | flagged(delta_bic>0, spurious): %d/%d\n",
            time()-t0, cl.method, count(cl.is_transit), length(cl.t))
    @printf("  raw σ=%.4e  ->  cleaned σ=%.4e   (%.1f× reduction)\n",
            raw_sd, det_sd, raw_sd/det_sd)
    out = joinpath(d, "$(pref)_cleaned_lc.csv")
    open(out, "w") do io
        println(io, "# notch-cleaned TESS LC | window=$(win) d | raw_sigma=$(raw_sd) cleaned_sigma=$(det_sd)")
        println(io, "bjd_tdb,pdcsap_flux_norm,pdcsap_flux_err_norm")
        for i in eachindex(cl.t)
            @printf(io, "%.8f,%.8f,%.8f\n", cl.t[i], cl.flux_detrended[i], cl.flux_err[i])
        end
    end
    @printf("  wrote %s (%d cadences)\n", out, length(cl.t))
end
println("\nDONE")
