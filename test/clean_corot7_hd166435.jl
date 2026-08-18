#!/usr/bin/env julia
# Notch-detrend the TESS LCs for CoRoT-7 (planet+activity positive test) and
# HD 166435 (activity-only NULL test). Per-sector CSVs were fetched from the
# exoautomata MinIO by /tmp/fetch_lcs.py. Notch preserves transits as residual
# dips (delta_bic>0), so CoRoT-7b needs no mask; HD 166435 has no planet so it
# acts as a local-polynomial activity detrender.
#
# Run: julia --project=. -t 6 Nereus.jl/test/clean_corot7_hd166435.jl

using Nereus, Statistics, Printf
const ROOT = abspath(joinpath(@__DIR__, "..", ".."))

# (target dir, file prefix, notch window [d] — < P_rot, > transit dur)
TARGETS = [
    ("CoRoT-7",  "corot7",   1.00),   # P_rot ~23 d, CoRoT-7b dur ~0.06 d
    ("HD166435", "hd166435", 0.50),   # P_rot ~3.8 d, no planet (NULL)
]

for (tdir, pref, win) in TARGETS
    d = joinpath(ROOT, "data", tdir)
    files = sort(filter(f -> occursin(Regex("^$(pref)_tess_s\\d+_lc\\.csv\$"), f), readdir(d)))
    isempty(files) && (@warn "no sector CSVs in $d for $pref"; continue)
    @printf("\n===== %s : %d sectors, notch window=%.2f d =====\n", tdir, length(files), win)
    secs = LightCurve[]
    for fn in files
        lc = load_tess_lc(joinpath(d, fn))
        push!(secs, LightCurve(lc.t, lc.flux, lc.flux_err))
        m = match(r"_s(\d+)_", fn)
        @printf("  %-26s n=%6d  rms=%.4f\n", fn, length(lc.t), std(lc.flux))
    end
    lc = LightCurve(secs)                          # concat sectors (sorted by time)
    t0 = time()
    cl = clean_lightcurve(lc; method=:notch, window=win)
    @printf("  notch done in %.0f s | method=%s | transit-flagged cadences: %d/%d\n",
            time()-t0, cl.method, count(cl.is_transit), length(cl.t))
    raw_sd = std(cl.flux); det_sd = std(cl.flux_detrended)
    @printf("  raw σ=%.4e  ->  cleaned σ=%.4e   (%.1f× reduction)\n",
            raw_sd, det_sd, raw_sd/det_sd)

    outdir = joinpath(ROOT, "data", tdir); mkpath(outdir)
    base = lowercase(replace(tdir, "-"=>""))
    out = joinpath(outdir, "$(base)_cleaned_lc.csv")     # load_tess_lc format
    open(out, "w") do io
        println(io, "# notch-cleaned TESS LC | window=$(win) d | raw_sigma=$(raw_sd) cleaned_sigma=$(det_sd)")
        println(io, "bjd_tdb,pdcsap_flux_norm,pdcsap_flux_err_norm")
        for i in eachindex(cl.t)
            @printf(io, "%.8f,%.8f,%.8f\n", cl.t[i], cl.flux_detrended[i], cl.flux_err[i])
        end
    end
    @printf("  wrote %s (%d cadences)\n", out, length(cl.t))
    # detrending plot (raw+trend / cleaned)
    try
        resdir = joinpath(ROOT, "results", "$(tdir)_clean"); mkpath(resdir)
        plot_detrending(cl.t, cl.flux, cl.flux_err, cl.trend;
                        output=joinpath(resdir, "$(base)_detrend.png"),
                        sector_id=cl.sector_id, name=tdir)
        @printf("  wrote %s/%s_detrend.png\n", resdir, base)
    catch err
        @warn "plot_detrending failed" exception=(err, catch_backtrace())
    end
end
println("\nDONE")
