#!/usr/bin/env julia
# Notch-detrend (Rizzuto+ 2017, transit-preserving) each WASP-47 segment
# with window=0.5 d, save the flattened LCs, and plot them for eyeballing.
# K2 has a strong slow systematic; TESS has per-orbit structure — the
# notch flattens both while keeping b/d/e transit dips.

using Nereus
using CairoMakie
using DelimitedFiles: writedlm
using Statistics: quantile, median, std
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const WIN = parse(Float64, get(ENV, "NOTCH_WINDOW", "0.5"))
const WTAG = replace(string(WIN), "." => "p")
files  = ["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
outs   = ["WASP-47_k2_c03_notch_w$WTAG.csv","WASP-47_tess_s42_notch_w$WTAG.csv","WASP-47_tess_s92_notch_w$WTAG.csv"]
labels = ["K2 C3", "TESS S42", "TESS S92"]
cols   = [:dodgerblue3, :darkorange2, :purple3]

fig = Figure(size = (1150, 950))
for (i, (fn, ofn, lab, col)) in enumerate(zip(files, outs, labels, cols))
    lc = load_tess_lc(joinpath(DATADIR, fn))
    t1 = time()
    nd = detrend_notch(lc.t, lc.flux, lc.flux_err; window = WIN)
    fd = nd.flux_detrended
    secs = time() - t1
    nclip = count(nd.outlier)
    @printf("%-9s n=%d  notch %.0fs  segs=%d  outliers=%d  resid_std=%.5f (raw %.5f)\n",
            lab, length(lc.t), secs, length(nd.segments), nclip, std(fd), std(lc.flux))
    # save flattened LC (same header convention as inputs)
    open(joinpath(DATADIR, ofn), "w") do io
        println(io, "bjd_tdb,pdcsap_flux_norm,pdcsap_flux_err_norm")
        writedlm(io, [lc.t fd lc.flux_err], ',')
    end
    t0 = minimum(lc.t)
    ax = Axis(fig[i, 1]; xlabel = i == 3 ? "time − t₀  [d]" : "",
              ylabel = "notch-detrended flux", xgridvisible = false, ygridvisible = false)
    scatter!(ax, lc.t .- t0, fd; markersize = 2.5, color = (col, 0.35), rasterize = 2)
    ql, qh = quantile(fd, [0.003, 0.997]); pad = 0.05 * (qh - ql)
    ylims!(ax, ql - pad, qh + pad)
    text!(ax, 0.005, 0.98;
          text = "$lab (notch w=$(WIN)d)  n=$(length(lc.t))  resid σ=$(round(Int, std(fd)*1e6))ppm",
          space = :relative, align = (:left, :top), fontsize = 15)
end
out = joinpath(@__DIR__, "WASP47_sectors_notch_w$WTAG.png")
save(out, fig; px_per_unit = 2)
println("saved $out")
