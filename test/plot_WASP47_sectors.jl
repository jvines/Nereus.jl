#!/usr/bin/env julia
# Eyeball the raw WASP-47 photometry per segment (K2 C3, TESS S42, S92)
# before notch-detrending. Flux vs time-within-segment, robust y-limits so
# the variability/systematics are visible (the ~6000 ppm short-period
# forest that buries d). One panel per segment.

using Nereus
using CairoMakie
using Statistics: quantile, median

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
files = ["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
labels = ["K2 C3", "TESS S42", "TESS S92"]
cols   = [:dodgerblue3, :darkorange2, :purple3]

fig = Figure(size = (1150, 950))
for (i, (fn, lab, col)) in enumerate(zip(files, labels, cols))
    lc = load_tess_lc(joinpath(DATADIR, fn))
    t0 = minimum(lc.t)
    ax = Axis(fig[i, 1];
              xlabel = i == 3 ? "time − t₀  [d]" : "",
              ylabel = "normalized flux",
              xgridvisible = false, ygridvisible = false)
    scatter!(ax, lc.t .- t0, lc.flux; markersize = 2.5,
             color = (col, 0.35), rasterize = 2)
    ql, qh = quantile(lc.flux, [0.003, 0.997])
    pad = 0.05 * (qh - ql)
    ylims!(ax, ql - pad, qh + pad)
    rng = round(maximum(lc.t) - t0, digits = 1)
    text!(ax, 0.005, 0.98;
          text = "$lab   n=$(length(lc.t))   span=$(rng) d   t₀=$(round(t0, digits=2)) BJD",
          space = :relative, align = (:left, :top), fontsize = 15, color = :black)
end
out = joinpath(@__DIR__, "WASP47_sectors_raw.png")
save(out, fig; px_per_unit = 2)
println("saved $out")
