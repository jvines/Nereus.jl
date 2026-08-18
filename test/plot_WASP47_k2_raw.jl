#!/usr/bin/env julia
# The raw K2 LC fed to the K2 BLS analysis, with formal error bars drawn,
# so the formal-error (~57 ppm) vs actual-scatter (~1700 ppm) mismatch is
# visible by eye.

using Nereus
using CairoMakie
using Statistics: median, std
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t = lc.t; f = lc.flux; e = lc.flux_err
t0 = minimum(t)
med_err = median(e)
ptp = std(diff(f))/sqrt(2)          # point-to-point scatter
@printf("K2 everest: n=%d  median formal err=%.1f ppm  point-to-point RMS=%.1f ppm  (ratio %.0fx)\n",
        length(t), med_err*1e6, ptp*1e6, ptp/med_err)

fig = Figure(size=(1200, 600))
ax = Axis(fig[1,1]; xlabel="time − t₀ [d]", ylabel="EVEREST normalized flux",
          xgridvisible=false, ygridvisible=false)
errorbars!(ax, t .- t0, f, e; color=(:red,0.5), linewidth=0.6)   # formal errors (tiny)
scatter!(ax, t .- t0, f; markersize=4, color=(:black,0.5))
text!(ax, 0.005, 0.98;
      text="K2 C3 EVEREST raw  n=$(length(t))  formal err=$(round(Int,med_err*1e6))ppm  scatter=$(round(Int,ptp*1e6))ppm",
      space=:relative, align=(:left,:top), fontsize=15)
out = joinpath(@__DIR__, "WASP47_k2_raw.png")
save(out, fig; px_per_unit=2)
println("saved $out")
