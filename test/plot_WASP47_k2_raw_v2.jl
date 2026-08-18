#!/usr/bin/env julia
# Raw K2 LC, FULL y-range (no clipping), with a neutral running-median
# trend (red) overlaid — so you can judge whether the upward bumps are
# really in the data (excursions above the running median) or an artifact
# of my GP/masking. Top: full range. Bottom: y-zoom on the locus.

using Nereus
using CairoMakie
using Statistics: median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; t0=minimum(t)

# running median (window ~49 cadences ≈ 1 d), a neutral smoother
function runmed(y, w)
    n=length(y); m=similar(y); h=w÷2
    for i in 1:n; lo=max(1,i-h); hi=min(n,i+h); m[i]=median(@view y[lo:hi]); end
    m
end
trend = runmed(f, 49)

fig=Figure(size=(1300,800))
ax1=Axis(fig[1,1]; ylabel="EVEREST flux (full range)", xgridvisible=false, ygridvisible=false)
scatter!(ax1, t.-t0, f; markersize=3, color=(:black,0.5))
lines!(ax1, t.-t0, trend; color=:red, linewidth=1.0)
text!(ax1,0.005,0.98; text="raw K2 (full y-range) + running median (red)", space=:relative, align=(:left,:top), fontsize=14)

ax2=Axis(fig[2,1]; xlabel="time − t₀ [d]", ylabel="EVEREST flux (locus zoom)", xgridvisible=false, ygridvisible=false)
scatter!(ax2, t.-t0, f; markersize=3, color=(:black,0.5))
lines!(ax2, t.-t0, trend; color=:red, linewidth=1.0)
ylims!(ax2, 0.989, 1.011)   # zoom on the locus so ~1000ppm bumps above the trend are visible
text!(ax2,0.005,0.98; text="locus zoom — upward bumps above the red trend = in the data; else my artifact", space=:relative, align=(:left,:top), fontsize=14)

out=joinpath(@__DIR__,"WASP47_k2_raw_v2.png"); save(out,fig; px_per_unit=2); println("saved $out")
