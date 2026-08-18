#!/usr/bin/env julia
# Phase-fold the GP-detrended WASP-47 LC on b, d, e. Scatter (light) +
# inverse-variance binned overlay — binning is essential to pull the
# shallow d (~840 ppm) and e (~210 ppm) dips out of the ~5500 ppm
# per-cadence TESS white noise.

using Nereus
using CairoMakie
using Statistics: quantile, mean

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
gp_files = ["WASP-47_k2_c03_everest_lc_gp.csv","WASP-47_tess_s42_lc_gp.csv","WASP-47_tess_s92_lc_gp.csv"]
t=Float64[]; f=Float64[]; e=Float64[]
for fn in gp_files
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(t,lc.t); append!(f,lc.flux); append!(e,lc.flux_err); end

# (name, P, T0, half-phase-window, nbins)
planets = [("b", 4.1591287, 2461.834+2_457_000, 0.05, 100),
           ("d", 9.030672,  2458.373+2_457_000, 0.05, 100),
           ("e", 0.789593,  2461.201+2_457_000, 0.15, 120)]

function binfold(ph, fl, fe, lo, hi, nb)
    edges = range(lo, hi; length=nb+1)
    bx=Float64[]; by=Float64[]; bs=Float64[]
    w = 1.0 ./ fe.^2
    for k in 1:nb
        sel = (ph .>= edges[k]) .& (ph .< edges[k+1])
        count(sel) < 5 && continue
        ww = w[sel]; W=sum(ww)
        push!(bx, (edges[k]+edges[k+1])/2)
        push!(by, sum(ww .* fl[sel])/W)
        push!(bs, 1/sqrt(W))
    end
    bx, by, bs
end

fig = Figure(size = (1100, 1150))
for (i,(nm,P,T0,win,nb)) in enumerate(planets)
    ph = @. mod((t - T0)/P + 0.5, 1.0) - 0.5
    sel = abs.(ph) .< win
    phs=ph[sel]; fls=f[sel]; fes=e[sel]
    ax = Axis(fig[i,1]; xlabel = i==3 ? "phase" : "", ylabel="GP-detrended flux",
              xgridvisible=false, ygridvisible=false)
    scatter!(ax, phs, fls; markersize=2, color=(:gray60,0.18), rasterize=2)
    bx,by,bs = binfold(phs, fls, fes, -win, win, nb)
    errorbars!(ax, bx, by, bs; color=:black, linewidth=1)
    scatter!(ax, bx, by; markersize=7, color=:red)
    # robust ylim from the binned curve so the shallow dip is visible
    lo=minimum(by)-3*maximum(bs); hi=maximum(by)+3*maximum(bs)
    ylims!(ax, lo, hi)
    text!(ax, 0.005, 0.98; text="$nm   P=$(P)d   (binned: red)",
          space=:relative, align=(:left,:top), fontsize=15)
end
out = joinpath(@__DIR__, "WASP47_phasefolds_gp.png")
save(out, fig; px_per_unit=2)
println("saved $out")
