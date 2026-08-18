#!/usr/bin/env julia
# TESS-only (s42+s92) phase folds. Period = literature (coherent over the
# 1381-d TESS span; the 7-yr K2 drift is what broke it, so K2 is dropped).
# T0 from a Tc-scan REFERENCED AT THE DATA START (lever-arm = baseline, not
# the BJD origin — that origin lever-arm was the bug that de-centered even
# b). Heavy binning to pull the shallow d/e above the ~5500 ppm TESS noise.

using Nereus
using CairoMakie
using Statistics: quantile

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
files = ["WASP-47_tess_s42_notch_w1p0.csv","WASP-47_tess_s92_notch_w1p0.csv"]
t=Float64[]; f=Float64[]; e=Float64[]
for fn in files
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(t,lc.t); append!(f,lc.flux); append!(e,lc.flux_err); end
p=sortperm(t); t=t[p]; f=f[p]; e=e[p]
t1=t[1]; w=1.0 ./ e.^2

# bin a fold (phase in [-0.5,0.5)) -> (binx, biny)
function binfold(ph, fl, ww; nb=140)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb; sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(ww[sel].*fl[sel])/sum(ww[sel])); end
    bx,by
end
refph(P,T0) = @. mod((t - T0)/P + 0.5, 1.0) - 0.5

# literature periods (Vanderburg+2017)
planets = [("b",4.1591287,0.06), ("d",9.030672,0.06), ("e",0.789593,0.18)]

fig=Figure(size=(1100,1100))
for (i,(nm,P,win)) in enumerate(planets)
    # Tc-scan referenced at t1: find the phase of the deepest binned bin,
    # then set T0 = t1 + φ_min·P so the transit sits at phase 0.
    ph0 = @. mod((t - t1)/P + 0.5, 1.0) - 0.5
    bx0,by0 = binfold(ph0, f, w)
    φmin = bx0[argmin(by0)]
    T0 = t1 + φmin*P
    ph = refph(P,T0)
    sel = abs.(ph) .< win
    ax=Axis(fig[i,1]; xlabel = i==3 ? "phase" : "", ylabel="notch flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, ph[sel], f[sel]; markersize=2, color=(:gray60,0.13), rasterize=2)
    bx,by = binfold(ph[sel], f[sel], w[sel])
    scatter!(ax, bx, by; markersize=7, color=:red)
    lo=minimum(by); hi=maximum(by); pad=0.25*(hi-lo)+5e-5; ylims!(ax, lo-pad, hi+pad)
    depth = round(Int, (1.0-lo)*1e6)
    text!(ax, 0.01, 0.97; text="$nm  P=$(P)d (lit)  T0=$(round(T0,digits=3))  binned depth≈$(depth)ppm",
          space=:relative, align=(:left,:top), fontsize=14)
    println(">> $nm: T0=$(round(T0,digits=4))  binned-min depth≈$(depth)ppm")
end
out=joinpath(@__DIR__,"WASP47_tess_folds.png"); save(out,fig; px_per_unit=2); println("saved $out")
