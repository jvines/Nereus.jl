#!/usr/bin/env julia
# K2 detrend with a LOCAL Savitzky-Golay smoother (transit-masked), which
# follows the complex EVEREST bowl without global Runge wiggles (poly) or
# GP overfitting/spurious peaks. K2 = 30-min cadence -> 1 d ≈ 49 cadences.

using Nereus
using CairoMakie
using Statistics: std
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err; t1=t[1]; w_iv=1.0 ./ e.^2; sid=ones(Int,length(t))

function binfold(flux, ph, ww; nb=140)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(ww[sel].*flux[sel])/sum(ww[sel]))
    end
    bx,by
end
function tc_scan(flux,P)
    ph=@. mod((t-t1)/P+0.5,1.0)-0.5; bx,by=binfold(flux,ph,w_iv); t1 + bx[argmin(by)]*P
end
mask = mask_transits(t,[Pb],[tc_scan(f,Pb)];window=0.04) .|
       mask_transits(t,[Pd],[tc_scan(f,Pd)];window=0.03) .|
       mask_transits(t,[Pe],[tc_scan(f,Pe)];window=0.06)

fig=Figure(size=(1200,1000)); t0=minimum(t)
for (i,wl) in enumerate([25,49,99])   # ~0.5, 1, 2 days
    res = detrend_savgol(t,f,e; window_length=wl, polyorder=3, transit_mask=mask, sector_id=sid)
    fc = res.flux_detrended
    oot = .!mask
    @printf("savgol wl=%d (%.1fd): OOT residual std=%.0f ppm\n", wl, wl/48, std(fc[oot])*1e6)
    ax=Axis(fig[i,1]; xlabel=i==3 ? "time − t₀ [d]" : "", ylabel="savgol-detrended flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, t.-t0, fc; markersize=4, color=(:black,0.5)); ylims!(ax, 0.998, 1.002)
    text!(ax,0.005,0.98; text="savgol window=$(wl) (~$(round(wl/48,digits=1))d)  OOT σ=$(round(Int,std(fc[oot])*1e6))ppm", space=:relative, align=(:left,:top), fontsize=14)
end
out=joinpath(@__DIR__,"WASP47_k2_savgol.png"); save(out,fig; px_per_unit=2); println("saved $out")
