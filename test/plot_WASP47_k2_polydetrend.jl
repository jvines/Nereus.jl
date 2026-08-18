#!/usr/bin/env julia
# K2 detrend with a SMOOTH low-order polynomial (the bowl is a slow trend;
# a flexible GP overfits and invents spurious peaks). Mask b/d/e, fit a
# polynomial to the out-of-transit flux, divide. No hyperparameters, no
# ringing. Try a few degrees so we can pick the smallest that flattens it.

using Nereus
using CairoMakie
using Statistics: std, mean
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err; t1=t[1]; w_iv=1.0 ./ e.^2

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

# mask transits (out-of-transit = fit region)
oot = .!(mask_transits(t,[Pb],[tc_scan(f,Pb)];window=0.04) .|
         mask_transits(t,[Pd],[tc_scan(f,Pd)];window=0.03) .|
         mask_transits(t,[Pe],[tc_scan(f,Pe)];window=0.06))
@printf("OOT points for the fit: %d / %d\n", count(oot), length(t))

# normalized time for conditioning
x = (t .- mean(t)) ./ (maximum(t)-minimum(t))
function polydetrend(deg)
    V = hcat([x[oot].^k for k in 0:deg]...)             # OOT design matrix
    c = V \ f[oot]                                       # least-squares poly coeffs
    trend = hcat([x.^k for k in 0:deg]...) * c           # evaluate everywhere
    f ./ trend
end

fig=Figure(size=(1200,1000)); t0=minimum(t)
for (i,deg) in enumerate([3,5,7])
    fc = polydetrend(deg)
    @printf("deg %d: OOT residual std = %.0f ppm\n", deg, std(fc[oot])*1e6)
    ax=Axis(fig[i,1]; xlabel = i==3 ? "time − t₀ [d]" : "", ylabel="poly-detrended flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, t.-t0, fc; markersize=4, color=(:black,0.5))
    ql=0.998; ylims!(ax, ql, 1.002)
    text!(ax,0.005,0.98; text="poly deg $deg  OOT σ=$(round(Int,std(fc[oot])*1e6))ppm", space=:relative, align=(:left,:top), fontsize=14)
end
out=joinpath(@__DIR__,"WASP47_k2_polydetrend.png"); save(out,fig; px_per_unit=2); println("saved $out")
