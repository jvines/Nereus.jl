#!/usr/bin/env julia
# The ACTUAL LC fed to the K2 iterative BLS: raw K2 -> 2-pass GP detrend
# (mask b, then b+d+e), exactly as in WASP47_dework.jl. Plot flux vs time
# so we can see what the BLS sees (bowl removed? transits preserved?
# artifacts?).

using Nereus
using CairoMakie
using Statistics: std
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err; inst=ones(Int,length(t)); t1=t[1]; w_iv=1.0 ./ e.^2

function binfold(flux, ph, ww; nb=140)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(ww[sel].*flux[sel])/sum(ww[sel]))
    end
    bx,by
end
function tc_scan(flux,P)
    ph=@. mod((t-t1)/P+0.5,1.0)-0.5
    bx,by=binfold(flux,ph,w_iv)
    t1 + bx[argmin(by)]*P
end

m1 = mask_transits(t,[Pb],[tc_scan(f,Pb)]; window=0.04)
fc1 = detrend_gp(t,f,e,CeleriteSHO(); transit_mask=m1, sector_id=inst, joint_segments=false).flux_detrended
T0b,T0d,T0e = tc_scan(fc1,Pb), tc_scan(fc1,Pd), tc_scan(fc1,Pe)
m2 = mask_transits(t,[Pb],[T0b];window=0.04) .| mask_transits(t,[Pd],[T0d];window=0.03) .| mask_transits(t,[Pe],[T0e];window=0.06)
fc = detrend_gp(t,f,e,CeleriteSHO(); transit_mask=m2, sector_id=inst, joint_segments=false).flux_detrended
@printf("GP-detrended K2: scatter=%.0fppm (raw bowl removed)\n", std(fc)*1e6)

t0=minimum(t)
fig=Figure(size=(1200,600))
ax=Axis(fig[1,1]; xlabel="time − t₀ [d]", ylabel="GP-detrended flux", xgridvisible=false, ygridvisible=false)
scatter!(ax, t.-t0, fc; markersize=4, color=(:black,0.5))
text!(ax,0.005,0.98; text="K2 C3 GP-detrended (mask b,d,e) — the LC the BLS uses", space=:relative, align=(:left,:top), fontsize=15)
out=joinpath(@__DIR__,"WASP47_k2_detrended.png"); save(out,fig; px_per_unit=2); println("saved $out")
