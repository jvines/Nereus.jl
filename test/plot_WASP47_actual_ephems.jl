#!/usr/bin/env julia
# Measure the ACTUAL ephemerides (P, T0) of b/d/e from THIS data (BLS +
# non-binned refinement, iteratively masking found planets), then phase-
# fold on the measured ephems. Folds are anchored on a transit near the
# data start (baseline lever-arm, not the BJD origin) so a precise period
# folds cleanly. Uses the notch-detrended (transit-preserving) LC.

using Nereus
using CairoMakie
using Statistics: quantile

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
# TESS only (s42+s92), K2 dropped — single coherent mission, no 7-yr drift.
files = ["WASP-47_tess_s42_notch_w1p0.csv","WASP-47_tess_s92_notch_w1p0.csv"]
t=Float64[]; f=Float64[]; e=Float64[]
for fn in files
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(t,lc.t); append!(f,lc.flux); append!(e,lc.flux_err); end
p=sortperm(t); t=t[p]; f=f[p]; e=e[p]

# find actual (P,T0) by BLS+refine on `flux`, around a literature period
function find_ephem(flux, Plo, Phi)
    periods = exp.(range(log(Plo), log(Phi); length=4000))
    P0,dep0,t00,snr0,dur0 = Nereus._box_least_squares(t, flux, e, periods; n_phase_bins=200, n_peaks=3)
    isempty(P0) && return (NaN,NaN,NaN,NaN)
    j = argmax(snr0); Pc=P0[j]; t0c=t00[j]; durc=dur0[j]<=0 ? 0.05*Pc : dur0[j]
    w = 1.0 ./ e.^2; W=sum(w); fm=sum(w.*flux)/W; wf=w.*(flux .- fm)
    Pr,_,t0r,_,snr = Nereus._refine_bls_period(t, w, wf, W, Pc, t0c, durc)
    Tref = t[1] + mod(t0r - t[1], Pr)
    (Pr, Tref, durc, snr)
end
fold(P,T0) = @. mod((t - T0)/P + 0.5, 1.0) - 0.5
function binned(ph, fl, fe; nb=110)
    edges=range(-0.5,0.5;length=nb+1); w=1.0 ./ fe.^2; bx=Float64[]; by=Float64[]
    for k in 1:nb; sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(w[sel].*fl[sel])/sum(w[sel])); end
    bx,by
end

fl = copy(f)
fig=Figure(size=(1100,1100))
# NARROW windows around the literature periods so the shallow d/e can't
# lose to noise peaks elsewhere (TESS-only ⇒ coherent over the 1381 d span).
for (i,(nm,Plo,Phi)) in enumerate([("b",4.10,4.22),("d",8.95,9.11),("e",0.783,0.797)])
    P,T0,dur,snr = find_ephem(fl, Plo, Phi)
    println(">> $nm: measured P=$(round(P,digits=6)) d  T0=$(round(T0,digits=4))  SNR=$(round(snr,digits=1))")
    ph = fold(P,T0); win = nm=="e" ? 0.2 : 0.08; sel=abs.(ph).<win
    ax=Axis(fig[i,1]; xlabel = i==3 ? "phase" : "", ylabel="notch flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, ph[sel], fl[sel]; markersize=2, color=(:gray60,0.15), rasterize=2)
    bx,by=binned(ph[sel], fl[sel], e[sel])
    keepb=abs.(bx).<win; scatter!(ax, bx[keepb], by[keepb]; markersize=7, color=:red)
    lo=minimum(by[keepb]); hi=maximum(by[keepb]); pad=0.25*(hi-lo)+5e-5; ylims!(ax, lo-pad, hi+pad)
    text!(ax, 0.01, 0.97; text="$nm  measured P=$(round(P,digits=5))d  T0=$(round(T0,digits=3))  SNR=$(round(snr,digits=1))",
          space=:relative, align=(:left,:top), fontsize=14)
    # mask this planet out before searching the next
    global fl = ifelse.(abs.(ph) .< (dur/P), 1.0, fl)
end
out=joinpath(@__DIR__,"WASP47_actual_ephems.png"); save(out,fig; px_per_unit=2); println("saved $out")
