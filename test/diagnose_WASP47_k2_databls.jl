#!/usr/bin/env julia
# Fully DATA-DRIVEN: no published Tc (those don't align this data — slc-b
# folded to 89ppm on the published Tc), no sample_map, no GP. Just:
#   runmed-flatten -> iterative BLS (find b, mask, find d, mask, find e) ->
#   fold each on its MEASURED period + empirical (tcscan) Tc.
# Tells us what is actually recoverable in the everest 30-min LC and at what
# ephemeris, with zero trust in external numbers.

using Nereus
using CairoMakie
using Statistics: std, median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb_lit, Pd_lit, Pe_lit = 4.1591287, 9.030672, 0.789593
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err

runmed(y,w)=(n=length(y); m=similar(y); h=w÷2; for i in 1:n; lo=max(1,i-h); hi=min(n,i+h); m[i]=median(@view y[lo:hi]); end; m)
flat = f ./ runmed(f,151)
@printf("everest 30-min: N=%d  flattened scatter=%.0fppm\n", length(t), std(flat)*1e6)

function binfold(flux, ph, ww; nb=160)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(ww[sel].*flux[sel])/sum(ww[sel]))
    end
    bx,by
end
# Robust phase-fold binning for the DISPLAY: bins span the plotted window with
# ~target cadences each (so b/d aren't over-binned and e isn't under-populated),
# robust MEDIAN per bin (kills outlier-pulled "nasty" points), plotted at the
# bin's MEAN phase, with a robust standard error per bin.
function binfold_plot(flux, ph, win; target=15)
    inwin = abs.(ph) .< win
    nb = clamp(round(Int, count(inwin)/target), 8, 60)
    edges = range(-win, win; length=nb+1); bx=Float64[]; by=Float64[]; be=Float64[]
    for k in 1:nb
        sel = inwin .& (ph .>= edges[k]) .& (ph .< edges[k+1]); n=count(sel); n<5 && continue
        fb = flux[sel]; m = median(fb)
        push!(bx, sum(ph[sel])/n); push!(by, m); push!(be, 1.4826*median(abs.(fb .- m))/sqrt(n))
    end
    bx,by,be
end
wn=1.0 ./ e.^2
function tcscan(flux,P,keep)
    ph=@. mod((t-t[1])/P+0.5,1.0)-0.5
    bx,by=binfold(flux[keep],ph[keep],wn[keep])
    isempty(by) && (@warn "tcscan: empty fold (count(keep)=$(count(keep)))"; return t[1])
    t[1]+bx[argmin(by)]*P
end

# iterative BLS, masking each found signal before the next search
keep=trues(length(t))
found=NamedTuple[]
for (lab,Plo,Phi,win) in [("1",2.0,6.0,0.04),("2",6.0,12.0,0.05),("3",0.6,1.0,0.10)]  # e in [0.6,1.0] -> fundamental, not 2x harmonic
    periods=exp.(range(log(Plo),log(Phi);length=12000))
    P0,dep,t0,snr,dur=Nereus._box_least_squares(t[keep],flat[keep],e[keep],periods; n_phase_bins=300, n_peaks=5)
    isempty(P0) && (println("search $lab: nothing"); continue)
    j=argmax(snr); P=P0[j]; durc=dur[j]<=0 ? 0.04*P : dur[j]
    @printf("search %s: count(keep)=%d  P=%.6f  SNR=%.1f  BLSdepth=%dppm  dur=%.3fd\n", lab,count(keep),P,snr[j],round(Int,dep[j]*1e6),durc)
    T0=tcscan(flat,P,keep)
    push!(found,(lab=lab,P=P,T0=T0,snr=snr[j],dep=dep[j],dur=durc))
    keep .&= .!mask_transits(t,[P],[T0]; window=clamp(1.5*durc, 0.05, 0.18))
end

# fold each found signal on its MEASURED P + empirical Tc
fig=Figure(size=(1200,1000))
for (i,fr) in enumerate(found)
    ph=@. mod((t-fr.T0)/fr.P+0.5,1.0)-0.5
    # which literature planet is this closest to?
    pl = argmin(abs.([fr.P-Pb_lit, fr.P-Pd_lit, fr.P-Pe_lit])); name=["b","d","e"][pl]; Plit=[Pb_lit,Pd_lit,Pe_lit][pl]
    win = name=="e" ? 0.18 : 0.06
    sel=abs.(ph).<win
    ax=Axis(fig[i,1]; xlabel=i==length(found) ? "phase" : "", ylabel="flat flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, ph[sel], flat[sel]; markersize=4, color=(:steelblue,0.5))
    bx,by,be = binfold_plot(flat, ph, win; target = name=="e" ? 25 : 12)
    errorbars!(ax, bx, by, be; color=(:red,0.45), whiskerwidth=0, linewidth=1.5)
    scatter!(ax, bx, by; markersize=9, color=:red)
    lo=minimum(by .- be); hi=maximum(by .+ be); pad=0.2*(hi-lo)+5e-5; ylims!(ax,lo-pad,hi+pad)
    depth0=1.0-minimum(by[abs.(bx).<0.02])
    @printf(">> search %s -> %s? measured P=%.6f (lit %.6f, Δ=%.1e)  depth@0=%dppm  SNR=%.1f\n",
            fr.lab,name,fr.P,Plit,(fr.P-Plit)/Plit, round(Int,depth0*1e6), fr.snr)
    text!(ax,0.01,0.97; text="search $(fr.lab) → likely $name: P=$(round(fr.P,digits=5))d  Tc=$(round(fr.T0,digits=4))  depth≈$(round(Int,depth0*1e6))ppm  SNR=$(round(fr.snr,digits=1))",
          space=:relative, align=(:left,:top), fontsize=13)
end
out=joinpath(@__DIR__,"WASP47_k2_databls.png"); save(out,fig; px_per_unit=2); println("saved $out")
