#!/usr/bin/env julia
# Is d/e present in the SHORT-CADENCE K2 LC? The 30-min everest/PDCSAP product
# folds to noise at d/e's published Tc (BLS locks onto 9.099 noise, not d at
# 9.03), despite d being a ~18σ transit in the discovery data. Hypothesis: the
# coarse processed LC smoothed the shallow transits away. Test the slc.
# No b-subtraction (b just scatters into d/e phase); just runmed-flatten + fold.

using Nereus
using CairoMakie
using Statistics: std, median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
const Tcb, Tcd, Tce = 2456979.7641, 2456982.349, 2456983.178

lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_slc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err
dt = median(diff(t))*1440  # cadence in minutes
@printf("slc: N=%d  cadence=%.1f min  baseline=%.1f d\n", length(t), dt, t[end]-t[1])

# remove slow bowl with a long running median (window >> transit durations)
function runmed(y,w); n=length(y); m=similar(y); h=w÷2; for i in 1:n; lo=max(1,i-h); hi=min(n,i+h); m[i]=median(@view y[lo:hi]); end; m; end
W = max(101, round(Int, 0.5/(dt/1440)) | 1)   # ~12 h window in points
flat = f ./ runmed(f, W)
@printf("flatten window=%d pts (~%.1f h)  flattened scatter=%.0fppm\n", W, W*dt/60, std(flat)*1e6)

function binfold(flux, ph, ww; nb=160)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<6 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(ww[sel].*flux[sel])/sum(ww[sel]))
    end
    bx,by
end
wn = 1.0 ./ e.^2

fig=Figure(size=(1200,1000))
for (i,(nm,P,Tc,win,exp_ppm)) in enumerate([("b",Pb,Tcb,0.05,12000),("d",Pd,Tcd,0.04,830),("e",Pe,Tce,0.18,145)])
    ph=@. mod((t-Tc)/P+0.5,1.0)-0.5; sel=abs.(ph).<win
    ax=Axis(fig[i,1]; xlabel=i==3 ? "phase" : "", ylabel="flat flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, ph[sel], flat[sel]; markersize=3, color=(:steelblue,0.35))
    bx,by=binfold(flat[sel], ph[sel], wn[sel]); scatter!(ax,bx,by; markersize=9, color=:red)
    lo=minimum(by); hi=maximum(by); pad=0.25*(hi-lo)+5e-5; ylims!(ax,lo-pad,hi+pad)
    depth0 = 1.0 - minimum(by[abs.(bx).<0.012])
    @printf(">> %s depth@phase0=%dppm  (deepest-bin=%dppm, expected ~%d)\n", nm, round(Int,depth0*1e6), round(Int,(1.0-lo)*1e6), exp_ppm)
    text!(ax,0.01,0.97; text="$nm  P=$(P)d  depth@0≈$(round(Int,depth0*1e6))ppm (exp ~$exp_ppm)", space=:relative, align=(:left,:top), fontsize=14)
end
out=joinpath(@__DIR__,"WASP47_slc_de.png"); save(out,fig; px_per_unit=2); println("saved $out")
