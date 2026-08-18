#!/usr/bin/env julia
# Working pipeline for WASP-47 b/d/e on TESS-only:
#   1. GP-detrend RAW TESS (CeleriteSHO, masks the transits so it does NOT
#      distort their timing like the notch did, and does NOT eat them).
#   2. Phase-fold b/d/e on the clean LC (lit P, data-anchored T0).
#   3. Iterative residual BLS, subtracting each planet by its EMPIRICAL
#      folded template (no model fit -> no railing, no wrong rr/LD; removes
#      exactly the observed transit).
# Two plots: folds + iterative BLS.

using Nereus
using CairoMakie
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
const DS = get(ENV, "DATASET", "tess")   # "tess" or "k2"
pm_names, files = DS=="k2" ?
    (["K2"], ["WASP-47_k2_c03_everest_lc.csv"]) :
    (["TESS42","TESS92"], ["WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"])
println("DATASET = $DS")
t=Float64[]; f=Float64[]; e=Float64[]; inst=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(t,lc.t); append!(f,lc.flux); append!(e,lc.flux_err); append!(inst,fill(ix,length(lc.t))); end
p=sortperm(t); t=t[p]; f=f[p]; e=e[p]; inst=inst[p]
t1=t[1]; w_iv=1.0 ./ e.^2

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

# ---- GP detrend: pass 1 mask b, then mask b+d+e ----
println("GP pass 1 (mask b) ...")
m1 = mask_transits(t,[Pb],[tc_scan(f,Pb)]; window=0.04)
fc1 = detrend_gp(t,f,e,CeleriteSHO(); transit_mask=m1, sector_id=inst, joint_segments=false).flux_detrended
T0b,T0d,T0e = tc_scan(fc1,Pb), tc_scan(fc1,Pd), tc_scan(fc1,Pe)
println("GP pass 2 (mask b,d,e) ...")
m2 = mask_transits(t,[Pb],[T0b];window=0.04) .| mask_transits(t,[Pd],[T0d];window=0.03) .| mask_transits(t,[Pe],[T0e];window=0.06)
fc = detrend_gp(t,f,e,CeleriteSHO(); transit_mask=m2, sector_id=inst, joint_segments=false).flux_detrended
T0b,T0d,T0e = tc_scan(fc,Pb), tc_scan(fc,Pd), tc_scan(fc,Pe)

# ---- Plot 1: phase folds on clean LC ----
fig1=Figure(size=(1100,1100))
for (i,(nm,P,T0,win)) in enumerate([("b",Pb,T0b,0.06),("d",Pd,T0d,0.06),("e",Pe,T0e,0.18)])
    ph=@. mod((t-T0)/P+0.5,1.0)-0.5; sel=abs.(ph).<win
    ax=Axis(fig1[i,1]; xlabel=i==3 ? "phase" : "", ylabel="GP-clean flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, ph[sel], fc[sel]; markersize=2, color=(:gray60,0.13), rasterize=2)
    bx,by=binfold(fc[sel], ph[sel], w_iv[sel]); scatter!(ax,bx,by; markersize=7, color=:red)
    lo=minimum(by); hi=maximum(by); pad=0.25*(hi-lo)+5e-5; ylims!(ax,lo-pad,hi+pad)
    @printf(">> fold %s: depth≈%dppm\n", nm, round(Int,(1.0-lo)*1e6))
    text!(ax,0.01,0.97; text="$nm  P=$(P)d  depth≈$(round(Int,(1.0-lo)*1e6))ppm", space=:relative, align=(:left,:top), fontsize=14)
end
save(joinpath(@__DIR__,"WASP47_folds_clean_$(DS).png"), fig1; px_per_unit=2)

# ---- empirical template subtraction ----
function tsub(flux,P,T0; nb=300)
    ph=@. mod((t-T0)/P+0.5,1.0)-0.5
    edges=range(-0.5,0.5;length=nb+1); tmpl=ones(nb)
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)>=4 && (tmpl[k]=sum(w_iv[sel].*flux[sel])/sum(w_iv[sel]))
    end
    bins=clamp.(floor.(Int,(ph .+ 0.5).*nb).+1, 1, nb)
    flux .- (tmpl[bins] .- 1.0)
end

periods = exp.(range(log(0.5), log(15.0); length=2500))
function bls_spectrum(flux)
    W=sum(w_iv); fm=sum(w_iv.*flux)/W; wf=w_iv.*(flux.-fm)
    nb=200; bw=zeros(nb); bwf=zeros(nb); durs=[0.01,0.02,0.04,0.08]; snr=similar(periods)
    @inbounds for i in eachindex(periods); snr[i],_,_,_=Nereus._bls_one_period(t,collect(w_iv),wf,W,periods[i],durs,nb,bw,bwf); end
    snr
end
snr_at(s,P0)=s[argmin(abs.(periods .- P0))]

# GREEDY: each round subtract the remaining planet with the HIGHEST BLS
# peak (within ±3% of its period), not a hardcoded order.
local_snr(s,P0;tol=0.03) = (sel=abs.(periods .- P0)./P0 .< tol; any(sel) ? maximum(s[sel]) : 0.0)
Pdict=Dict("b"=>Pb,"d"=>Pd,"e"=>Pe); T0dict=Dict("b"=>T0b,"d"=>T0d,"e"=>T0e)
remaining=["b","d","e"]; flux=copy(fc); stages=[("raw clean", bls_spectrum(fc))]; order=String[]
for r in 1:3
    global flux
    s=stages[end][2]
    best=remaining[argmax([local_snr(s,Pdict[p]) for p in remaining])]
    push!(order,best); flux=tsub(flux,Pdict[best],T0dict[best])
    deleteat!(remaining, findfirst(==(best),remaining))
    push!(stages, ("$best subtracted", bls_spectrum(flux)))
end
@printf("subtraction order (by BLS power): %s\n", join(order," → "))
fig2=Figure(size=(1200,1100))
for (i,(lab,s)) in enumerate(stages)
    ax=Axis(fig2[i,1]; xlabel=i==4 ? "period [d]" : "", ylabel="BLS SNR", xscale=log10, xgridvisible=false, ygridvisible=false)
    lines!(ax, periods, s; color=:black, linewidth=0.8)
    for (nm,P0,col) in [("b",Pb,:red),("d",Pd,:green),("e",Pe,:blue)]
        vlines!(ax,[P0]; color=(col,0.6), linewidth=1.2, linestyle=:dash); text!(ax,P0,maximum(s); text=nm, color=col, fontsize=12, align=(:center,:top)); end
    @printf(">> %-18s SNR@b=%.1f SNR@d=%.1f SNR@e=%.1f (top P=%.3f SNR=%.1f)\n", lab, snr_at(s,Pb),snr_at(s,Pd),snr_at(s,Pe), periods[argmax(s)], maximum(s))
    text!(ax,0.01,0.98; text=lab, space=:relative, align=(:left,:top), fontsize=14)
end
save(joinpath(@__DIR__,"WASP47_iterative_bls_$(DS).png"), fig2; px_per_unit=2)
println("saved WASP47_folds_clean.png + WASP47_iterative_bls.png")
