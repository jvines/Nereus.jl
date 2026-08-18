#!/usr/bin/env julia
# K2 clean d/e search: jitter-floor constrained-GP detrend (no over-fit
# bumps) + light sigma-clip of the few real outliers (transits protected),
# then greedy iterative template-subtract BLS (subtract the strongest
# remaining planet each round).

using Nereus
using CairoMakie
using Statistics: std, median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err; t1=t[1]; w0=1.0 ./ e.^2

function binfold(flux, ph, ww; nb=140)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(ww[sel].*flux[sel])/sum(ww[sel]))
    end
    bx,by
end
function tcscan(tt,flux,ww,P)
    ph = @. mod((tt-tt[1])/P+0.5,1.0)-0.5
    bx,by = binfold(flux,ph,ww)
    tt[1]+bx[argmin(by)]*P
end
function runmed(y,w); n=length(y); m=similar(y); h=w÷2; for i in 1:n; lo=max(1,i-h); hi=min(n,i+h); m[i]=median(@view y[lo:hi]); end; m; end

# transit mask on raw
mask = mask_transits(t,[Pb],[tcscan(t,f,w0,Pb)];window=0.05) .|
       mask_transits(t,[Pd],[tcscan(t,f,w0,Pd)];window=0.04) .|
       mask_transits(t,[Pe],[tcscan(t,f,w0,Pe)];window=0.07)
oot = .!mask
jit = 1.4826*median(abs.((f.-runmed(f,49))[oot] .- median((f.-runmed(f,49))[oot])))
e_eff = sqrt.(e.^2 .+ jit^2)
gp_priors = Dict{String,PriorSpec}("gp_log_omega0_phot"=>UniformPrior(-3.0,0.3), "gp_log_Q_phot"=>UniformPrior(-1.0,0.5))
fc0 = detrend_gp(t,f,e_eff,CeleriteSHO(); transit_mask=mask, sector_id=ones(Int,length(t)), joint_segments=false, gp_priors=gp_priors).flux_detrended

# light sigma-clip the few outliers, PROTECT transits
σ = 1.4826*median(abs.(fc0[oot] .- median(fc0[oot])))
keep = mask .| (abs.(fc0 .- 1.0) .< 5σ)
@printf("jitter=%.0fppm  OOT σ=%.0fppm  clipped %d/%d outliers\n", jit*1e6, σ*1e6, count(.!keep), length(t))
tk=t[keep]; fck=fc0[keep]; ek=e_eff[keep]; wk=1.0 ./ ek.^2; t1k=tk[1]

# template subtract + BLS on the clean LC
function tsub(flux,P)
    T0=tcscan(tk,flux,wk,P); ph=@. mod((tk-T0)/P+0.5,1.0)-0.5
    nb=300; edges=range(-0.5,0.5;length=nb+1); tmpl=ones(nb)
    for k in 1:nb; sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)>=4 && (tmpl[k]=sum(wk[sel].*flux[sel])/sum(wk[sel])); end
    bins=clamp.(floor.(Int,(ph .+ 0.5).*nb).+1,1,nb); flux .- (tmpl[bins] .- 1.0)
end
periods = exp.(range(log(0.5), log(15.0); length=2500))
function blsspec(flux)
    W=sum(wk); fm=sum(wk.*flux)/W; wf=wk.*(flux.-fm); nb=200; bw=zeros(nb); bwf=zeros(nb); durs=[0.01,0.02,0.04,0.08]; s=similar(periods)
    @inbounds for i in eachindex(periods); s[i],_,_,_=Nereus._bls_one_period(tk,wk,wf,W,periods[i],durs,nb,bw,bwf); end; s
end
lsnr(s,P0;tol=0.03)=(sel=abs.(periods .- P0)./P0 .< tol; any(sel) ? maximum(s[sel]) : 0.0)

Pd_=Dict("b"=>Pb,"d"=>Pd,"e"=>Pe); rem=["b","d","e"]; flux=copy(fck); stages=[("raw clean",blsspec(fck))]; ord=String[]
for r in 1:3
    global flux
    s=stages[end][2]; best=rem[argmax([lsnr(s,Pd_[p]) for p in rem])]
    push!(ord,best); flux=tsub(flux,Pd_[best]); deleteat!(rem,findfirst(==(best),rem)); push!(stages,("$best subtracted",blsspec(flux)))
end
@printf("subtraction order (by power): %s\n", join(ord," → "))
fig=Figure(size=(1200,1100))
for (i,(lab,s)) in enumerate(stages)
    ax=Axis(fig[i,1]; xlabel=i==4 ? "period [d]" : "", ylabel="BLS SNR", xscale=log10, xgridvisible=false, ygridvisible=false)
    lines!(ax,periods,s; color=:black, linewidth=0.8)
    for (nm,P0,c) in [("b",Pb,:red),("d",Pd,:green),("e",Pe,:blue)]; vlines!(ax,[P0]; color=(c,0.6), linestyle=:dash); text!(ax,P0,maximum(s); text=nm,color=c,fontsize=12,align=(:center,:top)); end
    @printf(">> %-16s SNR@b=%.1f SNR@d=%.1f SNR@e=%.1f (top P=%.3f SNR=%.1f)\n", lab, lsnr(s,Pb),lsnr(s,Pd),lsnr(s,Pe), periods[argmax(s)], maximum(s))
    text!(ax,0.01,0.98; text=lab, space=:relative, align=(:left,:top), fontsize=14)
end
save(joinpath(@__DIR__,"WASP47_k2_clean_bls.png"), fig; px_per_unit=2); println("saved WASP47_k2_clean_bls.png")
