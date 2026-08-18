#!/usr/bin/env julia
# The exact LC fed to the clean K2 BLS: jitter-floor constrained-GP detrend
# + 5σ outlier clip (transits protected). Full range + locus zoom.

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
function tcscan(P)
    ph=@. mod((t-t1)/P+0.5,1.0)-0.5; bx,by=binfold(f,ph,w0); t1+bx[argmin(by)]*P
end
function runmed(y,w); n=length(y); m=similar(y); h=w÷2; for i in 1:n; lo=max(1,i-h); hi=min(n,i+h); m[i]=median(@view y[lo:hi]); end; m; end

mask = mask_transits(t,[Pb],[tcscan(Pb)];window=0.05) .|
       mask_transits(t,[Pd],[tcscan(Pd)];window=0.04) .|
       mask_transits(t,[Pe],[tcscan(Pe)];window=0.07)
oot=.!mask
jit = 1.4826*median(abs.((f.-runmed(f,49))[oot] .- median((f.-runmed(f,49))[oot])))
e_eff=sqrt.(e.^2 .+ jit^2)
gpp=Dict{String,PriorSpec}("gp_log_omega0_phot"=>UniformPrior(-3.0,0.3),"gp_log_Q_phot"=>UniformPrior(-1.0,0.5))
fc0=detrend_gp(t,f,e_eff,CeleriteSHO(); transit_mask=mask, sector_id=ones(Int,length(t)), joint_segments=false, gp_priors=gpp).flux_detrended
σ=1.4826*median(abs.(fc0[oot] .- median(fc0[oot]))); keep=mask .| (abs.(fc0 .- 1.0) .< 5σ)
tk=t[keep]; fck=fc0[keep]
@printf("jitter=%.0fppm clipped=%d OOT σ=%.0fppm\n", jit*1e6, count(.!keep), std(fck[.!(mask[keep])])*1e6)

t0=minimum(t)
fig=Figure(size=(1300,800))
ax1=Axis(fig[1,1]; ylabel="detrended flux (full)", xgridvisible=false, ygridvisible=false)
scatter!(ax1, tk.-t0, fck; markersize=3, color=(:black,0.55))
text!(ax1,0.005,0.98; text="K2 jitter-GP detrended + 5σ clip — the LC the BLS used", space=:relative, align=(:left,:top), fontsize=14)
ax2=Axis(fig[2,1]; xlabel="time − t₀ [d]", ylabel="locus zoom", xgridvisible=false, ygridvisible=false)
scatter!(ax2, tk.-t0, fck; markersize=3, color=(:black,0.55)); ylims!(ax2, 0.9990, 1.0010)
text!(ax2,0.005,0.98; text="zoom ±1000ppm", space=:relative, align=(:left,:top), fontsize=14)
out=joinpath(@__DIR__,"WASP47_k2_final_lc.png"); save(out,fig; px_per_unit=2); println("saved $out")
