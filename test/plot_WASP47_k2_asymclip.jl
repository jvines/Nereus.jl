#!/usr/bin/env julia
# Remove the residual UPWARD bumps (flare/systematic) from the K2 detrend
# with an ASYMMETRIC iterated sigma-clip: tight on the upside (3σ — upward
# excursions can't be transits), loose on the downside (6σ, and transit-
# masked) so transit dips are never clipped. Iterate so clusters are caught.

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
gpp=Dict{String,PriorSpec}("gp_log_omega0_phot"=>UniformPrior(-3.0,0.3),"gp_log_Q_phot"=>UniformPrior(-1.0,0.5))
detr(ee)=detrend_gp(t,f,ee,CeleriteSHO(); transit_mask=mask, sector_id=ones(Int,length(t)), joint_segments=false, gp_priors=gpp).flux_detrended

oot=.!mask
jit = 1.4826*median(abs.((f.-runmed(f,49))[oot] .- median((f.-runmed(f,49))[oot])))
e_eff=sqrt.(e.^2 .+ jit^2)
keep=trues(length(t))
local fc
for it in 1:4
    global keep, fc
    ee = copy(e_eff); ee[.!keep] .= 1e3      # drop clipped points from the GP fit
    fc = detr(ee)
    o = oot .& keep
    σ = 1.4826*median(abs.(fc[o] .- median(fc[o])))
    newkeep = .!( ((fc .- 1.0) .> 3σ) .| (((fc .- 1.0) .< -6σ) .& .!mask) )
    n_new = count(keep .& .!newkeep)
    @printf("iter %d: σ=%.0fppm  newly clipped=%d  total clipped=%d\n", it, σ*1e6, n_new, count(.!newkeep))
    keep = newkeep
    n_new == 0 && break
end

t0=minimum(t)
fig=Figure(size=(1300,800))
ax1=Axis(fig[1,1]; ylabel="detrended flux (full)", xgridvisible=false, ygridvisible=false)
scatter!(ax1, t[keep].-t0, fc[keep]; markersize=3, color=(:black,0.55))
text!(ax1,0.005,0.98; text="K2 jitter-GP + ASYMMETRIC iterated clip (3σ up / 6σ down, transit-protected)", space=:relative, align=(:left,:top), fontsize=14)
ax2=Axis(fig[2,1]; xlabel="time − t₀ [d]", ylabel="locus zoom", xgridvisible=false, ygridvisible=false)
scatter!(ax2, t[keep].-t0, fc[keep]; markersize=3, color=(:black,0.55)); ylims!(ax2,0.9990,1.0010)
text!(ax2,0.005,0.98; text="zoom ±1000ppm — upward bumps gone?", space=:relative, align=(:left,:top), fontsize=14)
out=joinpath(@__DIR__,"WASP47_k2_asymclip.png"); save(out,fig; px_per_unit=2); println("saved $out")
