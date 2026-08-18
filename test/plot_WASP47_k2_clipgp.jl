#!/usr/bin/env julia
# K2 clean: constrained-GP detrend + sigma-clip the up/down outliers (which
# persist across GP configs => they're in the DATA, not GP artifacts), while
# PROTECTING the b/d/e transits. Then re-detrend the clipped LC.

using Nereus
using CairoMakie
using Statistics: std, median
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
trmask(tt) = mask_transits(tt,[Pb],[tc_scan(f,Pb)];window=0.05) .|
             mask_transits(tt,[Pd],[tc_scan(f,Pd)];window=0.04) .|
             mask_transits(tt,[Pe],[tc_scan(f,Pe)];window=0.07)

gp_priors = Dict{String,PriorSpec}(
    "gp_log_omega0_phot" => UniformPrior(-3.0, 0.3),
    "gp_log_Q_phot"      => UniformPrior(-1.0, 0.5),
)
detr(tt,ff,ee,m) = detrend_gp(tt,ff,ee,CeleriteSHO(); transit_mask=m, sector_id=ones(Int,length(tt)),
                              joint_segments=false, gp_priors=gp_priors).flux_detrended

# pass 1: detrend raw, find outliers in the flat residual
m0 = trmask(t)
fc1 = detr(t,f,e,m0)
oot = .!m0
σ = 1.4826*median(abs.(fc1[oot] .- median(fc1[oot])))
keep = m0 .| (abs.(fc1 .- 1.0) .< 5σ)         # PROTECT transits, clip 5σ outliers elsewhere
@printf("robust σ=%.0fppm; clipped %d/%d outliers (transits protected)\n", σ*1e6, count(.!keep), length(t))

# pass 2: re-detrend the clipped LC
tk=t[keep]; fk=f[keep]; ek=e[keep]; mk=m0[keep]
fc = detr(tk,fk,ek,mk)
@printf("clipped+detrended: OOT residual=%.0f ppm\n", std(fc[.!mk])*1e6)

t0=minimum(t)
fig=Figure(size=(1200,600))
ax=Axis(fig[1,1]; xlabel="time − t₀ [d]", ylabel="clipped GP-detrended flux", xgridvisible=false, ygridvisible=false)
scatter!(ax, tk.-t0, fc; markersize=4, color=(:black,0.5)); ylims!(ax, 0.998, 1.002)
text!(ax,0.005,0.98; text="K2 sigma-clipped + constrained-GP  clipped=$(count(.!keep))  OOT σ=$(round(Int,std(fc[.!mk])*1e6))ppm",
      space=:relative, align=(:left,:top), fontsize=15)
out=joinpath(@__DIR__,"WASP47_k2_clipgp.png"); save(out,fig; px_per_unit=2); println("saved $out")
