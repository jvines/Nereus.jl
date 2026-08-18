#!/usr/bin/env julia
# K2 GP detrend DONE PROPERLY: CeleriteSHO with the timescale (omega0) and
# quality factor (Q) constrained so the GP can ONLY capture the slow bowl,
# not overfit short-timescale noise/transits. The default omega0 prior
# (Uniform(-3,5)) allows ω0 up to e^5≈148/d (~1-hr timescale) which let the
# fit rail short and invent spurious peaks. Cap it to long timescales.

using Nereus
using CairoMakie
using Statistics: std
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pb, Pd, Pe = 4.1591287, 9.030672, 0.789593
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err; t1=t[1]; w_iv=1.0 ./ e.^2; sid=ones(Int,length(t))

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
mask = mask_transits(t,[Pb],[tc_scan(f,Pb)];window=0.04) .|
       mask_transits(t,[Pd],[tc_scan(f,Pd)];window=0.03) .|
       mask_transits(t,[Pe],[tc_scan(f,Pe)];window=0.06)
oot = .!mask

# CONSTRAINED priors: omega0 -> long timescales; Q -> OVERDAMPED (no ring).
#   ω0 ∈ exp[-3, 0.3] /d  ⇒ timescale ∈ [4.6, 125] d  (>> transit 0.1d)
#   Q  ∈ exp[-2.5,-0.7]   ⇒ 0.08–0.5 OVERDAMPED: SHO kernel = sum of real
#   exponentials, monotonic decay, NO oscillatory ringing around the
#   masked transit gaps (that ringing caused the upward bumps everywhere).
gp_priors = Dict{String,PriorSpec}(
    "gp_log_omega0_phot" => UniformPrior(-3.0, 0.3),
    "gp_log_Q_phot"      => UniformPrior(-2.5, -0.7),
)
res = detrend_gp(t,f,e,CeleriteSHO(); transit_mask=mask, sector_id=sid,
                 joint_segments=false, gp_priors=gp_priors)
fc = res.flux_detrended
gp = res.gp_params[1]
@printf("GP params [log_S0,log_Q,log_omega0] = %s\n", round.(gp; digits=3))
@printf("  ⇒ Q=%.2f  ω0=%.3f /d  timescale=%.1f d   OOT residual=%.0f ppm\n",
        exp(gp[2]), exp(gp[3]), 2π/exp(gp[3]), std(fc[oot])*1e6)

t0=minimum(t)
fig=Figure(size=(1200,600))
ax=Axis(fig[1,1]; xlabel="time − t₀ [d]", ylabel="GP-detrended flux (constrained)", xgridvisible=false, ygridvisible=false)
scatter!(ax, t.-t0, fc; markersize=4, color=(:black,0.5)); ylims!(ax, 0.998, 1.002)
text!(ax,0.005,0.98; text="K2 constrained-GP detrend  τ=$(round(2π/exp(gp[3]),digits=1))d  OOT σ=$(round(Int,std(fc[oot])*1e6))ppm", space=:relative, align=(:left,:top), fontsize=15)
out=joinpath(@__DIR__,"WASP47_k2_gp_constrained.png"); save(out,fig; px_per_unit=2); println("saved $out")
