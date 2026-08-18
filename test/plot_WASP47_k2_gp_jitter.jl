#!/usr/bin/env julia
# K2 GP detrend with a realistic JITTER floor. The bumps were GP-induced:
# the formal errors (~57 ppm) are far below the real scatter, so the GP
# trusted them and over-fit (its conditional mean wiggled below baseline →
# upward kicks in flux−GP). Inflating the per-point errors to a jitter
# floor (the real OOT scatter) lets the GP capture ONLY the smooth bowl.
# Constrained long timescale + transit mask as before.

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
function runmed(y,w)
    n=length(y); m=similar(y); h=w÷2
    for i in 1:n; lo=max(1,i-h); hi=min(n,i+h); m[i]=median(@view y[lo:hi]); end
    m
end
mask = mask_transits(t,[Pb],[tc_scan(f,Pb)];window=0.05) .|
       mask_transits(t,[Pd],[tc_scan(f,Pd)];window=0.04) .|
       mask_transits(t,[Pe],[tc_scan(f,Pe)];window=0.07)
oot = .!mask

# realistic jitter floor = robust OOT scatter about a running median
resid = f .- runmed(f,49)
jit = 1.4826*median(abs.(resid[oot] .- median(resid[oot])))
@printf("formal err=%.0fppm  jitter floor (real OOT scatter)=%.0fppm\n", median(e)*1e6, jit*1e6)
e_eff = sqrt.(e.^2 .+ jit^2)

gp_priors = Dict{String,PriorSpec}(
    "gp_log_omega0_phot" => UniformPrior(-3.0, 0.3),   # τ ∈ [4.6,125] d
    "gp_log_Q_phot"      => UniformPrior(-1.0, 0.5),   # ≤ critically damped
)
res = detrend_gp(t,f,e_eff,CeleriteSHO(); transit_mask=mask, sector_id=ones(Int,length(t)),
                 joint_segments=false, gp_priors=gp_priors)
fc = res.flux_detrended; gp=res.gp_params[1]
@printf("GP τ=%.1fd Q=%.2f   OOT residual=%.0fppm  (should ≈ jitter floor, NOT below it)\n",
        2π/exp(gp[3]), exp(gp[2]), std(fc[oot])*1e6)

t0=minimum(t)
fig=Figure(size=(1300,800))
ax1=Axis(fig[1,1]; ylabel="GP(jitter)-detrended flux", xgridvisible=false, ygridvisible=false)
scatter!(ax1, t.-t0, fc; markersize=3, color=(:black,0.5)); ylims!(ax1, 0.998, 1.002)
text!(ax1,0.005,0.98; text="K2 GP detrend with jitter floor=$(round(Int,jit*1e6))ppm  τ=$(round(2π/exp(gp[3]),digits=1))d", space=:relative, align=(:left,:top), fontsize=14)
ax2=Axis(fig[2,1]; xlabel="time − t₀ [d]", ylabel="zoom", xgridvisible=false, ygridvisible=false)
scatter!(ax2, t.-t0, fc; markersize=3, color=(:black,0.5)); ylims!(ax2, 0.9990, 1.0010)
text!(ax2,0.005,0.98; text="zoom ±1000ppm — bumps gone?", space=:relative, align=(:left,:top), fontsize=14)
out=joinpath(@__DIR__,"WASP47_k2_gp_jitter.png"); save(out,fig; px_per_unit=2); println("saved $out")
