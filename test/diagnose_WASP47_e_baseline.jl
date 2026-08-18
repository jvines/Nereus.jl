#!/usr/bin/env julia
# Is planet e's BLS hurt by a non-flat LOCAL baseline (vs a missing constant
# offset, which Kovács BLS already floats)? Fit+subtract b and d, fold the
# residual on e WIDE, and compare the transit depth measured against (a) the
# GLOBAL mean (what BLS uses) vs (b) a LOCAL linear baseline fit to the flanks.
# If (b) differs from (a), it's a local-TREND problem, not an offset problem.

using Nereus
using CairoMakie
using Statistics: std, median, mean
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err; t1=t[1]
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
flat=f./runmed(f,151)

function tcscan(flux,P)
    ph=@. mod((t-t1)/P+0.5,1.0)-0.5; w=1.0 ./ e.^2
    edges=range(-0.5,0.5;length=121); best=Inf; bc=0.0
    for k in 1:120
        m=(ph.>=edges[k]).&(ph.<edges[k+1]); count(m)<4 && continue
        v=sum(w[m].*flux[m])/sum(w[m]); v<best && (best=v; bc=(edges[k]+edges[k+1])/2)
    end
    t1+bc*P
end
function fit_transit(flux, Pseed, T0seed; sigP=1.5e-3)
    data=Data(; t_phot=t, flux=flux, flux_err=e, phot_inst=ones(Int,length(t)))
    pri=Dict{String,PriorSpec}()
    pri["P_k1"]=NormalPrior(Pseed,sigP,Pseed-6*sigP,Pseed+6*sigP)
    pri["Tc_k1"]=NormalPrior(T0seed,0.05,T0seed-0.2,T0seed+0.2)
    pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.003,0.20)
    pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(0.71,0.2,0.1,5.0)
    params=Params(;max_kplanet=1,planet_modes=[PM_ONLY],instruments=InstrumentConfig(pm=["K2"]),data=data,M_s=1.04,R_s=1.137,
        parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,stability=:none)
    res=sample_map(NereusTarget(params,data;unconstrained=true); n_starts=8, maxiter=4000)
    θ=Theta(params); for (nm,v) in zip(res.param_names,res.x_map); set_param!(θ,nm,v); end
    model,_=phot_predictions(θ,data); blk=params.layout.planet_blocks[1]
    (model=model, P=planet_P(θ,1), Tc=planet_time_anchor(θ,1), rr=θ.values[blk.r])
end

fb=fit_transit(flat, 4.158517, tcscan(flat,4.158517)); r1=flat .- fb.model .+ 1.0
fd=fit_transit(r1, 9.029180, tcscan(r1,9.029180));     r2=r1 .- fd.model .+ 1.0
Pe,Tce = 0.789759, tcscan(r2,0.789759)
@printf("e fold: P=%.6f Tc=%.4f\n", Pe, Tce)

ph=@. mod((t-Tce)/Pe+0.5,1.0)-0.5
# windows (e dur ~0.02 d -> half-phase q ~0.013)
q = 0.013
intr  = abs.(ph) .< q
flank = (abs.(ph) .>= q) .& (abs.(ph) .< 4q)     # just outside transit
far   = (abs.(ph) .>= 0.08) .& (abs.(ph) .< 0.30) # global OOT
# local linear baseline from the flanks, evaluated at phase 0
xf=ph[flank]; yf=r2[flank]; n=length(xf)
sx=sum(xf); sy=sum(yf); sxx=sum(xf.^2); sxy=sum(xf.*yf)
slope=(n*sxy-sx*sy)/(n*sxx-sx^2); icpt=(sy-slope*sx)/n
B_local = icpt                         # baseline at phase 0 from flanks
B_far   = median(r2[far])              # global OOT level (what BLS sees)
L       = median(r2[intr])             # in-transit level
σ = 1.4826*median(abs.(r2[far] .- B_far)); Nin=count(intr)
@printf("in-transit L = %.6f (%d pts)\n", L, Nin)
@printf("B_far  (global OOT) = %.6f  -> depth_vs_global = %d ppm\n", B_far, round(Int,(B_far-L)*1e6))
@printf("B_local(flank line) = %.6f  -> depth_vs_local  = %d ppm   (flank slope=%.3g /phase)\n", B_local, round(Int,(B_local-L)*1e6), slope)
@printf("baseline mismatch B_local-B_far = %d ppm  vs per-point σ=%d ppm, depth~103 ppm\n", round(Int,(B_local-B_far)*1e6), round(Int,σ*1e6))
@printf("SNR_global=%.1f  SNR_local=%.1f\n", (B_far-L)/(σ/sqrt(Nin)), (B_local-L)/(σ/sqrt(Nin)))

# wide fold plot
fig=Figure(size=(1100,420))
ax=Axis(fig[1,1]; xlabel="phase", ylabel="flux (b,d removed)", xgridvisible=false, ygridvisible=false)
sel=abs.(ph).<0.32
scatter!(ax, ph[sel], r2[sel]; markersize=3, color=(:steelblue,0.3))
xb,yb,eb=Nereus.bin_phasefold(ph[sel], r2[sel]; target=30, xlo=-0.32, xhi=0.32)
errorbars!(ax,xb,yb,eb; color=(:red,0.4), whiskerwidth=0); scatter!(ax,xb,yb; markersize=8, color=:red)
hlines!(ax,[B_far]; color=:black, linestyle=:dash, label="global OOT (BLS baseline)")
lines!(ax, [-4q,4q], [icpt-slope*4q, icpt+slope*4q]; color=:orange, linewidth=2.5, label="local flank baseline")
vlines!(ax,[-q,q]; color=(:gray,0.5), linestyle=:dot)
axislegend(ax; position=:rb, framevisible=false)
lo=minimum(yb.-eb); hi=maximum(yb.+eb); ylims!(ax, lo-0.3*(hi-lo), hi+0.3*(hi-lo))
save(joinpath(@__DIR__,"WASP47_e_baseline.png"), fig; px_per_unit=2); println("saved WASP47_e_baseline.png")
