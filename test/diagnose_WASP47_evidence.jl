#!/usr/bin/env julia
# Does the data actually FAVOR more planets? Compute the photometry ΔlogL gained
# by adding d and e to a b-only model (K2, where d/e live), at fitted params with
# the empirical post-fit noise. If ΔlogL >> the per-planet Occam penalty (~15-30
# nats), then b+d (≥Np=2) MUST have higher evidence than b-alone — so the
# trans-dim reporting P(Np=1)=0.97 is a SAMPLING failure, not correct Occam.

using Nereus
using Statistics: median, std
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
    end; t1+bc*P
end
function fit_transit(flux, Pseed, T0seed; sigP=1.5e-3)
    data=Data(; t_phot=t, flux=flux, flux_err=e, phot_inst=ones(Int,length(t)))
    pri=Dict{String,PriorSpec}(); pri["P_k1"]=NormalPrior(Pseed,sigP,Pseed-6*sigP,Pseed+6*sigP)
    pri["Tc_k1"]=NormalPrior(T0seed,0.05,T0seed-0.2,T0seed+0.2)
    pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.003,0.20)
    pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(0.71,0.2,0.1,5.0)
    params=Params(;max_kplanet=1,planet_modes=[PM_ONLY],instruments=InstrumentConfig(pm=["K2"]),data=data,M_s=1.04,R_s=1.137,
        parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,stability=:none)
    res=sample_map(NereusTarget(params,data;unconstrained=true); n_starts=8, maxiter=4000)
    θ=Theta(params); for (nm,v) in zip(res.param_names,res.x_map); set_param!(θ,nm,v); end
    phot_predictions(θ,data)[1]
end

mb = fit_transit(flat, 4.158517, tcscan(flat,4.158517))
r1 = flat .- mb .+ 1.0
md = fit_transit(r1, 9.029180, tcscan(r1,9.029180))
r2 = r1 .- md .+ 1.0
me = fit_transit(r2, 0.789554, tcscan(r2,0.789554); sigP=8e-4)
r3 = r2 .- me .+ 1.0

# empirical post-fit per-point noise (everything removed)
σ = 1.4826*median(abs.(r3 .- median(r3)))
chi2(r) = sum(((r .- median(r))./σ).^2)
c_b   = chi2(r1)               # residual of b-only model (d,e still in)
c_bd  = chi2(r2)               # b+d
c_bde = chi2(r3)               # b+d+e
N = length(t)
@printf("K2: N=%d  post-fit σ=%.0f ppm\n", N, σ*1e6)
@printf("χ²(b)=%.0f  χ²(b+d)=%.0f  χ²(b+d+e)=%.0f   (dof=N)\n", c_b, c_bd, c_bde)
@printf("ΔlogL adding d   = %.0f nats   (depth=%d ppm)\n", 0.5*(c_b-c_bd),  round(Int,(1-minimum(md))*1e6))
@printf("ΔlogL adding e   = %.0f nats   (depth=%d ppm)\n", 0.5*(c_bd-c_bde), round(Int,(1-minimum(me))*1e6))
@printf("ΔlogL adding d+e = %.0f nats\n", 0.5*(c_b-c_bde))
@printf("per-planet Occam penalty ≈ 15-30 nats  -> d+e favored by ~%.0fx the penalty\n", 0.5*(c_b-c_bde)/25)
println("\nVERDICT: if ΔlogL(d) >> ~25, b+d has higher evidence than b-alone,")
println("so trans-dim P(Np=1)=0.97 is a SAMPLING failure (mode not found), not Occam.")
