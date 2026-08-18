#!/usr/bin/env julia
# Test the two BLS generalizations:
#  A) a/Rs-scaled (period-dependent) duration grid  [rho_star kwarg]
#  B) local-baseline ("GBLS") depth                 [flank_frac kwarg]
# plus backward-compat of the default (global, fixed-duration) path.

using Nereus
using Statistics: median
using Random: seed!
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
top(r)=(p=r[1]; s=r[4]; i=argmax(s); (p[i], s[i]))

println("=== backward-compat: default BLS still finds b ===")
pall=exp.(range(log(2.0),log(6.0);length=4000))
P,S=top(Nereus._box_least_squares(t,flat,e,pall; n_phase_bins=300, n_peaks=3))
@printf("  default top peak P=%.5f SNR=%.1f  (expect b≈4.1585)\n", P,S)
# flank_frac=0 must be byte-identical
P0,S0=top(Nereus._box_least_squares(t,flat,e,pall; n_phase_bins=300, n_peaks=3, flank_frac=0.0))
@printf("  flank_frac=0 identical: %s\n", (P==P0 && S==S0))

println("\n=== Task A: a/Rs-scaled duration grid recovers e fundamental ===")
function fit_sub(flux, Pseed, T0seed; sigP=1.5e-3)
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
    flux .- phot_predictions(θ,data)[1] .+ 1.0
end
r1=fit_sub(flat,4.158517,tcscan(flat,4.158517)); r2=fit_sub(r1,9.029180,tcscan(r1,9.029180))
tk,fk,ek = t,r2,e
pe=exp.(range(log(0.6),log(3.0);length=4000))
Pn,Sn=top(Nereus._box_least_squares(tk,fk,ek,pe; durations=[0.004,0.008,0.016,0.03], n_phase_bins=300, n_peaks=3))
rho=1.04/1.137^3
Pa,Sa=top(Nereus._box_least_squares(tk,fk,ek,pe; n_phase_bins=300, n_peaks=3, rho_star=rho))
@printf("  NARROW fixed (max0.03): top P=%.4f SNR=%.1f  -> %s\n", Pn,Sn, abs(Pn-0.7896)<0.01 ? "Pe" : abs(Pn-2.369)<0.02 ? "3Pe ALIAS" : "?")
@printf("  a/Rs grid (rho=%.3f):    top P=%.4f SNR=%.1f  -> %s\n", rho, Pa,Sa, abs(Pa-0.7896)<0.01 ? "Pe ✓" : abs(Pa-2.369)<0.02 ? "3Pe ALIAS" : "?")
@printf("  duty grid: e(0.79)->%s  d(9.03)->%s\n",
        round.(Nereus._duty_grid(rho,0.79,[0.3,0.5,0.7,1.0,1.3]),digits=3),
        round.(Nereus._duty_grid(rho,9.03,[0.3,0.5,0.7,1.0,1.3]),digits=4))

println("\n=== Task B: local baseline recovers depth on a curved baseline ===")
seed!(42)
ts=collect(range(0.0, 27.0; step=0.5/24))          # 30-min cadence, 27 d
Pt=3.0; Tc=1.0; D=8e-4; qh=0.0133                    # 800 ppm box, half-width phase
phs=@. mod((ts-Tc)/Pt+0.5,1.0)-0.5
bowl=@. -0.004*phs^2                                 # 0→-1000 ppm bowl, min under transit
tr=@. ifelse(abs(phs)<qh, -D, 0.0)
fs=1.0 .+ bowl .+ tr .+ 1e-4 .* randn(length(ts))
es=fill(1e-4, length(ts))
ps=exp.(range(log(1.5),log(6.0);length=3000))
rg=Nereus._box_least_squares(ts,fs,es,ps; durations=[0.005,0.01,0.02,0.03], n_phase_bins=300, n_peaks=3)
rl=Nereus._box_least_squares(ts,fs,es,ps; durations=[0.005,0.01,0.02,0.03], n_phase_bins=300, n_peaks=3, flank_frac=1.0)
Pg,Sg=top(rg); dg=rg[2][argmax(rg[4])]
Pl,Sl=top(rl); dl=rl[2][argmax(rl[4])]
@printf("  truth: P=%.2f depth=%d ppm\n", Pt, round(Int,D*1e6))
@printf("  GLOBAL: P=%.3f depth=%d ppm SNR=%.1f  (bowl biases depth low)\n", Pg, round(Int,dg*1e6), Sg)
@printf("  LOCAL : P=%.3f depth=%d ppm SNR=%.1f  (flank baseline tracks bowl)\n", Pl, round(Int,dl*1e6), Sl)
