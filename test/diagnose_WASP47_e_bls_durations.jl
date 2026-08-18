#!/usr/bin/env julia
# If e's BLS is weak / aliases to 3×Pe, is it the BASELINE (Jose's offset idea)
# or the BLS DURATION GRID being too narrow for e's wide-for-a-USP transit
# (~0.08 phase) so the box only matches at the compressed harmonic? Compare a
# NARROW duration grid (what I'd been using) vs the Nereus default WIDE grid,
# searching [0.6,3.0] so both Pe (0.79) and 3×Pe (2.37) are in range.

using Nereus
using Statistics: median, mean
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
    (phot_predictions(θ,data)[1])
end
mb=fit_transit(flat,4.158517,tcscan(flat,4.158517)); r1=flat.-mb.+1.0
md=fit_transit(r1,9.029180,tcscan(r1,9.029180));      r2=r1.-md.+1.0

function pgram(flux, periods, durs)
    w=1.0 ./ e.^2; W=sum(w); fm=sum(w.*flux)/W; wf=w.*(flux.-fm)
    tt=t.-t1; bw=zeros(300); bwf=zeros(300); s=similar(periods)
    for (i,P) in enumerate(periods); s[i],_,_,_=Nereus._bls_one_period(tt,w,wf,W,P,durs,300,bw,bwf); end
    s
end
periods=exp.(range(log(0.6),log(3.0);length=4000))
NARROW=[0.004,0.008,0.016,0.03]; WIDE=[0.01,0.02,0.04,0.08]
snr_at(s,P)= s[argmin(abs.(periods.-P))]
for (lab,durs) in [("NARROW max0.03",NARROW),("WIDE max0.08 (default)",WIDE)]
    s=pgram(r2,periods,durs); jm=argmax(s)
    @printf("%-22s  global peak P=%.4f SNR=%.1f | SNR@Pe(0.79)=%.1f  SNR@3Pe(2.37)=%.1f  -> %s wins\n",
        lab, periods[jm], s[jm], snr_at(s,0.789759), snr_at(s,2.369),
        snr_at(s,0.789759) > snr_at(s,2.369) ? "Pe" : "3Pe")
end
