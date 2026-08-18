#!/usr/bin/env julia
# Iterative residual BLS on TESS-only, subtracting a PROPERLY FITTED transit
# at each stage (the depth-derived estimate was wrong-shaped and didn't
# remove b). Fit each planet on a DECIMATED LC with sample_map (tight P/Tc,
# free rr/impact/rho_s/LD — fast on ~19k pts), build the FULL-res model via
# phot_predictions, subtract, BLS the residual. Verify b's residual SNR
# drops at its period after subtraction.

using Nereus
using CairoMakie
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
pm_names = ["TESS42","TESS92"]
files = ["WASP-47_tess_s42_notch_w1p0.csv","WASP-47_tess_s92_notch_w1p0.csv"]
t=Float64[]; f=Float64[]; e=Float64[]; inst=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(t,lc.t); append!(f,lc.flux); append!(e,lc.flux_err); append!(inst,fill(ix,length(lc.t))); end
p=sortperm(t); t=t[p]; f=f[p]; e=e[p]; inst=inst[p]
t1=t[1]; w_iv=1.0 ./ e.^2

function binfold(ph, fl; nb=140)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(w_iv[sel].*fl[sel])/sum(w_iv[sel]))
    end
    bx,by
end
function tc_scan(flux,P)
    ph0=@. mod((t-t1)/P+0.5,1.0)-0.5; bx,by=binfold(ph0,flux); t1 + bx[argmin(by)]*P
end

function build_params(data, P_lit, T0)
    ic=InstrumentConfig(pm=pm_names); pri=Dict{String,PriorSpec}()
    pri["P_k1"]=NormalPrior(P_lit, 1e-4, P_lit-3e-3, P_lit+3e-3)
    pri["Tc_k1"]=NormalPrior(T0, 0.02, T0-0.12, T0+0.12)
    pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
    for n in pm_names; pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
        pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
    pri["rho_s"]=NormalPrior(0.71,0.3,0.05,8.0)
    Params(;max_kplanet=1,planet_modes=[PM_ONLY],instruments=ic,data=data,M_s=1.04,R_s=1.137,
        parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,stability=:none)
end

function fit_subtract(flux, P_lit, name; stride=8)
    T0 = tc_scan(flux, P_lit)
    idx = 1:stride:length(t)
    data_d=Data(; t_phot=t[idx], flux=flux[idx], flux_err=e[idx], phot_inst=inst[idx])
    res = sample_map(NereusTarget(build_params(data_d,P_lit,T0), data_d; unconstrained=true); n_starts=3, maxiter=2500)
    data_f=Data(; t_phot=t, flux=flux, flux_err=e, phot_inst=inst)
    params_f=build_params(data_f,P_lit,T0); θ=Theta(params_f)
    for (nm,v) in zip(res.param_names, res.x_map); set_param!(θ,nm,v); end
    blk=params_f.layout.planet_blocks[1]
    @printf("  fit %s: P=%.6f Tc=%.4f rr=%.4f b=%.3f conv=%s railed=%s\n",
            name, planet_P(θ,1), θ.values[blk.t], θ.values[blk.r], θ.values[blk.b],
            res.converged, res.railed)
    preds,_=phot_predictions(θ, data_f)
    flux .- preds .+ 1.0
end

periods = exp.(range(log(0.5), log(15.0); length=2500))
function bls_spectrum(flux)
    w=1.0 ./ e.^2; W=sum(w); fm=sum(w.*flux)/W; wf=w.*(flux.-fm)
    nb=200; bw=zeros(nb); bwf=zeros(nb); durs=[0.01,0.02,0.04,0.08]; snr=similar(periods)
    @inbounds for i in eachindex(periods); snr[i],_,_,_=Nereus._bls_one_period(t,w,wf,W,periods[i],durs,nb,bw,bwf); end
    snr
end
snr_at(s,P0)= s[argmin(abs.(periods .- P0))]

Pb,Pd,Pe = 4.1591287, 9.030672, 0.789593
s0=bls_spectrum(f)
rb=fit_subtract(f,Pb,"b");   s1=bls_spectrum(rb)
rbd=fit_subtract(rb,Pd,"d"); s2=bls_spectrum(rbd)
rbde=fit_subtract(rbd,Pe,"e"); s3=bls_spectrum(rbde)

stages=[("raw",s0),("b subtracted",s1),("b+d subtracted",s2),("b+d+e subtracted",s3)]
fig=Figure(size=(1200,1100))
for (i,(lab,s)) in enumerate(stages)
    ax=Axis(fig[i,1]; xlabel = i==4 ? "period [d]" : "", ylabel="BLS SNR", xscale=log10, xgridvisible=false, ygridvisible=false)
    lines!(ax, periods, s; color=:black, linewidth=0.8)
    for (nm,P0,col) in [("b",Pb,:red),("d",Pd,:green),("e",Pe,:blue)]
        vlines!(ax,[P0]; color=(col,0.6), linewidth=1.2, linestyle=:dash); text!(ax,P0,maximum(s); text=nm, color=col, fontsize=12, align=(:center,:top)); end
    @printf(">> %-18s SNR@b=%.1f  SNR@d=%.1f  SNR@e=%.1f  (top P=%.3f SNR=%.1f)\n",
            lab, snr_at(s,Pb), snr_at(s,Pd), snr_at(s,Pe), periods[argmax(s)], maximum(s))
    text!(ax, 0.01, 0.98; text=lab, space=:relative, align=(:left,:top), fontsize=14)
end
out=joinpath(@__DIR__,"WASP47_iterative_bls.png"); save(out,fig; px_per_unit=2); println("saved $out")
