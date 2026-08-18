#!/usr/bin/env julia
# Clean d/e folds: subtract b (NO add-back -> no b contamination filling the
# dip), MEASURE the d/e period from THIS K2 data (BLS+refine -> no beats
# from a slightly-wrong literature period), fold on the measured P/T0.

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
function tcscan(flux,P)
    ph=@. mod((t-t1)/P+0.5,1.0)-0.5; bx,by=binfold(flux,ph,w0); t1+bx[argmin(by)]*P
end
function runmed(y,w); n=length(y); m=similar(y); h=w÷2; for i in 1:n; lo=max(1,i-h); hi=min(n,i+h); m[i]=median(@view y[lo:hi]); end; m; end

# fit b, subtract (NO add-back)
T0b=tcscan(f,Pb)
data=Data(; t_phot=t, flux=f, flux_err=e, phot_inst=ones(Int,length(t)))
pri=Dict{String,PriorSpec}()
pri["P_k1"]=NormalPrior(Pb,1e-4,Pb-3e-3,Pb+3e-3); pri["Tc_k1"]=NormalPrior(T0b,0.02,T0b-0.12,T0b+0.12)
pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(0.71,0.2,0.1,5.0)
params=Params(;max_kplanet=1,planet_modes=[PM_ONLY],instruments=InstrumentConfig(pm=["K2"]),data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,stability=:none)
res=sample_map(NereusTarget(params,data;unconstrained=true); n_starts=6, maxiter=3000)
θ=Theta(params); for (nm,v) in zip(res.param_names,res.x_map); set_param!(θ,nm,v); end
bmodel,_=phot_predictions(θ,data); resid_b = f .- bmodel .+ 1.0

# GP on continuous b-subtracted flux (no mask gaps), renormalize. NO add-back.
jit=1.4826*median(abs.((resid_b.-runmed(resid_b,49)) .- median(resid_b.-runmed(resid_b,49))))
gpp=Dict{String,PriorSpec}("gp_log_omega0_phot"=>UniformPrior(-3.0,0.3),"gp_log_Q_phot"=>UniformPrior(-1.0,0.5))
fc=detrend_gp(t,resid_b,sqrt.(e.^2 .+ jit^2),CeleriteSHO(); sector_id=ones(Int,length(t)), joint_segments=false, gp_priors=gpp).flux_detrended
fc ./= median(fc)   # renormalize

# MEASURE d/e period from THIS data (BLS + non-binned refine)
function measureP(flux, Plo, Phi)
    periods=exp.(range(log(Plo),log(Phi);length=3000))
    P0,dep,t0,snr,dur=Nereus._box_least_squares(t,flux,e,periods; n_phase_bins=200, n_peaks=3)
    isempty(P0) && return (NaN,NaN)
    j=argmax(snr); Pc=P0[j]; t0c=t0[j]; durc=dur[j]<=0 ? 0.04*Pc : dur[j]
    w=1.0 ./ e.^2; W=sum(w); fm=sum(w.*flux)/W; wf=w.*(flux.-fm)
    Pr,_,_,_,sn=Nereus._refine_bls_period(t,w,wf,W,Pc,t0c,durc)
    (Pr, sn)
end
Pd_m,snd=measureP(fc, 8.95, 9.11); Pe_m,sne=measureP(fc, 0.785, 0.795)
@printf("measured: Pd=%.6f (lit %.6f, Δ=%.1e)  Pe=%.6f (lit %.6f, Δ=%.1e)\n",
        Pd_m,Pd,(Pd_m-Pd)/Pd, Pe_m,Pe,(Pe_m-Pe)/Pe)

# PUBLISHED transit times (Vanderburg+2017, BJD_TDB — inside the K2 epoch);
# tcscan noise-locks for shallow d/e, so center the fold on the real transit.
const Tcd, Tce = 2456982.349, 2456983.178
wn=1.0 ./ e.^2
# Diagnostic: fold PRE-GP (b-subtracted only, bowl removed by long running median
# that can't follow a transit) vs POST-GP, to see if the GP eats d/e.
bowl = runmed(resid_b, 301); rb_flat = resid_b ./ bowl   # remove slow bowl, preserve narrow transits
fig=Figure(size=(1300,800))
for (i,(nm,P,Tc,win)) in enumerate([("d",Pd,Tcd,0.06),("e",Pe,Tce,0.18)])
    ph=@. mod((t-Tc)/P+0.5,1.0)-0.5; sel=abs.(ph).<win
    for (j,(lab,flux)) in enumerate([("pre-GP (runmed-flat)",rb_flat),("post-GP",fc)])
        ax=Axis(fig[i,j]; xlabel=i==2 ? "phase" : "", ylabel=j==1 ? "flux (b removed)" : "", xgridvisible=false, ygridvisible=false)
        scatter!(ax, ph[sel], flux[sel]; markersize=4, color=(:steelblue,0.5))
        bx,by=binfold(flux[sel], ph[sel], wn[sel]); scatter!(ax,bx,by; markersize=9, color=:red)
        lo=minimum(by); hi=maximum(by); pad=0.25*(hi-lo)+5e-5; ylims!(ax,lo-pad,hi+pad)
        depth0 = 1.0 - minimum(by[abs.(bx).<0.012])
        @printf(">> %s [%s] depth@phase0=%dppm\n", nm, lab, round(Int,depth0*1e6))
        text!(ax,0.01,0.97; text="$nm  $lab  depth@0≈$(round(Int,depth0*1e6))ppm", space=:relative, align=(:left,:top), fontsize=13)
    end
end
save(joinpath(@__DIR__,"WASP47_k2_de_folds.png"), fig; px_per_unit=2); println("saved WASP47_k2_de_folds.png")
