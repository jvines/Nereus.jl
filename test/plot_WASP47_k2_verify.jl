#!/usr/bin/env julia
# Final K2 detrend pipeline + verification:
#   fit b -> subtract (continuous, no mask gaps) -> GP -> add b back ->
#   RENORMALIZE to OOT baseline = 1.0 (standard final step).
# Then phase-fold b/d/e to confirm the GP did NOT destroy d/e (depths must
# survive at ~b 12000 / d ~800 / e ~200 ppm).

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

# fit b, subtract
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

# GP on continuous b-subtracted flux, add b back
jit = 1.4826*median(abs.((resid_b.-runmed(resid_b,49)) .- median(resid_b.-runmed(resid_b,49))))
gpp=Dict{String,PriorSpec}("gp_log_omega0_phot"=>UniformPrior(-3.0,0.3),"gp_log_Q_phot"=>UniformPrior(-1.0,0.5))
fc = detrend_gp(t,resid_b,sqrt.(e.^2 .+ jit^2),CeleriteSHO(); sector_id=ones(Int,length(t)), joint_segments=false, gp_priors=gpp).flux_detrended
fc = fc .+ (bmodel .- 1.0)

# RENORMALIZE (standard): divide by the OOT baseline so OOT median = 1.0
allmask = mask_transits(t,[Pb],[T0b];window=0.05) .| mask_transits(t,[Pd],[tcscan(fc,Pd)];window=0.04) .| mask_transits(t,[Pe],[tcscan(fc,Pe)];window=0.07)
base = median(fc[.!allmask]); fc ./= base
@printf("renormalized by OOT baseline %.6f (offset %.0fppm removed)\n", base, (base-1)*1e6)

# verify all three transits survive
wn=1.0 ./ e.^2
fig=Figure(size=(1100,1100))
for (i,(nm,P,win,exp_ppm)) in enumerate([("b",Pb,0.06,12000),("d",Pd,0.06,800),("e",Pe,0.18,200)])
    T0=tcscan(fc,P); ph=@. mod((t-T0)/P+0.5,1.0)-0.5; sel=abs.(ph).<win
    ax=Axis(fig[i,1]; xlabel=i==3 ? "phase" : "", ylabel="renorm flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, ph[sel], fc[sel]; markersize=4, color=(:steelblue,0.5))
    bx,by=binfold(fc[sel], ph[sel], wn[sel]); scatter!(ax,bx,by; markersize=9, color=:red)
    lo=minimum(by); hi=maximum(by); pad=0.25*(hi-lo)+5e-5; ylims!(ax,lo-pad,hi+pad)
    dep=round(Int,(1.0-lo)*1e6)
    @printf(">> %s fold depth=%dppm (expected ~%d) — %s\n", nm, dep, exp_ppm, dep>exp_ppm*0.4 ? "PRESERVED" : "DESTROYED?")
    text!(ax,0.01,0.97; text="$nm  P=$(P)d  depth≈$(dep)ppm (exp ~$(exp_ppm))", space=:relative, align=(:left,:top), fontsize=14)
end
save(joinpath(@__DIR__,"WASP47_k2_verify.png"), fig; px_per_unit=2); println("saved WASP47_k2_verify.png")
