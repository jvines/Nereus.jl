#!/usr/bin/env julia
# TESS-only folds with deeper planets SUBTRACTED (fast: direct transit-model
# subtraction via phot_predictions, NO sample_map — sample_map is a
# multistart optimizer that is hopelessly slow on 152k photometry points).
# Build each planet's Mandel-Agol model from lit P + data-anchored T0 +
# depth-derived rr, subtract it, fold the next planet on the residual.

using Nereus
using CairoMakie
using Statistics: quantile
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
pm_names = ["TESS42","TESS92"]
files = ["WASP-47_tess_s42_notch_w1p0.csv","WASP-47_tess_s92_notch_w1p0.csv"]
t=Float64[]; f=Float64[]; e=Float64[]; inst=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(t,lc.t); append!(f,lc.flux); append!(e,lc.flux_err); append!(inst,fill(ix,length(lc.t))); end
p=sortperm(t); t=t[p]; f=f[p]; e=e[p]; inst=inst[p]
t1=t[1]; w=1.0 ./ e.^2

function binfold(ph, fl, ww; nb=140)
    edges=range(-0.5,0.5;length=nb+1); bx=Float64[]; by=Float64[]
    for k in 1:nb
        sel=(ph .>= edges[k]).&(ph .< edges[k+1]); count(sel)<4 && continue
        push!(bx,(edges[k]+edges[k+1])/2); push!(by,sum(ww[sel].*fl[sel])/sum(ww[sel]))
    end
    bx,by
end
function fold_depth(flux,P)              # data-anchored T0 + binned-min depth
    ph0 = @. mod((t-t1)/P+0.5,1.0)-0.5
    bx,by = binfold(ph0,flux,w)
    T0 = t1 + bx[argmin(by)]*P
    T0, max(1.0 - minimum(by), 1e-5)
end

# one-planet PM-only Theta builder; returns the phot model at given params
function transit_model(flux, P, T0, rr)
    data=Data(; t_phot=t, flux=flux, flux_err=e, phot_inst=inst)
    ic=InstrumentConfig(pm=pm_names)
    pri=Dict{String,PriorSpec}()
    pri["P_k1"]=LogUniformPrior(0.5,30.0); pri["Tc_k1"]=UniformPrior(t1-15.0,t1+15.0)
    pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.001,0.30)
    for n in pm_names
        pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
        pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
    pri["rho_s"]=NormalPrior(0.71,0.2,0.1,5.0)
    params=Params(;max_kplanet=1,planet_modes=[PM_ONLY],instruments=ic,data=data,M_s=1.04,R_s=1.137,
        parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,stability=:none)
    θ=Theta(params)
    set_param!(θ,"P_k1",P); set_param!(θ,"Tc_k1",T0); set_param!(θ,"rr_k1",rr)
    set_param!(θ,"b_k1",0.2); set_param!(θ,"rho_s",0.71)
    for n in pm_names; set_param!(θ,"offset_$n",0.0); set_param!(θ,"jitter_$n",3e-4); set_param!(θ,"q1_$n",0.3); set_param!(θ,"q2_$n",0.3); end
    preds,_=phot_predictions(θ,data); preds
end

P = (b=4.1591287, d=9.030672, e=0.789593)
T0b,depb = fold_depth(f, P.b);  rrb=sqrt(depb)
@printf("b: T0=%.4f depth=%dppm rr=%.3f → subtracting\n", T0b, round(Int,depb*1e6), rrb)
resid_b = f .- transit_model(f, P.b, T0b, rrb) .+ 1.0
T0d,depd = fold_depth(resid_b, P.d); rrd=sqrt(depd)
@printf("d: T0=%.4f depth=%dppm rr=%.3f → subtracting\n", T0d, round(Int,depd*1e6), rrd)
resid_bd = resid_b .- transit_model(resid_b, P.d, T0d, rrd) .+ 1.0

panels=[("b (raw)",f,P.b,0.06),("d (b subtracted)",resid_b,P.d,0.06),("e (b+d subtracted)",resid_bd,P.e,0.18)]
fig=Figure(size=(1100,1100))
for (i,(lab,flux,Pp,win)) in enumerate(panels)
    T0,_=fold_depth(flux,Pp); ph=@. mod((t-T0)/Pp+0.5,1.0)-0.5; sel=abs.(ph).<win
    ax=Axis(fig[i,1]; xlabel=i==3 ? "phase" : "", ylabel="flux", xgridvisible=false, ygridvisible=false)
    scatter!(ax, ph[sel], flux[sel]; markersize=2, color=(:gray60,0.13), rasterize=2)
    bx,by=binfold(ph[sel],flux[sel],w[sel]); scatter!(ax,bx,by; markersize=7, color=:red)
    lo=minimum(by); hi=maximum(by); pad=0.25*(hi-lo)+5e-5; ylims!(ax,lo-pad,hi+pad)
    @printf(">> %-22s depth≈%dppm\n", lab, round(Int,(1.0-lo)*1e6))
    text!(ax,0.01,0.97; text="$lab  P=$(Pp)d  depth≈$(round(Int,(1.0-lo)*1e6))ppm", space=:relative, align=(:left,:top), fontsize=14)
end
out=joinpath(@__DIR__,"WASP47_tess_folds_subtracted.png"); save(out,fig; px_per_unit=2); println("saved $out")
