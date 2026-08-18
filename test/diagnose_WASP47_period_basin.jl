#!/usr/bin/env julia
# WASP-47 period-basin verification. Hypothesis: the informed-birth period
# proposal σ_P=0.05 (5% in log-P) is orders of magnitude too wide for a
# transit to stay phase-aligned across the ~10-yr K2+TESS baseline (needs
# ~ transit_dur/baseline ~ 3e-5), so transit-sourced births land outside
# the photometry's razor-thin logL basin and get rejected -> sampler stuck
# at Np=0 / RV-ish aliases.
#
# This does a quick seeded JOINT fit to get a clean b transit solution,
# then scans the PHOTOMETRY logL across a fine period window at that fixed
# solution, and reports the basin half-width vs σ_P=0.05 and vs the BLS
# grid resolution.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const P_B = 4.1591287

pm_names=["K2","TESS42","TESS92"]
files=["WASP-47_k2_c03_everest_lc.csv","WASP-47_tess_s42_lc.csv","WASP-47_tess_s92_lc.csv"]
pt_t=Float64[]; pt_f=Float64[]; pt_e=Float64[]; pt_i=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(pt_t,lc.t); append!(pt_f,lc.flux)
    append!(pt_e,lc.flux_err); append!(pt_i,fill(ix,length(lc.t))); end
perm=sortperm(pt_t); pt_t=pt_t[perm]; pt_f=pt_f[perm]; pt_e=pt_e[perm]; pt_i=pt_i[perm]
baseline = pt_t[end]-pt_t[1]
@printf("photometry baseline = %.1f d  (%.1f periods of b)\n", baseline, baseline/P_B)

rr=readdlm(joinpath(DATADIR,"WASP47_RVs_combined.csv"),',',Any,'\n';header=true)
raw=rr[1]; hdr=vec(rr[2]); c(n)=findfirst(==(n),hdr); fcol=c("flag"); keep=trues(size(raw,1))
for i in 1:size(raw,1); f=raw[i,fcol]; f===missing&&continue
    String(strip(string(f))) in ("transit","transit_night","anomalous")&&(keep[i]=false); end
raw=raw[keep,:]; bjd=Float64.(raw[:,c("bjd")]); rv=Float64.(raw[:,c("rv")]); rve=Float64.(raw[:,c("rv_err")])
istr=String.(raw[:,c("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
rinst=[i2i[s] for s in istr]
data=Data(;t_rv=bjd,rv=rv,rv_err=rve,rv_inst=rinst,t_phot=pt_t,flux=pt_f,flux_err=pt_e,phot_inst=pt_i)
ic=InstrumentConfig(rv=inames,pm=pm_names)

pri=Dict{String,PriorSpec}()
pri["n_p"]=FixedPrior(1.0)
pri["P_k1"]=UniformPrior(4.1588,4.1595)          # ~3x the needle: decisive test
pri["K_k1"]=UniformPrior(0.0,250.0)
pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
pri["Tc_k1"]=UniformPrior(pt_t[1],pt_t[1]+P_B)
pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
    ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
for n in pm_names
    pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
pri["rho_s"]=NormalPrior(1.0,0.3,0.1,5.0)
params=Params(;max_kplanet=1,planet_modes=[RVPM],instruments=ic,data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,
    trend_order=0,stability=:none,external_priors=[ExternalPrior(:ecc,NormalPrior(0.0,0.3),true)])
target=NereusTarget(params,data;unconstrained=true)

println("quick seeded fit for a clean b solution ...")
res=sample_ptemcee(target,data;n_temps=8,n_walkers=100,n_steps=800,n_burnin=400,seed=7,show_progress=false)
ch=res.chains; gv(s)=median(vec(Array(ch[:,s,:])))
θ=Theta(params)
for s in ("P_k1","K_k1","sesinw_k1","secosw_k1","Tc_k1","b_k1","rr_k1","rho_s")
    set_param!(θ,s,gv(Symbol(s))); end
for n in pm_names, s in ("offset_","jitter_","q1_","q2_"); set_param!(θ,"$s$n",gv(Symbol("$s$n"))); end
for n in inames, s in ("gamma_","sigma_"); set_param!(θ,"$s$n",gv(Symbol("$s$n"))); end
P0=gv(:P_k1)
@printf("recovered b: P=%.6f  rr=%.4f  b=%.3f  Tc=%.4f\n", P0, gv(:rr_k1), gv(:b_k1), gv(:Tc_k1))

# ---- fine period scan of the PHOTOMETRY logL around P0 ----
L_at(P) = (set_param!(θ,"P_k1",P); transit_log_likelihood(θ,data))
Lpk = L_at(P0)
println("\nphotometry logL vs fractional period offset ΔP/P:")
@printf("  %-12s  %14s  %14s\n", "ΔP/P", "logL_phot", "Δ from peak")
for frac in [0.0, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.05]
    L = L_at(P0*(1+frac))
    @printf("  %-12.0e  %14.1f  %14.1f%s\n", frac, L, L-Lpk,
            abs(frac-0.05)<1e-9 ? "   <- typical σ_P birth proposal" : "")
end

# basin half-width: largest |ΔP/P| with logL within 10 of the peak
fr=10.0 .^ range(-6, -1.3; length=200); hw=0.0
for f in fr; (Lpk - L_at(P0*(1+f)) < 10) && (hw=f); end
need = 0.1/baseline   # transit-alignment tolerance ~ dur/baseline
@printf("\nbasin half-width (Δlog L<10): ΔP/P ~ %.1e\n", hw)
@printf("transit-alignment tol (dur/baseline): ΔP/P ~ %.1e\n", need)
@printf("informed-birth σ_P:                  ΔP/P ~ %.1e  (log-P space 0.05)\n", 0.05)
@printf("BLS grid resolution at P_b:           ΔP/P ~ %.1e\n",
        (log(30.0)-log(0.5))/2000)
@printf("\n>> σ_P is ~%.0f× wider than the basin. Fraction of births landing in-basin ~ %.1e\n",
        0.05/max(hw,1e-9), 2*hw/(0.05*sqrt(2π)))
