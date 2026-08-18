#!/usr/bin/env julia
# Proper "births'-eye" gate for d/e on the notch-detrended LCs: build a
# Theta with b ACTIVE at its known solution and call the SAME
# `_bls_peaks_filtered` the informed births use — it subtracts b's full
# Mandel-Agol model (not a crude mask) and harmonic-dedups b's aliases,
# then refines. If d (9.03) / e (0.79) survive as peaks here, the births
# can propose them and the full 4-slot run is worth launching.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
const Pd=9.030672; const Pe=0.789593; const Pb=4.159156

pm_names=["K2","TESS42","TESS92"]
files=["WASP-47_k2_c03_notch.csv","WASP-47_tess_s42_notch.csv","WASP-47_tess_s92_notch.csv"]
pt_t=Float64[]; pt_f=Float64[]; pt_e=Float64[]; pt_i=Int[]
for (ix,fn) in enumerate(files)
    lc=load_tess_lc(joinpath(DATADIR,fn)); append!(pt_t,lc.t); append!(pt_f,lc.flux)
    append!(pt_e,lc.flux_err); append!(pt_i,fill(ix,length(lc.t))); end
perm=sortperm(pt_t); pt_t=pt_t[perm]; pt_f=pt_f[perm]; pt_e=pt_e[perm]; pt_i=pt_i[perm]
# minimal RV (unused by the BLS path)
data=Data(;t_rv=[0.0,1.0,2.0],rv=[0.0,1.0,0.0],rv_err=[1.0,1.0,1.0],rv_inst=[1,1,1],
          t_phot=pt_t,flux=pt_f,flux_err=pt_e,phot_inst=pt_i)
ic=InstrumentConfig(rv=["X"],pm=pm_names)
pri=Dict{String,PriorSpec}()
pri["P_k1"]=LogUniformPrior(0.5,30.0); pri["K_k1"]=UniformPrior(0.0,200.0)
pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
pri["Tc_k1"]=UniformPrior(pt_t[1],pt_t[1]+30.0); pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
for k in 2:3
    pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["K_k$k"]=UniformPrior(0.0,200.0)
    pri["sesinw_k$k"]=UniformPrior(-0.7,0.7); pri["secosw_k$k"]=UniformPrior(-0.7,0.7)
    pri["Tc_k$k"]=UniformPrior(pt_t[1],pt_t[1]+30.0); pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
end
pri["gamma_X"]=UniformPrior(-10.0,10.0); pri["sigma_X"]=ModJeffreysPrior(0.1,50.0)
for n in pm_names
    pri["offset_$n"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_$n"]=LogUniformPrior(1e-5,5e-3)
    pri["q1_$n"]=UniformPrior(0.0,1.0); pri["q2_$n"]=UniformPrior(0.0,1.0); end
pri["rho_s"]=NormalPrior(0.71,0.2,0.1,5.0)
params=Params(;max_kplanet=3,planet_modes=[RVPM,RVPM,RVPM],instruments=ic,data=data,M_s=1.04,R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),priors=pri,trend_order=0,stability=:none)

# Theta with b ACTIVE at its known solution
td=TransDimState(max_planets=3); Nereus.activate_planet!(td,1)
th=Theta(params; td=td)
set_param!(th,"P_k1",Pb); set_param!(th,"Tc_k1",2456978.8126)
set_param!(th,"rr_k1",0.098); set_param!(th,"b_k1",0.293); set_param!(th,"K_k1",140.0)
set_param!(th,"sesinw_k1",0.0); set_param!(th,"secosw_k1",0.0); set_param!(th,"rho_s",0.71)
for n in pm_names; set_param!(th,"offset_$n",0.0); set_param!(th,"q1_$n",0.3); set_param!(th,"q2_$n",0.3); set_param!(th,"jitter_$n",3e-4); end

cache=Nereus.BLSCache()
P_ref,w,dep,t0,dur = Nereus._bls_peaks_filtered(th, data, 0.5, 30.0, cache; n_peaks=8)
println("=== _bls_peaks_filtered with b SUBTRACTED + harmonic-deduped (notch LCs) ===")
ord=sortperm(w;rev=true)
for j in ord
    tag = abs(P_ref[j]-Pd)/Pd<0.02 ? "  <== d" : abs(P_ref[j]-Pe)/Pe<0.02 ? "  <== e" : ""
    @printf("  P=%8.4f  w=%.3f  depth=%.5f  dur=%.4f%s\n", P_ref[j], w[j], dep[j], dur[j], tag)
end
for (nm,P0) in (("d",Pd),("e",Pe))
    hit=findfirst(j->abs(P_ref[j]-P0)/P0<0.02, eachindex(P_ref))
    @printf(">> %s (%.3f): %s\n", nm, P0, hit===nothing ? "NOT proposed" :
            "PROPOSABLE at P=$(round(P_ref[hit],digits=4)) w=$(round(w[hit],digits=3))")
end
