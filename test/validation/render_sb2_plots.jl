#!/usr/bin/env julia
# Render the RV plot surface for an SB2 fit (binary + circumprimary planet)
# so the component handling can be eyeballed. Short fit — plots only.
using Nereus, Statistics, Random
using Nereus: param_index, Theta

function kep_geom(t, P, Tc, e, w)
    f_tc=π/2-w; E_tc=2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2)); M_tc=E_tc-e*sin(E_tc)
    M=2π*(t-Tc)/P+M_tc; E=M; for _ in 1:80; E-=(E-e*sin(E)-M)/(1-e*cos(E)); end
    f=2*atan(sqrt((1+e)/(1-e))*tan(E/2)); cos(f+w)+e*cos(w)
end

const P_BIN,TC_BIN,E_BIN,W_BIN = 41.3,5.7,0.15,0.5
const K_A,K_B = 820.0,1240.0
const P_PL,TC_PL,E_PL,W_PL,K_PL = 7.83,2.1,0.10,1.0,24.0
const GAMMA,JIT = 15.0,2.0

Random.seed!(7)
nep=55; tep=sort!(300.0 .* rand(nep))
t_rv=vcat(tep,tep); rv_comp=vcat(fill(1,nep),fill(2,nep)); rv_err=fill(3.0,2nep)
rv=similar(t_rv)
for i in eachindex(t_rv)
    t=t_rv[i]
    rv[i]= rv_comp[i]==1 ? GAMMA+K_A*kep_geom(t,P_BIN,TC_BIN,E_BIN,W_BIN)+K_PL*kep_geom(t,P_PL,TC_PL,E_PL,W_PL) :
                            GAMMA-K_B*kep_geom(t,P_BIN,TC_BIN,E_BIN,W_BIN)
end
rv .+= JIT .* randn(2nep)
data=Data(; t_rv=t_rv, rv=rv, rv_err=rv_err, rv_inst=ones(Int,2nep), rv_comp=rv_comp)
ic=InstrumentConfig(rv=["SB2SPEC"]); par=ParametrizationConfig(time=:Tc)
pri=Dict{String,PriorSpec}(
    "n_p"=>FixedPrior(2.0),
    "P_k1"=>NormalPrior(P_BIN,0.05,P_BIN-0.5,P_BIN+0.5),
    "K_A_k1"=>UniformPrior(300.0,1500.0),"K_B_k1"=>UniformPrior(300.0,2200.0),
    "Tc_k1"=>NormalPrior(TC_BIN,0.3,TC_BIN-2,TC_BIN+2),
    "sesinw_k1"=>UniformPrior(-0.7,0.7),"secosw_k1"=>UniformPrior(-0.7,0.7),
    "P_k2"=>NormalPrior(P_PL,0.02,P_PL-0.2,P_PL+0.2),"K_k2"=>UniformPrior(0.0,80.0),
    "Tc_k2"=>NormalPrior(TC_PL,0.2,TC_PL-1,TC_PL+1),
    "sesinw_k2"=>UniformPrior(-0.5,0.5),"secosw_k2"=>UniformPrior(-0.5,0.5),
    "gamma_SB2SPEC"=>UniformPrior(-200.0,200.0),"sigma_SB2SPEC"=>LogUniformPrior(0.1,20.0))
params=Params(; max_kplanet=2, planet_modes=[BINARY_RV,RV_ONLY], instruments=ic,
                data=data, parametrization=par, priors=pri, stability=:none)
tgt=NereusTarget(params,data)
res=sample_ptemcee(tgt,data; n_temps=8,n_walkers=48,n_steps=3500,n_burnin=1800,
                   seed=5,init_strategy=:prior,show_progress=false)
ch=res.chains

out=joinpath(@__DIR__,"..","..","results","sb2_plots"); mkpath(out)
println("rendering SB2 RV plots → $out")
Nereus.plot_rv_sb2_timeseries(ch,params,data; output=out)             # component-colored timeseries
plot_rv_phasefold(ch,params,data; planet=1, output=out)               # binary → double-lined (delegates)
plot_rv_phasefold(ch,params,data; planet=2, output=joinpath(out,"planet"))  # planet → primary-only
println("done. files:")
for (r,_,fs) in walkdir(out), f in fs; println("  ", joinpath(r,f)); end
