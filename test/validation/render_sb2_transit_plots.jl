#!/usr/bin/env julia
# Render transit plots for an SB2 + circumprimary TRANSITING planet fit.
# Confirms the binary (has_geometry=false) is invisible to the transit and the
# planet's diluted transit renders correctly.
using Nereus, Statistics, Random
using Nereus: param_index, Theta, compute_transit_model_on_grid

function kep_geom(t,P,Tc,e,w)
    f_tc=π/2-w; E_tc=2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2)); M_tc=E_tc-e*sin(E_tc)
    M=2π*(t-Tc)/P+M_tc; E=M; for _ in 1:80; E-=(E-e*sin(E)-M)/(1-e*cos(E)); end
    f=2*atan(sqrt((1+e)/(1-e))*tan(E/2)); cos(f+w)+e*cos(w)
end
const P_BIN,TC_BIN,E_BIN,W_BIN=41.3,5.7,0.15,0.5
const K_A,K_B=820.0,1240.0
const P_PL,TC_PL,RR_PL,B_PL=7.83,2.10,0.11,0.30
const Q1,Q2,DIL,GAMMA,JIT,M_S,R_S=0.40,0.30,0.35,15.0,2.0,1.0,1.0
Random.seed!(9)
nep=45; tep=sort!(300.0 .* rand(nep))
t_rv=vcat(tep,tep); rv_comp=vcat(fill(1,nep),fill(2,nep)); rv_err=fill(3.0,2nep)
cad=2.0/(60*24); t_phot=Float64[]
for n in 0:3; tc=TC_PL+n*P_PL; append!(t_phot, tc-0.28:cad:tc+0.28); end
append!(t_phot,0.0:0.5:32.0); sort!(unique!(t_phot)); nph=length(t_phot); flux_err=fill(3e-4,nph)
data0=Data(; t_rv=t_rv,rv=zeros(2nep),rv_err=rv_err,rv_inst=ones(Int,2nep),rv_comp=rv_comp,
             t_phot=t_phot,flux=ones(nph),flux_err=flux_err,phot_inst=ones(Int,nph))
ic=InstrumentConfig(rv=["SB2SPEC"],pm=["TESS"]); par=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=false)
pri=Dict{String,PriorSpec}("n_p"=>FixedPrior(2.0),
    "P_k1"=>NormalPrior(P_BIN,0.05,P_BIN-0.5,P_BIN+0.5),"K_A_k1"=>UniformPrior(300.0,1500.0),
    "K_B_k1"=>UniformPrior(300.0,2200.0),"Tc_k1"=>NormalPrior(TC_BIN,0.3,TC_BIN-2,TC_BIN+2),
    "sesinw_k1"=>UniformPrior(-0.7,0.7),"secosw_k1"=>UniformPrior(-0.7,0.7),
    "P_k2"=>NormalPrior(P_PL,0.005,P_PL-0.05,P_PL+0.05),"K_k2"=>UniformPrior(0.0,80.0),
    "Tc_k2"=>NormalPrior(TC_PL,0.05,TC_PL-0.3,TC_PL+0.3),"sesinw_k2"=>UniformPrior(-0.3,0.3),
    "secosw_k2"=>UniformPrior(-0.3,0.3),"rr_k2"=>UniformPrior(0.02,0.25),"b_k2"=>UniformPrior(0.0,0.95),
    "q1_TESS"=>UniformPrior(0.0,1.0),"q2_TESS"=>UniformPrior(0.0,1.0),
    "gamma_SB2SPEC"=>UniformPrior(-200.0,200.0),"sigma_SB2SPEC"=>LogUniformPrior(0.1,20.0),
    "offset_TESS"=>NormalPrior(0.0,0.005,-1.0,1.0),"jitter_TESS"=>ModJeffreysPrior(1e-5,1e-2))
params=Params(; max_kplanet=2,planet_modes=[BINARY_RV,RVPM],instruments=ic,data=data0,
                parametrization=par,priors=pri,stability=:none,M_s=M_S,R_s=R_S)
# truth theta → synthetic diluted flux + RVs
sesinw(e,w)=sqrt(e)*sin(w); secosw(e,w)=sqrt(e)*cos(w)
th=Theta(params); setp!(n,v)=(th.values[param_index(params,n)]=v)
for (n,v) in (("P_k1",P_BIN),("K_A_k1",K_A),("K_B_k1",K_B),("Tc_k1",TC_BIN),
    ("sesinw_k1",sesinw(E_BIN,W_BIN)),("secosw_k1",secosw(E_BIN,W_BIN)),("P_k2",P_PL),
    ("K_k2",24.0),("Tc_k2",TC_PL),("sesinw_k2",0.0),("secosw_k2",0.0),("rr_k2",RR_PL),
    ("b_k2",B_PL),("q1_TESS",Q1),("q2_TESS",Q2),("gamma_SB2SPEC",GAMMA),
    ("sigma_SB2SPEC",JIT),("offset_TESS",0.0),("jitter_TESS",1e-4),("dilution_TESS",DIL))
    setp!(n,v)
end
flux=compute_transit_model_on_grid(th,data0,t_phot,1) .+ flux_err .* randn(nph)
rv=[rv_comp[i]==1 ? GAMMA+K_A*kep_geom(t_rv[i],P_BIN,TC_BIN,E_BIN,W_BIN)+24.0*kep_geom(t_rv[i],P_PL,TC_PL,0.0,0.0) :
                    GAMMA-K_B*kep_geom(t_rv[i],P_BIN,TC_BIN,E_BIN,W_BIN) for i in eachindex(t_rv)] .+ JIT .* randn(2nep)
data=Data(; t_rv=t_rv,rv=rv,rv_err=rv_err,rv_inst=ones(Int,2nep),rv_comp=rv_comp,
            t_phot=t_phot,flux=flux,flux_err=flux_err,phot_inst=ones(Int,nph))
tgt=NereusTarget(params,data)
res=sample_ptemcee(tgt,data; n_temps=8,n_walkers=48,n_steps=3000,n_burnin=1500,
                   seed=4,init_strategy=:prior,show_progress=false)
ch=res.chains
out=joinpath(@__DIR__,"..","..","results","sb2_transit_plots"); mkpath(out)
println("rendering transit plots → $out")
plot_pm_timeseries(ch,params,data; output=out)
plot_pm_phasefold(ch,params,data; planet=2, output=out)   # transiting circumprimary planet
println("done:"); for (r,_,fs) in walkdir(out), f in fs; println("  ",joinpath(r,f)); end
