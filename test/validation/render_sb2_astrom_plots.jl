#!/usr/bin/env julia
# Render astrometry plots for a BINARY (RV+AS+SB) fit: sky-plane orbit + the
# RV-astrometry joint phase-fold. Uses a self-consistent HGCA catalog.
using Nereus, Statistics, Random
using Nereus: param_index, Theta, star_reflex_pm, star_reflex_offset
const P=Nereus
const P_BIN,TC_BIN,E_BIN,W_BIN=2100.0,55000.0,0.30,0.9
const K_A,K_B,INC_BIN,OM_BIN,F_LIGHT,M_A,PLX,GAMMA,JIT=4200.0,6100.0,1.05,2.1,0.12,0.95,25.0,120.0,5.0
Random.seed!(21)
nep=40; tep=sort!(TC_BIN .+ 1600.0 .* rand(nep) .- 800.0)
t_rv=vcat(tep,tep); rv_comp=vcat(fill(1,nep),fill(2,nep)); rv_err=fill(8.0,2nep)
ep=P.mjd_epochs((1991.25,2004.5,2016.0))
hgca0=HGCAData(; epochs=ep,pmra=(0.,0.,0.),pmdec=(0.,0.,0.),sigma_pmra=(0.3,0.1,0.05),
                 sigma_pmdec=(0.3,0.1,0.05),plx=PLX,plx_err=0.05,hip_id=999999)
data0=Data(; t_rv=t_rv,rv=zeros(2nep),rv_err=rv_err,rv_inst=ones(Int,2nep),rv_comp=rv_comp,hgca=hgca0)
ic=InstrumentConfig(rv=["SB2SPEC"]); par=ParametrizationConfig(time=:Tc)
pri=Dict{String,PriorSpec}("n_p"=>FixedPrior(1.0),
    "P_k1"=>NormalPrior(P_BIN,20.0,P_BIN-200,P_BIN+200),"K_A_k1"=>UniformPrior(1000.,8000.),
    "K_B_k1"=>UniformPrior(1000.,10000.),"Tc_k1"=>NormalPrior(TC_BIN,50.,TC_BIN-400,TC_BIN+400),
    "sesinw_k1"=>UniformPrior(-0.9,0.9),"secosw_k1"=>UniformPrior(-0.9,0.9),
    "inc_k1"=>UniformPrior(0.0,π),"Omega_k1"=>UniformPrior(0.0,2π),
    "gamma_SB2SPEC"=>UniformPrior(-5000.,5000.),"sigma_SB2SPEC"=>LogUniformPrior(0.1,50.),
    "M_pri"=>FixedPrior(M_A),"plx"=>NormalPrior(PLX,0.05,PLX-1,PLX+1),
    "f_light"=>NormalPrior(F_LIGHT,0.02,0.0,0.5))
params=Params(; max_kplanet=1,planet_modes=[BINARY],instruments=ic,data=data0,parametrization=par,
                priors=pri,stability=:none,M_s=M_A,R_s=1.0)
sesinw(e,w)=sqrt(e)*sin(w); secosw(e,w)=sqrt(e)*cos(w)
th=Theta(params); setp!(n,v)=(th.values[param_index(params,n)]=v)
for (n,v) in (("P_k1",P_BIN),("K_A_k1",K_A),("K_B_k1",K_B),("Tc_k1",TC_BIN),
    ("sesinw_k1",sesinw(E_BIN,W_BIN)),("secosw_k1",secosw(E_BIN,W_BIN)),("inc_k1",INC_BIN),
    ("Omega_k1",OM_BIN),("gamma_SB2SPEC",GAMMA),("sigma_SB2SPEC",JIT),("M_pri",M_A),
    ("plx",PLX),("f_light",F_LIGHT)); setp!(n,v); end
orb,Msec_eff=P._planet_orbit(th,1,P.astrom_M_pri(th),P.astrom_plx(th),data0.t_ref)
function hgca_model(orb,ep,M)
    μh=star_reflex_pm(orb,ep[1],M); μg=star_reflex_pm(orb,ep[3],M)
    oH=star_reflex_offset(orb,ep[1],M); oG=star_reflex_offset(orb,ep[3],M); dt=(ep[3]-ep[1])/365.25
    (μh,((oG[1]-oH[1])/dt,(oG[2]-oH[2])/dt),μg)
end
mh,mhg,mg=hgca_model(orb,ep,Msec_eff); σ=(0.3,0.1,0.05)
hgca=HGCAData(; epochs=ep,pmra=(mh[1]+σ[1]*randn(),mhg[1]+σ[2]*randn(),mg[1]+σ[3]*randn()),
                pmdec=(mh[2]+σ[1]*randn(),mhg[2]+σ[2]*randn(),mg[2]+σ[3]*randn()),
                sigma_pmra=σ,sigma_pmdec=σ,plx=PLX,plx_err=0.05,hip_id=999999)
kg(t,A)=A*(let e=E_BIN,w=W_BIN,Pp=P_BIN,Tc=TC_BIN; f_tc=π/2-w; E_tc=2*atan(sqrt((1-e)/(1+e))*tan(f_tc/2))
    Mt=E_tc-e*sin(E_tc); M=2π*(t-Tc)/Pp+Mt; E=M; for _ in 1:80; E-=(E-e*sin(E)-M)/(1-e*cos(E)); end
    ff=2*atan(sqrt((1+e)/(1-e))*tan(E/2)); cos(ff+w)+e*cos(w) end)
rv=[rv_comp[i]==1 ? GAMMA+kg(t_rv[i],K_A) : GAMMA-kg(t_rv[i],K_B) for i in eachindex(t_rv)] .+ JIT .* randn(2nep)
data=Data(; t_rv=t_rv,rv=rv,rv_err=rv_err,rv_inst=ones(Int,2nep),rv_comp=rv_comp,hgca=hgca)
tgt=NereusTarget(params,data)
res=sample_ptemcee(tgt,data; n_temps=8,n_walkers=48,n_steps=3000,n_burnin=1500,
                   seed=6,init_strategy=:prior,show_progress=false)
ch=res.chains
out=joinpath(@__DIR__,"..","..","results","sb2_astrom_plots"); mkpath(out)
println("rendering astrometry plots → $out")
for (nm,f) in (("orbit_skyplane",()->plot_orbit_skyplane(ch,params,data; planet_idx=1,output=out)),
               ("rv_astrom_phasefold",()->plot_rv_astrom_phasefold(ch,params,data; planet_idx=1,output=out)))
    try; f(); println("  ok: $nm"); catch e; println("  FAIL $nm: ", sprint(showerror,e)); end
end
println("files:"); for (r,_,fs) in walkdir(out), f in fs; println("  ",joinpath(r,f)); end
