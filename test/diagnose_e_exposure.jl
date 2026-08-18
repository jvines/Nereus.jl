#!/usr/bin/env julia
# Is the +65 e-ceiling caused by the MISSING finite-exposure integration?
# e is a USP (T14≈63min) on 30-min-cadence K2: the data dip is smeared, but
# Nereus evaluates the Mandel-Agol model at cadence MIDPOINTS (sharp/deep).
# Compare e's photometric logL gain at its TRUE params, modeled two ways:
#   midpoint  — current production behaviour
#   supersampled — average the model over the 30-min exposure (Kipping 2010)
# If supersampled >> midpoint, the model can't represent e at long cadence
# and exposure integration is the fix (not a sampler change).

using Nereus
using Statistics: median
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
flat=f./runmed(f,151)
@printf("median cadence = %.1f min\n", median(diff(t))*1440)

data=Data(; t_phot=t, flux=flat, flux_err=e, phot_inst=ones(Int,length(t)))
pri=Dict{String,PriorSpec}()
for k in 1:3
    pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["Tc_k$k"]=UniformPrior(t[1],t[1]+30.0)
    pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
end
pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(0.71,0.3,0.1,5.0)
params=Params(;max_kplanet=3,planet_modes=[PM_ONLY,PM_ONLY,PM_ONLY],instruments=InstrumentConfig(pm=["K2"]),
    data=data,M_s=1.04,R_s=1.137,parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),
    priors=pri,stability=:none)

tds=TransDimState(; max_planets=3, n_noise=0)
θ=Theta{Float64}(params; td=tds)
set_param!(θ,"P_k1",4.159126); set_param!(θ,"Tc_k1",2456978.8266); set_param!(θ,"rr_k1",0.1088); set_param!(θ,"b_k1",0.68)
set_param!(θ,"P_k2",9.031728); set_param!(θ,"Tc_k2",2456979.2718); set_param!(θ,"rr_k2",0.0297); set_param!(θ,"b_k2",0.46)
set_param!(θ,"P_k3",0.789554); set_param!(θ,"Tc_k3",2456977.4029); set_param!(θ,"rr_k3",0.0122); set_param!(θ,"b_k3",0.1)
set_param!(θ,"offset_K2",0.0); set_param!(θ,"q1_K2",0.3); set_param!(θ,"q2_K2",0.3); set_param!(θ,"rho_s",0.71)
activate_planet!(θ.td,1); activate_planet!(θ.td,2)

jit=1.2e-4; σ2=e.^2 .+ jit^2
phot_logL(m)= -0.5*sum((flat .- m).^2 ./ σ2)

# b+d model (e inactive)
m_bd = compute_transit_model_on_grid(θ, data, t, 1)
L_bd = phot_logL(m_bd)

# activate e, build full model midpoint vs supersampled
activate_planet!(θ.td,3)
m_mid = compute_transit_model_on_grid(θ, data, t, 1)

texp = 29.4/1440                       # K2 long cadence [d]
for nss in (5,7,11)
    offs = range(-texp/2, texp/2; length=nss)
    m_ss = zeros(length(t))
    for o in offs; m_ss .+= compute_transit_model_on_grid(θ, data, t .+ o, 1); end
    m_ss ./= nss
    @printf("nss=%2d  ΔlogL(e) midpoint=%+.1f   supersampled=%+.1f\n",
            nss, phot_logL(m_mid)-L_bd, phot_logL(m_ss)-L_bd)
end

# folded depth check: observed vs model
ph=@. mod((t-2456977.4029)/0.789554+0.5,1.0)-0.5
intr=abs.(ph).<0.022
@printf("\ne folded: observed in-transit median dip = %d ppm  (model midpoint depth = %d ppm)\n",
        round(Int,(median(flat[.!intr .& (abs.(ph).<0.1)])-median(flat[intr]))*1e6),
        round(Int,(1-minimum(m_mid))*1e6))
