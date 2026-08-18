#!/usr/bin/env julia
# Why does e NEVER get born? Put a walker at the fitted b+d state (the modal
# state of the trans-dim runs), then:
#  1. call _bls_peaks_filtered exactly as the sampler's cache update does —
#     is e (P=0.7896) in the peak list? with what weight/σ?
#  2. draw 500 informed births from that state — where do proposed P land,
#     and what realized ΔlogL do the e-period proposals carry?

using Nereus
using Statistics: median
using Random
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
t=lc.t; f=lc.flux; e=lc.flux_err
runmed(y,w)=(n=length(y);m=similar(y);h=w÷2;for i in 1:n;lo=max(1,i-h);hi=min(n,i+h);m[i]=median(@view y[lo:hi]);end;m)
flat=f./runmed(f,151)

data=Data(; t_phot=t, flux=flat, flux_err=e, phot_inst=ones(Int,length(t)))
pri=Dict{String,PriorSpec}()
for k in 1:3
    pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["Tc_k$k"]=UniformPrior(t[1],t[1]+30.0)
    pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
end
pri["offset_K2"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_K2"]=LogUniformPrior(1e-5,5e-3)
pri["q1_K2"]=UniformPrior(0.0,1.0); pri["q2_K2"]=UniformPrior(0.0,1.0)
pri["rho_s"]=NormalPrior(0.71,0.3,0.1,5.0)
params=Params(;max_kplanet=3,planet_modes=[PM_ONLY,PM_ONLY,PM_ONLY],instruments=InstrumentConfig(pm=["K2"]),
    data=data,M_s=1.04,R_s=1.137,parametrization=ParametrizationConfig(time=:Tc,geom=:b_rr,use_rho_s=true),
    priors=pri,stability=:none)

# walker at the fitted b+d state (slots 1,2 active; slot 3 inactive)
tds=TransDimState(; max_planets=3, n_noise=0)
θ=Theta{Float64}(params; td=tds)
set_param!(θ,"P_k1",4.159126); set_param!(θ,"Tc_k1",2456978.8266)
set_param!(θ,"rr_k1",0.1088);  set_param!(θ,"b_k1",0.68)
set_param!(θ,"P_k2",9.031728); set_param!(θ,"Tc_k2",2456979.2718)
set_param!(θ,"rr_k2",0.0297);  set_param!(θ,"b_k2",0.46)
set_param!(θ,"offset_K2",0.0); set_param!(θ,"jitter_K2",1.2e-4)
set_param!(θ,"q1_K2",0.3); set_param!(θ,"q2_K2",0.3); set_param!(θ,"rho_s",0.71)
activate_planet!(θ.td,1); activate_planet!(θ.td,2)

# --- 1. the peak list the sampler would cache at this state ---
cache=Nereus.BLSCache()
Pk,wk,dk,t0k,durk = Nereus._bls_peaks_filtered(θ, data, 0.5, 30.0, cache)
println("== peak list at b+d state ==")
for i in eachindex(Pk)
    near = abs(Pk[i]-0.789554)/0.789554 < 0.01 ? "  <-- e!" :
           abs(Pk[i]-4.159)/4.159 < 0.01 ? "  (b resid)" :
           abs(Pk[i]-9.031)/9.031 < 0.01 ? "  (d resid)" : ""
    @printf("  P=%9.5f  w=%.3f  depth=%6.0fppm  dur=%.4f%s\n", Pk[i], wk[i], dk[i]*1e6, durk[i], near)
end

# --- 2. 500 informed birth draws from this state ---
ws = Nereus.PTWorkspace(params, 3, 0; n_obs=length(data.t_rv), n_phot=length(data.t_phot))
ll0 = Nereus.rv_log_likelihood(θ,data,ws) + Nereus.transit_log_likelihood(θ,data,ws)
@printf("\nlogL(b+d) = %.1f\n", ll0)
rng=MersenneTwister(7)
n_e=0; n_tot=0; dll_e=Float64[]; other=Dict{Float64,Int}()
for i in 1:500
    nt,lq = Nereus.propose_planet_birth(θ, rng, Nereus.JointInformedBirth(); data=data)
    isfinite(lq) || continue
    global n_tot += 1
    Pp = planet_P(nt,3)
    if abs(Pp-0.789554)/0.789554 < 0.01
        global n_e += 1
        ll = Nereus.rv_log_likelihood(nt,data,ws) + Nereus.transit_log_likelihood(nt,data,ws)
        push!(dll_e, ll-ll0)
    else
        key=round(Pp,digits=2); other[key]=get(other,key,0)+1
    end
end
@printf("\n== %d valid births: %d (%.0f%%) at e's period ==\n", n_tot, n_e, 100n_e/max(n_tot,1))
if !isempty(dll_e)
    @printf("e-proposal ΔlogL: median=%.0f  max=%.0f  frac>+50: %.2f\n",
            median(dll_e), maximum(dll_e), count(>(50.0), dll_e)/length(dll_e))
end
top=sort(collect(other); by=x->-x[2])[1:min(8,length(other))]
println("other proposed periods: ", join(["$(k)d×$(v)" for (k,v) in top], "  "))
