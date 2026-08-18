#!/usr/bin/env julia
# H1 vs H5: is e's gain strangled by SHARED transit-shape params (rho_s, q1/q2
# pinned by b+d), or is e genuinely marginal in K2-only?
# Conditional ladder from the locked b+d state — each case optimizes WITH e
# minus the SAME freedoms WITHOUT e (so shared-param slack doesn't overcount):
#   a: e params only            (what the sampler's birth effectively sees)
#   b: e + jitter               (current refinement freedom)
#   c: e + jitter + rho_s + q1/q2  (shape co-adaptation — H1's claim)
# H1 confirmed if Δ_c >> Δ_a,b (→ refinement must co-adapt shape params).
# H5 confirmed if Δ_c ≈ Δ_b ≈ +50-70 (→ e is honestly marginal in K2-only).

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
ws = Nereus.PTWorkspace(params, 3, 0; n_obs=length(data.t_rv), n_phot=length(data.t_phot))

function make_theta(; with_e::Bool)
    tds=TransDimState(; max_planets=3, n_noise=0)
    θ=Theta{Float64}(params; td=tds)
    set_param!(θ,"P_k1",4.159126); set_param!(θ,"Tc_k1",2456978.8266)
    set_param!(θ,"rr_k1",0.1088);  set_param!(θ,"b_k1",0.68)
    set_param!(θ,"P_k2",9.031728); set_param!(θ,"Tc_k2",2456979.2718)
    set_param!(θ,"rr_k2",0.0297);  set_param!(θ,"b_k2",0.46)
    set_param!(θ,"offset_K2",0.0); set_param!(θ,"jitter_K2",1.2e-4)
    set_param!(θ,"q1_K2",0.3); set_param!(θ,"q2_K2",0.3); set_param!(θ,"rho_s",0.71)
    activate_planet!(θ.td,1); activate_planet!(θ.td,2)
    if with_e
        set_param!(θ,"P_k3",0.789554); set_param!(θ,"Tc_k3",2456977.4029)
        set_param!(θ,"b_k3",0.10);     set_param!(θ,"rr_k3",0.0122)
        activate_planet!(θ.td,3)
    end
    θ
end
ll(θ) = Nereus.transit_log_likelihood(θ, data, ws)

# greedy adaptive hill-climb over a named free subset (multiplicative for jitter)
function climb!(θ, free::Vector{String}, sig0::Dict{String,Float64}; iters=6000, seed=1)
    rng=MersenneTwister(seed)
    best=ll(θ); vals=Dict(n=>get_param(θ,n) for n in free)
    sig=copy(sig0); accepted=0
    for it