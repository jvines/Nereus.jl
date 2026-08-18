#!/usr/bin/env julia
# H1 (shared-shape coupling) vs H5 (e genuinely marginal in K2):
# From the locked b+d state, optimize e's conditional ΔlogL three ways:
#   (a) e's own params only           — what the sampler's refinement does now
#   (b) e + jitter                    — current jitter-coupled refinement
#   (c) e + jitter + rho_s + q1/q2    — co-adapt the SHARED transit-shape params
# If (c) >> (a)/(b) and approaches the ~+210 sequential value, e's gain is
# strangled by rho_s/q1/q2 being pinned by b+d → H1, fixable. If (c) ~ (b),
# e is honestly marginal conditional on the shared stellar model → H5.

using Nereus
using Statistics: median
using Optim
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
ws = Nereus.PTWorkspace(params, 3, 0; n_obs=0, n_phot=length(t))

# locked b+d state (from the trans-dim runs / sequential fits)
function base_state()
    tds=TransDimState(; max_planets=3, n_noise=0)
    θ=Theta{Float64}(params; td=tds)
    set_param!(θ,"P_k1",4.159126); set_param!(θ,"Tc_k1",2456978.8266); set_param!(θ,"rr_k1",0.1088); set_param!(θ,"b_k1",0.68)
    set_param!(θ,"P_k2",9.031728); set_param!(θ,"Tc_k2",2456979.2718); set_param!(θ,"rr_k2",0.0297); set_param!(θ,"b_k2",0.46)
    set_param!(θ,"offset_K2",0.0); set_param!(θ,"jitter_K2",1.2e-4)
    set_param!(θ,"q1_K2",0.3); set_param!(θ,"q2_K2",0.3); set_param!(θ,"rho_s",0.71)
    activate_planet!(θ.td,1); activate_planet!(θ.td,2)
    θ
end
ll(θ)=Nereus.transit_log_likelihood(θ,data,ws)
θ0=base_state(); ll_bd=ll(θ0)
@printf("logL(b+d, jitter=120ppm) = %.1f\n", ll_bd)

# add e at its known ephemeris (slot 3)
function with_e(θ)
    set_param!(θ,"P_k3",0.789554); set_param!(θ,"Tc_k3",2456977.4029)
    set_param!(θ,"rr_k3",0.0122);  set_param!(θ,"b_k3",0.1)
    activate_planet!(θ.td,3); θ
end

# optimize a named subset, return best ΔlogL vs b+d (penalized by prior)
function opt_subset(free::Vector{String}; n_restart=4)
    best=-Inf
    for r in 1:n_restart
        θ=with_e(base_state())
        # jitter restart helps (c) find the low-jitter basin
        r>1 && set_param!(θ,"jitter_K2", [1.2e-4,5e-5,3e-4,8e-5][r])
        x0=[get_param(θ,nm) for nm in free]
        lo=[bounds(pri[_pk(nm)])[1] for nm in free]
        hi=[bounds(pri[_pk(nm)])[2] for nm in free]
        negobj=function(x)
            for (i,nm) in enumerate(free); set_param!(θ,nm,clamp(x[i],lo[i],hi[i])); end
            v=ll(θ); isfinite(v) ? -v : 1e12
        end
        res=optimize(negobj, x0, NelderMead(), Optim.Options(iterations=4000))
        best=max(best, -Optim.minimum(res))
    end
    best - ll_bd
end
# prior-key helper: strip instrument/planet suffix already matches dict keys
_pk(nm)=nm

println("\n== e conditional ΔlogL from locked b+d (true sequential value ≈ +210) ==")
da = opt_subset(["P_k3","Tc_k3","rr_k3","b_k3"])
@printf("(a) e params only          : ΔlogL = %+.1f\n", da)
db = opt_subset(["P_k3","Tc_k3","rr_k3","b_k3","jitter_K2"])
@printf("(b) e + jitter             : ΔlogL = %+.1f\n", db)
dc = opt_subset(["P_k3","Tc_k3","rr_k3","b_k3","jitter_K2","rho_s","q1_K2","q2_K2"])
@printf("(c) e + jitter + rho_s+LD  : ΔlogL = %+.1f\n", dc)
println()
if dc > 2*max(da,db,1.0) && dc > 100
    println(">> H1 CONFIRMED: e's gain is strangled by shared rho_s/q1/q2 pinned by b+d.")
    println("   Fix: refinement must co-adapt the shared transit-shape params.")
elseif dc < 1.5*max(db,1.0)
    println(">> H5: e is genuinely marginal conditional on the shared stellar model.")
    println("   Np=2 is the correct K2 answer; recover e only in K2+TESS+RV joint.")
else
    println(">> PARTIAL: shared params help but don't fully recover +210 — mixed H1.")
end
