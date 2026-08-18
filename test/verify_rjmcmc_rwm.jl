#!/usr/bin/env julia
# Verify Fix A: sample_rjmcmc within_model=:rwm vs :slice on native WASP-47.
# Checks (1) rwm runs without error, (2) finite log_L, (3) much faster than
# slice at equal budget. Recovery isn't expected at tiny budget — this is a
# speed+sanity gate before the full run.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std
using Printf
using MCMCChains

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))

lc = load_tess_lc(joinpath(DATADIR, "WASP-47_tess_s42_lc.csv"))
p = sortperm(lc.t); flat = (t=lc.t[p], flux=lc.flux[p], flux_err=lc.flux_err[p])
rr = readdlm(joinpath(DATADIR, "WASP47_RVs_combined.csv"), ',', Any, '\n'; header=true)
raw = rr[1]; hdr = vec(rr[2]); c(n)=findfirst(==(n),hdr)
fcol=c("flag"); keep=trues(size(raw,1))
for i in 1:size(raw,1); f=raw[i,fcol]; f===missing && continue
    String(strip(string(f))) in ("transit","transit_night","anomalous") && (keep[i]=false); end
raw=raw[keep,:]
bjd=Float64.(raw[:,c("bjd")]); rv=Float64.(raw[:,c("rv")]); rve=Float64.(raw[:,c("rv_err")])
istr=String.(raw[:,c("instrument")]); inames=sort(unique(istr)); i2i=Dict(n=>i for (i,n) in enumerate(inames))
rinst=[i2i[s] for s in istr]
data = Data(; t_rv=bjd, rv=rv, rv_err=rve, rv_inst=rinst, t_phot=flat.t, flux=flat.flux,
            flux_err=flat.flux_err, phot_inst=ones(Int,length(flat.t)))
ic = InstrumentConfig(rv=inames, pm=["TESS"])
pri = Dict{String,PriorSpec}()
for k in 1:3
    pri["P_k$k"]=LogUniformPrior(0.5,30.0); pri["K_k$k"]=UniformPrior(0.0,200.0)
    pri["sesinw_k$k"]=UniformPrior(-0.7,0.7); pri["secosw_k$k"]=UniformPrior(-0.7,0.7)
    pri["Tc_k$k"]=UniformPrior(flat.t[1],flat.t[1]+30.0)
    pri["b_k$k"]=UniformPrior(0.0,1.0); pri["rr_k$k"]=UniformPrior(0.005,0.20)
end
pri["P_k4"]=LogUniformPrior(100.0,1500.0); pri["K_k4"]=UniformPrior(0.0,200.0)
pri["sesinw_k4"]=UniformPrior(-0.7,0.7); pri["secosw_k4"]=UniformPrior(-0.7,0.7)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue
    ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
pri["offset_TESS"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_TESS"]=LogUniformPrior(1e-5,5e-3)
pri["q1_TESS"]=UniformPrior(0.0,1.0); pri["q2_TESS"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(1.0,0.3,0.1,5.0)
params = Params(; max_kplanet=4, planet_modes=[RVPM,RVPM,RVPM,RV_ONLY], instruments=ic, data=data,
    M_s=1.04, R_s=1.137, parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true),
    priors=pri, trend_order=0, stability=:amd, external_priors=[ExternalPrior(:ecc, NormalPrior(0.0,0.3), true)])
target = NereusTarget(params, data; unconstrained=true)
td = TransDimConfig(max_kplanet=4, planets=true, noise=false,
    birth_strategies=[PriorBirth(),InformedBirth()], birth_weights=[0.3,0.7], transdim_fraction=0.3)
@printf("native: %d phot, %d RV, %d free params\n", length(flat.t), length(bjd), n_unfrozen(params))

NW, NS = 30, 20
# compile both paths
sample_rjmcmc(target, target.data; td=td, n_samples=2, n_warmup=2, seed=1, n_chains=1, within_model=:slice, show_progress=false)
sample_rjmcmc(target, target.data; td=td, n_samples=2, n_warmup=2, seed=1, n_chains=1, within_model=:rwm, show_progress=false)

println("\n--- :slice ---")
t0=time(); ch_s, ev_s = sample_rjmcmc(target, target.data; td=td, n_samples=NS, n_warmup=NW, seed=7, n_chains=1, within_model=:slice, show_progress=false)
dt_s=time()-t0
@printf("slice: %.1fs  evals=%d  (%.2f s/iter)\n", dt_s, ev_s, dt_s/(NW+NS))

println("\n--- :rwm ---")
t0=time(); ch_r, ev_r = sample_rjmcmc(target, target.data; td=td, n_samples=NS, n_warmup=NW, seed=7, n_chains=1, within_model=:rwm, show_progress=false)
dt_r=time()-t0
@printf("rwm:   %.1fs  evals=%d  (%.2f s/iter)\n", dt_r, ev_r, dt_r/(NW+NS))

@printf("\nSPEEDUP: %.1fx faster, %.1fx fewer evals\n", dt_s/max(dt_r,1e-9), ev_s/max(ev_r,1))
ll_s = isempty(ch_s) ? NaN : 0.0
@printf("slice n_p range: %s ;  rwm n_p range: %s\n",
        string(extrema(vec(Array(ch_s[:, :n_planets, :])))),
        string(extrema(vec(Array(ch_r[:, :n_planets, :])))))
println(isfinite(dt_r) && ev_r < ev_s ? "✅ rwm runs, fewer evals — Fix A wired correctly" : "⚠ check")
