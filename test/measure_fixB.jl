#!/usr/bin/env julia
# Measure Fix B's speed gain: (1) per-eval phot-ll cost cache-hit vs cache-miss,
# (2) end-to-end rjmcmc :rwm wall time. Run this WITH Fix B (current code), then
# with the cache short-circuits disabled, and compare the rjmcmc times.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median
using Printf
using Random

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
const P = Nereus

# --- per-eval phot-ll cost: cache hit vs miss (no-transit, n_p=0) ---
rng = MersenneTwister(1)
theta = P.Theta{Float64}(params; td=P.TransDimState(max_planets=4, n_noise=0))
P._init_systemics_from_prior!(theta, rng)
ws = P.PTWorkspace(params, 4, 0; n_obs=length(bjd), n_phot=length(flat.t))
P.transit_log_likelihood(theta, data, ws)  # warm + populate cache
reps = 2000
t_hit = @elapsed for _ in 1:reps; P.transit_log_likelihood(theta, data, ws); end
t_miss = @elapsed for _ in 1:reps; ws.phot_ll_total_hash = zero(UInt); P.transit_log_likelihood(theta, data, ws); end
@printf("per-eval no-transit phot-ll:  hit=%.2f µs   miss=%.2f µs   (cache saves %.1fx)\n",
        1e6*t_hit/reps, 1e6*t_miss/reps, t_miss/max(t_hit,1e-12))

# --- end-to-end rjmcmc :rwm wall time (this is the with/without comparison) ---
sample_rjmcmc(target, target.data; td=td, n_samples=2, n_warmup=2, seed=1, n_chains=1, within_model=:slice, show_progress=false)
NW, NS = 100, 60
t0 = time()
ch, ev = sample_rjmcmc(target, target.data; td=td, n_samples=NS, n_warmup=NW, seed=7, n_chains=1, within_model=:slice, show_progress=false)
dt = time() - t0
@printf("rjmcmc :SLICE  %d+%d iters, 1 chain:  %.2f s  (%.1f ms/iter, %d evals)\n",
        NW, NS, dt, 1e3*dt/(NW+NS), ev)
