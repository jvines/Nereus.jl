#!/usr/bin/env julia
# Verify Rank 1 (route NS-family through the ws likelihood): (1) the 4 edited
# NS files compile, (2) ws path == non-ws path (correctness), (3) the per-eval
# allocation win that drives the GC reduction on long NS runs, (4) a tiny
# sample_nested run actually works with the new ws path.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median
using Printf

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
# Fixed-dim 1-planet RVPM (all planets active → transit path fires)
pri = Dict{String,PriorSpec}()
pri["P_k1"]=LogUniformPrior(0.5,30.0); pri["K_k1"]=UniformPrior(0.0,200.0)
pri["sesinw_k1"]=UniformPrior(-0.7,0.7); pri["secosw_k1"]=UniformPrior(-0.7,0.7)
pri["Tc_k1"]=UniformPrior(flat.t[1],flat.t[1]+10.0); pri["b_k1"]=UniformPrior(0.0,1.0); pri["rr_k1"]=UniformPrior(0.005,0.20)
for n in inames; ri=rv[rinst.==i2i[n]]; isempty(ri)&&continue; ce=median(ri); sp=max(maximum(ri)-minimum(ri),1.0)
    pri["gamma_$n"]=UniformPrior(ce-3sp,ce+3sp); pri["sigma_$n"]=ModJeffreysPrior(0.1,50.0); end
pri["offset_TESS"]=NormalPrior(0.0,1e-3,-5e-3,5e-3); pri["jitter_TESS"]=LogUniformPrior(1e-5,5e-3)
pri["q1_TESS"]=UniformPrior(0.0,1.0); pri["q2_TESS"]=UniformPrior(0.0,1.0); pri["rho_s"]=NormalPrior(1.0,0.3,0.1,5.0)
params = Params(; max_kplanet=1, planet_modes=[RVPM], instruments=ic, data=data, M_s=1.04, R_s=1.137,
    parametrization=ParametrizationConfig(time=:Tc, geom=:b_rr, use_rho_s=true), priors=pri, trend_order=0, stability=:none)
const P = Nereus

# theta at b's orbit
theta = P.Theta{Float64}(params)
sesinw_to = 0.0
theta.values[params.layout.name_to_idx["P_k1"]] = 4.159
theta.values[params.layout.name_to_idx["K_k1"]] = 140.0
theta.values[params.layout.name_to_idx["sesinw_k1"]] = 0.0
theta.values[params.layout.name_to_idx["secosw_k1"]] = 0.0
theta.values[params.layout.name_to_idx["Tc_k1"]] = flat.t[1]+2.0
theta.values[params.layout.name_to_idx["b_k1"]] = 0.3
theta.values[params.layout.name_to_idx["rr_k1"]] = 0.1
theta.values[params.layout.name_to_idx["rho_s"]] = 1.0
ws = P.PTWorkspace(params, params.config.max_kplanet, length(params.config.noise_models);
                   n_obs=length(bjd), n_phot=length(flat.t))

# correctness
ll_nonws = P.rv_log_likelihood(theta, data) + P.transit_log_likelihood(theta, data)
ll_ws    = P.rv_log_likelihood(theta, data, ws) + P.transit_log_likelihood(theta, data, ws)
@printf("correctness: nonws=%.6f  ws=%.6f  |Δ|=%.2e\n", ll_nonws, ll_ws, abs(ll_nonws-ll_ws))

# per-eval allocations (the GC driver on long NS runs)
P.rv_log_likelihood(theta, data); P.transit_log_likelihood(theta, data)       # warm
P.rv_log_likelihood(theta, data, ws); P.transit_log_likelihood(theta, data, ws)
a_nonws = @allocated (P.rv_log_likelihood(theta, data) + P.transit_log_likelihood(theta, data))
ws.phot_ll_total_hash = zero(UInt)  # force a real recompute, not a cache hit
a_ws = @allocated (P.rv_log_likelihood(theta, data, ws) + P.transit_log_likelihood(theta, data, ws))
@printf("allocations/eval: nonws=%d B   ws=%d B   (%.0fx less)\n", a_nonws, a_ws, a_nonws/max(a_ws,1))

# tiny nested run actually works with the ws path
println("running tiny sample_nested (ws path) ...")
chains, logZ = sample_nested(NereusTarget(params, data), data; n_live=80, dlogz=2.0, proposal=:rwalk)
@printf("sample_nested: logZ=%.1f  nsamp=%d  ✅ runs with ws path\n", logZ, size(chains,1))
