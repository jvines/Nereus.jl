#!/usr/bin/env julia
# Equivalence oracle for Fix B (total phot-ll cache). For a long random walk
# over theta (births/deaths + coordinate perturbations covering every param
# type), assert at EVERY state:
#   (1) cache correctness — the possibly-cached ws eval == a forced recompute
#       via the SAME code path, BIT-EXACT. A stale cache from an incomplete
#       hash (a phot param not hashed) makes these differ → caught.
#   (2) ws-path correctness — the recompute == the independent non-ws full
#       eval, to float tolerance.
# Exercises no-transit (n_p=0) AND with-transit states.

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
td = TransDimConfig(max_kplanet=4, planets=true, noise=false,
    birth_strategies=[PriorBirth(),InformedBirth()], birth_weights=[0.3,0.7], transdim_fraction=0.3)

# Internal handles
const P = Nereus
tll_ws(theta)  = P.transit_log_likelihood(theta, data, ws)
tll_full(theta) = P.transit_log_likelihood(theta, data)

rng = MersenneTwister(20260606)
td_state = P.TransDimState(max_planets=4, n_noise=0)
theta = P.Theta{Float64}(params; td=td_state)
P._init_systemics_from_prior!(theta, rng)
ws = P.PTWorkspace(params, 4, 0; n_obs=length(bjd), n_phot=length(flat.t))

log_pi = P.log_prior(theta); log_L = P._eval_ll(theta, data, nothing)
N = 8000
n_cache_fail = 0; n_path_fail = 0; max_cache_err = 0.0; max_path_err = 0.0
n_eval_fail = 0; max_eval_err = 0.0   # full _eval_ll(ws) vs nonws (ws-slice correctness)
n_finite = 0; n_transit_states = 0; n_notransit_states = 0
widths = P._prior_widths(params.layout)

for step in 1:N
    # --- mutate theta ---
    if rand(rng) < 0.3
        P._planet_move!(theta, data, td, rng, log_pi, log_L; ctr=nothing) do lpi, lL
            global log_pi = lpi; global log_L = lL
        end
    else
        aidx, apos = P._active_unfrozen(params, theta.td)
        if !isempty(aidx)
            r = rand(rng, 1:length(aidx)); idx = aidx[r]
            theta.values[idx] += randn(rng) * 0.5 * max(widths[apos[r]], 1e-3)
        end
        global log_pi = P.log_prior(theta)
        global log_L  = P._eval_ll(theta, data, nothing)
    end

    # --- compare at this state ---
    ll_cached = tll_ws(theta)             # may hit cache (stale if a bug)
    ws.phot_ll_total_hash = zero(UInt)    # invalidate
    ll_fresh  = tll_ws(theta)             # forced recompute, same code path
    ll_nonws  = tll_full(theta)           # independent ground truth

    # ws-slice correctness: full eval (rv+transit) ws vs nonws must agree
    ll_eval_ws = P._eval_ll(theta, data, nothing, ws)
    ll_eval_nw = P._eval_ll(theta, data, nothing)
    if isfinite(ll_eval_nw)
        ee = abs(ll_eval_ws - ll_eval_nw) / (1 + abs(ll_eval_nw))
        global max_eval_err = max(max_eval_err, ee)
        ee < 1e-6 || (global n_eval_fail += 1)
    end

    P.n_p(theta) >= 1 ? (global n_transit_states += 1) : (global n_notransit_states += 1)

    if isfinite(ll_fresh)
        global n_finite += 1
        ce = abs(ll_cached - ll_fresh)
        global max_cache_err = max(max_cache_err, ce)
        ce == 0.0 || (global n_cache_fail += 1)         # cache MUST be bit-exact
        if isfinite(ll_nonws)
            pe = abs(ll_fresh - ll_nonws) / (1 + abs(ll_nonws))
            global max_path_err = max(max_path_err, pe)
            if pe >= 1e-6
                global n_path_fail += 1
                if n_path_fail <= 8
                    np = P.n_p(theta)
                    @printf("  FAIL step=%d n_p=%d  ws=%.4f  nonws=%.4f  rel=%.3e\n",
                            step, np, ll_fresh, ll_nonws, pe)
                    for k in P.planet_indices(theta)
                        blk = params.layout.planet_blocks[k]
                        P.has_geometry(blk) || continue
                        Pk = P.planet_P(theta, k); e,w = P.planet_e_w(theta, k)
                        b,rrk = P.planet_b_rr(theta, k)
                        @printf("     planet k=%d  P=%.3f e=%.3f w=%.2f b=%.3f rr=%.3f\n",
                                k, Pk, e, w, b, rrk)
                    end
                end
            end
        end
    else
        # non-finite must agree in finiteness
        (isfinite(ll_cached) == isfinite(ll_fresh)) || (global n_cache_fail += 1)
    end
end

@printf("\nsteps=%d  finite=%d  (no-transit states=%d, transit states=%d)\n",
        N, n_finite, n_notransit_states, n_transit_states)
@printf("cache:   max |cached-fresh| = %.3e   failures(non-bit-exact) = %d\n", max_cache_err, n_cache_fail)
@printf("wspath:  max rel|fresh-nonws| = %.3e   failures(>1e-6) = %d\n", max_path_err, n_path_fail)
@printf("eval_ll: max rel|ws-nonws|   = %.3e   failures(>1e-6) = %d   (ws-slice correctness)\n", max_eval_err, n_eval_fail)
println((n_cache_fail == 0 && n_path_fail == 0 && n_eval_fail == 0) ?
        "✅ ALL EQUIVALENT — Fix B cache bit-exact, ws transit == nonws, ws _eval_ll == nonws" :
        "❌ MISMATCH — see failure counts")
