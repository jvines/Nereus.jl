#!/usr/bin/env julia
# Detailed-balance gate for the planet↔ActivityDecorrelation swap.
# A 15-d signal that BOTH a planet and AD (clean scaled indicator) can fit.
# Independent reference: fixed-dim evidences logZ(Np=1,no AD) and logZ(Np=0,AD)
# via sample_ptemcee (TI+/SS+/H+, validated to <0.01 nats on analytics).
# Then run the SWAP-enabled trans-dim (planets 0↔1 + AD toggle) and check the
# occupancy reproduces the evidence ratio:
#   logit(P(Np=0)) ≈ logZ(Np=0,AD) − logZ(Np=1,no AD)   (equal model priors).
# PASS ⇒ the swap samples P(M|D) (DB-correct). FAIL ⇒ Hastings bug.

using Nereus, MCMCChains, Statistics, Printf, Random

rng = MersenneTwister(2026)
const N = 90
const T = sort(rand(rng, N)) .* 100.0
const SIG = 2.0
const PSIG_AMP = 5.0
act  = PSIG_AMP .* sin.(2π .* T ./ 15.0 .+ 0.7)        # the shared 15-d signal
rv   = act .+ SIG .* randn(rng, N)
bis  = 10.0 .* sin.(2π .* T ./ 15.0 .+ 0.7) .+ 0.05 .* randn(rng, N)   # C_ols≈0.5 ∈ U(-1,1)
data = Data(t_rv=T, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
            indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))
ad = ActivityDecorrelation(indicators=["bis"])
rvmax = maximum(abs, rv)

base(maxk) = begin
    p = Dict{String,PriorSpec}("gamma_I1"=>UniformPrior(-3rvmax,3rvmax),
                                "sigma_I1"=>LogUniformPrior(0.2,12.0))
    for kk in 1:maxk
        p["P_k$kk"]=LogUniformPrior(2.0,40.0); p["K_k$kk"]=UniformPrior(0.0,30.0)
        p["sesinw_k$kk"]=UniformPrior(-0.6,0.6); p["secosw_k$kk"]=UniformPrior(-0.6,0.6)
        p["Mo_k$kk"]=UniformPrior(0.0,2π)
    end
    p
end
mkparams(maxk, noise) = Params(; max_kplanet=maxk,
    planet_modes=fill(RV_ONLY, maxk), instruments=InstrumentConfig(rv=["I1"]),
    data=data, parametrization=ParametrizationConfig(time=:Mo),
    priors=base(maxk), noise_models=noise, transdim_noise=!isempty(noise), stability=:none)

# ---- fixed-dim evidences (3 seeds each) ----
zpl = Float64[]; zad = Float64[]
for sd in 1:3
    rpl = sample_ptemcee(NereusTarget(mkparams(1, NoiseModel[]), data), data;
            n_temps=12, n_walkers=60, n_steps=8000, n_burnin=3000, seed=sd,
            init_strategy=:prior, show_progress=false)
    rad = sample_ptemcee(NereusTarget(mkparams(0, NoiseModel[ad]), data), data;
            n_temps=12, n_walkers=60, n_steps=8000, n_burnin=3000, seed=sd,
            init_strategy=:prior, show_progress=false)
    push!(zpl, rpl.log_evidence); push!(zad, rad.log_evidence)
end
Δz = mean(zad) - mean(zpl)
Δz_e = sqrt(std(zad)^2 + std(zpl)^2)/sqrt(3)
@printf("fixed-dim:  logZ(Np=1,noAD)=%.2f  logZ(Np=0,AD)=%.2f   ΔlogZ(AD−pl)=%+.2f ± %.2f\n",
        mean(zpl), mean(zad), Δz, Δz_e)

# ---- swap-enabled trans-dim ----
pT = mkparams(1, NoiseModel[ad])
td = TransDimConfig(; max_kplanet=1, planets=true, noise=true, toggleable=NoiseModel[ad])
pN0 = Float64[]
for sd in 1:3
    res = sample_transdim_ptemcee(NereusTarget(pT,data), data; td=td, n_temps=12,
            n_walkers=64, n_steps=14000, n_burnin=5000, n_birth_tries=8,
            n_birth_refine=10, seed=sd, show_progress=false)
    np = vec(Array(res.chains[:n_planets]))
    push!(pN0, mean(np .== 0))
    @printf("  td seed %d: P(Np=0)=%.3f\n", sd, mean(np .== 0))
end
p0 = clamp(mean(pN0), 1e-3, 1-1e-3)
obs = log(p0/(1-p0))
@printf("\nP(Np=0) trans-dim = %.3f  → logit = %+.2f\n", mean(pN0), obs)
@printf("predicted logit (=ΔlogZ) = %+.2f ± %.2f   discrepancy = %.2f nats\n",
        Δz, Δz_e, abs(obs - Δz))
println(abs(obs - Δz) < 3*max(Δz_e,0.3)+0.7 ?
        "✅ SWAP DB-CORRECT — occupancy reproduces the evidence ratio" :
        "❌ FAIL — swap occupancy ≠ evidence; Hastings bug")
