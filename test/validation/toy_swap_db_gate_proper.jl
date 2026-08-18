#!/usr/bin/env julia
# PROPER detailed-balance gate for the planet↔AD swap.
#
# The first gate (toy_swap_db_gate.jl) compared logZ(planet-15d, NO AD) vs
# logZ(AD) — but the swap-enabled trans-dim NEVER visits (Np=1, AD-off): it is
# at (AD-on) ≈99.8% of the time. The swap correctly moved the 15-d signal onto
# AD, so the residual "Np=1" is a spurious LOW-K planet COEXISTING with AD
# (K≈0.7, not ≈5). The model states that actually matter are therefore both
# AD-on:
#     logit(P(Np=1)/P(Np=0)) ?= logZ(Np=1, AD on) − logZ(Np=0, AD on)
# i.e. is the spurious-planet occupancy the genuine Occam penalty for adding a
# junk planet on top of AD? PASS ⇒ occupancy = P(M|D) for the models the chain
# really explores.
using Nereus, MCMCChains, Statistics, Printf, Random

rng = MersenneTwister(2026); const N = 90
const T = sort(rand(rng, N)) .* 100.0; const SIG = 2.0
act = 5.0 .* sin.(2π .* T ./ 15.0 .+ 0.7)
rv  = act .+ SIG .* randn(rng, N)
bis = 10.0 .* sin.(2π .* T ./ 15.0 .+ 0.7) .+ 0.05 .* randn(rng, N)
data = Data(t_rv=T, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
            indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))
ad = ActivityDecorrelation(indicators=["bis"]); rvmax = maximum(abs, rv)

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
# AD ALWAYS ON (transdim_noise=false ⇒ fixed model with the AD active), planet
# count fixed at maxk.
mkparams(maxk) = Params(; max_kplanet=maxk, planet_modes=fill(RV_ONLY, maxk),
    instruments=InstrumentConfig(rv=["I1"]), data=data,
    parametrization=ParametrizationConfig(time=:Mo), priors=base(maxk),
    noise_models=NoiseModel[ad], transdim_noise=false, stability=:none)

# ---- fixed-dim evidences, BOTH AD-on (3 seeds each) ----
z0 = Float64[]; z1 = Float64[]
for sd in 1:3
    r0 = sample_ptemcee(NereusTarget(mkparams(0), data), data;
            n_temps=12, n_walkers=60, n_steps=8000, n_burnin=3000, seed=sd,
            init_strategy=:prior, show_progress=false)
    r1 = sample_ptemcee(NereusTarget(mkparams(1), data), data;
            n_temps=12, n_walkers=60, n_steps=8000, n_burnin=3000, seed=sd,
            init_strategy=:prior, show_progress=false)
    push!(z0, r0.log_evidence); push!(z1, r1.log_evidence)
    @printf("  seed %d: logZ(Np0,ADon)=%.2f  logZ(Np1,ADon)=%.2f\n", sd, r0.log_evidence, r1.log_evidence)
end
Δz  = mean(z1) - mean(z0)                       # +log[P(Np1)/P(Np0)] predicted
Δze = sqrt(std(z0)^2 + std(z1)^2)/sqrt(3)
@printf("fixed-dim (both AD-on): logZ(Np0)=%.2f  logZ(Np1)=%.2f   Δ(Np1−Np0)=%+.2f ± %.2f\n",
        mean(z0), mean(z1), Δz, Δze)

# ---- swap-enabled trans-dim (planets 0↔1 + AD toggle) ----
pT = mkparams(1); pT = Params(; max_kplanet=1, planet_modes=[RV_ONLY],
    instruments=InstrumentConfig(rv=["I1"]), data=data,
    parametrization=ParametrizationConfig(time=:Mo), priors=base(1),
    noise_models=NoiseModel[ad], transdim_noise=true, stability=:none)
td = TransDimConfig(; max_kplanet=1, planets=true, noise=true, toggleable=NoiseModel[ad])
obs = Float64[]
for sd in 1:3
    res = sample_transdim_ptemcee(NereusTarget(pT,data), data; td=td, n_temps=12,
            n_walkers=64, n_steps=14000, n_burnin=5000, n_birth_tries=8,
            n_birth_refine=10, seed=sd, show_progress=false)
    np = vec(Array(res.chains[:n_planets]))
    p1 = clamp(mean(np .== 1), 1e-3, 1-1e-3)
    push!(obs, log(p1/(1-p1)))
    @printf("  td seed %d: P(Np=1)=%.3f  logit=%+.2f\n", sd, mean(np.==1), log(p1/(1-p1)))
end
lo = mean(obs)
@printf("\ntrans-dim  logit(P(Np1)/P(Np0)) = %+.2f ± %.2f\n", lo, std(obs)/sqrt(3))
@printf("predicted  (=ΔlogZ, both AD-on)  = %+.2f ± %.2f   discrepancy = %.2f nats\n",
        Δz, Δze, abs(lo - Δz))
println(abs(lo - Δz) < 3*max(Δze, 0.3) + 0.7 ?
        "✅ SWAP DB-CORRECT — occupancy reproduces the AD-on evidence ratio" :
        "❌ FAIL — junk-planet occupancy ≠ evidence")
