#!/usr/bin/env julia
# DEFINITIVE DB-correctness gate for the planet↔AD trans-dim system after the
# MoMS truncation-normalizer fix. Full realistic system (swap-enabled, AD
# toggling), clean DB config (no recovery aids: n_birth_tries=1, n_birth_refine=0),
# adequate PT ladder (24 temps so the discrete dimension mixes via swaps).
# Compare occupancy logit(P(Np1)/P(Np0)) to the high-temp evidence Δ=-3.38±0.02.
# PASS (≈ within MC) ⇒ trans-dim occupancy reproduces P(M|D): the toy fit is proper.
using Nereus, MCMCChains, Statistics, Printf, Random
rng = MersenneTwister(2026); const N = 90
const T = sort(rand(rng, N)) .* 100.0; const SIG = 2.0
act = 5.0 .* sin.(2π .* T ./ 15.0 .+ 0.7)
rv  = act .+ SIG .* randn(rng, N)
bis = 10.0 .* sin.(2π .* T ./ 15.0 .+ 0.7) .+ 0.05 .* randn(rng, N)
data = Data(t_rv=T, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
            indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))
ad = ActivityDecorrelation(indicators=["bis"]); rvmax = maximum(abs, rv)
pri = Dict{String,PriorSpec}("gamma_I1"=>UniformPrior(-3rvmax,3rvmax), "sigma_I1"=>LogUniformPrior(0.2,12.0),
    "P_k1"=>LogUniformPrior(2.0,40.0), "K_k1"=>UniformPrior(0.0,30.0),
    "sesinw_k1"=>UniformPrior(-0.6,0.6), "secosw_k1"=>UniformPrior(-0.6,0.6), "Mo_k1"=>UniformPrior(0.0,2π))
const ΔZ = -3.38; const ΔZe = 0.02      # robust, n_temps=28 (evidence_recheck_adon.jl)
p = Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=["I1"]),
    data=data, parametrization=ParametrizationConfig(time=:Mo), priors=pri,
    noise_models=NoiseModel[ad], transdim_noise=true, stability=:none)
td = TransDimConfig(; max_kplanet=1, planets=true, noise=true, toggleable=NoiseModel[ad])
obs = Float64[]
for sd in 1:3
    r = sample_transdim_ptemcee(NereusTarget(p,data), data; td=td, n_temps=24, n_walkers=64,
        n_steps=24000, n_burnin=8000, n_birth_tries=1, n_birth_refine=0, seed=sd, show_progress=false)
    m = vec(Array(r.chains[:n_planets])); p1 = clamp(mean(m.==1),1e-3,1-1e-3)
    push!(obs, log(p1/(1-p1)))
    @printf("  seed %d: P(Np1)=%.3f  logit=%+.2f\n", sd, mean(m.==1), log(p1/(1-p1)))
end
lo = mean(obs); loe = std(obs)/sqrt(3)
@printf("\ntrans-dim logit = %+.2f ± %.2f   evidence Δ = %+.2f ± %.2f   discrepancy = %.2f nats\n",
        lo, loe, ΔZ, ΔZe, abs(lo-ΔZ))
println(abs(lo-ΔZ) < 3*sqrt(loe^2+ΔZe^2)+0.4 ?
        "✅ DB-CORRECT — occupancy reproduces the evidence within MC" :
        "❌ residual remains")
