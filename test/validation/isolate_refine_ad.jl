#!/usr/bin/env julia
# Does post-burn-in refine_birth! (and burn-in multi-try) bias the planet-count
# STATIONARY in the AD context? Both are claimed DB-safe (within-model M-H /
# burn-in-only), so the occupancy P(Np=1) MUST be invariant to n_birth_refine
# and n_birth_tries. AD fixed-on, planets 0↔1, NO swap. Truth (high-temp evidence)
# logit ≈ -3.38 → P(Np1) ≈ 0.033. Baseline (refine=10,tries=8) gave ~0.10.
#   refine=0 ≪ refine=10  ⇒ refinement biases the stationary (sticky newborns)
#   all equal (~0.10)     ⇒ refinement DB-safe; residual is elsewhere
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
p = Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=["I1"]),
    data=data, parametrization=ParametrizationConfig(time=:Mo), priors=pri,
    noise_models=NoiseModel[ad], transdim_noise=false, stability=:none)
td = TransDimConfig(; max_kplanet=1, planets=true, noise=false)
@printf("truth P(Np1)≈0.033 (logit -3.38).  baseline refine=10/tries=8 ≈ 0.10\n")
for (refine, tries) in ((0,1),(0,8),(10,1),(10,8))
    r = sample_transdim_ptemcee(NereusTarget(p,data), data; td=td, n_temps=12, n_walkers=64,
        n_steps=20000, n_burnin=7000, n_birth_tries=tries, n_birth_refine=refine, seed=1, show_progress=false)
    m = Array(r.chains[:,:n_planets,:]); ni=size(m,1)
    @printf("refine=%2d tries=%d: P(Np1)=%.3f | thirds %.3f %.3f %.3f\n", refine, tries, mean(m.==1),
        mean(m[1:ni÷3,:].==1), mean(m[ni÷3+1:2ni÷3,:].==1), mean(m[2ni÷3+1:end,:].==1))
end
