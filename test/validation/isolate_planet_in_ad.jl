#!/usr/bin/env julia
# Localize the residual planet-count bias in the AD context (proper gate gave
# logit -2.19 vs evidence -3.24, residual ~1.0; isolated white-noise gave only
# ~0.4). MoMS off/scales are built from PRIORS (data-independent) so the planet
# birth/death truncation residual is identical here — the extra ~0.6 must be the
# SWAP or EQUILIBRATION. Arms:
#   (A) AD FIXED ON, planets 0↔1, NO swap (td.noise=false), 14k and 40k steps.
#   (B) FULL swap version (AD toggling), 40k steps.
# Δ(Np1−Np0, both AD-on) = -3.24 ± 0.04 (from toy_swap_db_gate_proper.jl).
#   A≈0.04        ⇒ swap adds the residual
#   A≈0.10        ⇒ planet-birth-in-AD-context or equilibration; 40k<14k ⇒ mixing
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
const ΔZ = -3.24
expect = exp(ΔZ)/(1+exp(ΔZ))
@printf("expect P(Np1)=%.3f (logit %+.2f)\n", expect, ΔZ)

# Arm A: AD fixed on (transdim_noise=false), planets toggle, NO swap (td.noise=false)
pA = Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=["I1"]),
    data=data, parametrization=ParametrizationConfig(time=:Mo), priors=pri,
    noise_models=NoiseModel[ad], transdim_noise=false, stability=:none)
tdA = TransDimConfig(; max_kplanet=1, planets=true, noise=false)
runA(ns,nb,sd) = begin
    r = sample_transdim_ptemcee(NereusTarget(pA,data), data; td=tdA, n_temps=12, n_walkers=64,
        n_steps=ns, n_burnin=nb, n_birth_tries=8, n_birth_refine=10, seed=sd, show_progress=false)
    m = Array(r.chains[:,:n_planets,:]); (mean(m.==1), m)
end
for (ns,nb) in ((14000,5000),(40000,12000))
    p,m = runA(ns,nb,1); ni=size(m,1)
    @printf("A AD-fixed-on noswap %dk: P(Np1)=%.3f | thirds %.3f %.3f %.3f\n", ns÷1000, p,
        mean(m[1:ni÷3,:].==1), mean(m[ni÷3+1:2ni÷3,:].==1), mean(m[2ni÷3+1:end,:].==1))
end

# Arm B: full swap version (AD toggling), 40k
pB = Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=["I1"]),
    data=data, parametrization=ParametrizationConfig(time=:Mo), priors=pri,
    noise_models=NoiseModel[ad], transdim_noise=true, stability=:none)
tdB = TransDimConfig(; max_kplanet=1, planets=true, noise=true, toggleable=NoiseModel[ad])
rB = sample_transdim_ptemcee(NereusTarget(pB,data), data; td=tdB, n_temps=12, n_walkers=64,
        n_steps=40000, n_burnin=12000, n_birth_tries=8, n_birth_refine=10, seed=1, show_progress=false)
mB = Array(rB.chains[:,:n_planets,:]); ni=size(mB,1)
@printf("B swap 40k: P(Np1)=%.3f | thirds %.3f %.3f %.3f\n", mean(mB.==1),
    mean(mB[1:ni÷3,:].==1), mean(mB[ni÷3+1:2ni÷3,:].==1), mean(mB[2ni÷3+1:end,:].==1))
