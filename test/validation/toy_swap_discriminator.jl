#!/usr/bin/env julia
# Mixing-vs-Hastings discriminator for the planet↔AD swap. Same toy as the DB
# gate (AD favored by ~10 nats → true P(Np=0)≈1). One long swap-enabled chain;
# report P(Np=0) over thirds of post-burnin iterations.
#   climbs across thirds → SLOW MIXING (extend / strengthen moves)
#   flat at ~0.80        → HASTINGS BIAS (wrong stationary dist)
using Nereus, MCMCChains, Statistics, Printf, Random

rng = MersenneTwister(2026); N = 90
T = sort(rand(rng, N)) .* 100.0; SIG = 2.0
act = 5.0 .* sin.(2π .* T ./ 15.0 .+ 0.7)
rv  = act .+ SIG .* randn(rng, N)
bis = 10.0 .* sin.(2π .* T ./ 15.0 .+ 0.7) .+ 0.05 .* randn(rng, N)
data = Data(t_rv=T, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
            indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))
ad = ActivityDecorrelation(indicators=["bis"]); rvmax = maximum(abs, rv)
pri = Dict{String,PriorSpec}("gamma_I1"=>UniformPrior(-3rvmax,3rvmax), "sigma_I1"=>LogUniformPrior(0.2,12.0),
    "P_k1"=>LogUniformPrior(2.0,40.0), "K_k1"=>UniformPrior(0.0,30.0),
    "sesinw_k1"=>UniformPrior(-0.6,0.6), "secosw_k1"=>UniformPrior(-0.6,0.6), "Mo_k1"=>UniformPrior(0.0,2π))
params = Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=["I1"]),
    data=data, parametrization=ParametrizationConfig(time=:Mo), priors=pri,
    noise_models=NoiseModel[ad], transdim_noise=true, stability=:none)
td = TransDimConfig(; max_kplanet=1, planets=true, noise=true, toggleable=NoiseModel[ad])

NS = parse(Int, get(ENV,"NS","42000")); NB = parse(Int, get(ENV,"NB","12000"))
@printf("LONG swap-enabled chain: %d steps, %d burnin (true P(Np=0)≈1.0, ΔlogZ≈10.4)\n", NS, NB)
res = sample_transdim_ptemcee(NereusTarget(params,data), data; td=td, n_temps=12,
        n_walkers=64, n_steps=NS, n_burnin=NB, n_birth_tries=8, n_birth_refine=10,
        seed=1, show_progress=false)
npmat = Array(res.chains[:,:n_planets,:])       # (iterations × walkers)
niter = size(npmat,1)
@printf("P(Np=0) overall = %.3f\n", mean(npmat .== 0))
for (lbl, lo, hi) in (("1st third",1,niter÷3), ("2nd third",niter÷3+1,2niter÷3), ("3rd third",2niter÷3+1,niter))
    @printf("  %s (iter %d:%d): P(Np=0)=%.3f\n", lbl, lo, hi, mean(npmat[lo:hi,:] .== 0))
end
