#!/usr/bin/env julia
# For max_kplanet=1 the MoMS birth/death is provably DB-correct post-truncation-
# fix (combinatorial terms vanish: _birth_death_probs gives (1,0)/(0,1), Hastings
# = ∓log_q). So the base residual in the CLEANEST config (refine=0,tries=1;
# AD context gave P(Np1)=0.063 vs truth 0.033) cannot be a Hastings bug — test
# whether it's PT-ladder mixing of the discrete dimension by sweeping n_temps.
# truth P(Np1) ≈ 0.033 (logit -3.38, robust high-temp evidence).
#   P(Np1) → 0.033 as temps↑  ⇒ ladder mixing; code correct, calibration needs temps
#   plateaus at 0.06          ⇒ deeper residual
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
@printf("truth P(Np1)≈0.033 (logit -3.38), cleanest config refine=0/tries=1\n")
for ntmp in (12, 20, 28)
    r = sample_transdim_ptemcee(NereusTarget(p,data), data; td=td, n_temps=ntmp, n_walkers=64,
        n_steps=20000, n_burnin=7000, n_birth_tries=1, n_birth_refine=0, seed=1, show_progress=false)
    m = Array(r.chains[:,:n_planets,:]); ni=size(m,1)
    @printf("n_temps=%2d: P(Np1)=%.3f | thirds %.3f %.3f %.3f\n", ntmp, mean(m.==1),
        mean(m[1:ni÷3,:].==1), mean(m[ni÷3+1:2ni÷3,:].==1), mean(m[2ni÷3+1:end,:].==1))
end
