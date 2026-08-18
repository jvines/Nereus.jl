#!/usr/bin/env julia
# Recovery-aids config drains spurious planets slowly because the MoMS death
# resets to prior-median off_values — tiny reverse-birth density at the data-mode
# where refinement parks the planet. A data-INFORMED birth/death pair proposes at
# the mode, so a well-fit junk planet is killable (high informed reverse density)
# AND real planets birth well. Test informed_birth_fraction ∈ {0,0.5,1} with the
# recovery aids (tries=8/refine=10) at MODERATE temps (16) on the AD null toy.
# evidence P(Np1)=0.038 (logit -3.38). MoMS-only (frac=0) was ~0.10 here.
#   frac>0 → ~0.04 at moderate temps ⇒ informed births/deaths drain efficiently = the fix
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
    noise_models=NoiseModel[ad], transdim_noise=true, stability=:none)
td = TransDimConfig(; max_kplanet=1, planets=true, noise=true, toggleable=NoiseModel[ad])
@printf("evidence P(Np1)=0.038 (logit -3.38).  recovery aids tries=8/refine=10, n_temps=16\n")
for frac in (0.0, 0.5, 1.0)
    r = sample_transdim_ptemcee(NereusTarget(p,data), data; td=td, n_temps=16, n_walkers=64,
        n_steps=20000, n_burnin=7000, n_birth_tries=8, n_birth_refine=10,
        informed_birth_fraction=frac, seed=1, show_progress=false)
    m = Array(r.chains[:,:n_planets,:]); ni=size(m,1)
    @printf("informed_frac=%.1f: P(Np1)=%.3f | thirds %.3f %.3f %.3f\n", frac, mean(m.==1),
        mean(m[1:ni÷3,:].==1), mean(m[ni÷3+1:2ni÷3,:].==1), mean(m[2ni÷3+1:end,:].==1))
end
