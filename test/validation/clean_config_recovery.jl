#!/usr/bin/env julia
# Does the CLEAN (default, DB-correct, calibrated) config — n_birth_tries=1,
# n_birth_refine=0 — also RECOVER a real planet, or are the (biased) recovery
# aids actually needed? If clean config recovers Toy A AND rejects null Toy B,
# the recovery-vs-calibration tension dissolves: use the clean config for both.
# Same toys as toy_planets_plus_noise_smoke.jl. n_temps=16 (adequate ladder).
using Nereus, MCMCChains, Statistics, Printf, Random
const N = 80; rng = MersenneTwister(11)
const T = sort(rand(rng, N)) .* 90.0; const SIG = 2.0
kep(K,P,ϕ) = K .* sin.(2π .* T ./ P .+ ϕ)
function run_toy(label, rv, bis, seed)
    data = Data(t_rv=T, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
                indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))
    rvmax = maximum(abs, rv)
    pri = Dict{String,PriorSpec}(
        "P_k1"=>LogUniformPrior(2.0,40.0), "K_k1"=>UniformPrior(0.0,30.0),
        "sesinw_k1"=>UniformPrior(-0.6,0.6), "secosw_k1"=>UniformPrior(-0.6,0.6), "Mo_k1"=>UniformPrior(0.0,2π),
        "P_k2"=>LogUniformPrior(2.0,40.0), "K_k2"=>UniformPrior(0.0,30.0),
        "sesinw_k2"=>UniformPrior(-0.6,0.6), "secosw_k2"=>UniformPrior(-0.6,0.6), "Mo_k2"=>UniformPrior(0.0,2π),
        "gamma_I1"=>UniformPrior(-3rvmax,3rvmax), "sigma_I1"=>LogUniformPrior(0.2,12.0))
    ad = ActivityDecorrelation(indicators=["bis"]); ma = MAModel(order=1); flr = IndicatorFloor(channels=[:bis])
    params = Params(; max_kplanet=2, planet_modes=fill(RV_ONLY,2), instruments=InstrumentConfig(rv=["I1"]),
        data=data, parametrization=ParametrizationConfig(time=:Mo), priors=pri,
        noise_models=NoiseModel[ad, ma, flr], transdim_noise=true, stability=:none)
    td = TransDimConfig(; max_kplanet=2, planets=true, noise=true, toggleable=NoiseModel[ad, ma])
    # CLEAN config: no recovery aids
    res = sample_transdim_ptemcee(NereusTarget(params,data), data; td=td, n_temps=16, n_walkers=64,
        n_steps=20000, n_burnin=7000, n_birth_tries=1, n_birth_refine=0, seed=seed, show_progress=false)
    np = vec(Array(res.chains[:n_planets]))
    @printf("\n[%s]  P(Np=0)=%.3f  P(Np=1)=%.3f  P(Np=2)=%.3f\n", label, mean(np.==0), mean(np.==1), mean(np.==2))
    if :K_k1 in names(res.chains,:parameters)
        K = vec(Array(res.chains[:,:K_k1,:])); P = vec(Array(res.chains[:,:P_k1,:])); sel = np.>=1
        any(sel) && @printf("  when Np≥1: K_k1=%.2f  P_k1=%.2f\n", median(K[sel]), median(P[sel]))
    end
    mean(np .>= 1)
end
rvA  = kep(6.0,10.0,1.0) .+ SIG .* randn(rng, N); bisA = randn(rng, N)        # real planet, uncorrelated indicator
act  = 5.0 .* sin.(2π .* T ./ 15.0 .+ 0.7); rvB = act .+ SIG .* randn(rng, N)
bisB = 10.0 .* sin.(2π .* T ./ 15.0 .+ 0.7) .+ 0.05 .* randn(rng, N)          # null: activity traced by indicator
pA = run_toy("A: real planet K=6@P=10 -> want Np=1", rvA, bisA, 1)
pB = run_toy("B: NULL activity 15d   -> want Np=0", rvB, bisB, 2)
@printf("\nCLEAN config: A P(Np≥1)=%.2f (want high)   B P(Np≥1)=%.2f (want low)\n", pA, pB)
println(pA>0.6 && pB<0.15 ? "✅ clean config RECOVERS real planet AND calibrates null — no aids needed" :
        pA>0.6 ? "~ recovers planet; null not as tight as evidence — more temps/steps" :
                 "✗ clean config missed the planet — aids needed for recovery")
