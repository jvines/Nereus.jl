#!/usr/bin/env julia
# SMOKE TEST of the untested trans-dim config: planets AND noise toggling
# simultaneously (planets=true, noise=true), with the planet count free to
# reach 0. Two controlled toys — checks both failure directions:
#   A (planet present): K=6 @ P=10d + white, indicator uncorrelated.
#       expect  Np=1 recovered, noise models OFF.
#   B (NULL: activity, no planet): 5·sin(2π t/15) + white, indicator traces it.
#       expect  Np=0 (NOT a fake 15d Keplerian), AD ON (absorbs the activity).
# Pass = both expectations met → the planets+noise combo works and the null
# mode doesn't manufacture planets.

using Nereus, MCMCChains, Statistics, Printf, Random
const N = 80
rng = MersenneTwister(11)
const T = sort(rand(rng, N)) .* 90.0
const SIG = 2.0

function kep(K, P, ϕ); K .* sin.(2π .* T ./ P .+ ϕ); end

function run_toy(label, rv, bis, seed)
    data = Data(t_rv=T, rv=rv, rv_err=fill(SIG, N), rv_inst=ones(Int, N),
                indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0, N)))
    rvmax = maximum(abs, rv)
    pri = Dict{String,PriorSpec}(
        "P_k1"=>LogUniformPrior(2.0,40.0), "K_k1"=>UniformPrior(0.0,30.0),
        "sesinw_k1"=>UniformPrior(-0.6,0.6), "secosw_k1"=>UniformPrior(-0.6,0.6), "Mo_k1"=>UniformPrior(0.0,2π),
        "P_k2"=>LogUniformPrior(2.0,40.0), "K_k2"=>UniformPrior(0.0,30.0),
        "sesinw_k2"=>UniformPrior(-0.6,0.6), "secosw_k2"=>UniformPrior(-0.6,0.6), "Mo_k2"=>UniformPrior(0.0,2π),
        "gamma_I1"=>UniformPrior(-3rvmax,3rvmax), "sigma_I1"=>LogUniformPrior(0.2,12.0))
    ad  = ActivityDecorrelation(indicators=["bis"])
    ma  = MAModel(order=1)
    flr = IndicatorFloor(channels=[:bis])
    toggle = NoiseModel[ad, ma]
    params = Params(; max_kplanet=2, planet_modes=fill(RV_ONLY,2), instruments=InstrumentConfig(rv=["I1"]),
        data=data, parametrization=ParametrizationConfig(time=:Mo), priors=pri,
        noise_models=NoiseModel[ad, ma, flr], transdim_noise=true, stability=:none)
    target = NereusTarget(params, data)
    td = TransDimConfig(; max_kplanet=2, planets=true, noise=true, toggleable=toggle)  # NO exclusion: clean planet+noise combo
    res = sample_transdim_ptemcee(target, data; td=td, n_temps=10, n_walkers=64,
        n_steps=9000, n_burnin=3500, n_birth_tries=8, n_birth_refine=10, seed=seed, show_progress=false)
    ch = res.chains
    np = vec(Array(ch[:n_planets]))
    @printf("\n[%s]\n", label)
    for k in 0:2; @printf("  P(Np=%d) = %.3f\n", k, mean(np .== k)); end
    nm = params.config.noise_models
    for (lbl,m) in (("AD",ad),("MA",ma))
        i = findfirst(==(m), nm); col = Symbol("noise_active_$i")
        col in names(ch,:parameters) && @printf("  P(%s on) = %.3f\n", lbl, mean(vec(Array(ch[col])).>0.5))
    end
    if :K_k1 in names(ch,:parameters)
        K = vec(Array(ch[:,:K_k1,:])); P = vec(Array(ch[:,:P_k1,:]))
        sel = np .>= 1
        if any(sel); @printf("  when Np≥1: K_k1=%.2f [%.2f,%.2f]  P_k1=%.2f\n",
            median(K[sel]),quantile(K[sel],0.16),quantile(K[sel],0.84), median(P[sel])); end
    end
    flush(stdout)
    return mean(np .>= 1)
end

# Toy A: real planet K=6 @ P=10d, indicator = uncorrelated white
rvA  = kep(6.0, 10.0, 1.0) .+ SIG .* randn(rng, N)
bisA = randn(rng, N)
# Toy B: NO planet, activity = 5·sin(2π t/15), indicator traces it (AD-absorbable)
act  = 5.0 .* sin.(2π .* T ./ 15.0 .+ 0.7)
rvB  = act .+ SIG .* randn(rng, N)
# amplitude 10 (≫ activity amp 5) so the OLS coefficient C_ols≈0.5 sits inside
# AD's U(-1,1) prior — mirrors HD18599's load_adfmt scaling indicators to the RV
# range. (With bis amp 1, C needed ≈5 → clamped to 1 → AD could only remove 1/5.)
bisB = 10.0 .* sin.(2π .* T ./ 15.0 .+ 0.7) .+ 0.05 .* randn(rng, N)

println("="^60); println("SMOKE: planets + noise both toggling (planet count free to 0)")
pA = run_toy("A: planet K=6 @ P=10, white  -> expect Np=1, noise off", rvA, bisA, 1)
pB = run_toy("B: NULL activity sin(15d), no planet -> expect Np=0, AD on", rvB, bisB, 2)
@printf("\nVERDICT: toy A P(Np≥1)=%.2f (want high)   toy B P(Np≥1)=%.2f (want LOW)\n", pA, pB)
println(pA > 0.6 && pB < 0.4 ? "✅ planets+noise combo OK (recovers real planet, null stays planet-free)" :
                                "❌ check — combo did not behave as expected")
