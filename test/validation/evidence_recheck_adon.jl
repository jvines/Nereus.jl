#!/usr/bin/env julia
# Is the proper-gate residual (occupancy logit -2.19 vs evidence Δ=-3.24) an
# OCCUPANCY bias or a fixed-dim PT-EVIDENCE bias? The planet birth/death Hastings
# is identical to the white-noise case (off/scales are prior-built) and verified
# clean there (0.13→0.043). A DB-correct move's stationary IS the true Bayes
# factor regardless of likelihood, so a residual that appears ONLY in the (ragged
# junk-planet+AD) likelihood points at the EVIDENCE estimate. 12-temp TI tends to
# UNDER-estimate logZ for ragged posteriors → Δ too negative.
# Recompute Δ(Np1−Np0, both AD-on) at 12 vs 20 vs 28 temps. If Δ RISES toward
# the occupancy's -2.19 as temps increase ⇒ evidence-estimator bias, occupancy OK.
using Nereus, Statistics, Printf, Random
rng = MersenneTwister(2026); const N = 90
const T = sort(rand(rng, N)) .* 100.0; const SIG = 2.0
act = 5.0 .* sin.(2π .* T ./ 15.0 .+ 0.7)
rv  = act .+ SIG .* randn(rng, N)
bis = 10.0 .* sin.(2π .* T ./ 15.0 .+ 0.7) .+ 0.05 .* randn(rng, N)
data = Data(t_rv=T, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
            indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))
ad = ActivityDecorrelation(indicators=["bis"]); rvmax = maximum(abs, rv)
base(maxk) = begin
    p = Dict{String,PriorSpec}("gamma_I1"=>UniformPrior(-3rvmax,3rvmax), "sigma_I1"=>LogUniformPrior(0.2,12.0))
    for kk in 1:maxk
        p["P_k$kk"]=LogUniformPrior(2.0,40.0); p["K_k$kk"]=UniformPrior(0.0,30.0)
        p["sesinw_k$kk"]=UniformPrior(-0.6,0.6); p["secosw_k$kk"]=UniformPrior(-0.6,0.6); p["Mo_k$kk"]=UniformPrior(0.0,2π)
    end; p
end
mkp(maxk) = Params(; max_kplanet=maxk, planet_modes=fill(RV_ONLY,maxk), instruments=InstrumentConfig(rv=["I1"]),
    data=data, parametrization=ParametrizationConfig(time=:Mo), priors=base(maxk),
    noise_models=NoiseModel[ad], transdim_noise=false, stability=:none)
@printf("occupancy logit (truth proxy from DB-correct chain) = -2.19\n")
for ntmp in (12, 20, 28)
    z0=Float64[]; z1=Float64[]
    for sd in 1:3
        r0 = sample_ptemcee(NereusTarget(mkp(0),data), data; n_temps=ntmp, n_walkers=60,
                n_steps=12000, n_burnin=4000, seed=sd, init_strategy=:prior, show_progress=false)
        r1 = sample_ptemcee(NereusTarget(mkp(1),data), data; n_temps=ntmp, n_walkers=60,
                n_steps=12000, n_burnin=4000, seed=sd, init_strategy=:prior, show_progress=false)
        push!(z0,r0.log_evidence); push!(z1,r1.log_evidence)
    end
    Δ=mean(z1)-mean(z0); Δe=sqrt(std(z0)^2+std(z1)^2)/sqrt(3)
    @printf("n_temps=%2d: logZ0=%.2f logZ1=%.2f  Δ=%+.2f ± %.2f\n", ntmp, mean(z0), mean(z1), Δ, Δe)
end
