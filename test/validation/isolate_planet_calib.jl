#!/usr/bin/env julia
# Isolate the planet birth/death calibration from AD/swap entirely.
# Pure WHITE NOISE (no planet, no activity), planet-only trans-dim 0↔1,
# inclusion_prior=0.5 (flat model prior) so the occupancy ratio must equal the
# Bayes factor:   logit(P(Np1)/P(Np0)) == logZ(Np1) − logZ(Np0).
# Run SHORT and LONG; report P(Np=1) for each + thirds of the long run.
#   LONG ≪ SHORT, →exp(ΔlogZ)  ⇒ slow draining (MIXING: death inefficient)
#   LONG ≈ SHORT  ≠ exp(ΔlogZ) ⇒ HASTINGS BIAS in planet birth/death
using Nereus, MCMCChains, Statistics, Printf, Random
rng = MersenneTwister(7); const N = 90
const T = sort(rand(rng, N)) .* 100.0; const SIG = 2.0
rv = SIG .* randn(rng, N)                                  # pure white noise
data = Data(t_rv=T, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N))
rvmax = maximum(abs, rv)
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
mkp(maxk; td=false) = Params(; max_kplanet=maxk, planet_modes=fill(RV_ONLY, maxk),
    instruments=InstrumentConfig(rv=["I1"]), data=data,
    parametrization=ParametrizationConfig(time=:Mo), priors=base(maxk),
    noise_models=NoiseModel[], transdim_noise=false, stability=:none)

z0=Float64[]; z1=Float64[]
for sd in 1:3
    r0 = sample_ptemcee(NereusTarget(mkp(0), data), data; n_temps=12, n_walkers=60,
            n_steps=8000, n_burnin=3000, seed=sd, init_strategy=:prior, show_progress=false)
    r1 = sample_ptemcee(NereusTarget(mkp(1), data), data; n_temps=12, n_walkers=60,
            n_steps=8000, n_burnin=3000, seed=sd, init_strategy=:prior, show_progress=false)
    push!(z0, r0.log_evidence); push!(z1, r1.log_evidence)
end
Δz = mean(z1)-mean(z0); Δze = sqrt(std(z0)^2+std(z1)^2)/sqrt(3)
@printf("fixed-dim: logZ(Np0)=%.2f logZ(Np1)=%.2f  Δ=%+.2f ± %.2f  → expect P(Np1)=%.3f\n",
        mean(z0), mean(z1), Δz, Δze, exp(Δz)/(1+exp(Δz)))

pT = Params(; max_kplanet=1, planet_modes=[RV_ONLY], instruments=InstrumentConfig(rv=["I1"]),
    data=data, parametrization=ParametrizationConfig(time=:Mo), priors=base(1),
    noise_models=NoiseModel[], transdim_noise=false, stability=:none)
td = TransDimConfig(; max_kplanet=1, planets=true, noise=false)
runchain(ns, nb, sd) = begin
    res = sample_transdim_ptemcee(NereusTarget(pT,data), data; td=td, n_temps=12, n_walkers=64,
            n_steps=ns, n_burnin=nb, n_birth_tries=8, n_birth_refine=10, seed=sd, show_progress=false)
    Array(res.chains[:,:n_planets,:])
end
for sd in 1:2
    m = runchain(14000, 5000, sd)
    @printf("SHORT seed %d: P(Np=1)=%.3f\n", sd, mean(m .== 1))
end
mL = runchain(40000, 12000, 1); ni = size(mL,1)
@printf("LONG  seed 1: P(Np=1)=%.3f  | thirds: %.3f %.3f %.3f\n",
        mean(mL .== 1),
        mean(mL[1:ni÷3,:].==1), mean(mL[ni÷3+1:2ni÷3,:].==1), mean(mL[2ni÷3+1:end,:].==1))
@printf("\nexpect P(Np1)=%.3f (logit %+.2f). If LONG thirds keep dropping → MIXING; flat&high → HASTINGS\n",
        exp(Δz)/(1+exp(Δz)), Δz)
