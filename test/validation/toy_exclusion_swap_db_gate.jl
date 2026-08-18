#!/usr/bin/env julia
# DB gate for the EXCLUSION-GROUP noise swap (A ↔ B, both fit the activity).
#
# toy_noise_model_selection.jl validates the EASY case: white ↔ MA(1), a single
# toggle where the off-state (white) is cheap to reach. The HD 18599 failure is
# the HARD case: two models (AD, GP-Rot, …) in a MUTUAL-EXCLUSION group that BOTH
# absorb the same activity signal. Switching A→B must cross the "none" valley
# (kill a FITTED incumbent, climb a challenger) — death/birth can't at β≈1, so
# whichever model a walker lands in during burn-in ENTRENCHES, and the recorded
# occupancy ≠ P(M|D).
#
# This gate builds the minimal hard case: AD-on-`bis` vs AD-on-`fwhm`, where the
# activity is correlated with BOTH indicators (bis a bit tighter ⇒ AD-bis wins by
# a COMPUTABLE evidence margin). Both members use the SAME OLS-informed birth, so
# it isolates the SWAP MECHANISM + its q-ratio (no GP proposal needed yet).
#
# PASS: logit(P(AD-bis) / P(AD-fwhm)) ≈ ΔlogZ(bis − fwhm) within a few × MC error.
# FAIL: the exclusion swap biases occupancy — occupancy is NOT P(M|D).
using Nereus, MCMCChains, Statistics, Printf, Random

rng = MersenneTwister(20260619); const N = 80
const Tt = sort(rand(rng, N)) .* 90.0; const SIG = 2.0
# one activity signal; bis tracks it tightly, fwhm more loosely (extra noise)
act  = 6.0 .* sin.(2π .* Tt ./ 13.0 .+ 0.6)
rv   = act .+ SIG .* randn(rng, N)
bis  = 12.0 .* sin.(2π .* Tt ./ 13.0 .+ 0.6) .+ 0.10 .* randn(rng, N)   # tight ⇒ AD-bis fits better
fwhm = 12.0 .* sin.(2π .* Tt ./ 13.0 .+ 0.6) .+ 1.20 .* randn(rng, N)   # loose
data = Data(t_rv=Tt, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
            indicators=Dict("bis"=>bis, "fwhm"=>fwhm),
            indicator_errs=Dict("bis"=>fill(1.0,N), "fwhm"=>fill(1.0,N)))
rvmax = maximum(abs, rv)

ad_bis  = ActivityDecorrelation(indicators=["bis"])
ad_fwhm = ActivityDecorrelation(indicators=["fwhm"])

# planet fixed at Np=0 (no Keplerian) ⇒ the ONLY thing the models do is absorb
# the activity, so the head-to-head is pure noise-model selection.
basepriors() = Dict{String,PriorSpec}(
    "gamma_I1"=>UniformPrior(-3rvmax,3rvmax),
    "sigma_I1"=>LogUniformPrior(0.2,12.0))

mkparams(noise_models; transdim) = Params(; max_kplanet=0, planet_modes=PlanetDataSources[],
    instruments=InstrumentConfig(rv=["I1"]), data=data,
    parametrization=ParametrizationConfig(time=:Mo), priors=basepriors(),
    noise_models=noise_models, transdim_noise=transdim, stability=:none)

# ---- Reference: two separate fixed-config evidences (3 seeds) ----
zb = Float64[]; zf = Float64[]
for sd in 1:3
    rb = sample_ptemcee(NereusTarget(mkparams(NoiseModel[ad_bis]; transdim=false), data), data;
            n_temps=12, n_walkers=60, n_steps=8000, n_burnin=3000, seed=sd,
            init_strategy=:prior, show_progress=false)
    rf = sample_ptemcee(NereusTarget(mkparams(NoiseModel[ad_fwhm]; transdim=false), data), data;
            n_temps=12, n_walkers=60, n_steps=8000, n_burnin=3000, seed=sd,
            init_strategy=:prior, show_progress=false)
    push!(zb, rb.log_evidence); push!(zf, rf.log_evidence)
    @printf("  seed %d: logZ(AD-bis)=%.2f  logZ(AD-fwhm)=%.2f  Δ=%+.2f\n",
            sd, rb.log_evidence, rf.log_evidence, rb.log_evidence - rf.log_evidence)
end
Δz  = mean(zb) - mean(zf)
Δze = sqrt(std(zb)^2 + std(zf)^2)/sqrt(3)
@printf("\nReference ΔlogZ (bis − fwhm) = %+.2f ± %.2f (3-seed)\n", Δz, Δze)

# ---- Trans-dim: AD-bis ↔ AD-fwhm in ONE exclusion group ----
pT = mkparams(NoiseModel[ad_bis, ad_fwhm]; transdim=true)
td = TransDimConfig(; max_kplanet=0, planets=false, noise=true,
                      toggleable=NoiseModel[ad_bis, ad_fwhm],
                      noise_exclusion_groups=[NoiseModel[ad_bis, ad_fwhm]])
obs = Float64[]
for sd in 1:3
    res = sample_transdim_ptemcee(NereusTarget(pT,data), data; td=td,
            n_temps=20, n_walkers=80, n_steps=12000, n_burnin=4000,
            n_birth_tries=8, n_birth_refine=10, seed=sd, show_progress=false)
    cn = names(res.chains, :parameters)
    a_bis  = :noise_active_1 in cn ? vec(Array(res.chains[:noise_active_1])) .> 0.5 : falses(0)
    a_fwhm = :noise_active_2 in cn ? vec(Array(res.chains[:noise_active_2])) .> 0.5 : falses(0)
    pb = mean(a_bis); pf = mean(a_fwhm)
    # condition on SOME group member active (exclusion ⇒ pb+pf≈1)
    denom = clamp(pb + pf, 1e-3, 1.0)
    pbc = clamp(pb/denom, 1e-3, 1-1e-3)
    push!(obs, log(pbc/(1-pbc)))
    @printf("  td seed %d: P(AD-bis)=%.3f  P(AD-fwhm)=%.3f  logit(bis|grp)=%+.2f\n",
            sd, pb, pf, log(pbc/(1-pbc)))
end
lo = mean(obs)
@printf("\ntrans-dim logit(P(bis)/P(fwhm)) = %+.2f ± %.2f\n", lo, std(obs)/sqrt(3))
@printf("predicted (=ΔlogZ)             = %+.2f ± %.2f   discrepancy = %.2f nats\n",
        Δz, Δze, abs(lo - Δz))
println(abs(lo - Δz) < 3*max(Δze, 0.3) + 0.7 ?
        "✅ EXCLUSION SWAP DB-CORRECT — occupancy reproduces the evidence ratio" :
        "❌ FAIL — exclusion occupancy ≠ evidence (entrenchment / biased swap)")
