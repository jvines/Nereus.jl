#!/usr/bin/env julia
# DB gate for the HARD exclusion case that fails on HD 18599: AD ↔ GP-Rot.
#
# toy_exclusion_swap_db_gate.jl (AD-bis ↔ AD-fwhm) PASSES — but BOTH members
# there use the OLS-informed AD birth. The HD 18599 failure pits AD against
# CeleriteRotation, whose birth is a BLIND PRIOR DRAW (proposals.jl line 478):
# a prior-drawn 5-param quasi-periodic GP almost never lands near the rotation
# mode, so the chain can't switch INTO GP-Rot post-burn-in and the occupancy
# under-represents GP-Rot vs its evidence.
#
# This toy reproduces that at small scale: activity = quasi-periodic (P_rot≈8d)
# so GP-Rot fits it, AND bis ∝ activity so AD fits it linearly. Mutual exclusion.
#
# DIAGNOSTIC (no move #1 yet): if occupancy ≠ evidence here, the blind GP birth
# is the culprit and the informed GP-Rot proposal (move #1) is required.
# Later, with move #1 enabled, this same gate must PASS (occupancy = evidence).
using Nereus, MCMCChains, Statistics, Printf, Random

rng = MersenneTwister(20260619); const N = 70
const Tt = sort(rand(rng, N)) .* 80.0; const SIG = 1.5
const P_ROT = 8.0
# quasi-periodic activity: fundamental + harmonic, slowly modulated amplitude
amp = 5.0 .* (1.0 .+ 0.3 .* sin.(2π .* Tt ./ 40.0))
act = amp .* (sin.(2π .* Tt ./ P_ROT) .+ 0.4 .* sin.(4π .* Tt ./ P_ROT .+ 0.5))
rv  = act .+ SIG .* randn(rng, N)
# MODEST-GAP design: bis tracks the activity with moderate scatter so AD and
# GP-Rot are CLOSE in evidence (a few nats) ⇒ occupancy lands at an intermediate
# value (no saturation) and the gate can quantitatively check occupancy=evidence
# for the GP model — the paper-grade extension of the AD↔AD gate to CeleriteRotation.
# Tracer noise sets the AD↔GP-Rot evidence gap. It is NOT hardcoded: a fixed
# level is exactly how this gate broke. The generator's RNG, N, or activity
# model changing silently moves the gap, and at |ΔlogZ| ≳ 10 the disfavoured
# model drops below the chain's resolution — at which point the test reports
# FAIL whatever the sampler does (a 13.7-nat gap was misread here as a 6.79-nat
# bias, when 6.79 was just clamp-minus-truth). The level the sibling script
# uses does not transfer either: a different time sampling flips the gap's SIGN.
#
# So scan for a level that puts |ΔlogZ| in a band that is both non-saturated
# and comfortably measurable, using short cheap chains, then run the real
# 3-seed reference there.
const TARGET_LO, TARGET_HI = 1.0, 4.0

mkdata(bn) = (r = MersenneTwister(20260619);
    tt = sort(rand(r, N)) .* 80.0;
    a = 5.0 .* (1.0 .+ 0.3 .* sin.(2π .* tt ./ 40.0));
    ac = a .* (sin.(2π .* tt ./ P_ROT) .+ 0.4 .* sin.(4π .* tt ./ P_ROT .+ 0.5));
    v = ac .+ SIG .* randn(r, N);
    b = ac .+ bn .* randn(r, N);
    Data(t_rv=tt, rv=v, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
         indicators=Dict("bis"=>b), indicator_errs=Dict("bis"=>fill(1.0,N))))

ad  = ActivityDecorrelation(indicators=["bis"])
rot = CeleriteRotation(channel=:rv)

# Only the always-present RV params; GP-Rot's gp_* priors are auto-filled from
# _default_noise_priors!(::CeleriteRotation) with the correct (suffixed) names.
basepriors(d) = (rvmax = maximum(abs, d.rv); Dict{String,PriorSpec}(
    "gamma_I1"=>UniformPrior(-3rvmax,3rvmax),
    "sigma_I1"=>LogUniformPrior(0.2,12.0)))

mkparams(d, noise_models; transdim) = Params(; max_kplanet=0, planet_modes=PlanetDataSources[],
    instruments=InstrumentConfig(rv=["I1"]), data=d,
    parametrization=ParametrizationConfig(time=:Mo), priors=basepriors(d),
    noise_models=noise_models, transdim_noise=transdim, stability=:none)

# ---- scan the tracer noise for a resolvable, non-saturated gap ----
zq(d, nms) = sample_ptemcee(NereusTarget(mkparams(d, nms; transdim=false), d), d;
        n_temps=10, n_walkers=40, n_steps=4000, n_burnin=1500, seed=1,
        init_strategy=:prior, show_progress=false).log_evidence
BN = let best = nothing, bestd = Inf
    for bn in (1.2, 1.4, 1.55, 1.7, 1.9)
        d = mkdata(bn)
        Δ = zq(d, NoiseModel[ad]) - zq(d, NoiseModel[rot])
        @printf("  scan bis_noise=%.2f  ΔlogZ(AD−Rot)=%+.2f\n", bn, Δ)
        # prefer inside the target band; otherwise nearest to its midpoint
        score = TARGET_LO <= abs(Δ) <= TARGET_HI ? 0.0 :
                abs(abs(Δ) - (TARGET_LO + TARGET_HI)/2)
        if score < bestd; bestd = score; best = bn; end
    end
    best
end
@printf("\nselected bis_noise = %.2f\n\n", BN)
data = mkdata(BN)

# ---- Reference: separate fixed-config evidences (3 seeds) ----
za = Float64[]; zr = Float64[]
for sd in 1:3
    ra = sample_ptemcee(NereusTarget(mkparams(data, NoiseModel[ad]; transdim=false), data), data;
            n_temps=14, n_walkers=60, n_steps=9000, n_burnin=3500, seed=sd,
            init_strategy=:prior, show_progress=false)
    rr = sample_ptemcee(NereusTarget(mkparams(data, NoiseModel[rot]; transdim=false), data), data;
            n_temps=14, n_walkers=60, n_steps=9000, n_burnin=3500, seed=sd,
            init_strategy=:prior, show_progress=false)
    push!(za, ra.log_evidence); push!(zr, rr.log_evidence)
    @printf("  seed %d: logZ(AD)=%.2f  logZ(GP-Rot)=%.2f  Δ(AD−Rot)=%+.2f\n",
            sd, ra.log_evidence, rr.log_evidence, ra.log_evidence - rr.log_evidence)
end
Δz  = mean(za) - mean(zr)
Δze = sqrt(std(za)^2 + std(zr)^2)/sqrt(3)
@printf("\nReference ΔlogZ (AD − GP-Rot) = %+.2f ± %.2f (3-seed)\n", Δz, Δze)

# ---- Trans-dim: AD ↔ GP-Rot in ONE exclusion group ----
pT = mkparams(data, NoiseModel[ad, rot]; transdim=true)
td = TransDimConfig(; max_kplanet=0, planets=false, noise=true,
                      toggleable=NoiseModel[ad, rot],
                      noise_exclusion_groups=[NoiseModel[ad, rot]])
obs = Float64[]; nvis = Int[]; nsamps = Int[]
for sd in 1:3
    res = sample_transdim_ptemcee(NereusTarget(pT,data), data; td=td,
            n_temps=20, n_walkers=80, n_steps=14000, n_burnin=5000,
            n_birth_tries=10, n_birth_refine=15, seed=sd, show_progress=false)
    cn = names(res.chains, :parameters)
    a_ad  = :noise_active_1 in cn ? vec(Array(res.chains[:noise_active_1])) .> 0.5 : falses(0)
    a_rot = :noise_active_2 in cn ? vec(Array(res.chains[:noise_active_2])) .> 0.5 : falses(0)
    pad = mean(a_ad); prot = mean(a_rot)
    nsamp = max(length(a_ad), length(a_rot))
    push!(nvis, min(count(a_ad), count(a_rot)))   # visits to the RARER member
    push!(nsamps, nsamp)
    denom = clamp(pad + prot, 1e-3, 1.0)
    padc = clamp(pad/denom, 1e-3, 1-1e-3)
    push!(obs, log(padc/(1-padc)))
    @printf("  td seed %d: P(AD)=%.3f  P(GP-Rot)=%.3f  logit=%+.2f  (rarer visited %d/%d)\n",
            sd, pad, prot, log(padc/(1-padc)), nvis[end], nsamp)
end
lo = mean(obs)
@printf("\ntrans-dim logit(P(AD)/P(GP-Rot)) = %+.2f ± %.2f\n", lo, std(obs)/sqrt(3))
@printf("predicted (=ΔlogZ)              = %+.2f ± %.2f   discrepancy = %.2f nats\n",
        Δz, Δze, abs(lo - Δz))

# ---- RESOLUTION GUARD ------------------------------------------------------
# An occupancy-vs-evidence test can only measure what the chain can visit. The
# rarer member is expected in a fraction 1/(1+exp|Δz|) of samples, i.e. λ
# expected visits in nsamp draws. When λ ≲ 1 a CORRECT sampler and one that can
# never enter the mode produce the IDENTICAL observation — zero visits — and
# the logit lands on whatever floor the clamp imposes. The script then prints
# FAIL regardless of the code, which is a permanent false alarm: it trains you
# to ignore a red result, and it cost real time here (a 13.7-nat gap was read
# as a 6.79-nat "bias" when 6.79 was just clamp-minus-truth).
#
# So: verdict only when the rare member is resolvable. Otherwise say so and
# say what to change.
const MIN_VISITS = 30
λ_expected = minimum(nsamps) / (1 + exp(abs(Δz)))
observed   = minimum(nvis)
@printf("\nresolution: rarer expected ≈%.1f visits, observed %d (need ≥%d for a verdict)\n",
        λ_expected, observed, MIN_VISITS)

if λ_expected < MIN_VISITS
    @printf("""
⚠️  UNRESOLVABLE — the %.2f-nat gap puts the disfavoured model below this
    chain's resolution, so NO sampler behaviour can be distinguished here and
    the discrepancy above is an artifact of the logit clamp, not a measurement.
    Fix by shrinking the gap (raise the tracer noise) or lengthening the chain
    to ≳%.0f samples. Not a code failure.
""", abs(Δz), MIN_VISITS * (1 + exp(abs(Δz))))
    exit(2)
end

if abs(lo - Δz) < 3*max(Δze, 0.3) + 1.0
    println("✅ AD↔GP-Rot occupancy reproduces evidence")
    exit(0)
else
    println("❌ FAIL — AD↔GP-Rot occupancy ≠ evidence " *
            "(gap is resolvable, so this IS a real bias)")
    exit(1)
end
