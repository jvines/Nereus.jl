#!/usr/bin/env julia
# IS THE 1.6-NAT AD<->GP-Rot RESIDUAL IN THE SAMPLER, OR IN THE REFERENCE?
#
# Everything sampler-side has now been eliminated:
#   * two IDENTICAL models split 0.5082/0.4918 (0.03 nat) with ~720k switches,
#     needing no reference at all -> no q-ratio bug, no entrenchment;
#   * blind-birth bias measured against dimension tops out at 0.148 nat (K=6),
#     and the annealed birth cuts that to 0.050 and flattens it;
#   * on AD<->GP-Rot the discrepancy sits at ~1.6 nat and does not move for ANY
#     sampler-side change -- swap on/off, rate 0/0.6/1.0, bridge 1/8.
#
# A floor immune to every sampler intervention is what a biased REFERENCE looks
# like. The reference is a PT evidence estimate, and every gate compares against
# it, so if it is off by ~1.6 nat on the GP model then the occupancy was right
# all along and the gates have been accusing the wrong component.
#
# This checks it the only way that does not beg the question: the SAME PT run
# reports four independent estimators (TI, TI+, SS+, H+) plus a mode-anchored
# Laplace cross-check. On the analytic 2D-Gauss x box they agree to <0.01 nat.
# If they agree here too, the reference is sound and the residual is real. If
# they scatter by ~1.6 nat on GP-Rot while agreeing on AD, the reference is the
# culprit -- and the spread across estimators is the error bar the gates should
# have been using instead of the 3-seed spread.
using Nereus, Statistics, Printf, Random

const N, SIG, P_ROT = 70, 1.5, 8.0
rng = MersenneTwister(20260619)
Tt  = sort(rand(rng, N)) .* 80.0
amp = 5.0 .* (1.0 .+ 0.3 .* sin.(2π .* Tt ./ 40.0))
act = amp .* (sin.(2π .* Tt ./ P_ROT) .+ 0.4 .* sin.(4π .* Tt ./ P_ROT .+ 0.5))
rv  = act .+ SIG .* randn(rng, N)
bis = act .+ 2.0 .* randn(rng, N)          # the validate_noise_swap level
data = Data(t_rv=Tt, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
            indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))
rvmax = maximum(abs, rv)

ad  = ActivityDecorrelation(indicators=["bis"])
rot = CeleriteRotation(channel=:rv)
mkp(nms) = Params(; max_kplanet=0, planet_modes=PlanetDataSources[],
    instruments=InstrumentConfig(rv=["I1"]), data=data,
    parametrization=ParametrizationConfig(time=:Mo),
    priors=Dict{String,PriorSpec}("gamma_I1"=>UniformPrior(-3rvmax,3rvmax),
                                   "sigma_I1"=>LogUniformPrior(0.2,12.0)),
    noise_models=nms, transdim_noise=false, stability=:none)

function report(label, nms)
    # Every estimator returns (value, its own standard error) — which is the
    # quantity the gates should have been using as the reference uncertainty,
    # instead of the 3-seed scatter of a single estimator.
    vals = Dict(k => Float64[] for k in (:ti,:ti_plus,:ss_plus,:hybrid,:lap))
    errs = Dict(k => Float64[] for k in (:ti,:ti_plus,:ss_plus,:hybrid))
    for sd in 1:3
        r = sample_ptemcee(NereusTarget(mkp(nms), data), data;
                n_temps=14, n_walkers=60, n_steps=9000, n_burnin=3500,
                seed=sd, init_strategy=:prior, show_progress=false)
        e = r.evidence
        for k in (:ti,:ti_plus,:ss_plus,:hybrid)
            v, er = getfield(e, k)
            push!(vals[k], v); push!(errs[k], er)
        end
        push!(vals[:lap], r.log_evidence_laplace)
    end
    m(k) = mean(vals[k])
    @printf("%-8s TI=%+8.2f  TI+=%+8.2f  SS+=%+8.2f  H+=%+8.2f  Laplace=%+8.2f\n",
            label, m(:ti), m(:ti_plus), m(:ss_plus), m(:hybrid), m(:lap))
    @printf("%-8s  own stderr:  %.2f      %.2f       %.2f       %.2f\n",
            "", mean(errs[:ti]), mean(errs[:ti_plus]),
            mean(errs[:ss_plus]), mean(errs[:hybrid]))
    four = [m(:ti), m(:ti_plus), m(:ss_plus), m(:hybrid)]
    @printf("%-8s  spread across estimators = %.2f nat;  3-seed scatter of H+ = %.2f\n",
            "", maximum(four)-minimum(four), std(vals[:hybrid]))
    return (ti=m(:ti), tip=m(:ti_plus), ss=m(:ss_plus), hy=m(:hybrid),
            lap=m(:lap), maxerr=maximum(mean(errs[k]) for k in keys(errs)))
end

println("Per-model evidence, four estimators from the SAME runs.\n")
A = report("AD",     NoiseModel[ad])
R = report("GP-Rot", NoiseModel[rot])

println()
for (nm, f) in (("TI",:ti), ("TI+",:tip), ("SS+",:ss), ("H+",:hy), ("Laplace",:lap))
    @printf("  dlogZ(AD - GP-Rot) by %-8s = %+6.2f\n", nm,
            getfield(A, f) - getfield(R, f))
end
ds = [getfield(A,f) - getfield(R,f) for f in (:ti,:tip,:ss,:hy)]
@printf("\nreference spread across estimators = %.2f nat\n", maximum(ds)-minimum(ds))
@printf("largest self-reported estimator error: AD %.2f, GP-Rot %.2f nat\n",
        A.maxerr, R.maxerr)
println("occupancy said dlogZ ~ -0.1 (rate 0.6); the gate assumed +1.62.\n")
if maximum(ds) - minimum(ds) > 1.0 || max(A.maxerr, R.maxerr) > 1.0
    println("✅ THE REFERENCE IS THE PROBLEM — estimators disagree by more than")
    println("   the discrepancy the gates were blaming on the sampler.")
    exit(0)
else
    println("❌ Reference is consistent — the residual is real and sampler-side.")
    exit(1)
end
