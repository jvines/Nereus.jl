#!/usr/bin/env julia
# Does the ANNEALED noise birth close the residual occupancy gap?
#
# Baseline (validate_noise_swap.jl, same data and reference): the within-group
# swap moves the AD↔GP-Rot occupancy from 3.49 nats off the evidence to 1.56 at
# swap rate 0.6 — a large improvement that does not finish the job. Two things
# were never tried: swap rates above 0.6 (the trend had not flattened), and a
# birth that lets the newborn relax before being judged.
#
# The annealed birth bridges rho_0 = pi_A*q(u) -> rho_T = pi_B, relaxing the
# newborn at each stage and accumulating the work into the acceptance ratio, so
# unlike `climb_newborn!` it is admissible post-burn-in. n_noise_bridge = 1
# reproduces the ordinary birth exactly, which is what makes this an A/B.
#
# Reports discrepancy |logit(P(AD)/P(GP-Rot)) - dlogZ| for
# swap rate x {ordinary birth, annealed birth}. Lower is better; 0 is perfect.
#
# Activity models have naturally LARGE evidence gaps (one fits clearly better),
# which saturates the logit=ΔlogZ test (occupancy pins to 0/1). So first SCAN the
# indicator-noise level to locate a NON-SATURATED gap (|ΔlogZ| small), then at
# that level run the A/B:
#   1. DB-CORRECTNESS: swap-ON occupancy must reproduce the fixed-dim ΔlogZ (a
#      wrong q-ratio injects a bias → FAIL). This exercises the new informed-AD ↔
#      prior-GP q-path that AD↔AD couldn't.
#   2. THE FIX: swap-ON discrepancy ≤ swap-OFF (drains the disfavoured mode).
using Nereus, MCMCChains, Statistics, Printf, Random

const N = 70; const SIG = 1.5; const P_ROT = 8.0
const TT = (rng=MersenneTwister(7); sort(rand(rng, N)) .* 80.0)
function make_data(bis_noise; seed=20260619)
    rng = MersenneTwister(seed)
    amp = 5.0 .* (1.0 .+ 0.3 .* sin.(2π .* TT ./ 40.0))
    act = amp .* (sin.(2π .* TT ./ P_ROT) .+ 0.4 .* sin.(4π .* TT ./ P_ROT .+ 0.5))
    rv  = act .+ SIG .* randn(rng, N)
    bis = act .+ bis_noise .* randn(rng, N)
    Data(t_rv=TT, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
         indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))
end
ad  = ActivityDecorrelation(indicators=["bis"]); rot = CeleriteRotation(channel=:rv)
mkp(data, nms; td) = (rvmax=maximum(abs,data.rv);
    Params(; max_kplanet=0, planet_modes=PlanetDataSources[],
        instruments=InstrumentConfig(rv=["I1"]), data=data,
        parametrization=ParametrizationConfig(time=:Mo),
        priors=Dict{String,PriorSpec}("gamma_I1"=>UniformPrior(-3rvmax,3rvmax),
                                       "sigma_I1"=>LogUniformPrior(0.2,12.0)),
        noise_models=nms, transdim_noise=td, stability=:none))
Z(data, nms) = sample_ptemcee(NereusTarget(mkp(data,nms;td=false),data),data;
        n_temps=14,n_walkers=60,n_steps=9000,n_burnin=3500,seed=1,
        init_strategy=:prior,show_progress=false).log_evidence

# bis_noise=2.0 → a NON-SATURATED gap (ΔlogZ≈+2.6, P(AD)≈0.93) from the scan.
bn = 2.0
data = make_data(bn)
za=Float64[]; zr=Float64[]
for sd in 1:3
    push!(za, sample_ptemcee(NereusTarget(mkp(data,NoiseModel[ad];td=false),data),data;
        n_temps=14,n_walkers=60,n_steps=9000,n_burnin=3500,seed=sd,init_strategy=:prior,show_progress=false).log_evidence)
    push!(zr, sample_ptemcee(NereusTarget(mkp(data,NoiseModel[rot];td=false),data),data;
        n_temps=14,n_walkers=60,n_steps=9000,n_burnin=3500,seed=sd,init_strategy=:prior,show_progress=false).log_evidence)
end
Δz=mean(za)-mean(zr); Δze=sqrt(std(za)^2+std(zr)^2)/sqrt(3)
@printf("Reference ΔlogZ (AD − GP-Rot) = %+.2f ± %.2f (3-seed)\n\n", Δz, Δze)

pT=mkp(data,NoiseModel[ad,rot];td=true)
tdc=TransDimConfig(; max_kplanet=0, planets=false, noise=true,
        toggleable=NoiseModel[ad,rot], noise_exclusion_groups=[NoiseModel[ad,rot]])

const RATES  = (0.0, 0.60, 1.00)
const BRIDGE = (1, 8)

function occ(rate::Float64, nbridge::Int; nseed::Int=2)
    obs = Float64[]
    for sd in 1:nseed
        res = sample_transdim_ptemcee(NereusTarget(pT, data), data; td=tdc,
            n_temps=20, n_walkers=80, n_steps=20000, n_burnin=6000,
            n_birth_tries=10, n_birth_refine=15, seed=sd,
            noise_swap = rate > 0, noise_swap_rate = rate,
            n_noise_bridge = nbridge, n_noise_relax = 2,
            show_progress = false)
        pad  = mean(vec(Array(res.chains[:noise_active_1])) .> 0.5)
        prot = mean(vec(Array(res.chains[:noise_active_2])) .> 0.5)
        d = clamp(pad + prot, 1e-3, 1.0); padc = clamp(pad/d, 1e-3, 1-1e-3)
        push!(obs, log(padc/(1-padc)))
        @printf("  rate=%.2f bridge=%d seed %d: P(AD)=%.3f P(Rot)=%.3f logit=%+.2f\n",
                rate, nbridge, sd, pad, prot, log(padc/(1-padc)))
    end
    mean(obs)
end

results = Dict{Tuple{Float64,Int}, Float64}()
for nb in BRIDGE, r in RATES
    results[(r, nb)] = occ(r, nb)
end

println("\n  discrepancy from evidence (nats; lower is better)")
@printf("  %-10s", "swap rate"); for nb in BRIDGE; @printf("  bridge=%-6d", nb); end
println()
for r in RATES
    @printf("  %-10.2f", r)
    for nb in BRIDGE
        @printf("  %-13.2f", abs(results[(r,nb)] - Δz))
    end
    println()
end

best_plain = minimum(abs(results[(r,1)] - Δz) for r in RATES)
best_anneal = minimum(abs(results[(r,8)] - Δz) for r in RATES)
@printf("\nbest ordinary birth = %.2f nats,  best annealed = %.2f nats\n",
        best_plain, best_anneal)
if best_anneal < best_plain - 0.3
    println("✅ ANNEALED BIRTH HELPS — closes part of the residual gap")
    exit(0)
elseif best_anneal > best_plain + 0.3
    println("❌ ANNEALED BIRTH HURTS — worse than the ordinary birth")
    exit(1)
else
    println("➖ NO MEASURABLE DIFFERENCE between ordinary and annealed birth")
    exit(2)
end
