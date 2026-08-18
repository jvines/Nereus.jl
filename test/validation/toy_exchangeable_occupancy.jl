#!/usr/bin/env julia
# THE ZERO-AMBIGUITY OCCUPANCY BENCHMARK.
#
# Every other occupancy gate compares the chain against a REFERENCE ΔlogZ that
# is itself estimated by sampling, so a discrepancy could always be blamed on
# the reference. This one needs no reference at all.
#
# Two ActivityDecorrelation models on the SAME indicator, distinguished only by
# a label (so they get disjoint parameter slots), placed in one mutual-exclusion
# group. They are the IDENTICAL model. Their evidences are equal by
# construction, exactly, with no estimation involved. Therefore
#
#     P(A) = P(B) = 0.5
#
# is not an approximation — it is forced by symmetry. Any deviation is sampler
# bias and nothing else, and the size of the deviation IS the size of the bias.
#
# Why this matters: the AD↔GP-Rot gates leave a residual (1.56 nats at best
# swap rate) that is hard to attribute — reference error, mixing, or a genuine
# q-ratio bug all look alike. Here there is nothing to attribute it to.
#
# Reports the split, the implied bias in nats, and whether the chain is simply
# frozen (never switching) versus switching but unbalanced — different diseases.
using Nereus, MCMCChains, Statistics, Printf, Random

const N, SIG, P_ROT = 60, 1.5, 8.0
rng = MersenneTwister(20260810)
t   = sort(rand(rng, N)) .* 70.0
act = 5.0 .* (sin.(2π .* t ./ P_ROT) .+ 0.4 .* sin.(4π .* t ./ P_ROT .+ 0.5))
rv  = act .+ SIG .* randn(rng, N)
bis = act .+ 1.5 .* randn(rng, N)
data = Data(t_rv=t, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
            indicators=Dict("bis"=>bis), indicator_errs=Dict("bis"=>fill(1.0,N)))

# Same indicator, same structure, different label ⇒ disjoint slots, identical model.
a1 = ActivityDecorrelation(indicators=["bis"], label="A")
a2 = ActivityDecorrelation(indicators=["bis"], label="B")
rvmax = maximum(abs, rv)
params = Params(; max_kplanet=0, planet_modes=PlanetDataSources[],
    instruments=InstrumentConfig(rv=["I1"]), data=data,
    priors=Dict{String,PriorSpec}("gamma_I1"=>UniformPrior(-3rvmax,3rvmax),
                                   "sigma_I1"=>LogUniformPrior(0.2,12.0)),
    noise_models=NoiseModel[a1,a2], transdim_noise=true, stability=:none)
tgt = NereusTarget(params, data)
td  = TransDimConfig(; max_kplanet=0, planets=false, noise=true,
        toggleable=NoiseModel[a1,a2], noise_exclusion_groups=[NoiseModel[a1,a2]])

function run_one(; rate, bridge, seed)
    r = sample_transdim_ptemcee(tgt, data; td=td, n_temps=16, n_walkers=60,
            n_steps=12000, n_burnin=4000, n_birth_tries=10, n_birth_refine=15,
            seed=seed, noise_swap = rate > 0, noise_swap_rate = rate,
            n_noise_bridge = bridge, n_noise_relax = 2, show_progress=false)
    x1 = vec(Array(r.chains[:noise_active_1])) .> 0.5
    x2 = vec(Array(r.chains[:noise_active_2])) .> 0.5
    p1, p2 = mean(x1), mean(x2)
    # how often the active member actually changes — frozen vs unbalanced
    lab = [a ? 1 : (b ? 2 : 0) for (a,b) in zip(x1,x2)]
    switches = count(i -> lab[i] != lab[i-1] && lab[i] != 0 && lab[i-1] != 0,
                     2:length(lab))
    return p1, p2, switches, length(lab)
end

println("Two labelled copies of ONE model. Symmetry forces P(A)=P(B)=0.5.\n")
@printf("%-6s %-7s %-9s %-9s %-10s %-9s\n",
        "rate","bridge","P(A)","P(B)","bias[nat]","switches")

# Collected rather than accumulated in a top-level loop: assigning to an outer
# name from inside a top-level `for` hits Julia's soft-scope rule and throws.
rows = Tuple{Float64,Int,Float64,Float64,Float64,Int}[]
for bridge in (1, 8), rate in (0.0, 0.6)
    ps1 = Float64[]; ps2 = Float64[]; sw = 0
    for sd in 1:3
        p1, p2, s, _ = run_one(rate=rate, bridge=bridge, seed=sd)
        push!(ps1, p1); push!(ps2, p2); sw += s
    end
    p1, p2 = mean(ps1), mean(ps2)
    d = clamp(p1 + p2, 1e-9, 1.0)
    frac = clamp(p1 / d, 1e-6, 1 - 1e-6)
    bias = abs(log(frac / (1 - frac)))
    push!(rows, (rate, bridge, p1, p2, bias, sw))
    @printf("%-6.2f %-7d %-9.4f %-9.4f %-10.2f %-9d\n",
            rate, bridge, p1, p2, bias, sw)
end
worst = maximum(r[5] for r in rows)

println()
if worst < 0.5
    println("✅ BALANCED — occupancy respects the symmetry (bias < 0.5 nat)")
    exit(0)
else
    @printf("❌ ASYMMETRIC — up to %.2f nats of bias between IDENTICAL models.\n", worst)
    println("   No reference is involved, so this is sampler bias, full stop.")
    println("   `switches` distinguishes the disease: ~0 means the chain is")
    println("   frozen in whichever member it entered; large means it moves")
    println("   but spends the wrong amount of time in each.")
    exit(1)
end
