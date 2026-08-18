#!/usr/bin/env julia
# HOW BADLY DOES A BLIND BIRTH SCALE WITH DIMENSION?
#
# toy_exchangeable_occupancy.jl showed the trans-dim plumbing is sound: two
# IDENTICAL models split 0.5082/0.4918 with ~720k switches, even with no swap
# and no bridge. So there is no q-ratio bug and no structural entrenchment.
#
# But that benchmark has a blind spot — both members are ActivityDecorrelation,
# so BOTH get the OLS-informed birth. It never exercises the thing blocker 3 is
# about: births that have to guess many parameters blind.
#
# This closes it while keeping the zero-ambiguity property. Two AD models on the
# SAME K indicators, distinguished only by a label, so they are the identical
# model and symmetry forces P(A) = P(B) = 0.5 exactly — no reference evidence
# anywhere. Then the OLS-informed birth is switched OFF, so both births are
# blind K-dimensional prior draws, and K is swept.
#
# Everything except the birth dimension is held fixed and symmetric. Whatever
# curve comes out is the blind-birth penalty as a function of dimension, which
# is the number the noise menu actually needs: its GP members carry 3-5
# parameters and ActivityGP carries 8.
using Nereus, MCMCChains, Statistics, Printf, Random

const N, SIG, P_ROT = 60, 1.5, 8.0
const KS = (1, 3, 6)

function build(K::Int)
    rng = MersenneTwister(20260811)
    t   = sort(rand(rng, N)) .* 70.0
    act = 5.0 .* (sin.(2π .* t ./ P_ROT) .+ 0.4 .* sin.(4π .* t ./ P_ROT .+ 0.5))
    rv  = act .+ SIG .* randn(rng, N)
    inds = Dict{String,Vector{Float64}}()
    errs = Dict{String,Vector{Float64}}()
    for k in 1:K
        inds["i$k"] = act .+ 1.5 .* randn(rng, N)
        errs["i$k"] = fill(1.0, N)
    end
    data = Data(t_rv=t, rv=rv, rv_err=fill(SIG,N), rv_inst=ones(Int,N),
                indicators=inds, indicator_errs=errs)
    names = ["i$k" for k in 1:K]
    a1 = ActivityDecorrelation(indicators=names, label="A")
    a2 = ActivityDecorrelation(indicators=names, label="B")
    rvmax = maximum(abs, rv)
    params = Params(; max_kplanet=0, planet_modes=PlanetDataSources[],
        instruments=InstrumentConfig(rv=["I1"]), data=data,
        priors=Dict{String,PriorSpec}("gamma_I1"=>UniformPrior(-3rvmax,3rvmax),
                                       "sigma_I1"=>LogUniformPrior(0.2,12.0)),
        noise_models=NoiseModel[a1,a2], transdim_noise=true, stability=:none)
    td = TransDimConfig(; max_kplanet=0, planets=false, noise=true,
            toggleable=NoiseModel[a1,a2], noise_exclusion_groups=[NoiseModel[a1,a2]])
    return NereusTarget(params, data), data, td
end

function measure(K::Int; informed::Bool, bridge::Int, rate::Float64)
    tgt, data, td = build(K)
    Nereus.AD_INFORMED_BIRTH[] = informed
    ps = Float64[]; sw = 0
    try
        for sd in 1:3
            r = sample_transdim_ptemcee(tgt, data; td=td, n_temps=16,
                    n_walkers=60, n_steps=12000, n_burnin=4000,
                    n_birth_tries=10, n_birth_refine=15, seed=sd,
                    noise_swap = rate > 0, noise_swap_rate = rate,
                    n_noise_bridge = bridge, n_noise_relax = 2,
                    show_progress=false)
            x1 = vec(Array(r.chains[:noise_active_1])) .> 0.5
            x2 = vec(Array(r.chains[:noise_active_2])) .> 0.5
            p1, p2 = mean(x1), mean(x2)
            d = clamp(p1+p2, 1e-9, 1.0)
            push!(ps, clamp(p1/d, 1e-6, 1-1e-6))
            lab = [a ? 1 : (b ? 2 : 0) for (a,b) in zip(x1,x2)]
            sw += count(i -> lab[i]!=lab[i-1] && lab[i]!=0 && lab[i-1]!=0,
                        2:length(lab))
        end
    finally
        Nereus.AD_INFORMED_BIRTH[] = true
    end
    f = mean(ps)
    return abs(log(f/(1-f))), sw
end

println("Two labelled copies of ONE K-coefficient model. Symmetry forces 0.5/0.5,")
println("so every number below is pure sampler bias.\n")
@printf("%-4s %-10s %-8s %-11s %-9s\n", "K", "birth", "bridge", "bias[nat]", "switches")
rows = Tuple{Int,String,Int,Float64,Int}[]
for K in KS
    for (lbl, inf, br) in (("informed", true, 1), ("blind", false, 1),
                           ("blind", false, 8))
        b, s = measure(K; informed=inf, bridge=br, rate=0.0)
        push!(rows, (K, lbl, br, b, s))
        @printf("%-4d %-10s %-8d %-11.3f %-9d\n", K, lbl, br, b, s)
    end
end

println()
blind1 = [r for r in rows if r[2]=="blind" && r[3]==1]
blind8 = [r for r in rows if r[2]=="blind" && r[3]==8]
worst_b1 = maximum(r[4] for r in blind1)
@printf("blind birth, no bridge : worst bias %.3f nat (K=%d)\n",
        worst_b1, blind1[argmax([r[4] for r in blind1])][1])
@printf("blind birth, 8 bridges : worst bias %.3f nat\n",
        maximum(r[4] for r in blind8))
if worst_b1 < 0.3
    println("\n➖ Blind birth is NOT the dominant problem at these dimensions.")
    exit(2)
else
    println("\n✅ Blind-birth bias reproduced and quantified vs dimension.")
    exit(0)
end
