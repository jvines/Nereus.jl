# Bridge sampling evaluates the posterior through the workspace likelihoods
# (src/samplers/bridge.jl, `_bridge_logdensity!`), the ones pt_emcee and the
# other samplers draw with, instead of the allocating `_logdensity_parts`.
#
# The two are the same function up to rounding. The RV part is bit-identical;
# the photometry is not: the workspace path computes each cadence's sky
# separation by a different but equivalent route (sin(ω+f) by the angle-sum
# identity) and sums every cadence in one pass, where `_logdensity_parts` sums
# fixed 4096-point chunks and then the chunk totals. Near the posterior of a
# 20k-point light curve they differ by ~1e-9 in log L, ~1e-14 relative.
# Anything larger is a bug in one of the two, not rounding.
using Test, Nereus, Random, MCMCChains
using Nereus: _BridgeEvaluator, _bridge_logdensity!, _logdensity_parts

Random.seed!(31)
const _BW_NRV = 40
const _BW_TRV = sort!(300 .* rand(_BW_NRV))
const _BW_RV = 30.0 .* sin.(2π .* (_BW_TRV .- 1.2) ./ 3.11) .+ 2.0 .* randn(_BW_NRV)
const _BW_NPH = 20_000
const _BW_TPH = collect(range(0.0, 40.0; length = _BW_NPH))
const _BW_FLUX = let ph = @. abs(mod(_BW_TPH - 1.2 + 3.11 / 2, 3.11) - 3.11 / 2)
    f = 1.0 .+ 3e-4 .* randn(_BW_NPH); f[ph .< 0.05] .-= 0.006; f
end

_bw_target(; unconstrained = true) = begin
    tg = build_target(
        planets = (b = (P = UniformPrior(3.06, 3.16), Tc = UniformPrior(1.15, 1.25),
                        K = UniformPrior(5.0, 60.0), b = UniformPrior(0.0, 0.9),
                        rr = UniformPrior(0.02, 0.15), sesinw = UniformPrior(-0.5, 0.5),
                        secosw = UniformPrior(-0.5, 0.5)),),
        rv = (SIM = (data = (t = _BW_TRV, rv = _BW_RV, rv_err = fill(2.0, _BW_NRV)),
                     sigma = LogUniformPrior(0.5, 10.0)),),
        phot = (TESS = (data = (t = _BW_TPH, flux = _BW_FLUX, flux_err = fill(3e-4, _BW_NPH)),),),
        M_s = 1.0, R_s = 1.0)
    unconstrained ? tg : Nereus.NereusTarget(tg.params, tg.data; unconstrained = false)
end

# Bounded-space points: half near the truth (in transit), half anywhere in the
# prior box (mostly no transit at all, or a very poor fit).
function _bw_points(tg, n; rng = MersenneTwister(4))
    L = tg.params.layout
    near = Dict("P_k1" => 3.11, "Tc_k1" => 1.2, "K_k1" => 30.0, "b_k1" => 0.3,
                "rr_k1" => 0.075, "sesinw_k1" => 0.0, "secosw_k1" => 0.0)
    pts = Vector{Vector{Float64}}()
    for k in 1:n
        x = Float64[]
        for (j, nm) in enumerate(L.unfrozen_names)
            ps = L.unfrozen_priors[j]
            lo, hi = ps.lo, ps.hi
            if !(isfinite(lo) && isfinite(hi))
                lo, hi = -1.0, 1.0
            end
            v = isodd(k) ? get(near, nm, (lo + hi) / 2) + 0.01 * (hi - lo) * randn(rng) :
                           lo + (hi - lo) * rand(rng)
            push!(x, clamp(v, ps.lo + 1e-9, ps.hi - 1e-9))
        end
        push!(pts, x)
    end
    return pts
end

_ref(tg, y) = (a = _logdensity_parts(tg, y); Float64(a[1]) + Float64(a[2]))

@testset "bridge through the workspace likelihood" begin
    for unconstrained in (true, false)
        tg = _bw_target(; unconstrained)
        @test length(tg.data.t_phot) > 4 * Nereus._PHOT_REDUCE_CHUNK
        xs = _bw_points(tg, 200)
        ys = unconstrained ? [Nereus.transform_forward(x, tg.transform) for x in xs] : xs
        # Points the prior or the transform rejects must stay rejected.
        bad = copy(ys[1]); bad[1] = NaN
        push!(ys, bad)
        ev = _BridgeEvaluator(tg)
        ref = [_ref(tg, y) for y in ys]
        got = [_bridge_logdensity!(ev, y) for y in ys]
        @test isfinite.(got) == isfinite.(ref)
        @test count(isfinite, ref) >= 150
        @test got[end] == -Inf
        fin = isfinite.(ref)
        rel = abs.(got[fin] .- ref[fin]) ./ max.(1.0, abs.(ref[fin]))
        @test maximum(rel) < 1e-12
        # The near-truth points are where the bridge's draws live.
        near = [i for i in 1:2:length(xs) if fin[i]]
        @test maximum(abs.(got[near] .- ref[near])) < 1e-8

        # The workspace caches flux per planet between calls. Revisiting a
        # point after others, and a fresh evaluator, must give the same bits.
        ev2 = _BridgeEvaluator(tg)
        again = [_bridge_logdensity!(ev, ys[i]) for i in length(ys):-1:1]
        @test all(reverse(again) .=== got)
        @test all(_bridge_logdensity!(ev2, ys[i]) === got[i] for i in 1:7:length(ys))
    end

    @testset "bridge_evidence matches the evaluator, threaded" begin
        tg = _bw_target()
        L = tg.params.layout
        X = reduce(vcat, permutedims.(_bw_points(tg, 1200; rng = MersenneTwister(8))[1:2:end]))
        ch = Chains(reshape(X, size(X, 1), size(X, 2), 1), Symbol.(L.unfrozen_names))
        b1 = bridge_evidence(tg, ch; n_proposal = 400, seed = 3, n_bootstrap = 10)
        b2 = bridge_evidence(tg, ch; n_proposal = 400, seed = 3, n_bootstrap = 10)
        @test b1.n_post == size(X, 1)
        @test isequal(b1, b2)
    end
end
