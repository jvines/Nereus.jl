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

# Gravity darkening. The workspace transit method has no gravity-darkened model
# (transit_likelihood.jl); it hands over to the allocating method only for TTVs,
# for exposures longer than 2 min and for photometric noise models. So on a fit
# with a :GD planet and every cadence at 2 min or shorter, an evaluator that
# called it integrated a posterior with no gravity darkening: flat in i_star
# and lambda, and thousands of nats away from `_logdensity_parts`.
const _BW_GD_P, _BW_GD_ARS = 2.827969, 6.815
const _BW_GD_T = collect(range(-0.09, 0.09; length = 301))
const _BW_GD_TRV = collect(range(0.0, 20.0; length = 30))

function _bw_gd_build(mode, flux)
    fx(v) = Dict{String, Any}("type" => "FixedPrior", "args" => [v])
    un(a, b) = Dict{String, Any}("type" => "UniformPrior", "args" => [a, b])
    priors = Dict{String, Any}(
        "P_k1" => fx(_BW_GD_P), "Tc_k1" => un(-0.01, 0.01), "b_k1" => un(0.5, 0.9),
        "rr_k1" => un(0.09, 0.14), "sesinw_k1" => fx(0.0), "secosw_k1" => fx(0.0),
        "rho_s" => fx(3π * _BW_GD_ARS^3 / (6.674e-8 * (_BW_GD_P * 86400)^2)),
        "offset_TESS" => fx(0.0), "jitter_TESS" => fx(0.0), "dilution_TESS" => fx(0.0),
        "q1_TESS" => fx(0.33), "q2_TESS" => fx(0.28),
        "lambda_k1" => un(-π, π), "v_sin_i_star" => fx(25_900.0))
    data = Dict{String, Any}("transit_photometry" => [Dict(
        "instrument" => "TESS", "exposure_time" => 120.0,
        "values" => Dict("bjd" => _BW_GD_T, "flux" => flux,
                         "flux_err" => fill(2.4e-5, length(_BW_GD_T))))])
    if mode == "RVPM_GD"
        priors["K_k1"] = un(10.0, 300.0)
        priors["gamma_SIM"] = un(-50.0, 50.0)
        priors["sigma_SIM"] = un(0.0, 20.0)
        rv = 120.0 .* sin.(2π .* _BW_GD_TRV ./ _BW_GD_P) .+ 5.0 .* randn(MersenneTwister(2), 30)
        data["rv"] = Dict("values" => Dict("bjd" => _BW_GD_TRV, "rv" => rv,
                                           "rv_err" => fill(5.0, 30),
                                           "instrument" => fill("SIM", 30)))
    end
    cfg = Dict{String, Any}(
        "star" => Dict("M_s" => 1.60, "R_s" => 1.47), "priors" => priors,
        "model" => Dict("max_kplanet" => 1, "planet_modes" => [mode],
                        "parametrization" => Dict("time" => "Tc", "use_rho_s" => true)),
        "data" => data)
    d, irv, ipm = Nereus._build_data(cfg["data"])
    params, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
    return params, d
end

# The light curve of a gravity-darkened transit at i_star = 20 deg and
# lambda = -60 deg, with white noise.
function _bw_gd_target(mode)
    params, d = _bw_gd_build(mode, ones(length(_BW_GD_T)))
    th = Nereus.Theta{Float64}(params)
    L = params.layout
    truth = Dict("Tc_k1" => 0.0, "b_k1" => 0.75, "rr_k1" => 0.116,
                 "lambda_k1" => deg2rad(-60.0), "i_star" => deg2rad(20.0))
    x = [get(truth, nm, (p.lo + p.hi) / 2) for (nm, p) in zip(L.unfrozen_names, L.unfrozen_priors)]
    Nereus.set_unfrozen!(th, x)
    pred, _ = Nereus.phot_predictions(th, d)
    params, d = _bw_gd_build(mode, pred .+ 2.4e-5 .* randn(MersenneTwister(1), length(pred)))
    return Nereus.NereusTarget(params, d; unconstrained = true), x
end

@testset "bridge evaluator on a gravity-darkened fit" begin
    for mode in ("PM_GD", "RVPM_GD")
        tg, x_true = _bw_gd_target(mode)
        names = tg.params.layout.unfrozen_names
        # The case the workspace method does not hand over by itself.
        @test Nereus._phot_n_super(tg.data) == 1
        @test "i_star" in names && "lambda_k1" in names
        rng = MersenneTwister(17)
        xs = [Nereus._draw_from_prior(tg, rng) for _ in 1:60]
        for _ in 1:60
            push!(xs, [clamp(v + 1e-3 * (p.hi - p.lo) * randn(rng), p.lo + 1e-9, p.hi - 1e-9)
                       for (v, p) in zip(x_true, tg.params.layout.unfrozen_priors)])
        end
        ys = [Nereus.transform_forward(x, tg.transform) for x in xs]
        ev = _BridgeEvaluator(tg)
        ref = [_ref(tg, y) for y in ys]
        got = [_bridge_logdensity!(ev, y) for y in ys]
        @test isfinite.(got) == isfinite.(ref)
        @test count(isfinite, ref) >= 100
        if mode == "PM_GD"
            # No RV data: the whole log density goes through the same methods.
            @test all(got .=== ref)
        else
            fin = isfinite.(ref)
            @test maximum(abs.(got[fin] .- ref[fin]) ./ max.(1.0, abs.(ref[fin]))) < 1e-12
        end

        # The evaluator sees i_star and lambda.
        j_is = findfirst(==("i_star"), names)
        j_la = findfirst(==("lambda_k1"), names)
        at(is, la) = (x = copy(x_true); x[j_is] = deg2rad(is); x[j_la] = deg2rad(la);
                      _bridge_logdensity!(ev, Nereus.transform_forward(x, tg.transform)))
        l_true = at(20.0, -60.0)
        @test l_true - at(85.0, -60.0) > 100
        @test l_true - at(20.0, 30.0) > 100
    end
end
