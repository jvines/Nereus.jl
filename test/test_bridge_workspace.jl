# Bridge sampling evaluates the posterior through the workspace likelihoods
# (src/samplers/bridge.jl, `_bridge_logdensity!`), the ones pt_emcee and the
# other samplers draw with, instead of the allocating `_logdensity_parts`.
#
# The two are the same function up to rounding, not to the bit. On a fit with
# no noise models at all (on any channel) the workspace RV method caches each
# planet's velocity curve and computes cos(f+ω) by the angle-sum identity; the
# workspace photometry sums every cadence in one pass, where `_logdensity_parts`
# sums fixed 4096-point chunks and then the chunk totals. Near the posterior of a
# 20k-point light curve they differ by a few 1e-9 nats. Relative to |log prior| +
# |log L_RV| + |log L_phot| the largest difference measured is 2.9e-11 (a
# sample maximum); a difference much larger than that, on that scale, is a bug
# in one of the two, not rounding. Relative to max(1, |log p|), the scale the
# tests below use, rounding has no bound: where |log p| < 1 those terms cancel
# and leave their rounding, up to 3.6e-7 relative in the samples measured.
#
# That holds because the evaluator takes the allocating methods at any planet
# whose e `true_anomaly` clamps (outside [0, 0.9999]), where the workspace
# methods compute another model, and for the photometry of a fit with a
# gravity-darkened planet. There it gives the same bits as `_logdensity_parts`.
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

_bw_target(; unconstrained = true, se = 0.5) = begin
    tg = build_target(
        planets = (b = (P = UniformPrior(3.06, 3.16), Tc = UniformPrior(1.15, 1.25),
                        K = UniformPrior(5.0, 60.0), b = UniformPrior(0.0, 0.9),
                        rr = UniformPrior(0.02, 0.15), sesinw = UniformPrior(-se, se),
                        secosw = UniformPrior(-se, se)),),
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

# Gravity darkening. The workspace transit method had no gravity-darkened model
# when the evaluator was written; it handed over to the allocating method only
# for TTVs, for supersampled exposures (then longer than 2 min, now longer than
# 3 min) and for photometric noise models. So on a fit with a :GD planet and no
# supersampled cadence, an evaluator that called it integrated a posterior with
# no gravity darkening: flat in i_star and lambda, and thousands of nats away
# from `_logdensity_parts`. The workspace method now has gravity darkening
# (test_gd_workspace_likelihood.jl); the evaluator still takes the allocating
# photometry for these fits, and must see i_star and lambda.
const _BW_GD_P, _BW_GD_ARS = 2.827969, 6.815
const _BW_GD_T = collect(range(-0.09, 0.09; length = 301))
const _BW_GD_TRV = collect(range(0.0, 20.0; length = 30))

# `noise`: nothing, "rv" (a MaternGP on the RV) or "phot" (a CeleriteSHO on the
# photometry only).
function _bw_gd_build(mode, flux; noise = nothing)
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
        "noise_models" => noise === nothing ? Any[] :
                          noise == "rv" ? Any[Dict("kind" => "MaternGP", "channel" => "rv")] :
                          Any[Dict("kind" => "CeleriteSHO", "channel" => "phot")],
        "data" => data)
    d, irv, ipm = Nereus._build_data(cfg["data"])
    params, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
    return params, d
end

# The light curve of a gravity-darkened transit at i_star = 20 deg and
# lambda = -60 deg, with white noise.
function _bw_gd_target(mode; noise = nothing)
    params, d = _bw_gd_build(mode, ones(length(_BW_GD_T)); noise)
    th = Nereus.Theta{Float64}(params)
    L = params.layout
    truth = Dict("Tc_k1" => 0.0, "b_k1" => 0.75, "rr_k1" => 0.116,
                 "lambda_k1" => deg2rad(-60.0), "i_star" => deg2rad(20.0))
    x = [get(truth, nm, (p.lo + p.hi) / 2) for (nm, p) in zip(L.unfrozen_names, L.unfrozen_priors)]
    Nereus.set_unfrozen!(th, x)
    pred, _ = Nereus.phot_predictions(th, d)
    params, d = _bw_gd_build(mode, pred .+ 2.4e-5 .* randn(MersenneTwister(1), length(pred));
                             noise)
    return Nereus.NereusTarget(params, d; unconstrained = true), x
end

@testset "bridge evaluator on a gravity-darkened fit" begin
    for (mode, noise) in (("PM_GD", nothing), ("RVPM_GD", nothing), ("RVPM_GD", "rv"),
                          ("RVPM_GD", "phot"))
        tg, x_true = _bw_gd_target(mode; noise)
        nms = tg.params.config.noise_models
        @test isempty(nms) == (noise === nothing)
        @test all(m -> Nereus.noise_channel(m) === Symbol(something(noise, "")), nms)
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
        # The photometry goes through the allocating method in both, so it
        # gives the same bits. Both RV methods choose their branch on
        # `config.noise_models` as a whole, not on the RV channel: with any
        # noise model, on the RV or only on the photometry, the RV gives the
        # same bits. With none at all it takes the workspace no-noise method
        # and its rounding, as on any other fit, and a gravity-darkened fit
        # with RV data is then equal to rounding only, not to the bit.
        if mode == "PM_GD" || noise !== nothing
            @test all(got .=== ref)
        else
            fin = isfinite.(ref)
            @test maximum(abs.(got[fin] .- ref[fin]) ./ max.(1.0, abs.(ref[fin]))) < 1e-12
        end
        # The RV on its own, since a large photometric term can absorb its last
        # bits in the sum.
        if mode == "RVPM_GD" && noise !== nothing
            ev2 = _BridgeEvaluator(tg)
            th2 = ev2.theta
            same = 0
            for x in xs
                Nereus.set_unfrozen!(th2, x)
                same += Nereus.rv_log_likelihood(th2, tg.data, ev2.ws) ===
                        Nereus.rv_log_likelihood(th2, tg.data)
            end
            @test same == length(xs)
        end

        # The evaluator sees i_star and lambda. A GP on the photometry absorbs
        # part of the change in the light curve.
        j_is = findfirst(==("i_star"), names)
        j_la = findfirst(==("lambda_k1"), names)
        at(is, la) = (x = copy(x_true); x[j_is] = deg2rad(is); x[j_la] = deg2rad(la);
                      _bridge_logdensity!(ev, Nereus.transform_forward(x, tg.transform)))
        l_true = at(20.0, -60.0)
        dmin = noise == "phot" ? 10 : 100
        @test l_true - at(85.0, -60.0) > dmin
        @test l_true - at(20.0, 30.0) > dmin
    end
end

# RV-only fits, where the RV is the whole likelihood. Without noise models the
# workspace RV method is not bit-identical to the other one (see the header);
# with them it evaluates each cadence the same way. Either way the evaluator
# must agree with `_logdensity_parts` to rounding, over the prior and near
# the best point, with eccentricities up to 0.9.
@testset "bridge evaluator on RV-only fits, with and without noise models" begin
    rng = MersenneTwister(41)
    t = sort!(500 .* rand(rng, 80))
    rv = 25 .* sin.(2π .* (t .- 1.0) ./ 7.3) .+ 8 .* sin.(2π .* (t .- 3.0) ./ 41.0) .+
         3 .* randn(rng, 80)
    pl = (b = (P = UniformPrior(7.0, 7.6), Tc = UniformPrior(0.5, 1.5), K = UniformPrior(1.0, 60.0),
               sesinw = UniformPrior(-0.95, 0.95), secosw = UniformPrior(-0.95, 0.95)),
          c = (P = UniformPrior(38.0, 44.0), Tc = UniformPrior(1.0, 5.0), K = UniformPrior(1.0, 30.0),
               sesinw = UniformPrior(-0.95, 0.95), secosw = UniformPrior(-0.95, 0.95)))
    rvd = (SIM = (data = (t = t, rv = rv, rv_err = fill(3.0, 80)),
                  sigma = LogUniformPrior(0.1, 20.0)),)
    for nm in (nothing, [MaternGP()])
        tg = build_target(planets = pl, rv = rvd, noise_models = nm)
        @test isempty(tg.data.t_phot)
        @test isempty(tg.params.config.noise_models) == (nm === nothing)
        r2 = MersenneTwister(5)
        xs = [Nereus._draw_from_prior(tg, r2) for _ in 1:150]
        ys = [Nereus.transform_forward(x, tg.transform) for x in xs]
        best = xs[argmax([_ref(tg, y) for y in ys])]
        pri = tg.params.layout.unfrozen_priors
        for _ in 1:150
            x = [clamp(v + 1e-3 * (p.hi - p.lo) * randn(r2), p.lo + 1e-9, p.hi - 1e-9)
                 for (v, p) in zip(best, pri)]
            push!(ys, Nereus.transform_forward(x, tg.transform))
        end
        ev = _BridgeEvaluator(tg)
        ref = [_ref(tg, y) for y in ys]
        got = [_bridge_logdensity!(ev, y) for y in ys]
        fin = isfinite.(ref)
        @test isfinite.(got) == fin
        @test count(fin) >= 250
        @test maximum(abs.(got[fin] .- ref[fin]) ./ max.(1.0, abs.(ref[fin]))) < 1e-12
        @test maximum(abs(got[i] - ref[i]) for i in 151:300 if fin[i]) < 1e-8
    end
end

# A two-planet RV + transit fit with ρ⋆ and the mean anomaly as parameters and
# the Gladman stability check, no noise models, two RV and two photometric
# instruments. On this kind of fit (with e up to 0.98 and b up to 1.3) a
# difference of 1.9e-10 nats was measured near the best point, where
# log p = -18.2, and 1.2e-10 nats on a prior draw where log p = 2.46: 1.03e-11
# and 4.7e-11 relative to max(1, |log p|), still rounding. Here e and b are
# narrower so that prior draws give finite densities. The relative bound
# allows for points where |log p| is that small, not for |log p| < 1.
@testset "bridge evaluator on a two-planet fit with rho_s, Mo and stability" begin
    un(a, b) = Dict{String, Any}("type" => "UniformPrior", "args" => [a, b])
    rng = MersenneTwister(11)
    trv = sort!(400 .* rand(rng, 60))
    rv = 30 .* sin.(2π .* (trv .- 1.2) ./ 3.11) .+ 12 .* sin.(2π .* (trv .- 2.0) ./ 8.0) .+
         3 .* randn(rng, 60)
    t1 = collect(range(0.0, 40.0; length = 10_000))
    t2 = collect(range(10.33, 10.73; length = 600))
    lc(t, d1, d2) = (f = 1.0 .+ 3e-4 .* randn(rng, length(t));
                     f[abs.(mod.(t .- 1.2 .+ 1.555, 3.11) .- 1.555) .< 0.05] .-= d1;
                     f[abs.(mod.(t .- 2.0 .+ 4.0, 8.0) .- 4.0) .< 0.07] .-= d2; f)
    pri = Dict{String, Any}(
        "P_k1" => un(3.05, 3.17), "P_k2" => un(7.8, 8.2), "K_k1" => un(1.0, 60.0),
        "K_k2" => un(1.0, 40.0), "b_k1" => un(0.0, 1.0), "b_k2" => un(0.0, 1.0),
        "rr_k1" => un(0.01, 0.2), "rr_k2" => un(0.01, 0.2), "rho_s" => un(0.3, 3.0),
        "sesinw_k1" => un(-0.6, 0.6), "secosw_k1" => un(-0.6, 0.6),
        "sesinw_k2" => un(-0.6, 0.6), "secosw_k2" => un(-0.6, 0.6),
        "dilution_NGTS" => un(0.0, 0.5),
        "gamma_HARPS" => un(-20.0, 20.0), "gamma_FEROS" => un(-20.0, 20.0),
        "sigma_HARPS" => un(0.0, 20.0), "sigma_FEROS" => un(0.0, 20.0))
    cfg = Dict{String, Any}(
        "star" => Dict("M_s" => 1.0, "R_s" => 1.0), "priors" => pri,
        "model" => Dict{String, Any}(
            "max_kplanet" => 2, "planet_modes" => ["RVPM", "RVPM"], "stability" => "gladman",
            "phot_trend_order" => 1,
            "parametrization" => Dict("mass" => "K_driven", "time" => "Mo", "ew" => "sesinw",
                                      "geom" => "b_rr", "use_rho_s" => true)),
        "noise_models" => Any[],
        "data" => Dict{String, Any}(
            "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv, "rv_err" => fill(3.0, 60),
                                          "instrument" => [isodd(i) ? "HARPS" : "FEROS" for i in 1:60])),
            "transit_photometry" => Any[
                Dict("instrument" => "TESS", "exposure_time" => 120.0,
                     "values" => Dict("bjd" => t1, "flux" => lc(t1, 0.006, 0.003),
                                      "flux_err" => fill(3e-4, length(t1)))),
                Dict("instrument" => "NGTS", "exposure_time" => 60.0,
                     "values" => Dict("bjd" => t2, "flux" => lc(t2, 0.004, 0.002),
                                      "flux_err" => fill(5e-4, length(t2))))]))
    d, irv, ipm = Nereus._build_data(cfg["data"])
    params, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
    tg = Nereus.NereusTarget(params, d; unconstrained = true)
    names = params.layout.unfrozen_names
    @test "rho_s" in names && "Mo_k1" in names && "Mo_k2" in names
    @test params.config.stability === :gladman && isempty(params.config.noise_models)
    r2 = MersenneTwister(7)
    xs = [Nereus._draw_from_prior(tg, r2) for _ in 1:250]
    ys = [Nereus.transform_forward(x, tg.transform) for x in xs]
    best = xs[argmax([_ref(tg, y) for y in ys])]
    ys = ys[1:60]
    for _ in 1:100
        x = [clamp(v + 3e-4 * (p.hi - p.lo) * randn(r2), p.lo + 1e-9, p.hi - 1e-9)
             for (v, p) in zip(best, params.layout.unfrozen_priors)]
        push!(ys, Nereus.transform_forward(x, tg.transform))
    end
    ev = _BridgeEvaluator(tg)
    ref = [_ref(tg, y) for y in ys]
    got = [_bridge_logdensity!(ev, y) for y in ys]
    fin = isfinite.(ref)
    @test isfinite.(got) == fin
    @test count(fin) >= 120
    @test maximum(abs.(got[fin] .- ref[fin]) ./ max.(1.0, abs.(ref[fin]))) < 1e-10
    @test maximum(abs(got[i] - ref[i]) for i in 61:160 if fin[i]) < 1e-8
end

# Eccentricities that `true_anomaly` clamps. The allocating RV and transit
# methods take the true anomaly from `true_anomaly`, which clamps e to 0.9999;
# the workspace photometry, and the workspace RV of a fit with no noise models,
# use the e they are given. Above 0.9999 those are two models, nats apart, not one
# model rounded two ways. The evaluator takes the allocating methods wherever a
# planet's e is clamped, so there it must give the same bits as
# `_logdensity_parts`; just below the clamp it keeps the workspace methods and
# agrees to rounding.
#
# The planet blocks are abstractly typed, so `planet_e_w`'s result is boxed.
# `_bridge_e_clamped` asserts its type, and must then allocate no more than
# that typed decode: 64 bytes per planet with :sesinw, where `planet_e_w` alone
# costs the same, and with :ew, where `planet_e_w` alone costs 32 and the
# assertion boxes the tuple for the other 32. Without the assertion the
# comparison after it was boxed as well (112 bytes per planet with :sesinw).
_bw_alloc_ec(th) = @allocated Nereus._bridge_e_clamped(th)
_bw_alloc_ew(th) = sum(k -> @allocated(Nereus.planet_e_w(th, k)::Tuple{Float64, Float64}),
                       Nereus.planet_indices(th))

# An RV-only fit with e and ω as parameters (:ew).
function _bw_ew_target(t, rv)
    un(a, b) = Dict{String, Any}("type" => "UniformPrior", "args" => [a, b])
    n = length(t)
    cfg = Dict{String, Any}(
        "star" => Dict("M_s" => 1.0, "R_s" => 1.0),
        "priors" => Dict{String, Any}(
            "P_k1" => un(7.0, 7.6), "Tc_k1" => un(0.5, 1.5), "K_k1" => un(1.0, 60.0),
            "ecc_k1" => un(0.0, 1.0), "w_k1" => un(-π, π),
            "gamma_SIM" => un(-50.0, 50.0), "sigma_SIM" => un(0.1, 20.0)),
        "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"],
                                     "parametrization" => Dict("time" => "Tc", "ew" => "ew")),
        "noise_models" => Any[],
        "data" => Dict{String, Any}("rv" => Dict("values" => Dict(
            "bjd" => t, "rv" => rv, "rv_err" => fill(3.0, n), "instrument" => fill("SIM", n)))))
    d, irv, ipm = Nereus._build_data(cfg["data"])
    params, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
    return Nereus.NereusTarget(params, d; unconstrained = true)
end

@testset "bridge evaluator where true_anomaly clamps e" begin
    rng = MersenneTwister(43)
    t = sort!(500 .* rand(rng, 80))
    rv = 25 .* sin.(2π .* (t .- 1.0) ./ 7.3) .+ 3 .* randn(rng, 80)
    rvonly = build_target(
        planets = (b = (P = UniformPrior(7.0, 7.6), Tc = UniformPrior(0.5, 1.5),
                        K = UniformPrior(1.0, 60.0), sesinw = UniformPrior(-1.0, 1.0),
                        secosw = UniformPrior(-1.0, 1.0)),),
        rv = (SIM = (data = (t = t, rv = rv, rv_err = fill(3.0, 80)),
                     sigma = LogUniformPrior(0.1, 20.0)),))
    for tg in (rvonly, _bw_target(; se = 1.0), _bw_ew_target(t, rv))
        names = tg.params.layout.unfrozen_names
        ew = tg.params.config.parametrization.ew
        @test ew in (:sesinw, :ew)
        j1 = findfirst(==(ew === :ew ? "ecc_k1" : "sesinw_k1"), names)
        j2 = findfirst(==(ew === :ew ? "w_k1" : "secosw_k1"), names)
        r2 = MersenneTwister(6)
        xs0 = [Nereus._draw_from_prior(tg, r2) for _ in 1:100]
        best = xs0[argmax([_ref(tg, Nereus.transform_forward(x, tg.transform)) for x in xs0])]
        at(e, ω) = (x = copy(best);
                    ew === :ew ? (x[j1] = e; x[j2] = ω) :
                                 (x[j1] = sqrt(e) * sin(ω); x[j2] = sqrt(e) * cos(ω));
                    Nereus.transform_forward(x, tg.transform))
        # With :ew, ω is a parameter bounded by [-π, π]; keep off the bound.
        ωs = range(-π, π; length = 9)[1:8] .+ (ew === :ew ? 0.05 : 0.0)
        hi = [at(e, ω) for e in (0.99990001, 0.99995, 0.99999, 0.9999999) for ω in ωs]
        lo = [at(e, ω) for e in (0.9998, 0.99989) for ω in ωs]
        th = Nereus.Theta{Float64}(tg.params)
        clamped(y) = (Nereus.set_unfrozen!(th, Nereus.transform_inverse(y, tg.transform));
                      Nereus._bridge_e_clamped(th))
        @test all(clamped, hi)
        @test !any(clamped, lo)
        # The check costs no more than decoding e with the type asserted: the
        # comparison after it allocates nothing.
        _bw_alloc_ec(th); _bw_alloc_ew(th)
        @test _bw_alloc_ec(th) <= _bw_alloc_ew(th)
        ev = _BridgeEvaluator(tg)
        ref_hi = [_ref(tg, y) for y in hi]
        got_hi = [_bridge_logdensity!(ev, y) for y in hi]
        @test count(isfinite, ref_hi) >= 24
        @test all(got_hi .=== ref_hi)
        ref_lo = [_ref(tg, y) for y in lo]
        got_lo = [_bridge_logdensity!(ev, y) for y in lo]
        fin = isfinite.(ref_lo)
        @test isfinite.(got_lo) == fin
        @test count(fin) >= 12
        @test maximum(abs.(got_lo[fin] .- ref_lo[fin]) ./ max.(1.0, abs.(ref_lo[fin]))) < 1e-12
    end
end
