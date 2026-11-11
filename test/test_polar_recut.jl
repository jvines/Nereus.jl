# A polar angle's chart moves (src/circular.jl, `polar_window`).
#
# The bug this guards against: a cyclic inclination (`WrappedSine`, `PolarFold`)
# was charted on the fixed [-π/2, 3π/2], whose walls sit on the mirror image of
# i = 90°. The posterior on the circle is the physical one twice over, at x and
# at -x, so every edge-on posterior -- an astrometric orbit of a transiting
# planet, a gravity-darkened star seen equator-on -- had its mirror mode cut in
# two by the walls, and raw draws piled against them (i★ = 90° by gravity
# darkening: 10.6% of pt_emcee's draws within 0.5° of a wall, 18.9% of moms').
# The density is 2π-periodic, so the chart [c, c + 2π] is moved like a
# full-circle angle's window: the seam goes where the folded draws and their
# mirror images are thinnest, a pole for an edge-on posterior, and stays on the
# equator's image for a face-on or inclined one.
#
# The assertions pin: the distributions on a moved chart (density, cdf/quantile,
# packing, serialisation) and their default chart unchanged to the bit; where
# the cut puts the seam; that the chart always holds [0, π]; the re-cut on a
# target (layout, packed priors, transform, log-density unchanged); the
# nested-sampling cube rotation; and a pt_emcee fit at i★ = 90° that ends with
# the seam at a pole, no raw draw near it, and resumes into the moved chart.

using Nereus, Distributions, LogDensityProblems, Random, Statistics, Test
using Nereus: WrappedSine, WrappedSinePrior, PolarFold, PolarFoldPrior, polar_window,
              polar_centre_lo, prior_centre, moved_off_value, seam_indices,
              fold_inclination, is_cyclic_inclination, pack_priors, eval_packed_logpdf,
              prior_to_dict, prior_from_dict, bounds, circular_windows, circular_groups

const _PR_FX(v) = Dict("type" => "FixedPrior", "args" => [v])

# Gravity darkening alone (the template of test_angles_never_rail.jl): i★ under
# its default U(0, π), lifted to `PolarFold`, and λ. The light curve is Nereus's
# own forward model at i★ = 90°, λ = 0.6 rad.
function _pr_gd(; flux = nothing, unconstrained = false)
    P, Tc, rp, b, aRs, n = 2.827969, 0.0, 0.116, 0.75, 6.815, 301
    t = collect(range(Tc - 0.09, Tc + 0.09; length = n))
    priors = Dict{String, Any}(
        "P_k1" => _PR_FX(P), "Tc_k1" => _PR_FX(Tc), "b_k1" => _PR_FX(b),
        "rr_k1" => _PR_FX(rp), "sesinw_k1" => _PR_FX(0.0), "secosw_k1" => _PR_FX(0.0),
        "rho_s" => _PR_FX(3π * aRs^3 / (6.674e-8 * (P * 86400)^2)),
        "offset_TESS" => _PR_FX(0.0), "jitter_TESS" => _PR_FX(0.0),
        "dilution_TESS" => _PR_FX(0.0), "q1_TESS" => _PR_FX(0.33),
        "q2_TESS" => _PR_FX(0.28), "v_sin_i_star" => _PR_FX(25_900.0))
    cfg = Dict{String, Any}(
        "star" => Dict("M_s" => 1.60, "R_s" => 1.47), "priors" => priors,
        "model" => Dict("max_kplanet" => 1, "planet_modes" => ["PM_GD"],
                        "parametrization" => Dict("time" => "Tc", "use_rho_s" => true)),
        "data" => Dict("transit_photometry" => [Dict(
            "instrument" => "TESS", "exposure_time" => 120.0,
            "values" => Dict("bjd" => t, "flux" => flux === nothing ? ones(n) : flux,
                             "flux_err" => fill(5e-6, n)))]))
    data, irv, ipm = Nereus._build_data(cfg["data"])
    params, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(cfg["star"]), irv, ipm)
    return NereusTarget(params, data; unconstrained)
end

const _PR_FLUX = let tg = _pr_gd()
    θ = Theta{Float64}(tg.params)
    θ.values[tg.params.layout.systemic.i_star] = π / 2
    set_param!(θ, "lambda_k1", 0.6)
    pred, _ = Nereus.phot_predictions(θ, tg.data)
    pred .+ 5e-6 .* randn(MersenneTwister(5), length(pred))
end

# Within `w` of either wall of the chart (lo, hi).
_pr_near(x, lo, hi, w) = count(v -> v - lo < w || hi - v < w, x)

@testset "polar charts move" begin

    @testset "the distributions on a moved chart" begin
        lifted = (WrappedSinePrior(),
                  PolarFoldPrior(UniformPrior(0.0, π)),
                  PolarFoldPrior(NormalPrior(1.2, 0.4, 0.0, π)))
        # The first chart is the old one, to the bit.
        ws = WrappedSine()
        @test (ws.lo, ws.hi) === (-π / 2, 3π / 2)
        for u in (1e-9, 0.1, 0.25, 0.4, 0.5, 0.75, 0.9, 1 - 1e-9)
            @test quantile(ws, u) === Nereus._wrapped_sine_quantile0(u)
        end
        for x in (-1.5, -0.3, 0.0, 1.0, π, 3.5, 4.7)
            @test cdf(ws, x) === Nereus._wrapped_sine_cdf0(x)
        end
        for ps0 in lifted, lo in (-π, 0.0, -2.2, 0.9)
            ps = Nereus._rewindow(ps0, lo, lo + 2π)
            d, d0 = ps.dist, ps0.dist
            @test bounds(ps) == (lo, lo + 2π) && (minimum(d), maximum(d)) == (lo, lo + 2π)
            @test is_cyclic_inclination(ps)
            # The same periodic density: one representative is enough.
            for x in range(lo + 0.01, lo + 2π - 0.01; length = 23)
                x0 = Nereus.circular_relabel(x, -π / 2, 3π / 2)
                @test logpdf(d, x) ≈ logpdf(d0, x0) atol = 1e-12
            end
            @test logpdf(d, lo - 0.01) == -Inf && logpdf(d, lo + 2π + 0.01) == -Inf
            @test !insupport(d, lo - 0.01) && insupport(d, lo + 1.0)
            xs = range(lo, lo + 2π; length = 40_001)
            @test sum(exp.(logpdf.(Ref(d), xs))) * step(xs) ≈ 1 rtol = 1e-3
            @test cdf(d, lo) == 0 && cdf(d, lo + 2π) == 1
            for u in (0.02, 0.2, 0.37, 0.5, 0.81, 0.99)
                q = quantile(d, u)
                @test lo <= q <= lo + 2π
                @test cdf(d, q) ≈ u atol = 1e-8
            end
            @test issorted(cdf.(Ref(d), xs[1:400:end]))
            # Draws land on the chart and fold to the base prior.
            r = rand(MersenneTwister(1), d, 4000)
            @test all(x -> lo <= x <= lo + 2π, r)
            r0 = fold_inclination.(rand(MersenneTwister(2), d0, 4000))
            @test abs(mean(fold_inclination.(r)) - mean(r0)) < 0.06
            # Packed with its chart, and serialised with it.
            pp = pack_priors([ps])
            for x in (lo - 0.5, lo + 0.3, lo + 2.0, lo + 4.0, lo + 2π + 0.5)
                @test eval_packed_logpdf(x, pp, 1) ≈ logpdf(ps, x) atol = 1e-10
            end
            back = prior_from_dict(prior_to_dict(ps))
            @test typeof(back.dist) == typeof(d) && bounds(back) == bounds(ps)
            @test (back.dist.lo, back.dist.hi) == (d.lo, d.hi)
            @test logpdf(back, lo + 1.0) == logpdf(ps, lo + 1.0)
        end
        # A dict written before charts moved has no chart: the first one.
        dold = prior_to_dict(WrappedSinePrior()); delete!(dold, "lo"); delete!(dold, "hi")
        @test bounds(prior_from_dict(dold)) == (-π / 2, 3π / 2)
    end

    @testset "where the cut puts the seam" begin
        rng = MersenneTwister(3)
        edge = π / 2 .+ 0.02 .* randn(rng, 3000)
        edge[1:3:end] .= .-edge[1:3:end]               # a third in the mirror copy
        # Edge-on: the seam goes to a pole, the chart holds [0, π].
        c = polar_window(edge, -π / 2)
        @test min(abs(c), abs(c + π)) < 1e-9
        @test -π <= c <= 0
        # The draws and their mirror images are as far from it as they can be.
        f = fold_inclination.(edge)
        both = Nereus.circular_relabel.(vcat(f, .-f), c, c + 2π)
        @test _pr_near(both, c, c + 2π, deg2rad(60)) == 0
        # Kept once there: a second cut is a no-op, exactly.
        @test polar_window(edge, c) === c
        # Face-on, retrograde face-on and inclined: the first seam stays.
        for i0 in (deg2rad(5.0), deg2rad(175.0), deg2rad(45.0), deg2rad(135.0))
            x = i0 .+ deg2rad(3.0) .* randn(rng, 3000)
            @test polar_window(x, -π / 2) === -π / 2
            @test polar_window(.-x, -π / 2) === -π / 2   # either copy votes the same
        end
        # A face-on posterior met at a pole moves to the other pole.
        face = deg2rad(5.0) .+ deg2rad(2.0) .* randn(rng, 3000)
        @test polar_window(face, 0.0) ≈ -π atol = 1e-9
        # A posterior that is the prior: |sin| vanishes at the poles.
        pr = rand(rng, WrappedSine(), 20_000)
        cp = polar_window(pr, -π / 2)
        @test min(abs(cp), abs(cp + π)) < deg2rad(15)
        # Every cut holds [0, π]; non-finite draws are ignored.
        for _ in 1:200
            x = rand(rng) * π .+ 0.05 .* randn(rng, 200)
            cc = polar_window(vcat(x, NaN, Inf), rand(rng, (-π / 2, -π, 0.0)))
            @test -π <= cc <= 0
        end
        @test polar_window(Float64[], -π / 2) === -π / 2
        # A single value (an init, an optimum) centres the chart on itself.
        @test polar_centre_lo(deg2rad(30.0)) == -π && polar_centre_lo(-deg2rad(30.0)) == -π
        @test polar_centre_lo(deg2rad(95.0)) == 0.0 && polar_centre_lo(deg2rad(265.0)) == 0.0
    end

    @testset "the re-cut on a target" begin
        tg = _pr_gd(flux = _PR_FLUX, unconstrained = true)
        p = tg.params
        L = p.layout
        ii = findfirst(==("i_star"), L.unfrozen_names)
        il = findfirst(==("lambda_k1"), L.unfrozen_names)
        @test L.unfrozen_priors[ii].dist isa PolarFold
        @test seam_indices(p) == sort([ii, il])
        @test Nereus.circular_indices(p) == [il]          # the reports keep them apart
        @test haskey(circular_windows(p), "i_star")
        # The star owns i★: owner 0, in its own group, permutable or not.
        for perm in (false, true)
            @test [(ii, 0)] in circular_groups(p; permutable = perm)
        end
        xs = [[π / 2 + 0.01, 0.6], [1.0, -2.0], [2.9, 3.0]]
        before = [LogDensityProblems.logdensity(tg, Nereus.transform_forward(x, tg.transform))
                  for x in xs]
        draws = [isodd(k) ? π / 2 + 0.01 * sin(k) : -(π / 2 + 0.01 * sin(k)) for k in 1:600]
        win = Nereus.recut_circular_param!(p, ii, draws; transforms = (tg.transform,))
        @test win !== nothing
        lo, hi = win
        @test min(abs(lo), abs(lo + π)) < 1e-9 && hi == lo + 2π
        @test bounds(L.unfrozen_priors[ii]) == (lo, hi)
        @test L.unfrozen_priors[ii].dist isa PolarFold && L.unfrozen_priors[ii].dist.lo == lo
        @test (L.packed_priors.lowers[ii], L.packed_priors.uppers[ii]) == (lo, hi)
        @test (tg.transform.lowers[ii], tg.transform.uppers[ii]) == (lo, hi)
        @test circular_windows(p)["i_star"] == (lo, hi)
        # The same density at every folded point, which every chart holds.
        after = [LogDensityProblems.logdensity(tg, Nereus.transform_forward(x, tg.transform))
                 for x in xs]
        @test after ≈ before rtol = 1e-12
        # Off-values and prior centres stay off the poles.
        ps = L.unfrozen_priors[ii]
        @test prior_centre(ps) ≈ π / 2 atol = 1e-12
        @test isfinite(logpdf(ps, prior_centre(ps)))
        @test moved_off_value(π / 2, ps, lo + π / 2) == π / 2
        @test moved_off_value(1.0, Nereus.UniformPrior(1.0, 1.0 + 2π), 0.5) == 1.5
        # A checkpoint's windows put the chart back.
        saved = circular_windows(p)
        Nereus.set_circular_window!(p, ii, -π / 2; transforms = (tg.transform,))
        @test Nereus._restore_shared_windows!(p, saved, (tg.transform,))
        @test bounds(L.unfrozen_priors[ii]) == (lo, hi)
    end

    @testset "nested sampling's cube follows the chart" begin
        tg = _pr_gd(flux = _PR_FLUX)
        p = tg.params
        ii = findfirst(==("i_star"), p.layout.unfrozen_names)
        @test Nereus._ns_prior(p.layout.unfrozen_priors[ii]) === p.layout.unfrozen_priors[ii]
        rng = MersenneTwister(9)
        ps0 = p.layout.unfrozen_priors[ii]
        live_x = [isodd(k) ? π / 2 + 0.02 * randn(rng) : 3π / 2 - 0.02 * randn(rng)
                  for k in 1:300]
        live_u = zeros(2, 300)
        live_u[ii, :] .= cdf.(Ref(ps0.dist), Nereus.circular_relabel.(live_x, ps0.lo, ps0.hi))
        live_u[3 - ii, :] .= rand(rng, 300)
        dead = [rand(rng, 2) for _ in 1:50]
        dead_x = [quantile(ps0.dist, u[ii]) for u in dead]
        @test Nereus._ns_recut!(live_u, (dead,), p)
        ps = p.layout.unfrozen_priors[ii]
        @test ps.lo != ps0.lo
        # Each point is the same angle in the new chart.
        for k in 1:300
            @test rem2pi(quantile(ps.dist, live_u[ii, k]) - live_x[k], RoundNearest) ≈ 0 atol = 1e-8
        end
        for (u, x) in zip(dead, dead_x)
            @test rem2pi(quantile(ps.dist, u[ii]) - x, RoundNearest) ≈ 0 atol = 1e-8
        end
        @test _pr_near(quantile.(Ref(ps.dist), live_u[ii, :]), ps.lo, ps.hi, deg2rad(30)) == 0
    end

    # A real fit: i★ = 90°, its mirror copy at -90° on the first chart's walls.
    @testset "pt_emcee at i★ = 90°: the seam at a pole, and resumed there" begin
        kw = (n_temps = 4, n_walkers = 16, n_burnin = 160, seed = 3, show_progress = false,
              bridge_headline = false)
        run(n; k...) = (tg = _pr_gd(flux = _PR_FLUX, unconstrained = true);
                        (tg, sample_pt_emcee(tg, tg.data; kw..., n_steps = n, k...)))
        tg, r = run(300)
        p = tg.params
        ii = findfirst(==("i_star"), p.layout.unfrozen_names)
        lo, hi = bounds(p.layout.unfrozen_priors[ii])
        @test min(abs(lo), abs(lo + π)) < 1e-9
        x = vec(Array(r.chains[:, :i_star, :]))
        @test _pr_near(x, lo, hi, deg2rad(1.0)) == 0
        f = fold_inclination.(x)
        @test abs(median(min.(f, π .- f)) - π / 2) < deg2rad(2.0)

        path = joinpath(mktempdir(), "pt_emcee_polar.jls")
        run(110; checkpoint = path)                        # past the move, in burn-in
        tg2, _ = run(110; checkpoint = path, resume = true)  # runs nothing ...
        @test circular_windows(tg2.params)["i_star"] == circular_windows(p)["i_star"]
        _, ext = run(300; checkpoint = path, resume = true)  # ... and on to the end
        @test ext.chains.value.data == r.chains.value.data
    end
end
