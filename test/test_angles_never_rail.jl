# No angle rails (src/circular.jl, src/priors.jl).
#
# Every angle is sampled on its own topology, whatever prior the user wrote:
#
# * a full-circle angle (Mo, phic, Ω, λ, ω under :ew) is a circle charted by a
#   one-period window with walls, whose seam the samplers move off the
#   posterior. A Normal, or a Uniform over more than one period, is replaced in
#   the layout by its pushforward onto the circle (`WrappedPushforward`), which
#   is charted and re-cut like the uniform. WrappedUniform, once a chart with no
#   walls, is the walled circular Uniform: unwalled, the stretch moves drifted
#   the ensemble out to |λ| ~ 1e12 rad (measured in a real RM fit);
# * a polar angle (an orbital inclination, the stellar inclination i★) under
#   any prior over all of [0, π] is folded at its poles (`WrappedSinePrior`,
#   `PolarFoldPrior`), its draws reported in [0, π];
# * gravity darkening alone cannot tell (i★, λ) from (π - i★, λ + π): both
#   are sampled and the draws put in one representative (`spin_flip`);
# * the live R-hat reads every angle in its reported representative;
# * a prior on part of an angle's range keeps its walls, and the reporters say
#   that a rail there is the prior's restriction.

using Nereus, Distributions, LogDensityProblems, MCMCChains, Random, Test
using Statistics: median, mean, quantile, std
using Nereus: PolarFold, PolarFoldPrior, WrappedPushforward, circular_lift, polar_lift,
              wrapped_push_logpdf, pack_priors, eval_packed_logpdf, prior_to_dict,
              prior_from_dict, fold_inclination, spin_flip, spin_canonical,
              fold_cyclic_inclinations!, AngleCanon, restricted_angles,
              is_cyclic_inclination, circular_relabel, set_param!, Theta

const _ANR_FX(v) = Dict("type" => "FixedPrior", "args" => [v])
_anr_un(a, b) = Dict{String, Any}("type" => "UniformPrior", "args" => [a, b])

# A GD-only transit (the template of test_gravity_darkening.jl), i★ and λ free.
function _anr_gd(; flux = nothing, extra = Dict{String, Any}(), mode = "PM_GD")
    P, Tc, rp, b, aRs, n = 2.827969, 0.0, 0.116, 0.75, 6.815, 301
    t = collect(range(Tc - 0.09, Tc + 0.09; length = n))
    priors = Dict{String, Any}(
        "P_k1" => _ANR_FX(P), "Tc_k1" => _ANR_FX(Tc), "b_k1" => _ANR_FX(b),
        "rr_k1" => _ANR_FX(rp), "sesinw_k1" => _ANR_FX(0.0), "secosw_k1" => _ANR_FX(0.0),
        "rho_s" => _ANR_FX(3π * aRs^3 / (6.674e-8 * (P * 86400)^2)),
        "offset_TESS" => _ANR_FX(0.0), "jitter_TESS" => _ANR_FX(0.0),
        "dilution_TESS" => _ANR_FX(0.0), "q1_TESS" => _ANR_FX(0.33),
        "q2_TESS" => _ANR_FX(0.28), "v_sin_i_star" => _ANR_FX(25_900.0))
    merge!(priors, extra)
    cfg = Dict{String, Any}(
        "star" => Dict("M_s" => 1.60, "R_s" => 1.47), "priors" => priors,
        "model" => Dict("max_kplanet" => 1, "planet_modes" => [mode],
                        "parametrization" => Dict("time" => "Tc", "use_rho_s" => true)),
        "data" => Dict("transit_photometry" => [Dict(
            "instrument" => "TESS", "exposure_time" => 120.0,
            "values" => Dict("bjd" => t, "flux" => flux === nothing ? ones(n) : flux,
                             "flux_err" => fill(2.4e-5, n)))]))
    data, irv, ipm = Nereus._build_data(cfg["data"])
    params, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(cfg["star"]), irv, ipm)
    return params, data
end

# An RV-only eccentric planet under :ew, ω and Mo free (data from Nereus's own
# forward model, e = 0.3, 2 m/s errors and 3 m/s of jitter).
function _anr_ew(; ω = π - 0.03, Mo = 0.02, wprior = _anr_un(-π, π))
    function build(t, rv)
        n = length(t)
        cfg = Dict{String, Any}(
            "star" => Dict("M_s" => 1.0, "R_s" => 1.0),
            "priors" => Dict{String, Any}(
                "P_k1" => _anr_un(12.0, 12.6), "Mo_k1" => _anr_un(0.0, 2π),
                "K_k1" => _anr_un(1.0, 60.0), "ecc_k1" => _anr_un(0.0, 0.9),
                "w_k1" => wprior, "gamma_SIM" => _anr_un(-50.0, 50.0),
                "sigma_SIM" => _anr_un(0.01, 20.0)),
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"],
                                         "parametrization" => Dict("ew" => "ew")),
            "noise_models" => Any[],
            "data" => Dict{String, Any}("rv" => Dict("values" => Dict(
                "bjd" => t, "rv" => rv, "rv_err" => fill(2.0, n),
                "instrument" => fill("SIM", n)))))
        d, irv, ipm = Nereus._build_data(cfg["data"])
        p, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
        return NereusTarget(p, d)
    end
    rng = MersenneTwister(42)
    t = sort(55000 .+ 200 .* rand(rng, 60))
    tg0 = build(t, zeros(60))
    th = Theta{Float64}(tg0.params)
    for (nm, v) in ("P_k1" => 12.3, "K_k1" => 25.0, "ecc_k1" => 0.3, "w_k1" => ω,
                    "Mo_k1" => Mo, "gamma_SIM" => 0.0, "sigma_SIM" => 0.01)
        set_param!(th, nm, v)
    end
    model, _ = Nereus.rv_predictions(th, tg0.data)
    return build(t, model .+ sqrt(13.0) .* randn(rng, 60))
end

_anr_cdist(a, b) = abs(rem2pi(a - b, RoundNearest))

@testset "angles never rail" begin

    @testset "PolarFold: the user's density reflected at both poles" begin
        for base in (UniformPrior(0.0, π), NormalPrior(1.2, 0.4, 0.0, π),
                     NormalPrior(3.0, 0.3, 0.0, π))
            ps = PolarFoldPrior(base)
            @test ps.dist isa PolarFold && bounds(ps) == (-π / 2, 3π / 2)
            @test is_cyclic_inclination(ps) && is_wrapped(ps)
            for x in (0.2, 1.0, 2.9)
                # x, -x and 2π - x are one inclination, each with half the mass
                @test logpdf(ps, x) ≈ logpdf(base, x) - log(2)
                -x >= -π / 2 && @test logpdf(ps, -x) ≈ logpdf(ps, x) atol = 1e-12
                2π - x <= 3π / 2 && @test logpdf(ps, 2π - x) ≈ logpdf(ps, x) atol = 1e-12
            end
            @test logpdf(ps, -2.0) == -Inf && logpdf(ps, 5.0) == -Inf
            # normalised on its chart, and cdf/quantile inverse
            xs = range(-π / 2, 3π / 2; length = 40_001)
            @test sum(exp.(logpdf.(Ref(ps), xs))) * step(xs) ≈ 1 rtol = 1e-3
            for u in (0.05, 0.3, 0.5, 0.8, 0.97)
                @test cdf(ps.dist, quantile(ps.dist, u)) ≈ u atol = 1e-8
            end
            # packed and serialised the same
            pp = pack_priors([ps])
            for x in (-1.0, 0.0, 0.7, 2.0, 3.5, 4.0, 5.0)
                @test eval_packed_logpdf(x, pp, 1) ≈ logpdf(ps, x) atol = 1e-10
            end
            r = prior_from_dict(prior_to_dict(ps))
            @test r.dist isa PolarFold && logpdf(r, 0.7) ≈ logpdf(ps, 0.7)
        end
        @test polar_lift(SinePrior()).dist isa Nereus.WrappedSine
        @test polar_lift(UniformPrior(0.0, π / 2)) == UniformPrior(0.0, π / 2)
        @test_throws ArgumentError PolarFoldPrior(UniformPrior(0.0, 2.0))
    end

    @testset "WrappedPushforward: the prior on the circle the user's induces" begin
        cases = (NormalPrior(1.0, 0.8),                    # untruncated
                 NormalPrior(3.0, 1.0, -2π, 2π),           # two periods
                 NormalPrior(0.0, 2.0, -π, π),             # one period: periodic extension
                 UniformPrior(-2π, 2π),                    # two whole periods
                 UniformPrior(0.0, 3π))                    # a period and a half: a step
        for ps in cases
            q = circular_lift(ps)
            @test q.dist isa WrappedPushforward
            @test q.hi - q.lo ≈ 2π
            # p̃(θ) = Σ_n p(θ + 2πn), periodic, normalised on its chart
            for θ in range(q.lo + 0.01, q.hi - 0.01; length = 7)
                s = sum(exp(logpdf(ps, θ + 2π * n)) for n in -30:30)
                @test exp(logpdf(q, θ)) ≈ s rtol = 1e-8
                @test wrapped_push_logpdf(θ + 2π, q.dist.kind, q.dist.μ, q.dist.σ,
                                          q.dist.blo, q.dist.bhi, q.dist.logZ) ≈
                      logpdf(q, θ) atol = 1e-10
            end
            xs = range(q.lo, q.hi; length = 40_001)
            @test sum(exp.(logpdf.(Ref(q), xs))) * step(xs) ≈ 1 rtol = 1e-3
            @test logpdf(q, q.hi + 0.1) == -Inf
            pp = pack_priors([q])
            for θ in range(q.lo + 0.01, q.hi - 0.01; length = 5)
                @test eval_packed_logpdf(θ, pp, 1) ≈ logpdf(q, θ) atol = 1e-10
            end
            for u in (0.1, 0.5, 0.9)
                @test cdf(q.dist, quantile(q.dist, u)) ≈ u atol = 1e-6
            end
            r = prior_from_dict(prior_to_dict(q))
            @test r.dist isa WrappedPushforward && logpdf(r, q.lo + 1) ≈ logpdf(q, q.lo + 1)
        end
        # Not lifted: a one-period uniform (already the circle) and an arc.
        @test circular_lift(UniformPrior(-π, π)) == UniformPrior(-π, π)
        @test circular_lift(UniformPrior(0.0, 1.0)) == UniformPrior(0.0, 1.0)
        @test circular_lift(NormalPrior(0.0, 0.3, -1.0, 1.0)).dist isa Distributions.Truncated
        # An untruncated Normal survives a dict round trip untruncated.
        n = prior_from_dict(prior_to_dict(NormalPrior(0.5, 0.2)))
        @test n.dist isa Normal && !isfinite(n.lo) && !isfinite(n.hi)
    end

    @testset "layout lifts angles, keeps everything else" begin
        tg = _anr_ew(; wprior = Dict{String, Any}("type" => "NormalPrior",
                                                  "args" => [3.0, 1.0]))
        p = tg.params
        L = p.layout
        iw = findfirst(==("w_k1"), L.unfrozen_names)
        @test p.config.priors["w_k1"].dist isa Normal                 # as written
        @test L.unfrozen_priors[iw].dist isa WrappedPushforward
        @test iw in Nereus.circular_indices(p)
        # a sub-range stays as written, and is reported as a restriction
        q = _anr_ew(; wprior = _anr_un(0.0, 1.5)).params
        jw = findfirst(==("w_k1"), q.layout.unfrozen_names)
        @test q.layout.unfrozen_priors[jw] == q.config.priors["w_k1"]
        @test restricted_angles(q) == Dict("w_k1" => "[0, 2π)")
        @test isempty(restricted_angles(p))
    end

    @testset "i★: folded like the orbit's inclination, spin flip exact" begin
        p, d = _anr_gd()
        L = p.layout
        ii = findfirst(==("i_star"), L.unfrozen_names)
        il = findfirst(==("lambda_k1"), L.unfrozen_names)
        @test L.unfrozen_priors[ii].dist isa PolarFold     # default U(0, π), lifted
        sf = spin_flip(p)
        @test sf !== nothing && sf.istar == ii && sf.lams == [il]
        tg = NereusTarget(p, d; unconstrained = false)
        for (i, λ) in ((0.4, -1.0), (1.2, 2.5), (2.9, 0.1))
            x = zeros(2); x[ii] = i; x[il] = λ
            v = tg(x)
            @test isfinite(v)
            # the folds of i★: -x, and 2π - x where it is on the chart
            xm = copy(x); xm[ii] = -i
            i < π / 2 && @test tg(xm) ≈ v rtol = 1e-12
            xr = copy(x); xr[ii] = 2π - i
            2π - i <= 3π / 2 && @test tg(xr) ≈ v rtol = 1e-12
            # the spin flip (π - i★, λ + π)
            xf = copy(x); xf[ii] = π - i; xf[il] = circular_relabel(λ + π, -π, π)
            @test tg(xf) ≈ v rtol = 1e-10
            # canonical representative: folded, positive side of the axis, same density
            for y in (copy(x), xf)
                fold_cyclic_inclinations!(y, p)
                @test 0 <= y[ii] <= π
                @test spin_canonical(y[ii], y[il], L.spin_axis)
                @test tg(y) ≈ v rtol = 1e-10
            end
        end
        # A measurement of the sense of rotation, or a λ that is not free over
        # the circle, breaks the symmetry: no flip.
        pf, _ = _anr_gd(; extra = Dict{String, Any}("lambda_k1" => _ANR_FX(0.3)))
        @test spin_flip(pf) === nothing
        pa, _ = _anr_gd(; extra = Dict{String, Any}("lambda_k1" => _anr_un(0.0, 1.0)))
        @test spin_flip(pa) === nothing
        pn, _ = _anr_gd(; extra = Dict{String, Any}(
            "i_star" => Dict("type" => "NormalPrior", "args" => [1.0, 0.3, 0.0, π])))
        @test spin_flip(pn) === nothing
        @test is_cyclic_inclination(pn.layout.unfrozen_priors[
            findfirst(==("i_star"), pn.layout.unfrozen_names)])

        # The axis is chosen where the draws are thinnest: an equator-on star
        # (i★ = 90° ± 11°, λ = 0.3) with half its draws in the mirror
        # representative is cut in λ, not at i★ = 90°, and comes back whole.
        rng = MersenneTwister(1)
        S = zeros(4000, 2)
        S[:, ii] .= π / 2 .+ 0.2 .* randn(rng, 4000)
        S[:, il] .= 0.3 .+ 0.15 .* randn(rng, 4000)
        for r in 1:2:4000
            S[r, ii] = π - S[r, ii]; S[r, il] = circular_relabel(S[r, il] + π, -π, π)
        end
        fill!(L.spin_axis, 0.0); L.spin_axis[3] = 1.0
        fold_cyclic_inclinations!(S, p; decide_from = S)
        @test abs(L.spin_axis[3]) < 0.3
        λc = S[:, il]
        # one lobe: λ ≈ 0.3 for every draw, or λ ≈ 0.3 - π for every draw (the
        # window is relabelled after, by the seam re-cut)
        c = angle(sum(cis.(λc)))
        @test all(_anr_cdist.(λc, c) .< 1.2)
        @test min(_anr_cdist(c, 0.3), _anr_cdist(c, 0.3 - π)) < 0.05
    end

    @testset "live R-hat reads every angle in its representative" begin
        p, _ = _anr_gd()
        L = p.layout
        ii = findfirst(==("i_star"), L.unfrozen_names)
        il = findfirst(==("lambda_k1"), L.unfrozen_names)
        fill!(L.spin_axis, 0.0); L.spin_axis[3] = 1.0
        rng = MersenneTwister(2)
        nw, ns = 16, 250
        M = zeros(ns * nw, 2)
        for s in 1:ns, w in 1:nw
            r = (s - 1) * nw + w
            i = 0.8 + 0.05randn(rng); λ = 3.0 + 0.05randn(rng)  # λ near +π
            iseven(w) && (i = π - i; λ = λ + π)                  # spin mirror, past the seam
            w % 3 == 0 && (i = -i)                               # fold mirror
            M[r, ii] = i; M[r, il] = λ
        end
        raw = Nereus._pt_convergence(M, size(M, 1), nw, [ii, il])
        can = Nereus._pt_convergence(M, size(M, 1), nw, [ii, il]; canon = AngleCanon(p))
        @test raw[2] > 1.5
        @test can[2] < 1.05
        # a full-circle angle split by its seam (half the walkers a period up)
        tg = _anr_ew()
        q = tg.params
        iw = findfirst(==("w_k1"), q.layout.unfrozen_names)
        X = randn(rng, ns * nw, length(q.layout.unfrozen_names))
        X[:, iw] .= π .+ 0.05 .* randn(rng, ns * nw)
        X[1:2:end, iw] .-= 2π
        cols = [iw]
        @test Nereus._pt_convergence(X, size(X, 1), nw, cols)[2] > 1.5
        @test Nereus._pt_convergence(X, size(X, 1), nw, cols; canon = AngleCanon(q))[2] < 1.05
    end

    @testset "a rail against a sub-range angle prior says it is the prior's" begin
        rng = MersenneTwister(3)
        n = 2000
        ch = Chains(reshape(hcat(1.5 .- 0.02 .* abs.(randn(rng, n)), randn(rng, n)), n, 2, 1),
                    [:w_k1, :K_k1])
        r = assess_fit(ch; prior_bounds = Dict(:w_k1 => (0.0, 1.5), :K_k1 => (-10.0, 10.0)),
                       angle_range = Dict("w_k1" => "[0, 2π)"))
        msg = only(c.message for c in r.checks if c.name === :prior_rail)
        @test occursin("w_k1", msg) && occursin("angle held by its own prior", msg)
        @test occursin("[0, 2π)", msg)
    end

    # The drift: a WrappedUniform λ is walled, its walkers stay on the chart.
    @testset "RM λ at 179°: no drift, no wall, the truth recovered" begin
        λt = deg2rad(179.0)
        P, Tc = 3.5, 2459000.0
        rng = MersenneTwister(11)
        sims = [Nereus.simulate_rm_night(; P = P, Tc = Tc + j * P, a_Rs = 8.0, λ = λt,
                                         vsini = 12.0, rr = 0.11, T14 = 3.0, sigma_rv = 6.0,
                                         b = 0.3, tag = "N$j", rng = rng) for j in 0:1]
        d, names = Nereus.obliquity_data([s.night for s in sims])
        p = Nereus.obliquity_params(d, names; P = P, Tc = Tc, b = (0.3, 0.03),
                                    a_Rs = (8.0, 0.2), rr = (0.11, 0.003),
                                    vsini = (12_000.0, 2000.0), arome = true,
                                    sigma0 = Dict(s.night.tag => s.night.σ0 for s in sims),
                                    rv_noise = :white)
        il = findfirst(==("lambda_k1"), p.layout.unfrozen_names)
        @test p.layout.unfrozen_priors[il].dist isa Nereus.WrappedUniform
        tg = NereusTarget(p, d)
        res = sample_pt_emcee(tg, d; n_temps = 4, n_walkers = 24, n_steps = 1200,
                              n_burnin = 600, seed = 3, show_progress = false,
                              bridge_headline = false)
        lo, hi = bounds(p.layout.unfrozen_priors[il])
        raw = vec(Array(res.chains[:, :lambda_k1, :]))
        @test all(lo .<= raw .<= hi)                  # was: 88% outside, |λ| ~ 1e10
        @test lo != -π                                # the seam moved off ±180°
        @test maximum(raw) - minimum(raw) < π         # one contiguous lobe
        @test quantile(raw, 0.025) < lo + mod(λt - lo, 2π) < quantile(raw, 0.975)
    end

    @testset "ω ≈ π and Mo ≈ 0 under :ew, a Normal on ω: no rail" begin
        tg = _anr_ew(; ω = -π + 0.03, Mo = 0.02,
                     wprior = Dict{String, Any}("type" => "NormalPrior", "args" => [3.0, 1.0]))
        res = sample_pt_emcee(tg, tg.data; n_temps = 6, n_walkers = 24, n_steps = 1500,
                              n_burnin = 750, seed = 3, show_progress = false,
                              bridge_headline = false)
        ch = res.chains
        Nereus.recenter_circular!(ch, tg.params; transforms = (tg.transform,))
        for (nm, tv) in (("w_k1", -π + 0.03), ("Mo_k1", 0.02))
            v = vec(Array(ch[:, Symbol(nm), :]))
            @test maximum(v) - minimum(v) < π
            t = median(v) + rem2pi(tv - median(v), RoundNearest)
            @test quantile(v, 0.025) < t < quantile(v, 0.975)
        end
        summary = Dict{String, Any}()
        Nereus._run_fit_health!(Dict{String, Any}("sampler" => "pt_emcee"), ch,
                                tg.params, summary)
        @test summary["fit_health"]["checks"]["prior_rail"]["status"] == "ok"
    end

    @testset "ensemble re-cuts a seam-centred angle at burn-in" begin
        tg = _anr_ew(; ω = 1.0, Mo = 0.02)
        im = findfirst(==("Mo_k1"), tg.params.layout.unfrozen_names)
        ch = sample_ensemble(tg; n_walkers = 32, n_steps = 4000, n_burnin = 2000,
                             seed = 5, show_progress = false)
        lo, _ = bounds(tg.params.layout.unfrozen_priors[im])
        @test lo != 0.0
        v = vec(Array(ch[:, :Mo_k1, :]))
        @test maximum(v) - minimum(v) < π
        @test _anr_cdist(median(v), 0.02) < 0.3
    end
end
