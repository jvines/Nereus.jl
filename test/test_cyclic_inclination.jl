# The orbital inclination is cyclic (src/circular.jl, `WrappedSine` in
# src/priors.jl).
#
# The bug this guards against: `SinePrior()` is sin(i)/2 on [0, π], and the
# unconstrained samplers reached it through a logit, which puts a wall at 0 and
# at π. A retrograde, nearly face-on orbit (i near 180°) has its posterior up
# against that wall: the draws piled at π and fit_health reported the
# inclination railed at its upper bound, though i = 180° is a face-on orbit, a
# point the orbit passes through, not an edge. The inclination is now sampled
# on a circle x, density |sin x|/4, where x, -x and x + 2π are one orbit, and
# the physical inclination is fold(x) = |rem2pi(x, RoundNearest)| in [0, π].
#
# The assertions pin: the distribution on its chart; that a prior over all of
# [0, π] on an inclination is converted, in the layout and nowhere else;
# that every likelihood sees the folded value; that the induced prior on i is
# still sin(i)/2; and that a fit near i = 180° samples across π, comes back
# folded, recovers the inclination and is not reported railed.

using Nereus, Distributions, LogDensityProblems, MCMCChains, Random, Statistics, Test
using Nereus: WrappedSine, WrappedSinePrior, fold_inclination, is_cyclic_inclination,
              is_wrapped, IADData

# A Gaia-4-like astrometric companion seen in the Hipparcos-style abscissae
# (the IAD template of test_node_flip.jl): a ≈ 1.2 AU, 0.01 M_sun, 220 scans
# over ~5 yr, here on a retrograde orbit 8° from face-on. The orbit is circular
# and the scans are at 0.02 mas, so the projected ellipse gives cos i to a few
# thousandths. (With e free, a face-on eccentric orbit and an inclined circular
# one project to similar ellipses: at e = 0.35 and 0.1 mas the posterior on i
# spans 134° to 174°.) The abscissae come from Nereus's own forward model.
const _CI_TRUTH = Dict("a_k1" => 1.18, "M_sec_k1" => 0.0105,
                       "inc_k1" => deg2rad(172.0), "Omega_k1" => 3.07,
                       "Mo_k1" => 0.5, "plx" => 13.6)
const _CI_RNG = MersenneTwister(11)
const _CI_N = 220
const _CI_SIG = 0.02
const _CI_T = sort(57000 .+ 1800 .* rand(_CI_RNG, _CI_N))
const _CI_PSI = 2π .* rand(_CI_RNG, _CI_N)
const _CI_NOISE = _CI_SIG .* randn(_CI_RNG, _CI_N)

function _ci_target(w)
    dt = (_CI_T .- mean(_CI_T)) ./ 365.25
    pf = sin.(2π .* (_CI_T .- 57000) ./ 365.25 .- _CI_PSI)
    iad = IADData(t = _CI_T, abscissa = w, abscissa_err = fill(_CI_SIG, _CI_N),
                  psi = _CI_PSI, parallax_factor = pf, pm_factor = dt)
    blk = (a = LogUniformPrior(0.3, 4.0), M_sec = LogUniformPrior(0.001, 0.05),
           sesinw = FixedPrior(0.0), secosw = FixedPrior(0.0), inc = SinePrior(),
           Omega = UniformPrior(0.0, 2π), Mo = UniformPrior(0.0, 2π))
    return build_target(M_pri = 0.644, planets = (b = blk,), iad = iad,
                        plx = NormalPrior(13.6, 0.02), M_s = 0.644)
end

function _ci_iad_target()
    tg0 = _ci_target(zeros(_CI_N))
    th = Theta{Float64}(tg0.params)
    for (nm, v) in _CI_TRUTH
        set_param!(th, nm, v)
    end
    _, orbs, Ms = Nereus._iad_active_orbits(th, Nereus.astrom_M_pri(th),
                                             Nereus.astrom_plx(th), tg0.data.t_ref)
    r = zeros(_CI_N)
    Nereus._iad_residuals!(r, tg0.data.iad, orbs, Ms, Nereus.astrom_plx(th))
    return _ci_target(-r .+ _CI_NOISE)
end

@testset "cyclic inclination" begin

    @testset "fold_inclination" begin
        for x in (0.0, 0.3, π / 2, 2.9, Float64(π))
            @test fold_inclination(x) === x               # [0, π] is its own representative
            @test fold_inclination(-x) === x              # the mirror, to the bit
            @test fold_inclination(x + 2π) ≈ x atol = 4e-15
            @test fold_inclination(2π - x) ≈ x atol = 4e-15
            @test fold_inclination(x - 4π) ≈ x atol = 8e-15
        end
        δ = 0.07
        @test fold_inclination(π + δ) ≈ π - δ atol = 1e-15
        @test fold_inclination(-δ) === δ
        @test all(x -> 0 <= fold_inclination(x) <= π, range(-20.0, 20.0; length = 2001))
    end

    @testset "WrappedSine" begin
        d = WrappedSine()
        lo, hi = minimum(d), maximum(d)
        @test (lo, hi) == (-π / 2, 3π / 2)
        @test hi - lo == Float64(2π)                     # one period, exactly
        # log|sin x| - log 4 on the chart: periodic (the chart is one period
        # of it, so a value and its relabelling into the chart have the same
        # density) and mirror-symmetric. 0 and π are interior points, where
        # the density is zero but there is no wall on either side.
        f(x) = log(abs(sin(x))) - log(4)
        for x in (-1.2, -0.4, -0.01, 0.01, 0.3, 1.0, 2.0, 3.0, π - 0.01, π + 0.01,
                  4.0, 4.6)
            @test logpdf(d, x) ≈ f(x) atol = 1e-14
            for k in (-3, -1, 1, 2)
                @test logpdf(d, Nereus.circular_relabel(x + 2π * k, lo, hi)) ≈
                      logpdf(d, x) atol = 1e-12
            end
            -x >= lo && @test logpdf(d, -x) == logpdf(d, x)
            2π - x <= hi && @test logpdf(d, 2π - x) ≈ logpdf(d, x) atol = 1e-12
        end
        @test logpdf(d, 0.0) == -Inf
        @test logpdf(d, Float64(π)) == f(Float64(π))     # sin(π) rounds to 1.2e-16
        @test isfinite(logpdf(d, lo)) && isfinite(logpdf(d, hi))
        # Zero past the chart's edges (the mirror image of i = 90°): on the
        # whole line the periodic density would be improper.
        @test logpdf(d, lo - 1e-9) == -Inf && logpdf(d, hi + 1e-9) == -Inf
        @test logpdf(d, 10.0) == -Inf && logpdf(d, -10.0) == -Inf
        @test logpdf(d, Inf) == -Inf && logpdf(d, NaN) == -Inf
        @test insupport(d, 0.0) && insupport(d, Float64(π)) && !insupport(d, 5.0)
        # Normalised over one period: ∫|sin x|/4 dx = 1 (midpoint rule).
        n = 200_000
        h = (hi - lo) / n
        @test sum(exp(logpdf(d, lo + (k - 0.5) * h)) for k in 1:n) * h ≈ 1 atol = 1e-9
        # cdf and quantile are inverse, monotone, and put a quarter of the mass
        # in each half-arch.
        @test cdf(d, lo) == 0 && cdf(d, hi) == 1
        @test cdf(d, 0.0) ≈ 0.25 && cdf(d, π / 2) ≈ 0.5 && cdf(d, Float64(π)) ≈ 0.75
        @test quantile(d, 0.0) == lo && quantile(d, 1.0) == hi
        @test quantile(d, 0.25) ≈ 0 atol = 1e-15
        @test quantile(d, 0.5) ≈ π / 2
        @test quantile(d, 0.75) ≈ π
        us = range(0.001, 0.999; length = 999)
        xs = quantile.(d, us)
        @test issorted(xs)
        @test all(lo .<= xs .<= hi)
        @test maximum(abs.(cdf.(d, xs) .- us)) < 1e-12
        @test maximum(abs.(quantile.(d, cdf.(d, xs)) .- xs)) < 1e-9
        # cdf is the integral of the density (midpoint rule, up to each x).
        for x in (-1.0, 0.5, 2.0, 3.5, 4.5)
            m = 20_000
            hx = (x - lo) / m
            @test sum(exp(logpdf(d, lo + (k - 0.5) * hx)) for k in 1:m) * hx ≈
                  cdf(d, x) atol = 1e-8
        end
        rng = MersenneTwister(3)
        r = [rand(rng, d) for _ in 1:20_000]
        @test all(lo .<= r .<= hi)
        @test median(r) ≈ π / 2 atol = 0.05

        ps = WrappedSinePrior()
        @test is_wrapped(ps) && is_cyclic_inclination(ps)
        @test !is_cyclic_inclination(SinePrior()) && !is_wrapped(SinePrior())
        @test !is_cyclic_inclination(WrappedUniformPrior(0.0, 2π))
        @test Nereus.bounds(ps) == (lo, hi)
        @test Nereus.in_support(ps, Float64(π)) && Nereus.in_support(ps, -0.3)
        @test !Nereus.in_support(ps, 10.0) && !Nereus.in_support(ps, Inf)
        # The packed prior evaluates the same density, and the same zero past
        # the chart.
        pk = Nereus.pack_priors([ps])
        @test pk.type_ids[1] == Nereus.PACKED_WRAPPED_SINE
        for x in (-3.0, -0.5, 0.2, 2.5, Float64(π), 3.3, 4.0, 9.0)
            @test Nereus.eval_packed_logpdf(x, pk.type_ids[1], pk.params[1, 1], pk.params[1, 2],
                                            pk.lowers[1], pk.uppers[1]) ≈
                  logpdf(d, x) atol = 1e-14
        end
    end

    @testset "serialisation round trip" begin
        for p in (WrappedSinePrior(), SinePrior())
            q = Nereus.prior_from_dict(Nereus.prior_to_dict(p))
            @test typeof(q) == typeof(p)
            @test Nereus.bounds(q) == Nereus.bounds(p)
            @test all(x -> logpdf(q.dist, x) == logpdf(p.dist, x), (0.3, 1.7, 2.9))
        end
    end

    # An RVAS planet with RV, relative astrometry and HGCA, as most astrometric
    # fits are set up.
    instruments = InstrumentConfig(rv = ["HARPS"], pm = String[])
    t_rv = collect(55000.0:30.0:55360.0)
    relast = RelAstromData(t = [55000.0, 55090.0, 55180.0, 55270.0],
        ra_off = [10.0, 8.0, -5.0, -10.0], dec_off = [0.0, 6.0, 9.0, 0.0],
        ra_err = ones(4), dec_err = ones(4))
    hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
        pmra = (5.0, 4.95, 4.9), pmdec = (-3.0, -3.05, -3.1),
        sigma_pmra = (0.2, 0.2, 0.2), sigma_pmdec = (0.2, 0.2, 0.2),
        plx = 25.0, plx_err = 0.05, hip_id = 1)
    data = Data(t_rv = t_rv, rv = 3.0 .* sin.(t_rv ./ 50), rv_err = ones(length(t_rv)),
                relastrom = relast, hgca = hgca)
    _params(priors = Dict{String,PriorSpec}()) =
        Params(parametrization = ParametrizationConfig(time = :Mo), max_kplanet = 1, planet_modes = [RVAS], instruments = instruments,
               data = data, stability = :none, M_s = 1.0, priors = priors)

    @testset "layout: a prior over all of [0, π] makes the inclination cyclic" begin
        p = _params()
        i = findfirst(==("inc_k1"), p.layout.unfrozen_names)
        @test p.config.priors["inc_k1"].dist isa Nereus.Sine      # as written
        @test p.layout.unfrozen_priors[i].dist isa WrappedSine
        @test p.layout.packed_priors.type_ids[i] == Nereus.PACKED_WRAPPED_SINE
        @test Nereus.cyclic_inclination_indices(p) == [i]
        @test collect(Nereus.cyclic_inclination_names(p)) == ["inc_k1"]
        tg = NereusTarget(p, data)
        @test tg.transform.type_ids[i] == 0                       # identity: no wall
        # Not a full-circle angle: it is folded, not re-charted.
        @test !(i in Nereus.circular_indices(p))

        # Re-pinned: a Uniform or a Normal covering all of [0, π] is lifted too
        # (`PolarFoldPrior`: the density reflected at both poles onto the same
        # chart, identity transform), so an inclination at 0 or π is never
        # against a wall whichever prior was written. A prior on part of
        # [0, π] keeps its walls.
        for other in (UniformPrior(0.0, π), NormalPrior(2.0, 0.2, 0.0, π))
            q = _params(Dict{String,PriorSpec}("inc_k1" => other))
            j = findfirst(==("inc_k1"), q.layout.unfrozen_names)
            @test q.layout.unfrozen_priors[j].dist isa Nereus.PolarFold
            @test q.layout.unfrozen_priors[j].dist.base === q.config.priors["inc_k1"]
            @test Nereus.cyclic_inclination_indices(q) == [j]
            @test NereusTarget(q, data).transform.type_ids[j] == 0
        end
        q = _params(Dict{String,PriorSpec}("inc_k1" => UniformPrior(1.0, 2.0)))
        j = findfirst(==("inc_k1"), q.layout.unfrozen_names)
        @test q.layout.unfrozen_priors[j] === q.config.priors["inc_k1"]
        @test isempty(Nereus.cyclic_inclination_indices(q))
        @test NereusTarget(q, data).transform.type_ids[j] != 0
    end

    @testset "trans-dim convergence gate reads the inclination folded" begin
        p = _params()
        cols = vcat(Symbol.(p.layout.unfrozen_names), [:n_planets, :planet_active_1])
        n_w, n_s = 16, 60
        rng = MersenneTwister(4)
        X = randn(rng, n_w * n_s, length(cols))
        X[:, findfirst(==(:n_planets), cols)] .= 1.0
        X[:, findfirst(==(:planet_active_1), cols)] .= 1.0
        ji = findfirst(==(:inc_k1), cols)
        X[:, ji] .= 2.5 .+ 0.05 .* randn(rng, n_w * n_s)
        X[1:2:end, ji] .= 2π .- X[1:2:end, ji]     # half the walkers in the mirror
        gate = Symbol[:n_planets; [c for c in cols if occursin(r"_k\d+$", String(c))]]
        fold = Nereus.cyclic_inclination_indices(p)
        @test fold == [ji]
        @test Nereus._td_convergence(X, size(X, 1), n_w, cols, gate, p; fold)[2] < 1.1
        @test Nereus._td_convergence(X, size(X, 1), n_w, cols, gate, p)[2] > 1.5
    end

    # Re-pinned (was "stays bounded"): the stellar inclination is a polar angle
    # too. Gravity darkening, its only reader, sees it through `system_i_star`,
    # folded like `planet_inc` -- x, -x and x + 2π are one star (the GD map,
    # disc mean and v_eq depend on i★ through sin² and cos² and the folded
    # angle) -- so a Sine prior on it is sampled on the circle as well.
    @testset "Sine on the stellar inclination is cyclic too" begin
        t_phot = collect(range(-0.1, 0.1; length = 60))
        d2 = Data(t_rv = t_rv, rv = zeros(length(t_rv)), rv_err = ones(length(t_rv)),
                  t_phot = t_phot, flux = ones(60), flux_err = fill(1e-3, 60),
                  phot_inst = ones(Int, 60))
        p = Params(parametrization = ParametrizationConfig(time = :Mo), max_kplanet = 1, planet_modes = [Nereus.RVPM_GD],
                   instruments = InstrumentConfig(rv = ["HARPS"], pm = ["TESS"]),
                   data = d2, stability = :none, M_s = 1.6, R_s = 1.5,
                   priors = Dict{String,PriorSpec}("i_star" => SinePrior()))
        j = findfirst(==("i_star"), p.layout.unfrozen_names)
        @test p.config.priors["i_star"].dist isa Nereus.Sine         # as written
        @test p.layout.unfrozen_priors[j].dist isa WrappedSine
        @test p.layout.packed_priors.type_ids[j] == Nereus.PACKED_WRAPPED_SINE
        @test Nereus.cyclic_inclination_indices(p) == [j]
        @test NereusTarget(p, d2).transform.type_ids[j] == 0       # identity: no wall
    end

    @testset "planet_inc and every likelihood see the folded value" begin
        p = _params()
        th = Theta{Float64}(p)
        for (nm, v) in ("P_k1" => 365.25, "M_sec_k1" => 0.01, "Omega_k1" => 0.5,
                        "Mo_k1" => 1.0, "plx" => 25.0)
            haskey(p.layout.name_to_idx, nm) && set_param!(th, nm, v)
        end
        slot = p.layout.name_to_idx["inc_k1"]
        val(x) = (th.values[slot] = x; th)
        lo, hi = Nereus.bounds(WrappedSinePrior())
        lls(t) = (rv_log_likelihood(t, data),
                  Nereus.relastrom_log_likelihood(t, data),
                  Nereus.hgca_log_likelihood(t, data))
        for x in (deg2rad(35.0), deg2rad(172.0))
            @test Nereus.planet_inc(val(x), 1) === x
            @test Nereus.planet_inc(val(-x), 1) === x
            @test Nereus.planet_inc(val(x + 2π), 1) ≈ x atol = 4e-15
            @test Nereus.planet_inc(val(2π - x), 1) ≈ x atol = 4e-15
            @test Nereus.planet_inc(val(π + 0.05), 1) ≈ π - 0.05 atol = 1e-15
            # RV, relative astrometry and HGCA: every representative.
            ref = lls(val(x))
            @test all(isfinite, ref)
            @test lls(val(-x)) == ref                       # bit for bit
            @test all(lls(val(2π - x)) .≈ ref)
            @test all(lls(val(x + 2π)) .≈ ref)
            @test all(lls(val(x - 2π)) .≈ ref)
            # The prior: the same at the mirror on the chart (-x at 35°,
            # 2π - x at 172°), and zero past the chart.
            m = -x >= lo ? -x : 2π - x
            @test lo <= m <= hi
            p0 = Nereus.log_prior(val(x))
            @test isfinite(p0) && Nereus.log_prior(val(m)) ≈ p0
            @test Nereus.log_prior(val(x + 2π)) == -Inf
        end

        # The Hipparcos IAD, on the IAD target used below. A transiting
        # planet has no inclination slot (its i comes from b), so the
        # transit likelihood, 0 here, must not move either.
        tg = _ci_iad_target()
        q = tg.params
        th = Theta{Float64}(q)
        for (nm, v) in _CI_TRUTH
            set_param!(th, nm, v)
        end
        slot = q.layout.name_to_idx["inc_k1"]
        x = _CI_TRUTH["inc_k1"]
        f(v) = (th.values[slot] = v;
                (Nereus.iad_log_likelihood(th, tg.data),
                 Nereus.transit_log_likelihood(th, tg.data),
                 rv_log_likelihood(th, tg.data)))
        ref = f(x)
        @test isfinite(ref[1]) && ref[1] != 0
        @test f(-x) == ref
        @test all(f(2π - x) .≈ ref)
        @test all(f(x + 2π) .≈ ref)
        # The whole log density, as the samplers see it: the same at the
        # mirror 2π - x on the chart, and -Inf past the chart (-x, at -172°).
        y = Float64[th.values[k] for k in q.layout.unfrozen_idx]
        k = findfirst(==("inc_k1"), q.layout.unfrozen_names)
        bt = NereusTarget(q, tg.data; unconstrained = false)
        lp(v) = (z = copy(y); z[k] = v; LogDensityProblems.logdensity(bt, z))
        @test isfinite(lp(x)) && lp(2π - x) ≈ lp(x)
        @test lp(-x) == -Inf
    end

    @testset "induced prior on the folded inclination is sin(i)/2" begin
        tg = NereusTarget(_params(), data)
        k = only(Nereus.cyclic_inclination_indices(tg.params))
        rng = MersenneTwister(11)
        n = 20_000
        # The samplers' prior draws (folded there) and raw draws on the chart
        # folded here: both must follow F(i) = (1 - cos i)/2 on [0, π].
        a = sort([Nereus._draw_from_prior(tg, rng)[k] for _ in 1:2_000])
        b = sort([fold_inclination(rand(rng, WrappedSine())) for _ in 1:n])
        ks(s) = maximum(abs.((1 .- cos.(s)) ./ 2 .- (1:length(s)) ./ length(s)))
        @test all(0 .<= a .<= π) && all(0 .<= b .<= π)
        @test ks(a) < 1.63 / sqrt(length(a))               # KS, 1% level
        @test ks(b) < 1.63 / sqrt(n)
        @test mean(cos.(b)) ≈ 0 atol = 0.02
    end

    @testset "fit_health: an inclination at 180° is not a rail" begin
        # Draws piled against i = π, as a nearly face-on retrograde orbit's are.
        # Under the bounded prior that is exactly what the rail check flags;
        # on the circle π is a point the orbit passes through.
        tg = _ci_iad_target()
        p = tg.params
        rng = MersenneTwister(4)
        nms = p.layout.unfrozen_names
        n_it, n_ch = 500, 4
        arr = Array{Float64}(undef, n_it, length(nms) + 1, n_ch)
        for (j, nm) in enumerate(nms)
            v = get(_CI_TRUTH, nm, 0.0)
            arr[:, j, :] .= nm == "inc_k1" ?
                π .- abs.(0.02 .* randn(rng, n_it, n_ch)) :
                v .+ 1e-3 .* abs(v) .* randn(rng, n_it, n_ch)
        end
        arr[:, end, :] .= 100 .+ randn(rng, n_it, n_ch)
        ch = Chains(arr, vcat(Symbol.(nms), :lp))
        summary = Dict{String,Any}()
        Nereus._run_fit_health!(Dict{String,Any}("sampler" => "pt_emcee"), ch, p,
                                summary)
        rail = summary["fit_health"]["checks"]["prior_rail"]
        @test rail["status"] == "ok"
        @test !occursin("inc_k1", rail["message"])
        # The same draws against the bounded prior's [0, π] do rail.
        r = assess_fit(ch; prior_bounds = Dict(:inc_k1 => (0.0, Float64(π))),
                       param_names = [:inc_k1])
        c = only(c for c in r.checks if c.name === :prior_rail)
        @test c.status !== :ok && occursin("inc_k1", c.message)
        # The science table does not flag it either.
        entries, _ = Nereus.science_fitted(ch, p)
        @test get(Dict(entries)["inc_k1"], "railed", false) === false
    end

    @testset "a fit near i = 180° samples across π" begin
        tg = _ci_iad_target()
        lo, hi = Nereus.bounds(WrappedSinePrior())
        res = sample_pt_emcee(tg, tg.data; n_temps = 8, n_walkers = 32,
                              n_steps = 1500, n_burnin = 750, seed = 1,
                              show_progress = false)
        ch = res.chains
        raw = vec(Array(ch[:, :inc_k1, :]))
        # The walkers crossed π (and nothing stopped them there): raw draws on
        # both sides of it. Each side is the other's mirror, so the folded
        # posterior is the same whichever side a walker was on. And they stay
        # on the chart: with the periodic density on the whole line, the hot
        # rungs spread over every period of it (-4.7e3 to 2.1e4 rad here).
        @test all(lo .<= raw .<= hi)
        @test count(>(π), raw) > 0.05 * length(raw)
        @test count(<(π), raw) > 0.3 * length(raw)
        Nereus.recenter_circular!(ch, tg.params; transforms = (tg.transform,))
        inc = vec(Array(ch[:, :inc_k1, :]))
        @test all(0 .<= inc .<= π)
        @test inc == fold_inclination.(raw)
        # Recovers the inclination, its mass running up to 180°.
        i0 = _CI_TRUTH["inc_k1"]
        @test quantile(inc, 0.025) < i0 < quantile(inc, 0.975)
        @test quantile(rad2deg.(inc), 0.99) > 175
        summary = Dict{String,Any}()
        Nereus._run_fit_health!(Dict{String,Any}("sampler" => "pt_emcee"), ch,
                                tg.params, summary)
        @test !occursin("inc_k1", summary["fit_health"]["checks"]["prior_rail"]["message"])
    end
end
