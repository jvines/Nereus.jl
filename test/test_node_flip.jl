# The astrometric node flip (src/samplers/node_flip.jl).
#
# Astrometry alone is invariant under (Ω, ω) → (Ω + π, ω + π), so an
# astrometry-only planet has two modes of exactly equal mass. The stretch move
# cannot cross between them and the PT ensembles weighted them by how often a
# swap happened to carry a walker across: Gaia-4 came back 61/39 with R-hat
# 1.003, and the synthetic target below comes back 92/8 (seed 1) and 83/17
# (seed 2) from pt_emcee with the flip off, 92/8 from transdim_pt_emcee.
# With it on: 50.5, 50.1 and 48.9 %.
#
# The assertions pin: the move is offered exactly where it can be accepted; the
# mirror really has the same posterior density in Nereus's own likelihood (the
# premise the whole move stands on); T∘T = id; each sampler then weights the
# modes 50/50; and a fit without an astrometry-only planet is bit-identical.

using Nereus, Random, Statistics, Test
using Nereus: IADData, NodeFlip, node_flips, node_flip!, logdensity_bounded

@testset "astrometric node flip" begin

    # Gaia-4-like: a ≈ 1.2 AU, 0.01 M_sun companion, e = 0.35, 220 scans over
    # ~5 yr at 0.1 mas. The abscissae come from Nereus's own forward model, so
    # the test cannot disagree with the likelihood about conventions.
    _blk(; Omega = UniformPrior(0.0, 2π), sesinw = UniformPrior(-1.0, 1.0),
           secosw = UniformPrior(-1.0, 1.0)) =
        (a = LogUniformPrior(0.3, 4.0), M_sec = LogUniformPrior(0.001, 0.05),
         sesinw = sesinw, secosw = secosw, inc = SinePrior(), Omega = Omega,
         Mo = UniformPrior(0.0, 2π))
    const_rng = MersenneTwister(11)
    n = 220
    t   = sort(57000 .+ 1800 .* rand(const_rng, n))
    dt  = (t .- mean(t)) ./ 365.25
    psi = 2π .* rand(const_rng, n)
    pf  = sin.(2π .* (t .- 57000) ./ 365.25 .- psi)
    _iad(w) = IADData(t = t, abscissa = w, abscissa_err = fill(0.1, n), psi = psi,
                      parallax_factor = pf, pm_factor = dt)
    _target(w; rv = NamedTuple(), kw...) =
        build_target(M_pri = 0.644, planets = (b = _blk(; kw...),), iad = _iad(w),
                     rv = rv, plx = NormalPrior(13.6, 0.02), M_s = 0.644)
    w_obs = let tg0 = _target(zeros(n))
        th = Theta{Float64}(tg0.params)
        e, ω = 0.35, 1.0
        for (nm, v) in ("a_k1" => 1.18, "M_sec_k1" => 0.0105,
                        "sesinw_k1" => sqrt(e) * sin(ω), "secosw_k1" => sqrt(e) * cos(ω),
                        "inc_k1" => 2.11, "Omega_k1" => 3.07, "Mo_k1" => 0.5,
                        "plx" => 13.6)
            set_param!(th, nm, v)
        end
        _, orbs, Ms = Nereus._iad_active_orbits(th, Nereus.astrom_M_pri(th),
                                                 Nereus.astrom_plx(th), tg0.data.t_ref)
        r = zeros(n)
        Nereus._iad_residuals!(r, tg0.data.iad, orbs, Ms, Nereus.astrom_plx(th))
        -r .+ 0.1 .* randn(const_rng, n)
    end
    as_target() = _target(w_obs)

    # The same planet under another parametrization: rebuilt from the
    # astrometry-only target's own config with the eccentricity pair swapped.
    function _reparam(; ew = :sesinw, time = :Mo, set = Dict{String,PriorSpec}(),
                        drop = String[])
        tg = as_target()
        cfg = tg.params.config
        pri = copy(cfg.priors)
        foreach(k -> delete!(pri, k), drop)
        merge!(pri, set)
        p = Params(max_kplanet = 1, planet_modes = cfg.planet_modes,
                   instruments = cfg.instruments, data = tg.data, M_s = cfg.M_s,
                   parametrization = ParametrizationConfig(mass = cfg.parametrization.mass,
                                                           ew = ew, time = time),
                   priors = pri)
        return NereusTarget(p, tg.data)
    end
    ew_target() = _reparam(ew = :ew, drop = ["sesinw_k1", "secosw_k1"],
                           set = Dict{String,PriorSpec}("ecc_k1" => UniformPrior(0.0, 0.95),
                                                        "w_k1" => UniformPrior(0.0, 2π)))
    circ_target() = _reparam(set = Dict{String,PriorSpec}("sesinw_k1" => FixedPrior(0.0),
                                                          "secosw_k1" => FixedPrior(0.0)))

    pos(tg, nm) = findfirst(==(nm), tg.params.layout.unfrozen_names)
    in_mode(ch) = (O = vec(Array(ch[:, :Omega_k1, :]));
                   mean(abs.(rem2pi.(O .- 3.07, RoundNearest)) .< π / 2))

    @testset "offered exactly where the mirror can be accepted" begin
        tg = as_target()
        # build_target makes an astrometry-only planet RVAS; no RV data, so it
        # still qualifies.
        @test only(tg.params.config.planet_modes) == RVAS
        f = only(node_flips(tg.params, tg.data))
        @test f.shift == [pos(tg, "Omega_k1")]
        @test f.negate == [pos(tg, "sesinw_k1"), pos(tg, "secosw_k1")]

        te = ew_target()
        @test only(node_flips(te.params, te.data)).shift ==
              [pos(te, "Omega_k1"), pos(te, "w_k1")]
        # e ≡ 0: the in-plane phase is ω + M, so Mo carries the flip.
        tc = circ_target()
        fc = only(node_flips(tc.params, tc.data))
        @test fc.shift == [pos(tc, "Omega_k1"), pos(tc, "Mo_k1")] && isempty(fc.negate)

        # RVs break the symmetry: K[cos(ν + ω) + e cos ω] changes sign.
        tr = _target(w_obs; rv = (HARPS = (data = (t = t[1:30], rv = zeros(30),
                                                   rv_err = ones(30)),
                                           sigma = LogUniformPrior(0.01, 10.0)),))
        @test isempty(node_flips(tr.params, tr.data))
        # Ω on an arc has no Ω + π in its window; Tc is defined through ω.
        ta = _target(w_obs; Omega = UniformPrior(0.0, π))
        @test isempty(node_flips(ta.params, ta.data))
        tt = _reparam(time = :Tc, drop = ["Mo_k1"],
                      set = Dict{String,PriorSpec}("Tc_k1" => UniformPrior(57500.0, 58500.0)))
        @test isempty(node_flips(tt.params, tt.data))
    end

    # Worst |log π(Tx) − log π(x)| and |T(Tx) − x| over prior draws. Prior
    # draws fit the data terribly (log L ~ −2e4), which is what makes a small
    # asymmetry visible at all.
    function _mirror_error(tg)
        f = only(node_flips(tg.params, tg.data))
        rng = MersenneTwister(1)
        Δ = 0.0; inv = 0.0; n_ok = 0
        for _ in 1:100
            x = Nereus._draw_from_prior(tg, rng)
            y = node_flip!(copy(x), f, tg.params)
            a, b = logdensity_bounded(tg, x), logdensity_bounded(tg, y)
            if isfinite(a)
                Δ = max(Δ, abs(a - b)); n_ok += 1
            end
            inv = max(inv, maximum(abs.(node_flip!(copy(y), f, tg.params) .- x)))
        end
        return Δ, inv, n_ok
    end

    @testset "the mirror has the same posterior density, and T∘T = id" begin
        # The e ≡ 0 target flips Mo, so it moves Tp = t_ref − Mo·P/2π by half a
        # period: it is exact only if the orbit's period really is the sampled
        # P. It was not — `a_from_P` used the Julian year, PlanetOrbits the
        # G = 4π² one — and this case measured 0.2 nats off until
        # KEPLER_YEAR_DAYS fixed it (test/astrometry/test_projection.jl pins
        # the round trip). Prior draws fit the data terribly (log L ~ −2e4),
        # which is what makes an asymmetry that size visible at all.
        for tg in (as_target(), ew_target(), circ_target())
            Δ, inv, n_ok = _mirror_error(tg)
            @test n_ok > 50
            @test Δ < 1e-8           # measured 4e-12 (sesinw), 3e-11 (:ew)
            @test inv < 1e-12
        end
    end

    @testset "windows that are not exactly one period" begin
        # `is_circular` admits |span - 2π| within `_CIRCULAR_SPAN_RTOL`, in
        # BOTH directions, and the two directions need opposite handling.
        #
        # LONGER than a period -- `Uniform(0, 6.2832)` is over by 1.5e-5 rad,
        # and unlike Mo the builder does not cap Omega at 2π. No wrap-based map
        # is an involution there: T∘T carries a point near `hi` to x - 2π.
        # Measured before the gate: |T∘T(x) - x| = 6.28 at x = 6.2832. Refused.
        @test isempty(node_flips(_target(w_obs; Omega = UniformPrior(0.0, 6.2832)).params,
                                 _target(w_obs; Omega = UniformPrior(0.0, 6.2832)).data))

        # SHORT of a period: offered, and an involution everywhere. `node_flip!`
        # used `circular_relabel`, which PINS a value past `hi` to `hi`
        # (src/circular.jl) -- right for relabelling a chart, wrong inside a
        # MOVE, where it maps the whole gap onto one point: an atom, not an
        # involution. Wrapped by the period instead, T∘T = id unconditionally
        # (x + 2π ≡ x) and a flip into the gap is out of support and rejected.
        tg = _target(w_obs; Omega = UniformPrior(0.0, 2π - 1e-5))
        f  = only(node_flips(tg.params, tg.data))
        L  = tg.params.layout
        d  = only(f.shift)
        lo, hi = bounds(L.unfrozen_priors[d])
        η  = Nereus.CIRCULAR_PERIOD - (hi - lo)
        @test 0 < η < 1e-4

        x0 = Theta{Float64}(tg.params).values[L.unfrozen_idx]
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0 - 1e-12)
            x = copy(x0); x[d] = lo + frac * (hi - lo)
            y = node_flip!(copy(x), f, tg.params)
            @test abs(node_flip!(copy(y), f, tg.params)[d] - x[d]) < 1e-12
        end

        # The point whose image lands INSIDE the gap. The old clamp returned
        # exactly `hi`; the wrap must return a value strictly above it, so the
        # prior rejects rather than an atom forming on the edge.
        x = copy(x0); x[d] = lo + π - η / 2
        y = node_flip!(copy(x), f, tg.params)
        @test y[d] > hi
        @test !isapprox(y[d], hi; atol = 1e-12)
        @test abs(node_flip!(copy(y), f, tg.params)[d] - x[d]) < 1e-12
    end

    @testset "pt_emcee weights the two modes equally" begin
        tg = as_target()
        res = sample_pt_emcee(tg, tg.data; n_temps = 8, n_walkers = 32,
                              n_steps = 1200, n_burnin = 600, seed = 1,
                              show_progress = false)
        @test abs(in_mode(res.chains) - 0.5) < 0.08
    end

    @testset "pt_whitening weights the two modes equally" begin
        tg = as_target()
        res = sample_pt_whitening(tg, tg.data; n_steps = 1200, n_burnin = 600,
                                  seed = 1, show_progress = false)
        @test abs(in_mode(res.chains) - 0.5) < 0.08
    end

    @testset "transdim_pt_emcee weights the two modes equally" begin
        tg = as_target()
        # At the DEFAULT informed_birth_fraction: the informed birth used to
        # take an astrometry-only fit down with "reducing over an empty
        # collection" from the RV periodogram it had no RVs to run.
        res = sample_transdim_pt_emcee(tg, tg.data; td = TransDimConfig(max_kplanet = 1),
                                       n_temps = 8, n_walkers = 32, n_steps = 1200,
                                       n_burnin = 600, seed = 1,
                                       show_progress = false)
        np = vec(Array(res.chains[:, :n_planets, :]))
        O = vec(Array(res.chains[:, :Omega_k1, :]))[np .>= 1]
        @test mean(np .>= 1) > 0.9
        @test abs(mean(abs.(rem2pi.(O .- 3.07, RoundNearest)) .< π / 2) - 0.5) < 0.08
    end

    @testset "a fit without an astrometry-only planet is bit-identical" begin
        rng = MersenneTwister(5)
        tv = sort(55000 .+ 300 .* rand(rng, 40))
        rv = 20 .* sin.(2π .* tv ./ 12.3) .+ 2 .* randn(rng, 40)
        mk() = build_target(planets = (b = (P = LogUniformPrior(10.0, 15.0),
                                            K = LogUniformPrior(1.0, 100.0),
                                            sesinw = UniformPrior(-1.0, 1.0),
                                            secosw = UniformPrior(-1.0, 1.0),
                                            Mo = UniformPrior(0.0, 2π)),),
                            rv = (HARPS = (data = (t = tv, rv = rv, rv_err = fill(2.0, 40)),
                                           sigma = LogUniformPrior(0.01, 10.0)),))
        run(nf) = (tg = mk(); sample_pt_emcee(tg, tg.data; n_temps = 4, n_walkers = 16,
                                               n_steps = 80, n_burnin = 40, seed = 3,
                                               node_flip = nf, show_progress = false))
        @test Array(run(0.0).chains) == Array(run(0.1).chains)
        @test_throws ArgumentError run(1.5)
    end
end
