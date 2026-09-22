# Circular parameters: the 0/2π seam of a full-circle prior is a chart, and the
# chart is moved so the seam sits in the emptiest arc (src/circular.jl).
#
# The bug this guards against: a posterior centred on 0 ≡ 2π came back as two
# lobes at opposite ends of [0, 2π), every linear summary of it was garbage, and
# fit_health reported "Mo_k1: median railed at upper bound (median 6.264, bound
# 6.283); Omega_k1: median railed at lower bound (median 0.01433, bound 0.0)" —
# a wall where the circle has none. The assertions below pin both halves of the
# fix: the samplers re-chart the angle, and everything downstream sees one
# contiguous posterior.

using Nereus, MCMCChains, Random, Statistics, Test

# Shortest signed angular distance a - b, in (-π, π].
_cdist(a, b) = rem2pi(a - b, RoundNearest)

@testset "circular parameters" begin

    _blk(; Mo = UniformPrior(0.0, 2π)) =
        (P = LogUniformPrior(12.0, 12.6), K = LogUniformPrior(1.0, 100.0),
         sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
         Mo = Mo)
    _target(t, rv; Mo = UniformPrior(0.0, 2π)) = build_target(
        planets = (b = _blk(; Mo),),
        rv = (HARPS = (data = (t = t, rv = rv, rv_err = fill(2.0, length(t))),
                       sigma = LogUniformPrior(0.01, 10.0)),))

    # An eccentric planet (e = 0.3, so Mo is not degenerate with ω) whose mean
    # anomaly at t_ref sits 0.02 rad from the seam: the RV posterior on Mo is a
    # few hundredths of a radian wide, so it straddles 0 ≡ 2π.
    Mo_true = 0.02
    function _seam_data()
        rng = MersenneTwister(42)
        N = 60
        t = sort(55000 .+ 200 .* rand(rng, N))
        tg0 = _target(t, zeros(N))
        th = Theta{Float64}(tg0.params)
        e, ω = 0.3, 1.0
        for (nm, v) in ("P_k1" => 12.3, "K_k1" => 25.0, "sesinw_k1" => sqrt(e) * sin(ω),
                        "secosw_k1" => sqrt(e) * cos(ω), "Mo_k1" => Mo_true,
                        "gamma_HARPS" => 0.0, "sigma_HARPS" => 0.01)
            set_param!(th, nm, v)
        end
        model, _ = rv_predictions(th, tg0.data)
        # 3 m/s of jitter on top of the 2 m/s errors, so sigma is interior and
        # the only edge anything could rail on is Mo's seam.
        return t, model .+ sqrt(2.0^2 + 3.0^2) .* randn(rng, N)
    end

    # Synthetic chains: every column uniform junk except Mo_k1 ~ N(0.05, 0.1)
    # folded into [0, 2π) — two lobes, the shape the samplers used to return.
    function _split_chains(p; rng = MersenneTwister(3), n = 500, nc = 4,
                           μ = 0.05, σ = 0.1)
        nm = p.layout.unfrozen_names
        X = zeros(n, length(nm) + 1, nc)
        for c in 1:nc, s in 1:n, (j, name) in enumerate(nm)
            X[s, j, c] = name == "Mo_k1" ? mod2pi(μ + σ * randn(rng)) : rand(rng)
        end
        return Chains(X, vcat(Symbol.(nm), :lp))
    end

    @testset "which parameters are circular" begin
        tg = _target(collect(55000.0:10:55300.0), zeros(31))
        cfg = tg.params.config
        @test Nereus.circular_names(tg.params) == Set(["Mo_k1"])
        @test Nereus.is_circular("Omega_k2", UniformPrior(0.0, 2π), cfg)
        @test Nereus.is_circular("lambda_k1", UniformPrior(-π, π), cfg)
        @test Nereus.is_circular("w_k1", UniformPrior(-π, π), cfg)
        # The user's walls stay walls: two periods, an arc, a non-uniform density.
        @test !Nereus.is_circular("Mo_k1", UniformPrior(-2π, 2π), cfg)
        @test !Nereus.is_circular("Mo_k1", UniformPrior(0.0, π), cfg)
        @test !Nereus.is_circular("Mo_k1", NormalPrior(0.1, 0.05, 0.0, 2π), cfg)
        # Not angles on a circle, or not angles at all.
        @test !Nereus.is_circular("inc_k1", UniformPrior(0.0, 2π), cfg)
        @test !Nereus.is_circular("gp_act_lambda_e", UniformPrior(0.0, 2π), cfg)
        @test !Nereus.is_circular("omega_1", UniformPrior(0.0, 2π), cfg)
        # TTV epoch numbering hangs off Tp = t_ref − Mo·P/2π: Mo is not periodic
        # in the likelihood once TTVs are modelled. ω, Ω and λ still are.
        ttv = (; planet_modes = [Nereus.RVPM_TTV])
        @test !Nereus.is_circular("Mo_k1", UniformPrior(0.0, 2π), ttv)
        @test Nereus.is_circular("w_k1", UniformPrior(-π, π), ttv)
        ttvnb = (; planet_modes = [Nereus.RVPM, Nereus.RVPM_TTV_NB])
        @test !Nereus.is_circular("Mo_k2", UniformPrior(0.0, 2π), ttvnb)
    end

    @testset "where the seam goes" begin
        rng = MersenneTwister(11)
        # Already interior: the window does not move, bit for bit.
        @test Nereus.circular_cut(mod2pi.(2.0 .+ 0.1 .* randn(rng, 2000)), 0.0) === 0.0
        @test Nereus.circular_window(mod2pi.(2.0 .+ 0.1 .* randn(rng, 2000)), 0.0, 0.0) === 0.0
        # A flat posterior has no emptiest arc to speak of: noise must not move
        # it. Many draws of it, not one lucky seed -- the least of 72 noisy arc
        # counts sits well below the typical one, and an earlier rule that
        # compared against the minimum moved a flat window in most runs.
        @test all(Nereus.circular_cut(2π .* rand(MersenneTwister(s), n), 0.0) === 0.0
                  for s in 1:10, n in (100, 1000, 5000))
        # Centred on the seam: the new seam lands opposite the mass.
        x = mod2pi.(0.05 .+ 0.1 .* randn(rng, 2000))
        c = Nereus.circular_cut(x, 0.0)
        @test abs(_cdist(c, 0.05 + π)) < 0.3
        # ... and the window is shifted by whole periods so the median sits in
        # the user's [0, 2π).
        lo = Nereus.circular_window(x, 0.0, 0.0)
        xs = [Nereus.circular_relabel(xi, lo, lo + 2π) for xi in x]
        @test 0 <= median(xs) < 2π
        @test maximum(xs) - minimum(xs) < π
        # Two modes 180° apart, one on the seam: the seam goes between them.
        bm = mod2pi.(vcat(0.1 .* randn(rng, 1000), π .+ 0.1 .* randn(rng, 1000)))
        c2 = Nereus.circular_cut(bm, 0.0)
        @test min(abs(_cdist(c2, 0.0)), abs(_cdist(c2, π))) > 1.2
    end

    @testset "relabelling" begin
        # Inside the window: untouched, bit for bit.
        @test Nereus.circular_relabel(0.3, -1.0, -1.0 + 2π) === 0.3
        @test Nereus.circular_relabel(6.2, -1.0, -1.0 + 2π) ≈ 6.2 - 2π
        @test Nereus.circular_relabel(-7.0, 0.0, 2π) ≈ -7.0 + 4π
        @test isnan(Nereus.circular_relabel(NaN, 0.0, 2π))
    end

    @testset "run_job / fit_* re-chart the returned draws" begin
        tg = _target(collect(55000.0:10:55300.0), zeros(31))
        p = tg.params
        i = findfirst(==("Mo_k1"), p.layout.unfrozen_names)
        ch = _split_chains(p)
        v0 = vec(Array(ch[:Mo_k1]))
        @test maximum(v0) - minimum(v0) > 6.0          # the two-lobe shape
        Nereus.recenter_circular!(ch, p; transforms = (tg.transform,))
        v = vec(Array(ch[:Mo_k1]))
        @test maximum(v) - minimum(v) < π              # one contiguous posterior
        @test median(v) ≈ 0.05 atol = 0.02
        @test std(v) ≈ 0.1 rtol = 0.1
        # Every copy of the bounds moved together; the user's prior did not.
        lo, hi = Nereus.bounds(p.layout.unfrozen_priors[i])
        @test hi - lo ≈ 2π
        @test (p.layout.packed_priors.lowers[i], p.layout.packed_priors.uppers[i]) == (lo, hi)
        @test (tg.transform.lowers[i], tg.transform.uppers[i]) == (lo, hi)
        @test Nereus.bounds(p.config.priors["Mo_k1"]) == (0.0, 2π)
        @test all(x -> lo <= x <= hi, v)
        # The prior density is unchanged at every relabelled draw.
        th = Theta{Float64}(p)
        for (j, idx) in enumerate(p.layout.unfrozen_idx)
            a, b = Nereus.bounds(p.layout.unfrozen_priors[j])
            th.values[idx] = (a + b) / 2
        end
        ref = Nereus.log_prior(th)
        @test all(v) do x
            th.values[p.layout.unfrozen_idx[i]] = x
            Nereus.log_prior(th) ≈ ref
        end
        # Idempotent: a second pass changes nothing.
        Nereus.recenter_circular!(ch, p)
        @test vec(Array(ch[:Mo_k1])) == v
    end

    @testset "draws outside the window are brought in" begin
        # OFTI folds Ω into [0, 2π) whatever the prior says; a [-π, π) prior then
        # gets draws above π. They must land back inside the layout window.
        tg = _target(collect(55000.0:10:55300.0), zeros(31);
                     Mo = UniformPrior(-π, π))
        p = tg.params
        ch = _split_chains(p; μ = 2.0, σ = 0.05)      # mod2pi'd: all in [0, 2π)
        Nereus.recenter_circular!(ch, p)
        v = vec(Array(ch[:Mo_k1]))
        lo, hi = Nereus.bounds(p.layout.unfrozen_priors[findfirst(==("Mo_k1"),
                                                                  p.layout.unfrozen_names)])
        @test all(x -> lo <= x <= hi, v)
        @test median(v) ≈ 2.0 atol = 0.02
    end

    @testset "fit_health measures a seam in the chart the engine used" begin
        tg = _target(collect(55000.0:10:55300.0), zeros(31))
        rail(r) = only(c for c in r.checks if c.name === :prior_rail)
        chk(ch, win) = rail(assess_fit(ch; prior_bounds = Dict(:Mo_k1 => win),
                                       param_names = [:Mo_k1], circular = ["Mo_k1"]))
        # The same posterior near 0, whatever chart the draws are stored in:
        # an engine that sampled in [0, 2π) had the seam through its mass
        # (two lobes, the user's case), one that moved its window off it did not.
        ch = _split_chains(tg.params)
        r = chk(ch, (0.0, 2π))
        @test r.status === :fail
        @test occursin("cut by its seam", r.message)
        @test chk(ch, (0.05 - π, 0.05 + π)).status === :ok
        # An engine that cannot cross the seam returns a truncated sliver (NUTS
        # on this posterior returned only the part below 2π). Recentring the
        # draws for reporting must not hide that: measured in the engine's
        # window it still piles against the seam.
        sliver = _split_chains(tg.params)
        j = findfirst(==(:Mo_k1), names(sliver))
        rng = MersenneTwister(9)
        sliver.value.data[:, j, :] .=
            2π .- abs.(0.03 .+ 0.02 .* randn(rng, size(sliver, 1), size(sliver, 3)))
        Nereus.recenter_circular!(sliver, tg.params)       # moves the reporting window
        @test chk(sliver, (0.0, 2π)).status === :fail
    end

    @testset "science table reports an angle near 0 as one interval" begin
        tg = _target(collect(55000.0:10:55300.0), zeros(31))
        p = tg.params
        ch = _split_chains(p)
        Nereus.recenter_circular!(ch, p)
        e = Nereus.sci_param_entry(ch, p, "Mo_k1")
        @test e.unit == "deg"
        @test e.stats.best ≈ rad2deg(0.05) atol = 1.5
        @test e.stats.ci1[2] - e.stats.ci1[1] < 20        # ~2σ = 11.5°, not ~360°
        # Ω and λ are angles too — reported in degrees, not raw radians.
        @test Nereus.sci_param_unit("Omega_k1", p) == ("deg", true)
        @test Nereus.sci_param_unit("lambda_k1", p) == ("deg", true)
    end

    @testset "derived ω straddling ±180° is one interval" begin
        tg = _target(collect(55000.0:10:55300.0), zeros(31))
        p = tg.params
        rng = MersenneTwister(5)
        nm = p.layout.unfrozen_names
        n = 2000
        X = zeros(n, length(nm) + 1, 1)
        for s in 1:n, (j, name) in enumerate(nm)
            ω = π + 0.05 * randn(rng)                  # retrograde-adjacent ω
            X[s, j, 1] = name == "sesinw_k1" ? sqrt(0.3) * sin(ω) :
                         name == "secosw_k1" ? sqrt(0.3) * cos(ω) :
                         name == "P_k1" ? 12.3 : name == "K_k1" ? 25.0 : rand(rng)
        end
        ch = Chains(X, vcat(Symbol.(nm), :lp))
        dp = compute_derived(ch, p)
        ωd = first(d for d in dp if haskey(d.values, "omega_deg")).values["omega_deg"]
        @test maximum(ωd) - minimum(ωd) < 180
        @test abs(rem(median(ωd) - 180, 360, RoundNearest)) < 2
    end

    @testset "pt_emcee re-charts a seam-centred Mo at burn-in" begin
        t, rv = _seam_data()
        tg = _target(t, rv)
        res = sample_pt_emcee(tg, tg.data; n_temps = 6, n_walkers = 32,
                              n_steps = 3000, n_burnin = 1500, seed = 3,
                              show_progress = false, bridge_headline = false)
        v = vec(Array(res.chains[:Mo_k1]))
        @test maximum(v) - minimum(v) < π
        @test abs(_cdist(median(v), Mo_true)) < 5 * std(v) + 0.02
        # The sampler moved the window off the mode.
        lo, _ = Nereus.bounds(tg.params.layout.unfrozen_priors[
            findfirst(==("Mo_k1"), tg.params.layout.unfrozen_names)])
        @test lo != 0.0
    end

    @testset "MAP at the seam is a converged fit, not a railed one" begin
        t, rv = _seam_data()
        tg = _target(t, rv)
        m = sample_map(tg; n_starts = 8, seed = 1)
        i = findfirst(==("Mo_k1"), m.param_names)
        @test !m.railed
        @test !("Mo_k1" in m.railed_params)
        # Right basin (the Mo posterior is ~0.09 rad wide here) ...
        @test abs(_cdist(m.x_map[i], Mo_true)) < 0.3
        # ... and interior to its window, not parked on a wall of the logit chart.
        lo, hi = Nereus.bounds(tg.params.layout.unfrozen_priors[i])
        @test lo + 1.0 < m.x_map[i] < hi - 1.0
    end
end
