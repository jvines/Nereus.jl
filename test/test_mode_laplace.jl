# mode_laplace_evidence, against its own stated assumption.
#
# It is reported by sample_pt_emcee and, before today, was SUBSTITUTED IN as the
# headline whenever the tempered path failed -- i.e. it produced the number on
# exactly the hard runs -- while having no test coverage at all.
#
# It cannot be checked the way bridge and reference-path are: those expose a
# core taking a raw log-density, so they can be pointed at a target whose log Z
# is known in closed form. This one takes a NereusTarget. So the test is its own
# premise instead: the estimator assumes the posterior is locally Gaussian about
# the mode, so on a posterior that genuinely IS near-Gaussian it must agree with
# bridge, which is validated against exact truth on curved 15-D targets.
#
# Measured on 51 Peg -- unimodal, tightly determined, its friendliest real case
# -- converged, it sits 5.5 nats above bridge (an earlier run, before the fixes
# below, put it 24 below). That is why it is reported, and headlined only when
# there is neither a usable bridge nor a corroborated tempered value.
#
# On an RV + Hipparcos-IAD companion pt_emcee's value sat about 1.1 nats above
# bridge and pt_hmc's 2.6 to 3.1 below, on the same target: pt_emcee's `:lp`, the
# bounded log posterior, was read as the unconstrained one without the map's
# log-Jacobian (-3.7 nats at the mode), and both used the covariance of the
# draws nearest the mode as it came, which a cut to the inner 30% shrinks
# (-2.6 nats for d = 6). With both corrected, both land within 0.25 of bridge.
using Test
using Nereus
using Random, Statistics, LinearAlgebra
using MCMCChains: Chains

# A high-SNR single Keplerian with these priors, its RVs at eccentricity `e0`
# (ω = 1). e0 = 0 is a pure sine.
function _ml_target(e0)
    rng = MersenneTwister(90210)
    n = 150
    t = sort!(600 .* rand(rng, n))
    P0, K0 = 4.23, 60.0
    rv = if e0 == 0
        K0 .* sin.(2π .* t ./ P0)
    else
        map(t) do ti
            M = mod2pi(2π * (ti - 1.3) / P0); E = M
            for _ in 1:60; E -= (E - e0 * sin(E) - M) / (1 - e0 * cos(E)); end
            ν = 2atan(sqrt((1 + e0) / (1 - e0)) * tan(E / 2))
            K0 * (cos(ν + 1.0) + e0 * cos(1.0))
        end
    end
    rv = rv .+ 1.5 .* randn(rng, n)
    return build_target(
        planets = (b = (P = LogUniformPrior(4.1, 4.4), K = LogUniformPrior(20.0, 120.0),
                        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                        Mo = UniformPrior(0.0, 2π)),),
        rv = (SIM = (data = (t = t, rv = rv, rv_err = fill(1.5, n)),
                     sigma = LogUniformPrior(0.5, 10.0)),),
    )
end

@testset "mode_laplace_evidence" begin
    # e = 0.3, well determined: the posterior is tight and close to Gaussian in
    # unconstrained space, which is where Laplace should be at its best.
    target = _ml_target(0.3)
    res = sample_pt_emcee(target, target.data; n_temps = 12, n_walkers = 30,
                         n_steps = 2000, n_burnin = 1000, seed = 11,
                         show_progress = false)

    lap = res.log_evidence_laplace
    br  = res.log_evidence_bridge
    @test isfinite(lap)
    @test isfinite(br)

    # Its premise: where the posterior is near-Gaussian, Laplace tracks the
    # validated estimator, and does not depend on how many of the draws nearest
    # the mode its covariance is taken from. Measured: 0.03 to 0.15 nats from
    # bridge at mode_frac 0.1 to 1 (seeds 11, 12; e = 0.3 and 0.5). Before the
    # Jacobian and the cut's shrink were put in, the bar here was 40 nats.
    @test abs(lap - br) < 0.5
    lf = [mode_laplace_evidence(target, res.chains; mode_frac = f) for f in (0.1, 0.5, 1.0)]
    @test maximum(lf) - minimum(lf) < 0.3

    # Where it is not: the pure sine, e near 0, where the posterior in
    # (√e cos ω, √e sin ω) is flat-topped (log L is quartic in them there) and
    # Mo follows ω around. The value then climbs with mode_frac, from 1.6 to
    # 4.0 nats above bridge (2.7 at the default 0.3). The bias of a Gaussian
    # approximation, not of the estimator; the uncorrected formula sat 0.4 below
    # here only because its two errors cancelled.
    tc = _ml_target(0.0)
    rc = sample_pt_emcee(tc, tc.data; n_temps = 12, n_walkers = 30, n_steps = 2000,
                         n_burnin = 1000, seed = 11, show_progress = false)
    @test 0 < rc.log_evidence_laplace - rc.log_evidence_bridge < 5

    # It must anchor on the MODE, not the mean. A mean-anchored Laplace on a
    # skewed posterior sits in a lower-density place and reads systematically
    # lower, so the mode anchor should not be BELOW a mean-anchored proxy by a
    # wide margin.
    @test lap > br - 200.0

    # Documented NaN paths, so a caller can rely on the sentinel.
    # Guard is N >= 2d+2 samples across ALL walkers, so slicing iterations alone
    # is not enough -- 3 iterations x 30 walkers is 90 samples, comfortably over.
    @test isnan(mode_laplace_evidence(target, res.chains[1:2, :, 1:1]))

    # The same value whichever density convention the chain's `:lp` carries:
    # pt_emcee's (bounded, no Jacobian), pt_hmc's and NUTS's (unconstrained,
    # Jacobian included), or none at all (every draw evaluated).
    nm = Symbol.(target.params.layout.unfrozen_names)
    X = Array(res.chains[:, nm, :])
    lq = similar(X, size(X, 1), 1, size(X, 3))
    for i in axes(X, 1), c in axes(X, 3)
        y = Nereus.transform_forward(X[i, :, c], target.transform)
        lq[i, 1, c] = sum(Nereus._logdensity_parts(target, y))
    end
    @test mode_laplace_evidence(target, Chains(cat(X, lq; dims = 2), vcat(nm, :lp))) ≈ lap rtol = 1e-10
    @test mode_laplace_evidence(target, Chains(X, nm)) ≈ lap rtol = 1e-10

    # And it is reported alongside, not silently swapped in: the headline is
    # bridge whenever bridge is available.
    @test res.log_evidence == res.log_evidence_bridge
    @test res.log_evidence != lap || isapprox(lap, br)
end
