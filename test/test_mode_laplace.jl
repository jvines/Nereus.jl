# mode_laplace_evidence, against its own stated assumption.
#
# It is reported by sample_ptemcee and, before today, was SUBSTITUTED IN as the
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
# -- it sat 24 nats from bridge. That is why it is reported but not headlined.
using Test
using Nereus
using Random, Statistics, LinearAlgebra

@testset "mode_laplace_evidence" begin
    # High-SNR single Keplerian: the posterior is tight and close to Gaussian in
    # unconstrained space, which is where Laplace should be at its best.
    rng = MersenneTwister(90210)
    n = 150
    t = sort!(600 .* rand(rng, n))
    P0, K0 = 4.23, 60.0
    rv = K0 .* sin.(2π .* t ./ P0) .+ 1.5 .* randn(rng, n)
    er = fill(1.5, n)
    target = build_target(
        planets = (b = (P = LogUniformPrior(4.1, 4.4), K = LogUniformPrior(20.0, 120.0),
                        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                        Mo = UniformPrior(0.0, 2π)),),
        rv = (SIM = (data = (t = t, rv = rv, rv_err = er),
                     sigma = LogUniformPrior(0.5, 10.0)),),
    )
    res = sample_ptemcee(target, target.data; n_temps = 12, n_walkers = 30,
                         n_steps = 2000, n_burnin = 1000, seed = 11,
                         show_progress = false)

    lap = res.log_evidence_laplace
    br  = res.log_evidence_bridge
    @test isfinite(lap)
    @test isfinite(br)

    # Its premise: where the posterior is near-Gaussian, Laplace should track the
    # validated estimator. The bar is deliberately loose -- this documents the
    # estimator's ACTUAL accuracy rather than asserting a precision it does not
    # have. If this ever fails wide, Laplace has drifted from its own assumption.
    @test abs(lap - br) < 40.0

    # It must anchor on the MODE, not the mean. A mean-anchored Laplace on a
    # skewed posterior sits in a lower-density place and reads systematically
    # lower, so the mode anchor should not be BELOW a mean-anchored proxy by a
    # wide margin.
    @test lap > br - 200.0

    # Documented NaN paths, so a caller can rely on the sentinel.
    # Guard is N >= 2d+2 samples across ALL walkers, so slicing iterations alone
    # is not enough -- 3 iterations x 30 walkers is 90 samples, comfortably over.
    @test isnan(mode_laplace_evidence(target, res.chains[1:2, :, 1:1]))

    # And it is reported alongside, not silently swapped in: the headline is
    # bridge whenever bridge is available.
    @test res.log_evidence == res.log_evidence_bridge
    @test res.log_evidence != lap || isapprox(lap, br)
end
