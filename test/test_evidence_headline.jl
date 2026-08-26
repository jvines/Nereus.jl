# What sample_ptemcee reports as `log_evidence`, and the estimators behind it.
#
# Measured on 51 Peg (1691 RVs, 15 free params, unimodal, P to 5 decimals --
# the friendliest target any of these will see):
#
#   TI+ / H+  -6073.24   SS+ -6097.41    ~178 nats low
#   Laplace   -5920.97                    ~24 nats low
#   bridge    -5897.0                     validated pair
#   refpath   -5893.1
#
# So the headline is bridge, Laplace is reported but not headlined, and the
# tempered stack is reported but never silently substituted.
#
# mode_laplace_evidence had NO test coverage at all before this file, despite
# being the value sample_ptemcee substituted in whenever the tempered path
# failed -- i.e. the number it reported on exactly the hard cases came from the
# one estimator nothing checked.
using Test
using Nereus
using LinearAlgebra, Statistics, Random, Distributions, MCMCChains

@testset "evidence headline and Laplace coverage" begin
    # A small, well-determined RV fit: unimodal, so every estimator here is
    # inside its stated preconditions.
    rng = MersenneTwister(20260826)
    n = 120
    t = sort!(1200 .* rand(rng, n))
    P0, K0 = 4.2308, 55.0
    rv = K0 .* sin.(2π .* t ./ P0) .+ 2.0 .* randn(rng, n)
    er = fill(2.0, n)
    target = build_target(
        planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 120.0),
                        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                        Mo = UniformPrior(0.0, 2π)),),
        rv = (SIM = (data = (t = t, rv = rv, rv_err = er),
                     sigma = LogUniformPrior(0.5, 20.0)),),
    )
    res = sample_ptemcee(target, target.data; n_temps = 12, n_walkers = 30,
                         n_steps = 1500, n_burnin = 800, seed = 7,
                         show_progress = false)

    # every estimator is still reported
    @test isfinite(res.log_evidence)
    @test res.evidence isa Nereus.EvidenceReport
    @test isfinite(res.evidence.ti_plus[1])

    # mode-Laplace: its first test. It must at least be finite and in the right
    # ballpark -- this is coverage where there was none, not a precision claim.
    @test isfinite(res.log_evidence_laplace)
    @test abs(res.log_evidence_laplace - res.log_evidence) < 200

    # bridge is computed, and IS the headline
    @test isfinite(res.log_evidence_bridge)
    @test res.log_evidence == res.log_evidence_bridge

    # ...and it is not merely echoing Laplace or the tempered value
    @test res.log_evidence_bridge != res.log_evidence_laplace
    @test res.log_evidence_bridge != res.evidence.ti_plus[1]

    # opting out returns the pre-existing behaviour rather than erroring
    res2 = sample_ptemcee(target, target.data; n_temps = 12, n_walkers = 30,
                          n_steps = 1500, n_burnin = 800, seed = 7,
                          bridge_headline = false, show_progress = false)
    @test isnan(res2.log_evidence_bridge)
    @test isfinite(res2.log_evidence)
end
