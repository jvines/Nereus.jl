# Reference-path evidence, against a case with a known answer.
#
# Same 22-D target as test_bridge_evidence.jl, and for the same reason: the
# beta-path estimators (TI/TI+/SS+/H+) miss the real 22-D HD 18599 posterior by
# ~118 nats, so an estimator that only works in 2-D would prove nothing.
using Test
using Nereus
using Nereus: _reference_path_core
using LinearAlgebra, Statistics, Random

@testset "reference-path evidence" begin
    d, L = 22, 20.0
    rng = MersenneTwister(7)
    A = randn(rng, d, d); Sig = A * A' / d + 0.5I
    Lc = cholesky(Symmetric(Sig)).L

    # Gaussian likelihood inside a wide uniform box: Z = (2L)^-d exactly.
    truth = -d * log(2L)
    iS = inv(Symmetric(Sig)); ld = logdet(Symmetric(Sig))
    lp(y) = any(abs.(y) .> L) ? -Inf :
            -0.5 * dot(y, iS * y) - 0.5d * log(2π) - 0.5ld - d * log(2L)

    Y = Lc * randn(rng, d, 40_000)          # posterior draws, to fit the reference

    r = _reference_path_core(lp, Y; n_particles = 512, n_beta = 64,
                             n_steps = 8, proposal = :gaussian, seed = 3)

    @test isfinite(r.log_z_ais)
    # 0.05 nats against TI's ~118-nat miss on the real target of this size.
    @test abs(r.log_z_ais - truth) < 0.05
    @test abs(r.log_z_ti  - truth) < 0.05
    # With a reference this well matched the two estimators must agree; a gap
    # here means the particles are not equilibrating.
    @test abs(r.log_z_ais - r.log_z_ti) < 0.05
    @test r.support_frac == 1.0            # q stays inside the target's support
    @test r.ess > 0.5 * r.n_particles      # weights not dominated by a few draws
    @test 0.05 < r.accept < 0.95           # kernel is actually moving

    # Seed-stability: this is a Monte Carlo estimator, so check it is not
    # accidentally right for one stream.
    for seed in (11, 29)
        rs = _reference_path_core(lp, Y; n_particles = 512, n_beta = 64,
                                  n_steps = 8, proposal = :gaussian, seed = seed)
        @test abs(rs.log_z_ais - truth) < 0.05
    end

    # AIS is unbiased for ANY invariant kernel, so it must survive a kernel too
    # short to equilibrate — which is exactly where the TI cross-check does not.
    rq = _reference_path_core(lp, Y; n_particles = 512, n_beta = 64,
                              n_steps = 2, proposal = :gaussian, seed = 3)
    @test abs(rq.log_z_ais - truth) < 0.10

    # Degenerate input returns cleanly rather than throwing.
    rbad = _reference_path_core(lp, Y[:, 1:3]; n_particles = 8, n_beta = 4)
    @test isnan(rbad.log_z)
end
