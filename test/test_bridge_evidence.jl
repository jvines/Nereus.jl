# Bridge-sampling evidence, against a case with a known answer.
#
# The dimensionality is deliberate: 22, the same as the HD 18599 joint
# RV+transit fit where the beta-path estimators (TI/TI+/SS+/H+) miss by 118
# nats. An estimator that only works in 2-D would prove nothing here.
using Test
using Nereus
using Nereus: _bridge_iterate
using LinearAlgebra, Statistics, Random

@testset "bridge sampling evidence" begin
    d, L = 22, 20.0
    rng = MersenneTwister(7)
    A = randn(rng, d, d); Sig = A * A' / d + 0.5I
    Lc = cholesky(Symmetric(Sig)).L
    mu = zeros(d)

    # Gaussian likelihood inside a wide uniform box: all the mass is interior,
    # so Z = (2L)^-d exactly.
    truth = -d * log(2L)
    iS = inv(Symmetric(Sig)); ld = logdet(Symmetric(Sig))
    lp(y) = any(abs.(y) .> L) ? -Inf :
            -0.5 * dot(y - mu, iS * (y - mu)) - 0.5d * log(2π) - 0.5ld - d * log(2L)

    N = 40_000
    Y = mu .+ Lc * randn(rng, d, N)
    mu_h = vec(mean(Y; dims = 2)); S_h = cov(Y; dims = 2) + 1e-10I
    Lh = cholesky(Symmetric(S_h)).L
    logq(y) = (z = Lh \ (y - mu_h); -0.5 * sum(abs2, z) - 0.5d * log(2π) - logdet(Lh))

    l1 = [lp(Y[:, i]) - logq(Y[:, i]) for i in 1:N]
    l2 = Float64[]
    for _ in 1:40_000
        y = mu_h + Lh * randn(rng, d); v = lp(y)
        isfinite(v) && push!(l2, v - logq(y))
    end

    logz, iters, converged = _bridge_iterate(l1, l2, 1000, 1e-12)

    @test converged
    @test iters < 50                       # the fixed point is fast; 4-7 in practice
    @test isfinite(logz)
    # 0.05 nats is ~4 orders of magnitude tighter than TI manages on the real
    # 22-D target, and the observed error here is ~0.003.
    @test abs(logz - truth) < 0.05

    # Symmetry: swapping which sample set plays which role must not move it.
    logz2, _, ok2 = _bridge_iterate(l1[1:20_000], l2[1:20_000], 1000, 1e-12)
    @test ok2
    @test abs(logz2 - truth) < 0.10

    # Degenerate inputs return cleanly rather than throwing.
    r_empty, _, _ = _bridge_iterate(Float64[], Float64[], 10, 1e-8)
    @test isfinite(r_empty) || isnan(r_empty) || isinf(r_empty)
end
