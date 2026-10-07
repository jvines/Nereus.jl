# Derivatives in e at and near e = 0.
#
# Kepler's solver returned its initial guess M + 0.85 sign(sin M) e whenever
# that already met the tolerance, which it does at e = 0 (and below e ~ 1e-10):
# an AD backend tracing the iteration then got ∂E/∂e = 0.85 sign(sin M), not
# sin M.
#
# The reference is the implicit-function derivatives of Kepler's equation.

using Test
using Nereus
using ForwardDiff
using ReverseDiff

@testset "Kepler's equation: derivatives of the solution" begin
    for e in (0.0, 1e-14, 1e-11, 1e-6, 0.3, 0.9, 0.99)
        for M in range(-3.1, 3.1; length = 63)
            E = kepler_solve(M, e)
            sE, cE = sincos(E)
            dEde, dEdM = sE / (1 - e * cE), 1 / (1 - e * cE)
            # ForwardDiff (the Dual method)
            @test ForwardDiff.derivative(x -> kepler_solve(M, x), e) ≈ dEde rtol = 1e-9 atol = 1e-14
            @test ForwardDiff.derivative(x -> kepler_solve(x, e), M) ≈ dEdM rtol = 1e-9
            # A backend that traces the iteration (the Newton step on convergence)
            g = ReverseDiff.gradient(v -> kepler_solve(v[1], v[2]), [M, e])
            @test g[1] ≈ dEdM rtol = 1e-9
            @test g[2] ≈ dEde rtol = 1e-9 atol = 1e-14
        end
    end
    # The Dual's value is the Float64 solution, to the bit; and an M far from
    # [-π, π) keeps the same derivatives.
    for (M, e) in ((1.3, 0.0), (-2.2, 0.4), (40.0, 0.7), (-1e4, 0.2))
        d = kepler_solve(ForwardDiff.Dual(M, 1.0), ForwardDiff.Dual(e, 0.0))
        @test ForwardDiff.value(d) === kepler_solve(M, e)
        E = kepler_solve(M, e)
        @test ForwardDiff.partials(d)[1] ≈ 1 / (1 - e * cos(E)) rtol = 1e-12
    end
    # Second derivative at e = 0: E = M + e sin M + e² sin M cos M + O(e³).
    for M in (-2.5, -0.4, 0.9, 2.0)
        d2 = ForwardDiff.derivative(e -> ForwardDiff.derivative(x -> kepler_solve(M, x), e), 0.0)
        @test d2 ≈ sin(2M) rtol = 1e-9
    end
    # Mixed tags (a Dual in M only, a Dual in e only)
    @test ForwardDiff.derivative(x -> kepler_solve(x, 0.0), 0.7) ≈ 1.0
    @test ForwardDiff.gradient(v -> kepler_solve(v[1], v[2]), [0.7, 0.0]) ≈ [1.0, sin(0.7)]
end
