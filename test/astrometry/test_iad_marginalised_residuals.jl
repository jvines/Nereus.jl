# The residuals the IAD marginalisation leaves behind.
#
# `iad_residuals.png` came out EMPTY on a real Gaia DR4 fit — no points, axes
# at Makie's default 0–10, χ²/N = NaN — while the sky-plane figure from the
# same posterior was perfect. The plotting code kept its own copy of the
# design row and that copy still had the pre-removal 5-wide catalogue layout
# (Δα₀, Δδ₀, ϖ, μα*, μδ):
#
#     corr = q[p]*s + q[p+1]*c + q[3]*plxf + q[4]*s*pmf + q[5]*c*pmf
#
# against the real design `cols = (p, p+1, 3, 4)`, `xs = (s, c, s·pmf, c·pmf)`.
# So it multiplied μα* by the parallax factor, shifted both proper-motion
# terms one column, and indexed q_opt[5] — which does not exist for a single
# instrument, where `_iad_n_q(1) == 4`. The loop was `@inbounds`, so instead
# of a BoundsError it read past the end of the vector and every residual came
# back NaN.
#
# The projection now lives beside the design it must match. These tests pin
# the property that makes it checkable: q_opt is the least-squares minimiser,
# so the weighted sum of squares of the residuals it leaves MUST equal the
# marginalised χ²_min that `_iad_solve` reports.

using Test, Nereus, LinearAlgebra, Random
using Nereus: IADData

@testset "IAD marginalised residuals" begin
    rng = MersenneTwister(20260922)
    n, t0 = 60, 57039.0
    t   = collect(range(t0, t0 + 1800.0, length = n))
    psi = 2π .* rand(rng, n)
    plxf = sin.(2π .* (t .- t0) ./ 365.25 .- psi)
    pmf  = (t .- t0) ./ 365.25
    plx  = 13.6
    err  = fill(0.3, n)
    absc = plx .* plxf .+ 0.7 .* sin.(psi) .+ 0.4 .* cos.(psi) .* pmf .+
           err .* randn(rng, n)
    iad = IADData(t = t, abscissa = absc, abscissa_err = err,
                  psi = psi, parallax_factor = plxf, pm_factor = pmf)

    n_inst = Nereus.n_iad_inst(iad)
    n_q    = Nereus._iad_n_q(n_inst)

    @testset "a single instrument has four columns, not five" begin
        # The out-of-bounds index that made the figure empty.
        @test n_inst == 1
        @test n_q == 4
        @test Nereus._iad_pos_cols(1) == [1]
    end

    r = Vector{Float64}(undef, n)
    Nereus._iad_residuals!(r, iad, Any[], Float64[], plx)
    @test all(isfinite, r)

    pos_col = Nereus._iad_pos_cols(n_inst)
    A = zeros(n_q, n_q); v = zeros(n_q)
    rWr, _ = Nereus._iad_normal_equations!(A, v, iad, r, iad.pm_factor, pos_col)
    chol = cholesky(Symmetric(A); check = false)
    @test issuccess(chol)
    q_opt = chol \ v

    resid = Nereus._iad_marginalised_residuals!(
        Vector{Float64}(undef, n), iad, r, q_opt, iad.pm_factor, pos_col)

    @testset "residuals are finite" begin
        # The whole failure mode: every one of these was NaN.
        @test all(isfinite, resid)
        @test !all(iszero, resid)
    end

    @testset "they reproduce the marginalised chi-squared" begin
        # q_opt minimises the weighted sum of squares, so what it leaves
        # behind must equal rᵀWr − vᵀA⁻¹v exactly. A projection using the
        # wrong columns cannot satisfy this.
        solved = Nereus._iad_solve(A, v, rWr, n_q)
        @test solved !== nothing
        chi2_min, _ = solved
        @test sum(resid .^ 2 ./ iad.abscissa_err .^ 2) ≈ chi2_min rtol=1e-8
    end

    @testset "chi2/N is finite and sane" begin
        # The number the figure prints in its corner.
        @test isfinite(sum(resid .^ 2 ./ iad.abscissa_err .^ 2) / n)
    end
end
