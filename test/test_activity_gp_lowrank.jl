# ActivityGP joint-likelihood solver: the quasi-periodic kernel blocks and
# the covariance builders that feed it.

using Nereus, LinearAlgebra, Random, Test
using Random: MersenneTwister

# Transcription of the closed forms in the header of src/noise/activity_gp.jl:
#   f   = -τ²/(2λe²) - sin²(πτ/P)/(2λp²)
#   f'  = -τ/λe² - π/(2Pλp²)·sin(2πτ/P)
#   f'' = -1/λe² - π²/(P²λp²)·cos(2πτ/P)
# and the blocks (k, -f'k, f'k, -(f''+f'²)k), written with the same operations
# as `activity_kernel_blocks` so the two agree bit for bit.
function _qp_blocks_transcribed(τ, amp, P, λe, λp)
    s   = sin(π * τ / P)
    s2  = sin(2π * τ / P)
    c2  = cos(2π * τ / P)
    f   = -τ^2 / (2 * λe^2) - s^2 / (2 * λp^2)
    fp  = -τ / λe^2 - π / (2 * P * λp^2) * s2
    fpp = -1.0 / λe^2 - π^2 / (P^2 * λp^2) * c2
    k   = amp^2 * exp(f)
    return (k, -fp * k, fp * k, -(fpp + fp^2) * k)
end

@testset "AGP kernel blocks and covariance builders keep their values" begin
    rng = MersenneTwister(20261003)
    for _ in 1:200
        τ   = 200 * (rand(rng) - 0.5)
        amp = 0.2 + 3rand(rng); P = 2 + 20rand(rng)
        λe  = 5 + 200rand(rng); λp = 0.2 + 2rand(rng)
        @test Nereus.activity_kernel_blocks(τ, amp, P, λe, λp) ===
              _qp_blocks_transcribed(τ, amp, P, λe, λp)
    end
    # τ = 0: no derivative cross-covariance, Var(Ġ) = amp²·(1/λe² + π²/(P²λp²)).
    kGG, kGd, kdG, kdd = Nereus.activity_kernel_blocks(0.0, 1.3, 8.7, 40.0, 0.6)
    @test kGG == 1.3^2
    @test kGd == 0 && kdG == 0
    @test kdd ≈ 1.3^2 * (1 / 40.0^2 + π^2 / (8.7^2 * 0.6^2)) rtol = 1e-14

    # The flat builder and the block-factored builder evaluate the same
    # expressions per pair. On a channel-major layout their upper triangles
    # agree exactly; below the diagonal the flat builder mirrors the upper
    # entry while the blocked one sums the two derivative terms in the other
    # order, so those entries can differ by an ulp.
    N, C = 23, 4
    t = sort!(60 .* rand(rng, N))
    ca = randn(rng, C); cb = randn(rng, C)
    amp, P, λe, λp = 1.1, 9.3, 35.0, 0.7
    t_flat = repeat(t, C)
    ch = repeat([:rv, :bis, :fwhm, :halpha], inner = N)
    a_flat = repeat(ca, inner = N); b_flat = repeat(cb, inner = N)
    Σd = activity_gp_covariance(t_flat, ch, a_flat, b_flat, amp, P, λe, λp)
    Σb = Nereus.activity_gp_covariance_blocked(t, ca, cb, amp, P, λe, λp)
    @test UpperTriangular(Σd) == UpperTriangular(Σb)
    @test Σd ≈ Σb rtol = 1e-14
    @test issymmetric(Σd)
end
