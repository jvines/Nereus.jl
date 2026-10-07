# The PPC residual autocorrelation on sparse, gapped sampling.
#
# Empty bins of the uniform grid were filled by linear interpolation, so on a
# typical RV campaign (short runs of nightly points, long gaps) the series was
# mostly straight lines and lag 1 came out near 1 for white residuals: 0.81 on
# the 36 NGTS-33 RVs, whose time-ordered lag-1 correlation is 0.02. The
# estimator now averages over pairs of occupied bins only.

using Nereus, Test, Random, Statistics

@testset "residual ACF uses occupied bins only" begin
    rng = MersenneTwister(7)
    # 20 seasons of 8 consecutive nights, one point a night, seasons 30 d apart
    t = sort(vec([30.0 * s + n + 0.05 * randn(rng) for n in 0:7, s in 0:19]))
    lags, ac = Nereus._residual_acf(t, randn(rng, length(t)))
    @test ac[1] ≈ 1.0
    @test lags[2] ≈ 1.0 atol = 0.1                 # median cadence: one night
    @test abs(ac[2]) < 0.25                         # white residuals: no lag-1 memory
    @test isnan(ac[findfirst(>(12.0), lags)])       # 12-25 d: no pair of points there

    # A real AR(1) process with gaps keeps its lag-1 correlation
    n = 4000; φ = 0.9
    x = zeros(n); x[1] = randn(rng)
    for i in 2:n; x[i] = φ * x[i-1] + sqrt(1 - φ^2) * randn(rng); end
    keep = [i <= 1500 || i > 2500 for i in 1:n]     # a 1000-sample gap
    lags2, ac2 = Nereus._residual_acf(collect(1.0:n)[keep], x[keep])
    @test ac2[2] ≈ φ atol = 0.05
end
