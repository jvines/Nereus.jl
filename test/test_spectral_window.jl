# The `:unit_signal` window is the spectral window of the sampling. GLS of a
# constant is not: GLS removes the mean and leaves ~1e-11 power everywhere.

using Nereus, Random, Test
using Random: MersenneTwister

# four visits every 10 d: the window peaks at 10 d and its harmonics
function _ten_day_sampling(seed)
    rng = MersenneTwister(seed)
    sort!(vcat([10.0 * k .+ 0.3 .* rand(rng, 4) for k in 0:30]...))
end

@testset "find_rv_planets: the :unit_signal window is the spectral window" begin
    t = _ten_day_sampling(3)
    n = length(t)
    rng = MersenneTwister(4)
    set = find_rv_planets(t, randn(rng, n), ones(n); window_def = :unit_signal,
                          fap_method = :analytic, samples_per_peak = 10)
    w = set.window
    @test w.frequencies == set.rv.frequencies             # one grid for every panel
    @test maximum(w.power) > 0.9
    @test abs(w.peaks[1].period - 10.0) < 0.1
    @test all(isnan, w.fap_thresholds) && all(p -> isnan(p.fap), w.peaks)
    # W(f) = |Σ exp(2πi f t)|² / N² at the top peak
    f = w.peaks[1].frequency
    @test w.peaks[1].power ≈ abs2(sum(cis.(2π .* f .* t))) / n^2
    # the default window is still GLS(t, t)
    d = find_rv_planets(t, randn(rng, n), ones(n); fap_method = :analytic,
                        samples_per_peak = 10)
    @test !all(isnan, d.window.fap_thresholds)
end

@testset "gls_rotation: the :unit_signal window is the spectral window" begin
    t = _ten_day_sampling(5)
    n = length(t)
    rng = MersenneTwister(6)
    flux = 1 .+ 1e-3 .* randn(rng, n)
    set = gls_rotation(t, flux, fill(1e-3, n); window_def = :unit_signal, gap_size = 5.0)
    @test set.window.frequencies == set.stitched.frequencies
    @test abs(set.window.peaks[1].period - 10.0) < 0.1
    @test maximum(set.window.power) > 0.9
end

@testset "find_rv_planets: the window takes the caller's FAP settings" begin
    t = _ten_day_sampling(7)
    n = length(t)
    set = find_rv_planets(t, randn(MersenneTwister(8), n), ones(n);
                          fap_method = :analytic, samples_per_peak = 10)
    ref = gls_periodogram(t, copy(t), ones(n); fap_method = :analytic,
                          samples_per_peak = 10)
    @test set.window.fap_thresholds == ref.fap_thresholds
end
