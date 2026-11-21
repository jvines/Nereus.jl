# Transit-panel y-limits and raw-cadence alpha (`plot_pm_phasefold`,
# `plot_pm_timeseries`). Source: an Argus fit of TIC-275647095 (2026-11-21).
#   * LCO Sutherland covers egress only: limits from the raw cadences'
#     2-98% quantiles cut the model's transit floor off the panel.
#   * NGTS 10 s cadences (±2.5% scatter, 1% transit): the same quantiles made
#     the axis ~5% tall, and point-count alpha (never below 0.75) plus
#     full-height error bars drew the transit as a solid block.

using Nereus, Test, Random, Statistics
using CairoMakie
const N = Nereus

@testset "fold flux limits frame the model and the bins" begin
    # Egress-only data: raw near the baseline, the model's floor at 0.99.
    model = vcat(fill(1.0, 50), fill(0.99, 100), fill(1.0, 50))
    raw = 1.0 .+ 1e-3 .* randn(MersenneTwister(1), 80)
    yb = [0.9995, 1.0002]; eb = [3e-4, 3e-4]
    lo, hi = N._fold_flux_ylims(model, yb, eb, raw)
    @test lo < 0.99 && hi > 1.0002 + 3e-4
    # Raw scatter does not set the limits when bins exist.
    noisy = 1.0 .+ 0.025 .* randn(MersenneTwister(2), 8000)
    lo2, hi2 = N._fold_flux_ylims(model, yb, eb, noisy)
    @test (lo2, hi2) == (lo, hi)
    @test hi2 - lo2 < 0.02
    # No bins: raw quantiles, still widened to the model's floor.
    lo3, hi3 = N._fold_flux_ylims(model, Float64[], Float64[], raw)
    @test lo3 < 0.99 && hi3 >= quantile(raw, 0.98)
    @test N._fold_flux_ylims(Float64[], Float64[], Float64[], Float64[]) === nothing
end

@testset "fold residual limits follow the binned residuals" begin
    raw = 25_000 .* randn(MersenneTwister(3), 8000)
    rb = [500.0, -400.0, 300.0, -600.0]; erb = fill(200.0, 4)
    lo, hi = N._fold_resid_ylims(rb, erb, raw)
    @test lo == -hi
    @test hi >= maximum(abs.(rb) .+ erb)
    @test hi < 5_000                         # not the raw 2-98% (~50 000 ppm)
    lo2, hi2 = N._fold_resid_ylims(Float64[], Float64[], raw)
    @test hi2 > 40_000                       # no bins: raw quantiles
end

function _alpha_on(x, y, err, ylims; ms = 4)
    fig = Figure(size = (1000, 1000)); ax = Axis(fig[1, 1])
    N._overlap_alpha(ax, x, y, err, extrema(x), ylims; markersize = ms)
end

@testset "alpha follows on-screen overlap, not point count" begin
    rng = MersenneTwister(4)
    # Sparse: 86 points with small bars -> opaque, bars drawn.
    x = collect(range(0, 1; length = 86)); y = 1e-3 .* randn(rng, 86)
    a, bars = _alpha_on(x, y, fill(1e-4, 86), (-5e-3, 5e-3))
    @test a == 1.0 && bars
    # Many points, still spread out on screen: count alone must not lower alpha.
    x = rand(rng, 3000); y = rand(rng, 3000)
    a, _ = _alpha_on(x, y, fill(1e-4, 3000), (0.0, 1.0))
    @test a == 1.0
    # A dense cloud piled into a narrow band: alpha drops toward the floor.
    x = rand(rng, 50_000); y = 0.5 .+ 0.01 .* randn(rng, 50_000)
    a, _ = _alpha_on(x, y, fill(1e-4, 50_000), (0.0, 1.0))
    @test 0.03 <= a < 0.2
    # Error bars spanning the panel: left out, and they do not set the alpha.
    x = rand(rng, 3000); y = rand(rng, 3000)
    a, bars = _alpha_on(x, y, fill(2.0, 3000), (0.0, 1.0))
    @test !bars && a == 1.0
end
