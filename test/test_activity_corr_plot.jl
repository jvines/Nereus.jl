# RV and RV residuals against the activity indicators (`plot_rv_activity_corr`,
# plot name "rv_activity_corr"), and the shape of the transit phase fold.
#
# The correlation figure is written whenever an RV instrument carries
# indicators, skipped when none do, and a missing (NaN) indicator value drops
# that point from its panel only. Each panel's weighted line is the exact
# Gaussian posterior of (a, b): the WLS mean and (AᵀWA)⁻¹.
#
# The transit fold is 3:2 and frames the 3σ band it draws (framed on the median
# alone, the band's transit floor was cut off at the bottom edge) and leaves a
# strip at the top for the instrument label.

using Nereus, Test, Random, MCMCChains, Statistics, Printf
using CairoMakie: Axis, Colorbar
using CairoMakie.Makie: Text

function _corr_fixture(; with_indicators = true, nan_bis = false)
    rng = MersenneTwister(5)
    n = 40
    t = sort(2460000.0 .+ 300 .* rand(rng, n))
    inst = [isodd(i) ? 1 : 2 for i in 1:n]
    ind = nothing
    if with_indicators
        bis = randn(rng, n); fwhm = randn(rng, n)
        nan_bis && (bis[7] = NaN; bis[8] = NaN)
        ind = Dict("bis" => bis, "fwhm" => fwhm)
    end
    data = Data(; t_rv = t, rv = zeros(n), rv_err = fill(3.0, n), rv_inst = inst,
                  indicators = ind)
    params = Params(; parametrization = ParametrizationConfig(time = :Mo), max_kplanet = 1,
                    planet_modes = [RV_ONLY], instruments = InstrumentConfig(rv = ["FEROS", "HARPS"]),
                    data = data, M_s = 1.0)
    vals = Dict("P_k1" => 11.3, "K_k1" => 25.0, "sesinw_k1" => 0.05, "secosw_k1" => -0.02,
                "Mo_k1" => 0.7, "gamma_FEROS" => 0.0, "gamma_HARPS" => 0.0,
                "sigma_FEROS" => 3.0, "sigma_HARPS" => 3.0)
    theta = Theta{Float64}(params)
    for (k, v) in vals
        set_param!(theta, k, v)
    end
    model = rv_predictions(theta, data)[1]
    data.rv .= model .+ 3.0 .* randn(MersenneTwister(9), n)
    nm = params.layout.unfrozen_names
    cols = vcat(Symbol.(nm), [:lp])
    arr = zeros(30, length(cols), 1)
    for i in 1:30
        arr[i, 1:length(nm), 1] = [vals[x] * (1 + 1e-4 * randn(rng)) for x in nm]
        arr[i, end, 1] = -0.1 * i
    end
    return params, data, Chains(arr, cols)
end

_texts(fig) = String[string(p.text[]) for ax in filter(c -> c isa Axis, fig.content)
                     for p in ax.scene.plots if p isa Text]

@testset "weighted line posterior is exactly Gaussian" begin
    rng = MersenneTwister(1)
    x = randn(rng, 60); σ = 0.5 .+ rand(rng, 60)
    y = 2.0 .+ 0.7 .* x .+ σ .* randn(rng, 60)
    a, b, cov, xm = Nereus._wls_line(x, y, σ)
    w = 1 ./ σ .^ 2
    A = hcat(ones(60), x)
    β = (A' * (w .* A)) \ (A' * (w .* y))
    @test b ≈ β[2] rtol = 1e-10
    @test a - b * xm ≈ β[1] rtol = 1e-10            # a is the line at xm
    @test cov[2, 2] ≈ inv(A' * (w .* A))[2, 2] rtol = 1e-10
    xs = collect(range(-2, 2; length = 50))
    q, _, bq, sd, _ = Nereus._line_bands(x, y, σ, xs; n = 6000)
    @test issorted(q[:, 25])                          # nested bands
    @test all(q[1, :] .< q[2, :] .< q[3, :] .< q[4, :] .< q[5, :] .< q[6, :] .< q[7, :])
    @test q[4, :] ≈ a .+ b .* (xs .- xm) atol = 0.03  # median line = WLS line
end

@testset "activity-correlation figure" begin
    @testset "produced per instrument when indicators exist" begin
        params, data, chains = _corr_fixture()
        out = mktempdir()
        figs = plot_rv_activity_corr(chains, params, data; output = out)
        @test length(figs) == 2
        for ins in ("FEROS", "HARPS")
            @test isfile(joinpath(out, "models", "RV_activity_corr_$ins.png"))
        end
        fig = first(figs)
        axes = filter(c -> c isa Axis, fig.content)
        @test length(axes) == 4                       # 2 indicators x (RV, residuals)
        @test length(filter(c -> c isa Colorbar, fig.content)) == 1
        # Pearson r of the first instrument, RV vs BIS, printed top-left
        m = data.rv_inst .== 1
        r = cor(data.indicators["bis"][m], data.rv[m] .- 0.0)
        @test any(s -> occursin(replace("r = " * Printf.@sprintf("%.2f", r), "-" => "−"), s), _texts(fig))
    end

    @testset "skipped when there are no indicators" begin
        params, data, chains = _corr_fixture(with_indicators = false)
        out = mktempdir()
        @test isempty(plot_rv_activity_corr(chains, params, data; output = out))
        @test !isdir(joinpath(out, "models"))
        @test !("rv_activity_corr" in Nereus._auto_plot_kinds(chains, params, data))
        @test Nereus._dispatch_plot("rv_activity_corr", chains, params, data, out, Dict{Symbol, Any}()) === nothing
    end

    @testset "a NaN indicator value drops that point only" begin
        params, data, chains = _corr_fixture(nan_bis = true)
        @test "rv_activity_corr" in Nereus._auto_plot_kinds(chains, params, data)
        out = mktempdir()
        figs = plot_rv_activity_corr(chains, params, data; output = out)
        @test length(figs) == 2
        # both indicators still have a column; the NaN points are not drawn in the BIS panels
        fig = first(figs)
        @test length(filter(c -> c isa Axis, fig.content)) == 4
        m = (data.rv_inst .== 1) .& isfinite.(data.indicators["bis"])
        r = cor(data.indicators["bis"][m], data.rv[m])
        @test any(s -> occursin(replace("r = " * Printf.@sprintf("%.2f", r), "-" => "−"), s), _texts(fig))
        @test !any(s -> occursin("NaN", s), _texts(fig))
        @test Nereus._dispatch_plot("rv_activity_corr", chains, params, data, out, Dict{Symbol, Any}()) ==
              "models/RV_activity_corr_*.png"
    end
end

@testset "transit fold limits frame the 3σ band and leave room for the label" begin
    x = collect(range(-0.025, 0.025; length = 400))
    median = 1 .- 0.01 .* exp.(-(x ./ 0.008) .^ 2)
    lo3 = median .- 0.0015; hi3 = median .+ 0.0015
    yb = [1.0, 0.995]; eb = [3e-4, 3e-4]
    plain = Nereus._fold_flux_ylims(median, yb, eb, Float64[])
    banded = Nereus._fold_flux_ylims(vcat(lo3, hi3, median), yb, eb, Float64[]; top_extra = 0.12)
    @test banded[1] < minimum(lo3)                    # the band's floor is inside the frame
    @test banded[2] > maximum(hi3)
    @test banded[1] < plain[1]
    span = (maximum(hi3) - minimum(lo3))
    @test banded[2] - maximum(hi3) > 0.2 * span       # a strip free above the data
    @test Nereus.FIG_PM_FOLD[1] / Nereus.FIG_PM_FOLD[2] ≈ 1.5
end

@testset "transit fold limits keep a sparse fold's raw points in view" begin
    model = vcat(fill(1.0, 50), fill(0.99, 100), fill(1.0, 50))
    yb = [1.0, 0.99]; eb = [2e-3, 2e-3]
    raw = vcat(0.9875, 1.0, 0.9975, 1.003, 0.99)           # one point just below the model floor
    lo, hi = Nereus._fold_flux_ylims(model, yb, eb, raw)
    @test lo < 0.9875 && hi > 1.003
    wild = vcat(raw, 0.9, 1.1)                              # the 1-99% range blows past 2x the signal
    @test Nereus._fold_flux_ylims(model, yb, eb, wild) == Nereus._fold_flux_ylims(model, yb, eb, Float64[])
end

@testset "RV fold frames the credible band it draws, unless the band spikes" begin
    vals = [-100.0, 50.0, 120.0]; errs = fill(5.0, 3); signal = [-120.0, 130.0]
    ci_fit = (lo3 = [-160.0, -150.0], hi3 = [140.0, 150.0], lo2 = [-140.0], hi2 = [135.0],
              lo1 = [-125.0], hi1 = [131.0])
    @test minimum(Nereus._band_extra(vals, errs, signal, ci_fit)) == -160.0     # 3σ fits
    ci_spike = (lo3 = [-5000.0], hi3 = [5000.0], lo2 = [-170.0], hi2 = [160.0],
                lo1 = [-125.0], hi1 = [131.0])
    @test extrema(Nereus._band_extra(vals, errs, signal, ci_spike)) == (-170.0, 160.0)  # 2σ instead
    ci_wild = (lo3 = [-5000.0], hi3 = [5000.0], lo2 = [-4000.0], hi2 = [4000.0],
               lo1 = [-3000.0], hi1 = [3000.0])
    @test isempty(Nereus._band_extra(vals, errs, signal, ci_wild))
    @test Nereus.FIG_RV_FOLD[1] / Nereus.FIG_RV_FOLD[2] ≈ 1.5
end
