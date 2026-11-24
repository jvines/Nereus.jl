# plot_rv_timeseries on a sparse, multi-instrument fit.
#
# On a 36-RV, 3-instrument fit (one Keplerian, per-instrument white jitter,
# a CeleriteRotation GP whose amplitude sampled down to ~0, no
# ActivityDecorrelation or other mean-modifier active) the figure had three
# defects:
#
#   1. The data and residual panels used a 2-98% quantile zoom, which on a
#      sparse series clips a real point outside the axes (a CORALIE point
#      above the top, a FEROS point below the bottom).
#   2. Every instrument was labelled "<name> (decorr.)" regardless of
#      whether any decorrelation model actually moved the plotted points --
#      a near-zero GP amplitude still returns a technically-nonzero (but
#      physically negligible) correction.
#   3. A spurious "raw RV" layer (and legend entry) was drawn under/over the
#      data for the same reason, duplicating it.
#
# This test builds exactly that scenario (no ActivityDecorrelation, a
# CeleriteRotation GP with gp_sigma essentially 0) and checks: every point,
# with its error bar, is framed in both panels -- the one true outlier
# included -- and the legend carries plain instrument names with no
# "(decorr.)" suffix and no "raw RV" entry. `robust_ylim = true` keeps the
# zoom, which frames every point but the 5-robust-sigma outlier.
#
# The SB2 time series and binary fold frame the same way, and `robust_ylim`
# reaches every RV figure from `output.plot_kwargs`.

using Nereus, Test, Random, MCMCChains
using CairoMakie: Axis, Legend

# y of every marker a panel draws
_marker_ys(ax) = reduce(vcat, [Float64[pt[2] for pt in p[1][]] for p in ax.scene.plots
                               if p isa Nereus.CairoMakie.Makie.Scatter]; init = Float64[])

# Every marker of both panels, with its error bar (`err`, one value for all
# points), inside the frame.
function _all_framed(fig, err)
    ax, ax_r = filter(c -> c isa Axis, fig.content)
    lo, hi = ax.limits[][2]
    rlo, rhi = ax_r.limits[][2]
    ys, rs = _marker_ys(ax), _marker_ys(ax_r)
    return !isempty(ys) && !isempty(rs) &&
           all(lo .< ys .- err) && all(ys .+ err .< hi) &&
           all(rlo .< rs .- err) && all(rs .+ err .< rhi)
end

# A tight, non-degenerate posterior around `vals`, best draw first.
function _tight_chains(params, vals, rng)
    nm = params.layout.unfrozen_names
    cols = vcat(Symbol.(nm), [:lp])
    ndraws = 40
    arr = zeros(ndraws, length(cols), 1)
    for i in 1:ndraws
        arr[i, 1:length(nm), 1] = [vals[x] * (1 + 1e-4 * randn(rng)) for x in nm]
        arr[i, end, 1] = -0.1 * i
    end
    return Chains(arr, cols)
end

function _sparse_multi_inst_rv_fixture()
    rng = MersenneTwister(31)
    n = 36
    t = sort(561.0 .* rand(rng, n))
    inst_names = ["FEROS", "CORALIE", "HARPS"]
    inst = [((i - 1) % 3) + 1 for i in 1:n]        # round-robin across 3 instruments
    data = Data(; t_rv = t, rv = zeros(n), rv_err = fill(3.0, n), rv_inst = inst)
    params = Params(; parametrization = ParametrizationConfig(time = :Mo), max_kplanet = 1, planet_modes = [RV_ONLY],
                       instruments = InstrumentConfig(rv = inst_names),
                       data = data, M_s = 1.0,
                       noise_models = NoiseModel[CeleriteRotation()])

    vals = Dict(
        "P_k1" => 11.3, "K_k1" => 25.0, "sesinw_k1" => 0.05, "secosw_k1" => -0.02,
        "Mo_k1" => 0.7,
        "gamma_FEROS" => 0.0, "gamma_CORALIE" => 0.0, "gamma_HARPS" => 0.0,
        "sigma_FEROS" => 30.0, "sigma_CORALIE" => 30.0, "sigma_HARPS" => 30.0,
        # GP "active" but amplitude sampled to the floor: a near-zero,
        # never-exactly-zero contribution at every data point.
        "gp_sigma" => 1e-9, "gp_period" => 18.0, "gp_Q0" => 1.3, "gp_dQ" => 1.8,
        "gp_f" => 0.35,
    )
    theta = Theta{Float64}(params)
    for (k, v) in vals
        set_param!(theta, k, v)
    end
    model = rv_predictions(theta, data)[1]
    noise = 30.0 .* randn(MersenneTwister(1384), n)
    outlier_idx = 5
    noise[outlier_idx] = 200.0                      # the one true outlier
    data.rv .= model .+ noise
    return params, data, _tight_chains(params, vals, rng), noise, outlier_idx, inst_names
end

# A double-lined binary, primary and secondary alternating, with one
# secondary point 5 km/s off its curve.
function _sb2_fixture()
    rng = MersenneTwister(7)
    n = 30
    t = sort(400.0 .* rand(rng, n))
    comp = [isodd(i) ? 1 : 2 for i in 1:n]
    data = Data(; t_rv = t, rv = zeros(n), rv_err = fill(50.0, n),
                  rv_inst = ones(Int, n), rv_comp = comp)
    params = Params(; parametrization = ParametrizationConfig(time = :Mo), max_kplanet = 1, planet_modes = [Nereus.BINARY_RV],
                       instruments = InstrumentConfig(rv = ["I1"]),
                       data = data, M_s = 1.0)
    value(nm) = startswith(nm, "P_")      ? 37.0     :
                startswith(nm, "K_A_")    ? 20_000.0 :
                startswith(nm, "K_B_")    ? 30_000.0 :
                startswith(nm, "sesinw_") ? 0.2      :
                startswith(nm, "secosw_") ? 0.1      :
                startswith(nm, "Mo_")     ? 1.0      :
                startswith(nm, "gamma_")  ? 0.0      :
                startswith(nm, "sigma_")  ? 100.0    :
                error("SB2 fixture: no value for $nm")
    vals = Dict(nm => value(nm) for nm in params.layout.unfrozen_names)
    theta = Theta{Float64}(params)
    for (k, v) in vals
        set_param!(theta, k, v)
    end
    model = rv_predictions(theta, data)[1]
    noise = 100.0 .* randn(rng, n)
    outlier_idx = 8                                  # a secondary point
    noise[outlier_idx] = 5000.0
    data.rv .= model .+ noise
    return params, data, _tight_chains(params, vals, rng), outlier_idx
end

@testset "sparse multi-instrument RV timeseries" begin
    params, data, chains, noise, outlier_idx, inst_names = _sparse_multi_inst_rv_fixture()
    fig = plot_rv_timeseries(chains, params, data)

    @testset "every point is framed with its error bar, the outlier included" begin
        ax = first(filter(c -> c isa Axis, fig.content))
        @test length(_marker_ys(ax)) == length(data.rv)
        @test _all_framed(fig, data.rv_err[1])
        @test _all_framed(Nereus.plot_rv_components(chains, params, data), data.rv_err[1])
    end

    @testset "robust_ylim: every non-outlier point is framed, the true outlier is not" begin
        fig_r = plot_rv_timeseries(chains, params, data; robust_ylim = true)
        ax, ax_r = filter(c -> c isa Axis, fig_r.content)
        lo, hi = ax.limits[][2]
        rlo, rhi = ax_r.limits[][2]
        inl = trues(length(noise)); inl[outlier_idx] = false
        # gamma = 0 and one planet with a negligible GP: the plotted points
        # are (up to the Keplerian) the data themselves.
        @test lo < minimum(data.rv[inl]) && maximum(data.rv[inl]) < hi
        @test rlo < minimum(noise[inl]) && maximum(noise[inl]) < rhi
        @test data.rv[outlier_idx] > hi
        @test noise[outlier_idx] > rhi
    end

    @testset "no decorrelation model active: plain labels, no raw-RV layer" begin
        legend = only(filter(c -> c isa Legend, fig.content))
        labels = [entry.label[] for grp in legend.entrygroups[] for entry in grp[2]]
        @test !any(l -> l !== nothing && occursin("decorr", lowercase(l)), labels)
        @test !any(==("raw RV"), labels)
        for ins in inst_names
            @test ins in labels
        end
    end
end

@testset "SB2 timeseries and binary fold frame every point" begin
    params, data, chains, outlier_idx = _sb2_fixture()
    ts   = Nereus.plot_rv_sb2_timeseries(chains, params, data)
    fold = plot_rv_phasefold(chains, params, data; planet = 1)   # → the binary fold
    for fig in (ts, fold)
        ax = first(filter(c -> c isa Axis, fig.content))
        @test length(_marker_ys(ax)) == length(data.rv)
        @test _all_framed(fig, data.rv_err[1])
    end
    # The zoom is still there on request: the 5 km/s residual leaves the frame.
    for fig in (Nereus.plot_rv_sb2_timeseries(chains, params, data; robust_ylim = true),
                plot_rv_phasefold(chains, params, data; planet = 1, robust_ylim = true))
        ax_r = filter(c -> c isa Axis, fig.content)[2]
        @test maximum(_marker_ys(ax_r)) > ax_r.limits[][2][2]
    end
end

@testset "robust_ylim reaches every RV figure from output.plot_kwargs" begin
    kw = Dict{Symbol, Any}(:robust_ylim => true)
    for f in (plot_rv_timeseries, Nereus.plot_rv_components,
              Nereus.plot_rv_sb2_timeseries, plot_rv_phasefold)
        @test Nereus._kw_for(f, kw) == (; robust_ylim = true)
    end
end

# An RV trend (`trend_order` 1 or 2) on two instruments with different γ. The
# time series drew the Keplerians alone around zero while the residuals took
# the trend out, so on a drifting star the curve missed the data; the
# component view left the trend out of its residuals as well.
function _trend_rv_fixture(order)
    rng = MersenneTwister(23)
    n = 40
    t = 55317.0 .+ sort(5000.0 .* rand(rng, n))
    inst = [isodd(i) ? 1 : 2 for i in 1:n]
    data = Data(; t_rv = t, rv = zeros(n), rv_err = fill(3.0, n), rv_inst = inst)
    params = Params(; parametrization = ParametrizationConfig(time = :Mo), max_kplanet = 1,
                    planet_modes = [RV_ONLY], instruments = InstrumentConfig(rv = ["FEROS", "HARPS"]),
                    data = data, M_s = 1.0, trend_order = order)
    vals = Dict("P_k1" => 2050.0, "K_k1" => 45.0, "sesinw_k1" => 0.1, "secosw_k1" => 0.05,
                "Mo_k1" => 1.0, "gamma_FEROS" => 12.0, "gamma_HARPS" => -30.0,
                "sigma_FEROS" => 2.0, "sigma_HARPS" => 2.0,
                "dvdt" => -0.018, "d2vdt2" => 4e-6)
    filter!(kv -> first(kv) in params.layout.unfrozen_names, vals)
    theta = Theta{Float64}(params)
    for (k, v) in vals
        set_param!(theta, k, v)
    end
    data.rv .= rv_predictions(theta, data)[1] .+ 4.0 .* randn(rng, n)
    return params, data, _tight_chains(params, vals, rng)
end

# x and y of the curve labelled `label` in axis `ax`
function _curve(ax, label)
    p = only(filter(p -> p isa Nereus.CairoMakie.Makie.Lines &&
                         get(p.attributes, :label, nothing) !== nothing &&
                         p.label[] == label, ax.scene.plots))
    return Float64[q[1] for q in p[1][]], Float64[q[2] for q in p[1][]]
end

@testset "RV trend: time-series curve is the likelihood model (trend_order = $order)" for order in (1, 2)
    params, data, chains = _trend_rv_fixture(order)
    theta = Theta{Float64}(params)
    Nereus.set_theta_best_lp!(theta, chains, params, collect(1:size(chains, 1)))
    preds = rv_predictions(theta, data)[1]
    gam = [rv_gamma(theta, i) for i in data.rv_inst]
    tmin = minimum(data.t_rv)

    # the curve routine at the data times is the likelihood's mean minus γ
    at_data = Nereus.compute_rv_model_on_grid(theta, data, data.t_rv; include_trend = true)
    @test maximum(abs, at_data .- (preds .- gam)) < 1e-9
    @test maximum(abs, Nereus.rv_trend(theta, data, data.t_rv) .-
                       (at_data .- Nereus.compute_rv_model_on_grid(theta, data, data.t_rv))) < 1e-9

    fig = plot_rv_timeseries(chains, params, data)
    ax, ax_r = filter(c -> c isa Axis, fig.content)
    x, y = _curve(ax, "Keplerian + trend")
    @test maximum(abs, y .- Nereus.compute_rv_model_on_grid(theta, data, x .+ tmin;
                                                            include_trend = true)) < 1e-9
    # the curve follows the data: residuals about it are the noise, not the drift
    @test sort(_marker_ys(ax_r)) ≈ sort(data.rv .- preds) atol = 1e-9

    figc = Nereus.plot_rv_components(chains, params, data)
    axc, axc_r = filter(c -> c isa Axis, figc.content)
    xt, yt = _curve(axc, "trend")
    @test maximum(abs, yt .- Nereus.rv_trend(theta, data, xt .+ tmin)) < 1e-9
    xT, yT = _curve(axc, "total")
    @test maximum(abs, yT .- Nereus.compute_rv_model_on_grid(theta, data, xT .+ tmin;
                                                             include_trend = true)) < 1e-9
    @test sort(_marker_ys(axc_r)) ≈ sort(data.rv .- preds) atol = 1e-9

    # the phase-fold takes the trend OUT of the data instead of drawing it
    figf = plot_rv_phasefold(chains, params, data; planet = 1)
    axf = first(filter(c -> c isa Axis, figf.content))
    folded = data.rv .- (preds .- Nereus.compute_rv_model_planet(theta, data, data.t_rv, 1))
    @test sort(_marker_ys(axf)) ≈ sort(folded) atol = 1e-9

    # every trend coefficient carries its unit in the fitted table
    ents = Nereus.science_fitted(chains, params)
    ents isa Tuple && (ents = first(ents))
    units = Dict(k => v["unit"] for (k, v) in ents)
    @test units["dvdt"] == "m/s/day"
    order >= 2 && @test units["d2vdt2"] == "m/s/day^2"
end

@testset "trend and CCF velocity units" begin
    params, _, _ = _trend_rv_fixture(2)
    @test Nereus.sci_param_unit("dvdt", params) == ("m/s/day", false)
    @test Nereus.sci_param_unit("d2vdt2", params) == ("m/s/day^2", false)
    @test Nereus.sci_param_unit("ccf_gdot_s1_A", params) == ("km/s/day", false)
    @test Nereus.sci_param_unit("ccf_gddot_s1_A", params) == ("km/s/day^2", false)
    for nm in ("ccf_gamma_s1_A", "ccf_K_s1_A", "ccf_fwhm_s1_A", "ccf_fwhm_s1")
        @test Nereus.sci_param_unit(nm, params) == ("km/s", false)
    end
    # rendered as units in the TeX table and the figure labels, not escaped text
    for u in ("m/s/day", "m/s/day^2", "km/s", "km/s/day", "km/s/day^2")
        @test !occursin("\\_", Nereus._texunit(u)) && !occursin("/", Nereus._texunit(u))
        @test !occursin("/", Nereus._sci_unit_latex(u))
    end
end
