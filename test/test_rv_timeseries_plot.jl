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
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
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
    params = Params(; max_kplanet = 1, planet_modes = [Nereus.BINARY_RV],
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
