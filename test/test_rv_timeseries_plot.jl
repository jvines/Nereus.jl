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
# CeleriteRotation GP with gp_sigma essentially 0) and checks: every point
# but one true, 5-robust-sigma outlier is framed in both panels, and the
# legend carries plain instrument names with no "(decorr.)" suffix and no
# "raw RV" entry.

using Nereus, Test, Random, MCMCChains
using CairoMakie: Axis, Legend

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

    nm = params.layout.unfrozen_names
    cols = vcat(Symbol.(nm), [:lp])
    ndraws = 40
    arr = zeros(ndraws, length(cols), 1)
    for i in 1:ndraws                               # a tight, non-degenerate posterior
        arr[i, 1:length(nm), 1] = [vals[x] * (1 + 1e-4 * randn(rng)) for x in nm]
        arr[i, end, 1] = -0.1 * i
    end
    return params, data, Chains(arr, cols), noise, outlier_idx, inst_names
end

@testset "sparse multi-instrument RV timeseries" begin
    params, data, chains, noise, outlier_idx, inst_names = _sparse_multi_inst_rv_fixture()
    fig = plot_rv_timeseries(chains, params, data)
    ax, ax_r = filter(c -> c isa Axis, fig.content)
    lo, hi = ax.limits[][2]
    rlo, rhi = ax_r.limits[][2]
    inl = trues(length(noise)); inl[outlier_idx] = false

    @testset "every non-outlier point is framed, the true outlier is not" begin
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
