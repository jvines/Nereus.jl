# The RV phase fold's y-zoom on a sparse fold.
#
# Below 51 kept points the fold draws no binned overlay, so the points are the
# only picture of the orbit. The zoom framed the 10-90% quantiles of the data
# all the same, and on a jitter-dominated hot Jupiter (33 RVs, K = 350 m/s,
# ~200 m/s scatter) seven points sat outside the frame, in both panels. Every
# point but a true outlier must be in view; the outlier still is not.

using Nereus, Test, Random, MCMCChains
using CairoMakie: Axis

function _sparse_jittery_fold()
    rng = MersenneTwister(33)
    t = sort(500.0 .* rand(rng, 33))
    data = Data(; t_rv = t, rv = zeros(length(t)), rv_err = fill(30.0, length(t)),
                  rv_inst = ones(Int, length(t)))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                      instruments = InstrumentConfig(rv = ["I1"]),
                      data = data, M_s = 1.0)
    vals = Dict("P_k1" => 2.83, "K_k1" => 350.0, "sesinw_k1" => 0.0,
                "secosw_k1" => 0.0, "Mo_k1" => 1.0,
                "gamma_I1" => 0.0, "sigma_I1" => 200.0)
    theta = Theta{Float64}(params)
    for (k, v) in vals
        set_param!(theta, k, v)
    end
    model = Nereus.rv_predictions(theta, data)[1]
    noise = 200.0 .* randn(rng, length(t))
    noise[17] = 5000.0                                  # the one true outlier
    data.rv .= model .+ noise
    nm = params.layout.unfrozen_names
    cols = vcat(Symbol.(nm), [:lp])
    n = 40
    arr = zeros(n, length(cols), 1)
    for i in 1:n                                        # a tight, non-degenerate posterior
        arr[i, 1:length(nm), 1] = [vals[x] * (1 + 1e-4 * randn(rng)) for x in nm]
        arr[i, end, 1] = -0.1 * i
    end
    return params, data, Chains(arr, cols), noise
end

@testset "sparse RV fold frames every point but true outliers" begin
    params, data, chains, noise = _sparse_jittery_fold()
    fig = plot_rv_phasefold(chains, params, data; n_draws = 50)
    ax, ax_r = filter(c -> c isa Axis, fig.content)
    lo, hi = ax.limits[][2]
    rlo, rhi = ax_r.limits[][2]
    inl = trues(length(noise)); inl[17] = false
    # gamma = 0 and one planet: the folded points are the data themselves
    @test lo < minimum(data.rv[inl]) && maximum(data.rv[inl]) < hi
    @test rlo < minimum(noise[inl]) && maximum(noise[inl]) < rhi
    @test data.rv[17] > hi && noise[17] > rhi
    # and the zoom is still a zoom: nothing like 5000 m/s of headroom
    @test hi < 0.5 * data.rv[17]
end
