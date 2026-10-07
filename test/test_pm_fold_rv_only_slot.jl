# A transit fold for an RV-only slot.
#
# run_job's pm_phasefold looped over every planet slot, and plot_pm_phasefold
# folded the light curves on whatever slot it was given. With an RV-only
# signal in slot 1 (an activity Keplerian next to a transiting planet in slot
# 2) that drew "Transit_phasefold_K1_*": the light curves folded on the
# RV-only period under a flat model, the real planet's transits smeared across
# the fold. Only slots with a photometric mode are folded now.

using Nereus, Test, MCMCChains

function _rv_only_plus_transit()
    t_rv = collect(0.0:1.0:40.0)
    t_ph = collect(0.0:0.004:30.0)
    data = Data(; t_rv = t_rv, rv = zeros(length(t_rv)), rv_err = ones(length(t_rv)),
                  rv_inst = ones(Int, length(t_rv)),
                  t_phot = t_ph, flux = ones(length(t_ph)), flux_err = fill(1e-3, length(t_ph)),
                  phot_inst = ones(Int, length(t_ph)))
    params = Params(; max_kplanet = 2, planet_modes = [RV_ONLY, RVPM],
                      instruments = InstrumentConfig(rv = ["I1"], pm = ["TESS"]),
                      data = data, M_s = 1.0, R_s = 1.0)
    vals = Dict("P_k1" => 1.39, "K_k1" => 20.0, "sesinw_k1" => 0.0, "secosw_k1" => 0.0,
                "Mo_k1" => 1.0,
                "P_k2" => 2.83, "K_k2" => 50.0, "sesinw_k2" => 0.0, "secosw_k2" => 0.0,
                "Mo_k2" => 2.0, "b_k2" => 0.3, "rr_k2" => 0.1,
                "gamma_I1" => 0.0, "sigma_I1" => 1.0,
                "offset_TESS" => 0.0, "jitter_TESS" => 1e-4, "q1_TESS" => 0.4, "q2_TESS" => 0.3)
    theta = Theta{Float64}(params)
    for (k, v) in vals
        set_param!(theta, k, v)
    end
    data.rv .= Nereus.rv_predictions(theta, data)[1]
    data.flux .= Nereus.phot_predictions(theta, data)[1]
    nm = params.layout.unfrozen_names
    missing_vals = setdiff(nm, keys(vals))
    isempty(missing_vals) || error("fixture lacks values for $missing_vals")
    cols = vcat(Symbol.(nm), [:lp])
    arr = zeros(8, length(cols), 1)
    for i in 1:8
        arr[i, 1:length(nm), 1] = [vals[x] * (1 + 1e-6 * i) for x in nm]
        arr[i, end, 1] = -0.1 * i
    end
    return params, data, Chains(arr, cols)
end

@testset "transit folds: transiting slots only" begin
    params, data, chains = _rv_only_plus_transit()
    out = mktempdir()
    @test_logs (:warn, r"has no transit") match_mode = :any begin
        @test plot_pm_phasefold(chains, params, data; planet = 1, output = out) === nothing
    end
    @test !isfile(joinpath(out, "models", "Transit_phasefold_K1_TESS.png"))
    plot_pm_phasefold(chains, params, data; planet = 2, output = out)
    @test isfile(joinpath(out, "models", "Transit_phasefold_K2_TESS.png"))
    # and run_job's dispatcher asks only for the transiting slot
    out2 = mktempdir()
    @test Nereus._dispatch_plot("pm_phasefold", chains, params, data, out2,
                                Dict{Symbol, Any}()) == "models/Transit_phasefold_K*_*.png"
    @test readdir(joinpath(out2, "models")) == ["Transit_phasefold_K2_TESS.png"]
end
