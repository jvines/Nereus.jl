# Which draw a figure pictures, on a trans-dim chain.
#
# A trans-dim chain stores EVERY slot's parameters on every row; a slot that
# is inactive in a draw keeps the values it was parked at -- a real orbit
# with a real amplitude. The figures rebuilt draws without the row's active
# set, so parked slots entered the model:
#
#   - `_theta_from_chain_row` / `_theta_best_lp` built `Theta(params)` with no
#     trans-dim state, so every slot counted as present (IAD residuals, the
#     sky-plane orbit, every posterior fan and band).
#   - `winning_td_state` switched on slots `1:modal_np`; after a death in the
#     middle of the slot list the live slots are others, so the RV fold drew
#     a parked slot and skipped the live one (`planet > modal_np`).
#   - per-planet figures conditioned on `n_planets ≥ k`, and fell back to ALL
#     draws when that met an empty best-fit cut.
#
# And several residual figures plotted the per-parameter MEDIAN, which is not
# a model on the posterior ridge. The fixtures here are noiseless: the live
# model fits exactly, so anything a parked slot leaks shows up as residual.

using Nereus, Test, Random, MCMCChains, LinearAlgebra
using Nereus: IADData

# Two RV slots. Slot 1 DIED mid-list and sits parked at a loud K = 30 m/s;
# slot 2 is alive and is the only signal in the data. n_planets == 1.
function _rv_middeath()
    t = collect(0.0:3.0:300.0)
    data = Data(; t_rv = t, rv = zeros(length(t)), rv_err = ones(length(t)),
                  rv_inst = ones(Int, length(t)))
    params = Params(; max_kplanet = 2, planet_modes = fill(RV_ONLY, 2),
                      instruments = InstrumentConfig(rv = ["I1"]),
                      data = data, M_s = 1.0)
    vals = Dict("P_k1" => 10.0, "K_k1" => 30.0, "sesinw_k1" => 0.0,
                "secosw_k1" => 0.0, "Mo_k1" => 1.0,
                "P_k2" => 37.0, "K_k2" => 5.0, "sesinw_k2" => 0.2,
                "secosw_k2" => 0.1, "Mo_k2" => 2.0,
                "gamma_I1" => 0.0, "sigma_I1" => 1.0)
    live = Theta{Float64}(params; td = let tds = TransDimState(; max_planets = 2)
        activate_planet!(tds, 2); tds end)
    for (k, v) in vals
        set_param!(live, k, v)
    end
    data.rv .= Nereus.rv_predictions(live, data)[1]
    nm = params.layout.unfrozen_names
    cols = vcat(Symbol.(nm), [:n_planets, :planet_active_1, :planet_active_2, :lp])
    arr = zeros(4, length(cols), 1)
    for i in 1:4
        arr[i, 1:length(nm), 1] = [vals[x] for x in nm]
        arr[i, length(nm)+1:end, 1] = [1.0, 0.0, 1.0, i == 1 ? 0.0 : -50.0 * i]
    end
    return params, data, Chains(arr, cols), live
end

# The same death mid-list, in transit photometry: slot 1 parked at a deep
# rr = 0.15, slot 2 live and the only transit in the data.
function _phot_middeath()
    t = collect(0.0:0.004:30.0)
    data = Data(; t_phot = t, flux = ones(length(t)), flux_err = fill(1e-3, length(t)),
                  phot_inst = ones(Int, length(t)))
    params = Params(; max_kplanet = 2, planet_modes = fill(PM_ONLY, 2),
                      instruments = InstrumentConfig(pm = ["TESS"]),
                      data = data, M_s = 1.0, R_s = 1.0)
    vals = Dict("P_k1" => 3.0, "sesinw_k1" => 0.0, "secosw_k1" => 0.0, "Mo_k1" => 0.5,
                "b_k1" => 0.2, "rr_k1" => 0.15,
                "P_k2" => 7.3, "sesinw_k2" => 0.0, "secosw_k2" => 0.0, "Mo_k2" => 2.0,
                "b_k2" => 0.3, "rr_k2" => 0.08,
                "offset_TESS" => 0.0, "jitter_TESS" => 1e-4, "q1_TESS" => 0.4,
                "q2_TESS" => 0.3)
    live = Theta{Float64}(params; td = let tds = TransDimState(; max_planets = 2)
        activate_planet!(tds, 2); tds end)
    for (k, v) in vals
        set_param!(live, k, v)
    end
    data.flux .= Nereus.phot_predictions(live, data)[1]
    nm = params.layout.unfrozen_names
    cols = vcat(Symbol.(nm), [:n_planets, :planet_active_1, :planet_active_2, :lp])
    arr = zeros(4, length(cols), 1)
    for i in 1:4
        arr[i, 1:length(nm), 1] = [vals[x] for x in nm]
        arr[i, length(nm)+1:end, 1] = [1.0, 0.0, 1.0, i == 1 ? 0.0 : -50.0 * i]
    end
    return params, data, Chains(arr, cols)
end

# Three RV slots; the second planet is in slot 2 in nine draws and was
# label-switched into slot 3 in the tenth -- which has the highest lp. Slot 1
# is live throughout. Winning model: slots {1, 2}.
function _rv_labelswitch()
    t = collect(0.0:3.0:300.0)
    data = Data(; t_rv = t, rv = zeros(length(t)), rv_err = ones(length(t)),
                  rv_inst = ones(Int, length(t)))
    params = Params(; max_kplanet = 3, planet_modes = fill(RV_ONLY, 3),
                      instruments = InstrumentConfig(rv = ["I1"]),
                      data = data, M_s = 1.0)
    orb(P, K, Mo) = Dict("P" => P, "K" => K, "sesinw" => 0.1, "secosw" => 0.1, "Mo" => Mo)
    vals = Dict{String, Float64}("gamma_I1" => 0.0, "sigma_I1" => 1.0)
    for (k, o) in ((1, orb(10.0, 8.0, 1.0)), (2, orb(37.0, 5.0, 2.0)), (3, orb(37.0, 5.0, 2.0)))
        for (n, v) in o
            vals["$(n)_k$k"] = v
        end
    end
    live = Theta{Float64}(params; td = let tds = TransDimState(; max_planets = 3)
        activate_planet!(tds, 1); activate_planet!(tds, 2); tds end)
    for (k, v) in vals
        set_param!(live, k, v)
    end
    data.rv .= Nereus.rv_predictions(live, data)[1]
    nm = params.layout.unfrozen_names
    cols = vcat(Symbol.(nm), [:n_planets, :planet_active_1, :planet_active_2,
                              :planet_active_3, :lp])
    arr = zeros(10, length(cols), 1)
    for i in 1:10
        arr[i, 1:length(nm), 1] = [vals[x] for x in nm]
        arr[i, length(nm)+1:end, 1] = i < 10 ? [2.0, 1.0, 1.0, 0.0, -1.0 * i] :
                                               [2.0, 1.0, 0.0, 1.0, 0.0]
    end
    return params, data, Chains(arr, cols)
end

_fig_texts(fig) = [string(p[:text][]) for c in fig.content if c isa Nereus.Axis
                   for p in c.scene.plots if p isa Nereus.CairoMakie.Makie.Text]

# The IAD fixture of test_epoch_astrometry_orbit.jl (included before this file).
@testset "trans-dim draws in the figures" begin

    @testset "a row's active set, from planet_active_<k>" begin
        params, _, chains, _ = _rv_middeath()
        tdc = Nereus._td_cols(chains, params)
        @test tdc !== nothing
        tds = Nereus._row_td_state(tdc, params, 1)
        @test tds.planet_active == [false, true]
        @test Nereus._planet_present_idx(chains, params, 1) == Int[]
        @test Nereus._planet_present_idx(chains, params, 2) == 1:4
        # the best-lp Theta carries it, so only slot 2 is in the model
        θ = Nereus._theta_best_lp(chains, params)
        @test collect(planet_indices(θ)) == [2]
        # a fixed-dimension chain has none
        fixed = chains[:, [Symbol(n) for n in params.layout.unfrozen_names] ∪ [:lp], :]
        @test Nereus._td_cols(fixed, params) === nothing
    end

    @testset "set_theta_best_lp! uses the draw's own slots, not 1:modal_np" begin
        params, data, chains, live = _rv_middeath()
        idx = collect(1:4)
        # winning_td_state now takes the modal live pattern, not 1:modal_np
        @test Nereus.winning_td_state(chains, params, idx, 1).planet_active == [false, true]
        # and set_theta_best_lp! overrides whatever mask it is handed
        θ = Theta{Float64}(params; td = let tds = TransDimState(; max_planets = 2)
            activate_planet!(tds, 1); tds end)
        Nereus.set_theta_best_lp!(θ, chains, params, idx)
        @test collect(planet_indices(θ)) == [2]
        @test maximum(abs, data.rv .- Nereus.rv_predictions(θ, data)[1]) < 1e-9
    end

    @testset "posterior bands leave parked slots out" begin
        params, data, chains, live = _rv_middeath()
        tg = collect(range(0.0, 300.0; length = 200))
        want = Nereus.compute_rv_model_on_grid(live, data, tg)
        b = Nereus.compute_ci_bands(chains, params, data, tg; n_draws = 20, bf_cutoff = Inf)
        # with slot 1 counted the band carries its 30 m/s
        @test maximum(abs, b.median .- want) < 1e-9
        # a slot that is never present has no band
        b1 = Nereus.compute_ci_bands(chains, params, data, tg; planet = 1, n_draws = 20,
                                     bf_cutoff = Inf)
        @test all(isnan, b1.median)
    end

    @testset "the RV fold draws the live slot and skips the parked one" begin
        params, data, chains, _ = _rv_middeath()
        out = mktempdir()
        plot_rv_phasefold(chains, params, data; planet = 2, output = out, n_draws = 20)
        @test isfile(joinpath(out, "models", "RV_phasefold_K2.png"))
        @test_logs (:warn, r"not active in the winning model") match_mode = :any begin
            plot_rv_phasefold(chains, params, data; planet = 1, output = out, n_draws = 20)
        end
        @test !isfile(joinpath(out, "models", "RV_phasefold_K1.png"))
        # and the runner publishes the pattern because K2 exists
        out2 = mktempdir()
        @test Nereus._dispatch_plot("rv_phasefold", chains, params, data, out2,
                                    Dict{Symbol, Any}(:n_draws => 20)) ==
              "models/RV_phasefold_K*.png"
        @test readdir(joinpath(out2, "models")) == ["RV_phasefold_K2.png"]
    end

    @testset "photometry: the live slot is folded, the parked one is not" begin
        params, data, chains = _phot_middeath()
        out = mktempdir()
        plot_pm_phasefold(chains, params, data; planet = 2, output = out)
        @test isfile(joinpath(out, "models", "Transit_phasefold_K2_TESS.png"))
        @test_logs (:warn, r"not active in the winning model") match_mode = :any begin
            plot_pm_phasefold(chains, params, data; planet = 1, output = out)
        end
        @test !isfile(joinpath(out, "models", "Transit_phasefold_K1_TESS.png"))
        plot_pm_timeseries(chains, params, data; output = out)
        @test isfile(joinpath(out, "models", "pm_timeseries_TESS.png"))
        @test !isempty(plot_transit_overlay_fit(chains, params, data; planet = 2,
                                                output = out).content)
        @test isempty(plot_transit_overlay_fit(chains, params, data; planet = 1,
                                               output = out).content)
        @test_logs (:warn, r"no draw has planet 1") match_mode = :any begin
            plot_ttv_oc(chains, data, params; planet_a_k = 1, planet_b_k = 0)
        end
    end

    @testset "IAD residuals: max-lp draw, parked slot not subtracted" begin
        target, _, chains = _epoch_orbit_target(; two = true, noise = false, active = [1])
        fig = plot_iad_residuals(chains, target.params, target.data)
        txt = only(filter(t -> occursin("χ²/N", t), _fig_texts(fig)))
        # noiseless and only slot 1 in the data: the fit is exact. With slot 2
        # subtracted as well, χ²/N was ~1 at σ = 0.08.
        @test parse(Float64, match(r"χ²/N = ([0-9.eE+-]+)", txt)[1]) < 1e-6
    end

    @testset "sky-plane orbit: nothing for a slot that is never active" begin
        target, _, chains = _epoch_orbit_target(; two = true, noise = false, active = [1])
        out = mktempdir()
        @test isempty(plot_orbit_skyplane(chains, target.params, target.data;
                                          planet_idx = 2, output = out).content)
        @test Nereus._dispatch_plot("orbit_skyplane", chains, target.params, target.data,
                                    out, Dict{Symbol, Any}()) == "models/orbit_skyplane_K*.png"
        @test readdir(joinpath(out, "models")) == ["orbit_skyplane_K1.png"]
    end

    @testset "science tables report the live slot, not 1:modal_np" begin
        params, _, chains, _ = _rv_middeath()
        fitted, cond = Nereus.science_fitted(chains, params)
        fnames = first.(fitted)
        @test "K_k2" in fnames
        @test !("K_k1" in fnames)
        @test cond["planet_slots"] == [2]
        derived, dcond = Nereus.science_derived(chains, params)
        @test any(e -> endswith(first(e), "_k2"), derived)
        @test !any(e -> endswith(first(e), "_k1"), derived)
        @test dcond["planet_slots"] == [2]
    end

    @testset "chains without :lp keep the draws' live slots" begin
        # rjmcmc, MoMS, daedalus and trans-dim PT write no :lp; the median
        # fallback used to keep the caller's 1:modal_np mask.
        params, data, chains, _ = _rv_middeath()
        nolp = chains[:, filter(!=(:lp), names(chains, :parameters)), :]
        θ = Theta{Float64}(params; td = let tds = TransDimState(; max_planets = 2)
            activate_planet!(tds, 1); tds end)
        Nereus.set_theta_best_lp!(θ, nolp, params, collect(1:4))
        @test collect(planet_indices(θ)) == [2]
        out = mktempdir()
        @test Nereus._dispatch_plot("rv_phasefold", nolp, params, data, out,
                                    Dict{Symbol, Any}(:n_draws => 10)) ==
              "models/RV_phasefold_K*.png"
        @test readdir(joinpath(out, "models")) == ["RV_phasefold_K2.png"]
    end

    @testset "slot identity is the modal pattern, not the max-lp draw's" begin
        params, data, chains = _rv_labelswitch()
        @test Nereus._winning_planet_slots(chains, params) == [1, 2]
        out = mktempdir()
        @test Nereus._dispatch_plot("rv_phasefold", chains, params, data, out,
                                    Dict{Symbol, Any}(:n_draws => 10)) ==
              "models/RV_phasefold_K*.png"
        @test sort(readdir(joinpath(out, "models"))) ==
              ["RV_phasefold_K1.png", "RV_phasefold_K2.png"]
        fnames = first.(first(Nereus.science_fitted(chains, params)))
        @test "K_k2" in fnames && !("K_k3" in fnames)
    end

    @testset "the timeseries Keplerian line carries live slots only" begin
        params, data, chains, _ = _rv_middeath()
        fig = plot_rv_timeseries(chains, params, data)
        ax = first(c for c in fig.content if c isa Nereus.Axis)
        kep = [p for p in ax.scene.plots if p isa Nereus.CairoMakie.Makie.Lines &&
               string(p[:label][]) == "Keplerian"]
        @test length(kep) == 1
        # slot 2 alone peaks near its K = 5 m/s; with the parked slot 1 it was 35
        @test maximum(abs(pt[2]) for pt in only(kep)[1][]) < 6.0
    end

    @testset "PPC and LOO rebuild each draw with its live slots" begin
        params, data, chains, _ = _rv_middeath()
        ppc = posterior_predictive_check(chains, params, data; n_draws = 4)
        # noiseless data, exact live model: every χ² is zero. Slots 1:n_planets
        # made it a 30 m/s mismatch at every draw.
        @test ppc.summary["rv_red_chi2_median_model"] < 1e-6
        @test ppc.summary["rv_red_chi2_per_draw_p50"] < 1e-6
        flat, _ = Nereus._flatten_chains(chains)
        @test collect(planet_indices(Nereus._theta_from_row(flat, 1, params))) == [2]
    end

    @testset "detection limits and the TTV envelope use live draws" begin
        params, _, chains, _ = _rv_middeath()
        # slot 1 is never live: no draws, not four parked ones
        @test_throws ArgumentError detection_limits(chains, params; planet = 1)
        pparams, _, pchains = _phot_middeath()
        @test_throws ErrorException Nereus.ttvc_envelope(pchains, pparams;
            tcs_observed = [1.0, 2.0], planet_a_k = 1, planet_b_k = 2)
    end

    @testset "runner: transit overlays per live transiting slot" begin
        params, data, chains = _phot_middeath()
        out = mktempdir()
        @test Nereus._dispatch_plot("transit_overlay", chains, params, data, out,
                                    Dict{Symbol, Any}()) == "models/transit_overlay_K*.png"
        @test readdir(joinpath(out, "models")) == ["transit_overlay_K2.png"]
    end

    @testset "_kw_for: forwarded keywords, and `planet` where the branch leaves it" begin
        kw = Dict{Symbol, Any}(:bf_cutoff => 1.0, :ylabel => "x", :planet_b_k => 0)
        @test Dict(pairs(Nereus._kw_for(plot_ttv_oc, kw;
                                        forwards_to = (Nereus.plot_ttv_diagram,)))) ==
              Dict(:ylabel => "x", :planet_b_k => 0)
        @test Nereus._kw_for(plot_rm, Dict{Symbol, Any}(:planet => 2)) == (; planet = 2)
    end

    @testset "runner: plot_kwargs reach only the figures that take them" begin
        target, _, chains = _epoch_orbit_target()
        out = mktempdir()
        cfg = Dict{String, Any}(
            "output" => Dict{String, Any}(
                "plots" => ["iad_residuals", "orbit_skyplane"],
                "plot_kwargs" => Dict{String, Any}("n_draws" => 5, "bogus_key" => 1),
                "show_progress" => false),
            "sampler" => Dict{String, Any}("name" => "pt_emcee",
                                           "kwargs" => Dict{String, Any}("n_walkers" => 1)))
        # `n_draws` is orbit_skyplane's and used to throw inside iad_residuals;
        # `bogus_key` is nobody's and is reported, not thrown.
        gen = @test_logs (:warn, r"plot_kwargs ignored") match_mode = :any begin
            Nereus._make_plots(cfg, chains, target.params, target.data, out)
        end
        @test "models/iad_residuals.png" in gen
        @test isfile(joinpath(out, "plots", "models", "iad_residuals.png"))
        @test Nereus._kw_for(plot_iad_residuals, Dict{Symbol, Any}(:n_draws => 5,
                                                                   :bf_cutoff => 3.0)) ==
              (; bf_cutoff = 3.0)
    end
end
