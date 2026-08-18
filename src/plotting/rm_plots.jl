# Rossiter-McLaughlin plot — the in-transit RV anomaly induced by a transiting
# planet crossing the rotating stellar disk (Hirano+ 2011 model, src/rm.jl).
# Top: RV with the Keplerian/systemic removed (= RM signal + noise), folded on
# the RM planet and zoomed to the transit window, with the posterior RM model
# curve + 1/2/3σ band. Bottom: residual after the full model (RM included).
# The shape encodes the sky-projected obliquity λ (v·sin i_* sets the amplitude).

using CairoMakie

# Dense RM-anomaly curve ΔRV_RM(t) at a parameter point, decoding planets exactly
# as rv_predictions/the likelihood do (so the curve matches the fitted model).
function _rm_curve_at(theta, data, t_grid)
    param = theta.params.config.parametrization
    t_ref = data.t_ref
    p_idx = planet_indices(theta)
    Ps = Float64[]; es = Float64[]; ws = Float64[]; Tps = Float64[]
    for k in p_idx
        has_K(theta.params.layout.planet_blocks[k]) || continue
        P = Float64(planet_P(theta, k)); e, w = planet_e_w(theta, k)
        e = clamp(Float64(e), 0.0, 0.999); w = Float64(w)
        ta = Float64(planet_time_anchor(theta, k))
        Tp = param.time === :Mo ? t_ref - ta * P / (2π) :
             param.time === :Tp ? ta : tc_to_tp(ta, P, e, w)
        push!(Ps, P); push!(es, e); push!(ws, w); push!(Tps, Tp)
    end
    n_rm, st = _decode_rm_state(theta, p_idx, Ps)
    (n_rm isa Int && n_rm > 0) || return zeros(length(t_grid))
    return Float64[rm_contribution(t, n_rm, st, Ps, es, ws, Tps) for t in t_grid]
end

# Transit centre + period of planet k (for the fold).
function _planet_tc_P(theta, data, k)
    param = theta.params.config.parametrization
    P = Float64(planet_P(theta, k)); e, w = planet_e_w(theta, k)
    e = clamp(Float64(e), 0.0, 0.999); w = Float64(w)
    ta = Float64(planet_time_anchor(theta, k))
    Tc = param.time === :Tc ? ta :
         param.time === :Tp ? tp_to_tc(ta, P, e, w) :
         tp_to_tc(data.t_ref - ta * P / (2π), P, e, w)
    return Tc, P
end

"""
    plot_rm(chains, params, data; planet=nothing, output=nothing, fmt=:png,
            save_pdf=false, figsize=FIG_RV, bf_cutoff=10.0, n_draws=400) -> Figure

Rossiter-McLaughlin in-transit RV anomaly for an RM-enabled fit. Picks the first
planet whose mode carries RM (override with `planet`). Throws if none.
Saved as `models/rm_anomaly_K<k>.<fmt>`.
"""
function plot_rm(chains, params, data;
                  planet::Union{Nothing, Int} = nothing,
                  output::Union{Nothing, String} = nothing,
                  fmt::Symbol = :png, save_pdf::Bool = false,
                  figsize = FIG_RM, bf_cutoff::Real = 10.0, n_draws::Int = 400)
    modes = params.config.planet_modes
    rm_k = planet === nothing ? findfirst(has_any_rm, modes) : planet
    rm_k === nothing && throw(ArgumentError("plot_rm: no RM-enabled planet (need an *_RM mode)"))

    with_theme(nereus_theme()) do
        # winning-model best-lp reference theta (same selection as rv_timeseries)
        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets])); modal_np = _mode_int_local(np_all)
            active_idx = findall(np_all .== modal_np)
            isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
            td_state = winning_td_state(chains, params, active_idx, modal_np)
            active_idx = winning_config_idx(chains, params, active_idx, td_state)
            theta_med = Theta{Float64}(params; td = td_state)
        else
            active_idx = collect(1:_n_flat_draws(chains)); theta_med = Theta{Float64}(params)
        end
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff = bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        set_theta_best_lp!(theta_med, chains, params, active_idx)

        # RM-free baseline: same theta with v·sin i_* = 0 → rv_predictions drops RM.
        theta0 = Theta{Float64}(params, copy(theta_med.values); td = theta_med.td)
        vidx = theta_med.params.layout.systemic.v_sin_i_star
        vidx == 0 && throw(ArgumentError("plot_rm: fit has no v_sin_i_star slot"))
        theta0.values[vidx] = 0.0
        pred_full, _ = rv_predictions(theta_med, data)
        pred_norm, _ = rv_predictions(theta0, data)
        rm_resid = data.rv .- pred_norm                 # RM signal + noise
        full_resid = data.rv .- pred_full               # noise (RM removed)

        Tc, P = _planet_tc_P(theta_med, data, rm_k)
        # in-transit window: where the RM model is non-negligible, + 40% margin
        wide = collect(range(Tc - P / 6, Tc + P / 6; length = 1500))
        rmw = _rm_curve_at(theta_med, data, wide)
        amax = maximum(abs, rmw)
        half = amax > 0 ? 1.4 * maximum(abs(wide[i] - Tc) for i in eachindex(wide)
                                        if abs(rmw[i]) > 0.02 * amax; init = 0.05) : 0.1
        hr_of(t) = (mod(t - Tc + P / 2, P) - P / 2) * 24.0        # hours from mid-transit

        fig = Figure(; size = figsize)
        ga = fig[1, 1] = GridLayout()
        # xtickalign/xminortickalign = 1 -> ticks point INWARD. The theme leaves
        # Makie's default (outward), which makes the top axis reserve protrusion
        # space below its bottom spine for the major+minor ticks. That protrusion
        # is NOT a layout gap, so rowgap! cannot remove it, and it shows up as a
        # white band between two panels meant to read as one stacked figure.
        ax  = Axis(ga[1, 1]; ylabel = rich("RM anomaly  ΔRV (m s", superscript("-1"), ")"),
                   xtickalign = 1, xminortickalign = 1,
                   ytickalign = 1, yminortickalign = 1)
        ax_r = Axis(ga[2, 1]; xlabel = "Time from mid-transit (hours)",
                    ylabel = "Residual",
                    xtickalign = 1, xminortickalign = 1,
                    ytickalign = 1, yminortickalign = 1)
        rowsize!(ga, 1, Auto(3)); rowsize!(ga, 2, Auto(1))
        linkxaxes!(ax, ax_r); hidexdecorations!(ax; grid = false, ticks = false)
        # Butt the residual panel against the main one. Without this the shared
        # x-axis still reserves Makie's default row gap, leaving a white band
        # between two panels that are meant to read as one stacked figure.
        rowgap!(ga, 0)

        # dense model curve + 1/2/3σ band over one transit window
        t_dense = collect(range(Tc - half, Tc + half; length = 600))
        hr_dense = (t_dense .- Tc) .* 24.0
        npool = min(n_draws, length(active_idx))
        sel = active_idx[unique(round.(Int, range(1, length(active_idx); length = npool)))]
        M = Matrix{Float64}(undef, length(t_dense), length(sel))
        for (d, idx) in enumerate(sel)
            M[:, d] = _rm_curve_at(_theta_from_chain_row(chains, params, idx), data, t_dense)
        end
        b = _bands123(M)
        _band_stack!(ax, hr_dense, b)
        # Model curve in the dedicated model color (black) — NOT post/inst blue,
        # which is the data-point color.
        lines!(ax, hr_dense, _rm_curve_at(theta_med, data, t_dense);
               color = NEREUS_COLORS.model, linewidth = MODEL_LW)
        hlines!(ax, 0.0; color = NEREUS_COLORS.zero_line, linestyle = :dash, linewidth = 1.0)

        # data within the zoom window, per instrument
        for i in 1:length(params.config.instruments.rv_names)
            mask = (data.rv_inst .== i) .& (abs.(hr_of.(data.t_rv)) .<= half * 24.0)
            any(mask) || continue
            hx = hr_of.(data.t_rv[mask]); err = data.rv_err[mask]
            mc = inst_color(i); mk = inst_marker(i)
            errorbars!(ax, hx, rm_resid[mask], err; color = mc, linewidth = ERRBAR_LW)
            scatter!(ax, hx, rm_resid[mask]; color = mc, marker = mk, markersize = 11,
                     strokewidth = 1.5, strokecolor = :black, label = params.config.instruments.rv_names[i])
            errorbars!(ax_r, hx, full_resid[mask], err; color = mc, linewidth = ERRBAR_LW)
            scatter!(ax_r, hx, full_resid[mask]; color = mc, marker = mk, markersize = 11,
                     strokewidth = 1.5, strokecolor = :black)
        end
        hlines!(ax_r, 0.0; color = NEREUS_COLORS.zero_line, linestyle = :dash, linewidth = MODEL_LW)
        xlims!(ax, -half * 24.0, half * 24.0)
        axislegend(ax; position = :rb, framevisible = false)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "rm_anomaly_K$(rm_k).$fmt"), fig;
                       save_pdf = save_pdf, px_per_unit = 3)
        end
        return fig
    end
end
