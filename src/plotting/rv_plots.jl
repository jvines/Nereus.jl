# RV plots: timeseries and phase-fold, matching EMPEROR style.

"""
    plot_rv_timeseries(chains, params, data;
                        output=nothing, fmt=:png, figsize=FIG_RV)

RV observations over time with best-fit model and residuals.
Multi-instrument data shown with different markers/colors.
3:1 data/residual panel layout.
"""
function plot_rv_timeseries(chains, params, data;
                             output::Union{Nothing, String}=nothing,
                             fmt::Symbol=:png,
                             save_pdf::Bool=false,
                             figsize=FIG_RV,
                             bf_cutoff::Real=10.0,
                             show_keplerian::Bool=true,
                             show_gp::Bool=true,
                             show_total::Bool=true,
                             show_activity::Bool=true,
                             decompose::Bool=false)
    with_theme(nereus_theme()) do
        fig = Figure(; size=figsize)
        ga = fig[1, 1] = GridLayout()

        ax = Axis(ga[1, 1]; ylabel=rich("RV (m s", superscript("-1"), ")"))
        ax_r = Axis(ga[2, 1]; ylabel="Residuals", yticks = LinearTicks(3),
                        ytickformat = vs -> [string(round(Int, v)) for v in vs])
        rowsize!(ga, 1, Auto(3))
        rowsize!(ga, 2, Auto(1))
        linkxaxes!(ax, ax_r)
        hidexdecorations!(ax; grid=false)
        # glue the residual panel to the main one: they share the x axis
        rowgap!(ga, 0)

        inst_names = params.config.instruments.rv_names
        tmin = minimum(data.t_rv)

        # Median-parameter model conditioned on the **winning** model:
        # use samples where N_p == modal_np and the modal noise-active
        # pattern. Anything else is a non-winning planet count or noise
        # configuration, and it doesn't belong in the headline plot.
        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            modal_np = _mode_int_local(np_all)
            active_idx = findall(np_all .== modal_np)
            isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
            td_state = winning_td_state(chains, params, active_idx, modal_np)
            # Condition on the WINNING noise configuration: the global max-lp
            # draw is the overfitting (GP-injecting) model, not the occupancy
            # winner. See winning_config_idx.
            active_idx = winning_config_idx(chains, params, active_idx, td_state)
            theta_med = Theta{Float64}(params; td=td_state)
        else
            active_idx = collect(1:_n_flat_draws(chains))
            theta_med = Theta{Float64}(params)
        end
        # EMPEROR best-fit cluster — keep samples within ln(bf_cutoff)
        # of max lp. Default BF=10 (Jeffreys "strong"), user-overridable.
        top_idx = _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff)
        active_idx = intersect(active_idx, top_idx)
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        # Max-lp reference theta (jointly consistent; see set_theta_best_lp!)
        set_theta_best_lp!(theta_med, chains, params, active_idx)

        preds, _ = rv_predictions(theta_med, data)
        residuals = data.rv .- preds

        # --- Activity model at the data times (what the noise model removes) ----
        # AD is a per-point regression on the activity indicators; a GP is a smooth
        # latent. Either way we evaluate its mean AT THE DATA TIMES and overlay the
        # DECORRELATED RV (raw − activity) on the faded raw RV, so the variance the
        # model removes is directly visible — the cloud collapses. The fast
        # Keplerian is NOT drawn as a continuous line over the (long) baseline: it
        # smears into an unreadable band; its shape lives in the phase-fold.
        act_at_data = zeros(length(data.t_rv))
        if show_activity
            # ActivityDecorrelation: full mean − mean with the C coefficients zeroed
            if any(nm isa ActivityDecorrelation for nm in params.config.noise_models)
                theta_noad = Theta{Float64}(params, copy(theta_med.values); td=theta_med.td)
                for nm in params.config.noise_models
                    nm isa ActivityDecorrelation || continue
                    for cnm in noise_param_names(nm, params.config.instruments)
                        startswith(cnm, "C_") || continue
                        idx = get(params.layout.name_to_idx, cnm, 0)
                        idx == 0 || (theta_noad.values[idx] = 0.0)
                    end
                end
                preds_noad, _ = rv_predictions(theta_noad, data)
                act_at_data .+= preds .- preds_noad
            end
            # GP/celerite latent mean at the data times, if a GP is active
            gp_at_data = try
                channel_gp_mean_at(theta_med, residuals, data.rv_err .^ 2,
                                   data.t_rv, data.t_rv, data.rv_inst, :rv; data=data)
            catch; nothing end
            gp_at_data !== nothing && (act_at_data .+= gp_at_data)
        end
        has_activity = any(!iszero, act_at_data)

        if decompose
            # === Component decomposition ====================================
            # Each Keplerian, the activity model, and their Total as separate
            # dense curves over the RAW (γ-subtracted) RV; residual panel shows
            # data − Total. The slow activity model reads cleanly in the time
            # domain; a short-period Keplerian necessarily renders as a fast comb
            # here (its shape lives in the phase-fold) — drawn anyway as its own
            # component, per request.
            t_dense = collect(range(minimum(data.t_rv), maximum(data.t_rv);
                                    length = 4000))
            pidx = (:n_planets in chain_names) ? collect(planet_indices(theta_med)) :
                   collect(1:params.config.max_kplanet)
            kep_dense = [compute_rv_model_planet(theta_med, data, t_dense, k) for k in pidx]
            keep = [any(!iszero, kd) for kd in kep_dense]      # drop inactive/zero slots
            pidx = pidx[keep]; kep_dense = kep_dense[keep]

            # Activity model on the dense grid (GP/AGP latent; AD has no t-only
            # curve → nothing, and we fall back to data-time activity only).
            act_dense = show_activity ? (try
                channel_gp_mean_at(theta_med, residuals, data.rv_err .^ 2,
                                   data.t_rv, t_dense, data.rv_inst, :rv; data=data)
            catch; nothing end) : nothing

            total_dense = reduce(.+, kep_dense; init = zeros(length(t_dense)))
            act_dense !== nothing && (total_dense = total_dense .+ act_dense)

            # Data (γ-subtracted) + residuals = data − Total at the data times.
            for (i, ins) in enumerate(inst_names)
                mask = data.rv_inst .== i
                any(mask) || continue
                t_i   = data.t_rv[mask] .- tmin
                raw_i = data.rv[mask] .- rv_gamma(theta_med, i)
                err_i = data.rv_err[mask]
                kep_at_i = zeros(count(mask))
                for k in pidx
                    kep_at_i .+= compute_rv_model_planet(theta_med, data, data.t_rv[mask], k)
                end
                res_i = raw_i .- kep_at_i .- act_at_data[mask]
                mc = inst_color(i); mk = inst_marker(i)
                errorbars!(ax, t_i, raw_i, err_i; color=mc, linewidth=ERRBAR_LW)
                scatter!(ax, t_i, raw_i; color=mc, marker=mk, markersize=12,
                         strokewidth=2.0, strokecolor=:black, label=ins)
                errorbars!(ax_r, t_i, res_i, err_i; color=mc, linewidth=ERRBAR_LW)
                scatter!(ax_r, t_i, res_i; color=mc, marker=mk, markersize=12,
                         strokewidth=2.0, strokecolor=:black)
            end

            # Component curves: each planet (gray when single, else along :cool),
            # the activity model (GP color), and the Total (bold black).
            np_draw = length(kep_dense)
            if show_keplerian
                for (j, k) in enumerate(pidx)
                    col = np_draw <= 1 ? NEREUS_COLORS.kep :
                          cgrad(NEREUS_CMAP)[(j - 0.5) / np_draw]
                    lines!(ax, t_dense .- tmin, kep_dense[j];
                           color=col, linewidth=MODEL_LW,
                           label="planet $(('a' + k))")    # star='a', first planet='b'
                end
            end
            (act_dense !== nothing && show_activity) &&
                lines!(ax, t_dense .- tmin, act_dense;
                       color=NEREUS_COLORS.gp, linewidth=MODEL_LW, label="activity")
            ncomp = np_draw + (act_dense !== nothing ? 1 : 0)
            (show_total && ncomp >= 2) &&
                lines!(ax, t_dense .- tmin, total_dense;
                       color=NEREUS_COLORS.model, linewidth=MODEL_LW + 0.6, label="total")
        else
            # === Legacy decorrelation view (raw vs activity-decorrelated) =====
            raw_labeled = false
            for (i, ins) in enumerate(inst_names)
                mask = data.rv_inst .== i
                any(mask) || continue
                t_i   = data.t_rv[mask] .- tmin
                rv_i  = data.rv[mask] .- rv_gamma(theta_med, i)     # raw (γ-subtracted)
                dec_i = rv_i .- act_at_data[mask]                   # activity-decorrelated
                err_i = data.rv_err[mask]
                res_i = residuals[mask]
                mc = inst_color(i); mk = inst_marker(i)

                if has_activity
                    scatter!(ax, t_i, rv_i; color=(:gray, 0.30), marker=mk,
                             markersize=10, strokewidth=0,
                             label = raw_labeled ? nothing : "raw RV")
                    raw_labeled = true
                end
                errorbars!(ax, t_i, dec_i, err_i; color=mc, linewidth=ERRBAR_LW)
                scatter!(ax, t_i, dec_i; color=mc, marker=mk, markersize=12,
                         strokewidth=2.0, strokecolor=:black,
                         label = has_activity ? "$ins (decorr.)" : ins)

                errorbars!(ax_r, t_i, res_i, err_i; color=mc, linewidth=ERRBAR_LW)
                scatter!(ax_r, t_i, res_i; color=mc, marker=mk, markersize=12,
                         strokewidth=2.0, strokecolor=:black)
            end
            if show_keplerian
                t_dense = collect(range(minimum(data.t_rv), maximum(data.t_rv);
                                        length = 4000))
                kep_dense = zeros(length(t_dense))
                for k in 1:params.config.max_kplanet
                    kep_dense .+= compute_rv_model_planet(theta_med, data, t_dense, k)
                end
                lines!(ax, t_dense .- tmin, kep_dense;
                       color = NEREUS_COLORS.kep, linewidth = MODEL_LW, label = "Keplerian")
            end
        end

        hlines!(ax_r, 0; color=NEREUS_COLORS.zero_line,
                 linestyle=:dash, linewidth=MODEL_LW)

        ax_r.xlabel = "Time (BJD - $(round(Int, tmin)))"
        # Legend below the panels — never overlaps the data, regardless
        # of orbit phase coverage or scatter range.
        Legend(ga[3, 1], ax;
                framevisible=false, labelsize=16,
                orientation=:horizontal,
                tellheight=true, tellwidth=false,
                nbanks=2)

        # Tight y-axes (percentile-based) — keeps single instrument
        # outliers (e.g. FEROS at the right edge in HD 18599) from
        # blowing up the data panel.
        rv_centered = vcat([data.rv[data.rv_inst .== i] .- rv_gamma(theta_med, i)
                             for i in 1:length(inst_names)]...)
        _tight_ylims!(ax,   rv_centered; pad_frac=0.10)
        _tight_ylims!(ax_r, residuals;   pad_frac=0.30, symmetric=true)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "rv_timeseries.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


"""
    plot_rv_components(chains, params, data; output=nothing, fmt=:png, ...)

RV **component decomposition** time-series: the raw (γ-subtracted) RV with each
Keplerian, the activity model, and their Total drawn as separate dense curves,
and a residual panel showing `data − Total`. Distinct from [`plot_rv_timeseries`]
(@ref) (the activity-decorrelated view) — this one keeps the raw data and shows
the model pieces. The activity curve is the GP/AGP latent in identified m/s on
the real data (an ActivityDecorrelation model contributes no t-only curve).
Short-period Keplerians read as a fast comb here by construction; their shape is
the phase-fold's job. Saved as `models/rv_components.<fmt>`.
"""
function plot_rv_components(chains, params, data;
                             output::Union{Nothing, String}=nothing,
                             fmt::Symbol=:png, save_pdf::Bool=false,
                             figsize=FIG_RV, bf_cutoff::Real=10.0,
                             show_keplerian::Bool=true, show_gp::Bool=true,
                             show_total::Bool=true, show_activity::Bool=true)
    fig = plot_rv_timeseries(chains, params, data;
                              output=nothing, fmt=fmt, save_pdf=save_pdf,
                              figsize=figsize, bf_cutoff=bf_cutoff,
                              show_keplerian=show_keplerian, show_gp=show_gp,
                              show_total=show_total, show_activity=show_activity,
                              decompose=true)
    if output !== nothing
        mkpath(joinpath(output, "models"))
        _save_plot(joinpath(output, "models", "rv_components.$fmt"), fig;
                    save_pdf=save_pdf, px_per_unit=3)
    end
    return fig
end


# SB2 double-lined component colors: the two ends of the Nereus :cool
# colormap — primary A = cyan, secondary B = magenta.
const SB2_PRI_COLOR = cgrad(NEREUS_CMAP)[0.0]
const SB2_SEC_COLOR = cgrad(NEREUS_CMAP)[1.0]

# True when the dataset carries any secondary (component-2) RV — i.e. an SB2 fit.
_is_sb2_data(data) = any(==(2), data.rv_comp)

# Winning-model, best-lp reference theta (shared selection logic).
function _sb2_plot_theta(chains, params, bf_cutoff)
    chain_names = Set(names(chains, :parameters))
    if :n_planets in chain_names
        np_all = vec(Array(chains[:n_planets])); modal_np = _mode_int_local(np_all)
        active_idx = findall(np_all .== modal_np)
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        td_state = winning_td_state(chains, params, active_idx, modal_np)
        active_idx = winning_config_idx(chains, params, active_idx, td_state)
        theta = Theta{Float64}(params; td=td_state)
    else
        active_idx = collect(1:_n_flat_draws(chains)); theta = Theta{Float64}(params)
    end
    top = _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff)
    active_idx = intersect(active_idx, top)
    isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
    set_theta_best_lp!(theta, chains, params, active_idx)
    return theta
end

"""
    plot_rv_sb2_timeseries(chains, params, data; output=nothing, …)

RV time series for an SB2 double-lined fit: primary (component 1) and
secondary (component 2) data in distinct colors, with BOTH the primary
(γ + K_A·binary + planet) and secondary (γ − K_B·binary) total model
curves overlaid on one panel. Residuals (data − component-gated
prediction) below. Saved as `models/rv_sb2_timeseries.<fmt>`.
"""
function plot_rv_sb2_timeseries(chains, params, data; output=nothing, fmt::Symbol=:png,
                                 save_pdf::Bool=false, figsize=FIG_RV, bf_cutoff::Real=10.0)
    with_theme(nereus_theme()) do
        fig = Figure(; size=figsize); ga = fig[1, 1] = GridLayout()
        ax   = Axis(ga[1, 1]; ylabel=rich("RV (m s", superscript("-1"), ")"))
        ax_r = Axis(ga[2, 1]; ylabel="Residuals", yticks = LinearTicks(3),
                        ytickformat = vs -> [string(round(Int, v)) for v in vs])
        rowsize!(ga, 1, Auto(3)); rowsize!(ga, 2, Auto(1))
        linkxaxes!(ax, ax_r); hidexdecorations!(ax; grid=false)
        # glue the residual panel to the main one: they share the x axis
        rowgap!(ga, 0)
        tmin = minimum(data.t_rv)

        theta_med = _sb2_plot_theta(chains, params, bf_cutoff)
        preds, _  = rv_predictions(theta_med, data)          # component-gated
        residuals = data.rv .- preds
        # per-point γ subtraction (barycentric γ shared across A/B; per-instrument
        # if multiple spectrographs) so both components centre on 0.
        g_at = [rv_gamma(theta_med, ins) for ins in data.rv_inst]

        # two total model curves (γ-free Keplerian + trend), primary and secondary
        t_dense = collect(range(minimum(data.t_rv), maximum(data.t_rv); length=4000))
        curve_pri = compute_rv_model_on_grid(theta_med, data, t_dense; include_trend=true, comp=1)
        curve_sec = compute_rv_model_on_grid(theta_med, data, t_dense; include_trend=true, comp=2)
        lines!(ax, t_dense .- tmin, curve_pri; color=SB2_PRI_COLOR, linewidth=MODEL_LW+0.4,
               label="primary (K_A + planet)")
        lines!(ax, t_dense .- tmin, curve_sec; color=SB2_SEC_COLOR, linewidth=MODEL_LW+0.4,
               label="secondary (−K_B)")

        for (cval, ccol, clab) in ((1, SB2_PRI_COLOR, "primary (A)"),
                                    (2, SB2_SEC_COLOR, "secondary (B)"))
            mask = data.rv_comp .== cval; any(mask) || continue
            t_i = data.t_rv[mask] .- tmin; raw_i = data.rv[mask] .- g_at[mask]
            err_i = data.rv_err[mask]; res_i = residuals[mask]
            errorbars!(ax, t_i, raw_i, err_i; color=ccol, linewidth=ERRBAR_LW)
            scatter!(ax, t_i, raw_i; color=ccol, markersize=11, strokewidth=1.5,
                     strokecolor=:black, label=clab)
            errorbars!(ax_r, t_i, res_i, err_i; color=ccol, linewidth=ERRBAR_LW)
            scatter!(ax_r, t_i, res_i; color=ccol, markersize=11, strokewidth=1.5, strokecolor=:black)
        end
        hlines!(ax_r, 0; color=NEREUS_COLORS.zero_line, linestyle=:dash, linewidth=MODEL_LW)
        ax_r.xlabel = "Time (BJD - $(round(Int, tmin)))"
        Legend(ga[3, 1], ax; framevisible=false, labelsize=16, orientation=:horizontal,
               tellheight=true, tellwidth=false, nbanks=2)
        _tight_ylims!(ax,   data.rv .- g_at; pad_frac=0.10)
        _tight_ylims!(ax_r, residuals;       pad_frac=0.30, symmetric=true)
        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "rv_sb2_timeseries.$fmt"), fig;
                       save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end

"""
    plot_rv_sb2_binary_fold(chains, params, data; binary_k=1, output=nothing, …)

Classic double-lined SB2 phase-fold on the binary period: primary
(component 1) points fold onto +K_A and secondary (component 2) onto
−K_B (anti-phase), two model curves crossing at the systemic velocity.
The circumprimary planet is subtracted from the primary. Saved as
`models/rv_sb2_binary_fold_P<binary_k>.<fmt>`.
"""
function plot_rv_sb2_binary_fold(chains, params, data; binary_k::Int=1, output=nothing,
                                  fmt::Symbol=:png, save_pdf::Bool=false, figsize=FIG_RV,
                                  bf_cutoff::Real=10.0)
    with_theme(nereus_theme()) do
        fig = Figure(; size=figsize); ga = fig[1, 1] = GridLayout()
        ax   = Axis(ga[1, 1]; ylabel=rich("RV (m s", superscript("-1"), ")"))
        ax_r = Axis(ga[2, 1]; xlabel="Phase", ylabel="Residuals")
        rowsize!(ga, 1, Auto(3)); rowsize!(ga, 2, Auto(1))
        linkxaxes!(ax, ax_r); hidexdecorations!(ax; grid=false)
        # glue the residual panel to the main one: they share the x axis
        rowgap!(ga, 0)

        theta_med = _sb2_plot_theta(chains, params, bf_cutoff)
        P  = planet_P(theta_med, binary_k)
        t0 = _planet_Tc(theta_med, data, binary_k)
        preds_all, _ = rv_predictions(theta_med, data)
        # binary's own contribution to each point (component-aware): +K_A on comp1,
        # −K_B on comp2. rv_folded = data − (everything except the binary itself).
        bin_pri = compute_rv_model_planet(theta_med, data, data.t_rv, binary_k; comp=1)
        bin_sec = compute_rv_model_planet(theta_med, data, data.t_rv, binary_k; comp=2)
        bin_at  = [data.rv_comp[i] == 1 ? bin_pri[i] : bin_sec[i] for i in eachindex(data.t_rv)]
        rv_folded = data.rv .- (preds_all .- bin_at)
        residuals = data.rv .- preds_all
        phases    = phase_fold(data.t_rv, P, t0)

        ph_fine = collect(range(-0.5, 0.5; length=1000)); t_fine = t0 .+ ph_fine .* P
        curve_pri = compute_rv_model_planet(theta_med, data, t_fine, binary_k; comp=1)
        curve_sec = compute_rv_model_planet(theta_med, data, t_fine, binary_k; comp=2)
        lines!(ax, ph_fine, curve_pri; color=SB2_PRI_COLOR, linewidth=MODEL_LW+0.4, label="primary (+K_A)")
        lines!(ax, ph_fine, curve_sec; color=SB2_SEC_COLOR, linewidth=MODEL_LW+0.4, label="secondary (−K_B)")
        hlines!(ax, 0; color=(:gray, 0.5), linestyle=:dot, linewidth=1)

        for (cval, ccol, clab) in ((1, SB2_PRI_COLOR, "primary (A)"),
                                    (2, SB2_SEC_COLOR, "secondary (B)"))
            mask = data.rv_comp .== cval; any(mask) || continue
            errorbars!(ax, phases[mask], rv_folded[mask], data.rv_err[mask]; color=ccol, linewidth=ERRBAR_LW)
            scatter!(ax, phases[mask], rv_folded[mask]; color=ccol, markersize=11,
                     strokewidth=1.5, strokecolor=:black, label=clab)
            errorbars!(ax_r, phases[mask], residuals[mask], data.rv_err[mask]; color=ccol, linewidth=ERRBAR_LW)
            scatter!(ax_r, phases[mask], residuals[mask]; color=ccol, markersize=11,
                     strokewidth=1.5, strokecolor=:black)
        end
        hlines!(ax_r, 0; color=NEREUS_COLORS.zero_line, linestyle=:dash, linewidth=MODEL_LW)
        Legend(ga[3, 1], ax; framevisible=false, labelsize=16, orientation=:horizontal,
               tellheight=true, tellwidth=false, nbanks=2)
        _tight_ylims!(ax,   rv_folded; pad_frac=0.10)
        _tight_ylims!(ax_r, residuals; pad_frac=0.30, symmetric=true)
        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "rv_sb2_binary_fold_P$binary_k.$fmt"), fig;
                       save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end

"""
    plot_rv_phasefold(chains, params, data;
                       planet=1, output=nothing, fmt=:png, figsize=FIG_RV,
                       n_draws=10000)

RV data phase-folded on one planet's period, other planets subtracted.
Points colored by observation time (cool colormap) with colorbar.
Matches EMPEROR's paint_fold_rv: 10K model points, 10K CI draws.

For an SB2 fit: folding the binary block delegates to the double-lined
`plot_rv_sb2_binary_fold`; folding a circumprimary planet uses ONLY the
primary (component-1) points (the planet is invisible in the secondary).
"""
function plot_rv_phasefold(chains, params, data;
                            planet::Int=1,
                            output::Union{Nothing, String}=nothing,
                            fmt::Symbol=:png,
                            save_pdf::Bool=false,
                            figsize=FIG_RV,
                            n_draws::Int=60000,
                            bf_cutoff::Real=10.0,
                            credmass::Real=0.85,
                            subtract_gp::Bool=true,
                            robust_ylim::Bool=true)
    # SB2 binary block → the dedicated double-lined fold.
    if planet <= length(params.layout.planet_blocks) &&
       params.layout.planet_blocks[planet] isa SB2Block
        return plot_rv_sb2_binary_fold(chains, params, data; binary_k=planet,
                                        output=output, fmt=fmt, save_pdf=save_pdf,
                                        figsize=figsize, bf_cutoff=bf_cutoff)
    end
    with_theme(nereus_theme()) do
        fig = Figure(; size=figsize)
        ga = fig[1, 1] = GridLayout()

        ax = Axis(ga[1, 1]; ylabel=rich("RV (m s", superscript("-1"), ")"))
        ax_r = Axis(ga[2, 1]; xlabel="Phase",
                     ylabel="Residuals", yticks = LinearTicks(3),
                        ytickformat = vs -> [string(round(Int, v)) for v in vs])
        rowsize!(ga, 1, Auto(3))
        rowsize!(ga, 2, Auto(1))
        linkxaxes!(ax, ax_r)
        hidexdecorations!(ax; grid=false)
        # glue the residual panel to the main one: they share the x axis
        rowgap!(ga, 0)

        # For trans-dim chains, condition on the **winning** model: only
        # samples with N_p == modal_np and the modal noise pattern.
        # Plotting K_k for k > modal_np is meaningless because the
        # winning model doesn't include that planet, so we skip and
        # warn.
        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            modal_np = _mode_int_local(np_all)
            if planet > modal_np
                @warn "Planet K=$planet not in winning model " *
                      "(modal_np=$modal_np); skipping phase-fold"
                return fig
            end
            active_idx = findall(np_all .== modal_np)
            if isempty(active_idx)
                @warn "No samples with N_p == $modal_np, skipping"
                return fig
            end
            td_state = winning_td_state(chains, params, active_idx, modal_np)
            # Condition the plotted draw on the WINNING noise configuration
            # (not the global max-lp overfitter). See winning_config_idx.
            active_idx = winning_config_idx(chains, params, active_idx, td_state)
            theta_med = Theta{Float64}(params; td=td_state)
        else
            active_idx = collect(1:_n_flat_draws(chains))
            theta_med = Theta{Float64}(params)
        end
        # Band draw pool = the JOINT credible region of this planet's orbital
        # params (NOT an lp cutoff): keep the central `credmass` of draws in
        # (P,K,sesinw,secosw), which excludes the prior-driven e→1 tail whose
        # periastron spikes otherwise wash the median and fray the band. The
        # old lp-cutoff (bf_cutoff) is a fixed nat-window — on a high-d chain it
        # either keeps the MAP sliver (fake-tight) or, opened up, the whole
        # degenerate tail (useless). See `_credible_region_pool`.
        top_idx = _credible_region_pool(chains, params, planet; credmass=credmass)
        active_idx = intersect(collect(active_idx), top_idx)
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))

        # Max-lp reference theta: the marginal-median theta is jointly
        # inconsistent and mis-subtracts the OTHER planets (b at K=141
        # leaking ±several m/s artifacts into small-planet folds).
        set_theta_best_lp!(theta_med, chains, params, active_idx)

        P = planet_P(theta_med, planet)
        # Fold around the planet's own inferior-CONJUNCTION epoch (the
        # constrained phase λ = Mo + ω), NOT periastron and NOT data.t_ref.
        # Periastron (ω) is ill-determined at low/moderate e, so folding there
        # decoheres the per-draw band; conjunction is fixed by λ. Data and band
        # MUST share this reference to overlay.
        t0 = _planet_Tc(theta_med, data, planet)

        # Subtract other planets + gamma + trend
        preds_all, _ = rv_predictions(theta_med, data)
        preds_planet = compute_rv_model_planet(theta_med, data, data.t_rv, planet)
        rv_folded = data.rv .- (preds_all .- preds_planet)
        phases_data = phase_fold(data.t_rv, P, t0)
        residuals = data.rv .- preds_all

        # Phase-fold = the pure ORBIT. On an RM-enabled fit, drop the in-transit
        # RV points: the Rossiter-McLaughlin anomaly is a sub-hour feature that is
        # invisible on a full-period fold and poisons the phase bins, so it lives
        # in the dedicated rm_anomaly plot instead. In-transit window = where the
        # RM model is non-negligible around the folded planet's conjunction t0.
        keep = trues(length(data.t_rv))
        # SB2: a circumprimary planet is invisible in the secondary's lines —
        # fold only the primary (component-1) points.
        _is_sb2_data(data) && (keep .&= data.rv_comp .== 1)
        if any(has_any_rm, params.config.planet_modes)
            wide = collect(range(t0 - P / 6, t0 + P / 6; length = 1000))
            rmw = _rm_curve_at(theta_med, data, wide)
            amax = maximum(abs, rmw)
            if amax > 0
                half = 1.2 * maximum((abs(wide[i] - t0) for i in eachindex(wide)
                                      if abs(rmw[i]) > 0.02 * amax); init = 0.0)
                @. keep &= abs(mod(data.t_rv - t0 + P / 2, P) - P / 2) >= half
            end
        end

        # GP cleaning: when a global GP is active on :rv, subtract its
        # mean at each data time so the phasefold shows the pure-orbit
        # signal (EMPEROR convention). Disable via `subtract_gp=false`.
        # Per-instrument restricted GPs are not yet handled.
        # `channel_gp_mean_at` dispatches on the active noise model:
        # celerite kernels use the passed residuals; the Rajpaul
        # ActivityGP (whose activity lives in the COVARIANCE, so a
        # mean-only subtraction would leave it smeared over the fold)
        # is routed through its multi-channel predictive mean via the
        # `data=` kwarg — same conventions as the likelihood.
        if subtract_gp
            gp_at_data = try
                channel_gp_mean_at(theta_med, residuals,
                                    data.rv_err .^ 2,
                                    data.t_rv, data.t_rv,
                                    data.rv_inst, :rv; data = data)
            catch err
                @warn "GP cleaning failed in phasefold; plotting raw residuals" exception=err
                nothing
            end
            if gp_at_data !== nothing
                rv_folded = rv_folded .- gp_at_data
                residuals = residuals .- gp_at_data
            end
        end

        # CI bands on 10K phase grid (using active samples only)
        # 1000 phase points is visually smooth for a fold curve; the budget
        # goes into `n_draws` instead, so the empirical percentile band is
        # well-sampled (smooth) rather than MC-hairy at the 3σ edge.
        ph_fine = collect(range(-0.5, 0.5; length=1000))
        t_fine = t0 .+ ph_fine .* P
        # Build sub-chain for CI computation
        if active_idx isa UnitRange
            ci_chains = chains
        else
            param_syms = names(chains, :parameters)
            mat = Matrix{Float64}(undef, length(active_idx), length(param_syms))
            for (j, sym) in enumerate(param_syms)
                mat[:, j] = vec(Array(chains[sym]))[active_idx]
            end
            ci_chains = MCMCChains.Chains(mat, param_syms)
        end
        # Fold each draw at its OWN period (phase-aligned) rather than on the
        # reference-period time grid t_fine — otherwise a broad period
        # posterior decoheres the per-draw curves into a kinky median + frayed
        # band (see compute_ci_bands).
        # ci_chains is ALREADY restricted to the credible pool above, so the
        # band must use all of it — disable compute_ci_bands' own lp cut.
        ci = compute_ci_bands(ci_chains, params, data, t_fine;
                               planet=planet, n_draws=n_draws,
                               bf_cutoff=Inf,
                               fold_phase=ph_fine, fold_t0=t0)

        # Nested 1σ/2σ/3σ predictive bands (lightest→darkest), concentric
        # with the model line below (which is the pointwise median = the
        # band's own centre). The periastron-referenced fold (compute_ci_bands)
        # keeps the per-draw curves phase-aligned, so the quantiles are smooth
        # rather than the eccentric-spike noise they were on a fixed-period grid.
        band!(ax, ph_fine, ci.lo3, ci.hi3; color=(NEREUS_COLORS.ci, CI_ALPHA_3SIGMA))
        band!(ax, ph_fine, ci.lo2, ci.hi2; color=(NEREUS_COLORS.ci, CI_ALPHA_2SIGMA))
        band!(ax, ph_fine, ci.lo1, ci.hi1; color=(NEREUS_COLORS.ci, CI_ALPHA_1SIGMA))

        # Time coloring
        tmin = minimum(data.t_rv)
        tmax = maximum(data.t_rv)

        # Plot per-instrument data (reduced alpha for dense datasets)
        inst_names = params.config.instruments.rv_names
        n_total = length(data.t_rv)
        data_alpha = n_total > 1000 ? 0.4 : n_total > 500 ? 0.6 : 0.9

        for (i, ins) in enumerate(inst_names)
            mask = (data.rv_inst .== i) .& keep

            errorbars!(ax, phases_data[mask], rv_folded[mask],
                        data.rv_err[mask];
                        color=(:gray, 0.15), linewidth=1.0)
            scatter!(ax, phases_data[mask], rv_folded[mask];
                      color=data.t_rv[mask] .- tmin,
                      colormap=NEREUS_CMAP,
                      colorrange=(0, tmax - tmin),
                      marker=inst_marker(i), markersize=13,
                      strokewidth=1.5, strokecolor=:black,
                      alpha=data_alpha, label=ins)

            errorbars!(ax_r, phases_data[mask], residuals[mask],
                        data.rv_err[mask];
                        color=(:gray, 0.15), linewidth=1.0)
            scatter!(ax_r, phases_data[mask], residuals[mask];
                      color=data.t_rv[mask] .- tmin,
                      colormap=NEREUS_CMAP,
                      colorrange=(0, tmax - tmin),
                      marker=inst_marker(i), markersize=13,
                      strokewidth=1.5, strokecolor=:black,
                      alpha=data_alpha)
        end

        # Phase-binned overlay — occupancy-scaled MEDIAN bins
        # (`bin_phasefold`, the same robust binner as the transit
        # folds): median bins shrug off outliers that drag a weighted
        # mean, which matters exactly when the fold is hard to read.
        xb = Float64[]; yb = Float64[]; eb = Float64[]
        if count(keep) > 50
            xb, yb, eb = bin_phasefold(phases_data[keep], rv_folded[keep];
                                        target = 12, xlo = -0.5, xhi = 0.5,
                                        min_count = 4)
            errorbars!(ax, xb, yb, eb;
                        color=NEREUS_COLORS.rv_bin, linewidth=2.5)
            scatter!(ax, xb, yb;
                      color=NEREUS_COLORS.rv_bin, markersize=14,
                      strokewidth=2, strokecolor=:black)

            xb_r, yb_r, eb_r = bin_phasefold(phases_data[keep], residuals[keep];
                                              target = 12, xlo = -0.5,
                                              xhi = 0.5, min_count = 4)
            errorbars!(ax_r, xb_r, yb_r, eb_r;
                        color=NEREUS_COLORS.rv_bin, linewidth=2.5)
            scatter!(ax_r, xb_r, yb_r;
                      color=NEREUS_COLORS.rv_bin, markersize=14,
                      strokewidth=2, strokecolor=:black)
        end

        # Model line = the pointwise predictive median (the band's own centre),
        # so the line and the 1/2/3σ bands are concentric and hug. The
        # periastron-referenced fold keeps draws phase-aligned, so this median
        # is smooth (no kinks); the best-lp orbit can be (e,ω)-atypical and sit
        # phase-shifted off the band centre, which read as a "weird" band.
        lines!(ax, ph_fine, ci.median;
                color=:black, linewidth=3)

        # Colorbar
        Colorbar(ga[1:2, 2]; colormap=NEREUS_CMAP,
                  limits=(0, tmax - tmin),
                  label="BJD - $(round(Int, tmin))",
                  labelsize=20, ticklabelsize=16)

        hlines!(ax_r, 0; color=NEREUS_COLORS.zero_line,
                 linestyle=:dash, linewidth=MODEL_LW)

        # Tight y-axes via percentile clipping — robust to single
        # outliers, matches the photometry convention.
        # Robust y-zoom: frame the SIGNAL (model median + binned points
        # + central data quantiles), NOT the CI bands (a GP band can
        # spike to ±100s at poorly constrained phases) and NOT the data
        # extremes — outliers land out of view instead of stretching the
        # axis until the orbit is unreadable (`robust_ylim=false` to
        # disable). This is the LAST limits-setting call on these axes —
        # an earlier version set limits mid-function and was silently
        # overwritten by a second mechanism down here.
        if robust_ylim
            _fmin(v) = (f = filter(isfinite, v); isempty(f) ? Inf  : minimum(f))
            _fmax(v) = (f = filter(isfinite, v); isempty(f) ? -Inf : maximum(f))
            fold_fin = filter(isfinite, rv_folded[keep])
            isempty(fold_fin) && (fold_fin = [0.0])
            zlo = min(quantile(fold_fin, 0.10), _fmin(ci.median))
            zhi = max(quantile(fold_fin, 0.90), _fmax(ci.median))
            if !isempty(yb)
                zlo = min(zlo, _fmin(yb .- eb))
                zhi = max(zhi, _fmax(yb .+ eb))
            end
            if isfinite(zlo) && isfinite(zhi) && zhi > zlo
                zpad = 0.15 * (zhi - zlo)
                ylims!(ax, zlo - zpad, zhi + zpad)
            end
            res_fin = filter(isfinite, residuals)
            if !isempty(res_fin)
                rmax = max(abs(quantile(res_fin, 0.10)),
                            abs(quantile(res_fin, 0.90)))
                rmax > 0 && ylims!(ax_r, -1.3 * rmax, 1.3 * rmax)
            end
        else
            _tight_ylims!(ax,   rv_folded; pad_frac=0.10)
            _tight_ylims!(ax_r, residuals; pad_frac=0.30, symmetric=true)
        end

        ph_lo, ph_hi = extrema(phases_data)
        ph_pad = 0.02 * (ph_hi - ph_lo)
        if isfinite(ph_lo) && isfinite(ph_hi) && ph_pad > 0
            xlims!(ax, ph_lo - ph_pad, ph_hi + ph_pad)
            xlims!(ax_r, ph_lo - ph_pad, ph_hi + ph_pad)
        end

        # In-axis legend; corner chosen by data density so the box
        # parks over the emptiest quadrant of the phase-fold. The
        # semi-transparent white background still keeps any stray
        # marker readable through it.
        corner = _best_legend_corner(phases_data, rv_folded)
        n_inst = length(inst_names)
        axislegend(ax;
                    position = corner,
                    framevisible = true,
                    framecolor = (:black, 0.3),
                    backgroundcolor = (:white, 0.85),
                    labelsize = 12,
                    patchsize = (18, 12),
                    margin = (10, 10, 10, 10),
                    nbanks = n_inst > 4 ? 2 : 1)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "RV_phasefold_P$planet.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end
