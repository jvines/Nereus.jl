# Photometry/transit plots: timeseries and phase-fold.

"""
    plot_pm_timeseries(chains, params, data;
                        output=nothing, fmt=:png, figsize=FIG_PM)

Transit photometry over time with best-fit transit model and
data-minus-model residuals. Per-instrument panels.

The model is the median-θ transit evaluated on a fine time grid
(overlay) and at each cadence (for residuals). Y-axes are scaled
to the actual data range, not auto-scaled, so transits don't get
swallowed by white space.
"""
function plot_pm_timeseries(chains, params, data;
                             output::Union{Nothing, String}=nothing,
                             fmt::Symbol=:png,
                             save_pdf::Bool=false,
                             figsize=FIG_PM)
    n_phot = length(data.t_phot)
    n_phot > 0 || return nothing

    with_theme(nereus_theme()) do
        # Median θ conditioned on the winning model (N_p == modal_np
        # plus modal noise pattern). See rv_plots.jl for rationale.
        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            modal_np = _mode_int_local(np_all)
            active_idx = findall(np_all .== modal_np)
            isempty(active_idx) && (active_idx = 1:_n_flat_draws(chains))
            td_state = winning_td_state(chains, params, active_idx, modal_np)
            active_idx = winning_config_idx(chains, params, active_idx, td_state)
            theta_med = Theta{Float64}(params; td=td_state)
        else
            active_idx = 1:_n_flat_draws(chains)
            theta_med = Theta{Float64}(params)
        end
        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue
            vals = vec(Array(chains[sym]))[active_idx]
            set_param!(theta_med, name, median(vals))
        end

        inst_names = params.config.instruments.pm_names

        for (ins_idx, ins) in enumerate(inst_names)
            mask = data.phot_inst .== ins_idx
            count(mask) == 0 && continue

            t_i    = data.t_phot[mask]
            flux_i = data.flux[mask]
            err_i  = data.flux_err[mask]

            # Real transit model at every cadence (for residuals)
            model_at_data = compute_transit_model_on_grid(theta_med, data,
                                                           t_i, ins_idx)
            res_ppm = (flux_i .- model_at_data) .* 1e6

            fig = Figure(; size = figsize)
            ga  = fig[1, 1] = GridLayout()

            ax   = Axis(ga[1, 1]; ylabel="Relative Flux", ylabelpadding = 6.0)
            ax_r = Axis(ga[2, 1]; xlabel="BJD", ylabel="O \u2212 C (ppm)", ylabelpadding = 4.0,
                        yticks = LinearTicks(3),
                        ytickformat = vs -> [string(round(Int, v)) for v in vs])
            text!(ax, 0.035, 0.92; text = ins, align = (:left, :top),
                   space = :relative,
                   fontsize = clamp(figsize[2] / 22, 12, 22))
            rowsize!(ga, 1, Relative(3/4))
            rowsize!(ga, 2, Relative(1/4))
            linkxaxes!(ax, ax_r)
            hidexdecorations!(ax; grid=false)
            # glue the residual panel to the main one: they share the x axis
            rowgap!(ga, 0)

            # Marker alpha: even on dense LCs we want the dips visible.
            n_pts = length(t_i)
            alpha = n_pts > 20000 ? 0.85 :
                    n_pts > 5000  ? 0.90 :
                    n_pts > 1000  ? 0.95 : 1.0

            errorbars!(ax, t_i, flux_i, err_i;
                        color=(NEREUS_COLORS.pm_marker, 0.6),
                        linewidth=1)
            scatter!(ax, t_i, flux_i;
                      color=(NEREUS_COLORS.pm_marker, alpha),
                      markersize=5, strokewidth=0)

            # Model overlay as a scatter at data cadences. With many
            # short transits across a long span, drawing model lines
            # collapses to vertical fence bars below pixel width.
            # Plotting model values as small markers at the data times
            # gives a visible "model trail" through the dips that
            # remains legible whatever the zoom.
            scatter!(ax, t_i, model_at_data;
                      color=NEREUS_COLORS.model, markersize=3,
                      strokewidth=0)

            scatter!(ax_r, t_i, res_ppm;
                      color=(NEREUS_COLORS.pm_marker, alpha),
                      markersize=5, strokewidth=0)
            hlines!(ax_r, 0; color=NEREUS_COLORS.zero_line,
                     linestyle=:dash, linewidth=MODEL_LW)

            # Tight y-axis: snug to actual data range with small pad
            _tight_ylims!(ax,   flux_i; pad_frac = 0.10)
            _tight_ylims!(ax_r, res_ppm; pad_frac = 0.30, symmetric=true)

            if output !== nothing
                mkpath(joinpath(output, "models"))
                _save_plot(joinpath(output, "models", "pm_timeseries_$ins.$fmt"), fig;
                            save_pdf=save_pdf)
            end
        end
    end
end


"""
    plot_pm_phasefold(chains, params, data;
                       planet=1, output=nothing, fmt=:png, figsize=FIG_PM,
                       n_bins=nothing,
                       phase_window=(-0.025, 0.025))

Transit data phase-folded on one planet's period, with median-θ
transit model overlaid and data-minus-model residuals.

`phase_window` defaults to ±0.025 — appropriate for short-period
hot-Jupiter / sub-Neptune transits where the dip lives in ≤ 5% of
the orbit. Pass a wider tuple for longer-duration grazing transits.
"""
function plot_pm_phasefold(chains, params, data;
                            planet::Int=1,
                            output::Union{Nothing, String}=nothing,
                            fmt::Symbol=:png,
                            save_pdf::Bool=false,
                            figsize=FIG_PM,
                            n_bins::Union{Nothing, Int}=nothing,
                            credmass::Real=0.85,
                            phase_window::Tuple{<:Real, <:Real}=(-0.025, 0.025))
    n_phot = length(data.t_phot)
    n_phot > 0 || return nothing

    with_theme(nereus_theme()) do
        chain_names = Set(names(chains, :parameters))

        # Active sample mask: condition on the winning N_p (modal). If
        # `planet` exceeds modal_np the requested transit is not part of
        # the winning model, so we skip.
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            modal_np = _mode_int_local(np_all)
            if planet > modal_np
                @warn "Planet K=$planet not in winning model " *
                      "(modal_np=$modal_np); skipping phase-fold"
                return nothing
            end
            active_idx = findall(np_all .== modal_np)
            isempty(active_idx) && (active_idx = 1:_n_flat_draws(chains))
            td_state = winning_td_state(chains, params, active_idx, modal_np)
            active_idx = winning_config_idx(chains, params, active_idx, td_state)
            theta_med = Theta{Float64}(params; td=td_state)
        else
            active_idx = 1:_n_flat_draws(chains)
            theta_med = Theta{Float64}(params)
        end

        # Band draw pool = JOINT credible region of this planet's params (the
        # Nereus default; same mechanism as the RV fold). Keeps the central
        # `credmass` of draws in (P, b, rr, sesinw, secosw, …), excluding the
        # degenerate tail that frays the band. See `_credible_region_pool`.
        cred_idx = _credible_region_pool(chains, params, planet; credmass=credmass)
        active_idx = intersect(collect(active_idx), cred_idx)
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))

        # Reference theta = the MAX-LP sample among the active draws, NOT
        # per-parameter marginal medians: the θ-median's jointly-inconsistent
        # (rho_s, b, rr, Tc) lies off the posterior ridge and its transit can
        # fall entirely OUTSIDE the (correct, predictive) CI bands — measured
        # on the WASP-47 trans-dim folds. Median is the no-:lp fallback.
        if :lp in chain_names
            lp_all = vec(Array(chains[:lp]))
            ibest = active_idx[argmax(@view lp_all[active_idx])]
            for name in params.layout.unfrozen_names
                sym = Symbol(name)
                sym in chain_names || continue
                set_param!(theta_med, name, vec(Array(chains[sym]))[ibest])
            end
        else
            for name in params.layout.unfrozen_names
                sym = Symbol(name)
                sym in chain_names || continue
                vals = vec(Array(chains[sym]))[active_idx]
                set_param!(theta_med, name, median(vals))
            end
        end

        P  = planet_P(theta_med, planet)
        ta = planet_time_anchor(theta_med, planet)
        parametrization = params.config.parametrization
        t0 = parametrization.time === :Tc ? ta :
             parametrization.time === :Tp ? ta : data.t_ref

        phases = phase_fold(data.t_phot, P, t0)

        inst_names = params.config.instruments.pm_names

        for (ins_idx, ins) in enumerate(inst_names)
            mask = data.phot_inst .== ins_idx
            count(mask) == 0 && continue

            ph_i   = phases[mask]
            flux_i = data.flux[mask]
            err_i  = data.flux_err[mask]
            t_i    = data.t_phot[mask]

            model_at_data = compute_transit_model_on_grid(theta_med, data,
                                                           t_i, ins_idx;
                                                           planet=planet)
            res_ppm = (flux_i .- model_at_data) .* 1e6

            fig = Figure(; size = figsize)
            ga  = fig[1, 1] = GridLayout()

            ax   = Axis(ga[1, 1]; ylabel="Relative Flux", ylabelpadding = 6.0)
            ax_r = Axis(ga[2, 1]; xlabel="Phase",
                                  ylabel="O \u2212 C (ppm)", ylabelpadding = 4.0,
                                  yticks = LinearTicks(3),
                        ytickformat = vs -> [string(round(Int, v)) for v in vs])
            # Name the band: one file is written per instrument, so without it
            # the panels are indistinguishable once they are in a paper.
            text!(ax, 0.035, 0.92; text = ins, align = (:left, :top),
                  space = :relative,
                  fontsize = clamp(figsize[2] / 22, 12, 22))
            rowsize!(ga, 1, Relative(3/4))
            rowsize!(ga, 2, Relative(1/4))
            linkxaxes!(ax, ax_r)
            hidexdecorations!(ax; grid=false)
            # glue the residual panel to the main one: they share the x axis
            rowgap!(ga, 0)

            ph_lo, ph_hi = phase_window
            xlims!(ax,   ph_lo, ph_hi)
            xlims!(ax_r, ph_lo, ph_hi)

            n_pts = count(mask)
            alpha = n_pts > 20000 ? 0.75 :
                    n_pts > 5000  ? 0.85 :
                    n_pts > 1000  ? 0.95 : 1.0

            errorbars!(ax, ph_i, flux_i, err_i;
                        color=(NEREUS_COLORS.pm_marker, 0.45),
                        linewidth=0.5)
            scatter!(ax, ph_i, flux_i;
                      color=(NEREUS_COLORS.pm_marker, alpha),
                      markersize=4, strokewidth=0)

            # CI bands on a fine phase grid using active samples only
            ph_fine = collect(range(ph_lo, ph_hi; length=2000))
            t_fine = t0 .+ ph_fine .* P
            ci_chains = if active_idx isa UnitRange
                chains
            else
                param_syms = names(chains, :parameters)
                mat = Matrix{Float64}(undef, length(active_idx),
                                              length(param_syms))
                for (j, sym) in enumerate(param_syms)
                    mat[:, j] = vec(Array(chains[sym]))[active_idx]
                end
                MCMCChains.Chains(mat, param_syms)
            end
            # ci_chains is already restricted to the credible pool → disable
            # compute_transit_ci_bands' own lp cut so it uses all of it.
            ci = compute_transit_ci_bands(ci_chains, params, data, t_fine,
                                             ins_idx; planet=planet,
                                             bf_cutoff=Inf,
                                             fold_phase=ph_fine,
                                             n_draws=min(2000,
                                                       length(active_idx)))

            band!(ax, ph_fine, ci.lo3, ci.hi3;
                   color=(NEREUS_COLORS.ci, CI_ALPHA_3SIGMA))
            band!(ax, ph_fine, ci.lo2, ci.hi2;
                   color=(NEREUS_COLORS.ci, CI_ALPHA_2SIGMA))
            band!(ax, ph_fine, ci.lo1, ci.hi1;
                   color=(NEREUS_COLORS.ci, CI_ALPHA_1SIGMA))

            # Model line = the bands' own 50% quantile (pointwise median of
            # the per-draw PHASE-ALIGNED curves) — inside the bands by
            # construction and smooth for aligned transits. A single max-lp
            # draw can be atypical in duration/depth (mild overfit of deep
            # scatter on shallow transits) and poke outside its own bands;
            # best-lp remains the residual-panel reference below.
            lines!(ax, ph_fine, ci.median;
                    color=NEREUS_COLORS.model, linewidth=MODEL_LW)

            # Binned overlay — robust occupancy-scaled median bins over the
            # window by default (`bin_phasefold` picks the bin count from the
            # in-window cadence count); an explicit `n_bins` overrides, but a
            # forced-fine grid drops below min_count and renders almost no
            # bins (measured: n_bins=40 over ~100 in-window cadences → 1 bin).
            xb, yb, eb = bin_phasefold(ph_i, flux_i;
                                        nbins=(n_bins isa Int && n_bins > 0 ?
                                               n_bins : nothing),
                                        xlo=ph_lo, xhi=ph_hi)
            if !isempty(xb)
                errorbars!(ax, xb, yb, eb;
                            color=NEREUS_COLORS.pm_bin,
                            linewidth=2.5)
                scatter!(ax, xb, yb;
                          color=NEREUS_COLORS.pm_bin, markersize=12,
                          strokewidth=1.5, strokecolor=:black)
            end

            scatter!(ax_r, ph_i, res_ppm;
                      color=(NEREUS_COLORS.pm_marker, alpha),
                      markersize=4, strokewidth=0)
            hlines!(ax_r, 0; color=NEREUS_COLORS.zero_line,
                     linestyle=:dash, linewidth=MODEL_LW)

            # Tight y-axes inside the phase window
            in_window = ph_lo .<= ph_i .<= ph_hi
            _tight_ylims!(ax,   flux_i[in_window]; pad_frac=0.10)
            _tight_ylims!(ax_r, res_ppm[in_window]; pad_frac=0.30,
                                                     symmetric=true)

            if output !== nothing
                mkpath(joinpath(output, "models"))
                _save_plot(joinpath(output, "models",
                            "Transit_phasefold_P$(planet)_$(ins).$fmt"), fig;
                            save_pdf=save_pdf)
            end
        end
    end
end

"""
    _tight_ylims!(ax, vals; pad_frac=0.10, symmetric=false,
                   q_lo=0.02, q_hi=0.98)

Tight y-limits using percentile-based clipping. We use 2-98% by
default rather than 0.5-99.5% so single-instrument outliers (FEROS at
the right edge of HD 18599 was the motivating case) don't blow up the
panel. `symmetric=true` produces ±m*(1+pad) limits — for residual
panels.
"""
function _tight_ylims!(ax, vals::AbstractVector{<:Real};
                         pad_frac::Float64 = 0.10,
                         symmetric::Bool = false,
                         q_lo::Float64 = 0.02,
                         q_hi::Float64 = 0.98)
    finite = filter(isfinite, vals)
    isempty(finite) && return
    lo = quantile(finite, q_lo)
    hi = quantile(finite, q_hi)
    if symmetric
        m = max(abs(lo), abs(hi))
        m > 0 || return
        ylims!(ax, -m * (1 + pad_frac), m * (1 + pad_frac))
    else
        span = hi - lo
        span > 0 || return
        ylims!(ax, lo - pad_frac * span, hi + pad_frac * span)
    end
end
