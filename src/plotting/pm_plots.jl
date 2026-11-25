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
            np_all = _draws(chains, :n_planets)
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
        # Max-lp draw of the winning model, with its active set -- NOT
        # per-parameter medians, whose (rho_s, b, rr, Tc) is off the posterior
        # ridge (see `set_theta_best_lp!` and the phase-fold below).
        set_theta_best_lp!(theta_med, chains, params, collect(active_idx))

        inst_names = params.config.instruments.pm_names

        for (ins_idx, ins) in enumerate(inst_names)
            mask = data.phot_inst .== ins_idx
            count(mask) == 0 && continue

            t_i    = data.t_phot[mask]
            flux_i = data.flux[mask]
            err_i  = data.flux_err[mask]

            # Real transit model at every cadence (for residuals), integrated
            # over each cadence's exposure as in the likelihood
            model_at_data = compute_transit_model_on_grid(theta_med, data,
                                                           t_i, ins_idx;
                                                           exposure = _cadence_exposures(data, mask))
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

            # Limits first, so the alpha below is judged on the final axis
            # box: the raw-cadence quantiles, widened to the model's dips
            # (`_fold_flux_ylims` with no bins), so a transit the cadences
            # under-sample is not clipped off the bottom.
            flims = _fold_flux_ylims(model_at_data, Float64[], Float64[], flux_i)
            flims === nothing || ylims!(ax, flims...)
            _tight_ylims!(ax_r, res_ppm; pad_frac = 0.30, symmetric=true)
            ylo, yhi = flims === nothing ? extrema(filter(isfinite, flux_i)) : flims
            alpha, alpha_bar = _overlap_alpha(ax, t_i, flux_i, err_i, extrema(t_i), (ylo, yhi);
                                          markersize = 5)
            _raw_points!(ax, t_i, flux_i, err_i, alpha, alpha_bar; markersize = 5)

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
transit model overlaid and data-minus-model residuals. The other planets'
transits are subtracted from the folded data, and every model is integrated
over the cadences' exposure times as in the likelihood.

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
    # A slot without a photometric mode (RV_ONLY, …) has no transit: folding
    # the light curves on its period drew a flat "transit model" over data
    # that happened to contain another planet's transits.
    if !has_pm(params.config.planet_modes[planet])
        @warn "Planet K=$planet has no transit (mode $(params.config.planet_modes[planet])); " *
              "skipping the transit fold"
        return nothing
    end

    with_theme(nereus_theme()) do
        chain_names = Set(names(chains, :parameters))

        # Active sample mask: condition on the winning model (modal N_p, its
        # modal live slots, modal noise). If slot `planet` is not live in it,
        # the transit is not part of the winning model and we skip (below).
        if :n_planets in chain_names
            np_all = _draws(chains, :n_planets)
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

        # Band draw pool = JOINT credible region of this planet's params (the
        # Nereus default; same mechanism as the RV fold). Keeps the central
        # `credmass` of draws in (P, b, rr, sesinw, secosw, …), excluding the
        # degenerate tail that frays the band. See `_credible_region_pool`.
        cred_idx = _credible_region_pool(chains, params, planet; credmass=credmass)
        let cut = _sorted_intersect(active_idx, cred_idx)   # empty: keep winners
            isempty(cut) || (active_idx = cut)
        end

        # Reference theta = the MAX-LP sample among the active draws, NOT
        # per-parameter marginal medians: the θ-median's jointly-inconsistent
        # (rho_s, b, rr, Tc) lies off the posterior ridge and its transit can
        # fall entirely OUTSIDE the (correct, predictive) CI bands — measured
        # on the WASP-47 trans-dim folds. Median is the no-:lp fallback.
        set_theta_best_lp!(theta_med, chains, params, collect(active_idx))
        # Is this SLOT live in the winning model? Not `planet ≤ modal_np`:
        # after a death mid-list the live slots are not 1:modal_np.
        if theta_med.td !== nothing && !(planet in planet_indices(theta_med))
            @warn "Planet K=$planet is not active in the winning model; " *
                  "skipping phase-fold"
            return nothing
        end

        P  = planet_P(theta_med, planet)
        ta = planet_time_anchor(theta_med, planet)
        parametrization = params.config.parametrization
        t0 = parametrization.time === :Tc   ? ta :
             parametrization.time === :phic ? phic_to_tc(ta, P, parametrization.t_ref_c) :
             parametrization.time === :Tp   ? ta : data.t_ref

        phases = phase_fold(data.t_phot, P, t0)

        inst_names = params.config.instruments.pm_names

        for (ins_idx, ins) in enumerate(inst_names)
            mask = data.phot_inst .== ins_idx
            count(mask) == 0 && continue

            ph_i   = phases[mask]
            flux_i = data.flux[mask]
            err_i  = data.flux_err[mask]
            t_i    = data.t_phot[mask]

            flux_i, model_at_data = _pm_fold_data(theta_med, data, mask, ins_idx, planet)
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

            # CI bands on a fine phase grid using active samples only
            ph_fine = collect(range(ph_lo, ph_hi; length=2000))
            t_fine = t0 .+ ph_fine .* P
            # The band samples the credible pool (`pool`), not
            # compute_transit_ci_bands' own lp cut. Passed as indices: copied out
            # as a sub-chain it cost every column at 85% of the draws, once per
            # instrument.
            # The curve is drawn over binned cadences: integrate it over the
            # instrument's exposure, or long-cadence ingress looks too sharp.
            ci = compute_transit_ci_bands(chains, params, data, t_fine,
                                             ins_idx; planet=planet, trend=false,
                                             exposure=_instrument_exposure(data, mask),
                                             pool=active_idx,
                                             fold_phase=ph_fine,
                                             n_draws=min(2000,
                                                       length(active_idx)))

            # Binned overlay — robust occupancy-scaled median bins over the
            # window by default (`bin_phasefold` picks the bin count from the
            # in-window cadence count); an explicit `n_bins` overrides, but a
            # forced-fine grid drops below min_count and renders almost no
            # bins (measured: n_bins=40 over ~100 in-window cadences → 1 bin).
            nb_kw = n_bins isa Int && n_bins > 0 ? n_bins : nothing
            xb, yb, eb = bin_phasefold(ph_i, flux_i; nbins = nb_kw,
                                        xlo = ph_lo, xhi = ph_hi)
            _, rb, erb = bin_phasefold(ph_i, res_ppm; nbins = nb_kw,
                                        xlo = ph_lo, xhi = ph_hi)

            # Limits BEFORE anything is drawn: the marker alpha below is set
            # from how the points crowd this exact axis box. Framed on the
            # model and the bins, not on the raw cadences (`_fold_flux_ylims`).
            in_window = ph_lo .<= ph_i .<= ph_hi
            flims = _fold_flux_ylims(ci.median, yb, eb, flux_i[in_window])
            flims === nothing || ylims!(ax, flims...)
            rlims = _fold_resid_ylims(rb, erb, res_ppm[in_window])
            rlims === nothing || ylims!(ax_r, rlims...)

            ylo, yhi = flims === nothing ? extrema(filter(isfinite, flux_i)) : flims
            alpha, alpha_bar = _overlap_alpha(ax, ph_i, flux_i, err_i, (ph_lo, ph_hi), (ylo, yhi);
                                          markersize = 4)
            _raw_points!(ax, ph_i, flux_i, err_i, alpha, alpha_bar; markersize = 4)

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

            if output !== nothing
                mkpath(joinpath(output, "models"))
                _save_plot(joinpath(output, "models",
                            "Transit_phasefold_K$(planet)_$(ins).$fmt"), fig;
                            save_pdf=save_pdf)
            end
        end
    end
end

"""
    _pm_fold_data(theta, data, mask, ins_idx, planet) -> (flux, model)

The cadences of instrument `ins_idx` (`mask`) as a phase fold of `planet` shows
them, and the model drawn through them. Models are integrated over each
cadence's exposure, as in the likelihood. The other planets' transits and the
instrument's baseline trend come out of the folded flux, so only this planet's
dip shows on a flat baseline at 1 + offset. The fold's model curve is evaluated
at the reference epoch, where a trend fitted to one night does not apply.
`flux - model` is the residual against the full model.
"""
function _pm_fold_data(theta, data, mask, ins_idx::Int, planet::Int)
    t_i = data.t_phot[mask]
    exp_i = _cadence_exposures(data, mask)
    model_full = compute_transit_model_on_grid(theta, data, t_i, ins_idx; exposure = exp_i)
    model = compute_transit_model_on_grid(theta, data, t_i, ins_idx; planet = planet,
                                          exposure = exp_i, trend = false)
    return data.flux[mask] .- model_full .+ model, model
end

"""
    _fold_flux_ylims(model, yb, eb, raw; pad_frac=0.10) -> (lo, hi) or nothing

Flux limits for a transit panel, framed on what carries the signal: the
model curve (its transit floor and its baseline) and the binned points
(`yb ± eb`). The raw cadences set the limits only when there are no bins,
and then the model's range is still included. Raw points may overflow the
axis; they do not set it.

Framed on the raw cadences' 2-98% quantiles, a light curve covering only
egress (LCO Sutherland on TIC-275647095) lost the model's floor off the
bottom of the panel, and NGTS's 10 s cadences (±2.5% scatter) made the axis
5% tall and flattened a 1% transit.
"""
function _fold_flux_ylims(model::AbstractVector{<:Real}, yb::AbstractVector{<:Real},
                          eb::AbstractVector{<:Real}, raw::AbstractVector{<:Real};
                          pad_frac::Float64 = 0.10, q_lo::Float64 = 0.02,
                          q_hi::Float64 = 0.98)
    fm = filter(isfinite, model)
    lo, hi = Inf, -Inf
    if !isempty(yb)
        lo = minimum(yb .- eb); hi = maximum(yb .+ eb)
    else
        fr = filter(isfinite, raw)
        isempty(fr) || ((lo, hi) = (quantile(fr, q_lo), quantile(fr, q_hi)))
    end
    if !isempty(fm)
        lo = min(lo, minimum(fm)); hi = max(hi, maximum(fm))
    end
    (isfinite(lo) && isfinite(hi) && hi > lo) || return nothing
    span = hi - lo
    return (lo - pad_frac * span, hi + pad_frac * span)
end

"""
    _fold_resid_ylims(rb, erb, raw; k=4.0, pad=1.15) -> (-m, m) or nothing

Symmetric residual limits. With binned residuals `rb ± erb`: `k` times their
rms about zero, widened to show every binned point and its bar. Without
bins: the raw residuals' 2-98% quantiles (as `_tight_ylims!`). Raw points
may overflow; on dense noisy cadences they otherwise set an axis on which
the residual structure the bins show is invisible.
"""
function _fold_resid_ylims(rb::AbstractVector{<:Real}, erb::AbstractVector{<:Real},
                           raw::AbstractVector{<:Real}; k::Float64 = 4.0,
                           pad::Float64 = 1.15, raw_pad::Float64 = 0.30)
    ok = isfinite.(rb) .& isfinite.(erb)
    if any(ok)
        r = rb[ok]; e = erb[ok]
        m = max(k * sqrt(sum(abs2, r) / length(r)), maximum(abs.(r) .+ e)) * pad
        m > 0 && return (-m, m)
    end
    fr = filter(isfinite, raw)
    isempty(fr) && return nothing
    m = max(abs(quantile(fr, 0.02)), abs(quantile(fr, 0.98)))
    m > 0 || return nothing
    return (-m * (1 + raw_pad), m * (1 + raw_pad))
end

"""
    _overlap_alpha(ax, x, y, err, xlims, ylims; markersize, c=1.5, α_min=0.03,
                   c_bar=3.0, α_bar_min=0.005, q=0.9) -> (alpha, alpha_bar)

Raw-cadence alphas from how much the points overlap ON SCREEN, not from how
many there are. The axis box (`xlims` × `ylims`, its size read from the axis
viewport) is cut into cells about one marker across. A crowd is the `q`
percentile of the counts in occupied cells -- a percentile, so one crowded
region or a sparse edge does not drive it.

- Markers: `alpha = clamp(c / d_marker, α_min, 1)`, `d_marker` counting each
  point in the cell of its marker -- about `c` opaque markers' worth of ink
  per cell.
- Error bars: `alpha_bar = clamp(c_bar / d_bar, α_bar_min, alpha / 2)`,
  `d_bar` counting each point in every cell its bar (`y ± err`) crosses, so
  the stack of bars in a cell carries about `c_bar` of ink.

The bars need their own crowd. On NGTS (10 s cadences, ±2.5% scatter, a ~2%
axis) the markers inside the axis barely overlap, but every bar runs the full
panel height: bars at half the marker alpha drew a solid block, and a single
alpha from the bars' crowd drove the markers to the floor. The point-count
rule before this never went below 0.75.
"""
function _overlap_alpha(ax, x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                        err::AbstractVector{<:Real}, xlims, ylims; markersize::Real,
                        c::Float64 = 1.5, α_min::Float64 = 0.03,
                        c_bar::Float64 = 3.0, α_bar_min::Float64 = 0.005,
                        q::Float64 = 0.9)
    w, h = Makie.widths(ax.scene.viewport[])
    (xlo, xhi), (ylo, yhi) = Float64.(xlims), Float64.(ylims)
    (w > 0 && h > 0 && xhi > xlo && yhi > ylo) || return (1.0, 0.5)
    nx = max(1, floor(Int, w / markersize)); ny = max(1, floor(Int, h / markersize))
    mk = zeros(Int, nx, ny); ink = zeros(Int, nx, ny)
    row(v) = clamp(1 + floor(Int, (v - ylo) / (yhi - ylo) * ny), 1, ny)
    @inbounds for i in eachindex(x, y, err)
        xi, yi, ei = x[i], y[i], err[i]
        (isfinite(xi) && isfinite(yi) && xlo <= xi <= xhi) || continue
        ei = isfinite(ei) ? abs(ei) : 0.0
        (yi + ei < ylo || yi - ei > yhi) && continue      # wholly off the axis
        col = min(nx, 1 + floor(Int, (xi - xlo) / (xhi - xlo) * nx))
        ylo <= yi <= yhi && (mk[col, row(yi)] += 1)
        for r in row(max(yi - ei, ylo)):row(min(yi + ei, yhi))
            ink[col, r] += 1
        end
    end
    crowd(a) = (v = filter(>(0), vec(a)); isempty(v) ? 0.0 : quantile(v, q))
    d_mk, d_bar = crowd(mk), crowd(ink)
    alpha = d_mk > 0 ? clamp(c / d_mk, α_min, 1.0) : 1.0
    alpha_bar = d_bar > 0 ? clamp(c_bar / d_bar, α_bar_min, alpha / 2) : alpha / 2
    return (alpha, alpha_bar)
end

"""
    _raw_points!(ax, x, y, err, alpha, alpha_bar; markersize)

The raw cadences of a photometry panel: error bars at `alpha_bar` and markers
at `alpha` (see `_overlap_alpha`).
"""
function _raw_points!(ax, x, y, err, alpha::Real, alpha_bar::Real; markersize::Real)
    errorbars!(ax, x, y, err; color = (NEREUS_COLORS.pm_marker, alpha_bar),
               linewidth = 0.5)
    scatter!(ax, x, y; color = (NEREUS_COLORS.pm_marker, alpha),
             markersize = markersize, strokewidth = 0)
    return nothing
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
