# TTV plotting helpers: O-C diagram + per-transit overlay.
#
# Nereus convention: no figure titles. Context goes in the file name,
# axis labels, per-panel subtitle (instrument / epoch), or the legend.
#
# Public API:
#   plot_ttv_diagram(transit_idx, δts_med, δts_lo, δts_hi; ...)
#   plot_ttv_diagram_multipanel(transit_idx, δt_med, δt_lo, δt_hi,
#                                 epoch_groups, epoch_labels; ...)
#   plot_transit_overlay(t_phot, flux, err, P, Tc, δts_med, model_fn; ...)

using CairoMakie
using Statistics: median, quantile

"""
    plot_ttv_diagram(transit_idx, δts_med [, δts_lo, δts_hi];
                      ylabel="O − C (min)", filename=nothing,
                      predicted=nothing, predicted_label="N-body") -> Figure

O − C (observed-minus-computed) diagram.

# Arguments
- `transit_idx::AbstractVector{<:Integer}` — transit number (1-based)
- `δts_med::AbstractVector{<:Real}` — median TTV offset per transit, days
- `δts_lo`, `δts_hi` — 16/84 percentile bands, optional (errorbars)
- `predicted::AbstractVector{<:Real}` — overlay (e.g. TTVFaster), days

Values internally converted to minutes for display.
"""
function plot_ttv_diagram(transit_idx::AbstractVector{<:Integer},
                            δts_med::AbstractVector{<:Real},
                            δts_lo::Union{AbstractVector{<:Real}, Nothing} = nothing,
                            δts_hi::Union{AbstractVector{<:Real}, Nothing} = nothing;
                            ylabel::AbstractString = "O − C [min]",
                            xlabel::AbstractString = "Transit number",
                            ylim::Union{Nothing, Tuple{<:Real, <:Real}} = nothing,
                            filename::Union{Nothing, AbstractString} = nothing,
                            save_pdf::Bool = false,
                            epoch_groups::Union{AbstractVector, Nothing} = nothing,
                            epoch_labels::Union{AbstractVector{<:AbstractString}, Nothing} = nothing,
                            predicted::Union{AbstractVector{<:Real}, Nothing} = nothing,
                            predicted_lo::Union{AbstractVector{<:Real}, Nothing} = nothing,
                            predicted_hi::Union{AbstractVector{<:Real}, Nothing} = nothing,
                            predicted_label::AbstractString = "N-body model")
    set_theme!(nereus_theme())
    fig = Figure(size = (1100, 480))
    ax = Axis(fig[1, 1]; xlabel = xlabel, ylabel = ylabel)

    xs = collect(transit_idx)
    y_min = 24 * 60 .* collect(δts_med)
    has_err = δts_lo !== nothing && δts_hi !== nothing
    lo_min = has_err ? 24 * 60 .* (collect(δts_med) .- collect(δts_lo)) : nothing
    hi_min = has_err ? 24 * 60 .* (collect(δts_hi) .- collect(δts_med)) : nothing

    # Publication-standard: thin black error bars + small filled markers.
    # Different markers per epoch (Becker 2015 / Hadden 2017 style).
    markers = (:circle, :rect, :utriangle, :diamond, :pentagon, :star5)

    # Dashed zero reference first (below points).
    hlines!(ax, [0.0]; color = (:black, 0.5), linestyle = :dash, linewidth = 1)

    if epoch_groups === nothing
        if has_err
            errorbars!(ax, xs, y_min, lo_min, hi_min;
                        color = :black, linewidth = 0.9, whiskerwidth = 4)
        end
        scatter!(ax, xs, y_min;
                  color = :black, marker = :circle, markersize = 7,
                  strokewidth = 0)
    else
        labels = epoch_labels === nothing ?
            ["epoch $k" for k in eachindex(epoch_groups)] :
            collect(epoch_labels)
        for (k, grp) in enumerate(epoch_groups)
            mk = markers[mod1(k, length(markers))]
            if has_err
                errorbars!(ax, xs[grp], y_min[grp], lo_min[grp], hi_min[grp];
                            color = :black, linewidth = 0.9, whiskerwidth = 4)
            end
            scatter!(ax, xs[grp], y_min[grp];
                      color = :black, marker = mk, markersize = 8,
                      strokewidth = 0, label = labels[k])
        end
    end

    if predicted !== nothing
        pred_min = 24 * 60 .* collect(predicted)
        ord = sortperm(xs)
        # CI band first (under the line) if 16/84 supplied. Bumped from
        # CI_ALPHA=0.25 to 0.5 because the TTV-OC band sits behind
        # high-contrast black data points and a bright cyan median line;
        # the standard 25% band was too faint to read.
        if predicted_lo !== nothing && predicted_hi !== nothing
            lo_min = 24 * 60 .* collect(predicted_lo)
            hi_min = 24 * 60 .* collect(predicted_hi)
            band!(ax, xs[ord], lo_min[ord], hi_min[ord];
                   color = (NEREUS_COLORS.ci, CI_ALPHA_1SIGMA))
        end
        lines!(ax, xs[ord], pred_min[ord];
                color = NEREUS_COLORS.pm_model, linewidth = 2,
                label = predicted_label)
    end

    if epoch_groups !== nothing || predicted !== nothing
        axislegend(ax; position = :rt, framevisible = false, labelsize = 18)
    end

    ylim === nothing || ylims!(ax, ylim[1], ylim[2])

    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end

"""
    plot_ttv_oc(chains, data, params;
                planet_a_k=1, planet_b_k=2,
                tcs_predicted=nothing,
                filename=nothing,
                half_window=0.5, ngrid=801,
                n_envelope_draws=400, seed=42,
                kwargs...) -> Figure

End-to-end TTV O−C diagram in Nereus publication style.

Combines:
  1. Per-transit photometric Tc measurements (data) from the phot LL
     profiled at the posterior median (via `measure_per_transit_tcs`)
  2. TTV-C / TTVFaster N-body envelope from posterior draws (via
     `ttvc_envelope`) — median line + 16/84 CI band

Both are passed to `plot_ttv_diagram` with the Nereus color scheme.
No fiddling required by the user: one call, publication-ready plot.

# Arguments
- `chains` — MCMCChains.Chains object (from `sample_*` result)
- `data::Data` — photometric data used in the fit
- `params::Params` — model layout
- `planet_a_k::Int=1` — transiting planet whose Tc's are measured
- `planet_b_k::Int=2` — perturber planet (non-transiting allowed) —
  set to `0` to skip the N-body envelope and plot just the data
- `tcs_predicted` — linear-ephemeris Tc grid; if `nothing`, computed
  from `P_k\$(planet_a_k)` and `Tc_k\$(planet_a_k)` posterior medians
"""
function plot_ttv_oc(chains, data::Data, params::Params;
                       planet_a_k::Int = 1, planet_b_k::Int = 2,
                       tcs_predicted::Union{Vector{Float64}, Nothing} = nothing,
                       filename::Union{Nothing, AbstractString} = nothing,
                       save_pdf::Bool = false,
                       half_window::Float64 = 0.5,
                       ngrid::Int = 801,
                       grid_min::Float64 = 0.4,
                       n_envelope_draws::Int = 400,
                       seed::Int = 42,
                       kwargs...)
    # 1. Posterior-median theta
    theta_med = Theta{Float64}(params)
    set_n_p!(theta_med, params.config.max_kplanet)
    pname_set = Set(string.(names(chains, :parameters)))
    for nm in keys(params.layout.name_to_idx)
        if nm in pname_set
            v = vec(Array(chains[:, Symbol(nm), :]))
            theta_med.values[params.layout.name_to_idx[nm]] = median(v)
        end
    end

    # 2. Predicted Tc grid
    P_a  = theta_med.values[params.layout.name_to_idx["P_k$(planet_a_k)"]]
    Tc_a = theta_med.values[params.layout.name_to_idx["Tc_k$(planet_a_k)"]]
    if tcs_predicted === nothing
        n_lo = ceil(Int,  (minimum(data.t_phot) + 0.1 - Tc_a) / P_a)
        n_hi = floor(Int, (maximum(data.t_phot) - 0.1 - Tc_a) / P_a)
        tcs_predicted = [Tc_a + n*P_a for n in n_lo:n_hi]
    end

    # 3. Per-transit Tc measurements (data points)
    meas = measure_per_transit_tcs(data, theta_med, params;
                                    tcs_predicted = tcs_predicted,
                                    half_window = half_window,
                                    grid_min = grid_min,
                                    ngrid = ngrid,
                                    planet_k = planet_a_k)

    # 4. N-body envelope (optional)
    pred = nothing; pred_lo = nothing; pred_hi = nothing
    if planet_b_k != 0
        env = ttvc_envelope(chains, params;
                              tcs_observed = tcs_predicted,
                              planet_a_k = planet_a_k,
                              planet_b_k = planet_b_k,
                              n_draws = n_envelope_draws,
                              seed = seed)
        pred    = env.med
        pred_lo = env.lo
        pred_hi = env.hi
    end

    # 5. Plot — data via δt_med/lo/hi, model via predicted/lo/hi
    return plot_ttv_diagram(meas.cycle_idx,
                              meas.mle_dt, meas.lo_dt, meas.hi_dt;
                              predicted = pred,
                              predicted_lo = pred_lo,
                              predicted_hi = pred_hi,
                              filename = filename,
                              save_pdf = save_pdf,
                              kwargs...)
end


"""
    plot_ttv_diagram_multipanel(transit_idx, δt_med, δt_lo, δt_hi,
                                  epoch_groups, epoch_labels;
                                  filename=nothing, predicted=nothing,
                                  predicted_label="N-body prediction")

O−C diagram with one panel per epoch. `epoch_groups[k]` selects indices
into `δt_med` that belong to epoch `k`. Per-panel subtitle uses
`epoch_labels[k]` (e.g. "K2_c03", "TESS_s42"). No overall figure title.
"""
function plot_ttv_diagram_multipanel(transit_idx::AbstractVector{<:Integer},
                                        δt_med::AbstractVector{<:Real},
                                        δt_lo::AbstractVector{<:Real},
                                        δt_hi::AbstractVector{<:Real},
                                        epoch_groups::AbstractVector,
                                        epoch_labels::AbstractVector{<:AbstractString};
                                        filename::Union{Nothing, AbstractString} = nothing,
                                        save_pdf::Bool = false,
                                        predicted::Union{AbstractVector{<:Real}, Nothing} = nothing,
                                        predicted_label::AbstractString = "N-body prediction")
    set_theme!(nereus_theme())
    n = length(epoch_groups)
    fig = Figure(size = (450 * n, 500))
    all_lo = 24 * 60 .* collect(δt_lo)
    all_hi = 24 * 60 .* collect(δt_hi)
    ymin = min(0.0, minimum(all_lo)) - 1.0
    ymax = max(0.0, maximum(all_hi)) + 1.0
    for (k, grp) in enumerate(epoch_groups)
        ax = Axis(fig[1, k];
                   xlabel = "Transit number",
                   ylabel = k == 1 ? "O − C (min)" : "")
        text!(ax, 0.04, 0.96; text = epoch_labels[k],
               align = (:left, :top), space = :relative, fontsize = 16)
        xs = collect(transit_idx[grp])
        y_min = 24 * 60 .* collect(δt_med[grp])
        lo_min = 24 * 60 .* (collect(δt_med[grp]) .- collect(δt_lo[grp]))
        hi_min = 24 * 60 .* (collect(δt_hi[grp]) .- collect(δt_med[grp]))
        errorbars!(ax, xs, y_min, lo_min, hi_min;
                    color = NEREUS_COLORS.post, linewidth = ERRBAR_LW)
        scatter!(ax, xs, y_min;
                  color = NEREUS_COLORS.post,
                  marker = :circle, markersize = 12,
                  strokewidth = 0.5, strokecolor = :black)
        hlines!(ax, [0.0]; color = NEREUS_COLORS.zero_line,
                 linestyle = :dash, linewidth = 1)
        if predicted !== nothing
            lines!(ax, xs, 24 * 60 .* collect(predicted[grp]);
                    color = NEREUS_COLORS.pm_model, linewidth = 2.5,
                    label = predicted_label)
            k == 1 && axislegend(ax; position = :rt, framevisible = false)
        end
        ylims!(ax, ymin, ymax)
    end
    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end

"""
    plot_transit_overlay(t_segments, flux_segments, err_segments,
                          model_segments; transit_idx, filename=nothing,
                          tcs=nothing) -> Figure

Per-transit panels: data + best-fit model. `t_segments` etc are vectors
of vectors (one per transit). Per-panel subtitle is "transit N"; no
overall figure title.
"""
function plot_transit_overlay(t_segs::AbstractVector{<:AbstractVector{<:Real}},
                                flux_segs::AbstractVector{<:AbstractVector{<:Real}},
                                err_segs::AbstractVector{<:AbstractVector{<:Real}},
                                model_segs::AbstractVector{<:AbstractVector{<:Real}};
                                transit_idx::AbstractVector{<:Integer} = 1:length(t_segs),
                                filename::Union{Nothing, AbstractString} = nothing,
                                save_pdf::Bool = false,
                                tcs::Union{AbstractVector{<:Real}, Nothing} = nothing)
    set_theme!(nereus_theme())
    n = length(t_segs)
    ncols = min(n, 4)
    nrows = ceil(Int, n / ncols)
    fig = Figure(size = (300 * ncols, 240 * nrows))
    for (k, idx) in enumerate(transit_idx)
        row = (k - 1) ÷ ncols + 1
        col = (k - 1) % ncols + 1
        ax = Axis(fig[row, col];
                   xlabel = "BJD − Tc (hr)",
                   ylabel = "rel. flux")
        # Bottom-left corner — out-of-transit + low-flux region is empty there,
        # so the label never overlaps the baseline data or the transit dip.
        text!(ax, 0.04, 0.05; text = "transit $idx",
               align = (:left, :bottom), space = :relative, fontsize = 12)
        ts = collect(t_segs[k])
        fs = collect(flux_segs[k])
        es = collect(err_segs[k])
        ms = collect(model_segs[k])
        tc = tcs === nothing ? median(ts) : tcs[k]
        x_hr = 24 .* (ts .- tc)
        errorbars!(ax, x_hr, fs, es;
                    color = (NEREUS_COLORS.pm_marker, 0.4),
                    linewidth = 1)
        scatter!(ax, x_hr, fs;
                  color = NEREUS_COLORS.pm_marker,
                  markersize = 5, strokewidth = 0)
        ord = sortperm(x_hr)
        lines!(ax, x_hr[ord], ms[ord];
                color = NEREUS_COLORS.model, linewidth = 2)
    end
    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end

"""
    plot_transit_overlay_fit(chains, params, data; planet=nothing, half_window=0.18,
                              output=nothing, fmt=:png, save_pdf=false) -> Figure

Per-transit light-curve gallery from a fit: one panel per transit of the
transiting planet, each showing that transit's photometry + the posterior-median
transit model (x = hours from Tc). Per-transit QC — catches a single bad transit
(spot crossing, partial/grazing, systematic) that the stacked phase-fold hides.
Saved as `models/transit_overlay_K<k>.<fmt>`.
"""
function plot_transit_overlay_fit(chains, params, data;
                                   planet::Union{Nothing, Int} = nothing,
                                   half_window::Real = 0.18,
                                   min_pts::Int = 5,
                                   output::Union{Nothing, String} = nothing,
                                   fmt::Symbol = :png, save_pdf::Bool = false)
    n_phot(data) > 0 || return Figure()
    pk = planet === nothing ? findfirst(has_pm, params.config.planet_modes) : planet
    pk === nothing && return Figure()

    # posterior-median theta (same construction as plot_ttv_oc)
    theta_med = Theta{Float64}(params)
    set_n_p!(theta_med, params.config.max_kplanet)
    pname_set = Set(string.(names(chains, :parameters)))
    for nm in keys(params.layout.name_to_idx)
        nm in pname_set &&
            (theta_med.values[params.layout.name_to_idx[nm]] = median(vec(Array(chains[:, Symbol(nm), :]))))
    end

    model_all, _ = phot_predictions(theta_med, data)
    P = planet_P(theta_med, pk); Tc = _planet_Tc(theta_med, data, pk)
    hw = Float64(half_window)
    n_lo = ceil(Int,  (minimum(data.t_phot) + 0.02 - Tc) / P)
    n_hi = floor(Int, (maximum(data.t_phot) - 0.02 - Tc) / P)

    t_segs = Vector{Float64}[]; flux_segs = Vector{Float64}[]
    err_segs = Vector{Float64}[]; model_segs = Vector{Float64}[]
    tcs = Float64[]; idx = Int[]
    for n in n_lo:n_hi
        c = Tc + n * P
        m = (data.t_phot .>= c - hw) .& (data.t_phot .<= c + hw)
        count(m) >= min_pts || continue
        push!(t_segs, data.t_phot[m]); push!(flux_segs, data.flux[m])
        push!(err_segs, data.flux_err[m]); push!(model_segs, model_all[m])
        push!(tcs, c); push!(idx, n - n_lo + 1)
    end
    isempty(t_segs) && return Figure()

    fn = output === nothing ? nothing :
         (mkpath(joinpath(output, "models")); joinpath(output, "models", "transit_overlay_K$(pk).$fmt"))
    return plot_transit_overlay(t_segs, flux_segs, err_segs, model_segs;
                                 transit_idx = idx, tcs = tcs, filename = fn, save_pdf = save_pdf)
end
