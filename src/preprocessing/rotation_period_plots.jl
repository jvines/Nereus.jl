# Plot for `find_rotation_period`. Same paper-quality conventions as
# the periodogram stack: no titles, tight axes, glued panels (when both
# ACF and PACF are shown), channel-name annotation upper-left, peak
# marker + period label above the dominant peak.

using CairoMakie
using Printf: @sprintf

const _ACF_LINE_LW = 1.0
const _ACF_CHANNEL_FONTSIZE = 16
const _ACF_PEAK_FONTSIZE = 14

"""
    plot_acf(result; filename=nothing, figsize=nothing,
              show_pacf=true, max_peak_labels=3) -> Figure

Render an ACF (and optionally PACF) diagnostic for a
`find_rotation_period` result.

When `show_pacf=true` AND `result.pacf !== nothing`, draws two glued
panels (ACF top, PACF bottom). When `show_pacf=false` or PACF wasn't
computed, draws a single panel.
"""
function plot_acf(result::NamedTuple;
                   filename::Union{Nothing, AbstractString} = nothing,
                   save_pdf::Bool = false,
                   figsize::Union{Nothing, Tuple{Int, Int}} = nothing,
                   show_pacf::Bool = true,
                   max_peak_labels::Int = 3)
    has_pacf = show_pacf && result.pacf !== nothing
    n_panels = has_pacf ? 2 : 1
    fs = figsize === nothing ?
         (900, has_pacf ? 320 : 200) :
         figsize

    set_theme!(nereus_theme())
    fig = Figure(; size = fs, figure_padding = (24, 12, 12, 8))

    axes = Axis[]
    # ---- ACF panel ---------------------------------------------------
    ax_acf = Axis(fig[1, 1];
                   xlabel = has_pacf ? "" : "Lag [d]",
                   ylabel = "ACF",
                   yticks = Makie.WilkinsonTicks(3),
                   xticksvisible          = !has_pacf,
                   xticklabelsvisible     = !has_pacf,
                   xminorticksvisible     = !has_pacf,
                   xticksize              = has_pacf ? 0.0 : 6.0,
                   xminorticksize         = has_pacf ? 0.0 : 3.0,
                   xticklabelpad          = has_pacf ? 0.0 : 4.0,
                   xticklabelspace        = has_pacf ?
                                            0.0 : Makie.Automatic(),
                   topspinevisible        = true,
                   bottomspinevisible     = true)
    push!(axes, ax_acf)
    _plot_acf_data!(ax_acf, result; max_peak_labels = max_peak_labels)
    text!(ax_acf, 0.012, 0.95;
           text = "ACF", align = (:left, :top),
           space = :relative, fontsize = _ACF_CHANNEL_FONTSIZE)

    # ---- PACF panel --------------------------------------------------
    if has_pacf
        ax_pacf = Axis(fig[2, 1];
                       xlabel = "Lag [d]",
                       ylabel = "PACF",
                       yticks = Makie.WilkinsonTicks(3),
                       topspinevisible    = false,
                       bottomspinevisible = true)
        push!(axes, ax_pacf)
        _plot_pacf_data!(ax_pacf, result)
        text!(ax_pacf, 0.012, 0.95;
               text = "PACF", align = (:left, :top),
               space = :relative, fontsize = _ACF_CHANNEL_FONTSIZE)
        linkxaxes!(axes...)
        rowgap!(fig.layout, 0)
    end

    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end


"""
    plot_acf_sectors(res; filename=nothing, save_pdf=false, panel_height=160,
                      max_peak_labels=3, max_lag=nothing, figwidth=900) -> Figure

Stacked per-sector ACF figure: one ACF panel per sector (glued, shared
lag x-axis), labelled with the sector name and its `P_rot`. `res` is the
output of `find_rotation_period_sectors`. There is no combined/stitched
ACF by design (see `find_rotation_period_sectors`). `max_lag` sets the
shared lag window [d]; defaults to the shortest sector's baseline so no
panel shows empty space. Pass `save_pdf=true` for a `.pdf` companion.
"""
function plot_acf_sectors(res::NamedTuple;
                           filename::Union{Nothing, AbstractString} = nothing,
                           save_pdf::Bool = false,
                           panel_height::Int = 160,
                           max_peak_labels::Int = 3,
                           max_lag::Union{Nothing, Real} = nothing,
                           figwidth::Int = 900)
    names = res.names
    results = res.results
    n = length(results)
    n >= 1 || throw(ArgumentError("no sectors to plot"))

    x_common = max_lag === nothing ?
               minimum(r.lags[end] for r in results) : Float64(max_lag)

    set_theme!(nereus_theme())
    fig = Figure(; size = (figwidth, panel_height * n + 50),
                   figure_padding = (24, 12, 12, 8))
    axes = Axis[]
    for i in 1:n
        is_bottom = i == n
        ax = Axis(fig[i, 1];
                   xlabel = is_bottom ? "Lag [d]" : "",
                   ylabel = "",
                   yticks = Makie.WilkinsonTicks(3),
                   xticksvisible      = is_bottom,
                   xticklabelsvisible = is_bottom,
                   xminorticksvisible = is_bottom,
                   xticksize          = is_bottom ? 6.0 : 0.0,
                   xminorticksize     = is_bottom ? 3.0 : 0.0,
                   xticklabelpad      = is_bottom ? 4.0 : 0.0,
                   xticklabelspace    = is_bottom ? Makie.Automatic() : 0.0,
                   topspinevisible    = i == 1,
                   bottomspinevisible = true)
        _plot_acf_data!(ax, results[i]; max_peak_labels = max_peak_labels)
        xlims!(ax, 0.0, x_common)            # shared lag window across panels
        lbl = isfinite(results[i].P_rot) ?
              @sprintf("%s   P = %.2f d", names[i], results[i].P_rot) :
              "$(names[i])   (no significant peak)"
        text!(ax, 0.012, 0.95; text = lbl, align = (:left, :top),
               space = :relative, fontsize = _ACF_CHANNEL_FONTSIZE)
        push!(axes, ax)
    end
    rowgap!(fig.layout, 0)
    Label(fig[1:n, 0], "ACF"; rotation = π/2, fontsize = 22,
           padding = (0, 8, 0, 0), tellheight = false)
    colgap!(fig.layout, 0)

    filename === nothing || _save_plot(filename, fig; save_pdf = save_pdf)
    return fig
end


"""
    plot_pacf_sectors(res; filename=nothing, save_pdf=false, panel_height=160,
                       max_lag=nothing, figwidth=900) -> Figure

Stacked per-sector PACF figure (one partial-autocorrelation panel per
sector, glued, shared lag x-axis). `res` is the output of
`find_rotation_period_sectors` — call it with `method=:both` (or
`:pacf`) so the PACF is available. The PACF disambiguates AR-noise from
genuine rotation. Pass `save_pdf=true` for a `.pdf` companion.
"""
function plot_pacf_sectors(res::NamedTuple;
                            filename::Union{Nothing, AbstractString} = nothing,
                            save_pdf::Bool = false,
                            panel_height::Int = 160,
                            max_lag::Union{Nothing, Real} = nothing,
                            figwidth::Int = 900)
    names = res.names
    results = res.results
    n = length(results)
    n >= 1 || throw(ArgumentError("no sectors to plot"))
    all(r -> r.pacf !== nothing, results) || throw(ArgumentError(
        "PACF not computed — call find_rotation_period_sectors with method=:both or :pacf"))

    _pacf_end(r) = length(r.pacf) * r.bin_cadence
    x_common = max_lag === nothing ?
               minimum(_pacf_end(r) for r in results) : Float64(max_lag)

    set_theme!(nereus_theme())
    fig = Figure(; size = (figwidth, panel_height * n + 50),
                   figure_padding = (24, 12, 12, 8))
    axes = Axis[]
    for i in 1:n
        is_bottom = i == n
        ax = Axis(fig[i, 1];
                   xlabel = is_bottom ? "Lag [d]" : "",
                   ylabel = "",
                   yticks = Makie.WilkinsonTicks(3),
                   xticksvisible      = is_bottom,
                   xticklabelsvisible = is_bottom,
                   xminorticksvisible = is_bottom,
                   xticksize          = is_bottom ? 6.0 : 0.0,
                   xminorticksize     = is_bottom ? 3.0 : 0.0,
                   xticklabelpad      = is_bottom ? 4.0 : 0.0,
                   xticklabelspace    = is_bottom ? Makie.Automatic() : 0.0,
                   topspinevisible    = i == 1,
                   bottomspinevisible = true)
        _plot_pacf_data!(ax, results[i])
        xlims!(ax, 0.0, x_common)
        lbl = isfinite(results[i].P_rot) ?
              @sprintf("%s   P = %.2f d", names[i], results[i].P_rot) :
              "$(names[i])   (no significant peak)"
        text!(ax, 0.012, 0.95; text = lbl, align = (:left, :top),
               space = :relative, fontsize = _ACF_CHANNEL_FONTSIZE)
        push!(axes, ax)
    end
    rowgap!(fig.layout, 0)
    Label(fig[1:n, 0], "PACF"; rotation = π/2, fontsize = 22,
           padding = (0, 8, 0, 0), tellheight = false)
    colgap!(fig.layout, 0)

    filename === nothing || _save_plot(filename, fig; save_pdf = save_pdf)
    return fig
end


# Shaded 95% CI fill colour (statsmodels light-blue look)
const _CI_FILL = RGB(0.20, 0.45, 0.80)

"Draw a discrete stem plot (statsmodels `plot_acf`/`plot_pacf` style):
vertical stems lag→value with markers at the tips."
function _stem!(ax::Axis, x::AbstractVector, y::AbstractVector)
    segs = Point2f[]
    @inbounds for i in eachindex(x)
        push!(segs, Point2f(x[i], 0.0))
        push!(segs, Point2f(x[i], y[i]))
    end
    linesegments!(ax, segs; color = :black, linewidth = 0.9)
    scatter!(ax, x, y; color = :black, markersize = 4, strokewidth = 0)
end

function _plot_acf_data!(ax::Axis, result::NamedTuple;
                          max_peak_labels::Int = 3)
    lags = result.lags; acf = result.acf
    N = max(result.n_bins, 2)

    # Bartlett 95% CI band — widens with lag: ci_k = 1.96·√((1+2·Σ_{j<k}ρ_j²)/N).
    ci = Vector{Float64}(undef, length(acf))
    csum = 0.0
    @inbounds for k in eachindex(acf)
        ci[k] = 1.96 * sqrt((1 + 2 * csum) / N)
        csum += acf[k]^2
    end
    band!(ax, lags, -ci, ci; color = (_CI_FILL, 0.30))
    hlines!(ax, [0.0]; color = (:black, 0.5), linewidth = 0.8)
    _stem!(ax, lags, acf)

    y_lo = min(minimum(acf), -maximum(ci)) * 1.10
    ylims!(ax, y_lo, 1.18)

    # Rotation aids (not in vanilla statsmodels): P_rot line + peak labels.
    if isfinite(result.P_rot)
        vlines!(ax, [result.P_rot]; color = (RGB(0.93, 0.34, 0.61), 0.85),
                 linestyle = :dash, linewidth = 1.0)
    end
    npk = min(max_peak_labels, length(result.peaks))
    if npk > 0
        ordered = sort(view(result.peaks, 1:npk); by = p -> p.lag)
        for pk in ordered
            text!(ax, pk.lag, min(pk.value + 0.08, 1.06);
                   text = @sprintf("%.2f", pk.lag),
                   align = (:center, :bottom), fontsize = _ACF_PEAK_FONTSIZE)
        end
    end
end


function _plot_pacf_data!(ax::Axis, result::NamedTuple)
    pacf = result.pacf
    pacf === nothing && return
    N = max(result.n_bins, 2)
    dt = result.bin_cadence
    pacf_lags = collect(1:length(pacf)) .* dt

    # PACF 95% CI band is constant: ±1.96/√N (Quenouille).
    ci = 1.96 / sqrt(N)
    band!(ax, [pacf_lags[1], pacf_lags[end]], [-ci, -ci], [ci, ci];
           color = (_CI_FILL, 0.30))
    hlines!(ax, [0.0]; color = (:black, 0.5), linewidth = 0.8)
    _stem!(ax, pacf_lags, pacf)
    if isfinite(result.P_rot)
        vlines!(ax, [result.P_rot]; color = (RGB(0.93, 0.34, 0.61), 0.85),
                 linestyle = :dash, linewidth = 1.0)
    end
end
