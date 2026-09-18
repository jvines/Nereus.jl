# Paper-quality periodogram plots — matches Vines+ 2023 (HD 18599)
# Figure 7 style for the stacked layout, with B&W-print-safe FAP lines
# (distinct linestyle per level) and a pastel cyan→magenta gradient
# tracking significance.

using CairoMakie
using Colors: RGB, RGBA, parse as _parse_color
using LaTeXStrings: @L_str
using Printf: @sprintf

# Frequency-axis label rendered in LaTeX (proper math fraction for 1/d
# rather than the engineering "[1/d]" bracket).
const _PGRAM_XLABEL = L"\log_{10}\left(\mathrm{Frequency}\left(\frac{1}{d}\right)\right)"

# --- FAP line styling --------------------------------------------------
# (linestyle, color, value-string) per typical 4-level FAP set. The
# styles cycle from "least significant" → "most significant":
#   10%   : dotted,    light pastel cyan
#   5%    : dashed,    cyan
#   1%    : dash-dot,  magenta-pink
#   0.1%  : solid,     hot pink
# Colors are sampled from the Nereus `:cool` colormap so they read in
# B&W as well (light → dark) AND in color as cyan → magenta.

const _FAP_LINESTYLES = Dict(
    0.1   => :dot,
    0.05  => :dash,
    0.01  => :dashdot,
    0.001 => :solid,
)

const _FAP_COLORS = Dict(
    0.1   => RGB(0.55, 0.86, 0.95),   # pastel cyan
    0.05  => RGB(0.30, 0.66, 0.85),   # cyan-blue
    0.01  => RGB(0.83, 0.54, 0.78),   # pinkish lavender
    0.001 => RGB(0.93, 0.34, 0.61),   # hot pink
)

const _PERIODOGRAM_PEAK_FONTSIZE = 14
const _PERIODOGRAM_CHANNEL_FONTSIZE = 16
const _FAP_LABEL_FONTSIZE = 11

"""
    _fap_style(level) -> (linestyle, color)

Look up the line style + colour for a given FAP value. Falls back to
solid black when the level isn't in the canonical set so user-supplied
FAPs still render (just less informatively).
"""
function _fap_style(level::Real)
    # Match by floating-point proximity (so 0.0500001 resolves to 0.05)
    for ref in keys(_FAP_LINESTYLES)
        if isapprox(level, ref; rtol = 1e-4)
            return (_FAP_LINESTYLES[ref], _FAP_COLORS[ref])
        end
    end
    return (:solid, RGB(0, 0, 0))
end

"""
    _fap_label_string(level) -> String

Format the FAP value as it should appear at the left-edge label. Uses
percentages for clarity (10%, 5%, 1%, 0.1%, …).
"""
function _fap_label_string(level::Real)
    pct = level * 100
    return pct >= 1 ? @sprintf("%.0f%%", pct) :
           pct >= 0.1 ? @sprintf("%.1f%%", pct) :
           @sprintf("%.2f%%", pct)
end

_periodogram_axis_label(::GLSPgram)   = "Power"
_periodogram_axis_label(::BGLSPgram)  = "Probability"
_periodogram_axis_label(::SBGLSPgram) = "Power"
_periodogram_axis_label(::L1Pgram)    = "Amplitude"

_periodogram_line_color(::Any) = :black


"""
    _periodogram_plot_to_axis!(ax, pgram; show_faps=true, channel_label=nothing,
                                 max_peak_labels=3,
                                 label_only_significant=false,
                                 fap_cutoff=0.1, fap_text_labels=true)

Render `pgram` into an existing `Axis`. Shared by `plot_periodogram`
and the stacked plot. `fap_text_labels` controls the per-line "10%"
annotation at the left edge.
"""
function _periodogram_plot_to_axis!(ax::Axis, pgram::Periodogram;
                                     show_faps::Bool = true,
                                     channel_label::Union{Nothing, AbstractString} = nothing,
                                     max_peak_labels::Int = 3,
                                     label_only_significant::Bool = false,
                                     fap_cutoff::Float64 = 0.1,
                                     fap_text_labels::Bool = true,
                                     skip_label_edge_frac::Float64 = 0.0)
    log_f = log10.(pgram.frequencies)
    pwr = power(pgram)

    # Tight x-limits — kill the default ~5% margins on both ends.
    xlims!(ax, log_f[1], log_f[end])

    # Power line — keep weight modest; the visual "thickness" of the
    # spectrum is driven by frequency-grid density (samples_per_peak),
    # not the line stroke.
    lines!(ax, log_f, pwr;
            color = _periodogram_line_color(pgram), linewidth = 1.0)

    # FAP thresholds — GLS only. Lines drawn at the actual threshold;
    # labels at the left edge with a per-label vertical offset that
    # prevents overlap when adjacent FAP values fall in a tight band
    # (very common at large N_obs).
    if show_faps && pgram isa GLSPgram && !isempty(pgram.fap_levels)
        # Sort FAP levels by threshold ascending so labels can stagger
        # upward without colliding.
        order = sortperm(pgram.fap_thresholds)
        # Per-label vertical pixel offset; later labels need extra
        # offset only when they'd collide. We compute that at draw time
        # using the axis's pixel-area heuristic: ~14 px per label row.
        prev_thr = -Inf
        # The y-axis is ultimately set by `ylims!` further down; use a
        # rough heuristic — `y_max_data` = max power — for the collision
        # test. FAP thresholds typically fall within (0, y_max_data×0.3).
        y_top_est = maximum(pwr) * 1.18
        # Convert "~16 pixel" label height to a data-y delta via the
        # axis height. We don't know the axis pixel height inside this
        # function, so use a fraction of the data range: 4% of full y.
        min_dy = 0.05 * y_top_est
        last_lbl_y = -Inf
        for ix in order
            lvl = pgram.fap_levels[ix]
            thr = pgram.fap_thresholds[ix]
            ls, col = _fap_style(lvl)
            hlines!(ax, [thr]; color = col, linestyle = ls, linewidth = 1.0)
            fap_text_labels || continue
            # Bump label y up if it would collide with the previous one.
            y_label = max(thr, last_lbl_y + min_dy)
            text!(ax, log_f[1], y_label;
                   text = _fap_label_string(lvl),
                   align = (:left, :center), color = col,
                   fontsize = _FAP_LABEL_FONTSIZE,
                   offset = (3.0, 0.0))
            last_lbl_y = y_label
        end
    end

    # Peak labels — period in days, "%.2f", thin reference line down to
    # the peak, vertical stagger on label collisions.
    labelable = collect(pgram.peaks)
    if label_only_significant && pgram isa GLSPgram
        labelable = [p for p in labelable
                       if isfinite(p.fap) && p.fap <= fap_cutoff]
    end
    npk = min(max_peak_labels, length(labelable))

    # A short channel label ("RV", "S02", "Window") tucks into the top-left
    # corner without touching anything. A long dataset name (e.g.
    # "ASAS-SN V / 609886212052") spans rightward into the peak-label band and
    # overprints the spectrum. For those, reserve a blank strip at the very top
    # and cap the peak labels below it so the two never collide.
    is_long_label = channel_label !== nothing && length(channel_label) > 12

    if npk > 0
        # Generous headroom — peak labels need breathing room AND we
        # need the top y-tick to stay below the label band.
        y_max_data = maximum(pwr)
        y_top = y_max_data * (is_long_label ? 1.80 : 1.55)
        ylims!(ax, 0.0, y_top)
        # Keep peak labels (which grow upward from their baseline) well below
        # the reserved long-label strip.
        peak_cap = y_max_data * (is_long_label ? 1.28 : Inf)
        ordered = sort(view(labelable, 1:npk);
                        by = p -> log10(p.frequency))
        x_min_sep = 0.06 * (log_f[end] - log_f[1])
        last_x = -Inf
        last_y = y_max_data * 1.26    # label baseline above data
        span = log_f[end] - log_f[1]
        for pk in ordered
            x  = log10(pk.frequency)
            # Skip near-edge labels: lowest/highest frequencies are the
            # least-reliable periods (≳ baseline) and the left edge collides
            # with the channel-name annotation.
            if skip_label_edge_frac > 0 && span > 0
                rel = (x - log_f[1]) / span
                (rel < skip_label_edge_frac || rel > 1 - skip_label_edge_frac) && continue
            end
            y_lbl = if x - last_x < x_min_sep
                last_y + y_max_data * 0.12
            else
                y_max_data * 1.26
            end
            y_lbl = min(y_lbl, peak_cap)
            lines!(ax, [x, x], [pk.power, y_lbl];
                    color = (:black, 0.6), linewidth = 0.6)
            text!(ax, x, y_lbl;
                   text = @sprintf("%.2f", pk.period),
                   align = (:center, :bottom),
                   fontsize = _PERIODOGRAM_PEAK_FONTSIZE)
            last_x = x; last_y = y_lbl
        end
    else
        y_max_data = maximum(pwr)
        ylims!(ax, 0.0, y_max_data * (is_long_label ? 1.40 : 1.08))
    end

    # Channel-name annotation (upper-left, in-axis text). Long labels park in
    # the reserved top strip (above the capped peak labels); short ones keep the
    # tight paper-style corner placement.
    if channel_label !== nothing
        text!(ax, 0.012, is_long_label ? 0.99 : 0.95;
               text = channel_label, align = (:left, :top), space = :relative,
               fontsize = is_long_label ? 13 : _PERIODOGRAM_CHANNEL_FONTSIZE)
    end

    return ax
end


"""
    plot_periodogram(pgram::Periodogram; filename=nothing, figsize=(900, 360),
                     channel_label=nothing, show_faps=true,
                     fap_text_labels=true, max_peak_labels=5) -> Figure

Single-panel periodogram. Tight axes, no padding.
"""
function plot_periodogram(pgram::Periodogram;
                           filename::Union{Nothing, AbstractString} = nothing,
                           save_pdf::Bool = false,
                           figsize::Tuple{Int, Int} = (900, 360),
                           channel_label::Union{Nothing, AbstractString} = nothing,
                           show_faps::Bool = true,
                           fap_text_labels::Bool = true,
                           max_peak_labels::Int = 5)
    set_theme!(nereus_theme())
    fig = Figure(; size = figsize)
    ax = Axis(fig[1, 1];
               xlabel = _PGRAM_XLABEL,
               ylabel = _periodogram_axis_label(pgram))
    _periodogram_plot_to_axis!(ax, pgram;
                                show_faps = show_faps,
                                channel_label = channel_label,
                                max_peak_labels = max_peak_labels,
                                fap_text_labels = fap_text_labels)
    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end


"""
    plot_periodograms_stacked(set::PgramSet; filename=nothing,
                               series_order=nothing, panel_height=180,
                               max_peak_labels=3,
                               label_only_significant=true,
                               fap_cutoff=0.1, show_faps=true,
                               fap_text_labels=true, figwidth=900) -> Figure

Stacked, shared-x, glued-row periodogram. Order: RV → indicators → Window.
"""
function plot_periodograms_stacked(set::PgramSet;
                                     filename::Union{Nothing, AbstractString} = nothing,
                                     save_pdf::Bool = false,
                                     series_order::Union{Nothing, Vector{String}} = nothing,
                                     panel_height::Int = 130,
                                     max_peak_labels::Int = 3,
                                     label_only_significant::Bool = true,
                                     fap_cutoff::Float64 = 0.1,
                                     show_faps::Bool = true,
                                     show_window_faps::Bool = false,
                                     fap_text_labels::Bool = true,
                                     figwidth::Int = 900)
    set_theme!(nereus_theme())
    panels = Tuple{String, Periodogram}[("RV", set.rv)]
    if series_order === nothing
        for (nm, pg) in set.indicators
            push!(panels, (nm, pg))
        end
    else
        ind_map = Dict{String, Periodogram}(set.indicators)
        for nm in series_order
            haskey(ind_map, nm) || continue
            push!(panels, (nm, ind_map[nm]))
        end
    end
    push!(panels, ("Window", set.window))

    n_panels = length(panels)
    fig = Figure(; size = (figwidth, panel_height * n_panels + 60),
                   figure_padding = (24, 12, 12, 8))

    axes = Axis[]
    for (i, (nm, pg)) in enumerate(panels)
        is_bottom = i == n_panels
        is_window = nm == "Window"
        ax = Axis(fig[i, 2];
                   xlabel = is_bottom ? _PGRAM_XLABEL : "",
                   ylabel = "",
                   # Sparse y-ticks so adjacent panels' tick labels
                   # don't collide at the shared border. 3 is the
                   # standard astronomy-paper count.
                   yticks = Makie.WilkinsonTicks(3),
                   # Kill EVERY bit of bottom protrusion on non-bottom
                   # axes so the box sits flush against the next one.
                   # matplotlib `hspace=0` equivalent.
                   xticksvisible          = is_bottom,
                   xticklabelsvisible     = is_bottom,
                   xminorticksvisible     = is_bottom,
                   xticksize              = is_bottom ? 6.0 : 0.0,
                   xminorticksize         = is_bottom ? 3.0 : 0.0,
                   xticklabelpad          = is_bottom ? 4.0 : 0.0,
                   xticklabelspace        = is_bottom ? Makie.Automatic() : 0.0,
                   # Shared boundary: every panel keeps its BOTTOM spine
                   # (it doubles as the next panel's top spine). Only
                   # panel 1 keeps its OWN top spine — the rest hide it
                   # so the shared edge is drawn exactly once.
                   topspinevisible        = i == 1,
                   bottomspinevisible     = true)
        # Decide FAP rendering — window panel skips them by default.
        show_faps_panel = show_faps && !(is_window && !show_window_faps)
        # Peak labels also suppressed on the window panel — they're
        # the sampling-pattern aliases, not science.
        peak_labels_panel = is_window ? 0 : max_peak_labels
        _periodogram_plot_to_axis!(ax, pg;
                                    show_faps = show_faps_panel,
                                    channel_label = nm,
                                    max_peak_labels = peak_labels_panel,
                                    label_only_significant = label_only_significant,
                                    fap_cutoff = fap_cutoff,
                                    fap_text_labels = fap_text_labels)
        push!(axes, ax)
    end
    linkxaxes!(axes...)
    rowgap!(fig.layout, 0)

    # Single shared Y label spanning every panel — replaces the
    # five identical "Power" axis labels.
    Label(fig[1:n_panels, 1], _periodogram_axis_label(set.rv);
           rotation = π/2, fontsize = 22, padding = (0, 8, 0, 0),
           tellheight = false)
    colgap!(fig.layout, 0)

    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end


"""
    plot_gls_sectors_stacked(set::GLSSectorSet; filename=nothing, save_pdf=false,
                              panel_height=130, max_peak_labels=3,
                              label_only_significant=true, fap_cutoff=0.1,
                              show_faps=true, show_window_faps=false,
                              fap_text_labels=true, figwidth=900) -> Figure

Stacked, shared-x, glued photometric-GLS figure: one panel per sector,
then the stitched panel, then the sampling window. Same Vines+ 2023
periodogram conventions as `plot_periodograms_stacked` (glued short
panels, 4 FAP lines, `log₁₀(Frequency)` x-axis, shared y-label, no
titles). Pass `save_pdf=true` for a publication-quality `.pdf` companion.
"""
function plot_gls_sectors_stacked(set::GLSSectorSet;
                                    filename::Union{Nothing, AbstractString} = nothing,
                                    save_pdf::Bool = false,
                                    panel_height::Int = 140,
                                    max_peak_labels::Int = 2,
                                    label_only_significant::Bool = true,
                                    fap_cutoff::Float64 = 0.1,
                                    show_faps::Bool = true,
                                    show_window_faps::Bool = false,
                                    fap_text_labels::Bool = true,
                                    figwidth::Int = 900)
    set_theme!(nereus_theme())
    panels = Tuple{String, Periodogram}[]
    for (nm, pg) in zip(set.names, set.sectors)
        push!(panels, (nm, pg))
    end
    push!(panels, ("Stitched", set.stitched))
    push!(panels, ("Window", set.window))

    n_panels = length(panels)
    # extra right padding so the last x-tick label isn't clipped
    fig = Figure(; size = (figwidth, panel_height * n_panels + 60),
                   figure_padding = (24, 30, 12, 8))

    axes = Axis[]
    for (i, (nm, pg)) in enumerate(panels)
        is_bottom = i == n_panels
        is_window = nm == "Window"
        ax = Axis(fig[i, 2];
                   xlabel = is_bottom ? _PGRAM_XLABEL : "",
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
        show_faps_panel   = show_faps && !(is_window && !show_window_faps)
        peak_labels_panel = is_window ? 0 : max_peak_labels
        _periodogram_plot_to_axis!(ax, pg;
                                    show_faps = show_faps_panel,
                                    channel_label = nm,
                                    max_peak_labels = peak_labels_panel,
                                    label_only_significant = label_only_significant,
                                    fap_cutoff = fap_cutoff,
                                    fap_text_labels = fap_text_labels,
                                    skip_label_edge_frac = 0.10)
        push!(axes, ax)
    end
    linkxaxes!(axes...)
    rowgap!(fig.layout, 0)
    Label(fig[1:n_panels, 1], _periodogram_axis_label(set.stitched);
           rotation = π/2, fontsize = 22, padding = (0, 8, 0, 0),
           tellheight = false)
    colgap!(fig.layout, 0)

    filename === nothing || _save_plot(filename, fig; save_pdf = save_pdf)
    return fig
end


"""
    plot_sbgls_heatmap(pgram::SBGLSPgram; filename=nothing,
                        figsize=(900, 480), max_peak_labels=3) -> Figure

Stacked-BGLS heatmap (Mortier & Collier Cameron 2017).
"""
function plot_sbgls_heatmap(pgram::SBGLSPgram;
                             filename::Union{Nothing, AbstractString} = nothing,
                             save_pdf::Bool = false,
                             figsize::Tuple{Int, Int} = (900, 480),
                             max_peak_labels::Int = 3)
    set_theme!(nereus_theme())
    fig = Figure(; size = figsize)
    ax = Axis(fig[1, 1];
               xlabel = _PGRAM_XLABEL,
               ylabel = "N obs")
    log_f = log10.(pgram.frequencies)
    n_rows = size(pgram.matrix, 1)
    heatmap!(ax, log_f, 1:n_rows, transpose(pgram.matrix);
              colormap = NEREUS_CMAP)
    xlims!(ax, log_f[1], log_f[end])
    Colorbar(fig[1, 2]; colormap = NEREUS_CMAP,
              limits = (0, 1), label = "Row-normalized")

    for pk in pgram.peaks[1:min(max_peak_labels, length(pgram.peaks))]
        x = log10(pk.frequency)
        lines!(ax, [x, x], [n_rows * 0.92, n_rows];
                color = :white, linewidth = 1.2)
        text!(ax, x, n_rows * 0.94;
               text = @sprintf("%.2f", pk.period),
               align = (:center, :bottom), color = :white,
               fontsize = _PERIODOGRAM_PEAK_FONTSIZE)
    end

    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end
