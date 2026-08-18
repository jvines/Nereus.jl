# Detrending diagnostic plots — broken-axis style for handling gaps.
#
# Produces a two-panel figure (top: original LC + model, bottom: detrended)
# with broken-axis horizontal layout: each contiguous segment gets its own
# x-axis range, separated by visible "//" break marks. This is the standard
# way to display TESS data with downlink gaps and sector boundaries.

# Cyan for data, black for model — matches EMPEROR convention.
const DETREND_DATA_COLOR  = colorant"#00bcd4"     # cyan
const DETREND_MODEL_COLOR = colorant"black"
const DETREND_MASK_COLOR  = colorant"#e91e63"     # pink transit shading
const DETREND_DATA_MS     = 3
const DETREND_MODEL_LW    = 1.8

"""
    plot_detrending(t, flux, flux_err, result; output=nothing,
                     title="", figsize=nothing,
                     show_mask=true, fmt=:both,
                     time_offset=:auto, segments=nothing)

Plot a detrending result. Two panels:
- top: original flux (cyan) with the trend model (black line)
- bottom: detrended flux (cyan) with zero/unity reference line

Each segment gets its own x-axis pane in a broken-axis layout, with
"//" break marks between panes. This matches the standard convention
for TESS data spanning multiple sectors with downlink gaps.

# Arguments
- `t, flux, flux_err` — light curve arrays
- `result` — return value of `detrend_savgol` or `detrend_gp`
   (a NamedTuple with `flux_detrended`, `trend`, `segments`,
   `transit_mask` fields)

# Keyword arguments
- `output::Union{Nothing, String}` — directory to save plots
- `title::String=""` — figure title (e.g., star name)
- `figsize` — `(width, height)` in pixels; auto-computed if `nothing`
- `show_mask::Bool=true` — shade transit-masked points if available
- `fmt::Symbol=:both` — `:png`, `:pdf`, or `:both`
- `time_offset::Union{Real, Symbol}=:auto` — subtract this offset from
   `t` before plotting. `:auto` chooses 2457000 (BJD → TJD, TESS) when
   the median time exceeds 2.45e6, else 0. Pass a number to override
   (e.g., `2400000` for MJD-style).
- `sector_id::Union{Nothing, AbstractVector{<:Integer}}` — per-cadence
   sector tag (1-indexed). Each unique value becomes its own plot pane
   with a `⫽` break to its neighbours. Mirrors `detrend_gp`'s API.
- `segments::Union{Nothing, AbstractVector}` — escape hatch for
   non-sector-based pane splits (pass UnitRange-of-Int directly).
   Takes precedence over `sector_id` when both are given.

When the figure has more than one segment pane and `output` is set,
per-segment zoom-in figures are also written.
"""
function plot_detrending(t::AbstractVector{<:Real},
                          flux::AbstractVector{<:Real},
                          flux_err::AbstractVector{<:Real},
                          result;
                          output::Union{Nothing, String}=nothing,
                          title::String="",
                          figsize=nothing,
                          show_mask::Bool=true,
                          fmt::Symbol=:png,
                          save_pdf::Bool=false,
                          time_offset::Union{Real, Symbol}=:auto,
                          segments=nothing,
                          sector_id::Union{Nothing,
                                            AbstractVector{<:Integer}}=nothing,
                          name::String="detrending")
    plot_segments = if segments !== nothing
        segments
    elseif sector_id !== nothing
        _segments_from_sector_id(sector_id)
    else
        result.segments
    end
    # Always rebuild `result.segments` from the chosen plot_segments —
    # the user may have passed `result` with `segments = nothing` (when
    # the result is a slice of a multi-sector cleaning) or with their
    # own gap-detection-based segments that don't align with how they
    # want the figure laid out.
    n_seg = length(plot_segments)
    t_offset = _resolve_time_offset(t, time_offset)
    tv = Vector{Float64}(t); fv = Vector{Float64}(flux); ev = Vector{Float64}(flux_err)

    # Paper-readable paging: at most 3 sectors/segments per figure, and when
    # there are more, split into BALANCED pages instead of cramming or leaving
    # an orphan — 4 → 2+2, 5 → 3+2, 6 → 3+3, 7 → 3+2+2. One figure for ≤3.
    pages = _paginate_segments(n_seg)
    npg = length(pages)
    figs = Any[]
    for (pi, rng) in enumerate(pages)
        pr  = merge(result, (segments = plot_segments[rng],))
        fig = _plot_detrend_broken(tv, fv, ev, pr;
                                   title=title, figsize=figsize,
                                   show_mask=show_mask, time_offset=t_offset)
        push!(figs, fig)
        if output !== nothing
            mkpath(joinpath(output, "detrend"))
            suffix = npg == 1 ? "" : "_p$(pi)"
            _save_fig(fig, joinpath(output, "detrend", name * suffix), fmt;
                      save_pdf=save_pdf)
        end
    end
    return npg == 1 ? figs[1] : figs
end

"""
    _paginate_segments(n; maxper=3) -> Vector{UnitRange{Int}}

Split `n` segments into balanced contiguous pages of at most `maxper` each.
Pages differ in size by at most one, so 4→[1:2,3:4], 5→[1:3,4:5],
7→[1:3,4:5,6:7] — never a lone orphan panel. `n ≤ maxper` ⇒ a single page.
"""
function _paginate_segments(n::Int; maxper::Int = 3)
    n <= maxper && return [1:n]
    npages = cld(n, maxper)
    base, rem = divrem(n, npages)
    ranges = UnitRange{Int}[]
    start = 1
    for i in 1:npages
        sz = base + (i <= rem ? 1 : 0)
        push!(ranges, start:(start + sz - 1))
        start += sz
    end
    return ranges
end

"""Resolve `time_offset` argument; `:auto` ⇒ 2457000 if data are BJD,
else 0. Numeric values pass through."""
function _resolve_time_offset(t::AbstractVector{<:Real},
                               offset::Union{Real, Symbol})
    if offset isa Symbol
        offset === :auto ||
            throw(ArgumentError("time_offset Symbol must be :auto"))
        med = median(t)
        return med > 2.45e6 ? 2457000.0 : 0.0
    else
        return Float64(offset)
    end
end

function _save_fig(fig, base::String, fmt::Symbol; save_pdf::Bool=false)
    # `fmt` is the explicit primary format. `save_pdf` is the opt-in
    # toggle to additionally save a `.pdf` companion when fmt isn't
    # already PDF. Default behavior: PNG only.
    if fmt === :png || fmt === :both
        save(base * ".png", fig; px_per_unit=3)
    end
    if fmt === :pdf || fmt === :both || save_pdf
        save(base * ".pdf", fig)
    end
end

"""Build the broken-axis two-panel plot. `time_offset` is subtracted
from `t` before plotting (e.g. 2457000 for BJD → TJD).
"""
function _plot_detrend_broken(t::Vector{Float64}, flux::Vector{Float64},
                                flux_err::Vector{Float64}, result;
                                title::String="", figsize=nothing,
                                show_mask::Bool=true,
                                time_offset::Float64 = 0.0)
    segments = result.segments
    trend = result.trend
    flux_dt = result.flux_detrended
    # Cadences to highlight as transit/dip. Result shape differs by detrender:
    # savgol/GP carry the input `transit_mask`; notch DETECTS dips (`delta_bic`>0);
    # LOCoR carries neither (it protects via the input mask but doesn't return it).
    mask = if hasproperty(result, :transit_mask) && result.transit_mask !== nothing
        result.transit_mask
    elseif hasproperty(result, :delta_bic)
        result.delta_bic .> 0
    else
        falses(length(t))
    end
    n_seg = length(segments)

    t_disp = t .- time_offset
    x_label = time_offset == 0 ? "Time [days]" :
              "Time [BJD − $(round(Int, time_offset))]"

    # Per-segment x-axis ranges (in display time)
    seg_ranges = [extrema(t_disp[seg]) for seg in segments]
    seg_widths = [r[2] - r[1] for r in seg_ranges]
    total_width = sum(seg_widths)

    # Figure size — width scales with number of segments
    if figsize === nothing
        per_seg = 700
        gap_px  = 30
        w = 200 + per_seg * n_seg + gap_px * (n_seg - 1)
        figsize = (min(w, 4000), 600)
    end

    with_theme(nereus_theme()) do
        fig = Figure(; size=figsize, figure_padding=(20, 20, 30, 30))
        ga = fig[1, 1] = GridLayout()

        # Two rows (top: original+trend, bottom: detrended), n_seg columns
        # Column widths proportional to segment time spans
        # Detect detrending convention: if median(flux_dt) ≈ 1, multiplicative
        # (SG style, flux/trend); otherwise additive (GP, flux − trend).
        med_dt = median(flux_dt)
        is_multiplicative = abs(med_dt - 1.0) < abs(med_dt)
        ref_line = is_multiplicative ? 1.0 : 0.0

        axes_top = Vector{Axis}(undef, n_seg)
        axes_bot = Vector{Axis}(undef, n_seg)

        # Shared y-limits: percentile-based, robust to outliers (cosmic
        # rays, edge artefacts) that would otherwise blow out the range
        # and crush the rotation modulation into a flat line. 0.5–99.5%
        # percentiles cover ~5σ of clean data; 8% padding above and
        # below leaves room for transit shading and tick labels.
        all_flux = vcat([flux[s] for s in segments]...)
        all_dt   = vcat([flux_dt[s] for s in segments]...)
        ylim_top = _robust_ylim(all_flux)
        ylim_bot = _robust_ylim(all_dt)

        for (k, seg) in enumerate(segments)
            t_seg     = t_disp[seg]
            flux_seg  = flux[seg]
            err_seg   = flux_err[seg]
            trend_seg = trend[seg]
            dt_seg    = flux_dt[seg]
            mask_seg  = mask === nothing ? nothing : mask[seg]

            # Marker size + alpha tuned to keep data visible at all
            # densities. Dense LCs (TESS 20-sec) need very small markers
            # to avoid becoming a solid cyan slab; sparse LCs deserve
            # bigger markers + errorbars. The 0.35 alpha floor is
            # deliberate — anything lower and the cyan disappears
            # against white background printing.
            n_pts = length(t_seg)
            data_alpha = clamp(5000 / n_pts, 0.35, 1.0)
            ms = n_pts > 50000 ? 1.0 :
                 n_pts > 5000  ? 1.5 :
                 n_pts > 1000  ? 2.5 : DETREND_DATA_MS
            # Errorbars are always drawn; alpha tapers with density so
            # sparse LCs get crisp bars and dense LCs get a faint
            # uncertainty band rather than a saturated slab.
            eb_alpha = clamp(2000 / n_pts, 0.03, 0.5)

            # Top panel
            ax_t = Axis(ga[1, k];
                        ylabel = k == 1 ? "Flux" : "",
                        xticklabelsvisible = false)
            axes_top[k] = ax_t

            if show_mask && mask_seg !== nothing && any(mask_seg)
                _shade_mask!(ax_t, t_seg, mask_seg, ylim_top)
            end

            errorbars!(ax_t, t_seg, flux_seg, err_seg;
                       color=(DETREND_DATA_COLOR, eb_alpha), linewidth=0.6)
            scatter!(ax_t, t_seg, flux_seg;
                     color=(DETREND_DATA_COLOR, data_alpha), markersize=ms,
                     strokewidth=0, rasterize=2)
            lines!(ax_t, t_seg, trend_seg;
                   color=DETREND_MODEL_COLOR, linewidth=DETREND_MODEL_LW)

            ylims!(ax_t, ylim_top)
            xlims!(ax_t, seg_ranges[k])

            # Bottom panel
            ax_b = Axis(ga[2, k];
                        xlabel = "",   # global label below, spans figure
                        ylabel = k == 1 ? "Detrended" : "",
                        xticks = WilkinsonTicks(4))
            axes_bot[k] = ax_b

            if show_mask && mask_seg !== nothing && any(mask_seg)
                _shade_mask!(ax_b, t_seg, mask_seg, ylim_bot)
            end

            errorbars!(ax_b, t_seg, dt_seg, err_seg;
                       color=(DETREND_DATA_COLOR, eb_alpha), linewidth=0.6)
            scatter!(ax_b, t_seg, dt_seg;
                     color=(DETREND_DATA_COLOR, data_alpha), markersize=ms,
                     strokewidth=0, rasterize=2)
            hlines!(ax_b, ref_line; color=DETREND_MODEL_COLOR,
                    linestyle=:dash, linewidth=1.0)

            ylims!(ax_b, ylim_bot)
            xlims!(ax_b, seg_ranges[k])

            # Hide y-axis labels, tick labels, AND tick marks on all
            # but the first column — duplicate ticks across linked y
            # axes are redundant and cluttered.
            if k > 1
                hideydecorations!(ax_t; grid=false)
                hideydecorations!(ax_b; grid=false)
            end

            # Hide right spine and add break marks for gaps
            if k < n_seg
                ax_t.rightspinevisible = false
                ax_b.rightspinevisible = false
            end
            if k > 1
                ax_t.leftspinevisible = false
                ax_b.leftspinevisible = false
            end
        end

        # Column widths proportional to time span (with a minimum)
        # to avoid one tiny segment squeezing the others.
        for (k, w) in enumerate(seg_widths)
            colsize!(ga, k, Auto(max(w, total_width * 0.05)))
        end
        rowsize!(ga, 1, Relative(2/3))
        rowsize!(ga, 2, Relative(1/3))
        rowgap!(ga, 5)
        colgap!(ga, 22)   # horizontal gap between sector panes

        # Single x-axis label spanning the full figure (instead of one
        # under the middle subplot) — matches publication convention
        # for broken-axis figures and stays centered regardless of how
        # many panes there are.
        Label(ga[3, :], x_label; fontsize=22, tellwidth=false,
              padding=(0, 0, 6, 0))
        rowsize!(ga, 3, Auto())

        # Link y-axes across columns
        n_seg > 1 && linkyaxes!(axes_top...)
        n_seg > 1 && linkyaxes!(axes_bot...)

        # Add break marks between segments — drawn on fig.scene in
        # pixel coords so they ride on top of the spine without
        # clipping. Must run AFTER the layout is otherwise complete
        # so xaxis.endpoints reflect final pixel positions.
        for k in 1:(n_seg - 1)
            _add_break_marks!(fig, axes_top, axes_bot, k)
        end

        return fig
    end
end

"""Shade transit-masked regions on an axis."""
function _shade_mask!(ax::Axis, t_seg::Vector{Float64},
                       mask_seg::AbstractVector{Bool},
                       ylim::Tuple)
    # Find runs of consecutive True values
    n = length(mask_seg)
    i = 1
    while i <= n
        if mask_seg[i]
            j = i
            while j <= n && mask_seg[j]
                j += 1
            end
            # Shade from t[i] to t[j-1] (with half-bin extension)
            t_lo = i > 1 ? (t_seg[i] + t_seg[i - 1]) / 2 : t_seg[i]
            t_hi = j - 1 < n ? (t_seg[j - 1] + t_seg[j]) / 2 : t_seg[j - 1]
            poly!(ax, [(t_lo, ylim[1]), (t_hi, ylim[1]),
                       (t_hi, ylim[2]), (t_lo, ylim[2])];
                  color=(DETREND_MASK_COLOR, 0.18), strokewidth=0)
            i = j
        else
            i += 1
        end
    end
end

"""Robust y-axis limits for a flux panel. Uses 0.5%/99.5% percentiles
so a few outliers (cosmic rays, edge ramps) don't crush the dynamic
range; pads by 8% of the inter-percentile range above and below."""
function _robust_ylim(y::AbstractVector{<:Real})
    if length(y) < 200
        lo, hi = extrema(y)
    else
        ys = sort(y)
        n = length(ys)
        lo = ys[max(1, round(Int, 0.005 * n))]
        hi = ys[min(n, round(Int, 0.995 * n))]
    end
    pad = 0.08 * (hi - lo)
    pad == 0 && (pad = 1e-6)
    return (lo - pad, hi + pad)
end

"""Draw matplotlib-style "//" break marks at the cut between pane k
and pane k+1, on the BOTTOM spine of BOTH the top panel and the bottom
panel (the x-axis line of each — not the top of either panel). Two
diagonal segments per side (`---// //---`) straddle each anchor so the
spine line appears interrupted by the slashes.

Plotted on `fig.scene` in figure pixel coordinates (sourced reactively
from `ax.xaxis.attributes.endpoints`), following the Makie discourse
recipe — that's the only approach that reliably renders break marks
on top of the axis spine without clipping at axis bounds (regular
`lines!` on an Axis can't extend past the spine even with
`clip_planes = Plane3f[]`).

Reference: discourse.julialang.org/t/break-axis-in-makie/62729
"""
function _add_break_marks!(fig::Figure,
                            axes_top::Vector{<:Axis},
                            axes_bot::Vector{<:Axis}, k::Int)
    angle      = π / 5      # ~36° from horizontal — strong diagonal
    linelength = 14.0       # pixel length of each slash
    sep_px     = 8.0        # pixel gap between the two slashes of one "//"

    segments = lift(
        axes_top[k].xaxis.attributes.endpoints,
        axes_top[k + 1].xaxis.attributes.endpoints,
        axes_bot[k].xaxis.attributes.endpoints,
        axes_bot[k + 1].xaxis.attributes.endpoints,
    ) do tk_eps, tk1_eps, bk_eps, bk1_eps
        # endpoints[1] = left pixel of bottom spine, [2] = right pixel
        anchors = Point2f[tk_eps[2], tk1_eps[1], bk_eps[2], bk1_eps[1]]
        out = Tuple{Point2f, Point2f}[]
        for p in anchors
            # Two parallel slashes around each anchor → "//"
            for x_off in (-sep_px / 2, sep_px / 2)
                c = p + Point2f(x_off, 0)
                a = c + Point2f(cos(angle), sin(angle)) * 0.5 * linelength
                b = c - Point2f(cos(angle), sin(angle)) * 0.5 * linelength
                push!(out, (a, b))
            end
        end
        out
    end

    linesegments!(fig.scene, segments; color = :black, linewidth = 1.8)
end

"""
    plot_transit_phasefold(t, result; sector_id=nothing, sector_names=nothing,
                           n_bins=60, xwin=0.025, min_snr=nothing,
                           output=nothing, fmt=:png, save_pdf=false) -> Figure

Phase-fold the `find_transits` detrended light curve at every significant
candidate period and stack one panel per candidate.

`result` is the NamedTuple from `find_transits` (its candidates are already
the significant ones — `snr ≥ score_threshold`, harmonic-deduped, capped at
`max_candidates`). `t` is the time vector you passed to `find_transits`
(same length as `result.flux_detrended`, which is input-aligned). No saved
LC is needed.

Panels share the x-axis (orbital phase, transit at 0), zoomed to `±xwin`,
and are glued with no vertical gap; a single shared y-label spans them.
Each panel overlays a binned-mean curve, marks the transit-duration window,
and labels its period/SNR in-axis. Robust y-limits keep the dip visible.

# Keyword arguments
- `sector_id` — per-cadence sector grouping (same length as `t`; any
  comparable eltype). When it spans >1 sector, points are colored per sector
  with a shared legend.
- `sector_names` — display labels aligned to the *sorted-unique* `sector_id`
  (e.g. `["S02","S03","S29","S30","S96","S97"]`). If omitted, each sector id
  is shown verbatim — never a fabricated `1..N` index.
- `min_snr` — optional plot-time SNR floor (tighter than the search's
  `score_threshold`), so you can drop weak candidates without re-running.
- `xwin` — half-width of the displayed phase window (default 0.025).
- `n_bins` — bins for the binned-mean overlay across the window.
- `output` (path without extension) saves via `_save_fig`; writes **PNG +
  PDF** by default (`save_pdf=true`), with the dense scatter rasterized so
  the PDF stays small while axes/text remain vector. Set `save_pdf=false`
  for PNG only.
"""
function plot_transit_phasefold(t::AbstractVector{<:Real}, result;
                                sector_id::Union{Nothing,AbstractVector} = nothing,
                                sector_names::Union{Nothing,AbstractVector} = nothing,
                                n_bins::Union{Nothing,Int} = nothing,
                                bin_target::Int = 15,
                                xwin::Real = 0.025,
                                min_snr::Union{Nothing,Real} = nothing,
                                output::Union{Nothing,AbstractString} = nothing,
                                fmt::Symbol = :png,
                                save_pdf::Bool = true)
    flux = result.flux_detrended
    # Per-point (leveled, fractional) photometric errors for the folded data,
    # when find_transits threaded them through; nothing for older results.
    ferr = (hasproperty(result, :flux_err_detrended) &&
            result.flux_err_detrended !== nothing &&
            length(result.flux_err_detrended) == length(flux)) ?
           result.flux_err_detrended : nothing
    length(t) == length(flux) || throw(ArgumentError(
        "t (len $(length(t))) must match result.flux_detrended (len $(length(flux))) " *
        "— pass the same time vector you gave find_transits"))
    sector_id === nothing || length(sector_id) == length(t) || throw(ArgumentError(
        "sector_id (len $(length(sector_id))) must match t (len $(length(t)))"))

    # `find_transits` already filtered to snr ≥ score_threshold; `min_snr`
    # is an optional plot-time tightening (no need to re-run the search).
    sel = min_snr === nothing ? collect(eachindex(result.periods)) :
          [i for i in eachindex(result.periods) if result.snr[i] >= min_snr]
    isempty(sel) && throw(ArgumentError(
        "no candidates to plot (min_snr=$min_snr filtered all $(length(result.periods)))"))
    P    = result.periods[sel]
    t0s  = result.t0s[sel]
    durs = result.durations[sel]
    snr  = result.snr[sel]
    ncand = length(P)

    sectors = sector_id === nothing ? [] : sort!(unique(sector_id))
    multi   = length(sectors) > 1
    # Legend labels: caller-supplied `sector_names` (aligned to sorted-unique
    # sector ids) override; otherwise use each sector id verbatim — never a
    # fabricated 1..N index. So pass the real identifiers ("TESS S02", 29, …).
    sector_names === nothing || length(sector_names) == length(sectors) ||
        throw(ArgumentError("sector_names (len $(length(sector_names))) must match " *
            "the number of unique sectors ($(length(sectors)))"))
    sec_label(si) = sector_names === nothing ? string(sectors[si]) :
                                               string(sector_names[si])
    finite_all = isfinite.(t) .& isfinite.(flux)

    with_theme(nereus_theme()) do
        fig = Figure(size = (900, 230 * ncand + 40))
        axs = Vector{Axis}(undef, ncand)
        for i in 1:ncand
            ax = Axis(fig[i, 1];
                      xlabel = i == ncand ? "Orbital phase" : "")
            axs[i] = ax
            ph = phase_fold(t, P[i], t0s[i])            # ∈ [-0.5, 0.5), transit at 0

            # Zoom to the transit; only render in-window points (dense LCs are
            # an unreadable smear at full phase, and the dip is ±~0.01 wide).
            inwin = finite_all .& (abs.(ph) .<= xwin)
            # Folded data points, visible, WITH per-point photometric error bars
            # (drawn under the markers). bars are faint so a few-hundred-point
            # window reads as a band, not a hairball; markers carry the eye.
            if multi
                for (si, s) in enumerate(sectors)
                    m = inwin .& (sector_id .== s)
                    any(m) || continue
                    ferr === nothing || errorbars!(ax, ph[m], flux[m], ferr[m];
                        color = (inst_color(si), 0.30), whiskerwidth = 0,
                        linewidth = 0.7, rasterize = 2)
                    scatter!(ax, ph[m], flux[m];
                             color = (inst_color(si), 0.6), markersize = 5,
                             strokewidth = 0, rasterize = 2)
                end
            else
                ferr === nothing || errorbars!(ax, ph[inwin], flux[inwin], ferr[inwin];
                    color = (NEREUS_COLORS.pm_marker, 0.30), whiskerwidth = 0,
                    linewidth = 0.7, rasterize = 2)
                scatter!(ax, ph[inwin], flux[inwin];
                         color = (NEREUS_COLORS.pm_marker, 0.6),
                         markersize = 5, strokewidth = 0, rasterize = 2)
            end

            # robust binned overlay over the displayed window: occupancy-scaled
            # bins, median estimator, per-bin error bars (see `bin_phasefold`).
            phf = ph[inwin]
            flf = flux[inwin]
            yb = Float64[]; eb = Float64[]
            if length(phf) >= 24
                xb, yb, eb = bin_phasefold(phf, flf; target = bin_target,
                                           nbins = n_bins, xlo = -xwin, xhi = xwin)
                errorbars!(ax, xb, yb, eb; color = (NEREUS_COLORS.pm_bin, 0.9),
                           whiskerwidth = 6, linewidth = 2)
                scatter!(ax, xb, yb; color = NEREUS_COLORS.pm_bin,
                         markersize = 10, strokewidth = 0.75, strokecolor = :black)
            end

            # transit-duration window: phase half-width = duration / (2P)
            hw = durs[i] / (2 * P[i])
            if isfinite(hw) && hw > 0
                vlines!(ax, [-hw, hw]; color = (:black, 0.6),
                        linestyle = :dash, linewidth = 2)
            end

            # Y-limits show the DATA scatter (now that the points + error bars
            # are visible), so a noise candidate reads as flat instead of being
            # auto-zoomed into a fake wiggle — and always extend to include the
            # binned dip + its error bars.
            if !isempty(flf)
                rlo, rhi = _robust_ylim(flf)
                if length(yb) >= 3
                    rlo = min(rlo, minimum(yb .- eb))
                    rhi = max(rhi, maximum(yb .+ eb))
                end
                ylims!(ax, rlo, rhi)
            end

            # in-axis period (+ SNR) label
            text!(ax, 0.02, 0.05;
                  text = "P = $(round(P[i], digits = 4)) d\nSNR = $(round(snr[i], digits = 1))",
                  align = (:left, :bottom), space = :relative, fontsize = 16)

            xlims!(ax, -xwin, xwin)
            # glue panels: hide x-decorations AND their reserved protrusion
            # space on every panel but the bottom, so rowgap=0 leaves no gap.
            # hide ALL x-decorations (incl. ticks) on upper panels so
            # rowgap=0 truly glues them — protruding ticks were the gap.
            i < ncand && hidexdecorations!(ax; grid = false)
        end
        ncand > 1 && linkxaxes!(axs...)
        if multi
            # explicit opaque swatches (the data scatter is near-transparent)
            elems  = [MarkerElement(color = inst_color(si), marker = :circle,
                                    markersize = 10) for si in eachindex(sectors)]
            labels = [sec_label(si) for si in eachindex(sectors)]
            axislegend(axs[1], elems, labels; position = :rt,
                       framevisible = false, labelsize = 14)
        end
        # single shared y-label spanning all panels (they're identical)
        Label(fig[1:ncand, 0], "Normalized flux"; rotation = π/2, fontsize = 22)
        rowgap!(fig.layout, 0)
        colgap!(fig.layout, 1, 8)
        output !== nothing && _save_fig(fig, String(output), fmt; save_pdf = save_pdf)
        return fig
    end
end
