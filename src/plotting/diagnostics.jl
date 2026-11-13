# Diagnostic plots: trace, posteriors, histograms, corner.

using Statistics: mean, std, median, median!, quantile, quantile!
using Distributions: fit, Normal
using PairPlots                 # top-level load avoids world-age in plot_corner
using PairPlots: pairplot       # pull the entry-point into local scope

"""
    plot_trace(chains, params;
                output=nothing, fmt=:png, figsize=FIG_TRACE, max_points=1500)

MCMC trace plots for all unfrozen parameters.
One figure per parameter with chain value vs iteration; returns them by name.

Every line is thinned to at most `max_points` points: every k-th iteration,
plotted at its true iteration number. 1500 is about one per pixel across the
1200-pixel figure, as in `plot_traces_grouped`. A figure of several walkers is
rasterized, so a `save_pdf` copy does not carry every walker as vector paths.
Unthinned, a pt_emcee run of 100 walkers x 30000 steps drew 3M vertices per
figure: the NGTS-33 global fit died partway through its trace set, and a
7-parameter synthetic chain of that shape took 23.9 s and 2.2 GB (4.4 s and
0.9 GB thinned).
"""
function plot_trace(chains, params;
                     output::Union{Nothing, String}=nothing,
                     fmt::Symbol=:png,
                     save_pdf::Bool=false,
                     figsize=FIG_TRACE,
                     max_points::Int=1500)
    figs = Dict{String, Figure}()
    with_theme(nereus_theme()) do
        chain_names = Set(names(chains, :parameters))

        for name in reported_param_names(params)
            sym = Symbol(name)
            sym in chain_names || continue

            samp_arr = _iter_chain_view(chains, sym)   # (n_iter, n_chain), in place
            fig = Figure(; size=figsize)
            ax = Axis(fig[1, 1];
                        xlabel="Iteration", ylabel=name)
            # Multi-chain: plot each chain as a separate thin line so the
            # eye can spot stuck walkers / mode-hopping; every walker is kept,
            # each thinned, and the lines are drawn as one bitmap. Single-chain
            # (or flat) case: one solid line, thinned the same way, so the trace
            # shows the macro trajectory rather than a black blob.
            if ndims(samp_arr) == 2 && size(samp_arr, 2) > 1
                n_iter, n_chain = size(samp_arr)
                xs = 1:cld(n_iter, max_points):n_iter
                α = clamp(0.9 / sqrt(n_chain), 0.15, 0.85)
                for c in 1:n_chain
                    lines!(ax, xs, samp_arr[xs, c];
                            color=(:black, α), linewidth=0.6, rasterize=2)
                end
            else
                samp = vec(samp_arr)
                xs = 1:cld(length(samp), max_points):length(samp)
                lines!(ax, xs, samp[xs];
                        color=(:black, 0.85), linewidth=1.0)
            end

            if output !== nothing
                mkpath(joinpath(output, "traces"))
                _save_plot(joinpath(output, "traces", "$name.$fmt"), fig;
                            save_pdf=save_pdf)
            end
            _keep_fig!(figs, name, fig)
        end
    end
    return figs
end


"""
    plot_histograms(chains, params;
                     output=nothing, fmt=:png, figsize=FIG_HIST,
                     n_bins=30)

Parameter posterior histograms with Gaussian fit and statistics.
One figure per parameter.
"""
function plot_histograms(chains, params;
                          output::Union{Nothing, String}=nothing,
                          fmt::Symbol=:png,
                          save_pdf::Bool=false,
                          figsize=FIG_HIST,
                          n_bins::Int=30)
    with_theme(nereus_theme()) do
        chain_names = Set(names(chains, :parameters))

        for name in reported_param_names(params)
            sym = Symbol(name)
            sym in chain_names || continue

            samp = _draws(chains, sym)
            # Skip degenerate (all-equal) columns — Makie's hist! can't
            # normalize a single-bin histogram and crashes downstream
            # with "reducing over an empty collection". Common on
            # nested-sampling chains where some params land at a prior
            # bound and the top-N-by-weight fallback gives a constant.
            samp_lo, samp_hi = extrema(samp)
            if samp_hi - samp_lo < 1e-12
                @info "plot_histograms: skipping '$name' — chain has " *
                      "no variance (constant at $(round(samp_lo; digits=6)))"
                continue
            end
            fig = Figure(; size=figsize)
            ax = Axis(fig[1, 1]; xlabel=name, ylabel="Density")

            hist!(ax, samp; bins=n_bins, normalization=:pdf,
                   color=(NEREUS_COLORS.hist_face, HIST_ALPHA),
                   strokewidth=0.5, strokecolor=:black)

            # Gaussian fit overlay
            d = fit(Normal, samp)
            x_fit = range(minimum(samp), maximum(samp); length=200)
            y_fit = [pdf(d, x) for x in x_fit]
            lines!(ax, collect(x_fit), y_fit;
                    color=NEREUS_COLORS.post, linewidth=3)

            # Stats text box
            # One scratch copy for the three order statistics, not one each.
            buf = copy(samp)
            med = median!(buf)
            q16, q84 = quantile!(buf, [0.16, 0.84])
            buf = nothing
            stats_text = "median = $(round(med; digits=4))\n" *
                         "+$(round(q84 - med; digits=4))\n" *
                         "-$(round(med - q16; digits=4))"
            text!(ax, 0.95, 0.95; text=stats_text,
                   align=(:right, :top), space=:relative,
                   fontsize=16)

            if output !== nothing
                mkpath(joinpath(output, "histograms"))
                _save_plot(joinpath(output, "histograms", "$name.$fmt"), fig;
                            save_pdf=save_pdf)
            end
        end
    end
end


# A parameter's draws against log-posterior, as a binned density on a log count
# scale in `cool` -- the corner's 2-D panels draw theirs the same way. As a
# scatter they needed an alpha low enough that the core did not saturate, and
# at that alpha (0.01 above 10k draws) the tails, which are single draws,
# vanished: a dim core over what looked like empty space. Non-finite
# log-posteriors are left out; a parameter whose draws do not vary has no width
# to bin, so it stays a column of points.
function _lnp_density!(ax, x::AbstractVector, y::AbstractVector)
    ok = isfinite.(x) .& isfinite.(y)
    x, y = x[ok], y[ok]
    isempty(x) && return nothing
    if minimum(x) == maximum(x) || minimum(y) == maximum(y)
        return scatter!(ax, x, y; color = NEREUS_COLORS.post, markersize = 4,
                        strokewidth = 0)
    end
    # One weighted point per hexagon: the same figure (`_hexbin_reduce`), without
    # the figure holding every draw.
    red = _hexbin_reduce(x, y, (100, 50))
    red === nothing &&
        return hexbin!(ax, x, y; bins = (100, 50), colormap = :cool, colorscale = log10)
    return hexbin!(ax, red[1]; weights = red[2], bins = (100, 50), colormap = :cool,
                   colorscale = log10)
end

"""
    plot_posteriors(chains, params;
                     output=nothing, fmt=:png, figsize=FIG_POST, max_points=200_000)

Parameter value vs log-posterior (EMPEROR style), drawn as a density
(`_lnp_density!`). Two versions per parameter: full posterior + inference
region. Returns the full-posterior figures by parameter name.

The draws are thinned to at most `max_points`, every k-th draw, the same draws
for every parameter. Unthinned, 100 walkers x 47000 steps put 4.7M points in
each of 32 figures and took 25 minutes; at a few hundred thousand draws the
density looks the same.
"""
function plot_posteriors(chains, params;
                          output::Union{Nothing, String}=nothing,
                          fmt::Symbol=:png,
                          save_pdf::Bool=false,
                          figsize=FIG_POST,
                          max_points::Int=200_000)
    figs = Dict{String, Any}()
    with_theme(nereus_theme()) do
        chain_names = Set(names(chains, :parameters))
        has_lp = :lp in chain_names
        if !has_lp
            @warn "plot_posteriors: chain has no `:lp` column — skipping. " *
                  "(Pigeons PT does not emit log-posterior per draw; " *
                  "use sample_pt_emcee or sample_nuts to get :lp.)"
            return
        end

        lp_all = _draws_view(chains, :lp)     # in place: only the kept draws are copied
        lp_max = -Inf
        for v in lp_all
            isfinite(v) && v > lp_max && (lp_max = v)
        end
        keep = 1:cld(length(lp_all), max_points):length(lp_all)
        lp = lp_all[keep]

        for name in reported_param_names(params)
            sym = Symbol(name)
            sym in chain_names || continue
            samp = _draws_view(chains, sym)[keep]   # the kept draws only

            # Full posterior
            fig = Figure(; size=figsize)
            ax = Axis(fig[1, 1]; xlabel=name, ylabel="Posterior")
            _lnp_density!(ax, samp, lp)
            _keep_fig!(figs, name, fig)
            if output !== nothing
                mkpath(joinpath(output, "posteriors"))
                _save_plot(joinpath(output, "posteriors", "$name.$fmt"), fig;
                            save_pdf=save_pdf, px_per_unit=3)
            end

            # Inference region (EMPEROR: within 2*ln(150) of max)
            cherry_mask = (lp_max .- lp) .< 2 * log(150)
            if count(cherry_mask) > 10
                fig2 = Figure(; size=figsize)
                ax2 = Axis(fig2[1, 1]; xlabel=name, ylabel="Posterior")
                _lnp_density!(ax2, samp[cherry_mask], lp[cherry_mask])
                if output !== nothing
                    _save_plot(joinpath(output, "posteriors",
                                "inference_$name.$fmt"), fig2;
                                save_pdf=save_pdf, px_per_unit=3)
                end
            end
        end
    end
    return figs
end


"""
    _CompactHexBin(; kwargs...)

`PairPlots.HexBin`, drawn from one weighted point per occupied hexagon instead
of every draw. The figure is the same: the same hexagons, counts, colours and
drawing order (see `_hexbin_reduce`), from the same bin counts.

A Makie `hexbin` keeps its input points for as long as the figure lives, one
`Point2d` per draw per panel: 3 MB for each of the 253 panels of a 23-parameter,
200,000-draw corner, 0.8 GB held until the PNG is written. Reduced, a panel
holds a few hundred points. `nbins` keeps each column's bin count, which
PairPlots computes from the column's effective sample size twice per PANEL, an
FFT over all the draws each time.
"""
struct _CompactHexBin <: PairPlots.VizTypeBody
    kwargs
    nbins::Dict{Symbol, Int}
end
_CompactHexBin(; kwargs...) = _CompactHexBin(kwargs, Dict{Symbol, Int}())

# One axis's hexbin bin count, as PairPlots' `bodyplot(::HexBin)` sets it: from
# `series.bins` under `key` (the ROW's name for x, as PairPlots has it), else the
# `bins` keyword, else from the effective sample size of the column `col` drawn
# on that axis.
function _hexbin_nbins(viz::_CompactHexBin, series, key, col, data)
    haskey(series.bins, key) && series.bins[key] isa Number && return series.bins[key]
    haskey(viz.kwargs, :bins) && return viz.kwargs[:bins]
    haskey(series.kwargs, :bins) && return series.kwargs[:bins]
    return get!(viz.nbins, Symbol(col)) do
        n_ess = PairPlots.MCMCDiagnosticTools.ess(data)
        isfinite(n_ess) || (n_ess = length(data))
        max(7, ceil(Int, 1.8log2(n_ess)) + 1)
    end
end

"""
    _hexbin_reduce(x, y, bins) -> (points, weights) or nothing

The draws `(x, y)` as Makie's `hexbin(x, y; bins)` bins them, reduced to one
point per occupied hexagon -- the first draw that falls in it, weighted by the
hexagon's count -- followed by the draws at the four extremes, weighted zero.

`hexbin(points; weights, bins)` then draws what `hexbin(x, y; bins)` draws:

  - the extremes give the same data limits, and so the same hexagon grid;
  - each hexagon gets its count back, as the same `Float64`;
  - the hexagons come out in the same order: Makie collects them in a `Dict`
    keyed by hexagon, and a `Dict`'s layout follows the order in which its keys
    first arrive, which the first-draw order keeps. The extremes arrive last,
    in hexagons already present, and add zero.

The binning is Makie's own (`_spacings_offsets_nbins`, `_nearest_center` and the
distance test of `plot!(::Hexbin)`), so it bins as the hexbin it replaces; the
corner tests compare the two. `nothing` when a coordinate is not finite, for the
caller to draw the draws themselves.
"""
function _hexbin_reduce(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, bins)
    n = length(x)
    (n == 0 || n != length(y)) && return nothing
    (all(isfinite, x) && all(isfinite, y)) || return nothing
    # The recipe's limits, from the extreme points.
    ixlo, ixhi = argmin(x), argmax(x)
    iylo, iyhi = argmin(y), argmax(y)
    r = Makie.Rect2(Makie.Point2d[(x[ixlo], y[iylo]), (x[ixhi], y[iyhi])])
    low = prevfloat.(minimum(r))
    high = nextfloat.(maximum(r))
    lims = Makie.Rect2d(low, high - low)
    spacing, offset, _ = Makie._spacings_offsets_nbins(bins, nothing, lims)
    sz = spacing .* Makie._hexbin_size_fact()
    yweight = sz[1] / sz[2]

    slot = Dict{NTuple{2, Int}, Int}()
    first_draw = Int[]
    count = Float64[]
    @inbounds for i in 1:n
        txy = Makie.Point2d(x[i], y[i])
        tx, ty = txy
        (nx, ny), (nxs, nys), (dvx, dvy) = Makie._nearest_center(txy, spacing, offset)
        d1 = (tx - nx)^2 + (yweight * (ty - ny))^2
        d2 = (tx - nxs)^2 + (yweight * (ty - nys))^2
        id = d1 < d2 ? (cld(dvx, 2), ifelse(iseven(dvy), dvy, dvy + 1)) :
                       (fld(dvx, 2), ifelse(iseven(dvy), dvy + 1, dvy))
        k = get(slot, id, 0)
        if k == 0
            push!(first_draw, i)
            push!(count, 1.0)
            slot[id] = length(first_draw)
        else
            count[k] += 1.0
        end
    end
    idx = vcat(first_draw, [ixlo, ixhi, iylo, iyhi])
    points = [Makie.Point2d(x[i], y[i]) for i in idx]
    return points, vcat(count, zeros(4))
end

function PairPlots.bodyplot(ax::Makie.Axis, viz::_CompactHexBin,
                            series::PairPlots.AbstractSeries, colname_row, colname_col)
    cn = PairPlots.columnnames(series)
    (colname_row in cn && colname_col in cn) || return nothing
    X = PairPlots.getcolumn(series, colname_col)
    Y = PairPlots.getcolumn(series, colname_row)
    try
        bins = (_hexbin_nbins(viz, series, colname_row, colname_col, X),
                _hexbin_nbins(viz, series, colname_col, colname_row, Y))
        kw = PairPlots.remove_attrs(Makie.hexbin; series.kwargs..., viz.kwargs...)
        red = _hexbin_reduce(X, Y, bins)
        if red === nothing
            Makie.hexbin!(ax, X, Y; colormap = Makie.cgrad([:transparent, :black]),
                          bins, kw...)
        else
            Makie.hexbin!(ax, red[1]; weights = red[2],
                          colormap = Makie.cgrad([:transparent, :black]), bins, kw...)
        end
    catch err
        err isa InexactError || rethrow(err)
        @warn "InexactError encountered in `hexbin`" colname_row colname_col
    end
    return nothing
end

"""
    _CornerContour(; sigmas = 0.5:0.5:2, bandwidth = 1.0, kwargs...)

`PairPlots.Contour`, with each column's KDE bandwidth computed once per figure.
PairPlots computes it in every panel for both axes, and each time it is an
effective sample size and a quantile copy over all the drawn draws: 19 of the
23 MB a panel allocated at 200,000 draws, 4.8 GB of garbage over a
23-parameter corner. The contours are PairPlots' own (`_corner_contours`).
"""
struct _CornerContour <: PairPlots.VizTypeBody
    sigmas
    bandwidth
    kwargs
    bw::Dict{Symbol, Float64}
end
_CornerContour(; sigmas = 0.5:0.5:2, bandwidth = 1.0, kwargs...) =
    _CornerContour(sigmas, bandwidth, kwargs, Dict{Symbol, Float64}())

# PairPlots' `prep_contours`, line for line, with the bandwidths of
# `default_bandwidth_ess` looked up in `viz.bw`.
function _corner_contours(viz::_CornerContour, series, colname_row, colname_col;
                          bandwidth = 1.0)
    PP = PairPlots
    xdat = PP.ustrip(PP.disallowmissing(PP.getcolumn(series, colname_col)))
    ydat = PP.ustrip(PP.disallowmissing(PP.getcolumn(series, colname_row)))
    bw(col, d) = get!(() -> PP.default_bandwidth_ess(d), viz.bw, Symbol(col))
    k = PP.KernelDensity.kde((xdat, ydat),
                             bandwidth = (bw(colname_col, xdat), bw(colname_row, ydat)) .* bandwidth)
    ik = PP.KernelDensity.InterpKDE(k)

    exx = extrema(xdat)
    exy = extrema(ydat)
    N = 100
    x = range(first(exx), last(exx), length = N)
    y = range(first(exy), last(exy), length = N)
    h = PP.pdf.(Ref(ik), x, y')

    levels = 1 .- exp.(-0.5 .* (1 ./ viz.sigmas) .^ 2)
    ii = sortperm(reshape(h, :))
    h2flat = h[ii]
    sm = cumsum(h2flat)
    sm /= sm[end]
    if all(isnan, sm) || length(h) <= 1
        @warn "Could not compute valid contours between $colname_col and $colname_row"
        V = [0]
    else
        V = sort(map(v0 -> h2flat[sm .≤ v0][end], levels))
        any(==(0), diff(V)) && @warn "Too few points to create valid contours"
    end

    pad_x = [first(x) - step(x); x; last(x) + step(x)]
    pad_y = [first(y) - step(y); y; last(y) + step(y)]
    pad_h = [
        zeros(size(h, 2))' 0 0
        zeros(size(h, 1)) h zeros(size(h, 1))
        zeros(size(h, 2))' 0 0
    ]
    return PP.ContourLib.contours(pad_x, pad_y, pad_h, V)
end

function PairPlots.bodyplot(ax::Makie.Axis, viz::_CornerContour, series,
                            colname_row, colname_col)
    cn = PairPlots.columnnames(series)
    (colname_row in cn && colname_col in cn) || return nothing
    bandwidth = viz.bandwidth
    if bandwidth isa AbstractDict
        bandwidth = get(bandwidth, (colname_row, colname_col), 1.0)
    end
    c = _corner_contours(viz, series, colname_row, colname_col; bandwidth)
    CL = PairPlots.ContourLib
    for level in CL.levels(c), poly in CL.lines(level)
        xs, ys = CL.coordinates(poly)
        Makie.lines!(ax, xs, ys;
                     PairPlots.remove_attrs(Makie.lines; series.kwargs..., viz.kwargs...)...)
    end
    return nothing
end


# A corner panel's limits from the full sample: its finite extrema padded 5%, as
# Makie's autolimits pad. No limits (`(;)`) for a constant or empty column. `f`
# is the increasing map the panel draws the column through (`rad2deg` for an
# angle), applied to the extrema rather than to a copy of the column.
function _corner_lims(x::AbstractVector, f = identity)
    lo, hi = Inf, -Inf                      # finite extrema, without a filtered copy
    for v in x
        isfinite(v) || continue
        v < lo && (lo = v)
        v > hi && (hi = v)
    end
    hi > lo || return (;)
    lo, hi = f(lo), f(hi)
    pad = 0.05 * (hi - lo)
    return (; lims = (; low = lo - pad, high = hi + pad))
end

"""
    plot_corner(chains, params;
                 output=nothing, fmt=:png,
                 params_to_plot=nothing, density=:hexbin, max_draws=200_000,
                 draws=nothing, stem="corner")

Corner plot using PairPlots.jl, of an explicit selection of parameters.
`params_to_plot`: optional list of parameter name strings to include
(default: every reported parameter -- the all-parameter corner, which costs
n² panels; `run_job` draws `plot_corners` instead). Each axis is labelled with
the parameter's symbol and unit, and an angle is drawn in degrees, as its label
says.

`draws` restricts the corner to those flat draws (indices into the chain's
draws, iterations fastest): `plot_corners` passes the draws in which a
trans-dimensional planet slot is live. Written to `output/<stem>.<fmt>`.

`density` picks how the 2-D panels show the sample cloud under the contours:

- `:hexbin` (default) — a binned density over EVERY draw.
- `:scatter` — one marker per draw, as this plot used to do unconditionally.

Both draw the same contours from the same full sample; only the cloud underneath
differs. The default is `:hexbin` because `:scatter` does not scale: CairoMakie
writes each marker as its own element in the PNG, so a 150,000-draw, 8-parameter
corner is 4.2 million path elements and the SAVE alone measured 16.4 s against
1.3 s to render it — 17.7 s in total, the single most expensive figure Nereus
produces. The hexbin draws the same 150,000 draws in 3.8 s.

`max_draws` caps what is drawn: above it, every k-th draw is plotted, and the
panel limits are set from the FULL sample (padded 5%, as Makie's own autolimits
are), because limits that follow a thinned subset move. The cost is linear in the
draws in both layers (at 3,000,000 draws of 7 parameters the hexbin alone took
11.8 s and the contours 14.8 s; both at 200,000 took 1.7 s), and the 37-parameter
NGTS-33 corner of 3,000,000 draws took 37 minutes. 200,000 draws give the same
contours; what thins out is the outermost hexes, the few drawn by a handful of
draws (on a heavy-tailed test parameter the cloud reached about 100 with every
draw and about 75 thinned, inside the same pinned limits). Pass a larger
`max_draws`, or `typemax(Int)`, to keep them. Below the cap nothing is thinned.

Thinning was measured and rejected as the fix for `:scatter` at 10,000 of
150,000 draws: the cloud all but vanishes at the same alpha, and the panel limits
moved with the thinned extrema (now pinned, as above). `rasterize` is not the fix
either — `PairPlots.Scatter` drops the keyword, and setting it on the plot
objects after `pairplot` returns produces a byte-identical PNG.
"""
function plot_corner(chains, params;
                      output::Union{Nothing, String}=nothing,
                      fmt::Symbol=:png,
                      save_pdf::Bool=false,
                      params_to_plot::Union{Nothing, Vector{String}}=nothing,
                      density::Symbol=:hexbin,
                      max_draws::Int=200_000,
                      draws::Union{Nothing, AbstractVector{<:Integer}}=nothing,
                      stem::AbstractString="corner")
    # PairPlots is loaded at the top of this file (see `using PairPlots`
    # there). The earlier pattern used `@eval using PairPlots` inside the
    # function body, which triggered Julia's world-age limitation:
    # PairPlots types newly brought into scope by `@eval` are not visible
    # to the current function frame, so `PairPlots.Scatter(...)` below
    # would error with `MethodError: ... method too new to be called from
    # this world context`. Loading PairPlots at the module level avoids
    # this entirely. `Base.invokelatest` on the inner call adds a final
    # safety net for any downstream extension methods registered
    # post-precompile (e.g. PairPlots-MCMCChainsExt).
    with_theme(nereus_theme()) do
        chain_names = Set(names(chains, :parameters))
        pnames = params_to_plot !== nothing ? params_to_plot :
                 [n for n in reported_param_names(params)
                  if Symbol(n) in chain_names]

        # A NamedTuple-of-vectors keyed by parameter name; the axis labels
        # (symbol and unit, `_sci_label`) go in PairPlots' `labels` Dict.
        # Each column is read in place (`_draws_view`): only the thinned draws
        # are copied, and the limits come from the full column without a copy of
        # it. Copied whole, the columns were the chain over again -- 1 GB for a
        # 23-parameter, 5.8M-draw fit -- held for the whole figure.
        # `draws` selects the sample (a view, still no copy); an angle is drawn
        # in degrees, as its label says.
        cols = [(c = _draws_view(chains, Symbol(name));
                 draws === nothing ? c : view(c, draws)) for name in pnames]
        to_unit = [last(sci_param_unit(name, params)) ? rad2deg : identity
                   for name in pnames]
        n_draws = isempty(cols) ? 0 : length(first(cols))
        keep = 1:max(1, cld(n_draws, max_draws)):n_draws     # every k-th draw
        syms = Tuple(Symbol.(pnames))
        data_nt = NamedTuple{syms}(Tuple(f.(view(c, keep)) for (c, f) in zip(cols, to_unit)))
        # Thinned: pin each panel to the full sample's range (see docstring).
        pin = length(keep) < n_draws ?
            (; axis = NamedTuple{syms}(Tuple(
                _corner_lims(c, f) for (c, f) in zip(cols, to_unit)))) : (;)
        labels = Dict{Symbol, Any}(Symbol(name) => _sci_label(name, params)
                                   for name in pnames)

        density in (:hexbin, :scatter) || throw(ArgumentError(
            "plot_corner: density must be :hexbin or :scatter, got $(repr(density))"))
        # `cool` per the house plotting convention, on a log count scale so a
        # thousand-fold density range does not render as two flat colours — the
        # tails of a well-sampled posterior hold single-digit counts while the
        # core holds thousands, and on a linear scale everything outside the core
        # saturates to one shade.
        layers = density === :hexbin ?
            (_CompactHexBin(colormap = :cool, colorscale = log10),   # = PairPlots.HexBin
             _CornerContour(color = :black)) :     # = PairPlots.Contour
            (PairPlots.Scatter(markersize = 1, color = (NEREUS_COLORS.post, 0.1)),
             _CornerContour())
        fig = Base.invokelatest(pairplot, data_nt => layers; labels, pin...)

        if output !== nothing
            mkpath(output)
            _save_plot(joinpath(output, "$stem.$fmt"), fig;
                        save_pdf=save_pdf)
        end
        return fig
    end
end
