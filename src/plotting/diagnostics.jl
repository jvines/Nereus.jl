# Diagnostic plots: trace, posteriors, histograms, corner.

using Statistics: mean, std, median, quantile
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

        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue

            samp_arr = Array(chains[sym])  # (n_iter, n_chain) or (n_iter,)
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
            figs[name] = fig
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

        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue

            samp = vec(Array(chains[sym]))
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
            med = median(samp)
            q16 = quantile(samp, 0.16)
            q84 = quantile(samp, 0.84)
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
    return hexbin!(ax, x, y; bins = (100, 50), colormap = :cool, colorscale = log10)
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

        lp_all = vec(Array(chains[:lp]))
        lp_max = maximum(lp_all[isfinite.(lp_all)])
        keep = 1:cld(length(lp_all), max_points):length(lp_all)
        lp = lp_all[keep]

        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue
            samp = vec(Array(chains[sym]))[keep]

            # Full posterior
            fig = Figure(; size=figsize)
            ax = Axis(fig[1, 1]; xlabel=name, ylabel="Posterior")
            _lnp_density!(ax, samp, lp)
            figs[name] = fig
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
    plot_corner(chains, params;
                 output=nothing, fmt=:png,
                 params_to_plot=nothing, density=:hexbin, max_draws=200_000)

Corner plot using PairPlots.jl.
`params_to_plot`: optional list of parameter name strings to include
(default: all unfrozen params).

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
# A corner panel's limits from the full sample: its finite extrema padded 5%, as
# Makie's autolimits pad. No limits (`(;)`) for a constant or empty column.
function _corner_lims(x::AbstractVector)
    f = filter(isfinite, x)
    isempty(f) && return (;)
    lo, hi = extrema(f)
    hi > lo || return (;)
    pad = 0.05 * (hi - lo)
    return (; lims = (; low = lo - pad, high = hi + pad))
end

function plot_corner(chains, params;
                      output::Union{Nothing, String}=nothing,
                      fmt::Symbol=:png,
                      save_pdf::Bool=false,
                      params_to_plot::Union{Nothing, Vector{String}}=nothing,
                      density::Symbol=:hexbin,
                      max_draws::Int=200_000)
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
                 [n for n in params.layout.unfrozen_names
                  if Symbol(n) in chain_names]

        # Build a NamedTuple-of-vectors so PairPlots picks axis labels
        # straight from the column names. (The `labels=Vector{String}`
        # kwarg form was deprecated; current API wants Dict{Symbol,…} —
        # avoid the issue entirely by going through a named column source.)
        cols = [vec(Array(chains[Symbol(name)])) for name in pnames]
        n_draws = isempty(cols) ? 0 : length(first(cols))
        keep = 1:max(1, cld(n_draws, max_draws)):n_draws     # every k-th draw
        data_nt = NamedTuple{Tuple(Symbol.(pnames))}(Tuple(c[keep] for c in cols))
        # Thinned: pin each panel to the full sample's range (see docstring).
        pin = length(keep) < n_draws ?
            (; axis = NamedTuple{Tuple(Symbol.(pnames))}(Tuple(
                _corner_lims(c) for c in cols))) : (;)

        density in (:hexbin, :scatter) || throw(ArgumentError(
            "plot_corner: density must be :hexbin or :scatter, got $(repr(density))"))
        # `cool` per the house plotting convention, on a log count scale so a
        # thousand-fold density range does not render as two flat colours — the
        # tails of a well-sampled posterior hold single-digit counts while the
        # core holds thousands, and on a linear scale everything outside the core
        # saturates to one shade.
        layers = density === :hexbin ?
            (PairPlots.HexBin(colormap = :cool, colorscale = log10),
             PairPlots.Contour(color = :black)) :
            (PairPlots.Scatter(markersize = 1, color = (NEREUS_COLORS.post, 0.1)),
             PairPlots.Contour())
        fig = Base.invokelatest(pairplot, data_nt => layers; pin...)

        if output !== nothing
            mkpath(output)
            _save_plot(joinpath(output, "corner.$fmt"), fig;
                        save_pdf=save_pdf)
        end
        return fig
    end
end
