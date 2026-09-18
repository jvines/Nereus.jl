# Diagnostic plots: trace, posteriors, histograms, corner.

using Statistics: mean, std, median, quantile
using Distributions: fit, Normal
using PairPlots                 # top-level load avoids world-age in plot_corner
using PairPlots: pairplot       # pull the entry-point into local scope

"""
    plot_trace(chains, params;
                output=nothing, fmt=:png, figsize=FIG_TRACE)

MCMC trace plots for all unfrozen parameters.
One figure per parameter with chain value vs iteration.
"""
function plot_trace(chains, params;
                     output::Union{Nothing, String}=nothing,
                     fmt::Symbol=:png,
                     save_pdf::Bool=false,
                     figsize=FIG_TRACE)
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
            # eye can spot stuck walkers / mode-hopping. Single-chain
            # (or flat) case: plot as one solid line. Ensemble samplers
            # (ptemcee) stack walkers into a single long row vector with
            # no chain axis — for those we thin to ≤3000 points so the
            # trace shows the macro trajectory rather than a black blob.
            if ndims(samp_arr) == 2 && size(samp_arr, 2) > 1
                n_iter, n_chain = size(samp_arr)
                α = clamp(0.9 / sqrt(n_chain), 0.15, 0.85)
                for c in 1:n_chain
                    lines!(ax, 1:n_iter, view(samp_arr, :, c);
                            color=(:black, α), linewidth=0.6)
                end
            else
                samp = vec(samp_arr)
                if length(samp) > 3000
                    stride = cld(length(samp), 3000)
                    xs = collect(1:stride:length(samp))
                    samp = samp[xs]
                else
                    xs = collect(1:length(samp))
                end
                lines!(ax, xs, samp;
                        color=(:black, 0.85), linewidth=1.0)
            end

            if output !== nothing
                mkpath(joinpath(output, "traces"))
                _save_plot(joinpath(output, "traces", "$name.$fmt"), fig;
                            save_pdf=save_pdf)
            end
        end
    end
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


"""
    plot_posteriors(chains, params;
                     output=nothing, fmt=:png, figsize=FIG_POST)

Posterior scatter plots: parameter value vs log-posterior (EMPEROR style).
Two versions per parameter: full posterior + inference region.
"""
function plot_posteriors(chains, params;
                          output::Union{Nothing, String}=nothing,
                          fmt::Symbol=:png,
                          save_pdf::Bool=false,
                          figsize=FIG_POST)
    with_theme(nereus_theme()) do
        chain_names = Set(names(chains, :parameters))
        has_lp = :lp in chain_names
        if !has_lp
            @warn "plot_posteriors: chain has no `:lp` column — skipping. " *
                  "(Pigeons PT does not emit log-posterior per draw; " *
                  "use sample_ptemcee or sample_nuts to get :lp.)"
            return
        end

        lp = vec(Array(chains[:lp]))
        lp_max = maximum(lp[isfinite.(lp)])

        # EMPEROR: alpha=0.3, or 0.01 if >10k unique posteriors
        alpha = length(unique(lp)) > 10000 ? 0.01 : 0.3

        for name in params.layout.unfrozen_names
            sym = Symbol(name)
            sym in chain_names || continue
            samp = vec(Array(chains[sym]))

            # Full posterior
            fig = Figure(; size=figsize)
            ax = Axis(fig[1, 1]; xlabel=name, ylabel="Posterior")
            scatter!(ax, samp, lp;
                      color=(NEREUS_COLORS.post, alpha),
                      markersize=4, strokewidth=0)
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
                scatter!(ax2, samp[cherry_mask], lp[cherry_mask];
                          color=(NEREUS_COLORS.post, alpha),
                          markersize=4, strokewidth=0)
                if output !== nothing
                    _save_plot(joinpath(output, "posteriors",
                                "inference_$name.$fmt"), fig2;
                                save_pdf=save_pdf, px_per_unit=3)
                end
            end
        end
    end
end


"""
    plot_corner(chains, params;
                 output=nothing, fmt=:png,
                 params_to_plot=nothing)

Corner plot using PairPlots.jl.
`params_to_plot`: optional list of parameter name strings to include
(default: all unfrozen params).
"""
function plot_corner(chains, params;
                      output::Union{Nothing, String}=nothing,
                      fmt::Symbol=:png,
                      save_pdf::Bool=false,
                      params_to_plot::Union{Nothing, Vector{String}}=nothing)
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
        data_nt = NamedTuple{Tuple(Symbol.(pnames))}(
            (vec(Array(chains[Symbol(name)])) for name in pnames))

        fig = Base.invokelatest(pairplot,
            data_nt => (PairPlots.Scatter(markersize=1, color=(NEREUS_COLORS.post, 0.1)),
                        PairPlots.Contour()))

        if output !== nothing
            mkpath(output)
            _save_plot(joinpath(output, "corner.$fmt"), fig;
                        save_pdf=save_pdf)
        end
        return fig
    end
end
