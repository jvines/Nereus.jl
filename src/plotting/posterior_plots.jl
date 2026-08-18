# EMPEROR-style log-posterior panels: ONE FIGURE PER PARAMETER, every
# posterior sample as a point (param value vs log-posterior), with the
# max-lp value (solid) and posterior mean (dashed) marked and quoted in the
# legend. The skyline shape gives an immediate read of mode location, width,
# skew, multimodality, and railing that summary statistics hide.

using Statistics: mean

"""
    plot_posteriors_lp(chains, params; output=nothing, fmt=:png,
                        save_pdf=false, max_points=150_000,
                        figsize=(900, 420), only::Union{Nothing,Vector{String}}=nothing)

One figure PER SAMPLED PARAMETER (planet, systemic, and noise params alike),
scattering parameter value against `:lp`. For trans-dim chains, planet-block
parameters are conditioned on the samples where that planet is ACTIVE
(`planet_active_k`, falling back to `n_planets ≥ k`); shared params use all
samples. Samples are stride-thinned to `max_points` for rendering. Files go
to `output/models/posteriors/lp_<param>.<fmt>`. `only` restricts to a subset
of parameter names.
"""
function plot_posteriors_lp(chains, params;
                             output::Union{Nothing, String} = nothing,
                             fmt::Symbol = :png,
                             save_pdf::Bool = false,
                             max_points::Int = 150_000,
                             figsize = (900, 420),
                             only::Union{Nothing, Vector{String}} = nothing,
                             planet::Union{Nothing, Int} = nothing)
    cn = Set(names(chains, :parameters))
    (:lp in cn) || throw(ArgumentError(
        "plot_posteriors_lp requires an :lp column in the chains"))
    lp = vec(Array(chains[:lp]))
    finite_idx = findall(isfinite, lp)
    isempty(finite_idx) && throw(ArgumentError("no finite-lp samples"))

    # Map each planet slot to the unfrozen names of its block, so planet
    # params can be conditioned on that planet's active samples.
    layout = params.layout
    owner = Dict{String, Int}()   # param name -> planet slot (absent = shared)
    for (k, blk) in enumerate(layout.planet_blocks)
        for slot in planet_slot_indices(blk)
            uf = findfirst(==(slot), layout.unfrozen_idx)
            uf === nothing && continue
            owner[layout.unfrozen_names[uf]] = k
        end
    end

    # Per-planet active index cache
    act_cache = Dict{Int, Vector{Int}}()
    function active_for(k::Int)
        get!(act_cache, k) do
            idx = finite_idx
            col = Symbol("planet_active_$k")
            if col in cn
                act = vec(Array(chains[col]))
                sub = [i for i in finite_idx if act[i] > 0.5]
                !isempty(sub) && (idx = sub)
            elseif :n_planets in cn
                npv = vec(Array(chains[:n_planets]))
                sub = [i for i in finite_idx if npv[i] >= k]
                !isempty(sub) && (idx = sub)
            end
            idx
        end
    end

    pnames = [nm for nm in layout.unfrozen_names if Symbol(nm) in cn]
    only !== nothing && (pnames = [nm for nm in pnames if nm in only])
    planet !== nothing && (pnames = [nm for nm in pnames if
                                      get(owner, nm, 0) == planet])
    isempty(pnames) && throw(ArgumentError("no matching sampled parameters"))

    outdir = output === nothing ? nothing :
             joinpath(output, "models", "posteriors")
    outdir !== nothing && mkpath(outdir)

    figs = Dict{String, Figure}()
    with_theme(nereus_theme()) do
        for nm in pnames
            k = get(owner, nm, 0)
            idx = k == 0 ? finite_idx : active_for(k)
            if length(idx) > max_points
                idx = idx[round.(Int, range(1, length(idx); length = max_points))]
            end
            v   = vec(Array(chains[Symbol(nm)]))[idx]
            lps = lp[idx]
            ibest = argmax(lps)

            fig = Figure(size = figsize)
            ax = Axis(fig[1, 1]; xlabel = nm, ylabel = "log P")
            scatter!(ax, v, lps;
                      color = (NEREUS_COLORS.pm_marker, 0.35),
                      markersize = 3, strokewidth = 0, rasterize = 2)
            vmax  = v[ibest]
            vmean = mean(v)
            vlines!(ax, [vmax]; color = :black, linewidth = 1.6,
                     label = "max = $(round(vmax, sigdigits = 7))")
            vlines!(ax, [vmean]; color = :black, linestyle = :dash,
                     linewidth = 1.6,
                     label = "mean = $(round(vmean, sigdigits = 7))")
            axislegend(ax; position = _best_legend_corner(v, lps),
                        framevisible = false, labelsize = 12)
            figs[nm] = fig
            if outdir !== nothing
                _save_plot(joinpath(outdir, "lp_$(nm).$fmt"), fig;
                            save_pdf = save_pdf, px_per_unit = 2)
            end
        end
    end
    return figs
end