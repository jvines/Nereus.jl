# transdim_plots.jl — visualize what a trans-dimensional run prefers across
# model dimensions: planet-count occupancy P(Nₚ|D), marginal noise-model
# occupancy P(active|D), and the n_planets trace (dimension-jump mixing).
# The text-only counterpart is print_transdim_summary (reporting.jl).

"""
    plot_transdim_occupancy(chains, params; td=nothing, output=nothing,
                            fmt=:png, save_pdf=false, figsize=(1100, 850)) -> Figure

Three-panel trans-dimensional summary:
  (1) planet-count occupancy   — P(Nₚ = k | D) bars, k = 0 … max_kplanet
  (2) noise-model occupancy    — marginal P(model active | D) for each
      toggleable noise model (needs `td` for the labels)
  (3) Nₚ trace                 — n_planets vs sample, to eyeball whether the
      sampler actually jumps dimensions (vs sticking)

Bars are colored along the `:cool` colormap. No titles (info in axes/labels).

`split=true` (with `output`) writes each panel to its OWN file instead of the
2×2 grid — `<base>_planets`, `_noise`, `_planets_trace`, `_noise_trace` — for
when the noise panel has too many models to read in a shared figure. Returns
the vector of per-panel figures.
"""
function plot_transdim_occupancy(chains, params=nothing;
                                  td::Union{Nothing, TransDimConfig}=nothing,
                                  noise_labels::Union{Nothing, Vector{<:AbstractString}}=nothing,
                                  max_kplanet::Union{Nothing, Int}=nothing,
                                  output::Union{Nothing, String}=nothing,
                                  fmt::Symbol=:png, save_pdf::Bool=false,
                                  split::Bool=false,
                                  figsize=(1100, 850))
    :n_planets in names(chains, :parameters) ||
        throw(ArgumentError("chains has no :n_planets column — not a trans-dim run"))
    lbl(p) = string(round(p; digits=3))
    # noise_active_* columns in index order; label from noise_labels (explicit),
    # else td (_detect_noise_columns), else generic. Lets us plot straight from a
    # saved chain without rebuilding params/td.
    # NB: `Base.split` is qualified — the `split` keyword arg shadows it here.
    _nac = sort([s for s in names(chains, :parameters)
                 if startswith(String(s), "noise_active_")];
                by = s -> parse(Int, Base.split(String(s), "_")[end]))
    ncols = if noise_labels !== nothing
        [_nac[i] => String(i <= length(noise_labels) ? noise_labels[i] : "model $i")
         for i in eachindex(_nac)]
    elseif td !== nothing
        _detect_noise_columns(chains, td)
    else
        [_nac[i] => "model $i" for i in eachindex(_nac)]
    end
    has_noise = !isempty(ncols)
    with_theme(nereus_theme()) do
        # Single Nₚ column when there are no toggleable noise models — don't
        # render empty noise panels.
        fig = Figure(; size = has_noise ? figsize :
                            (round(Int, figsize[1] * 0.52), figsize[2]))
        np = vec(Array(chains[:n_planets]))
        n_total = length(np)
        max_kp = max_kplanet !== nothing ? max_kplanet :
                 params !== nothing ? params.config.max_kplanet : Int(maximum(np))
        # Occupancy bars are colored ALONG the :cool colormap BY THEIR VALUE
        # (Nereus palette; a winning bar lands at the magenta end, low occupancy
        # at the cyan end — consistent across both panels). The noise-trace
        # raster's "on" cell uses the same magenta accent.
        cool_accent = cgrad(NEREUS_CMAP)[1.0]

        # noise-model occupancy panel (shared by both layouts)
        function draw_noise!(ax)
            if isempty(ncols)
                text!(ax, 0.5, 0.5; text="(no toggleable noise models)",
                      align=(:center, :center), space=:relative, color=:gray)
                hidedecorations!(ax); return
            end
            labels = [String(p.second) for p in ncols]
            nprobs = [count(vec(Array(chains[p.first])) .== 1.0) / n_total for p in ncols]
            barplot!(ax, 1:length(ncols), nprobs; color=nprobs,
                     colormap=NEREUS_CMAP, colorrange=(0, 1),
                     strokecolor=:black, strokewidth=1)
            hlines!(ax, 0.5; color=(:gray, 0.7), linestyle=:dash, linewidth=1.5)
            for (i, p) in enumerate(nprobs)
                text!(ax, i, p; text=lbl(p), align=(:center, :bottom),
                      fontsize=14, offset=(0, 3))
            end
            ax.xticks = (1:length(ncols), labels)
            ax.xticklabelrotation = π / 6
            ylims!(ax, 0, 1.10)
        end

        # split=true: emit each panel as its OWN file (declutter) instead of the
        # 2×2 grid. Filenames derive from `output`: <base>_planets, _noise,
        # _planets_trace, _noise_trace <ext>. Returns the vector of figures.
        if split && output !== nothing
            max_kp0 = max_kplanet !== nothing ? max_kplanet :
                      params !== nothing ? params.config.max_kplanet : Int(maximum(np))
            sidx0 = unique(round.(Int, range(1, n_total; length=min(2000, n_total))))
            ks0 = collect(0:max_kp0)
            probs0 = [count(np .== Float64(k)) / n_total for k in ks0]
            base, ext = splitext(output); mkpath(dirname(output))
            figs = Figure[]
            let f = Figure(; size = (560, 430)),
                ax = Axis(f[1, 1]; xlabel="Number of planets  Nₚ",
                          ylabel="P(Nₚ | D)", xticks=0:max_kp0)
                barplot!(ax, ks0, probs0; color=probs0, colormap=NEREUS_CMAP,
                         colorrange=(0,1), strokecolor=:black, strokewidth=1)
                for (k, p) in zip(ks0, probs0)
                    p > 0.005 && text!(ax, k, p; text=lbl(p), align=(:center,:bottom),
                                       fontsize=14, offset=(0,3))
                end
                ylims!(ax, 0, min(1.0, maximum(probs0)*1.18 + 0.02))
                _save_plot(base*"_planets"*ext, f; save_pdf=save_pdf, px_per_unit=2)
                push!(figs, f)
            end
            if has_noise
                let f = Figure(; size = (max(760, 95*length(ncols)), 500)),
                    ax = Axis(f[1, 1]; ylabel="P(model active | D)", xlabel="Noise model")
                    draw_noise!(ax)
                    _save_plot(base*"_noise"*ext, f; save_pdf=save_pdf, px_per_unit=2)
                    push!(figs, f)
                end
            end
            let f = Figure(; size = (820, 380)),
                ax = Axis(f[1, 1]; xlabel="Sample (post-burn-in, thinned)",
                          ylabel="Nₚ", yticks=0:max_kp0)
                lines!(ax, sidx0, np[sidx0]; color=(NEREUS_COLORS.post, 0.7), linewidth=0.8)
                ylims!(ax, -0.3, max_kp0 + 0.3)
                _save_plot(base*"_planets_trace"*ext, f; save_pdf=save_pdf, px_per_unit=2)
                push!(figs, f)
            end
            if has_noise
                let f = Figure(; size = (820, max(380, 46*length(ncols)))),
                    ax = Axis(f[1, 1]; xlabel="Sample (post-burn-in, thinned)",
                              ylabel="Noise model")
                    Mr = zeros(length(sidx0), length(ncols))
                    for (i, p) in enumerate(ncols)
                        Mr[:, i] = (vec(Array(chains[p.first]))[sidx0] .> 0.5)
                    end
                    heatmap!(ax, sidx0, 1:length(ncols), Mr;
                             colormap=cgrad([:white, cool_accent]), colorrange=(0,1))
                    ax.yticks = (1:length(ncols), [String(p.second) for p in ncols])
                    ylims!(ax, 0.5, length(ncols) + 0.5)
                    _save_plot(base*"_noise_trace"*ext, f; save_pdf=save_pdf, px_per_unit=2)
                    push!(figs, f)
                end
            end
            return figs
        end

        # 2×2: occupancy BARS on top (Nₚ | noise), trans-dim TRACES below
        # (Nₚ | noise) — posterior weight and the chain's actual dimension
        # switching read together.
        sidx = unique(round.(Int, range(1, n_total; length=min(2000, n_total))))

        # [1,1] Nₚ occupancy bar
        ax_npb = Axis(fig[1, 1]; xlabel="Number of planets  Nₚ",
                      ylabel="P(Nₚ | D)", xticks=0:max_kp)
        ks = collect(0:max_kp)
        probs = [count(np .== Float64(k)) / n_total for k in ks]
        barplot!(ax_npb, ks, probs; color=probs, colormap=NEREUS_CMAP,
                 colorrange=(0, 1), strokecolor=:black, strokewidth=1)
        for (k, p) in zip(ks, probs)
            p > 0.005 && text!(ax_npb, k, p; text=lbl(p), align=(:center, :bottom),
                               fontsize=14, offset=(0, 3))
        end
        ylims!(ax_npb, 0, min(1.0, maximum(probs) * 1.18 + 0.02))

        # [1,2] noise-model occupancy bar — only when there ARE noise models
        has_noise && draw_noise!(Axis(fig[1, 2];
                     ylabel="P(model active | D)", xlabel="Noise model"))

        # [2,1] Nₚ trace
        ax_npt = Axis(fig[2, 1]; xlabel="Sample (post-burn-in, thinned)",
                      ylabel="Nₚ", yticks=0:max_kp)
        lines!(ax_npt, sidx, np[sidx]; color=(NEREUS_COLORS.post, 0.7), linewidth=0.8)
        ylims!(ax_npt, -0.3, max_kp + 0.3)

        # [2,2] noise-model trace — raster of which toggleable models are active
        # over the run (rows = models, white = off, cool = on): the actual
        # trans-dim noise switching (or a model held on/off throughout).
        if has_noise
            ax_nt = Axis(fig[2, 2]; xlabel="Sample (post-burn-in, thinned)",
                         ylabel="Noise model")
            Mr = zeros(length(sidx), length(ncols))
            for (i, p) in enumerate(ncols)
                Mr[:, i] = (vec(Array(chains[p.first]))[sidx] .> 0.5)
            end
            heatmap!(ax_nt, sidx, 1:length(ncols), Mr;
                     colormap=cgrad([:white, cool_accent]), colorrange=(0, 1))
            ax_nt.yticks = (1:length(ncols), [String(p.second) for p in ncols])
            ylims!(ax_nt, 0.5, length(ncols) + 0.5)
        end
        rowsize!(fig.layout, 1, Auto(1.3))
        rowsize!(fig.layout, 2, Auto(1.0))

        if output !== nothing
            mkpath(dirname(output))
            _save_plot(output, fig; save_pdf=save_pdf, px_per_unit=2)
        end
        return fig
    end
end
