# Ladder diagnostics of a parallel-tempered run: what the temperatures, the
# swaps between them and the tempered evidence integrand did. The two figures
# astroEMPEROR draws from a reddemcee run as `rates` and `beta_ladder`, here
# from a `LadderHistory` (src/samplers/ladder_history.jl).

"""The `LadderHistory` of a sampler result, or `nothing` if it carries none."""
_ladder_of(L::LadderHistory) = L
_ladder_of(result) =
    (result !== nothing && hasproperty(result, :ladder) &&
     getproperty(result, :ladder) isa LadderHistory) ? getproperty(result, :ladder) : nothing

"""
    _running_mean(v, window) -> Vector{Float64}

Trailing mean of the last `window` values of `v` (fewer at the start, where
the window is still filling) -- EMPEROR's `running`.
"""
function _running_mean(v::AbstractVector{<:Real}, window::Integer)
    w = max(1, Int(window))
    out = Vector{Float64}(undef, length(v))
    acc = 0.0
    @inbounds for i in eachindex(v)
        acc += v[i]
        i > w && (acc -= v[i - w])
        out[i] = acc / min(i, w)
    end
    return out
end

"""
    plot_ladder_rates(result; output=nothing, fmt=:png, save_pdf=false,
                      figsize=(1000, 900), window=25)

How the temperature ladder behaved over a tempered run, against step on a
logarithmic axis. `result` is what `sample_pt_emcee` or
`sample_transdim_pt_emcee` returned (or its `LadderHistory`).

Three panels, one line per rung, coloured cold to hot:

  - `T` — the temperatures `1/β`. Flat unless `adapt_ladder = true`, and then
    they move only during burn-in.
  - `T_swap` — the fraction of swap proposals accepted between each pair of
    adjacent rungs. A pair that sinks toward zero has stopped exchanging: the
    rungs below it are sampling on their own, and the cold chain can be
    perfectly converged to one mode because of it.
  - `SMD` — the swap mean distance: how far, in prior widths, a swap proposal
    moves a walker on average, a rejected one counting as zero. A pair can
    accept often and still have SMD near zero, when both rungs already hold
    the same states; that pair is carrying nothing down the ladder.

The lower two are drawn for pairs, coloured as the colder rung of the pair.
`window` is a trailing running mean over that many steps (fewer while it
fills, so the first steps are a cumulative mean). A step's swap rate is a
fraction of `n_walkers` proposals and is that noisy; `window = 1` plots every
step as recorded.

Saves `betas/rates.<fmt>`. Returns the `Figure` — an empty one, with nothing
saved, when `result` has no ladder history or fewer than two rungs or steps.
"""
function plot_ladder_rates(result;
                           output::Union{Nothing, String} = nothing,
                           fmt::Symbol = :png,
                           save_pdf::Bool = false,
                           figsize = (1000, 900),
                           window::Integer = 25)
    L = _ladder_of(result)
    L === nothing && return Figure()
    n_steps, n_temps = size(L.betas)
    (n_steps >= 2 && n_temps >= 2) || return Figure()

    with_theme(nereus_theme()) do
        cols  = cool_pastel(n_temps)
        steps = 1:n_steps
        fig = Figure(; size = figsize)
        ax_T = Axis(fig[1, 1]; ylabel = "T", xscale = log10, yscale = log10)
        ax_s = Axis(fig[2, 1]; ylabel = rich("T", subscript("swap")), xscale = log10)
        ax_d = Axis(fig[3, 1]; ylabel = "SMD", xlabel = "Step", xscale = log10)
        linkxaxes!(ax_T, ax_s, ax_d)
        hidexdecorations!(ax_T; ticks = false, minorticks = false)
        hidexdecorations!(ax_s; ticks = false, minorticks = false)

        # Hot first, so the cold rungs -- the ones that are read -- end on top.
        for t in n_temps:-1:1
            T = 1 ./ view(L.betas, :, t)
            all(x -> isfinite(x) && x > 0, T) || continue   # β = 0 has no place on a log axis
            lines!(ax_T, steps, T; color = cols[t], linewidth = 2)
        end
        for t in (n_temps - 1):-1:1
            lines!(ax_s, steps, _running_mean(view(L.swap_rate, :, t), window);
                   color = (cols[t], 0.85), linewidth = 1.5)
            lines!(ax_d, steps, _running_mean(view(L.swap_distance, :, t), window);
                   color = (cols[t], 0.85), linewidth = 1.5)
        end
        xlims!(ax_d, 1, n_steps)

        if output !== nothing
            mkpath(joinpath(output, "betas"))
            _save_plot(joinpath(output, "betas", "rates.$fmt"), fig;
                       save_pdf = save_pdf, px_per_unit = 3)
        end
        fig
    end
end

"""The tempered evidence report a result carries, under either field name."""
function _evidence_report_of(result)
    for f in (:evidence, :evidence_report)
        hasproperty(result, f) || continue
        rep = getproperty(result, f)
        hasproperty(rep, :ti_plus) && return rep
    end
    return nothing
end

"""Which estimator a result's `log_evidence` is, for the annotation."""
function _reported_evidence_name(result, rep)
    hasproperty(result, :log_evidence) || return nothing
    lz = getproperty(result, :log_evidence)
    (lz isa Real && isfinite(lz)) || return nothing
    for (f, name) in ((:log_evidence_bridge, "bridge"),
                      (:log_evidence_laplace, "mode-Laplace"))
        hasproperty(result, f) && getproperty(result, f) == lz && return name
    end
    if rep !== nothing
        for (f, name) in ((:hybrid, "H+"), (:ti_plus, "TI+"), (:ti, "TI"))
            getproperty(rep, f)[1] == lz && return name
        end
    end
    return ""
end

"""
    plot_beta_ladder(result; output=nothing, fmt=:png, save_pdf=false,
                     figsize=(900, 700))

The thermodynamic-integration integrand of a tempered run: `⟨log L⟩` at each
rung against its `β`, one point per rung coloured cold to hot. The area
between the curve and zero, shaded, is `log Z` by thermodynamic integration.

A horizontal bar through a point spans the `β` that rung held after burn-in;
there is none when the ladder was fixed, which it is unless the run adapted
past burn-in.

The annotation gives the curvature-aware integral of the plotted curve (TI+,
with its Richardson error estimate; plain TI if TI+ is not finite), and below
it the evidence the run REPORTS when that is a different number -- for
`sample_pt_emcee` usually the bridge estimate. A large gap between the two is
the phase-transition signature: `⟨log L⟩` at the hot rungs is biased, and so
is every estimator that integrates this curve.

Saves `betas/beta_ladder.<fmt>`. Returns the `Figure` — an empty one, with
nothing saved, when `result` has no ladder history or fewer than two rungs
with a finite `⟨log L⟩` (a run with no post-burn-in steps).
"""
function plot_beta_ladder(result;
                          output::Union{Nothing, String} = nothing,
                          fmt::Symbol = :png,
                          save_pdf::Bool = false,
                          figsize = (900, 700))
    L = _ladder_of(result)
    L === nothing && return Figure()
    n_steps, n_temps = size(L.betas)
    n_steps >= 1 || return Figure()
    β = L.betas[end, :]
    keep = findall(isfinite, L.mean_logL)
    length(keep) >= 2 || return Figure()

    with_theme(nereus_theme()) do
        cols = cool_pastel(n_temps)
        fig = Figure(; size = figsize)
        ax = Axis(fig[1, 1]; xlabel = "β", xticks = 0:0.2:1,
                  ylabel = rich("E[log ", rich("L"; font = :italic), "]",
                                subscript("β")))

        ord = keep[sortperm(β[keep])]
        x, y = β[ord], L.mean_logL[ord]
        band!(ax, x, min.(y, 0.0), max.(y, 0.0); color = (:black, 0.22))
        lines!(ax, x, y; color = (:black, 0.45), linewidth = 1.5)

        # The β each rung held while the integrand was being accumulated.
        post = (L.n_burnin + 1):n_steps
        if !isempty(post)
            seg = Point2f[]; segcol = eltype(cols)[]
            for t in keep
                lo, hi = extrema(view(L.betas, post, t))
                hi > lo || continue
                push!(seg, Point2f(lo, L.mean_logL[t]), Point2f(hi, L.mean_logL[t]))
                push!(segcol, cols[t], cols[t])
            end
            isempty(seg) || linesegments!(ax, seg; color = segcol, linewidth = 3)
        end
        scatter!(ax, β[keep], L.mean_logL[keep]; color = cols[keep],
                 markersize = 16, strokewidth = 1.0, strokecolor = :black)

        span = max(maximum(y) - minimum(y), 1e-9)
        xlims!(ax, -0.02, 1.02)      # whole markers at β → 0 and β = 1
        ylims!(ax, minimum(y) - 0.05 * span, maximum(y) + 0.05 * span)

        rep = _evidence_report_of(result)
        txt = String[]
        shown = NaN
        if rep !== nothing
            if isfinite(rep.ti_plus[1])
                shown = rep.ti_plus[1]
                push!(txt, @sprintf("Evidence (TI+): %.3f ± %.3f", shown, rep.ti_plus[2]))
            elseif isfinite(rep.ti[1])
                shown = rep.ti[1]
                push!(txt, @sprintf("Evidence (TI): %.3f", shown))
            end
        end
        name = _reported_evidence_name(result, rep)
        if name !== nothing && getproperty(result, :log_evidence) != shown
            push!(txt, @sprintf("Reported%s: %.3f", isempty(name) ? "" : " ($name)",
                                getproperty(result, :log_evidence)))
        end
        isempty(txt) || text!(ax, 0.97, 0.04; text = replace(join(txt, "\n"), "-" => "−"),
                              space = :relative, align = (:right, :bottom),
                              fontsize = 18)

        if output !== nothing
            mkpath(joinpath(output, "betas"))
            _save_plot(joinpath(output, "betas", "beta_ladder.$fmt"), fig;
                       save_pdf = save_pdf, px_per_unit = 3)
        end
        fig
    end
end
