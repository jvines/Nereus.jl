# Paper-figure plot of the latent activity process G(t) inferred by
# an `ActivityGP` fit. Top panel: posterior over G(t) (mean line +
# 16/84 band) on a dense prediction grid. Optional bottom panel:
# RV observations with the inferred activity contribution
# (Vc·G + Vr·dG/dt) overplotted, so the user can read the
# decomposition off the figure.

using CairoMakie

# 1/2/3σ credible bands from a (n_grid × n_draws) draw matrix, filtering
# non-finite draws per row (failed GP solves come back as NaN columns). Same
# quantile language as the RV phase-fold so the AGP bands read identically.
function _bands123(M::AbstractMatrix{<:Real})
    n = size(M, 1)
    mn  = Vector{Float64}(undef, n)
    lo1 = similar(mn); hi1 = similar(mn); lo2 = similar(mn)
    hi2 = similar(mn); lo3 = similar(mn); hi3 = similar(mn)
    for j in 1:n
        finite = filter(isfinite, view(M, j, :))
        if isempty(finite)
            mn[j] = lo1[j] = hi1[j] = lo2[j] = hi2[j] = lo3[j] = hi3[j] = NaN
        else
            mn[j]  = mean(finite)
            lo1[j] = quantile(finite, 0.1587); hi1[j] = quantile(finite, 0.8413)
            lo2[j] = quantile(finite, 0.0228); hi2[j] = quantile(finite, 0.9772)
            lo3[j] = quantile(finite, 0.0013); hi3[j] = quantile(finite, 0.9987)
        end
    end
    return (; mn, lo1, hi1, lo2, hi2, lo3, hi3)
end

# Stacked 3σ→1σ credible band, matching the RV phase-fold alphas.
function _band_stack!(ax, x, b)
    band!(ax, x, b.lo3, b.hi3; color = (NEREUS_COLORS.ci, CI_ALPHA_3SIGMA))
    band!(ax, x, b.lo2, b.hi2; color = (NEREUS_COLORS.ci, CI_ALPHA_2SIGMA))
    band!(ax, x, b.lo1, b.hi1; color = (NEREUS_COLORS.ci, CI_ALPHA_1SIGMA))
end

"""
    plot_activity_gp_latent(chains, params, data;
                              filename=nothing, n_draws=100,
                              t_pred=nothing, figsize=(900, 400),
                              rng=Random.MersenneTwister(0)) -> Figure

Latent G(t) paper figure. Calls [`activity_gp_predict`](@ref) under
the hood and draws the standard band + mean line. Saves to
`filename` if given.

Throws `ArgumentError` if no `ActivityGP` is in the params noise
model list.
"""
function plot_activity_gp_latent(chains, params::Params, data::Data;
                                   filename::Union{Nothing, AbstractString} = nothing,
                                   save_pdf::Bool = false,
                                   n_draws::Int = 100,
                                   t_pred::Union{Nothing, AbstractVector{<:Real}} = nothing,
                                   figsize::Tuple{Int, Int} = (900, 400),
                                   rng::Random.AbstractRNG = Random.MersenneTwister(0))
    set_theme!(nereus_theme())
    pred = activity_gp_predict(chains, params, data;
                                t_pred = t_pred, n_draws = n_draws, rng = rng)

    fig = Figure(; size = figsize, figure_padding = (28, 14, 14, 10))
    ax = Axis(fig[1, 1];
               xlabel = "Time (BJD)",
               ylabel = "Latent G(t)",
               yticks = Makie.WilkinsonTicks(4))
    _band_stack!(ax, pred.t_pred, _bands123(pred.G_samples))
    lines!(ax, pred.t_pred, pred.G_mean;
             color = NEREUS_COLORS.post, linewidth = 2)
    hlines!(ax, [0.0]; color = NEREUS_COLORS.zero_line,
             linestyle = :dash, linewidth = 1.0)
    xlims!(ax, minimum(pred.t_pred), maximum(pred.t_pred))
    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end

"""
    plot_activity_gp_decomposition(chains, params, data;
                                     filename=nothing, n_draws=100,
                                     figsize=(900, 700),
                                     rng=Random.MersenneTwister(0)) -> Figure

Two-panel RV activity decomposition figure:

- **Top**: original RV data per-instrument scatter + inferred
  activity contribution `Vc·G(t) + Vr·dG/dt(t)` posterior band
  (calls [`activity_gp_decompose_rv`](@ref)).
- **Bottom**: activity-corrected RV residual `rv − ⟨activity⟩` with
  the corrected uncertainty (quadrature sum of measurement and
  activity uncertainties). What's left should look like noise +
  any Keplerian signal that survives the activity removal.
"""
function plot_activity_gp_decomposition(chains, params::Params, data::Data;
                                          filename::Union{Nothing, AbstractString} = nothing,
                                          save_pdf::Bool = false,
                                          n_draws::Int = 100,
                                          figsize::Tuple{Int, Int} = (900, 700),
                                          rng::Random.AbstractRNG = Random.MersenneTwister(0))
    set_theme!(nereus_theme())
    decomp = activity_gp_decompose_rv(chains, params, data;
                                        n_draws = n_draws, rng = rng)

    fig = Figure(; size = figsize, figure_padding = (28, 14, 14, 10))
    ax_top = Axis(fig[1, 1];
                   xlabel = "",
                   ylabel = "RV (m/s)",
                   yticks = Makie.WilkinsonTicks(4))
    ax_bot = Axis(fig[2, 1];
                   xlabel = "Time (BJD)",
                   ylabel = "Activity-corrected RV (m/s)",
                   yticks = Makie.WilkinsonTicks(4))
    linkxaxes!(ax_top, ax_bot)

    insts = sort!(unique(decomp.rv_inst))
    for ii in insts
        mask = decomp.rv_inst .== ii
        scatter!(ax_top, decomp.t_rv[mask], decomp.rv_data[mask];
                   color = inst_color(ii), marker = inst_marker(ii),
                   markersize = 7, strokewidth = 0.4, strokecolor = :black)
        scatter!(ax_bot, decomp.t_rv[mask], decomp.rv_corrected[mask];
                   color = inst_color(ii), marker = inst_marker(ii),
                   markersize = 7, strokewidth = 0.4, strokecolor = :black)
        errorbars!(ax_bot, decomp.t_rv[mask], decomp.rv_corrected[mask],
                     decomp.rv_corrected_err[mask];
                     color = inst_color(ii), whiskerwidth = 0)
    end

    # Activity band on top panel (re-sample on a smooth dense grid). Propagate
    # the RV-activity contribution PER DRAW — act_d(t) = Vc_d·G_d(t) + Vr_d·Ġ_d(t)
    # — then take quantiles, rather than multiplying mean Vc/Vr into the G band
    # (the quantile of a product ≠ product of quantiles).
    pred = activity_gp_predict(chains, params, data; n_draws = n_draws,
                                 rng = rng)
    act = similar(pred.G_samples)
    @inbounds for d in axes(pred.G_samples, 2)
        act[:, d] .= pred.Vc_samples[d] .* view(pred.G_samples, :, d) .+
                     pred.Vr_samples[d] .* view(pred.dG_dt_samples, :, d)
    end
    actb = _bands123(act)
    _band_stack!(ax_top, pred.t_pred, actb)
    lines!(ax_top, pred.t_pred, actb.mn;
             color = NEREUS_COLORS.post, linewidth = 2)

    hlines!(ax_bot, [0.0]; color = NEREUS_COLORS.zero_line,
              linestyle = :dash, linewidth = 1.0)

    rowsize!(fig.layout, 1, Relative(0.55))
    hidexdecorations!(ax_top; grid = false, ticks = false)

    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end
