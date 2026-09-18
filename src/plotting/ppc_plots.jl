# Posterior predictive check figure.
#
# Lays out, top-down:
#   1. RV residual time series  (data − mean model), with per-point
#      σ shaded as a band, colored per instrument.
#   2. RV residual GLS periodogram with FAP horizontals (signal you
#      didn't fit if a peak crosses 1%).
#   3. RV residual ACF (correlated noise you didn't model).
#   4. Photometry residual time series + ACF, when applicable.
#
# Channel blocks are omitted when the corresponding data block is
# absent.

using CairoMakie
using Statistics: median
using Printf: @sprintf

"""
    plot_ppc(ppc::PPCResult; filename=nothing, figwidth=900,
             panel_height=200, max_peak_labels=3, show_faps=true,
             fap_cutoff=0.1) -> Figure

Standard PPC diagnostic figure. Pass the result of
[`posterior_predictive_check`](@ref) and either save to disk via
`filename = "ppc.png"` or display the returned Figure.
"""
function plot_ppc(ppc::PPCResult;
                   filename::Union{Nothing, AbstractString} = nothing,
                   save_pdf::Bool = false,
                   figwidth::Int = 900,
                   panel_height::Int = 200,
                   max_peak_labels::Int = 3,
                   show_faps::Bool = true,
                   fap_cutoff::Float64 = 0.1)
    set_theme!(nereus_theme())

    panels = Symbol[]
    ppc.has_rv && push!(panels, :rv_resid)
    if ppc.has_rv && ppc.rv_pgram !== nothing
        push!(panels, :rv_pgram)
    end
    if ppc.has_rv && !isempty(ppc.rv_acf)
        push!(panels, :rv_acf)
    end
    ppc.has_pm && push!(panels, :phot_resid)
    if ppc.has_pm && !isempty(ppc.phot_acf)
        push!(panels, :phot_acf)
    end
    isempty(panels) &&
        throw(ArgumentError("PPCResult has no plottable channels"))

    n_panels = length(panels)
    fig = Figure(; size = (figwidth, panel_height * n_panels + 70),
                   figure_padding = (28, 14, 14, 10))

    row = 1
    for sym in panels
        if sym === :rv_resid
            _ppc_residual_axis!(fig, row, ppc.rv_t, ppc.rv_inst,
                                 ppc.rv_resid_med, ppc.rv_resid_std,
                                 "RV residual [m/s]")
        elseif sym === :rv_pgram
            _ppc_pgram_axis!(fig, row, ppc.rv_pgram::GLSPgram;
                              show_faps = show_faps,
                              max_peak_labels = max_peak_labels,
                              fap_cutoff = fap_cutoff)
        elseif sym === :rv_acf
            _ppc_acf_axis!(fig, row, ppc.rv_acf_lags, ppc.rv_acf,
                            "Lag [d]", "RV residual ACF")
        elseif sym === :phot_resid
            _ppc_residual_axis!(fig, row, ppc.phot_t, ppc.phot_inst,
                                 ppc.phot_resid_med, ppc.phot_resid_std,
                                 "Flux residual")
        elseif sym === :phot_acf
            _ppc_acf_axis!(fig, row, ppc.phot_acf_lags, ppc.phot_acf,
                            "Lag [d]", "Flux residual ACF")
        end
        row += 1
    end

    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end

# ---------------------------------------------------------------------
# Panel renderers
# ---------------------------------------------------------------------

function _ppc_residual_axis!(fig, row::Int,
                              t::Vector{Float64}, inst::Vector{Int},
                              resid::Vector{Float64}, sigma::Vector{Float64},
                              ylabel::AbstractString)
    ax = Axis(fig[row, 1];
               xlabel = "Time (BJD)",
               ylabel = ylabel,
               yticks = Makie.WilkinsonTicks(4))
    hlines!(ax, [0.0]; color = NEREUS_COLORS.zero_line,
             linestyle = :dash, linewidth = 1.0)

    insts = sort!(unique(inst))
    for ii in insts
        mask = inst .== ii
        any(mask) || continue
        col = inst_color(ii)
        errorbars!(ax, t[mask], resid[mask], sigma[mask];
                    color = col, whiskerwidth = 0)
        scatter!(ax, t[mask], resid[mask];
                  color = col, marker = inst_marker(ii),
                  markersize = 7, strokewidth = 0.4,
                  strokecolor = :black)
    end
    return ax
end

function _ppc_pgram_axis!(fig, row::Int, pg::GLSPgram;
                           show_faps::Bool = true,
                           max_peak_labels::Int = 3,
                           fap_cutoff::Float64 = 0.1)
    ax = Axis(fig[row, 1];
               xlabel = L"\log_{10}\left(\mathrm{Frequency}\left(\frac{1}{d}\right)\right)",
               ylabel = "Power",
               yticks = Makie.WilkinsonTicks(3))
    x = log10.(pg.frequencies)
    lines!(ax, x, pg.power; color = NEREUS_COLORS.post, linewidth = 1.2)

    if show_faps && !isempty(pg.fap_thresholds)
        linestyles = (:solid, :dashdot, :dash, :dot)
        fap_colors = (NEREUS_COLORS.pm_marker, NEREUS_COLORS.pm_bin,
                       NEREUS_COLORS.gp, NEREUS_COLORS.kep)
        for (i, (lvl, th)) in enumerate(zip(pg.fap_levels, pg.fap_thresholds))
            hlines!(ax, [th];
                     color = fap_colors[mod1(i, length(fap_colors))],
                     linestyle = linestyles[mod1(i, length(linestyles))],
                     linewidth = 1.2,
                     label = string(round(100 * lvl; digits = 2), "% FAP"))
        end
        axislegend(ax; position = :rt, framevisible = false,
                    labelsize = 14, padding = (4, 4, 2, 2))
    end

    n_labels = 0
    for pk in pg.peaks
        n_labels >= max_peak_labels && break
        pk.fap <= fap_cutoff || continue
        text!(ax, log10(pk.frequency), pk.power;
               text = @sprintf("%.2f d", pk.period),
               align = (:center, :bottom),
               fontsize = 14, color = :black, offset = (0, 4))
        n_labels += 1
    end
    return ax
end

function _ppc_acf_axis!(fig, row::Int,
                         lags::Vector{Float64}, acf::Vector{Float64},
                         xlabel::AbstractString, ylabel::AbstractString)
    ax = Axis(fig[row, 1];
               xlabel = xlabel,
               ylabel = ylabel,
               yticks = Makie.WilkinsonTicks(3))
    hlines!(ax, [0.0]; color = NEREUS_COLORS.zero_line,
             linestyle = :dash, linewidth = 1.0)
    lines!(ax, lags, acf; color = NEREUS_COLORS.post, linewidth = 1.4)
    return ax
end
