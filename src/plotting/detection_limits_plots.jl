# Detection-limit figure: log-log K_lim(P) with the "detectable above"
# region shaded. Optional overlay of the actual posterior on (P, K)
# from the same chain for context.

using CairoMakie
using Printf: @sprintf

"""
    plot_detection_limits(res::DetectionLimitResult;
                           filename=nothing, figsize=(900, 500),
                           K_min=nothing, K_max=nothing,
                           color=NEREUS_COLORS.post,
                           shade=true) -> Figure

Plot `K_lim(P)` from a `DetectionLimitResult`. Log-log axes, with the
"detectable" region above the curve shaded (when `shade = true`).
NaN bins (insufficient samples) are skipped so the curve breaks
visibly across coverage gaps.
"""
function plot_detection_limits(res::DetectionLimitResult;
                                filename::Union{Nothing, AbstractString} = nothing,
                                save_pdf::Bool = false,
                                figsize::Tuple{Int, Int} = (900, 500),
                                K_min::Union{Nothing, Real} = nothing,
                                K_max::Union{Nothing, Real} = nothing,
                                color = NEREUS_COLORS.post,
                                shade::Bool = true)
    set_theme!(nereus_theme())
    fig = Figure(; size = figsize, figure_padding = (24, 16, 14, 10))
    ax = Axis(fig[1, 1];
               xscale = log10,
               yscale = log10,
               xlabel = "Period (d)",
               ylabel = @sprintf("K upper limit (%g%% credible) (m/s)",
                                  100 * res.confidence),
               yticks = Makie.WilkinsonTicks(5))

    valid = isfinite.(res.K_limit) .& (res.K_limit .> 0)
    P_v   = res.P_centers[valid]
    K_v   = res.K_limit[valid]

    if isempty(K_v)
        @warn "plot_detection_limits: no valid bins to plot"
        filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
        return fig
    end

    K_lo_axis = K_min === nothing ? max(minimum(K_v) * 0.5, 1e-3) : Float64(K_min)
    K_hi_axis = K_max === nothing ? maximum(K_v) * 2.0 : Float64(K_max)

    # Shaded "detectable above" region — band from K_v up to K_hi_axis.
    if shade
        band!(ax, P_v, K_v, fill(K_hi_axis, length(K_v));
               color = (color, 0.15))
    end

    lines!(ax, P_v, K_v; color = color, linewidth = 2.0)
    scatter!(ax, P_v, K_v; color = color, markersize = 6,
              strokewidth = 0.4, strokecolor = :black)

    # Mark bins with marginal sample count for caller awareness.
    sparse = (res.n_in_bin .< 50) .& valid
    if any(sparse)
        scatter!(ax, res.P_centers[sparse], res.K_limit[sparse];
                  color = NEREUS_COLORS.kep, marker = :xcross,
                  markersize = 10, strokewidth = 0)
    end

    ylims!(ax, K_lo_axis, K_hi_axis)
    xlims!(ax, minimum(P_v) * 0.9, maximum(P_v) * 1.1)

    filename === nothing || _save_plot(filename, fig; save_pdf=save_pdf)
    return fig
end
