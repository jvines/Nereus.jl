using Printf: @sprintf
using LinearAlgebra: Symmetric, cholesky, I
using Statistics: cor, quantile
using Random: MersenneTwister

# RV and RV residuals against the activity indicators, one figure per RV
# instrument. Drawn in the style of the RV phase fold (time-coloured markers,
# nested 1/2/3σ bands, black median line, same fonts and colourbar).

# Display name of an indicator column, and the order the panels come in:
# the classical CCF/spectral indicators first, anything else alphabetically.
const _IND_DISPLAY = Dict("bis" => "BIS", "bis_span" => "BIS", "fwhm" => "FWHM",
                          "contrast" => "Contrast", "s_index" => "S-index",
                          "sindex" => "S-index", "s-index" => "S-index",
                          "halpha" => "Hα", "h_alpha" => "Hα")
const _IND_ORDER = ("bis", "bis_span", "fwhm", "contrast", "s_index", "sindex",
                    "s-index", "halpha", "h_alpha")

_ind_display(name::AbstractString) = get(_IND_DISPLAY, lowercase(name), String(name))

function _ind_sorted(names)
    rank(n) = (i = findfirst(==(lowercase(n)), _IND_ORDER); i === nothing ? length(_IND_ORDER) + 1 : i)
    return sort!(collect(String, names); by = n -> (rank(n), n))
end

"""
    _wls_line(x, y, σ; scale_to_scatter=false) -> (a, b, cov, xm)

Weighted least-squares line `y = a + b (x - xm)` (`xm` the weighted mean of `x`,
which only conditions the solve). With flat priors on `(a, b)` and Gaussian
errors σ the posterior of `(a, b)` is exactly Gaussian with this mean and
covariance `(AᵀWA)⁻¹`, so it needs no sampler.

`scale_to_scatter = true` multiplies the covariance by the reduced χ² of the
line when it exceeds 1 (errors σ scaled by a common factor fitted to the
scatter about the line). The activity-correlation panels use it: their top row
still holds the planets' signal, which no error bar accounts for.
"""
function _wls_line(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                   σ::AbstractVector{<:Real}; scale_to_scatter::Bool = false)
    w = 1 ./ Float64.(σ) .^ 2
    xm = sum(w .* x) / sum(w)
    A = hcat(ones(length(x)), Float64.(x) .- xm)
    M = A' * (w .* A)
    cov = inv(Symmetric(M))
    β = cov * (A' * (w .* y))
    if scale_to_scatter && length(x) > 2
        χ2r = sum(w .* (y .- A * β) .^ 2) / (length(x) - 2)
        cov = cov .* max(χ2r, 1.0)
    end
    return β[1], β[2], Matrix(Symmetric(cov)), xm
end

"""
    _line_bands(x, y, σ, xs; n=4000, rng) -> (q, a, b, sd_b, cov)

The weighted line's credible band on the grid `xs`: draws of `(a, b)` from
the exact Gaussian posterior ([`_wls_line`](@ref)), and the 0.15, 2.5, 16, 50,
84, 97.5, 99.85 percentiles of the drawn lines (rows of `q`, one column per
`xs`): the nested 68/95/99.7% bands and the median.
"""
function _line_bands(x, y, σ, xs; n::Int = 4000, rng = MersenneTwister(7),
                     scale_to_scatter::Bool = false)
    a, b, cov, xm = _wls_line(x, y, σ; scale_to_scatter)
    L = cholesky(Symmetric(cov + 1e-300I)).L
    draws = [[a, b] .+ L * randn(rng, 2) for _ in 1:n]
    lines = [d[1] .+ d[2] .* (xs .- xm) for d in draws]
    probs = (0.0015, 0.025, 0.16, 0.5, 0.84, 0.975, 0.9985)
    q = zeros(length(probs), length(xs))
    for j in eachindex(xs)
        col = [l[j] for l in lines]
        for (i, p) in enumerate(probs)
            q[i, j] = quantile(col, p)
        end
    end
    return q, a, b, sqrt(cov[2, 2]), cov
end

"""
    plot_rv_activity_corr(chains, params, data; output=nothing, fmt=:png,
                          save_pdf=false, figsize=nothing, subtract_gp=true,
                          n_samples=4000) -> Vector{Figure}

Correlation of the RVs with the activity indicators the fit carries
(`data.indicators`), one figure per RV instrument. Columns are indicators; the
top row is the RV (the instrument's fitted γ removed) and the bottom row the RV
residuals after the adopted model. The residuals are computed as the residual
panel of [`plot_rv_phasefold`](@ref) does: data minus the reference draw's full
model (all planets, trend, activity decorrelation) minus the GP mean at the
data times when a GP is active (`subtract_gp`).

Points carry the time colour of the phase fold (`cool`, black edge, a
`BJD - t₀` colourbar) and light-grey RV error bars. In each panel the black
line and the nested 68/95/99.7% bands are the weighted linear fit
`y = a + b·x` weighted by the RV errors with the instrument's fitted jitter
added in quadrature; the `(a, b)` posterior (flat priors, Gaussian errors) is
exactly Gaussian, so it is sampled directly, and its covariance is scaled by
the reduced χ² of the line when that exceeds 1 (the top row carries the
planets' signal). The Pearson `r` of the panel and the slope in units of its
posterior standard deviation, `b/σ_b`, are printed top-left, in the headroom
above the data.

Indicators enter as `Data` holds them (per instrument median-subtracted and
RMS-scaled when `normalize_indicators`, the default), so the x axes are in
those units; `r` is unaffected. A missing (non-finite) indicator value drops
that point from that panel only. An indicator with fewer than 3 finite points
(or no spread) for an instrument gets no column, and an instrument with no
usable indicator gets no figure. Nothing is drawn, and `Figure[]` returned, when
the data carry no indicators. Files: `models/RV_activity_corr_<instrument>.<fmt>`.
"""
function plot_rv_activity_corr(chains, params, data;
                               output::Union{Nothing, String} = nothing,
                               fmt::Symbol = :png,
                               save_pdf::Bool = false,
                               figsize = nothing,
                               bf_cutoff::Real = 10.0,
                               subtract_gp::Bool = true,
                               n_samples::Int = 4000)
    figs = Figure[]
    (n_rv(data) > 0 && !isempty(data.indicators)) || return figs

    with_theme(nereus_theme()) do
        # Reference draw: the winning model's max-lp draw, as the RV fold and
        # time series choose it.
        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = _draws(chains, :n_planets)
            modal_np = _mode_int_local(np_all)
            active_idx = findall(np_all .== modal_np)
            isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
            td_state = winning_td_state(chains, params, active_idx, modal_np)
            active_idx = winning_config_idx(chains, params, active_idx, td_state)
            theta_med = Theta{Float64}(params; td = td_state)
        else
            active_idx = collect(1:_n_flat_draws(chains))
            theta_med = Theta{Float64}(params)
        end
        top_idx = _top_lp_draw_pool(chains; bf_cutoff = bf_cutoff)
        let cut = _sorted_intersect(active_idx, top_idx)
            isempty(cut) || (active_idx = cut)
        end
        set_theta_best_lp!(theta_med, chains, params, active_idx)

        preds, _ = rv_predictions(theta_med, data)
        residuals = data.rv .- preds
        if subtract_gp
            gp_at_data = try
                channel_gp_mean_at(theta_med, residuals, data.rv_err .^ 2,
                                   data.t_rv, data.t_rv, data.rv_inst, :rv; data = data)
            catch err
                @warn "GP mean failed in the activity-correlation plot; residuals keep it" exception = err
                nothing
            end
            gp_at_data === nothing || (residuals = residuals .- gp_at_data)
        end

        inst_names = params.config.instruments.rv_names
        ind_names = _ind_sorted(keys(data.indicators))
        for (ii, ins) in enumerate(inst_names)
            inst_mask = data.rv_inst .== ii
            _is_sb2_data(data) && (inst_mask = inst_mask .& (data.rv_comp .== 1))
            count(inst_mask) >= 3 || continue

            gamma = try
                Float64(get_param(theta_med, "gamma_$ins"))
            catch
                0.0
            end
            rv = data.rv[inst_mask] .- gamma
            res = residuals[inst_mask]
            err = data.rv_err[inst_mask]
            # line weights: RV errors with the instrument's fitted jitter in quadrature
            jit = try
                Float64(get_param(theta_med, "sigma_$ins"))
            catch
                0.0
            end
            werr = sqrt.(err .^ 2 .+ jit^2)
            tt = data.t_rv[inst_mask]

            # one column per indicator with something to correlate
            cols = Tuple{String, Vector{Float64}}[]
            for nm in ind_names
                x = Float64.(data.indicators[nm][inst_mask])
                ok = isfinite.(x) .& isfinite.(rv) .& isfinite.(res) .& isfinite.(err)
                count(ok) >= 3 || continue
                length(unique(x[ok])) >= 2 || continue
                push!(cols, (nm, x))
            end
            isempty(cols) && continue
            ncol = length(cols)

            tmin = minimum(tt); tmax = maximum(tt)
            span_t = tmax > tmin ? tmax - tmin : 1.0
            n_total = length(tt)
            data_alpha = n_total > 1000 ? 0.4 : n_total > 500 ? 0.6 : 0.9
            fsize = figsize === nothing ? (380 * ncol + 220, 700) : Tuple(Int.(round.(collect(figsize))))   # also takes the JSON array of plot_kwargs

            fig = Figure(; size = fsize)
            ga = fig[1, 1] = GridLayout()
            axs = [Axis(ga[r, c]) for r in 1:2, c in 1:ncol]
            for c in 1:ncol
                linkxaxes!(axs[1, c], axs[2, c])
                hidexdecorations!(axs[1, c]; grid = false)
                axs[1, c].xticks = LinearTicks(3); axs[2, c].xticks = LinearTicks(3)
                axs[2, c].xlabel = _ind_display(cols[c][1])
            end
            for r in 1:2
                linkyaxes!(axs[r, :]...)
                for c in 2:ncol
                    hideydecorations!(axs[r, c]; grid = false)
                end
            end
            axs[1, 1].ylabel = rich("RV (m s", superscript("-1"), ")")
            axs[2, 1].ylabel = rich("Residuals (m s", superscript("-1"), ")")
            colgap!(ga, 0); rowgap!(ga, 0)
            rowsize!(ga, 1, Auto(1.6)); rowsize!(ga, 2, Auto(1))

            rng = MersenneTwister(7)
            for (c, (nm, x)) in enumerate(cols)
                for (r, y) in enumerate((rv, res))
                    ax = axs[r, c]
                    ok = isfinite.(x) .& isfinite.(y) .& isfinite.(err)
                    xo = x[ok]; yo = y[ok]; eo = err[ok]; wo = werr[ok]
                    xlo, xhi = extrema(xo)
                    xpad = 0.06 * (xhi - xlo)
                    xs = collect(range(xlo - xpad, xhi + xpad; length = 200))
                    q, _, bsl, sdb, _ = _line_bands(xo, yo, wo, xs; n = n_samples, rng = rng,
                                                    scale_to_scatter = true)
                    band!(ax, xs, q[1, :], q[7, :]; color = (NEREUS_COLORS.ci, CI_ALPHA_3SIGMA))
                    band!(ax, xs, q[2, :], q[6, :]; color = (NEREUS_COLORS.ci, CI_ALPHA_2SIGMA))
                    band!(ax, xs, q[3, :], q[5, :]; color = (NEREUS_COLORS.ci, CI_ALPHA_1SIGMA))
                    lines!(ax, xs, q[4, :]; color = :black, linewidth = 3)
                    r == 2 && hlines!(ax, 0; color = NEREUS_COLORS.zero_line,
                                      linestyle = :dash, linewidth = MODEL_LW)
                    errorbars!(ax, xo, yo, eo; color = (:gray, 0.15), linewidth = 1.0)
                    scatter!(ax, xo, yo; color = tt[ok] .- tmin, colormap = NEREUS_CMAP,
                             colorrange = (0, span_t), marker = :circle, markersize = 13,
                             strokewidth = 1.5, strokecolor = :black, alpha = data_alpha)
                    rho = length(xo) >= 3 ? cor(xo, yo) : NaN
                    rtxt = isfinite(rho) ? replace("r = " * @sprintf("%.2f", rho), "-" => "−") : "r = n/a"
                    isfinite(bsl / sdb) &&
                        (rtxt *= replace(@sprintf(", b/σ = %.1f", bsl / sdb), "-" => "−"))
                    text!(ax, 0.05, 0.95; text = rtxt, space = :relative,
                          align = (:left, :top), fontsize = 20)
                    xlims!(ax, xs[1], xs[end])
                end
            end
            # y limits shared per row, with headroom above the data for the r labels
            for (r, y) in enumerate((rv, res))
                okr = isfinite.(y) .& isfinite.(err)
                lo = minimum(y[okr] .- err[okr]); hi = maximum(y[okr] .+ err[okr])
                r == 2 && (lo = min(lo, 0.0); hi = max(hi, 0.0))
                span = hi > lo ? hi - lo : 1.0
                ylims!(axs[r, 1], lo - 0.08 * span, hi + 0.28 * span)
            end

            Colorbar(ga[1:2, ncol + 1]; colormap = NEREUS_CMAP, limits = (0, span_t),
                     label = "BJD - $(round(Int, tmin))", labelsize = 20, ticklabelsize = 16)

            if output !== nothing
                mkpath(joinpath(output, "models"))
                _save_plot(joinpath(output, "models", "RV_activity_corr_$(ins).$fmt"), fig;
                           save_pdf = save_pdf, px_per_unit = 3)
            end
            push!(figs, fig)
        end
    end
    return figs
end
