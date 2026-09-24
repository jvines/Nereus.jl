# Joint RV + astrometry plots: sky-plane orbit, HGCA PM residuals, joint
# RV/astrometry phase-fold. Visual sanity checks for fits combining
# `RelAstromData`, `HGCAData`, and `Data.t_rv`.
#
# Conventions:
#   - Sky-plane axes: x = ΔRA·cos δ (mas), y = Δδ (mas).
#     Standard astronomy convention is north up, east left, so the
#     RA axis is *flipped* (RA increases to the LEFT).
#   - Posterior fans: `n_draws` random samples drawn from the chain;
#     each draw gets its own orbit traced over one full period.
#   - The host star sits at the origin of every sky-plane plot.
#
# Internal: chain → orbit conversion uses `_planet_orbit(theta, k,
# M_pri, plx, t_ref)` which is parametrization-aware (handles
# :K_driven, :M_sec_driven, :a_driven mass modes and :Mo/:Tp/:Tc
# time anchors). Building a Theta from a chain row is the same
# pattern used in `utils.jl::compute_ci_bands`.

using Statistics: mean, median, std, quantile
using LinearAlgebra: eigen, Symmetric, cholesky, issuccess
import LinearAlgebra

# =====================================================================
# Internal helpers
# =====================================================================

"""
    _theta_from_chain_row(chains, params, idx; tdc=_td_cols(chains, params)) -> Theta

Build a `Theta{Float64}` populated from row `idx` of `chains`. Frozen
slots are auto-populated by the Theta constructor; only chain-present
parameters are set. On a trans-dim chain the row's active set is attached
(see `_row_td_state`); pass a precomputed `tdc` when calling in a loop.
"""
function _theta_from_chain_row(chains, params, idx::Int;
                               tdc  = _td_cols(chains, params),
                               cols = _chain_cols(chains, params))
    theta = Theta{Float64}(params; td = _row_td_state(tdc, params, idx))
    @inbounds for (name, col) in cols
        set_param!(theta, name, col[idx])
    end
    return theta
end

"""
    _chain_cols(chains, params) -> Vector{Tuple{String,Vector{Float64}}}

Every unfrozen parameter's flat column, materialised ONCE.

`chains[sym]` is an AxisArray view, so `vec(Array(chains[sym]))` COPIES the whole
column. Doing that inside `_theta_from_chain_row` — which is called once per
DRAW — made every band plot O(n_draws × n_samples) instead of O(n_draws): on the
HD 114762 posterior that is 13 columns × 446,250 Float64 rebuilt per draw, 90.7
MiB and 3.66 ms of the 3.80 ms each draw cost (96%). One `plot_rv_phasefold` at
its shipped `n_draws = 60_000` spent 440 s and ~5 TB of transient allocation
getting there.

Hoist it beside `tdc` and pass both when calling in a loop. The default argument
keeps a bare single-row call working exactly as before.
"""
function _chain_cols(chains, params)
    chain_names = Set(names(chains, :parameters))
    cols = Tuple{String,Vector{Float64}}[]
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in chain_names || continue
        push!(cols, (name, vec(Array(chains[sym]))))
    end
    return cols
end

"""
    _theta_median(chains, params; active_idx=nothing) -> Theta

Posterior-median Theta. If `active_idx` is given, takes the median over
that subset only. Only the fallback of `_theta_best_lp` for a chain with no
`:lp`: per-parameter medians are not a model on the posterior ridge.
On a trans-dim chain the subset's modal active pattern is attached
(`_row_td_state`).
"""
function _theta_median(chains, params; active_idx = nothing)
    chain_names = Set(names(chains, :parameters))
    rows = active_idx === nothing ? (1:_n_flat_draws(chains)) : active_idx
    theta = Theta{Float64}(params;
                           td = _row_td_state(_td_cols(chains, params), params, rows))
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in chain_names || continue
        vals = vec(Array(chains[sym]))
        if active_idx !== nothing
            vals = vals[active_idx]
        end
        set_param!(theta, name, median(vals))
    end
    return theta
end

"""
    _best_lp_row(chains, idxs) -> Int or nothing

The draw with the highest finite `:lp` among `idxs` (all draws when
`nothing`); `nothing` when the chain has no `:lp` or none is finite.
"""
function _best_lp_row(chains, idxs = nothing)
    (:lp in Set(names(chains, :parameters))) || return nothing
    lp = vec(Array(chains[:lp]))
    cand = filter(i -> isfinite(lp[i]), idxs === nothing ? (1:length(lp)) : idxs)
    isempty(cand) && return nothing
    return cand[argmax(@view lp[cand])]
end

"""
    _theta_best_lp(chains, params; active_idx=nothing) -> Theta

Theta at the MAXIMUM-log-posterior sample (within `active_idx` when
given), with that draw's trans-dim active set. This is the EMPEROR best-fit
convention and the right CENTRAL CURVE for overlays and residuals: it is a
real, smooth model on the posterior ridge. The per-parameter
marginal-median theta is NOT (circular Ω/Mo and the sesinw/secosw mapping
put it off-ridge — its curve sat ~90 mas off the HD 159062 fan, and on
Gaia-4 the IAD χ²/N is 1.65 there against 1.38 at max lp), and a
pointwise predictive median is an order statistic that kinks wherever the
draw curves cross. Falls back to `_theta_median` when the chain carries no
finite `:lp`.
"""
function _theta_best_lp(chains, params; active_idx = nothing)
    best = _best_lp_row(chains, active_idx)
    best === nothing && return _theta_median(chains, params; active_idx = active_idx)
    return _theta_from_chain_row(chains, params, best)
end

"""
    _has_astrom_orbit(params, k) -> Bool

Whether slot `k` carries an astrometric orbit (inc, Ω). A per-companion
astrometry figure has nothing to draw for any other slot -- an RV-only
companion threw inside `planet_inc`, or duplicated the K1 figure.
"""
_has_astrom_orbit(params, k::Int) =
    1 <= k <= length(params.layout.planet_blocks) && has_AS(params.layout.planet_blocks[k])

"""
    _planet_draw(chains, params, planet_idx; bf_cutoff) -> (theta, pool) or nothing

The draw a per-companion figure pictures, and the pool its posterior fan
samples: the draws in which companion `planet_idx` EXISTS, cut to the
best-fit cluster (`_top_lp_draw_pool`), and the max-lp one of them with its
active set attached. `nothing` when a trans-dim chain never has the
companion active -- the figure has nothing to show.

A companion with low occupancy can miss the global best-fit cluster; the
pool then stays the draws in which it exists, rather than falling back to
all draws, most of which would carry it parked.
"""
function _planet_draw(chains, params, planet_idx::Int; bf_cutoff::Real)
    tdc = _td_cols(chains, params)
    present = _planet_present_idx(chains, params, planet_idx; tdc = tdc)
    isempty(present) && return nothing
    pool = intersect(present, _top_lp_draw_pool(chains; bf_cutoff = bf_cutoff))
    isempty(pool) && (pool = present)
    return _theta_best_lp(chains, params; active_idx = pool), pool
end

"""
    _orbit_period_days(theta, k) -> P_days

Convenience: orbital period of planet `k` in days, parametrization-
aware (handles `:a_driven` mode where the period is derived from a).
"""
@inline _orbit_period_days(theta::Theta, k::Int) = planet_P(theta, k)

"""
    _trace_orbit(orb, t_start, P; n_pts=400) -> (Δra, Δdec)

Sample the companion's sky-plane offset over one full orbital period
starting at `t_start` (MJD). Returns vectors of (mas, mas).
"""
function _trace_orbit(orb, t_start::Real, P::Real; n_pts::Int = 400)
    ts = collect(range(t_start, t_start + P; length = n_pts))
    Δra  = Vector{Float64}(undef, n_pts)
    Δdec = Vector{Float64}(undef, n_pts)
    @inbounds for (i, t) in enumerate(ts)
        r, d = relastrom_offset(orb, t)
        Δra[i]  = r
        Δdec[i] = d
    end
    return Δra, Δdec, ts
end

"""
    _orbit_phase_at(t, t_peri, P) -> phase ∈ [0, 1)

Phase of the orbit at time t, with phase 0 at periastron. Wraps to
[0, 1). Used to colour-code RV epochs by where the companion is in
its orbit.
"""
@inline function _orbit_phase_at(t::Real, t_peri::Real, P::Real)
    return mod((t - t_peri) / P, 1.0)
end

"""
    _flip_xaxis!(ax)

Flip the x-axis so values increase to the left (standard RA convention
for sky-plane plots).
"""
function _flip_xaxis!(ax)
    ax.xreversed = true
    return nothing
end


# =====================================================================
# 1. Sky-plane orbit overlay
# =====================================================================

"""
    plot_orbit_skyplane(chains, params, data;
                        n_draws=100, planet_idx=1,
                        output=nothing, fmt=:png, figsize=(900, 900))

Sky-plane (ΔRA·cos δ, Δδ) overlay of the relative-astrometry data with
the best-fit (max-lp) orbit + a faint fan of `n_draws` posterior-sampled
orbits.

The host star sits at the origin (star glyph, which no other element of the
figure uses); periastron of the best-fit orbit is a red diamond. The x-axis is
reversed so RA increases to the LEFT (north up, east left convention).

Returns the `Figure`.
"""
function plot_orbit_skyplane(chains, params, data;
                              n_draws::Int = 0,        # no fan by default
                              planet_idx::Int = 1,
                              output::Union{Nothing, String} = nothing,
                              fmt::Symbol = :png,
                              save_pdf::Bool = false,
                              figsize = (900, 900),
                              bf_cutoff::Real = 5.0)
    relast = data.relastrom
    _has_astrom_orbit(params, planet_idx) || return Figure()   # no astrometric orbit

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ga = fig[1, 1] = GridLayout()
        ax = Axis(ga[1, 1];
                   xlabel = rich("ΔRA·cos δ (mas)"),
                   ylabel = "Δδ (mas)",
                   aspect = DataAspect())
        _flip_xaxis!(ax)

        # The best-fit draw and the fan pool: draws in which this companion
        # exists, in the EMPEROR best-fit cluster (see `_planet_draw`). A
        # trans-dim slot that is never active has no orbit to draw.
        drawn = _planet_draw(chains, params, planet_idx; bf_cutoff = bf_cutoff)
        drawn === nothing && return Figure()
        theta_best, active_idx = drawn
        tdc = _td_cols(chains, params)
        cols = _chain_cols(chains, params)   # ONCE, not once per draw

        n_samples = length(active_idx)

        # ---- 1. Posterior fan -------------------------------------
        ndraw = min(n_draws, n_samples)
        if ndraw > 0
            draw_idx = active_idx[rand(1:n_samples, ndraw)]
            for i in draw_idx
                theta = _theta_from_chain_row(chains, params, i; tdc = tdc, cols = cols)
                P = _orbit_period_days(theta, planet_idx)
                (isfinite(P) && P > 0) || continue
                M_pri = astrom_M_pri(theta)
                plx   = astrom_plx(theta)
                try
                    orb, _ = Nereus._planet_orbit(theta, planet_idx,
                                                    M_pri, plx, data.t_ref)
                    Δra, Δdec, _ = _trace_orbit(orb, data.t_ref, P;
                                                  n_pts = 250)
                    lines!(ax, Δra, Δdec;
                            color = (NEREUS_COLORS.ci, 0.15),
                            linewidth = 0.8)
                catch
                    # Skip degenerate samples (e>1 etc.)
                end
            end
        end

        # ---- 2. Best-fit orbit (max-lp draw; EMPEROR convention) ----
        theta_med = theta_best
        P_med = _orbit_period_days(theta_med, planet_idx)
        periastron_pos = nothing  # set after orbit trace; drawn last
        orbit_xspan = 0.0; orbit_yspan = 0.0   # sky extent → drives figure aspect
        if isfinite(P_med) && P_med > 0
            M_pri_med = astrom_M_pri(theta_med)
            plx_med   = astrom_plx(theta_med)
            try
                orb_med, _ = Nereus._planet_orbit(theta_med, planet_idx,
                                                    M_pri_med, plx_med,
                                                    data.t_ref)
                Δra_m, Δdec_m, ts_m = _trace_orbit(orb_med, data.t_ref,
                                                     P_med; n_pts = 600)
                orbit_xspan = maximum(Δra_m) - minimum(Δra_m)
                orbit_yspan = maximum(Δdec_m) - minimum(Δdec_m)
                lines!(ax, Δra_m, Δdec_m;
                        color = NEREUS_COLORS.model,
                        linewidth = 2.5, label = "Best fit")

                # Compute periastron position now, defer the scatter
                # call so it sits on top of the data markers.
                _, ω_med = planet_e_w(theta_med, planet_idx)
                t_anc = planet_time_anchor(theta_med, planet_idx)
                time_kind = params.config.parametrization.time
                e_med, _ = planet_e_w(theta_med, planet_idx)
                tp_med = if time_kind === :Mo
                    mo_to_tp(t_anc, P_med, data.t_ref)
                elseif time_kind === :Tp
                    t_anc
                else
                    tc_to_tp(t_anc, P_med, e_med, ω_med)
                end
                periastron_pos = relastrom_offset(orb_med, tp_med)
            catch err
                @warn "Could not trace median orbit" exception=err
            end
        end

        # ---- 4. RV-epoch markers on the orbit ---------------------
        # Project each RV observation onto the median sky-plane orbit.
        # Per-instrument marker shape, colour-coded by BJD with the
        # cool colormap (matches plot_rv_phasefold's convention).
        plotted_inst_obs = false
        if isfinite(P_med) && P_med > 0 && n_rv(data) > 0
            try
                orb_for_obs, _ = Nereus._planet_orbit(theta_med, planet_idx,
                                                       astrom_M_pri(theta_med),
                                                       astrom_plx(theta_med),
                                                       data.t_ref)
                t_obs = data.t_rv
                t_min = minimum(t_obs)
                t_max = maximum(t_obs)
                inst_names = params.config.instruments.rv_names
                for (i, ins) in enumerate(inst_names)
                    mask = data.rv_inst .== i
                    any(mask) || continue
                    t_i = t_obs[mask]
                    Δra_i  = Vector{Float64}(undef, length(t_i))
                    Δdec_i = Vector{Float64}(undef, length(t_i))
                    @inbounds for k in eachindex(t_i)
                        Δra_i[k], Δdec_i[k] = relastrom_offset(orb_for_obs, t_i[k])
                    end
                    scatter!(ax, Δra_i, Δdec_i;
                              color = t_i .- t_min,
                              colormap = NEREUS_CMAP,
                              colorrange = (0, t_max - t_min),
                              marker = sky_inst_marker(i),
                              markersize = 11,
                              strokewidth = 1.0, strokecolor = :black,
                              label = ins)
                end
                Colorbar(ga[1, 2]; colormap = NEREUS_CMAP,
                          limits = (0, t_max - t_min),
                          label = "BJD - $(round(Int, t_min))")
                plotted_inst_obs = true
            catch err
                @warn "Could not project RV epochs onto orbit" exception=err
            end
        end

        # ---- 5. relAST data points — drawn AFTER the RV-epoch markers:
        # the companion barely moves over the RV baseline, so dozens of
        # projected RV epochs pile up exactly where the (few) relAST
        # measurements sit and bury them (IAD/GOST-only targets have no
        # sky-plane points to overlay).
        if relast !== nothing
            mask = relast.planet_idx .== planet_idx
            if any(mask)
                r_obs  = relast.ra_off[mask]
                d_obs  = relast.dec_off[mask]
                σr     = relast.ra_err[mask]
                σd     = relast.dec_err[mask]

                errorbars!(ax, r_obs, d_obs, σr;
                            direction = :x, color = :black,
                            linewidth = ERRBAR_LW)
                errorbars!(ax, r_obs, d_obs, σd;
                            direction = :y, color = :black,
                            linewidth = ERRBAR_LW)
                scatter!(ax, r_obs, d_obs;
                          color = NEREUS_COLORS.rv_bin,
                          markersize = 16, strokewidth = 1.8,
                          strokecolor = :black,
                          label = "relAST data")
            end
        end

        # ---- 6. Periastron (drawn AFTER data so it stays on top) --
        if periastron_pos !== nothing
            Δra_p, Δdec_p = periastron_pos
            scatter!(ax, [Δra_p], [Δdec_p];
                      color = :red, marker = :diamond,
                      markersize = 20, strokewidth = 1.5,
                      strokecolor = :black,
                      label = "Periastron")
        end

        # ---- 6. Host star at origin -------------------------------
        scatter!(ax, [0.0], [0.0];
                  color = :gold, marker = :star5,
                  markersize = 22, strokewidth = 1.5,
                  strokecolor = :black, label = "Host star")

        # Legend below the axis to keep the orbit/markers unobscured
        # regardless of orientation.
        Legend(ga[2, 1], ax;
                framevisible = false, labelsize = 14,
                orientation = :horizontal,
                tellheight = true, tellwidth = false,
                nbanks = 2)

        # The sky-plane axis uses DataAspect() (equal RA/Dec so the orbit is
        # geometrically true); on a wide-short orbit a square frame then wastes
        # vertical space and stretches the colorbar. Size the figure to the
        # orbit's own aspect so the panel fills the frame. Colorbar (~190 px) and
        # the bottom legend (~150 px) add fixed margins.
        if orbit_xspan > 0 && orbit_yspan > 0
            pw = 720.0
            ph = clamp(pw * orbit_yspan / orbit_xspan, 200.0, 720.0)
            resize!(fig.scene, round(Int, pw + 190), round(Int, ph + 150))
        end

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models",
                        "orbit_skyplane_K$planet_idx.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


# =====================================================================
# 1b. Hipparcos IAD along-scan residual diagnostic
# =====================================================================

"""
    plot_iad_residuals(chains, params, data;
                        output=nothing, fmt=:png,
                        figsize=(1000, 800), bf_cutoff=10.0)

Two-panel along-scan residual diagnostic for intermediate astrometry
(Hipparcos IAD, Gaia DR4 epoch astrometry).

For each transit `j`:
  residual_j = abscissa_j  −  Δη_orbit_j(θ)  −  ϖ(θ)·f_ϖ,j  −  X_jᵀ q_opt

where `θ` is the max-lp draw of the best-fit cluster (with its trans-dim
active set), `Δη_orbit_j` the along-scan reflex of every astrometrically
active companion, and `q_opt` the catalogue correction the marginalisation
fits — all through the likelihood's own helpers (`_iad_oc`), so χ²/N is
`χ²_min / N` of `iad_log_likelihood`. Per-transit `σ_j` is shown as
errorbars.

Panels:
  top    — residual vs ψ (scan position angle, rad)
  bottom — residual vs MJD

Multi-instrument aware: the marginalisation it subtracts is the same one
`iad_log_likelihood` performs — a shared (μα*, μδ) and one along-scan
zero point per instrument, the sampled parallax subtracted — so the
residuals shown are the residuals the fit actually sees. Instruments are distinguished by marker shape; colour
stays on time.

Returns the `Figure`. No-op (returns empty Figure) if `data.iad` is
nothing or has no more transits than the marginalisation has parameters
(4 for one instrument, 6 for two) — the likelihood's own requirement.
"""
function plot_iad_residuals(chains, params, data;
                             output::Union{Nothing, String} = nothing,
                             fmt::Symbol = :png,
                             save_pdf::Bool = false,
                             figsize = (1000, 800),
                             bf_cutoff::Real = 10.0)
    iad = data.iad
    iad === nothing && (return Figure())
    n = n_iad(iad)
    n_inst = n_iad_inst(iad)
    n_q = Nereus._iad_n_q(n_inst)
    n > n_q || (return Figure())

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ax_psi = Axis(fig[1, 1];
                       xlabel = "ψ (rad)",
                       ylabel = "abscissa residual (mas)")
        ax_t   = Axis(fig[2, 1];
                       xlabel = "MJD",
                       ylabel = "abscissa residual (mas)")

        # The max-lp draw of the best-fit cluster, with its trans-dim active
        # set. NOT the per-parameter median: on Gaia-4 that sits off the ridge
        # (χ²/N 1.45 in this cluster against 1.38 at max lp), and on a
        # trans-dim chain a Theta without the active set counts parked slots
        # as companions.
        theta = _theta_best_lp(chains, params;
                               active_idx = _top_lp_draw_pool(chains; bf_cutoff = bf_cutoff))
        # The likelihood's own residuals (`_iad_oc`): every active companion's
        # reflex and the sampled parallax subtracted, the catalogue solution
        # marginalised. So the χ²/N below is χ²_min/N of the fit.
        resid = _iad_oc(theta, data).oc
        σs = iad.abscissa_err

        # Marker shape carries the instrument; colour is already spent on time.
        _inst_marker = (:circle, :rect, :utriangle, :diamond, :cross, :star5)
        markers = [_inst_marker[mod1(m, length(_inst_marker))] for m in iad.inst]

        # Plot vs ψ
        errorbars!(ax_psi, iad.psi, resid, σs;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_psi, iad.psi, resid;
                  color = iad.t .- minimum(iad.t),
                  colormap = NEREUS_CMAP,
                  colorrange = (0, maximum(iad.t) - minimum(iad.t)),
                  marker = markers,
                  markersize = 11, strokewidth = 1.0,
                  strokecolor = :black)
        hlines!(ax_psi, 0; color = :black, linestyle = :dash, linewidth = 1.5)

        # Plot vs t
        errorbars!(ax_t, iad.t, resid, σs;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_t, iad.t, resid;
                  color = iad.t .- minimum(iad.t),
                  colormap = NEREUS_CMAP,
                  colorrange = (0, maximum(iad.t) - minimum(iad.t)),
                  marker = markers,
                  markersize = 11, strokewidth = 1.0,
                  strokecolor = :black)
        hlines!(ax_t, 0; color = :black, linestyle = :dash, linewidth = 1.5)

        # Summary statistics as a corner annotation (titles are not used
        # in Nereus plots; goodness-of-fit info goes in a small text box).
        rms = sqrt(sum(resid.^2 ./ σs.^2) / n)
        text!(ax_psi, 0.02, 0.95;
               text = "χ²/N = $(round(rms^2; digits=2))",
               align = (:left, :top), space = :relative, fontsize = 16)

        # Colorbar for the time encoding (markers colored by MJD)
        t_min_iad = minimum(iad.t); t_span_iad = maximum(iad.t) - t_min_iad
        Colorbar(fig[1:2, 2];
                  colormap = NEREUS_CMAP,
                  limits = (0.0, t_span_iad),
                  label = "MJD − $(round(Int, t_min_iad))",
                  labelsize = 16, ticklabelsize = 14)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "iad_residuals.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


# =====================================================================
# 2. HGCA PM residuals with model curves
# =====================================================================

"""
    _cov_ellipse_pts(cx, cy, Σ; n_σ=1, n_pts=120) -> (xs, ys)

Generate an n-σ confidence ellipse for a 2×2 covariance matrix Σ
centered at (cx, cy). Eigen-decomposition of Σ gives the principal
axes; we sample a unit circle and rotate / scale.
"""
function _cov_ellipse_pts(cx::Real, cy::Real, Σ::AbstractMatrix;
                           n_σ::Real = 1, n_pts::Int = 120)
    e = eigen(Symmetric(Float64.(Σ)))
    λ1, λ2 = max(e.values[1], 1e-30), max(e.values[2], 1e-30)
    V = e.vectors
    θ = range(0, 2π; length = n_pts)
    a = n_σ * sqrt(λ1)
    b = n_σ * sqrt(λ2)
    xs = Vector{Float64}(undef, n_pts)
    ys = Vector{Float64}(undef, n_pts)
    @inbounds for i in 1:n_pts
        u = a * cos(θ[i])
        v = b * sin(θ[i])
        xs[i] = cx + V[1, 1] * u + V[1, 2] * v
        ys[i] = cy + V[2, 1] * u + V[2, 2] * v
    end
    return xs, ys
end

"""
    _hgca_mub(hgca, μra_mod, μdec_mod) -> (μb_ra, μb_dec, χ²) or nothing

The barycentric proper motion `hgca_log_likelihood` marginalizes, for the
model PMs `μ_mod` at the three HGCA epochs: `μ_b = A⁻¹v` with the per-epoch
2×2 covariances, and the marginalized χ² `Σ rᵀC⁻¹r − vᵀA⁻¹v`. `nothing` when
`A` is singular.
"""
function _hgca_mub(hgca, μra_mod, μdec_mod)
    A11 = 0.0; A12 = 0.0; A22 = 0.0; v1 = 0.0; v2 = 0.0; rCr = 0.0
    @inbounds for k in 1:3
        a = hgca.cov_ep[k][1, 1]; b = hgca.cov_ep[k][1, 2]; c = hgca.cov_ep[k][2, 2]
        detC = a * c - b * b
        detC > 0 || continue
        i11 = c / detC; i12 = -b / detC; i22 = a / detC
        rra = hgca.pmra[k] - μra_mod[k]; rdec = hgca.pmdec[k] - μdec_mod[k]
        rCr += i11 * rra * rra + 2 * i12 * rra * rdec + i22 * rdec * rdec
        v1  += i11 * rra + i12 * rdec
        v2  += i12 * rra + i22 * rdec
        A11 += i11; A12 += i12; A22 += i22
    end
    detA = A11 * A22 - A12 * A12
    detA > 0 || return nothing
    μb1 = ( A22 * v1 - A12 * v2) / detA
    μb2 = (-A12 * v1 + A11 * v2) / detA
    return μb1, μb2, rCr - (v1 * μb1 + v2 * μb2)
end

"""
    _g23h_mub(g23h, μra_mod, μdec_mod) -> (μb_ra, μb_dec) or nothing

The barycentric proper motion `g23h_log_likelihood` marginalizes, for the
model PMs at the five G23H epochs, under the full 10×10 covariance.
`nothing` when the covariance or the normal matrix is singular.
"""
function _g23h_mub(g23h, μra_mod, μdec_mod)
    cholΣ = cholesky(Symmetric(Matrix{Float64}(g23h.cov)); check = false)
    LinearAlgebra.issuccess(cholΣ) || return nothing
    r10 = Vector{Float64}(undef, 10)
    @inbounds for k in 1:5
        r10[2k - 1] = g23h.pmra[k]  - μra_mod[k]
        r10[2k    ] = g23h.pmdec[k] - μdec_mod[k]
    end
    Σinv_r = cholΣ \ r10
    X = zeros(10, 2)
    @inbounds for k in 1:5
        X[2k - 1, 1] = 1.0
        X[2k    , 2] = 1.0
    end
    Y = cholΣ \ X
    A11 = 0.0; A12 = 0.0; A22 = 0.0; v1 = 0.0; v2 = 0.0
    @inbounds for k in 1:5
        v1 += Σinv_r[2k - 1]; v2 += Σinv_r[2k]
        A11 += Y[2k - 1, 1]; A12 += Y[2k - 1, 2]; A22 += Y[2k, 2]
    end
    detA = A11 * A22 - A12 * A12
    detA > 0 || return nothing
    return (A22 * v1 - A12 * v2) / detA, (-A12 * v1 + A11 * v2) / detA
end

"""
    plot_pm_residuals(chains, params, data;
                       planet_idx=1, output=nothing, fmt=:png,
                       save_pdf=false, figsize=(1100, 900), bf_cutoff=10.0)

HGCA proper-motion residuals at the three catalogue epochs (Hipparcos /
Hipparcos–Gaia / Gaia): `Δμ = μ_obs − μ_model − μ_b` in α* (top) and δ
(bottom) vs MJD, with the per-epoch 1σ, and the marginalized χ² — the
fit's, not an approximation of it:

  - `μ_model` is `_hgca_model_pm`, the model `hgca_log_likelihood` uses
    (every active companion's reflex; the Hipparcos–Gaia epoch as the mean
    reflex velocity over the baseline; the Gaia epoch through GOST when scan
    plans are supplied), at the max-lp draw in which companion `planet_idx`
    exists;
  - `μ_b` is the barycentric PM, analytically marginalized with the same
    2×2 within-epoch covariances as the likelihood.

The residuals are those of the whole model, so with several companions the
`K<k>` figures differ only through which draws have companion `k`.

Requires `data.hgca` to be non-`nothing`.
"""
function plot_pm_residuals(chains, params, data;
                            planet_idx::Int = 1,
                            output::Union{Nothing, String} = nothing,
                            fmt::Symbol = :png,
                            save_pdf::Bool = false,
                            figsize = (1100, 900),
                            bf_cutoff::Real = 10.0)
    hgca = data.hgca
    hgca === nothing && throw(ArgumentError(
        "plot_pm_residuals requires data.hgca (got nothing)"))
    _has_astrom_orbit(params, planet_idx) || return Figure()   # no astrometric orbit

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)

        labels = ("Hipparcos", "Hipparcos–Gaia", "Gaia")

        # The max-lp draw among those in which this companion exists, with its
        # trans-dim active set (see `_planet_draw`) -- not per-parameter
        # medians, which are not a model on the posterior ridge.
        drawn = _planet_draw(chains, params, planet_idx; bf_cutoff = bf_cutoff)
        drawn === nothing && return Figure()
        theta_med = first(drawn)

        # The model PM the likelihood compares with the catalogue: EVERY
        # active companion's reflex, the Hipparcos–Gaia epoch as the mean
        # reflex velocity over the baseline, the Gaia epoch through GOST when
        # scan plans are supplied (`_hgca_model_pm`, shared with
        # `hgca_log_likelihood`). This figure used to take planet
        # `planet_idx`'s instantaneous reflex alone, so its residuals and χ²
        # were not the fit's whenever there were other companions or the
        # period was comparable to the 25-yr baseline.
        med_μra, med_μdec = try
            Nereus._hgca_model_pm(theta_med, hgca, data, astrom_M_pri(theta_med),
                                  astrom_plx(theta_med), data.t_ref)
        catch err
            @warn "plot_pm_residuals: the HGCA model failed at the best-fit draw; " *
                  "nothing to plot" exception = err
            return Figure()
        end
        # The barycentric PM, marginalized as in `hgca_log_likelihood`.
        mb = _hgca_mub(hgca, med_μra, med_μdec)
        if mb === nothing
            @warn "plot_pm_residuals: singular HGCA covariance; nothing to plot"
            return Figure()
        end
        μb1, μb2, chi2_total = mb
        med_ok = true

        # Per-epoch residuals + 1-σ errors (diagonal σ from cov_ep)
        ts_ep    = collect(hgca.epochs)
        res_ra   = Float64[hgca.pmra[k]  - med_μra[k]  - μb1 for k in 1:3]
        res_dec  = Float64[hgca.pmdec[k] - med_μdec[k] - μb2 for k in 1:3]
        σ_ra_ep  = Float64[sqrt(hgca.cov_ep[k][1, 1]) for k in 1:3]
        σ_dec_ep = Float64[sqrt(hgca.cov_ep[k][2, 2]) for k in 1:3]

        ax_ra  = Axis(fig[1, 1];
                       ylabel = rich("Δμ", subscript("α*"), " (mas/yr)"))
        ax_dec = Axis(fig[2, 1]; xlabel = "MJD",
                                   ylabel = rich("Δμ", subscript("δ"), " (mas/yr)"))
        linkxaxes!(ax_ra, ax_dec)
        hidexdecorations!(ax_ra; grid = false, ticks = false)

        hlines!(ax_ra,  0; color = :black, linestyle = :dash, linewidth = 1)
        hlines!(ax_dec, 0; color = :black, linestyle = :dash, linewidth = 1)

        t_min = minimum(ts_ep); t_span = maximum(ts_ep) - t_min
        errorbars!(ax_ra, ts_ep, res_ra, σ_ra_ep;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_ra, ts_ep, res_ra;
                  color = ts_ep .- t_min,
                  colormap = NEREUS_CMAP,
                  colorrange = (0.0, t_span),
                  markersize = 14, strokewidth = 1.0, strokecolor = :black)
        errorbars!(ax_dec, ts_ep, res_dec, σ_dec_ep;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_dec, ts_ep, res_dec;
                  color = ts_ep .- t_min,
                  colormap = NEREUS_CMAP,
                  colorrange = (0.0, t_span),
                  markersize = 14, strokewidth = 1.0, strokecolor = :black)

        # Epoch labels offset from each point (anchored in data coords)
        for k in 1:3
            text!(ax_ra, ts_ep[k], res_ra[k];
                   text = " " * labels[k], align = (:left, :center),
                   fontsize = 13)
        end

        # χ² annotation lower-left of top panel
        if med_ok
            text!(ax_ra, 0.02, 0.05;
                   text = "marginalized χ² = $(round(chi2_total; digits=2))",
                   align = (:left, :bottom), space = :relative, fontsize = 14)
        end

        Colorbar(fig[1:2, 2];
                  colormap = NEREUS_CMAP,
                  limits = (0.0, t_span),
                  label = "MJD − $(round(Int, t_min))",
                  labelsize = 16, ticklabelsize = 14)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models",
                        "hgca_pm_residuals_K$planet_idx.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


# =====================================================================
# 3. Joint RV + astrometry phase fold
# =====================================================================

"""
    plot_rv_astrom_phasefold(chains, params, data;
                              n_draws=100, planet_idx=1,
                              output=nothing, fmt=:png, figsize=(1100, 1300))

Two-panel joint diagnostic for an RV+astrometry fit:

  Top: RV phase-folded on planet `planet_idx`'s period (other planets
       and per-instrument γ subtracted), with the posterior median
       Keplerian curve and an `n_draws` posterior fan.

  Bottom: Sky-plane companion track over one orbital period, colour-
       coded by orbital phase (0–1, NEREUS_CMAP). Each RV epoch
       is overplotted as a marker showing where the companion was at
       that epoch, colour-keyed to its phase. relAST data are also
       overlaid as black points with errorbars.

This visually connects the two observable channels.

Returns the `Figure`. Requires both RV and (relAST or HGCA)
astrometry.
"""
function plot_rv_astrom_phasefold(chains, params, data;
                                    n_draws::Int = 20000,
                                    planet_idx::Int = 1,
                                    output::Union{Nothing, String} = nothing,
                                    fmt::Symbol = :png,
                                    save_pdf::Bool = false,
                                    bf_cutoff::Real = 10.0,
                                    credmass::Real = 0.85,
                                    subtract_gp::Bool = true,
                                    figsize = (1100, 1300))
    _has_astrom_orbit(params, planet_idx) || return Figure()   # no astrometric orbit
    n_rv(data) > 0 || throw(ArgumentError(
        "plot_rv_astrom_phasefold requires RV data"))
    has_astrometry(data) || throw(ArgumentError(
        "plot_rv_astrom_phasefold requires relAST or HGCA astrometry"))

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)

        ax_rv = Axis(fig[1, 1];
                      xlabel = "Phase",
                      ylabel = rich("RV (m s", superscript("-1"), ")"))
        ax_sky = Axis(fig[2, 1];
                       xlabel = rich("ΔRA·cos δ (mas)"),
                       ylabel = "Δδ (mas)",
                       aspect = DataAspect())
        _flip_xaxis!(ax_sky)

        # Reference theta = max-lp draw (EMPEROR best-fit) among the draws in
        # which this companion exists, with its trans-dim active set: a real
        # smooth model ON the posterior ridge (see `_planet_draw`).
        drawn = _planet_draw(chains, params, planet_idx; bf_cutoff = bf_cutoff)
        drawn === nothing && return Figure()
        theta_med, active_idx = drawn
        P_med = _orbit_period_days(theta_med, planet_idx)
        if !(isfinite(P_med) && P_med > 0)
            @warn "Best-fit period not finite; aborting joint phase-fold"
            return fig
        end

        e_med, ω_med = planet_e_w(theta_med, planet_idx)
        t_anc = planet_time_anchor(theta_med, planet_idx)
        time_kind = params.config.parametrization.time
        tp_med = if time_kind === :Mo
            mo_to_tp(t_anc, P_med, data.t_ref)
        elseif time_kind === :Tp
            t_anc
        else
            tc_to_tp(t_anc, P_med, e_med, ω_med)
        end

        # ---- TOP: RV phase fold ----------------------------------
        # Data de-trend (other planets + γ + trend) and the black curve
        # both come from the SAME max-lp theta, so the data sit on the
        # curve by construction; the fan shows posterior spread. (A
        # θ-marginal-median curve sat visibly off the data, and a
        # pointwise predictive median is an order statistic — kinked.)
        preds_all, _ = rv_predictions(theta_med, data)
        preds_planet = compute_rv_model_planet(theta_med, data,
                                                 data.t_rv, planet_idx)
        rv_folded = data.rv .- (preds_all .- preds_planet)
        if subtract_gp
            residuals_full = data.rv .- preds_all
            gp_at_data = try
                channel_gp_mean_at(theta_med, residuals_full,
                                    data.rv_err .^ 2,
                                    data.t_rv, data.t_rv,
                                    data.rv_inst, :rv)
            catch err
                @warn "GP cleaning failed in astrom phasefold" exception=err
                nothing
            end
            gp_at_data !== nothing && (rv_folded = rv_folded .- gp_at_data)
        end
        # Phase fold on tp so phase=0 = periastron (consistent with
        # bottom panel's colour key).
        # x axis runs [-0.5, 0.5] with periastron CENTRED at 0. `phases_rv01`
        # keeps the original [0,1) phase for the colour mapping, which is shared
        # with the sky panel's colour key (0 = periastron) — only the RV panel's
        # horizontal placement is re-centred.
        phases_rv01 = mod.((data.t_rv .- tp_med) ./ P_med, 1.0)
        phases_rv   = mod.(phases_rv01 .+ 0.5, 1.0) .- 0.5

        # Posterior uncertainty: 1/2/3σ CI BANDS, identical in construction to
        # plot_rv_phasefold, so the two RV folds of the same target agree.
        #
        # This used to be a spaghetti fan of `n_draws` individual lines. Two
        # problems: (a) it disagreed visually with plot_rv_phasefold's bands for
        # no reason, and (b) the draw pool came from `_top_lp_draw_pool`, which
        # is a silent NO-OP on any chain without an `:lp` column (Pigeons PT
        # chains have none) — so the fan quietly included the whole degenerate
        # tail, and near-face-on draws with K→0 rendered as flat lines pinned at
        # zero. `_credible_region_pool` is parameter-based and doesn't depend on
        # `lp` existing.
        ph_fine = collect(range(-0.5, 0.5; length = 800))
        cred_idx = _credible_region_pool(chains, params, planet_idx;
                                          credmass = credmass)
        band_idx = intersect(collect(active_idx), cred_idx)
        isempty(band_idx) && (band_idx = collect(active_idx))
        if !isempty(band_idx)
            param_syms = names(chains, :parameters)
            mat = Matrix{Float64}(undef, length(band_idx), length(param_syms))
            for (j, sym) in enumerate(param_syms)
                mat[:, j] = vec(Array(chains[sym]))[band_idx]
            end
            ci_chains = MCMCChains.Chains(mat, param_syms)
            # PHASE ORIGIN. compute_ci_bands ignores fold_t0 and references
            # every draw to ITS OWN inferior conjunction Tc (deliberately — λ is
            # well determined where periastron is not). This panel, however,
            # folds the data and the black model curve on PERIASTRON tp_med. Ask
            # for the grid shifted by Δ = (Tc − Tp)/P so the returned band comes
            # back on the panel's periastron axis; without this the band sits a
            # constant ~Δ in phase away from the curve it is meant to bracket.
            Δφ = (_planet_Tc(theta_med, data, planet_idx) - tp_med) / P_med
            ci = compute_ci_bands(ci_chains, params, data,
                                   tp_med .+ ph_fine .* P_med;
                                   planet = planet_idx, n_draws = n_draws,
                                   bf_cutoff = Inf,
                                   fold_phase = ph_fine .- Δφ, fold_t0 = tp_med)
            band!(ax_rv, ph_fine, ci.lo3, ci.hi3; color = (NEREUS_COLORS.ci, CI_ALPHA_3SIGMA))
            band!(ax_rv, ph_fine, ci.lo2, ci.hi2; color = (NEREUS_COLORS.ci, CI_ALPHA_2SIGMA))
            band!(ax_rv, ph_fine, ci.lo1, ci.hi1; color = (NEREUS_COLORS.ci, CI_ALPHA_1SIGMA))
        end

        inst_names = params.config.instruments.rv_names
        # SB2: the astrometric reflex pairs with the PRIMARY (star A) RV — show
        # only component-1 points here; the secondary lives in the dedicated
        # double-lined binary fold, not this joint RV–astrometry plot.
        sb2_pri = _is_sb2_data(data) ? (data.rv_comp .== 1) : trues(length(data.t_rv))
        for (i, ins) in enumerate(inst_names)
            mask = (data.rv_inst .== i) .& sb2_pri
            count(mask) == 0 && continue
            errorbars!(ax_rv, phases_rv[mask], rv_folded[mask],
                        data.rv_err[mask];
                        color = (:gray, 0.3), linewidth = 1.0)
            scatter!(ax_rv, phases_rv[mask], rv_folded[mask];
                      color = phases_rv01[mask],
                      colormap = NEREUS_CMAP,
                      colorrange = (0.0, 1.0),
                      marker = inst_marker(i), markersize = 12,
                      strokewidth = 0.8,
                      strokecolor = (:black, 0.4),
                      label = ins)
        end

        # Best-fit Keplerian curve (smooth, on-ridge by construction)
        t_fine = tp_med .+ ph_fine .* P_med
        model_fine = compute_rv_model_planet(theta_med, data, t_fine,
                                               planet_idx)
        lines!(ax_rv, ph_fine, model_fine;
                color = :black, linewidth = 2.5)

        xlims!(ax_rv, -0.5, 0.5)
        # Per-instrument legend for the RV panel (marker shapes are the only
        # way to tell instruments apart; the figure-level legend below only
        # carries the sky-panel entries).
        # Upper-LEFT. The fold is periastron-referenced, so the curve climbs out
        # of its minimum at phase 0 and the top-left corner is the one region
        # both the model and the data reliably avoid. :rb put the legend on top
        # of the phase≈0.95 data cluster and clipped its text at the axis edge.
        length(inst_names) > 0 && axislegend(ax_rv; merge = true, position = :lt,
                                              framevisible = false,
                                              labelsize = 12)

        # ---- BOTTOM: sky-plane track + RV epoch markers -----------
        M_pri_med = astrom_M_pri(theta_med)
        plx_med   = astrom_plx(theta_med)

        try
            orb_med, _ = Nereus._planet_orbit(theta_med, planet_idx,
                                                M_pri_med, plx_med,
                                                data.t_ref)

            # Colour-coded orbit track (sample at fine phase grid).
            n_track = 600
            ph_track = collect(range(0.0, 1.0; length = n_track))
            t_track = tp_med .+ ph_track .* P_med
            Δra_t  = Vector{Float64}(undef, n_track)
            Δdec_t = Vector{Float64}(undef, n_track)
            @inbounds for (j, t) in enumerate(t_track)
                r, d = relastrom_offset(orb_med, t)
                Δra_t[j]  = r
                Δdec_t[j] = d
            end

            # Lines! supports per-vertex colors when given a vector.
            lines!(ax_sky, Δra_t, Δdec_t;
                    color = ph_track,
                    colormap = NEREUS_CMAP,
                    colorrange = (0.0, 1.0),
                    linewidth = 2.5)

            # RV epochs projected onto the orbit, colour-keyed by phase.
            Δra_rv  = Vector{Float64}(undef, length(data.t_rv))
            Δdec_rv = Vector{Float64}(undef, length(data.t_rv))
            @inbounds for (j, t) in enumerate(data.t_rv)
                r, d = relastrom_offset(orb_med, t)
                Δra_rv[j]  = r
                Δdec_rv[j] = d
            end
            scatter!(ax_sky, Δra_rv, Δdec_rv;
                      color = phases_rv01,
                      colormap = NEREUS_CMAP,
                      colorrange = (0.0, 1.0),
                      marker = :utriangle,
                      markersize = 10, strokewidth = 0.8,
                      strokecolor = (:black, 0.6),
                      label = "RV epochs")
        catch err
            @warn "Could not project median orbit on sky-plane" exception=err
        end

        # Overlay relAST data for this planet
        if data.relastrom !== nothing
            relast = data.relastrom
            mask = relast.planet_idx .== planet_idx
            if any(mask)
                errorbars!(ax_sky, relast.ra_off[mask],
                            relast.dec_off[mask], relast.ra_err[mask];
                            direction = :x, color = :black,
                            linewidth = ERRBAR_LW)
                errorbars!(ax_sky, relast.ra_off[mask],
                            relast.dec_off[mask], relast.dec_err[mask];
                            direction = :y, color = :black,
                            linewidth = ERRBAR_LW)
                scatter!(ax_sky, relast.ra_off[mask],
                          relast.dec_off[mask];
                          color = :white, marker = :circle,
                          markersize = 12, strokewidth = 1.5,
                          strokecolor = :black,
                          label = "relAST data")
            end
        end

        # Host star
        scatter!(ax_sky, [0.0], [0.0];
                  color = :gold, marker = :star5,
                  markersize = 22, strokewidth = 1.5,
                  strokecolor = :black, label = "Host star")

        Colorbar(fig[1:2, 2];
                  colormap = NEREUS_CMAP,
                  limits = (0.0, 1.0),
                  label = "Orbital phase (0 = periastron)",
                  labelsize = 18, ticklabelsize = 14)

        Legend(fig[3, 1:2], ax_sky;
                framevisible = false, labelsize = 14,
                orientation = :horizontal,
                tellheight = true, tellwidth = false,
                nbanks = 2)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models",
                        "rv_astrom_phasefold_K$planet_idx.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


# =====================================================================
# 5. Relative astrometry — separation + PA vs time (sky-plane companion)
# =====================================================================

"""
    _sep_pa_from_offset(Δra, Δdec) -> (ρ, PA)

Convert RA/Dec offsets [mas] to (separation [mas], position angle [deg]).
PA measured from North (+δ) through East (+α*), in [0, 360).
"""
@inline function _sep_pa_from_offset(Δra::Real, Δdec::Real)
    ρ  = hypot(Δra, Δdec)
    PA = mod(atand(Δra, Δdec), 360.0)
    return ρ, PA
end

"""
    _sep_pa_errs(Δra, Δdec, σra, σdec, corr) -> (σ_ρ, σ_PA)

Linearized error propagation on the (Δra, Δdec)→(ρ, PA) transformation,
including the off-diagonal correlation. PA error in degrees.
"""
function _sep_pa_errs(Δra::Real, Δdec::Real,
                      σra::Real, σdec::Real, corr::Real)
    ρ2 = Δra*Δra + Δdec*Δdec
    ρ  = sqrt(ρ2)
    ρ < 1e-12 && return (hypot(σra, σdec), 360.0)
    # ∂ρ/∂α = α/ρ, ∂ρ/∂δ = δ/ρ
    σ_ρ2 = (Δra*σra)^2 / ρ2 + (Δdec*σdec)^2 / ρ2 +
            2 * Δra*Δdec * σra * σdec * corr / ρ2
    σ_ρ = sqrt(max(σ_ρ2, 0.0))
    # PA = atan2(α, δ): ∂PA/∂α = δ/ρ², ∂PA/∂δ = -α/ρ² (radians)
    σ_PA_rad2 = (Δdec * σra)^2 / ρ2^2 + (Δra * σdec)^2 / ρ2^2 -
                 2 * Δra * Δdec * σra * σdec * corr / ρ2^2
    σ_PA_deg = sqrt(max(σ_PA_rad2, 0.0)) * (180/π)
    return σ_ρ, σ_PA_deg
end

"""
    plot_relastrom_timeseries(chains, params, data;
                              n_draws=100, planet_idx=1,
                              output=nothing, fmt=:png, figsize=(1100, 900),
                              bf_cutoff=5.0)

Two-panel publication plot of the relative-astrometry data:
  top    — Separation ρ [mas] vs time
  bottom — Position angle [deg] vs time

Observed epochs drawn as data points with errorbars from RA/Dec
error propagation (including off-diagonal correlation). Posterior
median orbit traced as a solid line; `n_draws` posterior fan draws
shown as faint orchid lines.

Returns a `Figure`. No-op if `data.relastrom` is `nothing`.
"""
function plot_relastrom_timeseries(chains, params, data;
                                     n_draws::Int = 100,
                                     planet_idx::Int = 1,
                                     output::Union{Nothing, String} = nothing,
                                     fmt::Symbol = :png,
                                     save_pdf::Bool = false,
                                     figsize = (1100, 900),
                                     bf_cutoff::Real = 5.0)
    _has_astrom_orbit(params, planet_idx) || return Figure()   # no astrometric orbit
    relast = data.relastrom
    relast === nothing && return Figure()
    mask = relast.planet_idx .== planet_idx
    any(mask) || return Figure()

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ax_sep = Axis(fig[1, 1]; ylabel = "Separation (mas)")
        ax_pa  = Axis(fig[2, 1]; xlabel = "MJD",
                                  ylabel = "PA (deg)")
        linkxaxes!(ax_sep, ax_pa)
        hidexdecorations!(ax_sep; grid = false, ticks = false)

        # Draws in which this companion exists, in the best-fit cluster, and
        # the max-lp one of them (see `_planet_draw`).
        drawn = _planet_draw(chains, params, planet_idx; bf_cutoff = bf_cutoff)
        drawn === nothing && return Figure()
        theta_med, active_idx = drawn
        tdc = _td_cols(chains, params)
        cols = _chain_cols(chains, params)   # ONCE, not once per draw
        P_med = _orbit_period_days(theta_med, planet_idx)
        M_pri = astrom_M_pri(theta_med)
        plx   = astrom_plx(theta_med)

        # Time grid: 10% padding around data span
        t_obs = relast.t[mask]
        t_lo  = minimum(t_obs); t_hi = maximum(t_obs)
        span  = t_hi - t_lo
        t_grid = collect(range(t_lo - 0.10*span, t_hi + 0.10*span; length = 600))

        # --- Posterior fan ----------------------------------------------
        ndraw = min(n_draws, length(active_idx))
        if ndraw > 0
            draw_idx = active_idx[rand(1:length(active_idx), ndraw)]
            for i in draw_idx
                theta = _theta_from_chain_row(chains, params, i; tdc = tdc, cols = cols)
                P = _orbit_period_days(theta, planet_idx)
                (isfinite(P) && P > 0) || continue
                try
                    orb, _ = Nereus._planet_orbit(theta, planet_idx,
                                                    astrom_M_pri(theta),
                                                    astrom_plx(theta),
                                                    data.t_ref)
                    sep_g = Vector{Float64}(undef, length(t_grid))
                    pa_g  = Vector{Float64}(undef, length(t_grid))
                    @inbounds for (k, t) in enumerate(t_grid)
                        r, d = relastrom_offset(orb, t)
                        sep_g[k], pa_g[k] = _sep_pa_from_offset(r, d)
                    end
                    lines!(ax_sep, t_grid, sep_g;
                            color = (NEREUS_COLORS.ci, 0.12), linewidth = 0.8)
                    lines!(ax_pa, t_grid, pa_g;
                            color = (NEREUS_COLORS.ci, 0.12), linewidth = 0.8)
                catch
                end
            end
        end

        # --- Best-fit curve (max-lp draw; smooth + on-ridge) --------------
        # theta_med here is `_theta_best_lp` — the θ-marginal-median curve
        # sat ~+90 mas / ~+3° off the fan on HD 159062 (circular Ω/Mo +
        # sesinw/secosw put the marginal-median θ off the posterior ridge),
        # and a pointwise predictive median kinks at draw-curve crossings.
        if isfinite(P_med) && P_med > 0
            try
                orb_med, _ = Nereus._planet_orbit(theta_med, planet_idx,
                                                    M_pri, plx, data.t_ref)
                sep_m = Vector{Float64}(undef, length(t_grid))
                pa_m  = Vector{Float64}(undef, length(t_grid))
                @inbounds for (k, t) in enumerate(t_grid)
                    r, d = relastrom_offset(orb_med, t)
                    sep_m[k], pa_m[k] = _sep_pa_from_offset(r, d)
                end
                lines!(ax_sep, t_grid, sep_m;
                        color = NEREUS_COLORS.model, linewidth = 2.0,
                        label = "Best fit")
                lines!(ax_pa, t_grid, pa_m;
                        color = NEREUS_COLORS.model, linewidth = 2.0)
            catch err
                @warn "Could not trace best-fit orbit (sep/PA)" exception=err
            end
        end

        # --- Observed data points + errorbars --------------------------
        ra_obs  = relast.ra_off[mask]
        dec_obs = relast.dec_off[mask]
        σ_ra    = relast.ra_err[mask]
        σ_dec   = relast.dec_err[mask]
        corr_v  = relast.corr[mask]
        sep_obs = similar(ra_obs); pa_obs = similar(ra_obs)
        σ_sep   = similar(ra_obs); σ_pa  = similar(ra_obs)
        for i in eachindex(ra_obs)
            sep_obs[i], pa_obs[i] = _sep_pa_from_offset(ra_obs[i], dec_obs[i])
            σ_sep[i], σ_pa[i] = _sep_pa_errs(ra_obs[i], dec_obs[i],
                                              σ_ra[i], σ_dec[i], corr_v[i])
        end

        errorbars!(ax_sep, t_obs, sep_obs, σ_sep;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_sep, t_obs, sep_obs;
                  color = NEREUS_COLORS.rv_bin, markersize = 11,
                  strokewidth = 1.0, strokecolor = :black,
                  label = "relAST data")

        errorbars!(ax_pa, t_obs, pa_obs, σ_pa;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_pa, t_obs, pa_obs;
                  color = NEREUS_COLORS.rv_bin, markersize = 11,
                  strokewidth = 1.0, strokecolor = :black)

        axislegend(ax_sep; position = :rt, framevisible = false,
                    labelsize = 14)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models",
                        "relastrom_timeseries_K$planet_idx.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


# =====================================================================
# 6. Relative astrometry — sep/PA residuals (observed − median model)
# =====================================================================

"""
    plot_relastrom_residuals(chains, params, data;
                             planet_idx=1, output=nothing, fmt=:png,
                             figsize=(1100, 700), bf_cutoff=5.0)

Two-panel residual diagnostic for relative-astrometry epochs:
  top    — Δρ  = ρ_obs − ρ_med  vs MJD  [mas]
  bottom — ΔPA = PA_obs − PA_med vs MJD [deg, wrapped to (-180, 180]]

Posterior-median orbit is used to compute the model `ρ_med`, `PA_med`
at each observed epoch. Errorbars from per-epoch RA/Dec uncertainty
propagation. Dashed zero reference per panel.
"""
function plot_relastrom_residuals(chains, params, data;
                                    planet_idx::Int = 1,
                                    output::Union{Nothing, String} = nothing,
                                    fmt::Symbol = :png,
                                    save_pdf::Bool = false,
                                    figsize = (1100, 700),
                                    bf_cutoff::Real = 5.0)
    _has_astrom_orbit(params, planet_idx) || return Figure()   # no astrometric orbit
    relast = data.relastrom
    relast === nothing && return Figure()
    mask = relast.planet_idx .== planet_idx
    any(mask) || return Figure()

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ax_sep = Axis(fig[1, 1]; ylabel = "Δρ (mas)")
        ax_pa  = Axis(fig[2, 1]; xlabel = "MJD",
                                  ylabel = "ΔPA (deg)")
        linkxaxes!(ax_sep, ax_pa)
        hidexdecorations!(ax_sep; grid = false, ticks = false)

        # The max-lp draw among those in which this companion exists -- not
        # per-parameter medians, whose orbit is off the posterior ridge and
        # puts residuals where the fit has none (see `_planet_draw`).
        drawn = _planet_draw(chains, params, planet_idx; bf_cutoff = bf_cutoff)
        drawn === nothing && return Figure()
        theta_med = first(drawn)
        P_med = _orbit_period_days(theta_med, planet_idx)

        t_obs = relast.t[mask]
        ra_obs = relast.ra_off[mask]; dec_obs = relast.dec_off[mask]
        σ_ra = relast.ra_err[mask]; σ_dec = relast.dec_err[mask]
        corr_v = relast.corr[mask]

        res_sep = fill(NaN, length(t_obs))
        res_pa  = fill(NaN, length(t_obs))
        σ_sep   = fill(0.0, length(t_obs))
        σ_pa    = fill(0.0, length(t_obs))
        if isfinite(P_med) && P_med > 0
            try
                orb_med, _ = Nereus._planet_orbit(theta_med, planet_idx,
                                                    astrom_M_pri(theta_med),
                                                    astrom_plx(theta_med),
                                                    data.t_ref)
                for i in eachindex(t_obs)
                    sep_o, pa_o = _sep_pa_from_offset(ra_obs[i], dec_obs[i])
                    r_m, d_m = relastrom_offset(orb_med, t_obs[i])
                    sep_m, pa_m = _sep_pa_from_offset(r_m, d_m)
                    res_sep[i] = sep_o - sep_m
                    Δpa = mod(pa_o - pa_m + 180.0, 360.0) - 180.0
                    res_pa[i] = Δpa
                    σ_sep[i], σ_pa[i] = _sep_pa_errs(ra_obs[i], dec_obs[i],
                                                      σ_ra[i], σ_dec[i],
                                                      corr_v[i])
                end
            catch err
                @warn "Could not compute model sep/PA at epochs" exception=err
            end
        end

        hlines!(ax_sep, 0; color = :black, linestyle = :dash, linewidth = 1)
        errorbars!(ax_sep, t_obs, res_sep, σ_sep;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_sep, t_obs, res_sep;
                  color = NEREUS_COLORS.rv_bin, markersize = 11,
                  strokewidth = 1.0, strokecolor = :black)

        hlines!(ax_pa, 0; color = :black, linestyle = :dash, linewidth = 1)
        errorbars!(ax_pa, t_obs, res_pa, σ_pa;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_pa, t_obs, res_pa;
                  color = NEREUS_COLORS.rv_bin, markersize = 11,
                  strokewidth = 1.0, strokecolor = :black)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models",
                        "relastrom_residuals_K$planet_idx.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


# =====================================================================
# 7. G23H (Thompson+ 2026) per-epoch PM residuals
# =====================================================================

"""
    plot_g23h_residuals(chains, params, data;
                        output=nothing, fmt=:png, figsize=(1100, 800),
                        bf_cutoff=10.0)

Per-epoch proper-motion residuals for G23H (Thompson+ 2026) catalog
data. Five epochs: Hip, HG long-baseline, Gaia DR2, DR3-DR2, DR3.

  top    — Δμ_α* = μ_α*_obs − μ_α*_model  vs MJD   [mas/yr]
  bottom — Δμ_δ  = μ_δ_obs  − μ_δ_model  vs MJD   [mas/yr]

Model PM = instantaneous reflex PM at each epoch (Mode A; no
window-averaging). Errorbars from the diagonal of the G23H 10×10
within-epoch covariance.
"""
function plot_g23h_residuals(chains, params, data;
                              output::Union{Nothing, String} = nothing,
                              fmt::Symbol = :png,
                              save_pdf::Bool = false,
                              figsize = (1100, 800),
                              bf_cutoff::Real = 10.0)
    g23h = data.g23h
    g23h === nothing && return Figure()

    epoch_names = ("Hip", "HG", "DR2", "DR3−DR2", "DR3")

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ax_ra  = Axis(fig[1, 1]; ylabel = rich("Δμ", subscript("α*"), " (mas/yr)"))
        ax_dec = Axis(fig[2, 1]; xlabel = "MJD",
                                   ylabel = rich("Δμ", subscript("δ"), " (mas/yr)"))
        linkxaxes!(ax_ra, ax_dec)
        hidexdecorations!(ax_ra; grid = false, ticks = false)

        # The max-lp draw of the best-fit cluster with its trans-dim active
        # set, not per-parameter medians (off the posterior ridge).
        theta_med = _theta_best_lp(chains, params;
                                   active_idx = _top_lp_draw_pool(chains; bf_cutoff = bf_cutoff))
        M_pri = astrom_M_pri(theta_med)
        plx   = astrom_plx(theta_med)
        t_ref = data.t_ref

        ts = collect(g23h.epochs)
        μra_obs  = collect(g23h.pmra)
        μdec_obs = collect(g23h.pmdec)
        # Diagonal σs from the 10x10 covariance, ordering [pmra_k, pmdec_k]_k=1..5
        σ_ra  = [sqrt(g23h.cov[2k-1, 2k-1]) for k in 1:5]
        σ_dec = [sqrt(g23h.cov[2k,   2k  ]) for k in 1:5]

        # The likelihood's own model (`_g23h_model_pm`: every active, coupled
        # companion; GOST Mode B at DR3 when scan plans are supplied) and its
        # marginalized barycentric PM. This figure used the instantaneous
        # reflex at all five epochs, so its residuals were not the fit's
        # whenever GOST was in the fit.
        μra_mod, μdec_mod = Nereus._g23h_model_pm(theta_med, g23h, data, M_pri, plx, t_ref)
        μb1, μb2 = something(_g23h_mub(g23h, μra_mod, μdec_mod), (0.0, 0.0))
        res_ra  = μra_obs  .- μra_mod  .- μb1
        res_dec = μdec_obs .- μdec_mod .- μb2

        hlines!(ax_ra,  0; color = :black, linestyle = :dash, linewidth = 1)
        hlines!(ax_dec, 0; color = :black, linestyle = :dash, linewidth = 1)

        # Per-epoch marker so the 3 close-in-time Gaia epochs are
        # distinguishable. Time colormap kept for visual coherence with
        # the IAD / HGCA plots.
        epoch_markers = (:circle, :rect, :utriangle, :diamond, :dtriangle)
        t_min = minimum(ts); t_span = maximum(ts) - t_min
        ax_ra_handles  = []
        for k in 1:5
            errorbars!(ax_ra, [ts[k]], [res_ra[k]], [σ_ra[k]];
                        color = :black, linewidth = ERRBAR_LW)
            h = scatter!(ax_ra, [ts[k]], [res_ra[k]];
                          color = [ts[k] - t_min],
                          colormap = NEREUS_CMAP,
                          colorrange = (0.0, t_span),
                          marker = epoch_markers[k],
                          markersize = 14, strokewidth = 1.0,
                          strokecolor = :black,
                          label = epoch_names[k])
            push!(ax_ra_handles, h)

            errorbars!(ax_dec, [ts[k]], [res_dec[k]], [σ_dec[k]];
                        color = :black, linewidth = ERRBAR_LW)
            scatter!(ax_dec, [ts[k]], [res_dec[k]];
                      color = [ts[k] - t_min],
                      colormap = NEREUS_CMAP,
                      colorrange = (0.0, t_span),
                      marker = epoch_markers[k],
                      markersize = 14, strokewidth = 1.0,
                      strokecolor = :black)
        end

        # Epoch legend in the top-right of the upper panel — solid black
        # markers (legend ignores the colormap so we render them as
        # neutral). Avoids the per-point text-overlap at the Gaia cluster.
        axislegend(ax_ra; position = :rt, framevisible = false,
                    labelsize = 13, nbanks = 5,
                    orientation = :horizontal)

        # Time colorbar so the cool-colormap encoding is interpretable
        Colorbar(fig[1:2, 2];
                  colormap = NEREUS_CMAP,
                  limits = (0.0, t_span),
                  label = "MJD − $(round(Int, t_min))",
                  labelsize = 16, ticklabelsize = 14)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "g23h_residuals.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


# =====================================================================
# 8. Proper-motion-anomaly trajectory along GOST scan window
# =====================================================================

"""
    plot_pm_anomaly(chains, params, data;
                    n_draws=80, planet_idx=1,
                    output=nothing, fmt=:png, figsize=(1100, 900),
                    bf_cutoff=5.0)

Two-panel reflex proper-motion trajectory across the GOST scan window:
  top    — μ_α*_reflex(t) (mas/yr)   vs MJD
  bottom — μ_δ_reflex(t)  (mas/yr)   vs MJD

Best-fit (max-lp) curve in cyan; the 16-84% band of `n_draws` posterior
draws in orchid. Curve and band are the same quantity -- the summed reflex
of every companion the likelihood couples to the astrometry -- drawn among
the draws in which companion `planet_idx` exists. HGCA / G23H observed PMs
at their tabulated epochs, less the barycentric PM each likelihood
marginalizes (`_hgca_mub`, `_g23h_mub`), beside open markers for the
likelihood's own model at those epochs (baseline-mean at Hipparcos–Gaia,
GOST window at Gaia -- not points on the instantaneous curve).

Requires `data.gost`.
"""
function plot_pm_anomaly(chains, params, data;
                          n_draws::Int = 80,
                          planet_idx::Int = 1,
                          output::Union{Nothing, String} = nothing,
                          fmt::Symbol = :png,
                          save_pdf::Bool = false,
                          figsize = (1100, 900),
                          bf_cutoff::Real = 5.0)
    _has_astrom_orbit(params, planet_idx) || return Figure()   # no astrometric orbit
    gost = data.gost
    gost === nothing && return Figure()

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ax_ra  = Axis(fig[1, 1]; ylabel = rich("μ", subscript("α*"), " reflex (mas/yr)"))
        ax_dec = Axis(fig[2, 1]; xlabel = "MJD",
                                   ylabel = rich("μ", subscript("δ"), " reflex (mas/yr)"))
        linkxaxes!(ax_ra, ax_dec)
        hidexdecorations!(ax_ra; grid = false, ticks = false)

        # Draws in which this companion exists, in the best-fit cluster, and
        # the max-lp one of them -- not per-parameter medians, which are off
        # the posterior ridge (see `_planet_draw`).
        drawn = _planet_draw(chains, params, planet_idx; bf_cutoff = bf_cutoff)
        drawn === nothing && return Figure()
        theta_med, active_idx = drawn
        tdc = _td_cols(chains, params)
        cols = _chain_cols(chains, params)   # ONCE, not once per draw

        # Dense time grid spanning all astrometric epochs (Hip + Gaia +
        # GOST + relAST) so the model curve is shown wherever a data
        # point is plotted, not just inside the GOST window.
        t_min_data = minimum(gost.t); t_max_data = maximum(gost.t)
        if data.hgca !== nothing
            t_min_data = min(t_min_data, minimum(data.hgca.epochs))
            t_max_data = max(t_max_data, maximum(data.hgca.epochs))
        end
        if data.g23h !== nothing
            t_min_data = min(t_min_data, minimum(data.g23h.epochs))
            t_max_data = max(t_max_data, maximum(data.g23h.epochs))
        end
        if data.relastrom !== nothing
            t_min_data = min(t_min_data, minimum(data.relastrom.t))
            t_max_data = max(t_max_data, maximum(data.relastrom.t))
        end
        pad = 0.05 * (t_max_data - t_min_data)
        t_grid = collect(range(t_min_data - pad, t_max_data + pad; length = 800))

        # --- Posterior CI band: 16/84 percentile per time-grid point ---
        # Collect μ_α*, μ_δ at every grid point across `ndraw` posterior
        # draws, then summarize to a band instead of plotting individual
        # noisy lines (was unreadable at low draw count).
        ndraw = min(n_draws, length(active_idx))
        if ndraw > 0
            draw_idx = active_idx[rand(1:length(active_idx), ndraw)]
            ng = length(t_grid)
            μra_mat  = fill(NaN, ndraw, ng)
            μdec_mat = fill(NaN, ndraw, ng)
            for (q, i) in enumerate(draw_idx)
                theta = _theta_from_chain_row(chains, params, i; tdc = tdc, cols = cols)
                P = _orbit_period_days(theta, planet_idx)
                (isfinite(P) && P > 0) || continue
                try
                    # The same quantity as the best-fit curve below: the
                    # summed reflex of the companions active in this draw.
                    _, orbs_i, Ms_i = Nereus._iad_active_orbits(
                        theta, astrom_M_pri(theta), astrom_plx(theta), data.t_ref)
                    @inbounds for k in 1:ng
                        μa = 0.0; μd = 0.0
                        for j in eachindex(orbs_i)
                            a_j, d_j = star_reflex_pm(orbs_i[j], t_grid[k], Ms_i[j])
                            μa += a_j; μd += d_j
                        end
                        if isfinite(μa) && isfinite(μd)
                            μra_mat[q,  k] = μa
                            μdec_mat[q, k] = μd
                        end
                    end
                catch
                end
            end
            lo_ra  = Vector{Float64}(undef, ng)
            hi_ra  = Vector{Float64}(undef, ng)
            lo_dec = Vector{Float64}(undef, ng)
            hi_dec = Vector{Float64}(undef, ng)
            @inbounds for k in 1:ng
                col_ra  = filter(isfinite, view(μra_mat,  :, k))
                col_dec = filter(isfinite, view(μdec_mat, :, k))
                if length(col_ra) >= 4
                    lo_ra[k]  = quantile(col_ra,  0.16)
                    hi_ra[k]  = quantile(col_ra,  0.84)
                else
                    lo_ra[k]  = NaN; hi_ra[k]  = NaN
                end
                if length(col_dec) >= 4
                    lo_dec[k] = quantile(col_dec, 0.16)
                    hi_dec[k] = quantile(col_dec, 0.84)
                else
                    lo_dec[k] = NaN; hi_dec[k] = NaN
                end
            end
            ok_ra  = isfinite.(lo_ra)  .& isfinite.(hi_ra)
            ok_dec = isfinite.(lo_dec) .& isfinite.(hi_dec)
            if any(ok_ra)
                band!(ax_ra,  t_grid[ok_ra],  lo_ra[ok_ra],  hi_ra[ok_ra];
                       color = (NEREUS_COLORS.ci, CI_ALPHA_2SIGMA))
            end
            if any(ok_dec)
                band!(ax_dec, t_grid[ok_dec], lo_dec[ok_dec], hi_dec[ok_dec];
                       color = (NEREUS_COLORS.ci, CI_ALPHA_2SIGMA))
            end
        end

        # --- GOST scan window + scan epochs ---------------------------
        # Mode B fits the Gaia PM as the AVERAGE over the forecast scan
        # epochs; without rendering them the GOST data are invisible in
        # every plot. Shade the scan window and rug the individual scan
        # times on both panels.
        vspan!(ax_ra,  minimum(gost.t), maximum(gost.t);
                color = (NEREUS_COLORS.pm_marker, 0.10))
        vspan!(ax_dec, minimum(gost.t), maximum(gost.t);
                color = (NEREUS_COLORS.pm_marker, 0.10))
        vlines!(ax_ra, gost.t; ymin = 0.0, ymax = 0.05,
                 color = (NEREUS_COLORS.pm_marker, 0.8), linewidth = 0.8,
                 label = "GOST scans ($(length(gost.t)))")
        vlines!(ax_dec, gost.t; ymin = 0.0, ymax = 0.05,
                 color = (NEREUS_COLORS.pm_marker, 0.8), linewidth = 0.8)

        # --- Best-fit curve: the summed reflex of the companions the
        # likelihood sums (astrometric mode, active, coupled), selected by the
        # same helper the IAD path uses.
        M_pri = astrom_M_pri(theta_med)
        plx   = astrom_plx(theta_med)
        _, active_orbs, active_Msec = Nereus._iad_active_orbits(theta_med, M_pri, plx,
                                                                data.t_ref)
        if !isempty(active_orbs)
            μra_m  = zeros(length(t_grid))
            μdec_m = zeros(length(t_grid))
            for q in eachindex(active_orbs)
                for (k, t) in enumerate(t_grid)
                    μa, μd = star_reflex_pm(active_orbs[q], t, active_Msec[q])
                    μra_m[k]  += μa
                    μdec_m[k] += μd
                end
            end
            lines!(ax_ra,  t_grid, μra_m;
                    color = NEREUS_COLORS.model, linewidth = 2.0,
                    label = "Best fit")
            lines!(ax_dec, t_grid, μdec_m;
                    color = NEREUS_COLORS.model, linewidth = 2.0)
        end

        # --- Catalog overlays (HGCA / G23H) ---
        # Observed PM minus the barycentric PM each likelihood marginalizes
        # (`_hgca_mub` / `_g23h_mub`), at the best-fit draw -- NOT minus the
        # catalogue's Hipparcos–Gaia value, which carries the mean reflex over
        # the baseline and so shifted every point by the signal itself when
        # P ≫ 25 yr. The likelihood's own model at each epoch is drawn as an
        # open marker: the Hipparcos–Gaia epoch is a baseline-mean reflex and
        # the Gaia epoch a GOST-window average, neither a point on the
        # instantaneous curve.
        function _overlay!(ts, μra_obs, μdec_obs, σ_ra, σ_dec, μra_mod, μdec_mod, mb,
                           color, marker, label)
            μb1, μb2 = mb === nothing ? (0.0, 0.0) : mb[1:2]
            for (ax, obs, σ, mod, μb) in ((ax_ra, μra_obs, σ_ra, μra_mod, μb1),
                                          (ax_dec, μdec_obs, σ_dec, μdec_mod, μb2))
                errorbars!(ax, ts, obs .- μb, σ; color = :black, linewidth = ERRBAR_LW)
                scatter!(ax, ts, obs .- μb; color = color, marker = marker,
                         markersize = 13, strokewidth = 1.0, strokecolor = :black,
                         label = ax === ax_ra ? label : nothing)
                scatter!(ax, ts, collect(mod); color = (:white, 0.0), marker = marker,
                         markersize = 13, strokewidth = 1.5, strokecolor = color,
                         label = ax === ax_ra ? "$label model" : nothing)
            end
        end
        if data.hgca !== nothing
            hg = data.hgca
            μm = try
                Nereus._hgca_model_pm(theta_med, hg, data, M_pri, plx, data.t_ref)
            catch
                nothing
            end
            if μm !== nothing
                _overlay!(collect(hg.epochs), collect(hg.pmra), collect(hg.pmdec),
                          [sqrt(hg.cov_ep[k][1, 1]) for k in 1:3],
                          [sqrt(hg.cov_ep[k][2, 2]) for k in 1:3],
                          μm[1], μm[2], _hgca_mub(hg, μm[1], μm[2]),
                          :orangered, :diamond, "HGCA")
            end
        end
        if data.g23h !== nothing
            g = data.g23h
            μm = try
                Nereus._g23h_model_pm(theta_med, g, data, M_pri, plx, data.t_ref)
            catch
                nothing
            end
            if μm !== nothing
                _overlay!(collect(g.epochs), collect(g.pmra), collect(g.pmdec),
                          [sqrt(g.cov[2k-1, 2k-1]) for k in 1:5],
                          [sqrt(g.cov[2k, 2k]) for k in 1:5],
                          μm[1], μm[2], _g23h_mub(g, μm[1], μm[2]),
                          :seagreen, :utriangle, "G23H")
            end
        end

        axislegend(ax_ra; position = :rt, framevisible = false, labelsize = 14)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "pm_anomaly_K$planet_idx.$fmt"), fig;
                        save_pdf=save_pdf, px_per_unit=3)
        end
        return fig
    end
end


# =====================================================================
# 9. Epoch astrometry on the sky — abscissae along their scan axes
# =====================================================================

"""
    _iad_normal_points(iad, e, gap) -> NamedTuple

Group along-scan residuals `e` into normal points: consecutive transits of
the SAME instrument, at the SAME scan angle, no more than `gap` days apart.

For Gaia that is one field-of-view transit — the 8-9 CCD abscissae read
within ~40 s at an identical ψ; the next FoV transit is 106.5 min later, so
any `gap` between ~1 and ~60 min gives the same grouping. A Hipparcos IAD
record is already one abscissa per satellite orbit, so its groups are
singletons and a normal point IS the abscissa.

The scan-angle condition is not decoration: a weighted mean of residuals
measured along DIFFERENT directions is not a residual along any of them, so
a group never spans a change of ψ, whatever `gap` is.

Each normal point is the inverse-variance weighted mean residual with its
formal error `1/√Σσ⁻²`, at the weighted mean epoch, along the weighted mean
scan direction `u = (sin ψ, cos ψ)`.
"""
function _iad_normal_points(iad, e::AbstractVector, gap::Real)
    n = length(e)
    ord = sortperm(collect(zip(iad.inst, iad.t)))
    groups = Vector{Vector{Int}}()
    for j in ord
        if !isempty(groups)
            k = groups[end][end]
            if iad.inst[j] == iad.inst[k] && iad.t[j] - iad.t[k] <= gap &&
               abs(rem2pi(iad.psi[j] - iad.psi[k], RoundNearest)) < 1e-3
                push!(groups[end], j)
                continue
            end
        end
        push!(groups, [j])
    end
    m = length(groups)
    t̄, ē, σ̄, ux, uy = zeros(m), zeros(m), zeros(m), zeros(m), zeros(m)
    inst = zeros(Int, m)
    for (g, idx) in enumerate(groups)
        w  = [1 / iad.abscissa_err[j]^2 for j in idx]
        W  = sum(w)
        t̄[g] = sum(w[i] * iad.t[j] for (i, j) in enumerate(idx)) / W
        ē[g] = sum(w[i] * e[j]     for (i, j) in enumerate(idx)) / W
        σ̄[g] = 1 / sqrt(W)
        sx = sum(w[i] * sin(iad.psi[j]) for (i, j) in enumerate(idx))
        sy = sum(w[i] * cos(iad.psi[j]) for (i, j) in enumerate(idx))
        h  = hypot(sx, sy)
        ux[g], uy[g] = sx / h, sy / h
        inst[g] = iad.inst[idx[1]]
    end
    return (t = t̄, e = ē, σ = σ̄, ux = ux, uy = uy, inst = inst,
            size = length.(groups))
end

"""
    _iad_oc(theta, data) -> NamedTuple or nothing

The along-scan O−C the IAD likelihood leaves behind at `theta`, with the
orbits it subtracted: `(oc, active_ks, orbs, M_secs, plx)`.

Built from the likelihood's own helpers, never a copy of them:
`_iad_active_orbits` picks the companions (astrometric mode, active in
`theta`'s trans-dim state, coupled to the astrometry), `_iad_residuals!`
subtracts their reflex and the sampled parallax, `_iad_normal_equations!`
sets up the catalogue marginalisation and `_iad_marginalised_residuals!`
removes its solution. So `Σ(oc/σ)²` is `χ²_min` of `iad_log_likelihood`.
A copy of that design in the plotting code once went stale and emptied
`iad_residuals.png`; `plot_iad_residuals` and
`plot_epoch_astrometry_orbit` both come through here.

`nothing` when there is no IAD, or no more transits than the
marginalisation has parameters (where the likelihood returns 0).
"""
function _iad_oc(theta, data)
    iad = data.iad
    iad === nothing && return nothing
    n = n_iad(iad)
    n_inst = n_iad_inst(iad)
    n_q = Nereus._iad_n_q(n_inst)
    n > n_q || return nothing

    plx = astrom_plx(theta)
    active_ks, orbs, M_secs = Nereus._iad_active_orbits(theta, astrom_M_pri(theta),
                                                        plx, data.t_ref)
    r = Vector{Float64}(undef, n)
    Nereus._iad_residuals!(r, iad, orbs, M_secs, plx)
    pos_col = Nereus._iad_pos_cols(n_inst)
    A = zeros(n_q, n_q)
    v = zeros(n_q)
    Nereus._iad_normal_equations!(A, v, iad, r, iad.pm_factor, pos_col)
    chol = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A); check = false)
    # Rank-deficient design: the likelihood falls back to q ≡ 0, and so do we.
    q_opt = LinearAlgebra.issuccess(chol) ? (chol \ v) : zeros(n_q)
    oc = Nereus._iad_marginalised_residuals!(
        Vector{Float64}(undef, n), iad, r, q_opt, iad.pm_factor, pos_col)
    return (oc = oc, active_ks = active_ks, orbs = orbs, M_secs = M_secs, plx = plx)
end

"""
    _epoch_astrometry_oc(theta, data, planet_idx) -> NamedTuple or nothing

Everything `plot_epoch_astrometry_orbit` draws, before it is drawn: planet
`planet_idx`'s orbit and reflex mass as the likelihood builds them, and the
O−C of `_iad_oc` -- with EVERY other active companion already removed from
the data. `nothing` as for `_iad_oc`, or when `planet_idx` has no active
astrometric orbit at `theta`.
"""
function _epoch_astrometry_oc(theta, data, planet_idx::Int)
    fit = _iad_oc(theta, data)
    fit === nothing && return nothing
    ki = findfirst(==(planet_idx), fit.active_ks)
    ki === nothing && return nothing
    return (orb = fit.orbs[ki], M_sec = fit.M_secs[ki], oc = fit.oc, plx = fit.plx)
end

"""
    plot_epoch_astrometry_orbit(chains, params, data; planet_idx=1,
                                output=nothing, fmt=:png, save_pdf=false,
                                figsize=(1000, 1000), bf_cutoff=5.0,
                                normal_point_gap=0.01, n_track=1200)

The astrometric orbit of an epoch-astrometry target (Hipparcos IAD, Gaia DR4
along-scan) on the sky, with every abscissa placed along its own scan axis.

Epoch astrometry is ONE-DIMENSIONAL: a transit measures the abscissa `w` along
the scan direction `u = (sin ψ, cos ψ)` and nothing across it, so it pins the
photocentre to a line, not a point. The standard picture (Sahlmann et al.
2011, A&A 525, A95, Fig. 20; Holl et al. 2023, A&A 674, A10, Figs. 12-16, the
Gaia DR3 NSS convention) puts each measurement at the model position plus its
O−C along `u`:

    P = M(t) + (O − C)·u

and draws its error bar along `u` too. Across the scan the point sits exactly
where the model puts it, so scatter about the ellipse is information and
agreement across the scan is not evidence. This is a picture of the fit, not
2-D astrometry.

What is drawn:
  - the orbit at the MAXIMUM-log-posterior draw of the best-fit cluster.
    NOT the per-parameter median: on Gaia-4 the circular median of Ω lands
    off the ridge and χ²/N rises from 1.38 to 1.65. On a trans-dim chain the
    draw is taken among those where planet `planet_idx` is active, and that
    draw's active set decides which other companions are subtracted.
  - individual abscissae, small and grey;
  - normal points (see `_iad_normal_points`: one per Gaia field-of-view
    transit, `normal_point_gap` days), coloured by epoch, with a ±1σ bar
    along the scan axis and a dashed connector from the model position —
    the O−C itself, drawn along ψ;
  - the host star at the barycentre (gold star) and periastron (red
    diamond), and an arrow for the sense of motion;
  - below, the along-scan O−C against orbital phase (0 = periastron).

The O−C are the likelihood's own (`_epoch_astrometry_oc`, the path
`plot_iad_residuals` also takes), so with several companions this panel shows
planet `planet_idx`'s orbit with the others already removed from the data.

The annotation gives χ²/N over the abscissae (what the likelihood sees) and
over the normal points. With independent CCD errors both sit near 1, the
normal-point one a little lower (the fitted orbit and catalogue parameters
take a larger share of fewer points, ~0.1 at ~100 transits); a normal-point
χ²/N well ABOVE the per-abscissa one says the excess noise is correlated
within a transit.

When a Gaia DR3 five-parameter solution is also supplied (`data.gaia_dr3`
with `data.gost`), the fit marginalises the catalogue solution against it
as well; this figure, like `plot_iad_residuals`, uses the IAD-only
marginalisation.

Saves `models/epoch_astrometry_orbit_K<planet_idx>.<fmt>`. Returns the
`Figure`; an empty one (nothing saved) when there is no IAD, too few transits
for the marginalisation, planet `planet_idx` carries no astrometric orbit, or
a trans-dim chain never has it active.
"""
function plot_epoch_astrometry_orbit(chains, params, data;
                                      planet_idx::Int = 1,
                                      output::Union{Nothing, String} = nothing,
                                      fmt::Symbol = :png,
                                      save_pdf::Bool = false,
                                      figsize = (1000, 1000),
                                      bf_cutoff::Real = 5.0,
                                      normal_point_gap::Real = 0.01,
                                      n_track::Int = 1200)
    iad = data.iad
    iad === nothing && return Figure()
    n = n_iad(iad)
    n_inst = n_iad_inst(iad)
    n_q = Nereus._iad_n_q(n_inst)
    # The likelihood needs MORE than n_q transits (at n == n_q every orbit
    # fits exactly); below that there is no fit to picture.
    n > n_q || return Figure()

    # The max-lp draw among those in which this companion exists, with its
    # trans-dim active set -- so a parked slot is neither subtracted from
    # the O−C nor drawn (see `_planet_draw`).
    drawn = _planet_draw(chains, params, planet_idx; bf_cutoff = bf_cutoff)
    drawn === nothing && return Figure()         # never present: nothing to draw
    theta = first(drawn)

    fit = _epoch_astrometry_oc(theta, data, planet_idx)
    fit === nothing && return Figure()
    (; orb, M_sec, oc, plx) = fit

    np = _iad_normal_points(iad, oc, normal_point_gap)
    m  = length(np.t)
    has_raw = any(>(1), np.size)       # singletons: a normal point IS the abscissa

    with_theme(nereus_theme()) do
        # model positions, and each measurement along its scan axis
        st, ct = sin.(iad.psi), cos.(iad.psi)
        mod_raw = [star_reflex_offset(orb, iad.t[j], M_sec) for j in 1:n]
        x_raw = [mod_raw[j][1] + oc[j] * st[j] for j in 1:n]
        y_raw = [mod_raw[j][2] + oc[j] * ct[j] for j in 1:n]
        mod_np = [star_reflex_offset(orb, np.t[g], M_sec) for g in 1:m]
        xm = first.(mod_np); ym = last.(mod_np)
        xn = xm .+ np.e .* np.ux
        yn = ym .+ np.e .* np.uy

        P_d = _orbit_period_days(theta, planet_idx)
        t_first = minimum(iad.t)
        tp  = PlanetOrbits.periastron(orb)
        x_p, y_p = star_reflex_offset(orb, tp, M_sec)
        # Traced uniformly in ECCENTRIC anomaly, not in time: at high e a time
        # grid puts periastron between two samples and the line cuts the
        # corner, leaving the periastron marker off the drawn orbit.
        e_orb = PlanetOrbits.eccentricity(orb)
        tt  = [tp + P_d / 2π * (E - e_orb * sin(E))
               for E in range(0, 2π; length = n_track)]
        trk = [star_reflex_offset(orb, t, M_sec) for t in tt]
        x_t = first.(trk); y_t = last.(trk)
        a0  = abs(PlanetOrbits.semimajoraxis(orb) * M_sec /
                  PlanetOrbits.totalmass(orb) * plx)

        # Frame the orbit and the normal points, not the individual
        # abscissae: a handful of CCD outliers (HD 114762 has 14 beyond 5σ,
        # one at 41σ) otherwise set the limits and shrink the orbit into a
        # corner. The ones left outside are counted in the annotation.
        xs = vcat(x_t, xn .- np.σ .* abs.(np.ux), xn .+ np.σ .* abs.(np.ux))
        ys = vcat(y_t, yn .- np.σ .* abs.(np.uy), yn .+ np.σ .* abs.(np.uy))
        pad = 0.08 * max(maximum(xs) - minimum(xs), maximum(ys) - minimum(ys))
        xlo, xhi = minimum(xs) - pad, maximum(xs) + pad
        ylo, yhi = minimum(ys) - pad, maximum(ys) + pad
        n_out = has_raw ? count(j -> !(xlo <= x_raw[j] <= xhi && ylo <= y_raw[j] <= yhi), 1:n) : 0
        xspan, yspan = xhi - xlo, yhi - ylo

        # Size the figure to that frame's aspect (DataAspect), as
        # plot_orbit_skyplane does, so the sky panel fills its box and the
        # O−C strip below it is the same width.
        pw = float(figsize[1]) - 190
        ph = clamp(pw * yspan / max(xspan, eps()), 0.5pw, 1.3pw)
        rh = 0.3pw
        fig = Figure(; size = (round(Int, pw + 190), round(Int, ph + rh + 230)))
        ga = fig[1, 1] = GridLayout()
        ax = Axis(ga[1, 1];
                  xlabel = "ΔRA·cos δ (mas)",
                  ylabel = "Δδ (mas)",
                  aspect = DataAspect())
        _flip_xaxis!(ax)
        ax_r = Axis(ga[2, 1];
                    xlabel = "orbital phase",
                    ylabel = "O−C (mas)")
        rowsize!(ga, 2, rh)       # a Real is a fixed size; `Fixed` is taken by Nereus

        tcol   = np.t .- t_first
        t_span = max(maximum(tcol), 1.0)
        mk     = [sky_inst_marker(i) for i in np.inst]

        lines!(ax, x_t, y_t; color = NEREUS_COLORS.model, linewidth = 2,
               label = "Best fit")
        # sense of motion: a short arrow a tenth of a period past periastron
        let t_a = tp + 0.1 * P_d, dt = P_d / 200
            p1 = star_reflex_offset(orb, t_a, M_sec)
            p2 = star_reflex_offset(orb, t_a + dt, M_sec)
            d  = (p2[1] - p1[1], p2[2] - p1[2])
            h  = hypot(d...)
            # h == 0 when the reflex mass is zero (an SB2 whose light ratio
            # equals its mass ratio): no motion, so no arrow, not a NaN one.
            if h > 0
                s = 0.06 * max(xspan, yspan) / h
                arrows2d!(ax, [Point2f(p1...)], [Vec2f(s * d[1], s * d[2])];
                          color = NEREUS_COLORS.model, shaftwidth = 2,
                          tipwidth = 12, tiplength = 12)
            end
        end
        if has_raw
            scatter!(ax, x_raw, y_raw; color = (:gray, 0.35), markersize = 4,
                     strokewidth = 0, label = "Individual abscissae")
        end
        # O−C connectors, model → normal point, along the scan axis
        seg = Point2f[]
        for g in 1:m
            push!(seg, Point2f(xm[g], ym[g]), Point2f(xn[g], yn[g]))
        end
        linesegments!(ax, seg; color = (:gray30, 0.8), linewidth = 0.9,
                      linestyle = :dash)
        # ±1σ along the scan axis
        bar = Point2f[]
        for g in 1:m
            dx, dy = np.σ[g] * np.ux[g], np.σ[g] * np.uy[g]
            push!(bar, Point2f(xn[g] - dx, yn[g] - dy), Point2f(xn[g] + dx, yn[g] + dy))
        end
        linesegments!(ax, bar; color = :black, linewidth = ERRBAR_LW)
        sc = scatter!(ax, xn, yn; color = tcol, colormap = NEREUS_CMAP,
                      colorrange = (0, t_span), marker = mk, markersize = 11,
                      strokewidth = 1.0, strokecolor = :black,
                      label = has_raw ? "Normal points" : "Abscissae")
        scatter!(ax, [x_p], [y_p]; color = :red, marker = :diamond, markersize = 18,
                 strokewidth = 1.5, strokecolor = :black, label = "Periastron")
        scatter!(ax, [0.0], [0.0]; color = :gold, marker = :star5, markersize = 22,
                 strokewidth = 1.5, strokecolor = :black, label = "Host star (barycentre)")
        limits!(ax, xlo, xhi, ylo, yhi)

        # along-scan O−C of the normal points vs orbital phase. The individual
        # abscissae stay out: their scatter would set the scale and flatten the
        # normal points onto the zero line (plot_iad_residuals shows them).
        ph_np = mod.((np.t .- tp) ./ P_d, 1.0)
        errorbars!(ax_r, ph_np, np.e, np.σ; color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_r, ph_np, np.e; color = tcol, colormap = NEREUS_CMAP,
                 colorrange = (0, t_span), marker = mk, markersize = 9,
                 strokewidth = 0.8, strokecolor = :black)
        hlines!(ax_r, 0; color = NEREUS_COLORS.zero_line, linestyle = :dash,
                linewidth = 1.5)
        xlims!(ax_r, 0, 1)

        Colorbar(ga[1:2, 2], sc; label = "MJD − $(round(Int, t_first))")
        Legend(ga[3, 1], ax; framevisible = false, labelsize = 14,
               orientation = :horizontal, tellheight = true, tellwidth = false,
               nbanks = 2)

        χ2_raw = sum(abs2, oc ./ iad.abscissa_err) / n
        χ2_np  = sum(abs2, np.e ./ np.σ) / m
        txt = has_raw ?
            @sprintf("%d abscissae in %d normal points\na₀ = %.3f mas\nχ²/N = %.2f (abscissae), %.2f (normal points)",
                     n, m, a0, χ2_raw, χ2_np) :
            @sprintf("%d abscissae\na₀ = %.3f mas\nχ²/N = %.2f", n, a0, χ2_raw)
        n_out > 0 && (txt *= "\n$n_out abscissa$(n_out == 1 ? "" : "e") outside the frame")
        text!(ax, 0.02, 0.98; text = txt, space = :relative,
              align = (:left, :top), fontsize = 15)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models",
                                "epoch_astrometry_orbit_K$(planet_idx).$fmt"), fig;
                       save_pdf = save_pdf, px_per_unit = 3)
        end
        fig
    end
end
