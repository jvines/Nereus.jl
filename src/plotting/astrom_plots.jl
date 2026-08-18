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
    _theta_from_chain_row(chains, params, idx) -> Theta

Build a `Theta{Float64}` populated from row `idx` of `chains`. Frozen
slots are auto-populated by the Theta constructor; only chain-present
parameters are set.
"""
function _theta_from_chain_row(chains, params, idx::Int)
    chain_names = Set(names(chains, :parameters))
    theta = Theta{Float64}(params)
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in chain_names || continue
        v = vec(Array(chains[sym]))[idx]
        set_param!(theta, name, v)
    end
    return theta
end

"""
    _theta_median(chains, params; active_idx=nothing) -> Theta

Posterior-median Theta. If `active_idx` is given, takes the median over
that subset only (used for trans-dim chains conditioned on planet
activation).
"""
function _theta_median(chains, params; active_idx = nothing)
    chain_names = Set(names(chains, :parameters))
    theta = Theta{Float64}(params)
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
    _theta_best_lp(chains, params; active_idx=nothing) -> Theta

Theta at the MAXIMUM-log-posterior sample (within `active_idx` when
given). This is the EMPEROR best-fit convention and the right CENTRAL
CURVE for overlays: it is a real, smooth model on the posterior ridge.
The per-parameter marginal-median theta is NOT (circular Ω/Mo and the
sesinw/secosw mapping put it off-ridge — its curve sat ~90 mas off the
HD 159062 fan), and a pointwise predictive median is an order statistic
that kinks wherever the draw curves cross. Falls back to `_theta_median`
when the chain carries no `:lp`.
"""
function _theta_best_lp(chains, params; active_idx = nothing)
    chain_names = Set(names(chains, :parameters))
    (:lp in chain_names) ||
        return _theta_median(chains, params; active_idx = active_idx)
    lp = vec(Array(chains[:lp]))
    idxs = active_idx === nothing ? (1:length(lp)) : active_idx
    best = idxs[argmax(@view lp[idxs])]
    theta = Theta{Float64}(params)
    for name in params.layout.unfrozen_names
        sym = Symbol(name)
        sym in chain_names || continue
        set_param!(theta, name, vec(Array(chains[sym]))[best])
    end
    return theta
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
the posterior median orbit + a faint fan of `n_draws` posterior-sampled
orbits.

The host star sits at the origin (marked with a star glyph). The
periastron direction of the median orbit is annotated. The x-axis is
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

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ga = fig[1, 1] = GridLayout()
        ax = Axis(ga[1, 1];
                   xlabel = rich("ΔRA·cos δ (mas)"),
                   ylabel = "Δδ (mas)",
                   aspect = DataAspect())
        _flip_xaxis!(ax)

        # Trans-dim conditioning
        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            active_idx = findall(np_all .>= planet_idx)
            isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        else
            active_idx = collect(1:_n_flat_draws(chains))
        end
        # EMPEROR best-fit cluster — samples within ln(bf_cutoff) of max lp
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))

        n_samples = length(active_idx)

        # ---- 1. Posterior fan -------------------------------------
        ndraw = min(n_draws, n_samples)
        if ndraw > 0
            draw_idx = active_idx[rand(1:n_samples, ndraw)]
            for i in draw_idx
                theta = _theta_from_chain_row(chains, params, i)
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
        theta_med = _theta_best_lp(chains, params; active_idx = active_idx)
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
                              marker = inst_marker(i),
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
                      color = :red, marker = :star5,
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

Two-panel along-scan residual diagnostic for Hipparcos IAD data.

For each IAD transit `j`:
  residual_j = abscissa_j  −  Δη_orbit_j(theta_med)  −  X_jᵀ q_opt

where `Δη_orbit_j` is the orbit-induced along-scan reflex at the
median theta, and `q_opt` is the analytic best-fit 5-parameter catalog
correction obtained from the same Cholesky factorization used inside
`iad_log_likelihood`. Per-transit `σ_j` is shown as errorbars.

Panels:
  top    — residual vs ψ (scan position angle, rad)
  bottom — residual vs MJD

Returns the `Figure`. No-op (returns empty Figure) if `data.iad` is
nothing or has fewer than 5 transits.
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
    n >= 5 || (return Figure())

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ax_psi = Axis(fig[1, 1];
                       xlabel = "ψ (rad)",
                       ylabel = "abscissa residual (mas)")
        ax_t   = Axis(fig[2, 1];
                       xlabel = "MJD",
                       ylabel = "abscissa residual (mas)")

        # Best-fit cluster median
        chain_names = Set(names(chains, :parameters))
        active_idx = collect(1:_n_flat_draws(chains))
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        theta_med = _theta_median(chains, params; active_idx = active_idx)

        M_pri  = astrom_M_pri(theta_med)
        plx    = astrom_plx(theta_med)
        t_ref  = data.t_ref

        # Per-planet orbits at median theta (only astrometry-active)
        active_orbs = Any[]
        active_Msec = Float64[]
        for k in planet_indices(theta_med)
            block = theta_med.params.layout.planet_blocks[k]
            has_AS(block) || continue
            try
                orb_k, M_sec_k = Nereus._planet_orbit(theta_med, k, M_pri, plx, t_ref)
                push!(active_orbs, orb_k)
                push!(active_Msec, M_sec_k)
            catch
                # skip
            end
        end

        # Stage 1: per-transit orbit-subtracted residuals
        r = Vector{Float64}(undef, n)
        @inbounds for j in 1:n
            t_j = iad.t[j]
            ψ_j = iad.psi[j]
            Δη_mod = 0.0
            for ki in eachindex(active_orbs)
                Δra, Δdec = star_reflex_offset(active_orbs[ki], t_j, active_Msec[ki])
                Δη_mod += along_scan_projection(Δra, Δdec, ψ_j)
            end
            r[j] = iad.abscissa[j] - Δη_mod
        end

        # Stage 2: solve for the 5-param marginalized correction.
        # Same design as iad_log_likelihood:
        #   x_j = (sin ψ, cos ψ, plxf, sin ψ · pmf, cos ψ · pmf)
        A = zeros(5, 5)
        v = zeros(5)
        @inbounds for j in 1:n
            s, c = sincos(iad.psi[j])
            plxf = iad.parallax_factor[j]
            pmf  = iad.pm_factor[j]
            xj = (s, c, plxf, s*pmf, c*pmf)
            w  = 1.0 / iad.abscissa_err[j]^2
            rj = r[j]
            for a in 1:5
                v[a] += w * xj[a] * rj
                for b in 1:5
                    A[a, b] += w * xj[a] * xj[b]
                end
            end
        end
        chol = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A); check = false)
        q_opt = LinearAlgebra.issuccess(chol) ? (chol \ v) : zeros(5)

        # Stage 3: final residuals after catalog-correction marginalization
        resid = Vector{Float64}(undef, n)
        @inbounds for j in 1:n
            s, c = sincos(iad.psi[j])
            plxf = iad.parallax_factor[j]
            pmf  = iad.pm_factor[j]
            corr = q_opt[1]*s + q_opt[2]*c + q_opt[3]*plxf +
                   q_opt[4]*s*pmf + q_opt[5]*c*pmf
            resid[j] = r[j] - corr
        end
        σs = iad.abscissa_err

        # Plot vs ψ
        errorbars!(ax_psi, iad.psi, resid, σs;
                    color = :black, linewidth = ERRBAR_LW)
        scatter!(ax_psi, iad.psi, resid;
                  color = iad.t .- minimum(iad.t),
                  colormap = NEREUS_CMAP,
                  colorrange = (0, maximum(iad.t) - minimum(iad.t)),
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
    plot_pm_residuals(chains, params, data;
                       n_draws=100, planet_idx=1,
                       output=nothing, fmt=:png, figsize=(1100, 900))

HGCA proper-motion observed-vs-modelled at the three reference epochs
(Hipparcos / Hipparcos-Gaia / Gaia). Each panel shows:

  - the observed (μ_α*, μ_δ) point with its 1-σ within-epoch covariance
    ellipse,
  - a cloud of `n_draws` posterior-drawn model reflex PM points,
  - the posterior median model PM (and its barycentre-marginalized
    correction is *not* applied — these are raw reflex contributions
    at the catalog-tabulated epoch).

A title summary shows the χ² of the median orbit against the three
HGCA points (sum of within-epoch Mahalanobis distances; the
barycentric μ_b is *not* re-marginalized here, this is for visual
inspection only).

Requires `data.hgca` to be non-`nothing`.
"""
function plot_pm_residuals(chains, params, data;
                            n_draws::Int = 100,
                            planet_idx::Int = 1,
                            output::Union{Nothing, String} = nothing,
                            fmt::Symbol = :png,
                            save_pdf::Bool = false,
                            figsize = (1100, 900),
                            bf_cutoff::Real = 10.0)
    hgca = data.hgca
    hgca === nothing && throw(ArgumentError(
        "plot_pm_residuals requires data.hgca (got nothing)"))

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)

        labels = ("Hipparcos", "Hipparcos–Gaia", "Gaia")
        colors = (colorant"#1f77b4", colorant"#2ca02c", colorant"#d62728")

        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            active_idx = findall(np_all .>= planet_idx)
            isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        else
            active_idx = collect(1:_n_flat_draws(chains))
        end
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))

        n_samples = length(active_idx)
        ndraw = min(n_draws, n_samples)
        draw_idx = ndraw > 0 ? active_idx[rand(1:n_samples, ndraw)] :
                                Int[]

        # Compute model μ_α*, μ_δ at each HGCA epoch for each draw and
        # for the median.
        μra_draws  = [Float64[] for _ in 1:3]
        μdec_draws = [Float64[] for _ in 1:3]

        for i in draw_idx
            theta = _theta_from_chain_row(chains, params, i)
            M_pri = astrom_M_pri(theta)
            plx   = astrom_plx(theta)
            try
                orb, M_sec = Nereus._planet_orbit(theta, planet_idx,
                                                    M_pri, plx, data.t_ref)
                @inbounds for k in 1:3
                    μr, μd = star_reflex_pm(orb, hgca.epochs[k], M_sec)
                    if isfinite(μr) && isfinite(μd)
                        push!(μra_draws[k], μr)
                        push!(μdec_draws[k], μd)
                    end
                end
            catch
                # Skip degenerate samples
            end
        end

        # Median orbit reflex
        theta_med = _theta_median(chains, params; active_idx = active_idx)
        med_μra = zeros(3); med_μdec = zeros(3)
        med_ok = true
        try
            M_pri = astrom_M_pri(theta_med)
            plx   = astrom_plx(theta_med)
            orb, M_sec = Nereus._planet_orbit(theta_med, planet_idx,
                                                M_pri, plx, data.t_ref)
            @inbounds for k in 1:3
                μr, μd = star_reflex_pm(orb, hgca.epochs[k], M_sec)
                med_μra[k]  = μr
                med_μdec[k] = μd
            end
        catch
            med_ok = false
        end

        # Analytically marginalized barycentric μ_b (RA, Dec) — mirrors
        # the closed-form solution inside hgca_log_likelihood. Without
        # this subtraction the absolute observed PM (~100s of mas/yr)
        # buries the orbital reflex (~mas/yr) on the plot.
        μb1 = 0.0; μb2 = 0.0; chi2_total = 0.0
        if med_ok
            A11 = 0.0; A12 = 0.0; A22 = 0.0
            v1  = 0.0; v2  = 0.0
            rCr = 0.0
            @inbounds for k in 1:3
                a = hgca.cov_ep[k][1, 1]
                b = hgca.cov_ep[k][1, 2]
                c = hgca.cov_ep[k][2, 2]
                detC = a * c - b * b
                detC > 0 || continue
                invC11 =  c / detC
                invC12 = -b / detC
                invC22 =  a / detC
                rra  = hgca.pmra[k]  - med_μra[k]
                rdec = hgca.pmdec[k] - med_μdec[k]
                rCr += invC11 * rra * rra + 2 * invC12 * rra * rdec +
                       invC22 * rdec * rdec
                v1  += invC11 * rra + invC12 * rdec
                v2  += invC12 * rra + invC22 * rdec
                A11 += invC11; A12 += invC12; A22 += invC22
            end
            detA = A11 * A22 - A12 * A12
            if detA > 0
                μb1 = ( A22 * v1 - A12 * v2) / detA
                μb2 = (-A12 * v1 + A11 * v2) / detA
                chi2_total = rCr - (v1 * μb1 + v2 * μb2)
            end
        end

        # Per-epoch residuals + 1-σ errors (diagonal σ from cov_ep)
        ts_ep    = collect(hgca.epochs)
        res_ra   = Float64[hgca.pmra[k]  - med_μra[k]  - μb1 for k in 1:3]
        res_dec  = Float64[hgca.pmdec[k] - med_μdec[k] - μb2 for k in 1:3]
        σ_ra_ep  = Float64[sqrt(hgca.cov_ep[k][1, 1]) for k in 1:3]
        σ_dec_ep = Float64[sqrt(hgca.cov_ep[k][2, 2]) for k in 1:3]

        ax_ra  = Axis(fig[1, 1];
                       ylabel = rich("Δμ", subscript("α*"), " [mas/yr]"))
        ax_dec = Axis(fig[2, 1]; xlabel = "MJD",
                                   ylabel = rich("Δμ", subscript("δ"), " [mas/yr]"))
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

        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            active_idx = findall(np_all .>= planet_idx)
            isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        else
            active_idx = collect(1:_n_flat_draws(chains))
        end
        # EMPEROR best-fit cluster (BF = 5; samples within ln 5 of
        # max log-posterior) — median + fan from this cluster only.
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))

        # Reference theta = max-lp draw (EMPEROR best-fit): a real smooth
        # model ON the posterior ridge — see `_theta_best_lp`.
        theta_med = _theta_best_lp(chains, params; active_idx = active_idx)
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

        # Trans-dim + bf-cluster active draws
        chain_names = Set(names(chains, :parameters))
        active_idx = collect(1:_n_flat_draws(chains))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            active_idx = findall(np_all .>= planet_idx)
            isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        end
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))

        theta_med = _theta_best_lp(chains, params; active_idx = active_idx)
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
                theta = _theta_from_chain_row(chains, params, i)
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

        active_idx = collect(1:_n_flat_draws(chains))
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        theta_med = _theta_median(chains, params; active_idx = active_idx)
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
        ax_ra  = Axis(fig[1, 1]; ylabel = "Δμ_α* (mas/yr)")
        ax_dec = Axis(fig[2, 1]; xlabel = "MJD",
                                   ylabel = "Δμ_δ (mas/yr)")
        linkxaxes!(ax_ra, ax_dec)
        hidexdecorations!(ax_ra; grid = false, ticks = false)

        active_idx = collect(1:_n_flat_draws(chains))
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        theta_med = _theta_median(chains, params; active_idx = active_idx)
        M_pri = astrom_M_pri(theta_med)
        plx   = astrom_plx(theta_med)
        t_ref = data.t_ref

        # Sum of reflex PMs across astrometry-active companions
        active_orbs = Any[]
        active_Msec = Float64[]
        for k in planet_indices(theta_med)
            block = theta_med.params.layout.planet_blocks[k]
            has_AS(block) || continue
            try
                orb_k, M_sec_k = Nereus._planet_orbit(theta_med, k,
                                                       M_pri, plx, t_ref)
                push!(active_orbs, orb_k)
                push!(active_Msec, M_sec_k)
            catch
            end
        end

        ts = collect(g23h.epochs)
        μra_obs  = collect(g23h.pmra)
        μdec_obs = collect(g23h.pmdec)
        # Diagonal σs from the 10x10 covariance, ordering [pmra_k, pmdec_k]_k=1..5
        σ_ra  = [sqrt(g23h.cov[2k-1, 2k-1]) for k in 1:5]
        σ_dec = [sqrt(g23h.cov[2k,   2k  ]) for k in 1:5]

        μra_mod  = zeros(5)
        μdec_mod = zeros(5)
        for k in 1:5
            for q in eachindex(active_orbs)
                μ_a, μ_d = star_reflex_pm(active_orbs[q], ts[k], active_Msec[q])
                μra_mod[k]  += μ_a
                μdec_mod[k] += μ_d
            end
        end

        # Analytically marginalize the system barycentric PM μ_b — same
        # closed-form solution used inside g23h_log_likelihood. Without
        # this, residuals are dominated by the star's barycentric motion
        # (orders of magnitude larger than any orbital reflex).
        r10 = Vector{Float64}(undef, 10)
        @inbounds for k in 1:5
            r10[2k - 1] = μra_obs[k]  - μra_mod[k]
            r10[2k    ] = μdec_obs[k] - μdec_mod[k]
        end
        cholΣ = cholesky(Symmetric(Matrix{Float64}(g23h.cov)); check = false)
        μb1 = 0.0; μb2 = 0.0
        if LinearAlgebra.issuccess(cholΣ)
            Σinv_r = cholΣ \ r10
            v1 = 0.0; v2 = 0.0
            @inbounds for k in 1:5
                v1 += Σinv_r[2k - 1]
                v2 += Σinv_r[2k    ]
            end
            X = zeros(10, 2)
            @inbounds for k in 1:5
                X[2k - 1, 1] = 1.0
                X[2k    , 2] = 1.0
            end
            Y = cholΣ \ X
            A11 = 0.0; A12 = 0.0; A22 = 0.0
            @inbounds for k in 1:5
                A11 += Y[2k - 1, 1]
                A12 += Y[2k - 1, 2]
                A22 += Y[2k    , 2]
            end
            detA = A11 * A22 - A12 * A12
            if detA > 0
                μb1 = ( A22 * v1 - A12 * v2) / detA
                μb2 = (-A12 * v1 + A11 * v2) / detA
            end
        end
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
  top    — μ_α*_reflex(t) [mas/yr]   vs MJD
  bottom — μ_δ_reflex(t)  [mas/yr]   vs MJD

Posterior median curve in cyan; `n_draws` posterior fan in orchid.
HGCA / G23H observed PMs overlaid as colored markers with errorbars
where available, at their tabulated epochs.

Requires `data.gost`. Other companions' reflex contributions are
summed.
"""
function plot_pm_anomaly(chains, params, data;
                          n_draws::Int = 80,
                          planet_idx::Int = 1,
                          output::Union{Nothing, String} = nothing,
                          fmt::Symbol = :png,
                          save_pdf::Bool = false,
                          figsize = (1100, 900),
                          bf_cutoff::Real = 5.0)
    gost = data.gost
    gost === nothing && return Figure()

    with_theme(nereus_theme()) do
        fig = Figure(; size = figsize)
        ax_ra  = Axis(fig[1, 1]; ylabel = "μ_α* reflex (mas/yr)")
        ax_dec = Axis(fig[2, 1]; xlabel = "MJD",
                                   ylabel = "μ_δ reflex (mas/yr)")
        linkxaxes!(ax_ra, ax_dec)
        hidexdecorations!(ax_ra; grid = false, ticks = false)

        active_idx = collect(1:_n_flat_draws(chains))
        chain_names = Set(names(chains, :parameters))
        if :n_planets in chain_names
            np_all = vec(Array(chains[:n_planets]))
            active_idx = findall(np_all .>= planet_idx)
            isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))
        end
        active_idx = intersect(active_idx, _top_lp_draw_pool(chains; bf_cutoff=bf_cutoff))
        isempty(active_idx) && (active_idx = collect(1:_n_flat_draws(chains)))

        theta_med = _theta_median(chains, params; active_idx = active_idx)

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
                theta = _theta_from_chain_row(chains, params, i)
                P = _orbit_period_days(theta, planet_idx)
                (isfinite(P) && P > 0) || continue
                try
                    M_pri_i = astrom_M_pri(theta)
                    plx_i   = astrom_plx(theta)
                    orb_i, M_sec_i = Nereus._planet_orbit(theta, planet_idx,
                                                            M_pri_i, plx_i,
                                                            data.t_ref)
                    @inbounds for k in 1:ng
                        μa, μd = star_reflex_pm(orb_i, t_grid[k], M_sec_i)
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

        # --- Median curve (sum over astrometry-active companions) ---
        M_pri = astrom_M_pri(theta_med)
        plx   = astrom_plx(theta_med)
        active_orbs = Any[]
        active_Msec = Float64[]
        for k in planet_indices(theta_med)
            block = theta_med.params.layout.planet_blocks[k]
            has_AS(block) || continue
            try
                orb_k, M_sec_k = Nereus._planet_orbit(theta_med, k,
                                                       M_pri, plx, data.t_ref)
                push!(active_orbs, orb_k)
                push!(active_Msec, M_sec_k)
            catch
            end
        end
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
                    label = "Posterior median")
            lines!(ax_dec, t_grid, μdec_m;
                    color = NEREUS_COLORS.model, linewidth = 2.0)
        end

        # --- Catalog overlays (HGCA / G23H) ---
        if data.hgca !== nothing
            hg = data.hgca
            ts = collect(hg.epochs)
            σ_ra  = [sqrt(hg.cov_ep[k][1, 1]) for k in 1:3]
            σ_dec = [sqrt(hg.cov_ep[k][2, 2]) for k in 1:3]
            # HGCA tabulates absolute μ; we need to subtract the
            # barycentric μ to compare with reflex-only model. Without
            # the barycentre we plot anomaly relative to the HG epoch
            # (the "long-baseline" PM is by construction the barycentre).
            μra_bary  = hg.pmra[2]
            μdec_bary = hg.pmdec[2]
            μra_anom  = collect(hg.pmra)  .- μra_bary
            μdec_anom = collect(hg.pmdec) .- μdec_bary
            errorbars!(ax_ra,  ts, μra_anom,  σ_ra;
                        color = :black, linewidth = ERRBAR_LW)
            scatter!(ax_ra, ts, μra_anom;
                      color = :orangered, marker = :diamond,
                      markersize = 13, strokewidth = 1.0, strokecolor = :black,
                      label = "HGCA")
            errorbars!(ax_dec, ts, μdec_anom, σ_dec;
                        color = :black, linewidth = ERRBAR_LW)
            scatter!(ax_dec, ts, μdec_anom;
                      color = :orangered, marker = :diamond,
                      markersize = 13, strokewidth = 1.0, strokecolor = :black)
        end
        if data.g23h !== nothing
            g = data.g23h
            ts = collect(g.epochs)
            σ_ra  = [sqrt(g.cov[2k-1, 2k-1]) for k in 1:5]
            σ_dec = [sqrt(g.cov[2k,   2k  ]) for k in 1:5]
            μra_bary  = g.pmra[2]
            μdec_bary = g.pmdec[2]
            μra_anom  = collect(g.pmra)  .- μra_bary
            μdec_anom = collect(g.pmdec) .- μdec_bary
            errorbars!(ax_ra,  ts, μra_anom,  σ_ra;
                        color = :black, linewidth = ERRBAR_LW)
            scatter!(ax_ra, ts, μra_anom;
                      color = :seagreen, marker = :utriangle,
                      markersize = 13, strokewidth = 1.0, strokecolor = :black,
                      label = "G23H")
            errorbars!(ax_dec, ts, μdec_anom, σ_dec;
                        color = :black, linewidth = ERRBAR_LW)
            scatter!(ax_dec, ts, μdec_anom;
                      color = :seagreen, marker = :utriangle,
                      markersize = 13, strokewidth = 1.0, strokecolor = :black)
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


"""
    plot_epoch_astrometry_orbit(chains, params, data; output=nothing, ...)

The astrometric orbit of an epoch-astrometry (IAD / Gaia DR4 along-scan) target,
with the measurements placed on it.

Gaia epoch astrometry is ONE-DIMENSIONAL: each transit measures an along-scan
abscissa `w` at scan angle ψ, which constrains the photocentre to a LINE on the
sky, not a point. `plot_orbit_skyplane` therefore has nothing to overlay for
such a target and can only draw a bare model ellipse.

This function uses the standard reconstruction (Panuzzo et al. 2024, A&A,
arXiv:2404.10486, their Fig. 2 — "the position of the photocentre on the sky
corresponding to each measurement is derived combining the measured
one-dimensional AL position and the assumed orbital solution"): put each epoch
on its own measurement line, at the point closest to the model orbit.

    offset = w_obs − (Δα*_mod·sinψ + Δδ_mod·cosψ)
    Δα*_rec = Δα*_mod + offset·sinψ
    Δδ_rec  = Δδ_mod  + offset·cosψ

CAVEAT, and it is a real one: the reconstructed points can only depart from the
model ALONG the scan direction. Across-scan they sit wherever the model puts
them. Scatter about the ellipse is therefore informative; apparent agreement
across-scan is not evidence. This is a visualisation of the fit, not raw 2-D
astrometry.

The parallax + proper motion are removed first by re-fitting the 5-parameter
solution to the ORBIT-SUBTRACTED abscissae, so the astrometric parameters do
not absorb the orbit being displayed.

Returns the `Figure`. No-op (empty `Figure`) if `data.iad` is nothing or has
fewer than 5 transits.
"""
function plot_epoch_astrometry_orbit(chains, params, data;
                                      planet_idx::Int = 1,
                                      output::Union{Nothing, String} = nothing,
                                      fmt::Symbol = :png,
                                      save_pdf::Bool = false,
                                      figsize = (1000, 1000),
                                      n_track::Int = 1200,
                                      n_phase_bins::Int = 0)
    iad = data.iad
    iad === nothing && (return Figure())
    n = n_iad(iad)
    n >= 5 || (return Figure())

    with_theme(nereus_theme()) do
        theta_med = _theta_median(chains, params)
        M_pri = astrom_M_pri(theta_med)
        plx   = astrom_plx(theta_med)
        orb, M_sec = Nereus._planet_orbit(theta_med, planet_idx, M_pri, plx, data.t_ref)

        # model photocentre offsets at the observed epochs
        Δra_m  = Vector{Float64}(undef, n)
        Δdec_m = Vector{Float64}(undef, n)
        for j in 1:n
            Δra_m[j], Δdec_m[j] = star_reflex_offset(orb, iad.t[j], M_sec)
        end
        model_al = [along_scan_projection(Δra_m[j], Δdec_m[j], iad.psi[j]) for j in 1:n]

        # 5-param solution re-fitted to the orbit-subtracted abscissae
        st, ct = sin.(iad.psi), cos.(iad.psi)
        D = hcat(st, ct, iad.parallax_factor, st .* iad.pm_factor, ct .* iad.pm_factor)
        W = 1.0 ./ iad.abscissa_err
        p = (D .* W) \ ((iad.abscissa .- model_al) .* W)
        w_obs = iad.abscissa .- D * p              # observed ORBITAL along-scan signal

        # place each epoch on its measurement line, closest to the model
        off    = w_obs .- model_al
        Δra_r  = Δra_m  .+ off .* st
        Δdec_r = Δdec_m .+ off .* ct

        # model ellipse over one full period
        P_d = _orbit_period_days(theta_med, planet_idx)
        tt  = range(iad.t[1], iad.t[1] + P_d; length = n_track)
        trk = [star_reflex_offset(orb, t, M_sec) for t in tt]
        Δra_t  = [x[1] for x in trk]
        Δdec_t = [x[2] for x in trk]

        fig = Figure(; size = figsize)
        ax = Axis(fig[1, 1];
                  xlabel = rich("ΔRA·cos δ (mas)"),
                  ylabel = "Δδ (mas)",
                  aspect = DataAspect())
        _flip_xaxis!(ax)

        lines!(ax, Δra_t, Δdec_t; color = NEREUS_COLORS.model, linewidth = 2)
        # line of nodes + barycentre
        scatter!(ax, [0.0], [0.0]; color = :gold, marker = :cross,
                 markersize = 18, strokewidth = 1, strokecolor = :black)
        tcol = iad.t .- minimum(iad.t)
        if n_phase_bins > 0
            # LOW-SNR MODE. When a0 is only a few × the per-transit error the
            # reconstructed cloud swamps the ellipse. Bin in ORBITAL PHASE and
            # plot the per-bin median: N transits per bin buy √N, and the
            # ellipse becomes legible without hiding the scatter (raw points
            # stay underneath, faint). Medians, not means — a handful of
            # discrepant transits shouldn't drag a bin off the orbit.
            P_b = _orbit_period_days(theta_med, planet_idx)
            ph  = mod.((iad.t .- iad.t[1]) ./ P_b, 1.0)
            scatter!(ax, Δra_r, Δdec_r; color = (:gray, 0.20), markersize = 4)
            edges = range(0, 1; length = n_phase_bins + 1)
            bra, bdec, bph = Float64[], Float64[], Float64[]
            for b in 1:n_phase_bins
                m = (ph .>= edges[b]) .& (ph .< edges[b + 1])
                count(m) >= 3 || continue
                push!(bra,  median(Δra_r[m]))
                push!(bdec, median(Δdec_r[m]))
                push!(bph,  0.5 * (edges[b] + edges[b + 1]))
            end
            sc = scatter!(ax, bra, bdec; color = bph, colormap = NEREUS_CMAP,
                          colorrange = (0.0, 1.0), markersize = 17,
                          strokewidth = 0.9, strokecolor = :black)
            Colorbar(fig[1, 2], sc; label = "orbital phase")
        else
            # residual sticks: model → reconstructed, i.e. the along-scan miss
            for j in 1:n
                lines!(ax, [Δra_m[j], Δra_r[j]], [Δdec_m[j], Δdec_r[j]];
                       color = (:gray, 0.45), linewidth = 0.7)
            end
            sc = scatter!(ax, Δra_r, Δdec_r; color = tcol, colormap = NEREUS_CMAP,
                          markersize = 9, strokewidth = 0.4, strokecolor = :black)
            Colorbar(fig[1, 2], sc; label = "MJD − $(round(Int, minimum(iad.t)))")
        end

        rms = sqrt(sum(abs2, off) / n)
        text!(ax, 0.02, 0.98;
              text = @sprintf("%d transits\na₀ ≈ %.2f mas\nalong-scan residual RMS %.3f mas",
                              n, maximum(hypot.(Δra_t, Δdec_t)), rms),
              space = :relative, align = (:left, :top), fontsize = 16)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models", "epoch_astrometry_orbit.$fmt"), fig;
                        save_pdf = save_pdf, px_per_unit = 3)
        end
        fig
    end
end
