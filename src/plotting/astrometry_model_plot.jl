# The two figures of an epoch-astrometry fit: the reflex orbit on the sky
# against the abscissae of every mission that constrains it.
#
#   plot_astrometry_model        orbit, a zoom on the best mission, and each
#                                mission's O−C against EPOCH
#   plot_epoch_astrometry_orbit  orbit, and each mission's O−C against orbital
#                                PHASE
#
# They picture ONE fit and so share ONE scene (`_astrometry_model_scene`) and
# ONE way of drawing it (`_am_draw_sky!`, `_am_draw_oc!`): the same draw, the
# same binning, the same line for an abscissa, the same colours, markers and
# legend. They used to be two independent drawings and disagreed on all of
# those -- lines in one and points in the other, 19 binned Hipparcos abscissae
# in one and 37 in the other. What differs between them now is only what each
# is FOR: the missions separated in time, or one orbit folded in phase.
#
# astroEMPEROR's `astrometry_model` (the Feng et al. layout) shows the orbit
# with Hipparcos and the Gaia CATALOGUES: DR2/DR3 positions and proper-motion
# wedges, and the GOST-predicted epochs they were built from. Nereus fits the
# intermediate astrometric data of both missions directly, so the catalogue
# layer has nothing to show here; in its place every mission's abscissae are
# drawn the way EMPEROR bins Hipparcos's -- per epoch, as a line along the scan
# axis, sitting at the model position plus the O−C.

"""Julian year of an MJD (the inverse of `jyear_to_mjd`)."""
_mjd_to_jyear(mjd::Real) = 2000.0 + (mjd - 51544.5) / 365.25

"""
    _iad_mission_names(iad; names=nothing) -> Vector{String}

A display name per IAD instrument. `IADData` numbers its instruments and does
not name them, so the name is read off the epochs: before J2000 is Hipparcos
(1989.85-1993.21), from mid-2014 on is Gaia, anything else is
`"Instrument m"`. Two instruments of one mission are numbered. `names`
overrides it, one entry per instrument.
"""
function _iad_mission_names(iad; names = nothing)
    n_inst = n_iad_inst(iad)
    if names !== nothing
        length(names) == n_inst || throw(ArgumentError(
            "mission_names has $(length(names)) entries for $n_inst IAD instrument(s)"))
        return String[String(x) for x in names]
    end
    out = String[]
    for m in 1:n_inst
        t_mid = median(iad.t[iad.inst .== m])
        push!(out, t_mid < 51544.5 ? "Hipparcos" :
                   t_mid >= 56800.0 ? "Gaia" : "Instrument $m")
    end
    for nm in unique(out)
        idx = findall(==(nm), out)
        length(idx) > 1 || continue
        for (j, i) in enumerate(idx)
            out[i] = "$nm $j"
        end
    end
    return out
end

"""File-name form of a mission name: `"Gaia 2"` → `"gaia_2"`."""
_mission_slug(name::AbstractString) =
    strip(replace(lowercase(name), r"[^a-z0-9]+" => "_"), '_')

"""A square frame around `(xs, ys)`, padded by `pad` of its side."""
function _square_frame(xs, ys; pad::Real = 0.08)
    xlo, xhi = extrema(xs)
    ylo, yhi = extrema(ys)
    half = (1 + 2pad) * max(xhi - xlo, yhi - ylo, 1e-9) / 2
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    return (cx - half, cx + half, cy - half, cy + half)
end

"""
    _astrometry_model_scene(chains, params, data, planet_idx; ...) -> NamedTuple or nothing

Everything the two epoch-astrometry figures draw, before it is drawn, so that
both of them, and every panel of each, picture one and the same fit.

`nothing` when there is nothing to picture: no IAD, no more transits than the
catalogue marginalisation has parameters, companion `planet_idx` carries no
astrometric orbit, or a trans-dim chain never has it active.

The draw is the maximum-log-posterior one among those in which the companion
exists (`_planet_draw`), the O−C are the likelihood's (`_epoch_astrometry_oc`),
with every other active companion already removed from the data.
"""
function _astrometry_model_scene(chains, params, data, planet_idx::Int;
                                 bf_cutoff::Real = 5.0,
                                 normal_point_gap::Real = 0.01,
                                 psi_tol::Real = 0.02,
                                 n_track::Int = 1200,
                                 mission_names = nothing)
    iad = data.iad
    iad === nothing && return nothing
    n_inst = n_iad_inst(iad)
    n_iad(iad) > Nereus._iad_n_q(n_inst) || return nothing
    _has_astrom_orbit(params, planet_idx) || return nothing
    drawn = _planet_draw(chains, params, planet_idx; bf_cutoff = bf_cutoff)
    drawn === nothing && return nothing
    theta = first(drawn)
    fit = _epoch_astrometry_oc(theta, data, planet_idx)
    fit === nothing && return nothing
    (; orb, M_sec, oc) = fit

    np = _iad_normal_points(iad, oc, normal_point_gap; psi_tol = psi_tol)
    mod_np = [star_reflex_offset(orb, t, M_sec) for t in np.t]
    xm = Float64[p[1] for p in mod_np]
    ym = Float64[p[2] for p in mod_np]
    xn = xm .+ np.e .* np.ux
    yn = ym .+ np.e .* np.uy

    # The orbit, traced uniformly in eccentric anomaly so periastron is on it.
    P_d = _orbit_period_days(theta, planet_idx)
    tp  = PlanetOrbits.periastron(orb)
    e_orb = PlanetOrbits.eccentricity(orb)
    trk = [star_reflex_offset(orb, tp + P_d / 2π * (E - e_orb * sin(E)), M_sec)
           for E in range(0, 2π; length = n_track)]
    x_t = Float64[p[1] for p in trk]
    y_t = Float64[p[2] for p in trk]
    peri = star_reflex_offset(orb, tp, M_sec)

    # Sense of motion: a short step a tenth of a period past periastron.
    p1 = star_reflex_offset(orb, tp + 0.1 * P_d, M_sec)
    p2 = star_reflex_offset(orb, tp + 0.1 * P_d + P_d / 200, M_sec)

    # The zoom is on the mission that resolves the orbit best: the one whose
    # binned abscissae have the smallest errors. One mission needs no zoom.
    med_σ = [median(np.σ[np.inst .== m]) for m in 1:n_inst]
    zoom = n_inst >= 2 ? argmin(med_σ) : 0

    return (; iad, oc, np, xm, ym, xn, yn, x_t, y_t, peri, p1, p2, n_inst, zoom,
              med_σ, tp, P_d, names = _iad_mission_names(iad; names = mission_names),
              cols = cool_pastel(n_inst))
end

"""
Frame of a sky panel showing instruments `insts`: the orbit, the barycentre
and those instruments' binned abscissae -- with their ±1σ lines when `bars`,
which is what a zoom wants and a wide view of mas-level Hipparcos lines around
a sub-mas orbit does not.
"""
function _am_sky_frame(S, insts; bars::Bool)
    xs = vcat(S.x_t, 0.0)
    ys = vcat(S.y_t, 0.0)
    for g in eachindex(S.np.t)
        S.np.inst[g] in insts || continue
        d = bars ? S.np.σ[g] : 0.0
        push!(xs, S.xn[g] - d * abs(S.np.ux[g]), S.xn[g] + d * abs(S.np.ux[g]))
        push!(ys, S.yn[g] - d * abs(S.np.uy[g]), S.yn[g] + d * abs(S.np.uy[g]))
    end
    return _square_frame(xs, ys)
end

"""
    _am_wide_frame(S, orbit_frame) -> (xlo, xhi, ylo, yhi)

Frame of the all-missions sky panel: the orbit, with room around it -- NOT
every abscissa.

Hipparcos abscissae scatter by milliarcseconds about an orbit that is often a
fraction of one. A frame that holds all of them shows the scatter of Hipparcos
and leaves the orbit a speck in the middle. So the frame is a square centred
on the orbit, `orbit_frame` times its extent on a side, and the lines of a
mission that does not resolve the orbit run off the edge.

Two things bound it. It never zooms OUT further than the abscissae need: when
every binned centre already fits in less, that tighter frame is used. And it
always contains the frame of the zoom panel, so the box drawn for the zoom lies
inside it even when the orbit is smaller than the zoom mission's error bars.
"""
function _am_wide_frame(S, orbit_frame::Real)
    full = _am_sky_frame(S, 1:S.n_inst; bars = false)
    f = _orbit_square(S.x_t, S.y_t, orbit_frame)
    f[2] - f[1] >= full[2] - full[1] && return full
    S.zoom == 0 && return f
    z = _am_sky_frame(S, [S.zoom]; bars = true)
    (f[1] <= z[1] && z[2] <= f[2] && f[3] <= z[3] && z[4] <= f[4]) && return f
    return _square_frame((min(f[1], z[1]), max(f[2], z[2])),
                         (min(f[3], z[3]), max(f[4], z[4])); pad = 0.0)
end

# In the all-missions view the lines of a mas-level mission cross everything,
# so they are drawn thinner there than in a single-mission panel.
_am_wide_linewidth(S) = S.zoom == 0 ? 2.5 : 1.5

const _AM_XLABEL = "ΔRA·cos δ (mas)"
const _AM_YLABEL = "Δδ (mas)"

"""A sky-plane axis: equal scales, East to the left."""
function _am_sky_axis(pos; kw...)
    ax = Axis(pos; xlabel = _AM_XLABEL, ylabel = _AM_YLABEL,
              aspect = DataAspect(), kw...)
    _flip_xaxis!(ax)
    return ax
end

_am_legend!(pos, ax; kw...) =
    Legend(pos, ax; framevisible = false, labelsize = 16, orientation = :horizontal,
           tellheight = true, tellwidth = false, kw...)

"""
Draw a sky panel into `ax`: the orbit, and for each instrument in `insts` its
binned abscissae as lines along the scan axis.

An abscissa is one-dimensional: it fixes the photocentre along the scan
direction `u = (sin ψ, cos ψ)` and says nothing across it. So each one is drawn
as the line it is -- centred on the model position plus its O−C along `u`,
`±1σ` long, with a small marker at its centre -- and a dashed connector back to
where the model puts the star at that epoch. Scatter of the lines about the
orbit is information; their agreement with it ACROSS the scan is not.

Back to front: the least precise mission's lines first, the orbit over all of
the lines, the markers, then the arrow over everything. So mas-long Hipparcos
lines bury neither a sub-mas orbit nor Gaia's abscissae nor the sense of motion.

EVERY sky panel of both figures comes through here. That is the point: an
abscissa looks the same wherever it is drawn.
"""
function _am_draw_sky!(ax, S, insts; frame, box = nothing, tag = nothing,
                       linewidth::Real = 2.5)
    order = sort(collect(insts); by = m -> S.med_σ[m], rev = true)
    for m in order
        sel = findall(==(m), S.np.inst)
        isempty(sel) && continue
        con = Point2f[]
        bar = Point2f[]
        for g in sel
            push!(con, Point2f(S.xm[g], S.ym[g]), Point2f(S.xn[g], S.yn[g]))
            dx, dy = S.np.σ[g] * S.np.ux[g], S.np.σ[g] * S.np.uy[g]
            push!(bar, Point2f(S.xn[g] - dx, S.yn[g] - dy),
                       Point2f(S.xn[g] + dx, S.yn[g] + dy))
        end
        linesegments!(ax, con; color = (:gray30, 0.6), linewidth = 0.8,
                      linestyle = :dash)
        linesegments!(ax, bar; color = S.cols[m], linewidth = linewidth)
    end
    lines!(ax, S.x_t, S.y_t; color = NEREUS_COLORS.model, linewidth = 2,
           label = "Best fit")
    for m in order
        sel = findall(==(m), S.np.inst)
        isempty(sel) && continue
        scatter!(ax, S.xn[sel], S.yn[sel]; color = S.cols[m],
                 marker = sky_inst_marker(m), markersize = 9,
                 strokewidth = 0.8, strokecolor = :black,
                 label = "$(S.names[m]) IAD (binned)")
    end
    scatter!(ax, [S.peri[1]], [S.peri[2]]; color = :red, marker = :diamond,
             markersize = 15, strokewidth = 1.2, strokecolor = :black,
             label = "Periastron")
    scatter!(ax, [0.0], [0.0]; color = :black, marker = :cross, markersize = 20,
             strokewidth = 0, label = "Barycentre")
    let side = frame[2] - frame[1],
        extent = max(maximum(S.x_t) - minimum(S.x_t), maximum(S.y_t) - minimum(S.y_t))
        # Last, over the lines and the markers. Not for an orbit too small in
        # this frame to carry one: an arrow sized to the frame would be many
        # times the orbit it describes.
        extent > 0.1 * side && _sky_motion_arrow!(ax, S.p1, S.p2, side)
    end
    if box !== nothing
        lines!(ax, [box[1], box[2], box[2], box[1], box[1]],
               [box[3], box[3], box[4], box[4], box[3]];
               color = (:black, 0.5), linestyle = :dash, linewidth = 1)
    end
    _sky_limits!(ax, frame...)
    tag === nothing || text!(ax, 0.04, 0.97; text = tag, space = :relative,
                             align = (:left, :top), font = :bold, fontsize = 18)
    return ax
end

"""
Draw instrument `m`'s along-scan O−C into `ax`, against epoch in years
(`x = :epoch`) or orbital phase from periastron (`x = :phase`): every abscissa
as a grey dot where binning merged any, the binned ones with their error bars
on top, in the mission's colour and marker. The limits follow the binned
points, so a handful of CCD outliers do not flatten them onto the zero line;
`χ²/N` is over the abscissae, which is what the likelihood sees.
"""
function _am_draw_oc!(ax, S, m::Int; x::Symbol = :epoch, tag = nothing)
    iad = S.iad
    raw = findall(==(m), iad.inst)
    sel = findall(==(m), S.np.inst)
    xof(t) = x === :phase ? mod.((t .- S.tp) ./ S.P_d, 1.0) : _mjd_to_jyear.(t)
    if any(>(1), S.np.size[sel])
        # No error bars on these: 800 CCD-level bars are a grey curtain.
        scatter!(ax, xof(iad.t[raw]), S.oc[raw]; color = (:gray, 0.4),
                 markersize = 4, strokewidth = 0)
    end
    xs = xof(S.np.t[sel])
    errorbars!(ax, xs, S.np.e[sel], S.np.σ[sel]; color = :black,
               linewidth = ERRBAR_LW)
    scatter!(ax, xs, S.np.e[sel]; color = S.cols[m], marker = sky_inst_marker(m),
             markersize = 9, strokewidth = 0.8, strokecolor = :black)
    hlines!(ax, 0; color = NEREUS_COLORS.zero_line, linestyle = :dash,
            linewidth = 1.5)
    half = 1.3 * maximum(abs.(S.np.e[sel]) .+ S.np.σ[sel])
    ylims!(ax, -half, half)
    x === :phase && xlims!(ax, 0, 1)
    χ2 = sum(abs2, S.oc[raw] ./ iad.abscissa_err[raw]) / length(raw)
    text!(ax, 0.98, 0.04; text = @sprintf("χ²/N = %.2f", χ2), space = :relative,
          align = (:right, :bottom), fontsize = 15)
    tag === nothing || text!(ax, 0.02, 0.96; text = tag, space = :relative,
                             align = (:left, :top), font = :bold, fontsize = 18)
    return ax
end

"""
    plot_astrometry_model(chains, params, data; planet_idx=1, output=nothing,
                          fmt=:png, save_pdf=false, figsize=nothing,
                          panels=true, orbit_frame=1.5, bf_cutoff=5.0,
                          normal_point_gap=0.01, psi_tol=0.02, n_track=1200,
                          mission_names=nothing)

The astrometric orbit of companion `planet_idx` against the epoch astrometry
of every mission in `data.iad` -- astroEMPEROR's `astrometry_model` figure,
with the intermediate astrometric data in place of its Gaia DR2/DR3 catalogue
positions and GOST model.

Panels:
  - **(a)** the reflex orbit of the star about the barycentre (`+`), with every
    mission's abscissae as BINNED LINES along their scan axes (see below), the
    periastron, an arrow for the sense of motion, and a dashed box marking the
    frame of (b). It is framed on the ORBIT: a square `orbit_frame` orbit
    extents on a side, never wider than the abscissae need. Lines of a mission
    that does not resolve the orbit (Hipparcos, usually) run off the edge;
  - **(b)** the same, zoomed on the mission that resolves the orbit best (the
    smallest binned errors; Gaia, when it is there) and showing only that
    mission. Not drawn when there is only one mission;
  - one panel per mission: its along-scan O−C against epoch, the individual
    abscissae in grey and the binned ones on top, with `χ²/N` over the
    abscissae.

An abscissa is one-dimensional. It is drawn as a line along its scan direction
`u = (sin ψ, cos ψ)`, `±1σ` long, centred at the model position plus its O−C
along `u`, and joined to that model position by a dashed connector. Abscissae
are binned first (`_iad_normal_points`): consecutive transits of one mission no
more than `normal_point_gap` days apart whose scan angles agree within
`psi_tol` radians. That is one line per Gaia field-of-view transit (its 8-9 CCD
abscissae share one ψ) and one per Hipparcos satellite orbit (whose records
differ by a few 1e-3 rad).

The orbit is the maximum-log-posterior draw among those in which the companion
exists, and the O−C are the likelihood's own with every other active companion
already subtracted. [`plot_epoch_astrometry_orbit`](@ref) draws the same scene
the same way, with the O−C folded in orbital phase instead. When the fit is a
non-detection the orbit is far smaller than the abscissa errors, and the lines
of (a) all pass near the barycentre: that is the data saying so, not a drawing
fault.

Missions are named from their epochs (`_iad_mission_names`); `mission_names`
overrides that, one name per IAD instrument. `figsize = nothing` picks a size
for the layout: wide with a zoom column, tall with one mission.

Saves `models/astrometry_model_K<k>.<fmt>` and, with `panels = true`, each panel
as its own figure beside it: `..._sky`, `..._sky_<mission>` (the zoom) and
`..._oc_<mission>`. Returns the combined `Figure`; an empty one, with nothing
saved, when there is nothing to picture (`_astrometry_model_scene`).
"""
function plot_astrometry_model(chains, params, data;
                               planet_idx::Int = 1,
                               output::Union{Nothing, String} = nothing,
                               fmt::Symbol = :png,
                               save_pdf::Bool = false,
                               figsize = nothing,
                               panels::Bool = true,
                               orbit_frame::Real = 1.5,
                               bf_cutoff::Real = 5.0,
                               normal_point_gap::Real = 0.01,
                               psi_tol::Real = 0.02,
                               n_track::Int = 1200,
                               mission_names = nothing)
    _check_orbit_frame(orbit_frame)
    S = _astrometry_model_scene(chains, params, data, planet_idx;
                                bf_cutoff, normal_point_gap, psi_tol, n_track,
                                mission_names)
    S === nothing && return Figure()
    insts = collect(1:S.n_inst)
    frame_all  = _am_wide_frame(S, orbit_frame)
    frame_zoom = S.zoom == 0 ? nothing : _am_sky_frame(S, [S.zoom]; bars = true)
    base = output === nothing ? nothing :
           joinpath(output, "models", "astrometry_model_K$(planet_idx)")
    output === nothing || mkpath(joinpath(output, "models"))
    save_fig(fig, suffix) = base === nothing ? nothing :
        _save_plot("$(base)$(suffix).$fmt", fig; save_pdf = save_pdf, px_per_unit = 3)
    oc_axis(pos; kw...) = Axis(pos; xlabel = "Epoch (yr)", ylabel = "O−C (mas)", kw...)
    wide_lw = _am_wide_linewidth(S)

    with_theme(nereus_theme()) do
        # ---- each panel on its own -------------------------------------
        if panels && base !== nothing
            let fig = Figure(; size = (1000, 1100))
                ax = _am_sky_axis(fig[1, 1])
                _am_draw_sky!(ax, S, insts; frame = frame_all, box = frame_zoom,
                              linewidth = wide_lw)
                _am_legend!(fig[2, 1], ax; nbanks = 2)
                save_fig(fig, "_sky")
            end
            if S.zoom != 0
                fig = Figure(; size = (1000, 1100))
                ax = _am_sky_axis(fig[1, 1])
                _am_draw_sky!(ax, S, [S.zoom]; frame = frame_zoom)
                _am_legend!(fig[2, 1], ax; nbanks = 2)
                save_fig(fig, "_sky_" * _mission_slug(S.names[S.zoom]))
            end
            for m in insts
                fig = Figure(; size = (1100, 520))
                _am_draw_oc!(oc_axis(fig[1, 1]), S, m; tag = S.names[m])
                save_fig(fig, "_oc_" * _mission_slug(S.names[m]))
            end
        end

        # ---- the combined figure ---------------------------------------
        letters = Iterators.Stateful('a':'z')
        tag(extra = "") = "(" * string(popfirst!(letters)) * ")" *
                          (isempty(extra) ? "" : " " * extra)
        fig = Figure(; size = something(figsize, S.zoom == 0 ? (1000, 1350) : (1650, 1080)))
        # Row 1 is the legend, row 2 the panels.
        ax_a = _am_sky_axis(fig[2, 1])
        _am_draw_sky!(ax_a, S, insts; frame = frame_all, box = frame_zoom,
                      tag = tag(), linewidth = wide_lw)
        if S.zoom == 0
            # One mission: its O−C under the orbit.
            _am_draw_oc!(oc_axis(fig[3, 1]), S, 1; tag = tag(S.names[1]))
            rowsize!(fig.layout, 3, Makie.Relative(0.24))
            _am_legend!(fig[1, 1], ax_a)
        else
            # Left: the wide view, square. Right: the zoom over the O−C panels.
            gr = fig[2, 2] = GridLayout()
            ax_b = _am_sky_axis(gr[1, 1]; yaxisposition = :right)
            _am_draw_sky!(ax_b, S, [S.zoom]; frame = frame_zoom,
                          tag = tag(S.names[S.zoom]))
            rowsize!(gr, 1, Makie.Relative(0.46))
            for (i, m) in enumerate(insts)
                _am_draw_oc!(oc_axis(gr[1 + i, 1]; yaxisposition = :right), S, m;
                             tag = tag(S.names[m]))
            end
            colsize!(fig.layout, 1, Makie.Aspect(2, 1.0))
            _am_legend!(fig[1, 1:2], ax_a)
        end
        save_fig(fig, "")
        fig
    end
end

"""
    plot_epoch_astrometry_orbit(chains, params, data; planet_idx=1,
                                output=nothing, fmt=:png, save_pdf=false,
                                figsize=nothing, orbit_frame=1.5, bf_cutoff=5.0,
                                normal_point_gap=0.01, psi_tol=0.02,
                                n_track=1200, mission_names=nothing)

The astrometric orbit of an epoch-astrometry target (Hipparcos IAD, Gaia DR4
along-scan) on the sky with every mission's abscissae on it, and below it the
along-scan O−C folded in ORBITAL PHASE (0 = periastron), one strip per mission.

Epoch astrometry is ONE-DIMENSIONAL: a transit measures the abscissa `w` along
the scan direction `u = (sin ψ, cos ψ)` and nothing across it, so it pins the
photocentre to a line, not a point. The standard picture (Sahlmann et al.
2011, A&A 525, A95, Fig. 20; Holl et al. 2023, A&A 674, A10, Figs. 12-16, the
Gaia DR3 NSS convention) puts each measurement at the model position plus its
O−C along `u`:

    P = M(t) + (O − C)·u

and draws it as a line along `u`, `±1σ` long. Across the scan it sits exactly
where the model puts it, so scatter about the ellipse is information and
agreement across the scan is not evidence. This is a picture of the fit, not
2-D astrometry.

This is the same scene [`plot_astrometry_model`](@ref) draws, drawn by the
same routines -- its sky panel IS that figure's panel (a): the same maximum-
log-posterior draw, the same binning (`normal_point_gap`, `psi_tol`), the same
line and marker for an abscissa, the same mission colours, the same frame
(`orbit_frame` orbit extents; a mission that does not resolve the orbit runs
off the edge). The two differ only in what the O−C is plotted against: epoch
there, which separates the missions; orbital phase here, which shows whether
the residuals follow the orbit.

Each strip has the mission's individual abscissae in grey, the binned ones with
their error bars on top, and `χ²/N` over the abscissae -- what the likelihood
sees. A strip per mission, because Hipparcos O−C of several mas on one axis
flatten Gaia's onto the zero line.

The O−C are the likelihood's own (`_epoch_astrometry_oc`, the path
`plot_iad_residuals` also takes), so with several companions this shows planet
`planet_idx`'s orbit with the others already removed from the data. On a
trans-dim chain the draw is taken among those where the planet is active, and
that draw's active set decides which other companions are subtracted. When a
Gaia DR3 five-parameter solution is also supplied (`data.gaia_dr3` with
`data.gost`), the fit marginalises the catalogue solution against it as well;
this figure, like `plot_iad_residuals`, uses the IAD-only marginalisation.

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
                                     figsize = nothing,
                                     orbit_frame::Real = 1.5,
                                     bf_cutoff::Real = 5.0,
                                     normal_point_gap::Real = 0.01,
                                     psi_tol::Real = 0.02,
                                     n_track::Int = 1200,
                                     mission_names = nothing)
    _check_orbit_frame(orbit_frame)
    S = _astrometry_model_scene(chains, params, data, planet_idx;
                                bf_cutoff, normal_point_gap, psi_tol, n_track,
                                mission_names)
    S === nothing && return Figure()
    insts = collect(1:S.n_inst)
    strip_h = 230                       # px per O−C strip

    with_theme(nereus_theme()) do
        # Room on the right for the last phase tick label, which sits on the edge.
        fig = Figure(; size = something(figsize, (1000, 1130 + strip_h * S.n_inst)),
                     figure_padding = (10, 30, 10, 10))
        # Row 1 is the legend, row 2 the sky panel, then one strip per mission.
        ax = _am_sky_axis(fig[2, 1])
        _am_draw_sky!(ax, S, insts; frame = _am_wide_frame(S, orbit_frame),
                      linewidth = _am_wide_linewidth(S))
        for (i, m) in enumerate(insts)
            last = i == length(insts)
            ax_r = Axis(fig[2 + i, 1]; xlabel = "Orbital phase", ylabel = "O−C (mas)",
                        xlabelvisible = last, xticklabelsvisible = last)
            _am_draw_oc!(ax_r, S, m; x = :phase, tag = S.names[m])
            rowsize!(fig.layout, 2 + i, strip_h - (last ? 0 : 50))
        end
        _am_legend!(fig[1, 1], ax)
        # The sky frame is square: give it a square cell, and let the figure
        # take the height that leaves, rather than float the panel in slack.
        rowsize!(fig.layout, 2, Makie.Aspect(1, 1.0))
        figsize === nothing && resize_to_layout!(fig)

        if output !== nothing
            mkpath(joinpath(output, "models"))
            _save_plot(joinpath(output, "models",
                                "epoch_astrometry_orbit_K$(planet_idx).$fmt"), fig;
                       save_pdf = save_pdf, px_per_unit = 3)
        end
        fig
    end
end
