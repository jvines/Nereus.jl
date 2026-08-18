#!/usr/bin/env julia
# Render the Gaia DR4 epoch-astrometry reader validation as figures.
#
# Companion to validate_gaia_dr4_epoch.jl — same fit_5p, same oracle, but the
# two numeric gates come out as panels instead of PASS/FAIL lines:
#
#   (a) gate A — |Nereus − ESA gaiasupdate| per source over (ϖ, μα*, μδ).
#       Every point must sit below the 1e-3 mas line, i.e. the BINARY2 parser
#       + AL design matrix reproduce ESA's reference solution.
#   (b) gate C — post-fit along-scan residual RMS of a SINGLE-STAR (5-param)
#       model. The nine non-orbital sources pile up on a ~1 mas floor; the
#       three published orbital systems sit above it. That excess IS the
#       orbit signal the fits in test/astrometry/ then model.
#
# Usage:  julia --project=. test/validation/plot_gaia_dr4_epoch_validation.jl
#   env NEREUS_GAIA_DR4_XML=<path>  reuse a local VOTable (else it fetches)
# Output: test/validation/plots_gaia_dr4/reader_validation.png

using Nereus, LinearAlgebra, Printf, Statistics, JSON3
using CairoMakie
using Nereus: nereus_theme, NEREUS_CMAP

OUTDIR = joinpath(@__DIR__, "plots_gaia_dr4")
mkpath(OUTDIR)

const ORBITAL = Dict(
    3937211745905473024 => "HD 114762",
    1457486023639239296 => "Gaia-4",
    4318465066420528000 => "Gaia BH3",
)

# n colors sampled evenly from :cool, blended toward white (house style).
function cool_pastel(n; strength = 0.45)
    cs = cgrad(NEREUS_CMAP)
    [RGBf((1 - strength) * c.r + strength,
          (1 - strength) * c.g + strength,
          (1 - strength) * c.b + strength)
     for c in (cs[x] for x in range(0, 1, length = n))]
end

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = Nereus.fetch_gaia_dr4_prerelease())
sources = read_gaia_epoch_votable(xml)
oracle  = JSON3.read(read(joinpath(@__DIR__, "gaia_dr4_oracle.json"), String))
@printf("parsed %d sources\n", length(sources))

# 5-parameter along-scan LSQ — identical to validate_gaia_dr4_epoch.jl
function fit_5p(iad)
    st = sin.(iad.psi); ct = cos.(iad.psi)
    D  = hcat(st, ct, iad.parallax_factor, st .* iad.pm_factor, ct .* iad.pm_factor)
    y  = iad.abscissa; σ = iad.abscissa_err; W = 1.0 ./ σ
    p  = (D .* W) \ (y .* W)
    post = sqrt(mean(((y .- D * p) ./ σ) .^ 2))
    return (; plx = p[3], pmra = p[4], pmdec = p[5], post)
end

# --- assemble the per-source table, sorted by post-fit RMS -------------------
rows = map(collect(keys(sources))) do sid
    f = fit_5p(sources[sid].iad)
    o = oracle[Symbol(string(sid))]
    (; sid,
       name    = get(ORBITAL, sid, ""),
       n       = length(sources[sid].iad.t),
       post    = f.post,
       d_plx   = abs(f.plx   - o.plx),
       d_pmra  = abs(f.pmra  - o.pmra),
       d_pmdec = abs(f.pmdec - o.pmdec))
end
sort!(rows, by = r -> r.post)

# axis tick labels: orbital systems by name, the rest by trailing source_id
ticklab = [isempty(r.name) ? "…" * string(r.sid)[end-4:end] : r.name for r in rows]
isorb   = [!isempty(r.name) for r in rows]
xs      = 1:length(rows)

pal   = cool_pastel(3)
c_orb = RGBf(0.85, 0.30, 0.55)      # orbital sources — off-palette so they read as flagged

with_theme(nereus_theme()) do
    fig = Figure(; size = (1150, 950))

    # (a) parser vs ESA reference ------------------------------------------------
    ax1 = Axis(fig[1, 1];
               ylabel = "|Nereus − ESA|  (mas, mas/yr)",
               yscale = log10,
               yticks = ([1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                         ["10⁻⁷", "10⁻⁶", "10⁻⁵", "10⁻⁴", "10⁻³"]),
               xticks = (xs, ticklab),
               xticklabelsvisible = false,
               xticksvisible = false)
    for (j, (key, lab, mk)) in enumerate([(:d_plx,   "ϖ",   :circle),
                                          (:d_pmra,  "μα*", :utriangle),
                                          (:d_pmdec, "μδ",  :diamond)])
        scatter!(ax1, xs, [max(getfield(r, key), 1e-16) for r in rows];
                 color = pal[j], marker = mk, markersize = 13,
                 strokewidth = 0.6, strokecolor = :black, label = lab)
    end
    hlines!(ax1, [1e-3]; color = :black, linestyle = :dash, linewidth = 1.5)
    text!(ax1, length(rows) + 0.35, 1.15e-3; text = "validation gate  10⁻³",
          align = (:right, :bottom), fontsize = 15, color = :black)
    ylims!(ax1, 3e-8, 4e-3)
    axislegend(ax1; position = :rb, framevisible = true, patchsize = (12, 12))

    # (b) single-star post-fit residual RMS --------------------------------------
    ax2 = Axis(fig[2, 1];
               xlabel = "DR4 pre-release source  (ordered by post-fit residual RMS)",
               ylabel = "single-star post-fit AL residual RMS  (mas)",
               yscale = log10,
               yticks = ([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"]),
               xticks = (xs, ticklab),
               xticklabelrotation = π / 4)
    # the ~1 mas floor set by the nine non-orbital sources
    floorhi = maximum(r.post for r in rows if isempty(r.name))
    hspan!(ax2, minimum(r.post for r in rows), floorhi;
           color = (pal[1], 0.35))
    text!(ax2, length(rows) + 0.35, floorhi;
          text = @sprintf("non-orbital floor ≤ %.2f mas", floorhi),
          align = (:right, :bottom), fontsize = 15, color = :black)
    scatter!(ax2, xs[.!isorb], [r.post for r in rows[.!isorb]];
             color = pal[1], marker = :circle, markersize = 14,
             strokewidth = 0.6, strokecolor = :black, label = "single star / QSO")
    scatter!(ax2, xs[isorb], [r.post for r in rows[isorb]];
             color = c_orb, marker = :star5, markersize = 20,
             strokewidth = 0.6, strokecolor = :black, label = "published orbital system")
    for (i, r) in enumerate(rows)
        isempty(r.name) && continue
        text!(ax2, i, r.post * 1.35; text = @sprintf("%.0f× floor", r.post / floorhi),
              align = (:center, :bottom), fontsize = 15, color = c_orb)
    end
    ylims!(ax2, 0.75, 260)
    axislegend(ax2; position = :ct, framevisible = true, patchsize = (12, 12))

    linkxaxes!(ax1, ax2)
    rowgap!(fig.layout, 8)

    for (lbl, ax) in [("(a)", ax1), ("(b)", ax2)]
        text!(ax, 0.012, 0.955; text = lbl, space = :relative,
              align = (:left, :top), fontsize = 19, font = :bold)
    end

    out = joinpath(OUTDIR, "reader_validation.png")
    save(out, fig)
    println("saved ", out)
end

# --- console summary matching the figure -------------------------------------
@printf("\n%-12s %6s %10s %12s\n", "source", "N", "post_rms", "max|Δ| vs ESA")
for r in rows
    @printf("%-12s %6d %10.2f %12.2e  %s\n",
            isempty(r.name) ? string(r.sid)[end-4:end] : r.name,
            r.n, r.post, max(r.d_plx, r.d_pmra, r.d_pmdec), r.name)
end
