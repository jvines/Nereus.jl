# Nereus plotting theme — matches astroEMPEROR publication style.
#
# EMPEROR style: serif 22pt, orchid CI bands, cool colormap for time,
# 3:1 data/residual panels, per-instrument markers/colors.

using CairoMakie
using Colors: colorant

# =====================================================================
# Style constants (ported from emperors_canvas.py __config())
# =====================================================================

const NEREUS_COLORS = (
    ci        = colorant"orchid",          # credibility interval fill
    rv_bin    = colorant"lime",            # binned RV points
    pm_marker = colorant"#17becf",         # unbinned photometry (Tableau tab:cyan)
    pm_model  = colorant"#3dffd5",         # transit model line
    pm_bin    = colorant"#e377c2",         # binned photometry (tab:pink)
    post      = colorant"#1f77b4",         # posterior scatter (tab:blue)
    hist_face = colorant"#1f77b4",         # histogram fill
    gp        = colorant"rosybrown",       # GP component
    activity  = colorant"#ff7f0e",         # activity-decorrelation (AD) (tab:orange)
    model     = colorant"black",           # combined model
    kep       = colorant"#7f7f7f",         # Keplerian component (tab:gray)
    zero_line = colorant"black",           # residual zero line
)

const NEREUS_CMAP = :cool    # time coloring for RV phase-fold

# Per-instrument markers (matching EMPEROR's 12 marker cycle)
const INST_MARKERS = [
    :circle, :utriangle, :rect, :diamond, :star5, :pentagon,
    :hexagon, :cross, :dtriangle, :rtriangle, :ltriangle, :octagon,
]

# Per-instrument colors (Tableau 10 cycle)
const INST_COLORS = [
    colorant"#1f77b4", colorant"#ff7f0e", colorant"#2ca02c",
    colorant"#d62728", colorant"#9467bd", colorant"#8c564b",
    colorant"#e377c2", colorant"#7f7f7f", colorant"#bcbd22",
    colorant"#17becf",
]

const CI_ALPHA = 0.25

# Nested 1σ/2σ/3σ credible-band opacities, darkest→lightest. These are FINAL
# rendered values: there is deliberately no theme-level Band alpha (Makie would
# multiply the two together — see nereus_theme below). Tuned so a wide band
# reads as a translucent credible region rather than a solid slab, and a narrow
# one is still visible against white.
const CI_ALPHA_1SIGMA = 0.30
const CI_ALPHA_2SIGMA = 0.18
const CI_ALPHA_3SIGMA = 0.10
const HIST_ALPHA = 0.5
const MODEL_LW = 2
const ERRBAR_LW = 1.75

# Figure sizes in pixels (100 dpi equivalent of EMPEROR's inch sizes)
const FIG_RV   = (1000, 1000)   # 10×10 inches
# RM anomaly is a two-panel stacked figure sharing one x-axis, so a 10x10 square
# wastes space and reads oversized. Wider than tall, and smaller overall.
const FIG_RM   = (820, 620)     # 8.2×6.2
const FIG_PM   = (1000, 1000)   # 10×10
const FIG_TRACE = (1200, 700)   # 12×7
const FIG_POST  = (1000, 800)   # 10×8
const FIG_HIST  = (1200, 800)   # 12×8

# =====================================================================
# Theme
# =====================================================================

"""
    _serif_family() -> NamedTuple

Pick the first serif family actually present on this machine.

Times New Roman is a Microsoft font and is NOT installed on most Linux systems.
When CairoMakie cannot resolve the requested family it silently drops the text,
so every axis label and tick label vanishes — which is what happened to the
transit plots generated on the compute node. Liberation Serif is
metrically compatible with Times New Roman and ships with essentially every
Linux distribution, so it is the first fallback.
"""
function _font_available(name::AbstractString)
    # Makie.to_font does NOT throw for a missing family -- it silently substitutes
    # whatever FreeType hands back. So asking "did it succeed?" always says yes,
    # and the bug survives. The only reliable test is to resolve the font and check
    # that the family we got back is the family we asked for.
    try
        f = Makie.to_font(String(name))
        got = lowercase(String(Makie.FreeTypeAbstraction.family_name(f)))
        want = lowercase(String(name))
        return got == want || startswith(got, want) || startswith(want, got)
    catch
        return false
    end
end

function _serif_family()
    candidates = (("Times New Roman", "Times New Roman Bold", "Times New Roman Italic"),
                  ("Liberation Serif", "Liberation Serif Bold", "Liberation Serif Italic"),
                  ("DejaVu Serif", "DejaVu Serif Bold", "DejaVu Serif Italic"),
                  ("TeX Gyre Termes", "TeX Gyre Termes Bold", "TeX Gyre Termes Italic"))
    for (r, b, i) in candidates
        _font_available(r) || continue
        # bold/italic faces are often packaged separately; fall back per face
        return (; regular = r,
                  bold   = _font_available(b) ? b : r,
                  italic = _font_available(i) ? i : r)
    end
    @warn "nereus_theme: no serif family found (tried Times New Roman, Liberation \
           Serif, DejaVu Serif, TeX Gyre Termes). Falling back to the Makie default; \
           axis labels will still render but the style will differ."
    return (; regular = "TeX Gyre Heros Makie",
              bold = "TeX Gyre Heros Bold Makie",
              italic = "TeX Gyre Heros Italic Makie")
end

"""
    nereus_theme() -> Makie.Theme

Publication-quality theme matching astroEMPEROR's visual style.
Apply with `set_theme!(nereus_theme())` or `with_theme(nereus_theme()) do ... end`.
"""
function nereus_theme()
    Theme(
        fontsize = 22,
        fonts = _serif_family(),
        figure_padding = 10,
        Axis = (;
            xlabelsize = 22,
            ylabelsize = 22,
            xticklabelsize = 20,
            yticklabelsize = 20,
            xminorticksvisible = true,
            yminorticksvisible = true,
            xgridvisible = false,
            ygridvisible = false,
            topspinevisible = true,
            rightspinevisible = true,
        ),
        Lines = (; linewidth = 2),
        Scatter = (; markersize = 10, strokewidth = 0.5, strokecolor = :black),
        # NO theme-level Band alpha. Makie MULTIPLIES a theme alpha into the
        # colour's own alpha, and every band! call site in this package already
        # passes an explicit alpha (0.22 / 0.45 / 0.72 for the 3/2/1σ stack).
        # With `Band = (; alpha = 0.25)` here those became 0.055 / 0.11 / 0.18 —
        # i.e. the CI bands were being drawn and were effectively invisible.
        Errorbars = (; linewidth = ERRBAR_LW, whiskerwidth = 0),
    )
end

"""Marker for instrument index (cycles through INST_MARKERS)."""
inst_marker(i::Int) = INST_MARKERS[mod1(i, length(INST_MARKERS))]

"""Color for instrument index (cycles through INST_COLORS)."""
inst_color(i::Int) = INST_COLORS[mod1(i, length(INST_COLORS))]
