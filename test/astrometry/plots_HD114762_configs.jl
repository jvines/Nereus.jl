#!/usr/bin/env julia
# How much of HD 114762 b's astrometric mass is the DATA, and how much is the
# nuisance model? Five joint RV+DR4 fits that differ only in how the outer
# companion and the RV jitter are handled, on identical data.
#
#   trend1        linear trend  + LogUniform(0.5,50) jitter   (committed default)
#   trend2        quadratic     + LogUniform(0.5,50) jitter
#   defaultjitter linear trend  + Nereus default jitter prior
#   fixedjitter   linear trend  + jitter FIXED at 3.0 m/s
#   2kep          B as a 2nd Keplerian, no trend, default jitter
#
# All five pass the script's sin(i)-break gates. They disagree on the true mass
# by 3.5×, and every one of them puts i below Kiefer+ 2019's ~7°.
#
# Usage:  julia --project=. test/astrometry/plots_HD114762_configs.jl
# Output: test/astrometry/plots_HD114762_configs/mass_vs_config.png

using Nereus
using MCMCChains
using CairoMakie
using Colors: @colorant_str
using Statistics: quantile, median
using Printf
using Nereus: nereus_theme, NEREUS_CMAP

const MJUP_PER_MSUN = 1047.57
const KIEFER_M, KIEFER_I = 0.10, 7.0
const DEUTERIUM_LIMIT, HYDROGEN_LIMIT = 13.0, 80.0

DIR = joinpath(@__DIR__, "plots_HD114762_configs")

# order = the narrative order, not alphabetical
CONFIGS = [
    ("trend1.nc",        "linear trend\nLogU jitter\n(committed default)"),
    ("trend2.nc",        "quadratic trend\nLogU jitter"),
    ("defaultjitter.nc", "linear trend\ndefault jitter prior"),
    ("fixedjitter.nc",   "linear trend\njitter fixed 3 m/s"),
    ("2kep.nc",          "B as 2nd Keplerian\nno trend"),
]

function cool_pastel(n; strength = 0.42)
    cs = cgrad(NEREUS_CMAP)
    [RGBf((1 - strength) * c.r + strength,
          (1 - strength) * c.g + strength,
          (1 - strength) * c.b + strength)
     for c in (cs[x] for x in range(0, 1, length = n))]
end

rows = map(CONFIGS) do (fn, lab)
    ch, _ = load_chains(joinpath(DIR, fn))
    M = vec(Array(ch[:, :M_sec_k1, :]))                      # M_sun (RVAS ⇒ true mass)
    i = [x > 90 ? 180 - x : x for x in rad2deg.(vec(Array(ch[:, :inc_k1, :])))]
    jit = Float64[]
    for s in (:sigma_HIRES, :sigma_Lick)
        s in names(ch, :parameters) && append!(jit, vec(Array(ch[:, s, :])))
    end
    (; lab, M, i, jit)
end

for r in rows
    @printf("%-28s M = %.3f [%.3f, %.3f]   i = %.2f°   jitter med %s\n",
            replace(r.lab, "\n" => " / "),
            quantile(r.M, 0.5), quantile(r.M, 0.16), quantile(r.M, 0.84),
            quantile(r.i, 0.5),
            isempty(r.jit) ? "fixed" : @sprintf("%.1f m/s", median(r.jit)))
end

pal = cool_pastel(length(rows))
ys  = 1:length(rows)

with_theme(nereus_theme()) do
    fig = Figure(; size = (1250, 850))

    # (a) true mass — the quantity the sin(i) break is supposed to deliver
    ax1 = Axis(fig[1, 1];
               xlabel = "true companion mass M  (M_Jup)",
               xscale = log10,
               xticks = ([30, 50, 100, 200, 300, 500],
                         ["30", "50", "100", "200", "300", "500"]),
               yticks = (ys, [r.lab for r in rows]),
               ygridvisible = false)
    # No regime bands here: every configuration lands above the 80 M_Jup
    # hydrogen-burning limit, so a shaded "star" band would cover the whole
    # panel and carry no information. The disagreement is WITHIN the stellar
    # regime, which is the point.
    vlines!(ax1, [KIEFER_M * MJUP_PER_MSUN]; color = :black, linewidth = 2.5,
            linestyle = :dash)
    text!(ax1, KIEFER_M * MJUP_PER_MSUN, length(rows) + 0.46;
          text = " Kiefer+ 2019  0.10 M⊙", align = (:left, :top), fontsize = 16)
    for (k, r) in enumerate(rows)
        v = r.M .* MJUP_PER_MSUN
        lines!(ax1, [quantile(v, 0.0015), quantile(v, 0.9985)], [k, k];
               color = (pal[k], 0.45), linewidth = 5)
        lines!(ax1, [quantile(v, 0.16), quantile(v, 0.84)], [k, k];
               color = pal[k], linewidth = 14)
        scatter!(ax1, [quantile(v, 0.5)], [k]; color = :white, strokecolor = :black,
                 strokewidth = 1.5, markersize = 15)
        text!(ax1, quantile(v, 0.5), k + 0.20;
              text = @sprintf("%.3f M⊙", quantile(r.M, 0.5)),
              align = (:center, :bottom), fontsize = 16)
    end
    xlims!(ax1, 100, 900)
    ylims!(ax1, 0.4, length(rows) + 0.85)

    # (b) the inclination that sets it — every config sits below Kiefer
    ax2 = Axis(fig[1, 2];
               xlabel = "inclination i  (deg)",
               yticks = (ys, ["" for _ in rows]),
               ygridvisible = false)
    vlines!(ax2, [KIEFER_I]; color = :black, linewidth = 2.5, linestyle = :dash)
    text!(ax2, KIEFER_I, length(rows) + 0.46; text = " ≈7°",
          align = (:left, :top), fontsize = 16)
    for (k, r) in enumerate(rows)
        lines!(ax2, [quantile(r.i, 0.0015), quantile(r.i, 0.9985)], [k, k];
               color = (pal[k], 0.45), linewidth = 5)
        lines!(ax2, [quantile(r.i, 0.16), quantile(r.i, 0.84)], [k, k];
               color = pal[k], linewidth = 14)
        scatter!(ax2, [quantile(r.i, 0.5)], [k]; color = :white, strokecolor = :black,
                 strokewidth = 1.5, markersize = 15)
        text!(ax2, quantile(r.i, 0.5), k + 0.20;
              text = @sprintf("%.2f°", quantile(r.i, 0.5)),
              align = (:center, :bottom), fontsize = 16)
    end
    xlims!(ax2, 0, 9)
    ylims!(ax2, 0.4, length(rows) + 0.85)

    colsize!(fig.layout, 1, Relative(0.60))
    Label(fig[2, 1:2],
          "identical data — only the outer-companion and jitter treatment differ;\n" *
          "all five pass the script's sin(i)-break gates",
          fontsize = 18, padding = (0, 0, 4, 10))

    for (lbl, ax) in [("(a)", ax1), ("(b)", ax2)]
        text!(ax, 0.02, 0.985; text = lbl, space = :relative,
              align = (:left, :top), fontsize = 19, font = :bold)
    end

    out = joinpath(DIR, "mass_vs_config.png")
    save(out, fig)
    println("\nsaved ", out)
end
