#!/usr/bin/env julia
# Render the HD 114762 joint RV + Gaia DR4 epoch-astrometry fit
# (fit_HD114762_rv_dr4_epoch.jl) — the sin(i) break.
#
# Reads the posterior the fit saved, so plot style can be iterated without
# re-running Pigeons. NEREUS_OUTDIR selects which trend_order variant to
# render (the fit writes chains.nc there).
#
# Figures:
#   sini_break.png     — M sin i vs true mass across the planet/BD/star regimes
#   iad_residuals.png  — DR4 along-scan residuals vs ψ and vs MJD
#   rv_timeseries.png / rv_phasefold.png
#   orbit_skyplane.png, corner.png
#
# Usage:  julia --project=. test/astrometry/plots_HD114762_dr4.jl
#   env HD114762_RV=<path>           4-col RV file (BJD RV eRV inst)
#   env HD114762_TREND=<n>           must match the fit that wrote chains.nc
#   env NEREUS_OUTDIR=<dir>         defaults to plots_HD114762_dr4
#   env NEREUS_GAIA_DR4_XML=<path>  reuse a local VOTable

using Nereus
using MCMCChains
using CairoMakie
using Colors: @colorant_str
using Statistics: quantile
using Printf
using Nereus: nereus_theme, NEREUS_COLORS

const HD114762_SID = 3937211745905473024
const M_PRI   = 0.83
const PLX     = 25.36
const PLX_ERR = 0.30

# Mass-regime boundaries, in M_Jup
const DEUTERIUM_LIMIT = 13.0     # planet | brown dwarf
const HYDROGEN_LIMIT  = 80.0     # brown dwarf | star
const KIEFER_MSUN     = 0.10     # Kiefer+ 2019 true mass
const MJUP_PER_MSUN   = 1047.57

OUTDIR = get(ENV, "NEREUS_OUTDIR",
             joinpath(@__DIR__, "plots_HD114762_dr4"))
CHAINS = joinpath(OUTDIR, "chains.nc")
isfile(CHAINS) ||
    error("no chains at $CHAINS — run fit_HD114762_rv_dr4_epoch.jl first")

# --- rebuild the target exactly as the fit did -------------------------------
rvfile = get(ENV, "HD114762_RV", "")
isfile(rvfile) || error("set HD114762_RV to the 4-col RV file (BJD RV eRV inst)")
tb = Dict{String, Vector{Float64}}()
for line in eachline(rvfile)
    s = strip(line); (isempty(s) || startswith(s, "#")) && continue
    t, rv, er, ins = split(s)
    d = get!(() -> Float64[], tb, ins)
    append!(d, (parse(Float64, t) - 2_400_000.5, parse(Float64, rv), parse(Float64, er)))
end
mkrv(ins) = let v = reshape(tb[ins], 3, :)
    (t = v[1, :], rv = v[2, :], rv_err = v[3, :])
end
hires = mkrv("j"); lick = mkrv("lick")

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, HD114762_SID)

target = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a      = LogUniformPrior(0.30, 0.45),
        M_sec  = LogUniformPrior(0.003, 0.5),
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2π),
    ),),
    rv = (
        HIRES = (data = hires, sigma = LogUniformPrior(0.5, 50.0)),
        Lick  = (data = lick,  sigma = LogUniformPrior(0.5, 50.0)),
    ),
    iad = src.iad,
    plx = NormalPrior(PLX, PLX_ERR),
    M_s = M_PRI,
    trend_order = parse(Int, get(ENV, "HD114762_TREND", "1")),
)
params, data = target.params, target.data

chains, meta = load_chains(CHAINS)
@printf("%d RV + %d abscissae; posterior %d iter × %d chains\n",
        length(hires.t) + length(lick.t), length(src.iad.t),
        size(chains, 1), size(chains, 3))

# --- the standard suite -------------------------------------------------------
for (fn, name) in [(plot_iad_residuals,      "iad_residuals"),
                   (plot_rv_timeseries,      "rv_timeseries"),
                   (plot_rv_phasefold,       "rv_phasefold"),
                   (plot_rv_astrom_phasefold, "rv_astrom_phasefold"),
                   (plot_orbit_skyplane,     "orbit_skyplane")]
    try
        fn(chains, params, data; output = OUTDIR)
        println("saved $name")
    catch err
        println("$name FAILED: ", sprint(showerror, err))
    end
end

try
    plot_corner(chains, params; output = OUTDIR,
                params_to_plot = ["a_k1", "M_sec_k1", "sesinw_k1", "secosw_k1",
                                  "inc_k1", "Omega_k1"])
    println("saved corner")
catch err
    println("corner FAILED: ", sprint(showerror, err))
end

# --- the sin(i) break ---------------------------------------------------------
# RV alone measures M sin i and cannot reach i. The DR4 abscissae pin i, and
# the same posterior that looks planetary in M sin i is stellar in M.
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))
M_J   = M_sec .* MJUP_PER_MSUN
Msini = M_J .* sin.(inc_v)
i_deg = rad2deg.(inc_v)
i_fold = [x > 90 ? 180 - x : x for x in i_deg]

qb(x) = (quantile(x, 0.16), quantile(x, 0.5), quantile(x, 0.84))

with_theme(nereus_theme()) do
    fig = Figure(; size = (1150, 900))

    # (a) the two mass estimates as intervals, over the regime boundaries.
    # NOT histograms: both posteriors are ~delta functions (σ/μ < 1%) and any
    # binning of them is sub-pixel on a two-decade log axis.
    ax = Axis(fig[1, 1];
              xlabel = "companion mass  (M_Jup)",
              xscale = log10,
              xticks = ([1, 3, 10, 30, 100, 300],
                        ["1", "3", "10", "30", "100", "300"]),
              yticks = ([1, 2], ["RV alone\nM sin i", "RV + DR4\nastrometry\nM"]),
              ygridvisible = false)
    vspan!(ax, 0.5, DEUTERIUM_LIMIT;            color = (colorant"#7fffd4", 0.25))
    vspan!(ax, DEUTERIUM_LIMIT, HYDROGEN_LIMIT; color = (colorant"#c9a0ff", 0.22))
    vspan!(ax, HYDROGEN_LIMIT, 500;             color = (colorant"#ff9ecb", 0.25))

    lo_s, md_s, hi_s = qb(Msini)
    lo_t, md_t, hi_t = qb(M_J)
    for (y, v, med, lo, hi, col) in
            [(1, Msini, md_s, lo_s, hi_s, NEREUS_COLORS.post),
             (2, M_J,   md_t, lo_t, hi_t, colorant"#d62728")]
        # 3σ whisker behind the 1σ bar, so the tails are visible at all
        lines!(ax, [quantile(v, 0.0015), quantile(v, 0.9985)], [y, y];
               color = (col, 0.45), linewidth = 5)
        lines!(ax, [lo, hi], [y, y]; color = col, linewidth = 14)
        scatter!(ax, [med], [y]; color = :white, strokecolor = :black,
                 strokewidth = 1.5, markersize = 16)
        text!(ax, med, y + 0.30;
              text = @sprintf("%.4g M_J", med), align = (:center, :bottom),
              fontsize = 18, color = :black)
    end
    vlines!(ax, [KIEFER_MSUN * MJUP_PER_MSUN]; color = :black,
            linewidth = 2.5, linestyle = :dash)
    text!(ax, KIEFER_MSUN * MJUP_PER_MSUN, 2.44;
          text = "Kiefer+ 2019\ntrue mass ", align = (:right, :bottom),
          fontsize = 16)
    # low enough to contain the M sin i 99.7% tail rather than clip it
    xlims!(ax, 1.5, 400)
    ylims!(ax, 0.4, 2.8)

    for (x, lab) in [(0.13, "planet"), (0.42, "brown dwarf"), (0.82, "star")]
        text!(ax, x, 0.97; text = lab, space = :relative,
              align = (:center, :top), fontsize = 17, color = :black)
    end

    text!(ax, 0.03, 0.62;
          text = @sprintf("M sin i = %.2f [+%.2f, −%.2f] M_J\nM = %.1f [+%.1f, −%.1f] M_J = %.3f M_⊙\ninflation factor 1/sin i = %.1f×",
                          md_s, hi_s - md_s, md_s - lo_s,
                          md_t, hi_t - md_t, md_t - lo_t, md_t / MJUP_PER_MSUN,
                          md_t / md_s),
          space = :relative, align = (:left, :top), fontsize = 17)
    Legend(fig[1, 1],
           [LineElement(color = NEREUS_COLORS.post, linewidth = 14),
            LineElement(color = (NEREUS_COLORS.post, 0.45), linewidth = 5)],
           ["68% credible interval", "99.7% credible interval"];
           tellwidth = false, tellheight = false,
           halign = :right, valign = :bottom, margin = (10, 10, 10, 10),
           framevisible = true, patchsize = (24, 12))

    # (b) the inclination that does the work
    ax2 = Axis(fig[2, 1];
               xlabel = "inclination  (deg, folded to [0°, 90°])",
               ylabel = "posterior density")
    # explicit edges over the plotted window — i_fold ranges to 90°, so a bare
    # bin count would give ~1° bins and smear a 0.08° posterior
    i_hi = 1.25 * max(quantile(i_fold, 0.999), 7.0)
    hist!(ax2, i_fold; bins = range(0, i_hi, length = 110), normalization = :pdf,
          color = (NEREUS_COLORS.hist_face, 0.6),
          strokewidth = 0.4, strokecolor = :black)
    lo_i, md_i, hi_i = qb(i_fold)
    vspan!(ax2, lo_i, hi_i; color = (NEREUS_COLORS.ci, 0.25))
    vlines!(ax2, [md_i]; color = NEREUS_COLORS.model, linewidth = 2)
    vlines!(ax2, [7.0]; color = :crimson, linewidth = 2, linestyle = :dash)
    text!(ax2, 0.03, 0.95;
          text = @sprintf("i = %.2f° [+%.2f, −%.2f]   (Kiefer+ 2019: ≈7°)\nnear face-on ⇒ sin i ≈ %.3f",
                          md_i, hi_i - md_i, md_i - lo_i, sind(md_i)),
          space = :relative, align = (:left, :top), fontsize = 17)
    # frame the posterior, keeping the 7° literature value in view
    xlims!(ax2, -0.4, i_hi)

    rowsize!(fig.layout, 2, Auto(0.55))
    for (lbl, a) in [("(a)", ax), ("(b)", ax2)]
        text!(a, 0.985, 0.96; text = lbl, space = :relative,
              align = (:right, :top), fontsize = 19, font = :bold)
    end

    out = joinpath(OUTDIR, "sini_break.png")
    save(out, fig)
    println("saved sini_break.png")
end

println("plots in $OUTDIR")
