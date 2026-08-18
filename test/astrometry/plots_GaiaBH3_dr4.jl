#!/usr/bin/env julia
# Render the Gaia BH3 fit (fit_GaiaBH3_dr4_epoch.jl) — a 33 M_sun black hole
# recovered from DR4 epoch astrometry alone.
#
# No phase binning here, unlike Gaia-4: BH3's photocentre orbit is a0 ≈ 27 mas
# against ~0.085 mas per-transit errors (~300σ), so the raw reconstructed
# positions trace the ellipse directly — cf. Panuzzo et al. 2024 Fig. 2.
#
# Usage:  julia --project=. test/astrometry/plots_GaiaBH3_dr4.jl
#   env NEREUS_GAIA_DR4_XML=<path>, NEREUS_OUTDIR=<dir>

using Nereus
using MCMCChains
using CairoMakie
using Statistics: quantile
using Printf
using Nereus: nereus_theme, NEREUS_COLORS

const BH3_SID = 4318465066420528000
const M_PRI, PLX, PLX_ERR = 0.76, 1.6747, 0.0094
const PUB = (P = 4194.7, e = 0.7262, i = 110.659, M = 32.03)

OUTDIR = get(ENV, "NEREUS_OUTDIR", joinpath(@__DIR__, "plots_GaiaBH3_dr4"))
CHAINS = joinpath(OUTDIR, "chains.nc")
isfile(CHAINS) || error("no chains at $CHAINS — run fit_GaiaBH3_dr4_epoch.jl first")

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, BH3_SID)

target = build_target(
    M_pri = M_PRI,
    planets = (BH = (
        a      = LogUniformPrior(3.0, 80.0),
        M_sec  = LogUniformPrior(0.5, 100.0),
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2π),
    ),),
    iad = src.iad,
    plx = NormalPrior(PLX, PLX_ERR),
    M_s = M_PRI,
)
params, data = target.params, target.data
chains, _ = load_chains(CHAINS)
@printf("%d abscissae; posterior %d iter × %d chains\n",
        length(src.iad.t), size(chains, 1), size(chains, 3))

for (fn, name, kw) in [(plot_iad_residuals,          "iad_residuals",          ()),
                       (plot_epoch_astrometry_orbit, "epoch_astrometry_orbit", ())]
    try
        fn(chains, params, data; output = OUTDIR, kw...)
        println("saved $name.png")
    catch err
        println("$name FAILED: ", sprint(showerror, err))
    end
end

try
    plot_corner(chains, params; output = OUTDIR,
                params_to_plot = ["a_k1", "M_sec_k1", "sesinw_k1", "secosw_k1",
                                  "inc_k1", "Omega_k1"])
    println("saved corner.png")
catch err
    println("corner FAILED: ", sprint(showerror, err))
end

# --- recovery vs Panuzzo+ 2024 ------------------------------------------------
a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))

derived = [
    ("P  (d)",        [365.25 * sqrt(a_v[j]^3 / (M_PRI + M_sec[j])) for j in eachindex(a_v)], PUB.P),
    ("M_sec  (M_⊙)",  M_sec,                   PUB.M),
    ("e",             ses .^ 2 .+ sec .^ 2,    PUB.e),
    ("i  (deg)",      rad2deg.(inc_v),         PUB.i),
]

with_theme(nereus_theme()) do
    fig = Figure(; size = (1150, 850))
    for (k, (lab, v, pub)) in enumerate(derived)
        r, c = divrem(k - 1, 2) .+ (1, 1)
        ax = Axis(fig[r, c]; xlabel = lab, ylabel = "posterior density")
        w0 = min(quantile(v, 0.001), pub); w1 = max(quantile(v, 0.999), pub)
        pad = 0.18 * max(w1 - w0, eps())
        hist!(ax, v; bins = range(w0 - pad, w1 + pad, length = 60),
              normalization = :pdf, color = (NEREUS_COLORS.hist_face, 0.55),
              strokewidth = 0.5, strokecolor = :black)
        lo, med, hi = quantile(v, 0.16), quantile(v, 0.5), quantile(v, 0.84)
        vspan!(ax, lo, hi; color = (NEREUS_COLORS.ci, 0.30))
        vlines!(ax, [med]; color = NEREUS_COLORS.model, linewidth = 2)
        vlines!(ax, [pub]; color = :crimson, linewidth = 2, linestyle = :dash)
        text!(ax, 0.03, 0.96;
              text = @sprintf("%.4g  [+%.2g, −%.2g]\npublished %.4g",
                              med, hi - med, med - lo, pub),
              space = :relative, align = (:left, :top), fontsize = 16)
        xlims!(ax, w0 - pad, w1 + pad)
    end
    Legend(fig[3, 1:2],
           [LineElement(color = NEREUS_COLORS.model, linewidth = 2),
            PolyElement(color = (NEREUS_COLORS.ci, 0.30)),
            LineElement(color = :crimson, linewidth = 2, linestyle = :dash)],
           ["posterior median", "68% credible interval",
            "Panuzzo+ 2024 (astrometric solution)"];
           orientation = :horizontal, framevisible = false)
    rowsize!(fig.layout, 3, Auto(0.12))
    save(joinpath(OUTDIR, "recovery.png"), fig)
    println("saved recovery.png")
end
println("plots in $OUTDIR")
