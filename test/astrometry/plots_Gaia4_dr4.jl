#!/usr/bin/env julia
# Render the Gaia-4 b DR4 epoch-astrometry fit (fit_Gaia4_dr4_epoch.jl).
#
# Reads the posterior that fit saved to plots_Gaia4_dr4/chains.nc, so plot
# style can be iterated without re-running Pigeons. Rebuilding the target is
# deterministic and costs nothing; only the sampler is expensive.
#
# Figures:
#   iad_residuals.png  — DR4 along-scan residuals vs ψ and vs MJD
#   orbit_skyplane.png — sky-projected reflex orbit
#   corner.png         — orbital-parameter posterior
#   recovery.png       — P, M, e, i posteriors against Stefansson+ 2024
#
# Usage:  julia --project=. test/astrometry/plots_Gaia4_dr4.jl
#   env NEREUS_GAIA_DR4_XML=<path>  reuse a local VOTable (else it fetches)

using Nereus
using MCMCChains
using CairoMakie
using Statistics: quantile
using Printf
using Nereus: nereus_theme, NEREUS_COLORS, NEREUS_CMAP

const GAIA4_SID = 1457486023639239296
const M_PRI     = 0.644
const PLX_GAIA  = 13.628
const PLX_ERR   = 0.021

# Stefansson+ 2024 (arXiv:2410.05654), Gaia+RV adopted solution
const PUB = (P = 571.3, M_J = 11.8, e = 0.338, i = 116.9)

OUTDIR = joinpath(@__DIR__, "plots_Gaia4_dr4")
CHAINS = joinpath(OUTDIR, "chains.nc")
isfile(CHAINS) || error("no chains at $CHAINS — run fit_Gaia4_dr4_epoch.jl first")

# --- rebuild the target exactly as the fit did -------------------------------
xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, GAIA4_SID)

target = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a      = LogUniformPrior(0.3, 4.0),
        M_sec  = LogUniformPrior(0.001, 0.05),
        sesinw = UniformPrior(-1.0, 1.0),
        secosw = UniformPrior(-1.0, 1.0),
        Mo     = UniformPrior(0.0, 2π),
        inc    = SinePrior(),
        Omega  = UniformPrior(0.0, 2π),
    ),),
    iad = src.iad,
    plx = NormalPrior(PLX_GAIA, PLX_ERR),
    M_s = M_PRI,
)
params, data = target.params, target.data

chains, meta = load_chains(CHAINS)
@printf("%d abscissae; posterior %d iter × %d chains\n",
        length(src.iad.t), size(chains, 1), size(chains, 3))

# --- the standard astrometry suite -------------------------------------------
# NOT plot_orbit_skyplane: DR4 epoch astrometry is 1-D along-scan abscissae, so
# there are no 2-D sky positions to overlay and that plot can only draw a bare
# model ellipse with zero data on it. al_orbit_signal.png below is the
# data-bearing equivalent.
try
    plot_iad_residuals(chains, params, data; output = OUTDIR)
    println("saved iad_residuals.png")
catch err
    println("iad_residuals FAILED: ", sprint(showerror, err))
end

# Gaia-4's photocentre orbit is a0 ≈ 0.3 mas against ~0.10 mas per-transit
# errors — ~3σ per point, so the raw reconstructed cloud swamps the ellipse.
# 824 transits binned into 24 phase bins buy the √N needed to see it.
try
    plot_epoch_astrometry_orbit(chains, params, data; output = OUTDIR,
                                 n_phase_bins = 24)
    println("saved epoch_astrometry_orbit.png")
catch err
    println("epoch_astrometry_orbit FAILED: ", sprint(showerror, err))
end

# --- the orbit as it actually appears in the data ----------------------------
# Strip the 5-parameter single-star solution from the abscissae, leaving the
# orbital reflex, and compare it against the fitted orbit's along-scan
# projection. This is the "is the orbit really in the data" figure.
let
    theta_med = Nereus._theta_median(chains, params)
    M_pri_m = astrom_M_pri(theta_med)
    plx_m   = astrom_plx(theta_med)
    orb, M_sec_m = Nereus._planet_orbit(theta_med, 1, M_pri_m, plx_m, data.t_ref)

    iad = data.iad
    n   = length(iad.t)
    model_al = Vector{Float64}(undef, n)
    for j in 1:n
        Δra, Δdec = Nereus.star_reflex_offset(orb, iad.t[j], M_sec_m)
        model_al[j] = Nereus.along_scan_projection(Δra, Δdec, iad.psi[j])
    end

    # 5-param solution fitted to the ORBIT-SUBTRACTED abscissae, so the
    # astrometric parameters don't soak up the orbit we are trying to show
    st, ct = sin.(iad.psi), cos.(iad.psi)
    D = hcat(st, ct, iad.parallax_factor, st .* iad.pm_factor, ct .* iad.pm_factor)
    W = 1.0 ./ iad.abscissa_err
    p = (D .* W) \ ((iad.abscissa .- model_al) .* W)
    data_al = iad.abscissa .- D * p          # astrometric solution removed, orbit kept
    resid   = data_al .- model_al

    @printf("AL orbit signal: data RMS %.3f mas, model RMS %.3f mas, residual RMS %.3f mas\n",
            sqrt(sum(abs2, data_al) / n), sqrt(sum(abs2, model_al) / n),
            sqrt(sum(abs2, resid) / n))

    with_theme(nereus_theme()) do
        fig = Figure(; size = (1150, 900))
        tcol = iad.t .- minimum(iad.t)

        ax1 = Axis(fig[1, 1];
                   xlabel = "model along-scan reflex  (mas)",
                   ylabel = "observed along-scan signal  (mas)")
        lo = minimum(vcat(data_al, model_al)) - 0.05
        hi = maximum(vcat(data_al, model_al)) + 0.05
        lines!(ax1, [lo, hi], [lo, hi]; color = :black, linestyle = :dash, linewidth = 2)
        errorbars!(ax1, model_al, data_al, iad.abscissa_err;
                   color = (:gray, 0.5), linewidth = 1)
        sc = scatter!(ax1, model_al, data_al; color = tcol, colormap = NEREUS_CMAP,
                      markersize = 11, strokewidth = 0.4, strokecolor = :black)
        Colorbar(fig[1, 2], sc; label = "MJD − $(round(Int, minimum(iad.t)))")
        text!(ax1, 0.03, 0.96;
              text = @sprintf("%d abscissae\ndata RMS %.3f mas\nresidual RMS %.3f mas",
                              n, sqrt(sum(abs2, data_al) / n), sqrt(sum(abs2, resid) / n)),
              space = :relative, align = (:left, :top), fontsize = 17)
        limits!(ax1, lo, hi, lo, hi)

        ax2 = Axis(fig[2, 1]; xlabel = "MJD",
                   ylabel = "observed − model  (mas)")
        hlines!(ax2, [0.0]; color = :black, linestyle = :dash)
        errorbars!(ax2, iad.t, resid, iad.abscissa_err; color = (:gray, 0.5), linewidth = 1)
        scatter!(ax2, iad.t, resid; color = tcol, colormap = NEREUS_CMAP,
                 markersize = 10, strokewidth = 0.4, strokecolor = :black)
        rowsize!(fig.layout, 2, Auto(0.5))

        out = joinpath(OUTDIR, "al_orbit_signal.png")
        save(out, fig)
        println("saved al_orbit_signal.png")
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

# --- recovery against the published solution ---------------------------------
# The fit's PASS/FAIL gates as a figure: derived physical posteriors with the
# Stefansson+ 2024 value marked. Astrometry-only vs their Gaia+RV adopted fit,
# so ~1-2σ offsets are expected, not error.
a_v   = vec(Array(chains[:, :a_k1, :]))
M_sec = vec(Array(chains[:, :M_sec_k1, :]))
ses   = vec(Array(chains[:, :sesinw_k1, :]))
sec   = vec(Array(chains[:, :secosw_k1, :]))
inc_v = vec(Array(chains[:, :inc_k1, :]))

derived = [
    ("P  (d)",        [365.25 * sqrt(a_v[j]^3 / (M_PRI + M_sec[j])) for j in eachindex(a_v)], PUB.P),
    ("M_sec  (M_J)",  M_sec .* 1047.57,        PUB.M_J),
    ("e",             ses .^ 2 .+ sec .^ 2,    PUB.e),
    ("i  (deg)",      rad2deg.(inc_v),         PUB.i),
]

with_theme(nereus_theme()) do
    fig = Figure(; size = (1150, 850))
    for (k, (lab, v, pub)) in enumerate(derived)
        r, c = divrem(k - 1, 2) .+ (1, 1)
        ax = Axis(fig[r, c]; xlabel = lab, ylabel = "posterior density")
        # Bin over the window this panel actually shows. A bare bin count
        # spans the full prior support, and after the zoom below only two or
        # three bins survive — the panel then reads as one solid block.
        w0 = min(quantile(v, 0.001), pub)
        w1 = max(quantile(v, 0.999), pub)
        wpad = 0.18 * max(w1 - w0, eps())
        hist!(ax, v; bins = range(w0 - wpad, w1 + wpad, length = 60),
              normalization = :pdf,
              color = (NEREUS_COLORS.hist_face, 0.55),
              strokewidth = 0.5, strokecolor = :black)
        lo, med, hi = quantile(v, 0.16), quantile(v, 0.5), quantile(v, 0.84)
        vspan!(ax, lo, hi; color = (NEREUS_COLORS.ci, 0.25))
        vlines!(ax, [med]; color = NEREUS_COLORS.model, linewidth = 2)
        vlines!(ax, [pub]; color = :crimson, linewidth = 2, linestyle = :dash)
        text!(ax, 0.03, 0.96;
              text = @sprintf("%.3g  [+%.2g, −%.2g]\npublished %.4g",
                              med, hi - med, med - lo, pub),
              space = :relative, align = (:left, :top), fontsize = 16)
        # Frame the posterior, not the prior — but always keep the published
        # value in view so the comparison the panel exists for stays visible.
        xlims!(ax, w0 - wpad, w1 + wpad)
    end
    Legend(fig[3, 1:2],
           [LineElement(color = NEREUS_COLORS.model, linewidth = 2),
            PolyElement(color = (NEREUS_COLORS.ci, 0.25)),
            LineElement(color = :crimson, linewidth = 2, linestyle = :dash)],
           ["posterior median", "68% credible interval",
            "Stefansson+ 2024 (Gaia+RV)"];
           orientation = :horizontal, framevisible = false)
    rowsize!(fig.layout, 3, Auto(0.12))
    out = joinpath(OUTDIR, "recovery.png")
    save(out, fig)
    println("saved recovery.png")
end

println("plots in $OUTDIR")
