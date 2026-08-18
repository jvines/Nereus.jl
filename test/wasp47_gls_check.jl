#!/usr/bin/env julia
# WASP-47 GLS diagnostic — show whether c, d, e signals are actually
# present in the residuals after subtracting WASP-47b.
#
# Steps:
#   1. Load combined multi-instrument RV CSV
#   2. Drop flagged rows (transit, transit_night, anomalous)
#   3. Median-subtract per instrument
#   4. GLS periodogram on raw centered RVs
#   5. Subtract median-b Keplerian model (Bryant 2022 values, e=0)
#   6. GLS on residuals
#   7. Plot both with vertical markers at literature periods

using DelimitedFiles, Statistics, Printf
using LombScargle
using CairoMakie

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results",
                            "WASP47_joint_search", "gls")
mkpath(OUT_DIR)

# Literature (Bryant & Bayliss 2022)
const Pb_LIT, Kb_LIT  = 4.1591492, 140.84
const Tc_b_LIT        = 2457007.932103
const Pc_LIT          = 588.4
const Pd_LIT          = 9.030672
const Pe_LIT          = 0.789593

# ---- Load ------------------------------------------------------------
raw = readdlm(joinpath(DATADIR, "WASP47_RVs_combined.csv"),
              ',', Any, '\n'; header=true)
mat, hdr = raw[1], vec(raw[2])
col(name) = findfirst(==(name), hdr)

flag_col = col("flag")
keep = trues(size(mat, 1))
for i in 1:size(mat, 1)
    f = mat[i, flag_col]
    f === missing && continue
    s = String(strip(string(f)))
    s in ("transit", "transit_night", "anomalous") && (keep[i] = false)
end
mat = mat[keep, :]

bjd  = Float64.(mat[:, col("bjd")])
rv   = Float64.(mat[:, col("rv")])
σ_rv = Float64.(mat[:, col("rv_err")])
inst = String.(mat[:, col("instrument")])

@printf("Loaded %d RVs after flag-filter, span %.1f d\n",
        length(bjd), maximum(bjd) - minimum(bjd))

# ---- Per-instrument median-subtract ----------------------------------
inst_names = sort(unique(inst))
rv_centered = copy(rv)
for name in inst_names
    mask = inst .== name
    n_i = count(mask)
    n_i == 0 && continue
    med = median(rv[mask])
    rv_centered[mask] .= rv[mask] .- med
    @printf("  %-13s n=%3d  median=%+.2f m/s  σ_centered=%.2f m/s\n",
            name, n_i, med, std(rv_centered[mask]))
end

# ---- GLS on centered raw RVs -----------------------------------------
# Period grid: 0.5 to 2000 days, log-spaced.
const PMIN, PMAX = 0.5, 2000.0
freq_grid = exp.(range(log(1/PMAX), log(1/PMIN); length = 5000))

pgram_raw = lombscargle(bjd, rv_centered, σ_rv;
                         frequencies = freq_grid,
                         normalization = :standard)
periods_raw = 1 ./ freq(pgram_raw)
power_raw   = power(pgram_raw)

# False alarm probability (FAP) thresholds via bootstrap-free analytic
# estimate (Cumming 2004): FAP ≈ M × exp(-z), with M ≈ N_indep ≈
# (df * t_span) where df is mean independent freq spacing. LombScargle
# returns this via fap and fapinv:
fap_levels = (1e-1, 1e-2, 1e-3)
fap_z      = [fapinv(pgram_raw, p) for p in fap_levels]

# ---- Compute b's Keplerian RV at every cadence ----------------------
# Circular orbit: K * cos(2π (t − Tc + P/4) / P)  (transit at ν = 90°
# means RV = 0 with negative-going zero crossing at conjunction; for
# circular orbit the convention is RV = -K sin(2π (t − Tc) / P), so
# at t = Tc the radial velocity is 0 and falls toward redshift.
# Use the negative-sin form:
#       RV_b(t) = -K sin(2π (t − Tc) / P)
# This matches Nereus's Keplerian convention for circular fits.
function rv_b_circular(t)
    return -Kb_LIT * sin(2π * (t - Tc_b_LIT) / Pb_LIT)
end

rv_b_pred = rv_b_circular.(bjd)
rv_resid  = rv_centered .- rv_b_pred

@printf("\nAfter b-subtract (circular): σ_resid by instrument:\n")
for name in inst_names
    mask = inst .== name
    @printf("  %-13s σ=%.2f m/s\n", name, std(rv_resid[mask]))
end

pgram_resid = lombscargle(bjd, rv_resid, σ_rv;
                           frequencies = freq_grid,
                           normalization = :standard)
power_resid = power(pgram_resid)
fap_z_resid = [fapinv(pgram_resid, p) for p in fap_levels]

# Identify top peaks (P_max ≥ FAP=10⁻³) in residuals
peak_periods = Float64[]
peak_powers  = Float64[]
let z3 = fap_z_resid[3]   # FAP = 10⁻³
    n = length(power_resid)
    for i in 2:(n-1)
        if power_resid[i] > power_resid[i-1] &&
           power_resid[i] > power_resid[i+1] &&
           power_resid[i] > z3
            push!(peak_periods, periods_raw[i])
            push!(peak_powers, power_resid[i])
        end
    end
end
order = sortperm(peak_powers; rev = true)
peak_periods = peak_periods[order]
peak_powers  = peak_powers[order]
@printf("\nResidual GLS peaks above FAP=10⁻³:\n")
for k in 1:min(8, length(peak_periods))
    @printf("  rank %d: P = %.4f d   power = %.3f\n",
            k, peak_periods[k], peak_powers[k])
end

# ---- Plot ------------------------------------------------------------
fig = Figure(; size = (1200, 700))

ax1 = Axis(fig[1, 1]; xscale = log10, ylabel = "GLS power",
           title = "WASP-47 — raw RVs (per-instrument median subtracted)")
ax2 = Axis(fig[2, 1]; xscale = log10, xlabel = "Period (days)",
           ylabel = "GLS power",
           title  = "After subtracting circular WASP-47b model")
linkxaxes!(ax1, ax2)

# Periodograms
lines!(ax1, periods_raw, power_raw;   color = :black, linewidth = 1.0)
lines!(ax2, periods_raw, power_resid; color = :black, linewidth = 1.0)

# FAP horizontal levels
fap_colors = (:dodgerblue, :darkorange, :crimson)
for (lvl, z, c) in zip(fap_levels, fap_z, fap_colors)
    hlines!(ax1, [z]; color = c, linestyle = :dash, linewidth = 1.0,
            label = "FAP = $(lvl)")
end
for (z, c) in zip(fap_z_resid, fap_colors)
    hlines!(ax2, [z]; color = c, linestyle = :dash, linewidth = 1.0)
end

# Literature period markers (vertical)
for (P, name, col) in [(Pe_LIT, "e", :green),
                        (Pb_LIT, "b", :red),
                        (Pd_LIT, "d", :blue),
                        (Pc_LIT, "c", :purple)]
    vlines!(ax1, [P]; color = col, linewidth = 1.5,
            linestyle = :dot)
    vlines!(ax2, [P]; color = col, linewidth = 1.5,
            linestyle = :dot)
end

# Annotate the literature period markers on the top axis
for (P, name, col) in [(Pe_LIT, "e", :green),
                        (Pb_LIT, "b", :red),
                        (Pd_LIT, "d", :blue),
                        (Pc_LIT, "c", :purple)]
    text!(ax1, P, maximum(power_raw) * 0.92;
          text = name, color = col, align = (:center, :bottom),
          fontsize = 14)
end

xlims!(ax1, 0.5, 2000)
xlims!(ax2, 0.5, 2000)
axislegend(ax1; position = :rt, framevisible = false, labelsize = 11)

save(joinpath(OUT_DIR, "wasp47_gls_raw_and_residual.png"), fig;
     px_per_unit = 3)
save(joinpath(OUT_DIR, "wasp47_gls_raw_and_residual.pdf"), fig)
println("\nWrote $(joinpath(OUT_DIR, "wasp47_gls_raw_and_residual.png"))")
