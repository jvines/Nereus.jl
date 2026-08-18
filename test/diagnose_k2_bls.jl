# BLS on K2 EVEREST raw photometry, b masked, to verify d (P=9.03d,
# 970 ppm) and e (P=0.79d, 170 ppm) are detectable in our binary data
# the same way Sinukoff+ 2017 found them.
#
# K2 σ_med ≈ 60 ppm → d S/N per transit ≈ 16, e S/N per transit ≈ 3.
# Cumulative S/N over 70 d baseline:
#   d ≈ 7 transits × √2 in-transit pts × 16 ≈ 60σ
#   e ≈ 88 transits × √2 × 3 ≈ 40σ
# Both should be unambiguously detectable.

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, mean, quantile
using Printf
using BoxLeastSquares: BLS, BLSPeriodogram
using LombScargle
using Random
using CairoMakie

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results",
                            "WASP47_joint_search")
mkpath(OUT_DIR)

const Pb = 4.1591287
const Pd = 9.030672
const Pe = 0.789593
const T_DUR_b = 0.127

# Load K2 raw
k2 = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
@printf("K2: %d points, σ_med=%.2e (%.0f ppm)\n",
        length(k2.t), median(k2.flux_err), median(k2.flux_err)*1e6)

# Mask b transits
function coarse_tc(t, f, e, P, T_dur)
    Tc_grid = range(t[1], t[1] + P; length=800)
    best = Inf; best_Tc = t[1]
    for Tc in Tc_grid
        χ² = 0.0
        for i in eachindex(t)
            Δt = t[i] - Tc
            Δt -= P * round(Δt / P)
            if abs(Δt) < T_dur
                model_f = 1 - 0.021 * (1 - (Δt / T_dur)^2)
                χ² += ((f[i] - model_f) / e[i])^2
            end
        end
        χ² < best && (best = χ²; best_Tc = Tc)
    end
    return best_Tc
end

Tc_b = coarse_tc(k2.t, k2.flux, k2.flux_err, Pb, T_DUR_b)
@printf("Tc_b on K2: %.5f\n", Tc_b)

mask_b = mask_transits(k2.t, [Pb], [Tc_b]; window = 0.04)
keep = .!mask_b
t = k2.t[keep]; f = k2.flux[keep]; e = k2.flux_err[keep]
@printf("After masking b: %d points\n", length(t))

# Lomb-Scargle on b-masked K2
freqs = collect(range(1/30, 1/0.4; length=30_000))
ls = lombscargle(t, f .- 1.0, 1 ./ e.^2; frequencies=freqs)
freqs_out, powers = freqpower(ls)
periods = 1 ./ freqs_out

sorted = sortperm(powers; rev=true)
top = Float64[]
println("\nLS top 10 peaks (K2, b masked):")
for i in sorted
    P = periods[i]
    if all(abs(log10(P) - log10(P0)) > 0.02 for P0 in top)
        push!(top, P)
        @printf("  P = %7.4f d   power = %.4f\n", P, powers[i])
        length(top) >= 10 && break
    end
end

println("\nLit period LS power:")
for (name, P_lit) in (("e", Pe), ("d", Pd), ("32.5d (rot)", 32.5))
    j = argmin(abs.(periods .- P_lit))
    @printf("  %-12s P=%7.4f → power=%.4f\n", name, P_lit, powers[j])
end

# ---------------------------------------------------------------------
# BLS on b-masked K2 — the right tool for transit detection
# ---------------------------------------------------------------------
println("\n--- Running BLS on K2 (b masked) ---")
# BoxLeastSquares.BLS expects (time, flux, flux_err)
# Search periods 0.5d - 30d, durations 1hr - 6hr
periods_search = collect(range(0.5, 30.0; length=20_000))
durations = [0.5/24, 1/24, 1.5/24, 2/24, 3/24, 4/24, 6/24]  # hours → days

bls_pgram = BLS(t, f, e; duration=durations, periods=periods_search)
println("BLS pgram type: ", typeof(bls_pgram))

# Top BLS peaks
@printf("BLS pgram fields: %s\n", propertynames(bls_pgram))

# Use whatever the field names are
P_arr = bls_pgram.periods
pow_arr = bls_pgram.power
dur_arr = hasproperty(bls_pgram, :duration) ? bls_pgram.duration :
          fill(NaN, length(P_arr))

sorted_bls = sortperm(pow_arr; rev=true)
top_bls = Float64[]
println("\nBLS top 10 peaks:")
for i in sorted_bls
    P = P_arr[i]
    if all(abs(log10(P) - log10(P0)) > 0.02 for P0 in top_bls)
        push!(top_bls, P)
        @printf("  P = %7.4f d   power = %.4f   dur = %.4f d\n",
                P, pow_arr[i], dur_arr[i])
        length(top_bls) >= 10 && break
    end
end

println("\nLit period BLS power:")
for (name, P_lit) in (("e", Pe), ("d", Pd))
    j = argmin(abs.(P_arr .- P_lit))
    @printf("  %-3s P=%7.4f → power=%.4f, dur=%.4f d\n",
            name, P_lit, pow_arr[j], dur_arr[j])
end

# Plot
let fig = Figure(size=(1200, 720))
    ax1 = Axis(fig[1, 1], xlabel="Period (d)", ylabel="LS power",
               xscale=log10, title="K2 EVEREST (b masked) — Lomb-Scargle")
    lines!(ax1, periods, powers, color=:black, linewidth=0.7)
    for (name, P_lit) in (("e", Pe), ("d", Pd))
        vlines!(ax1, [P_lit]; color=:red, linestyle=:dash)
        text!(ax1, P_lit, maximum(powers) * 0.9; text=name,
              align=(:center, :bottom), color=:red)
    end

    ax2 = Axis(fig[2, 1], xlabel="Period (d)", ylabel="BLS power",
               xscale=log10, title="K2 EVEREST (b masked) — BLS")
    lines!(ax2, P_arr, pow_arr, color=:black, linewidth=0.7)
    for (name, P_lit) in (("e", Pe), ("d", Pd))
        vlines!(ax2, [P_lit]; color=:red, linestyle=:dash)
        text!(ax2, P_lit, maximum(bls_pgram.power) * 0.9; text=name,
              align=(:center, :bottom), color=:red)
    end

    save(joinpath(OUT_DIR, "diag_k2_ls_bls.pdf"), fig)
    println("\n✓ saved diag_k2_ls_bls.pdf")
end
