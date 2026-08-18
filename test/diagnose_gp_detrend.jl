# Raw (UNBINNED) photometry + GP rotation detrend, then LS to see if
# d (P=9.03d, depth 970 ppm) and e (P=0.79d, depth 170 ppm) emerge once
# the rotation/activity baseline is removed.
#
# - K2 C03 EVEREST: ~30-min cadence
# - TESS sectors 42, 92: 2-min or 30-min cadence (whatever the loader gives)
# - mask b's transits at lit ephemeris (P=4.159, Tc coarse-fit per sector)
# - detrend_gp with CeleriteRotation kernel seeded at P_rot ~ 32d
# - LS on detrended residuals over [0.5d, 30d]

using Nereus
using DelimitedFiles: readdlm
using Statistics: median, std, quantile, mean
using Printf
using LombScargle: lombscargle, freqpower
using Random
using CairoMakie

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DATADIR   = joinpath(REPO_ROOT, "data", "WASP47")
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results",
                            "WASP47_joint_search")
mkpath(OUT_DIR)

const Pb = 4.1591287
const T_DUR = 0.127  # b's transit half-duration (~3 hr → 0.127d full / 2)

# 1. Load raw K2 EVEREST (no binning)
k2 = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))
@printf("K2 raw: %d points, baseline=%.1f d, σ_med=%.2e\n",
        length(k2.t), k2.t[end] - k2.t[1], median(k2.flux_err))

# 2. Load raw TESS sectors (no binning)
sector_files = sort(filter(f -> occursin(r"WASP-47_tess_s\d+_lc\.csv$", f),
                            readdir(DATADIR; join=true)))
tess_data = []
for fp in sector_files
    lc = load_tess_lc(fp)
    push!(tess_data, lc)
    @printf("  TESS %s: %d points, baseline=%.1f d, σ_med=%.2e\n",
            basename(fp), length(lc.t), lc.t[end]-lc.t[1], median(lc.flux_err))
end

# 3. Combine into a single time series with sector_id for the GP
all_t = Float64[]; all_f = Float64[]; all_e = Float64[]; all_sec = Int[]
append!(all_t, k2.t); append!(all_f, k2.flux); append!(all_e, k2.flux_err)
append!(all_sec, fill(1, length(k2.t)))
for (i, lc) in enumerate(tess_data)
    append!(all_t, lc.t); append!(all_f, lc.flux); append!(all_e, lc.flux_err)
    append!(all_sec, fill(i + 1, length(lc.t)))
end
perm = sortperm(all_t)
all_t = all_t[perm]; all_f = all_f[perm]; all_e = all_e[perm]; all_sec = all_sec[perm]
@printf("Combined: %d raw points, %d sectors\n",
        length(all_t), length(unique(all_sec)))

# 4. Coarse-fit b's Tc on the K2 data (which has the cleanest first transits)
function coarse_tc(t, f, e, P, T_dur)
    n_grid = 800
    t_low = t[1]; t_hi = t[1] + P
    Tc_grid = range(t_low, t_hi; length=n_grid)
    best = Inf; best_Tc = t_low
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
        if χ² < best
            best = χ²; best_Tc = Tc
        end
    end
    return best_Tc
end

Tc_b = coarse_tc(k2.t, k2.flux, k2.flux_err, Pb, T_DUR)
@printf("\nCoarse-fit b's Tc on K2: %.5f\n", Tc_b)

# 5. Mask b's transits across the full combined LC (±2× transit duration)
mask_b = mask_transits(all_t, [Pb], [Tc_b]; window = 0.05)
@printf("Masked %d b-transit cadences (%.2f%%)\n",
        count(mask_b), 100*count(mask_b)/length(all_t))

# 6. Build CeleriteRotation kernel seeded at P_rot=32.5d (Becker+ 2015)
kernel = CeleriteRotation(channel = :phot, instruments = String[])
# σ (overall amplitude) capped at 5e-4 = 500 ppm — well below e's
# 170 ppm transit depth doesn't matter (we wouldn't detrend e anyway),
# but keeps GP from absorbing d's 970 ppm transits. Stellar rotation
# amplitudes for slow rotators are typically 100-1000 ppm.
gp_priors = Dict{String, PriorSpec}(
    "gp_sigma_phot"   => LogUniformPrior(1e-5, 5e-4),
    "gp_period_phot"  => LogUniformPrior(20.0, 60.0),
    "gp_Q0_phot"      => LogUniformPrior(1.0, 30.0),
    "gp_dQ_phot"      => LogUniformPrior(1.0, 30.0),
    "gp_f_phot"       => UniformPrior(0.05, 0.95),
)
# RAW init params (CeleriteRotation uses unlogged values directly).
init_params = [2e-4, 32.5, 5.0, 5.0, 0.5]

println("\nFitting GP (CeleriteRotation, init P_rot=32.5d)...")
result = detrend_gp(all_t, all_f, all_e, kernel;
                     transit_mask = mask_b,
                     init_params  = init_params,
                     gp_priors    = gp_priors,
                     joint_segments = true,
                     sector_id    = all_sec)
@printf("GP fit MAP params: %s\n", result.gp_params[1])
@printf("Per-sector offsets: %s\n", result.offsets)

flux_clean = result.flux_detrended

# 7. Lomb-Scargle on cleaned residuals + b removed
# Use raw weights (transit_mask was for the GP fit only)
freqs = collect(range(1/30.0, 1/0.5; length=20_000))
ls = lombscargle(all_t, flux_clean .- mean(flux_clean), 1 ./ all_e.^2;
                 frequencies = freqs)
freqs_out, powers = freqpower(ls)
periods = 1 ./ freqs_out

# Top peaks
sorted = sortperm(powers; rev=true)
top = Float64[]
println("\nTop 10 peaks in GP-detrended LC, b masked:")
for i in sorted
    P = periods[i]
    if all(abs(log10(P) - log10(P0)) > 0.02 for P0 in top)
        push!(top, P)
        @printf("  P = %7.4f d   power = %.4f\n", P, powers[i])
        length(top) >= 10 && break
    end
end

# Specific lit-period checks
println("\nLit period checks (post-detrend):")
for (name, P_lit) in (("e", 0.789593), ("b (masked)", Pb), ("d", 9.030672),
                       ("c (no transit)", 588.4))
    if P_lit > 30
        @printf("  %s P=%.4f d → outside LS range\n", name, P_lit)
        continue
    end
    j = argmin(abs.(periods .- P_lit))
    @printf("  %s P=%7.4f d → power=%.4f\n", name, P_lit, powers[j])
end

# Also check rotation harmonics
println("\nRotation harmonics:")
for k in (1, 2, 3, 4)
    P_h = 32.5 / k
    P_h > 30 && continue
    j = argmin(abs.(periods .- P_h))
    @printf("  P_rot/%d = %6.2f d → power=%.4f\n", k, P_h, powers[j])
end

# Plot the LS
let fig = Figure(size=(1200, 360))
    ax = Axis(fig[1, 1], xlabel="Period (d)", ylabel="LS power",
              xscale=log10,
              title="GP-detrended phot LC, b masked — LS")
    lines!(ax, periods, powers, color=:black, linewidth=0.7)
    for (name, P_lit) in (("e", 0.789593), ("d", 9.030672))
        vlines!(ax, [P_lit]; color=:red, linestyle=:dash, linewidth=1.0)
        text!(ax, P_lit, maximum(powers) * 0.9; text=name,
              align=(:center, :bottom), color=:red, fontsize=12)
    end
    save(joinpath(OUT_DIR, "diag_gp_detrend_ls.pdf"), fig)
    println("\n✓ saved $OUT_DIR/diag_gp_detrend_ls.pdf")
end

# Save the GP-detrended LC for re-use in v14 search
open(joinpath(OUT_DIR, "wasp47_gp_detrended.csv"), "w") do io
    println(io, "t,flux,flux_err,sector_id")
    for i in eachindex(all_t)
        @printf(io, "%.6f,%.6e,%.6e,%d\n",
                all_t[i], flux_clean[i] + 1.0, all_e[i], all_sec[i])
    end
end
println("✓ saved GP-detrended LC to $OUT_DIR/wasp47_gp_detrended.csv")
