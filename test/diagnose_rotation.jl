# Lomb-Scargle on raw K2+TESS binned LCs (b transits masked) — looks
# for the dominant non-transit periodicity. Expected: ~32 d (Becker+
# 2015 measured P_rot from K2).

ENV["WASP47_PT_DIAG_ONLY"] = "1"
include(joinpath(@__DIR__, "wasp47_pt_only.jl"))

using Nereus, Printf
using LombScargle: lombscargle, freqpower

# Simple b-transit mask: phase-fold the LC at b's literature ephemeris
# and discard ±1.5×duration around mid-transit.
const Pb = 4.1591287
# Use the first phot point as a coarse starting reference; transit
# duration is ~3.05 hr ≈ 0.127 d for b → mask ±0.20 d around centre.
const T_DUR = 0.127
const MASK_HALF = 1.5 * T_DUR

# Need an actual Tc. Use a simple BLS-like search to find b's first
# transit centre by scanning Tc over [t1, t1+P) at fine resolution.
function find_first_b_tc(t, f, e)
    Tc_grid = range(t[1], t[1] + Pb; length=400)
    best_χ² = Inf; best_Tc = t[1]
    for Tc in Tc_grid
        χ² = 0.0
        for i in eachindex(t)
            Δt = t[i] - Tc
            Δt -= Pb * round(Δt / Pb)
            if abs(Δt) < T_DUR
                # depth ~21,000 ppm
                model_f = 1 - 0.021 * (1 - (Δt / T_DUR)^2)
                χ² += ((f[i] - model_f) / e[i])^2
            end
        end
        if χ² < best_χ²
            best_χ² = χ²; best_Tc = Tc
        end
    end
    return best_Tc
end

# Use TESS data only for the rotation search (longer baseline, cleaner).
# Find b's Tc on TESS, mask, run LS.
phot_t   = target.data.t_phot
phot_f   = target.data.flux
phot_e   = target.data.flux_err
inst     = target.data.phot_inst
tess     = inst .== 2
t_t = phot_t[tess]; f_t = phot_f[tess]; e_t = phot_e[tess]

println("TESS subset: $(length(t_t)) points, baseline=$(round(t_t[end]-t_t[1]; digits=1)) d")

Tc_b = find_first_b_tc(t_t, f_t, e_t)
@printf("Coarse-fit b's Tc on TESS: %.4f\n", Tc_b)

# Mask b's transits
mask_b = trues(length(t_t))
@inbounds for i in eachindex(t_t)
    Δt = t_t[i] - Tc_b
    Δt -= Pb * round(Δt / Pb)
    if abs(Δt) < MASK_HALF
        mask_b[i] = false
    end
end
n_masked = count(.!mask_b)
@printf("Masked %d points around b's transits (%.1f%% of TESS)\n",
        n_masked, 100*n_masked/length(t_t))

t_clean = t_t[mask_b]; f_clean = f_t[mask_b]; e_clean = e_t[mask_b]

# Lomb-Scargle on b-masked TESS LC, search 1 d to 100 d
freqs = collect(range(1/100, 1/1.0; length=10_000))
ls = lombscargle(t_clean, f_clean .- 1.0, 1 ./ e_clean.^2; frequencies=freqs)
freqs_out, powers = freqpower(ls)
periods = 1 ./ freqs_out

# Top peaks (anti-aliased)
sorted = sortperm(powers; rev=true)
top = Float64[]
println("\nTop 10 peaks in b-masked TESS LC:")
for i in sorted
    P = periods[i]
    if all(abs(log10(P) - log10(P0)) > 0.02 for P0 in top)
        push!(top, P)
        @printf("  P = %7.4f d   power = %.4f\n", P, powers[i])
        length(top) >= 10 && break
    end
end

# Specifically check for P_rot and harmonics
println("\nChecks at expected rotation values (from Becker+ 2015 ~32.5d):")
for P_check in (32.5, 32.5/2, 32.5/3, 32.5/4, 32.5*2)
    j = argmin(abs.(periods .- P_check))
    @printf("  P=%6.2f d (P_rot×%.2f) → power=%.4f\n",
            P_check, P_check/32.5, powers[j])
end

println("\nv13's mystery N_p=3 planets:")
for P_check in (7.52, 20.4)
    j = argmin(abs.(periods .- P_check))
    @printf("  P=%6.2f d → power=%.4f  (≈ P_rot/%.1f)\n",
            P_check, powers[j], 32.5 / P_check)
end
