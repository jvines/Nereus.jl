#!/usr/bin/env julia
# Verify the noise-limited-posterior hypothesis: at each target's
# *published* orbit, what is the expected along-scan reflex amplitude
# per IAD transit, and how does it compare to the per-transit σ?
#
# If the cumulative √N × (signal/σ) is large, IAD should pin
# inclination tightly. If marginal, broad posteriors are expected.

using Nereus
using Statistics: std, mean, median
using Printf

include_target(hip, name, M_pri, plx, P_d, M_sec_MJ, e, ω_deg, Tp_jd, i_deg, Ω_deg) =
    (hip=hip, name=name, M_pri=M_pri, plx=plx, P_d=P_d,
     M_sec=M_sec_MJ/1047.57, e=e, ω=deg2rad(ω_deg),
     Tp=Tp_jd, i=deg2rad(i_deg), Ω=deg2rad(Ω_deg))

targets = [
    include_target(24205, "HD 33636 (Bean+ 2007)",
                   1.02, 35.25,  2447.3, 142.0, 0.4805, 169.4,
                   2_451_772.0, 4.1, 0.0),
    include_target(27253, "HD 38529 c (Benedict+ 2010)",
                   1.48, 24.34,  2134.0,  17.6, 0.36,   17.7,
                   2_450_240.4, 48.0, 0.0),
    include_target(16537, "ε Eri (Llop-Sayson+ 2021)",
                   0.82, 311.0,  2700.0,  0.74, 0.07,    0.0,
                   2_454_500.0, 78.0, 0.0),
]

println("=" ^ 78)
println("IAD signal-to-noise verification")
println("=" ^ 78)
println(rpad("Target", 32),
        rpad("a_⋆(mas)", 11), rpad("|Δη|(mas)", 11),
        rpad("σ_med(mas)", 12), rpad("N", 5),
        rpad("S/N_med", 9), "cum S/N")
println("-" ^ 78)

for tgt in targets
    iad = fetch_hip_iad(tgt.hip; verbose=false)
    n = n_iad(iad)
    σ_med = median(iad.abscissa_err)

    # Build orbit at published values
    orb = Nereus.build_orbit(tgt.P_d, tgt.e, tgt.ω, tgt.Ω, tgt.i,
                              tgt.M_pri, tgt.M_sec, tgt.Tp, tgt.plx)

    # Stellar reflex semi-major axis in mas
    a_AU = Nereus.a_from_P(tgt.P_d, tgt.M_pri + tgt.M_sec)
    a_star_AU = a_AU * tgt.M_sec / (tgt.M_pri + tgt.M_sec)
    a_star_mas = a_star_AU * tgt.plx

    # Compute along-scan reflex Δη at each IAD transit
    Δη = Float64[]
    for j in 1:n
        Δα, Δδ = Nereus.star_reflex_offset(orb, iad.t[j], tgt.M_sec)
        push!(Δη, Nereus.along_scan_projection(Δα, Δδ, iad.psi[j]))
    end
    rms_signal = sqrt(mean(Δη .^ 2))
    snr_med = rms_signal / σ_med
    cum_snr = sqrt(n) * snr_med

    @printf("%-32s%-11.3f%-11.3f%-12.3f%-5d%-9.2f%.1f\n",
            tgt.name, a_star_mas, rms_signal, σ_med, n, snr_med, cum_snr)
end

println("-" ^ 78)
println("\nInterpretation:")
println("  cum S/N ≥ 10  → IAD strongly constrains inclination")
println("  cum S/N ~ 5   → IAD marginal; broad i posterior expected")
println("  cum S/N ≲ 3   → IAD essentially uninformative\n")
