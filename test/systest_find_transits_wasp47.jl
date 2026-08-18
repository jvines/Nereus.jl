#!/usr/bin/env julia
# System test: production find_transits (detrend + BLS + harmonic dedup) on the
# WASP-47 K2 LC. Compare scale_durations OFF vs ON (now the default). Pass =
# all three of b(4.159)/d(9.03)/e(0.789) recovered as candidates, and e on its
# FUNDAMENTAL (not the 3×Pe=2.37 alias) with scale_durations on.

using Nereus
using Printf

const DATADIR = abspath(joinpath(@__DIR__, "..", "..", "data", "WASP47"))
lc = load_tess_lc(joinpath(DATADIR, "WASP-47_k2_c03_everest_lc.csv"))

idname(P) = abs(P-4.159)<0.05 ? "b" : abs(P-9.03)<0.12 ? "d" :
            abs(P-0.7896)<0.01 ? "e" : abs(P-2.369)<0.03 ? "(3×Pe alias)" : ""

for sd in (false, true)
    res = find_transits(lc.t, lc.flux, lc.flux_err;
                        method=:bls, scale_durations=sd,
                        detrend=:savgol, detrend_kwargs=(window_length=151,),
                        R_s=1.137, M_s=1.04,
                        period_min=0.5, period_max=12.0,
                        n_peaks=40, score_threshold=7.0, max_candidates=12)
    @printf("\n=== scale_durations=%-5s : %d candidates ===\n", sd, length(res.periods))
    got = String[]
    for i in eachindex(res.periods)
        nm = idname(res.periods[i])
        nm in ("b","d","e") && push!(got, nm)
        @printf("  P=%8.4f d  SNR=%6.1f  depth=%6d ppm  Rp=%5.1f Re   %s\n",
                res.periods[i], res.snr[i], round(Int,res.depths[i]*1e6),
                res.R_p_earth[i], nm)
    end
    @printf("  --> recovered: %s  %s\n", join(sort(unique(got)), " "),
            all(x->x in got, ["b","d","e"]) ? "ALL THREE ✓" : "MISSING " * join(setdiff(["b","d","e"], got), ","))
end
