#!/usr/bin/env julia
# Notch on a WELL-BEHAVED light curve — the regime it's actually built for and
# the one that matters operationally: notch DETECTS the transits (delta_bic), we
# mask them, and a GP models the stellar variability afterwards. So this asserts
# BOTH halves: (a) the detrended baseline is white-clean and the transit depth
# survives, and (b) every injected transit is DETECTED (a dip lands on it) with
# no spurious detections away from a transit. Quiet star: gentle slow rotation
# (P=18 d → ~1.5 cycles, low curvature) + a slow drift + clean 1% transits.
using Nereus, Random, Printf, Statistics

out = joinpath(@__DIR__, "..", "..", "results", "detrend_check"); mkpath(out)
rng = MersenneTwister(20260629)

# --- well-behaved LC: 27 d, 10-min cadence ----------------------------------
P_rot, P_tr, T0, depth, dur = 18.0, 4.05, 2.0, 0.010, 0.12     # days; depth frac
t = collect(0.0:(10/1440):27.0); n = length(t)
ph = mod.(t .- T0 .+ 0.5P_tr, P_tr) ./ P_tr .- 0.5
intr = abs.(ph) .< (dur / 2) / P_tr
σ = 0.0008
cont = 1.0 .+ 0.008 .* sin.(2π .* t ./ P_rot) .+ 0.002 .* (t ./ 27)   # gentle activity + drift
flux = cont .+ intr .* (-depth) .+ σ .* randn(rng, n)
t = Vector{Float64}(t); flux = Vector{Float64}(flux); flux_err = fill(σ, n)
tcenters = [T0 + k * P_tr for k in 0:Int(fld(27.0 - T0, P_tr))]    # injected transit times
@printf("LC: %d cadences (10-min, 27 d), %d transits @ P=%.2f d, depth %.1f%%, σ=%d ppm\n",
        n, length(tcenters), P_tr, depth * 100, round(Int, σ * 1e6))

ok = true
chk(nm, c) = (global ok; ok &= c; @printf("    [%s] %s\n", c ? "PASS" : "FAIL", nm))

res = detrend_notch(t, flux, flux_err; window = 1.0)
fd  = res.flux_detrended
oot = .!intr

# (a) detrend quality -------------------------------------------------------
base_std = std(fd[oot]); base_med = median(fd[oot])
rec_depth = base_med - median(fd[intr])
# correlated (binned) residual vs the white-noise floor
edges = 0.0:0.5:27.0; bm = Float64[]
for k in 1:length(edges)-1
    m = (t .>= edges[k]) .& (t .< edges[k+1]) .& oot
    count(m) >= 5 && push!(bm, median(fd[m]) - 1)
end
white = σ / sqrt(0.5 / (10 / 1440)); excess = std(bm) / white
@printf("  detrend : OOT std=%.4f med=%.4f | depth rec=%.4f (truth %.4f) | binned-resid %.1f× white\n",
        base_std, base_med, rec_depth, depth, excess)
chk("baseline flattened (OOT std < 1.5 σ)",        base_std < 1.5σ)
chk("baseline level ≈ 1",                          abs(base_med - 1) < 0.001)
chk("transit depth preserved (within 15%)",        abs(rec_depth - depth) / depth < 0.15)
chk("baseline white-clean (binned < 1.8× floor)",  excess < 1.8)

# (b) detection: real transits tower over the noise floor -------------------
# Detection = a Δbic threshold (the default -1 is the per-cadence detrending
# accept, NOT a candidate threshold). What matters is SEPARATION: every
# transit's peak Δbic must sit far above the strongest spurious (noise) dip
# away from any transit — then any threshold in the gap recovers all and only
# the real transits.
db = res.delta_bic
real_peak = [maximum(db[abs.(t .- tc) .< 0.12]) for tc in tcenters]
far = trues(n); for tc in tcenters; far .&= abs.(t .- tc) .>= 0.3; end
far_max = maximum(db[far])
sep = minimum(real_peak) / max(far_max, 1.0)
@printf("  detect  : %d/%d transits | peak Δbic %.0f–%.0f | worst spurious Δbic=%.0f | separation=%.1f×\n",
        count(real_peak .> 5 * far_max), length(tcenters),
        minimum(real_peak), maximum(real_peak), far_max, sep)
chk("every transit detected (peak Δbic > 0)",            all(real_peak .> 0))
chk("transits cleanly separable (peak ≫ 5× worst noise)", sep > 5)

Nereus.plot_detrending(t, flux, flux_err, res; output = out, name = "notch_wellbehaved")

println(ok ? "\n✅ NOTCH DETECTION (well-behaved LC) PASS" :
             "\n❌ NOTCH DETECTION — FAILURES ABOVE")
@printf("plot: %s\n", joinpath(out, "detrend", "notch_wellbehaved.png"))
