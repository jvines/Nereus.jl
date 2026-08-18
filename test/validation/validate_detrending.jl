#!/usr/bin/env julia
# Detrending validation. Inject stellar variability (rotation 1.5% + slow drift)
# + a 1% transit into a clean LC, run each detrender, and assert BOTH:
#   (a) out-of-transit baseline is flattened to ~the noise level (trend removed),
#   (b) the transit DEPTH survives (the detrender didn't eat the signal).
# (b) is the one that catches the classic detrender failure. Renders the
# plot_detrending diagnostic for each method. Covers savgol / GP / notch / LOCoR.
using Nereus, Random, Printf, Statistics

out = joinpath(@__DIR__, "..", "..", "results", "detrend_check"); mkpath(out)
rng = MersenneTwister(20260628)

# --- synthetic LC: 20 d, 20-min cadence -------------------------------------
P_rot, P_tr, T0, depth, dur = 4.0, 3.0, 1.5, 0.010, 0.125     # days; depth frac; 3 hr
t = collect(0.0:(20/1440):20.0); n = length(t)
ph = mod.(t .- T0 .+ 0.5P_tr, P_tr) ./ P_tr .- 0.5
intr = abs.(ph) .< (dur / 2) / P_tr                          # in-transit cadences
σ = 0.0010
flux = 1.0 .+ 0.015 .* sin.(2π .* t ./ P_rot) .+            # rotation (trend)
       0.005 .* (t ./ 20) .+                                # slow drift (trend)
       (intr .* (-depth)) .+ σ .* randn(rng, n)             # transit + white noise
flux_err = fill(σ, n)
t = Vector{Float64}(t); flux = Vector{Float64}(flux); flux_err = Vector{Float64}(flux_err)
tmask = mask_transits(t, [P_tr], [T0])
@printf("LC: %d cadences, %d in-transit; trend amp ≈ 1.5%%, transit depth %.1f%%\n",
        n, count(intr), depth * 100)

ok = true
chk(nm, c) = (global ok; ok &= c; @printf("    [%s] %s\n", c ? "PASS" : "FAIL", nm))

function assess(name, res)
    fd = res.flux_detrended
    oot = .!intr
    base_std = std(fd[oot])                       # OOT scatter after detrending
    base_med = median(fd[oot])                    # OOT level (want ≈ 1)
    rec_depth = base_med - median(fd[intr])       # recovered transit depth
    @printf("  %-7s: OOT median=%.4f std=%.4f (raw var≈1.5%%) | transit depth recovered=%.4f (truth %.4f)\n",
            name, base_med, base_std, rec_depth, depth)
    chk("$name flattens baseline (OOT std < 0.3%)", base_std < 0.003)
    chk("$name OOT level ≈ 1 (|med-1|<0.3%)", abs(base_med - 1.0) < 0.003)
    chk("$name PRESERVES transit (depth within 25%)", abs(rec_depth - depth) / depth < 0.25)
    try
        Nereus.plot_detrending(t, flux, flux_err, res;
                                output = out, name = name)
    catch e; @warn "plot_detrending($name) failed" exception=e; end
end

println("\n=== Savitzky-Golay (transit-masked) ===")
assess("savgol", detrend_savgol(t, flux, flux_err; window_length = 121, polyorder = 2,
                                transit_mask = tmask))

println("\n=== GP / celerite (transit-masked) ===")
assess("gp", detrend_gp(t, flux, flux_err, CeleriteRotation(); transit_mask = tmask))

println("\n=== Notch filter (self-detects the dip) ===")
assess("notch", detrend_notch(t, flux, flux_err; window = 1.0))

println("\n=== LOCoR (rotation; transit-masked) ===")
assess("locor", detrend_locor(t, flux, flux_err; P_rot = P_rot, transit_mask = tmask))

println(ok ? "\n✅ DETRENDING VALIDATION PASS" : "\n❌ DETRENDING VALIDATION — FAILURES ABOVE")
@printf("plots: %s\n", out)
