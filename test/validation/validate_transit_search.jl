#!/usr/bin/env julia
# Transit-search (find_transits) injection-recovery. Inject a transiting planet
# into a quiet star with stellar variability, run the FULL pipeline
# (detrend -> BLS), and assert it recovers the injected period / epoch / depth.
# Runs the notch detrender too, so it exercises the committed two-pass fix end to
# end (detect -> mask -> search). Renders the phase-fold for eyeballing.
using Nereus, Random, Printf, Statistics

out = joinpath(@__DIR__, "..", "..", "results", "transit_search_check"); mkpath(out)
rng = MersenneTwister(20260629)

# --- quiet star + ONE transiting planet: 27 d, 10-min cadence ---------------
P0, T0, depth, dur = 4.0567, 2.0, 0.010, 0.13          # injected ephemeris
t = collect(0.0:(10/1440):27.0); n = length(t)
ph = mod.(t .- T0 .+ 0.5P0, P0) ./ P0 .- 0.5
intr = abs.(ph) .< (dur / 2) / P0
σ = 0.0007
flux = 1.0 .+ 0.008 .* sin.(2π .* t ./ 19.0) .+ 0.0015 .* (t ./ 27) .+   # activity + drift
       intr .* (-depth) .+ σ .* randn(rng, n)
t = Vector{Float64}(t); flux = Vector{Float64}(flux); flux_err = fill(σ, n)
@printf("LC: %d cadences (10-min, 27 d) | injected P=%.4f d, T0=%.3f, depth=%.1f%%, σ=%d ppm\n",
        n, P0, T0, depth * 100, round(Int, σ * 1e6))

ok = true
chk(nm, c) = (global ok; ok &= c; @printf("    [%s] %s\n", c ? "PASS" : "FAIL", nm))

# phase offset of a recovered epoch vs the truth, folded to [-0.5,0.5]·P [days]
ephem_off(t0rec) = (d = mod(t0rec - T0 + 0.5P0, P0) - 0.5P0; abs(d))

function search(label, method, detrend, kw)
    res = find_transits(t, flux, flux_err; method = method, detrend = detrend,
                        detrend_kwargs = kw, period_min = 1.0, period_max = 12.0)
    isempty(res.periods) && (chk("$label returns ≥1 candidate", false); return)
    Pr, t0r, dr, sr = res.periods[1], res.t0s[1], res.depths[1], res.snr[1]
    @printf("  %-12s: top P=%.4f d (truth %.4f) | epoch off=%.3f d | depth=%.4f (truth %.4f) | SNR=%.1f\n",
            label, Pr, P0, ephem_off(t0r), dr, depth, sr)
    chk("$label recovers P within 1%",              abs(Pr - P0) / P0 < 0.01)
    chk("$label recovers epoch (< 0.1 d)",          ephem_off(t0r) < 0.1)
    chk("$label recovers depth (within 30%)",       abs(dr - depth) / depth < 0.30)
    chk("$label significant (SNR > 7)",             sr > 7)
    return res
end

println("\n=== BLS + notch (committed two-pass detrender) ===")
res_n = search("bls+notch", :bls, :notch, (; window = 1.0))
println("\n=== BLS + savgol (cross-check) ===")
res_s = search("bls+savgol", :bls, :savgol, (; window_length = 301, polyorder = 2))
println("\n=== TLS + notch (alternative search method) ===")
res_t = search("tls+notch", :tls, :notch, (; window = 1.0))

# phase-fold plot from the notch run
if res_n !== nothing
    try
        Nereus.plot_transit_phasefold(t, res_n; output = out)
    catch e; @warn "plot_transit_phasefold failed" exception = e; end
end

println(ok ? "\n✅ TRANSIT SEARCH VALIDATION PASS" :
             "\n❌ TRANSIT SEARCH — FAILURES ABOVE")
@printf("plots: %s\n", out)
