#!/usr/bin/env julia
# Verify sigma_clip: synthetic LC with slow variability + a deep transit + extreme
# point-outliers. Sigma-clip must (1) flag the outliers, (2) NOT flag the protected
# transit, (3) leave the bulk alone. Also exercises clean_lightcurve(sigma_clip=...)
# and sigma_clip_lightcurve, and the never-bin / trim semantics.

using Nereus
using Statistics, Random, Printf

rng = MersenneTwister(1)
n = 6000
t = collect(range(0.0, 12.0, length=n))            # 12 days
trend = 1.0 .+ 0.01 .* sin.(2π .* t ./ 3.0)         # 3-day stellar variability
flux  = trend .+ 0.001 .* randn(rng, n)             # 1000 ppm white noise
ferr  = fill(0.001, n)

# A deep (2%) transit over a ~3.6 h window — 20σ, would look like outliers
intransit = (t .>= 4.0) .& (t .<= 4.15)
flux[intransit] .-= 0.02

# 15 extreme outliers scattered in normal data (cosmic rays / hot pixels)
out_idx = unique(rand(rng, 1:n, 20))[1:15]
true_out = falses(n); true_out[out_idx] .= true
for i in out_idx; flux[i] += (rand(rng) < 0.5 ? -1 : 1) * 0.3; end   # ±0.3 ≈ 300σ

println("=== sigma_clip_mask, protecting the transit ===")
mask = sigma_clip_mask(t, flux; sigma=5.0, window=101, protect_mask=intransit)
tp   = sum(mask .& true_out)                          # outliers caught
tr   = sum(mask .& intransit)                         # transit points wrongly clipped
fp   = sum(mask .& .!true_out .& .!intransit)         # bulk wrongly clipped
@printf("injected=%d  flagged=%d  caught=%d/%d  transit-clipped=%d (want 0)  bulk-clipped=%d\n",
        sum(true_out), sum(mask), tp, sum(true_out), tr, fp)

println("\n=== protection mechanism: protect the outliers themselves → none flagged ===")
# Direct test of protect_mask: the 15 points ARE outlier-magnitude, but protecting
# them must prevent any from being flagged.
mask_prot = sigma_clip_mask(t, flux; sigma=5.0, window=101, protect_mask=true_out)
@printf("outliers flagged when protected = %d (want 0);  unprotected = %d (want 15)\n",
        sum(mask_prot .& true_out), sum(mask .& true_out))

println("\n=== sigma_clip_lightcurve trims them out entirely ===")
lc = LightCurve(t, flux, ferr, ones(Int, n), nothing, nothing, Vector{Bool}(intransit), nothing)
lc2 = sigma_clip_lightcurve(lc; sigma=5.0, window=101, protect_transit=true)
@printf("cadences: %d -> %d  (removed %d; injected outliers %d)\n",
        length(lc.t), length(lc2.t), length(lc.t)-length(lc2.t), sum(true_out))
@printf("any injected outlier survived? %s\n", any(in.(t[out_idx], Ref(lc2.t))) ? "YES (bad)" : "no")

println("\n=== find_transits(sigma_clip=...) drops outliers before the search ===")
res = find_transits(t, flux, ferr; method=:bls, detrend=:none, sigma_clip=5.0,
                    period_min=1.0, period_max=6.0, n_periods=2000)
@printf("find_transits kept %d / %d cadences (dropped %d; outliers %d)\n",
        sum(res.kept), length(t), length(t)-sum(res.kept), sum(true_out))

ok = (tp == sum(true_out)) && (tr == 0) && (fp == 0) &&
     (sum(mask_prot .& true_out) == 0) &&
     (length(t) - sum(res.kept) == sum(true_out))
println(ok ? "\n✅ sigma_clip works: outliers caught, transit protected, protect_mask honored, find_transits wired" :
             "\n⚠ check the counts above")
