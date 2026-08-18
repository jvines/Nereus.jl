#!/usr/bin/env julia
# Multi-sector light-curve handling + persistence. Build a 3-sector LightCurve
# (inter-sector gaps + per-sector flux offsets, the real TESS situation), run
# clean_lightcurve, and assert: (a) each sector is detrended to a flat ≈1
# baseline despite its offset (per-segment leveling), (b) the common transit
# survives in every sector, (c) the notch is_transit flags land on the transits,
# and (d) save_lightcurve -> load_lightcurve round-trips every array exactly.
using Nereus, Random, Printf, Statistics

out = joinpath(@__DIR__, "..", "..", "results", "lc_multisector_check"); mkpath(out)
rng = MersenneTwister(20260629)

# --- 4 sectors, 9 d each, ~5 d gaps, distinct flux offsets -------------------
# 4 sectors exercises the paper-readable paging: ≤3 per figure ⇒ 2 balanced
# pages of 2 sectors each (not 3+1).
P0, T0, depth, dur = 4.0, 2.0, 0.010, 0.13
σ = 0.0008
offsets = [1.000, 1.020, 0.985, 1.010]    # per-sector instrumental level
starts  = [0.0, 14.0, 28.0, 42.0]         # sector start times (gaps between)
seclcs = LightCurve[]
for (s, (off, t0s)) in enumerate(zip(offsets, starts))
    ts = collect(t0s:(10/1440):(t0s + 9.0))
    ph = mod.(ts .- T0 .+ 0.5P0, P0) ./ P0 .- 0.5
    intr = abs.(ph) .< (dur / 2) / P0
    fl = off .+ 0.006 .* sin.(2π .* ts ./ 13.0) .+ intr .* (-depth .* off) .+ σ .* randn(rng, length(ts))
    push!(seclcs, LightCurve(ts, fl, fill(σ, length(ts)), fill(s, length(ts))))
end
lc = LightCurve(seclcs)                    # concat + time-sort, sector_id = 1,2,3
nsec = length(unique(lc.sector_id))
@printf("multi-sector LC: %d cadences, %d sectors, offsets %s\n",
        length(lc.t), nsec, offsets)

ok = true
chk(nm, c) = (global ok; ok &= c; @printf("    [%s] %s\n", c ? "PASS" : "FAIL", nm))

# transit cadence mask on the stitched LC (truth, for assertions)
ph_all = mod.(lc.t .- T0 .+ 0.5P0, P0) ./ P0 .- 0.5
intr_all = abs.(ph_all) .< (dur / 2) / P0

# --- clean (detrend into a LightCurve) --------------------------------------
cl = clean_lightcurve(lc; method = :notch, window = 1.0)
chk("clean_lightcurve returns detrended flux", cl.flux_detrended !== nothing)
chk("method recorded as :notch",               cl.method === :notch)

# (a) each sector flattened to ≈1 despite its offset
for s in 1:nsec
    m = (lc.sector_id .== s) .& .!intr_all
    sd = std(cl.flux_detrended[m]); md = median(cl.flux_detrended[m])
    @printf("  sector %d: OOT median=%.4f std=%.4f (offset was %.3f)\n", s, md, sd, offsets[s])
    chk("sector $s flattened to ≈1 (|med-1|<0.2%, std<2σ)", abs(md - 1) < 0.002 && sd < 2σ)
end
# (b) transit preserved across all sectors
rec = median(cl.flux_detrended[.!intr_all]) - median(cl.flux_detrended[intr_all])
@printf("  global transit depth recovered=%.4f (truth %.4f)\n", rec, depth)
chk("transit preserved across sectors (depth within 20%)", abs(rec - depth) / depth < 0.20)
# (c) is_transit flags land on the transits
det = cl.is_transit
purity = count(det .& .!intr_all) / max(count(det), 1)
@printf("  is_transit: %d flagged | %.0f%% on real transits | hit-rate %.2f\n",
        count(det), 100 * (1 - purity), count(det .& intr_all) / max(count(intr_all), 1))
chk("is_transit flags overlap the transits", any(det .& intr_all))

# --- (d) save / load round-trip ---------------------------------------------
path = joinpath(out, "multisector.nc")
save_lightcurve(path, lc.t, lc.flux, lc.flux_err;
                flux_detrended = cl.flux_detrended, trend = cl.trend,
                sector_id = lc.sector_id)
ld = load_lightcurve(path)
rt(a, b) = length(a) == length(b) && all(isapprox.(a, b; atol = 1e-9))
@printf("  round-trip: t=%s flux=%s err=%s detrended=%s sector=%s\n",
        rt(ld[:t], lc.t), rt(ld[:flux], lc.flux), rt(ld[:flux_err], lc.flux_err),
        rt(ld[:flux_detrended], cl.flux_detrended), rt(Int.(ld[:sector_id]), lc.sector_id))
chk("save/load round-trips raw arrays exactly", rt(ld[:t], lc.t) && rt(ld[:flux], lc.flux) && rt(ld[:flux_err], lc.flux_err))
chk("save/load round-trips detrended + sector_id", rt(ld[:flux_detrended], cl.flux_detrended) && rt(Int.(ld[:sector_id]), lc.sector_id))

# --- plot the multi-sector detrend (paged: 4 sectors → 2 figures of 2) -------
res = detrend_notch(lc.t, lc.flux, lc.flux_err; sector_id = lc.sector_id, window = 1.0)
p1 = joinpath(out, "detrend", "multisector_p1.png")
p2 = joinpath(out, "detrend", "multisector_p2.png")
p3 = joinpath(out, "detrend", "multisector_p3.png")
rm(p1, force = true); rm(p2, force = true); rm(p3, force = true)
try
    Nereus.plot_detrending(lc.t, lc.flux, lc.flux_err, res; output = out, name = "multisector")
catch e; @warn "plot_detrending failed" exception = e; end
chk("4 sectors → 2 balanced pages of 2 (p1+p2 exist, no p3)",
    isfile(p1) && isfile(p2) && !isfile(p3))

println(ok ? "\n✅ MULTI-SECTOR LC VALIDATION PASS" : "\n❌ MULTI-SECTOR LC — FAILURES ABOVE")
@printf("plots: %s , %s\n", p1, p2)
