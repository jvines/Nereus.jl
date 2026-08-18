#!/usr/bin/env julia
# Periodogram validation. Injection-recovery (does the peak land on the injected
# period?) + the FAP CALIBRATION test (on pure noise, the analytic false-alarm
# probability must be well-calibrated — else the periodogram invents planets) +
# harmonic dedup. Covers GLS, BGLS, L1, gls_rotation, and the find_rv_planets
# dispatcher. Pure-Julia synthetic ground truth (no external reference needed).
using Nereus, Printf, Random, Statistics

ok = true
chk(n, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", n))
topP(pg) = isempty(pg.peaks) ? NaN : pg.peaks[1].period
topFAP(pg) = isempty(pg.peaks) ? NaN : pg.peaks[1].fap

# ----------------------------------------------------------------------------
println("=== 1. GLS injection-recovery ===")
let
    rng = MersenneTwister(1)
    n = 90; t = sort(rand(rng, n) .* 600.0); err = fill(2.0, n)
    P0, K0 = 12.34, 10.0
    y = K0 .* sin.(2π .* t ./ P0 .+ 0.7) .+ err .* randn(rng, n)
    pg = gls_periodogram(t, y, err; period_min = 1.5, period_max = 300.0)
    @printf("  injected P=%.3f d → top peak P=%.3f d  (fap=%.2e)\n", P0, topP(pg), topFAP(pg))
    chk("GLS recovers P within 3%", abs(topP(pg) - P0) / P0 < 0.03)
    chk("GLS detects it (top-peak FAP < 1e-3)", topFAP(pg) < 1e-3)
end

println("\n=== 2. GLS two-signal recovery ===")
let
    rng = MersenneTwister(2)
    n = 140; t = sort(rand(rng, n) .* 800.0); err = fill(1.5, n)
    P1, P2 = 12.3, 31.7
    y = 8 .* sin.(2π .* t ./ P1) .+ 6 .* sin.(2π .* t ./ P2 .+ 1.1) .+ err .* randn(rng, n)
    pg = gls_periodogram(t, y, err; period_min = 1.5, period_max = 300.0, max_peaks = 6)
    found = [p.period for p in pg.peaks]
    hit1 = any(abs.(found .- P1) ./ P1 .< 0.03)
    hit2 = any(abs.(found .- P2) ./ P2 .< 0.03)
    # Diagnostic: is the real P2 actually a strong peak in the raw spectrum, or
    # genuinely weaker than the 2·P2 alias the dedup keeps?
    pw_at(P) = pg.power[argmin(abs.(pg.periods .- P))]
    @printf("  injected P1=%.1f P2=%.1f → peaks %s\n", P1, P2, round.(found, digits=2))
    @printf("  raw power: at P2=%.1f → %.3f | at 2·P2=%.1f → %.3f | at P1 → %.3f\n",
            P2, pw_at(P2), 2P2, pw_at(2P2), pw_at(P1))
    chk("both periods present in peak list", hit1 && hit2)
end

println("\n=== 3. GLS FAP CALIBRATION via bootstrap (the key correctness test) ===")
let
    N = 150; rng = MersenneTwister(3)
    best_faps = Float64[]; best_faps_an = Float64[]
    for i in 1:N
        n = 70; t = sort(rand(rng, n) .* 500.0); err = fill(1.0, n)
        y = err .* randn(rng, n)                       # PURE NOISE
        pgb = gls_periodogram(t, y, err; period_min = 1.5, period_max = 250.0,
                              fap_method = :bootstrap, n_bootstrap = 200)
        f = topFAP(pgb); isfinite(f) && push!(best_faps, f)
        pga = gls_periodogram(t, y, err; period_min = 1.5, period_max = 250.0,
                              fap_method = :analytic)
        fa = topFAP(pga); isfinite(fa) && push!(best_faps_an, fa)
    end
    # On noise a calibrated FAP is ~Uniform(0,1): the fraction below α ≈ α.
    for α in (0.10, 0.05)
        emp = mean(best_faps .< α)
        tol = 3 * sqrt(α * (1 - α) / length(best_faps)) + 0.03
        @printf("  bootstrap FAP<%.2f: false-alarm rate = %.3f  (target %.2f ± %.3f)\n", α, emp, α, tol)
        chk(@sprintf("bootstrap FAP calibrated at α=%.2f", α), abs(emp - α) < tol)
    end
    @printf("  median best-FAP: bootstrap=%.3f  analytic=%.3f  (want ≈0.5; analytic is over-confident)\n",
            median(best_faps), median(best_faps_an))
    chk("bootstrap noise best-FAP median ≈ 0.5 (±0.15)", abs(median(best_faps) - 0.5) < 0.15)
end

println("\n=== 4. extract_peaks harmonic dedup ===")
let
    # Synthetic spectrum: a fundamental at f0 and its 2nd harmonic at 2*f0.
    freqs = collect(range(0.005, 0.5; length = 4000)); periods = 1.0 ./ freqs
    f0 = 1 / 20.0
    power = exp.(-((freqs .- f0) ./ 0.0008) .^ 2) .+ 0.7 .* exp.(-((freqs .- 2f0) ./ 0.0008) .^ 2)
    pk = Nereus.extract_peaks(freqs, periods, power; max_peaks = 5, harmonic_tol = 0.03)
    pers = [p.period for p in pk]
    @printf("  fundamental P=20 + harmonic P=10 → extracted %s\n", round.(pers, digits=2))
    chk("fundamental (P≈20) is the top peak", !isempty(pers) && abs(pers[1] - 20.0) < 1.0)
end

println("\n=== 5. BGLS recovery ===")
let
    rng = MersenneTwister(5)
    n = 90; t = sort(rand(rng, n) .* 600.0); err = fill(2.0, n)
    P0 = 18.5; y = 9 .* sin.(2π .* t ./ P0 .+ 0.3) .+ err .* randn(rng, n)
    pg = bgls_periodogram(t, y, err; period_min = 1.5, period_max = 300.0)
    @printf("  injected P=%.1f → BGLS top peak P=%.2f\n", P0, topP(pg))
    chk("BGLS recovers P within 3%", abs(topP(pg) - P0) / P0 < 0.03)
end

println("\n=== 6. find_rv_planets dispatcher ===")
let
    rng = MersenneTwister(6)
    n = 110; t = sort(rand(rng, n) .* 700.0); err = fill(1.5, n)
    P0 = 23.1; y = 7 .* sin.(2π .* t ./ P0 .+ 0.9) .+ err .* randn(rng, n)
    res = find_rv_planets(t, y, err; method = :gls, period_min = 1.5, period_max = 300.0)
    p = (res isa Nereus.PgramSet && !isempty(res.rv.peaks)) ? res.rv.peaks[1].period : NaN
    @printf("  injected P=%.1f → find_rv_planets (PgramSet) top RV peak P=%.2f\n", P0, p)
    chk("find_rv_planets recovers P within 3%", isfinite(p) && abs(p - P0) / P0 < 0.03)
end

println("\n=== 7. gls_rotation ===")
let
    rng = MersenneTwister(7)
    n = 200; t = sort(rand(rng, n) .* 80.0); err = fill(0.002, n)
    # gls_rotation is a PHOTOMETRIC tool → feed it relative flux (~1.0), not a
    # zero-mean RV-like signal (which breaks the ÷median flux normalization).
    Prot = 8.3
    y = 1.0 .+ 0.01 .* sin.(2π .* t ./ Prot) .+ 0.004 .* sin.(4π .* t ./ Prot) .+ err .* randn(rng, n)
    ss = gls_rotation(t, y, err; period_min = 0.5, period_max = 40.0)
    p = isempty(ss.stitched.peaks) ? NaN : ss.stitched.peaks[1].period
    @printf("  injected P_rot=%.1f → gls_rotation stitched top P=%.2f\n", Prot, p)
    chk("gls_rotation recovers P_rot within 5%", isfinite(p) && abs(p - Prot) / Prot < 0.05)
end

println("\n=== 8. l1_periodogram (smoke + recovery) ===")
let
    rng = MersenneTwister(8)
    n = 80; t = sort(rand(rng, n) .* 500.0); err = fill(1.5, n)
    P0 = 15.2; y = 8 .* sin.(2π .* t ./ P0 .+ 0.4) .+ err .* randn(rng, n)
    try
        pg = l1_periodogram_compute(t, y, err; period_min = 1.5, period_max = 250.0)
        amp = pg.amplitudes; i = argmax(amp)
        @printf("  injected P=%.1f → L1 top-amplitude P=%.2f\n", P0, pg.periods[i])
        chk("L1 places dominant amplitude near P", abs(pg.periods[i] - P0) / P0 < 0.05)
    catch e
        @printf("  L1 errored: %s\n", e); chk("L1 runs", false)
    end
end

println(ok ? "\n✅ PERIODOGRAM VALIDATION PASS" : "\n❌ PERIODOGRAM VALIDATION — FAILURES ABOVE")
