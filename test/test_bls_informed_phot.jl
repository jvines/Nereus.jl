# BLS / informed-photometry unit tests.
#
# Covers three features that had zero automated coverage, each with a
# recorded real-data failure behind it:
#   1. `_duty_grid` — ρ⋆-scaled duration grid (q ∝ P^(-2/3)).  A FIXED
#      fractional grid put the P≈0.79 d WASP-47 e below the noise floor.
#   2. `flank_frac` — local ("GBLS") baseline; depth measured against the
#      bins flanking each box, for a wandering/under-detrended baseline.
#   3. `_box_least_squares` itself — the kernel behind `find_transits` and
#      the trans-dim informed birth proposal.
#
# All light curves are generated here (box transits, no external files).
# The headline assertions use NOISELESS flux with a nominal `flux_err`, so
# the BLS SNRs are deterministic — no RNG-stream dependence.

using Nereus
using Random
using Test

const _BLS = Nereus._box_least_squares
const _DUTY = Nereus._duty_grid

# Box transit LC. `dur` in days; `sigma` only sets `flux_err` unless
# `noise=true`, in which case it is also injected.
function _mk_box_lc(; P, depth, dur, t0, base = 30.0, cad = 10 / 1440,
                      sigma = 5e-4, noise = false, seed = 1,
                      wander_amp = 0.0, wander_per = 4.9, wander_phase = 0.7)
    t = collect(0.0:cad:base)
    f = ones(length(t))
    if noise
        f .+= sigma .* randn(MersenneTwister(seed), length(t))
    end
    @inbounds for i in eachindex(t)
        ph = mod(t[i] - t0 + P / 2, P) - P / 2
        abs(ph) < dur / 2 && (f[i] -= depth)
        wander_amp > 0 &&
            (f[i] += wander_amp * sin(2π * t[i] / wander_per + wander_phase))
    end
    return t, f, fill(sigma, length(t))
end

# Central circular T14/P for a star of density `rho` [ρ_sun] at period P.
_q_true(rho, P) = asin(min(1.0, 1.0 / Nereus.rho_s_to_a_Rs(rho, P))) / π

@testset "BLS / informed photometry" begin

# =====================================================================
# 1. ρ⋆-scaled duration grid
# =====================================================================
@testset "_duty_grid: q ∝ P^(-2/3)" begin
    # Analytic scaling (a/R⋆ ≫ 1, so asin x ≈ x): doubling P shrinks the
    # duty cycle by 2^(-2/3); an 8× denser star shrinks it by 8^(-1/3).
    q10 = _DUTY(1.0, 10.0, [1.0])[1]
    q20 = _DUTY(1.0, 20.0, [1.0])[1]
    @test q20 / q10 ≈ 2.0^(-2 / 3) rtol = 1e-3
    @test _DUTY(8.0, 5.0, [1.0])[1] / _DUTY(1.0, 5.0, [1.0])[1] ≈
          8.0^(-1 / 3) rtol = 1e-3

    # The physical point the fixed grid gets wrong: for a Sun-like host a
    # USP has a ~10% duty cycle and a 9 d planet <2%. The DEFAULT fixed
    # grid [0.01,0.02,0.04,0.08] tops out at 0.08 (too narrow at 0.79 d)
    # and bottoms out at 0.01 (5× too wide at 9 d).
    @test _DUTY(1.0, 0.7896, [1.0])[1] > 0.08
    @test _DUTY(1.0, 9.0, [1.0])[1] < 0.02

    # Grid brackets the central value and is ordered by multiplier.
    g = _DUTY(1.0, 0.45, [0.3, 0.5, 0.7, 1.0, 1.3])
    @test issorted(g)
    @test g[4] ≈ _q_true(1.0, 0.45) rtol = 1e-6      # mult 1.0 == central q
    @test g[1] < g[4] < g[5]

    # Clamps: [0.002, 0.30].
    @test all(_DUTY(0.01, 0.3, [1.0, 1.3]) .== 0.30)   # a/R⋆ ≤ 1 → saturates
    @test all(_DUTY(10.0, 200.0, [0.3, 1.0]) .== 0.002)
end

@testset "scaled duration grid recovers a USP the fixed grid clips" begin
    # A short-period, large-duty-cycle transit: P = 0.45 d around a
    # Sun-density star ⇒ a/R⋆ = 2.47, q = 0.133, T14 = 1.43 h. The default
    # fixed grid's widest box (0.08 P) is 1.7× too narrow.
    P, rho = 0.45, 1.0
    q = _q_true(rho, P)                       # 0.1326
    dur = q * P                               # 0.0597 d
    @test q > 0.08                            # the fixed grid CANNOT match it
    t, f, e = _mk_box_lc(; P = P, depth = 1e-3, dur = dur, t0 = 0.13)

    fixed  = _BLS(t, f, e, [P]; n_phase_bins = 100, n_peaks = 1)
    scaled = _BLS(t, f, e, [P]; n_phase_bins = 100, n_peaks = 1, rho_star = rho)

    # (a) The recovered DURATION. The fixed grid rails at its widest box;
    #     the scaled grid lands on the true duration.
    @test fixed[5][1]  ≈ 0.08 * P  rtol = 1e-6      # railed at the grid cap
    @test fixed[5][1]  < 0.7 * dur
    @test scaled[5][1] ≈ dur       rtol = 0.05      # true duration recovered

    # (b) The SNR the box-width mismatch costs (noiseless ⇒ deterministic).
    #     Measured 31.0 (fixed) vs 37.3 (scaled) = 1.20×.
    @test scaled[4][1] > 1.15 * fixed[4][1]

    # (c) Both still find the depth and epoch — the loss is sensitivity,
    #     not bias.
    @test scaled[2][1] ≈ 1e-3 rtol = 0.20
    @test scaled[3][1] ≈ mod(0.13, P) atol = P / 100   # t0 = center phase × P
end

@testset "scale_durations knob reaches the kernel (find_transits)" begin
    P, rho = 0.45, 1.0
    dur = _q_true(rho, P) * P
    t, f, e = _mk_box_lc(; P = P, depth = 1.5e-3, dur = dur, t0 = 0.13)
    common = (; detrend = :none, period_min = 0.3, period_max = 1.5,
                n_periods = 2000, max_candidates = 3, R_s = 1.0, M_s = 1.0)
    r_fix = find_transits(t, f, e; scale_durations = false, common...)
    r_scl = find_transits(t, f, e; scale_durations = true,  common...)
    @test r_fix[1][1] ≈ P rtol = 0.01
    @test r_scl[1][1] ≈ P rtol = 0.01
    # Same peak, better matched box: higher SNR and the true duration.
    @test r_scl[5][1] > 1.15 * r_fix[5][1]
    @test r_scl[4][1] ≈ dur rtol = 0.10
    @test r_fix[4][1] < 0.75 * dur
end

# =====================================================================
# 2. flank_frac / local-baseline depth
# =====================================================================
@testset "flank_frac = 0 is byte-identical to the global default" begin
    t, f, e = _mk_box_lc(; P = 2.5, depth = 3e-3, dur = 0.12, t0 = 0.9,
                           sigma = 3e-4, noise = true, seed = 11)
    pg = exp.(range(log(1.0), log(6.0); length = 600))
    a = _BLS(t, f, e, pg; n_phase_bins = 100, n_peaks = 4)
    b = _BLS(t, f, e, pg; n_phase_bins = 100, n_peaks = 4, flank_frac = 0.0)
    @test all(a[i] == b[i] for i in 1:5)          # exact equality, not ≈
end

@testset "local baseline survives a wandering baseline the global one loses to" begin
    # Un-detrended rotational modulation: a 4.9 d sinusoid of amplitude
    # 6e-3 — 2× the 3e-3 transit depth — on top of a P = 3.7 d transit.
    P = 3.7
    dur = _q_true(1.0, P) * P                       # 0.117 d
    t, f, e = _mk_box_lc(; P = P, depth = 3e-3, dur = dur, t0 = 1.1,
                           wander_amp = 6e-3, wander_per = 4.9)
    pg = exp.(range(log(2.0), log(6.0); length = 800))

    glob = _BLS(t, f, e, pg; n_phase_bins = 100, n_peaks = 3)
    locl = _BLS(t, f, e, pg; n_phase_bins = 100, n_peaks = 3, flank_frac = 1.0)

    # Global baseline: the top peak is the WANDER period, not the transit,
    # and its "depth" is the sinusoid amplitude, not the transit depth.
    @test glob[1][1] ≈ 4.9 rtol = 0.05
    @test !isapprox(glob[1][1], P; rtol = 0.02)
    @test glob[2][1] > 4e-3                        # inflated by the sinusoid

    # Local baseline: the top peak IS the transit, at roughly its depth.
    @test locl[1][1] ≈ P rtol = 0.01
    @test locl[2][1] ≈ 3e-3 rtol = 0.35            # flanks still tilt slightly
end

@testset "local baseline costs SNR on a flat baseline (why it is opt-in)" begin
    t, f, e = _mk_box_lc(; P = 2.5, depth = 3e-3, dur = 0.12, t0 = 0.9)
    # Pin the box width so the two paths differ ONLY in how the
    # out-of-transit level is measured.
    glob = _BLS(t, f, e, [2.5]; n_peaks = 1, durations = [0.048])
    locl = _BLS(t, f, e, [2.5]; n_peaks = 1, durations = [0.048],
                 flank_frac = 1.0)
    @test locl[4][1] < 0.9 * glob[4][1]            # less out-of-transit data
    @test locl[2][1] ≈ glob[2][1] rtol = 0.05      # …but the same depth
end

# =====================================================================
# 3. _box_least_squares core
# =====================================================================
@testset "_box_least_squares: recovery, conventions, determinism" begin
    P, depth, dur, t0, sigma = 2.5, 4e-3, 0.12, 0.9, 3e-4
    t, f, e = _mk_box_lc(; P = P, depth = depth, dur = dur, t0 = t0,
                           sigma = sigma, noise = true, seed = 11)
    pg = exp.(range(log(1.0), log(6.0); length = 1000))
    r = _BLS(t, f, e, pg; n_phase_bins = 100, n_peaks = 4,
              durations = [0.02, 0.04, 0.06, 0.08])

    # Top peak is the true period; t0 is a PHASE×P in [0,P), not an epoch.
    @test r[1][1] ≈ P rtol = 0.005                 # top peak = true period
    @test r[3][1] ≈ mod(t0, P) atol = P / 100
    @test 0.0 <= r[3][1] < P
    # Depth/duration land on the best of the discrete box grid, so the
    # tolerances must straddle the two boxes that bracket the truth
    # (0.04 P = 0.10 d and 0.06 P = 0.15 d for a true 0.12 d transit) —
    # which of the two wins is noise-realization dependent.
    @test r[2][1] ≈ depth rtol = 0.25              # bin quantization dilutes
    @test r[5][1] ≈ dur rtol = 0.30
    @test r[4][1] > 50                             # SNR at this depth
    @test issorted(r[4]; rev = true)               # peaks ordered by SNR
    @test length(r[1]) == 4 && all(length(x) == 4 for x in r)

    # Recovered depth is LINEAR in injected depth (same box ⇒ same
    # dilution factor), i.e. the estimator is not depth-dependent.
    ratios = Float64[]
    for d in (1e-3, 4e-3, 1e-2)
        _, fd, ed = _mk_box_lc(; P = P, depth = d, dur = dur, t0 = t0,
                                 sigma = sigma)
        rd = _BLS(t, fd, ed, [P]; n_peaks = 1, durations = [0.048])
        push!(ratios, rd[2][1] / d)
    end
    @test all(x -> isapprox(x, ratios[1]; rtol = 0.02), ratios)

    # Threaded and serial paths are numerically identical.
    rs = _BLS(t, f, e, pg; n_phase_bins = 100, n_peaks = 4, threaded = false)
    rt = _BLS(t, f, e, pg; n_phase_bins = 100, n_peaks = 4, threaded = true)
    @test all(rs[i] == rt[i] for i in 1:5)

    # Peak guard band: returned peaks are ≥ min_peak_spacing grid samples apart.
    r2 = _BLS(t, f, e, pg; n_peaks = 6, min_peak_spacing = 50)
    idx = [findfirst(==(p), pg) for p in r2[1]]
    @test all(abs(i - j) >= 50 for i in idx, j in idx if i != j)

    # Empty inputs return five empty vectors rather than throwing.
    @test all(isempty, _BLS(Float64[], Float64[], Float64[], pg))
    @test all(isempty, _BLS(t, f, e, Float64[]))

    # A transit-free light curve stays below the find_transits default
    # score_threshold of 7.
    _, fflat, eflat = _mk_box_lc(; P = P, depth = 0.0, dur = dur, t0 = t0,
                                   sigma = sigma, noise = true, seed = 5)
    rflat = _BLS(t, fflat, eflat, pg; n_peaks = 3)
    @test maximum(rflat[4]) < 7.0
end

end # testset
