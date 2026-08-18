# Photometry-side helpers for JointInformedBirth.
#
# Building blocks (kept independent of any BirthStrategy so they can be
# tested in isolation):
#   * _box_least_squares — binned BLS (Hippke & Heller 2019 §2-style)
#     returning (period, depth, t0, score) for top peaks
#   * _mass_radius_chenkipping — Chen & Kipping 2017 piecewise M-R relation
#   * _K_from_mass — Keplerian RV semi-amplitude from M_p, M_*, P (circular)
#   * _radius_from_depth — R_p[R_E] from BLS depth + R_*[R_sun]
#   * _compute_phot_residuals — photometry minus offsets minus active
#     transit models (uses transit.jl machinery for consistency with the
#     Nereus likelihood)
#
# All functions take Float64 inputs since they live entirely on the
# proposal side and don't need to AD-thread.

using Statistics: mean, median

# =====================================================================
# Mass–Radius relation (Chen & Kipping 2017, Forecaster)
# =====================================================================

"""
    _mass_radius_chenkipping(R_p_earth) -> M_p_earth

Predict planet mass [M_⊕] from radius [R_⊕] using the Chen & Kipping
(2017) "Forecaster" piecewise power-law relation. Branches:
  * Terran      (R < 1.23 R_⊕)         — rocky
  * Neptunian   (1.23 ≤ R < 11.1 R_⊕)
  * Jovian      (R ≥ 11.1 R_⊕)         — saturates near Jupiter mass
                                          since R is mass-insensitive
  * Stellar regime is not handled here (planet-mass priors cover it
    well before this clause kicks in).

Returns Jupiter mass (318 M_⊕) for any input above the Jovian threshold;
the actual M-R relation in that regime is multi-valued (R is roughly
mass-independent for inflated hot Jupiters and brown dwarfs), so a
single point estimate is the best we can do for a proposal density.
"""
function _mass_radius_chenkipping(R_p_earth::Float64)
    R_p_earth <= 0 && return 0.0
    if R_p_earth < 1.23
        return (R_p_earth / 1.008) ^ (1.0 / 0.279)
    elseif R_p_earth < 11.1
        return (R_p_earth / 0.808) ^ (1.0 / 0.589)
    else
        # Jovian regime: R nearly mass-independent. Default to Jupiter mass.
        return 318.0
    end
end

# =====================================================================
# Keplerian K from mass
# =====================================================================

"""
    _K_from_mass(M_p_earth, P_days, M_s_solar) -> K [m/s]

Circular-orbit, edge-on (sin i = 1) RV semi-amplitude. Useful as a
default K_est when InformedBirth has only photometric depth to work
from. The chain refines K via within-model MCMC; this estimator just
needs to be in the right order of magnitude so the proposed K isn't
absurd.

Convention: K[m/s] = 0.6391 · P[d]^(-1/3) · M_p[M_⊕] · M_s[M_⊙]^(-2/3)
which matches Lovis & Fischer (2010) eq 14 with sin i=1, e=0.
"""
function _K_from_mass(M_p_earth::Float64, P_days::Float64, M_s_solar::Float64)
    (P_days <= 0 || M_s_solar <= 0) && return 0.0
    return 0.6391 * P_days ^ (-1.0/3.0) * M_p_earth * M_s_solar ^ (-2.0/3.0)
end

"""
    _radius_from_depth(depth, R_s_solar) -> R_p [R_⊕]

Inverse of depth ≈ (R_p/R_*)². Returns R_p in Earth radii using the
conversion 1 R_⊙ = 109.2 R_⊕ (Allen's Astrophysical Quantities).
"""
function _radius_from_depth(depth::Float64, R_s_solar::Float64)
    depth <= 0 && return 0.0
    return R_s_solar * sqrt(depth) * 109.2
end

# --- Public aliases for the photometric / mass-radius helpers --------
# Internal call sites (`InformedBirth`, `find_transits`) still use the
# underscore names; these are thin wrappers that take `Real` so callers
# don't have to convert to Float64 by hand.

"""
    radius_from_depth(depth, R_s_solar) -> R_p [R_⊕]

Transit depth → planet radius assuming `depth = (R_p / R_*)²`.
"""
radius_from_depth(depth::Real, R_s_solar::Real) =
    _radius_from_depth(Float64(depth), Float64(R_s_solar))

"""
    chen_kipping_mass(R_p_earth) -> M_p [M_⊕]

Chen & Kipping (2017) deterministic mass-radius relation. Three
piecewise power laws (Earth-like / sub-Neptune / Jovian) with the
Jovian regime returning a constant Jupiter mass (318 M_⊕) because R is
mass-independent for inflated hot Jupiters.
"""
chen_kipping_mass(R_p_earth::Real) =
    _mass_radius_chenkipping(Float64(R_p_earth))

"""
    K_estimate(M_p_earth, P_days, M_s_solar) -> K [m/s]

Circular, edge-on (`sin i = 1`) RV semi-amplitude (Lovis & Fischer 2010
eq. 14 with e=0).
"""
K_estimate(M_p_earth::Real, P_days::Real, M_s_solar::Real) =
    _K_from_mass(Float64(M_p_earth), Float64(P_days), Float64(M_s_solar))

# =====================================================================
# Box Least Squares (BLS) — binned, fast enough to cache every N calls
# =====================================================================

# Expected central-circular transit duty cycle (T14/P, as a fraction of the
# period) at period `P` for a star of density `rho_star` [ρ_sun]. From Kepler's
# 3rd law a/R* ∝ P^(2/3), so the duty cycle q ∝ P^(-2/3): a USP planet has a
# ~10% duty cycle while a long-period one has <1%. A FIXED fractional duration
# grid is therefore physically wrong across periods — it either misses wide USP
# transits (box too narrow → power leaks to harmonics) or wastes wide boxes on
# long periods. `_duty_grid` returns a small grid bracketing q0(P) to allow for
# impact parameter (b>0 shortens) and mild eccentricity.
@inline function _duty_grid(rho_star::Float64, P::Float64,
                            mults::Vector{Float64})
    a_Rs = rho_s_to_a_Rs(rho_star, P)
    x  = a_Rs > 1.0 ? 1.0 / a_Rs : 1.0
    q0 = asin(x) / π                         # central circular T14/P
    return clamp.(q0 .* mults, 0.002, 0.30)
end

# Score a single trial period. Factored out so the serial loop (used
# from the trans-dim birth proposals) and the threaded `find_transits`
# search share one numerically-identical kernel. `bin_w`/`bin_wf` are
# caller-owned scratch (length `n_phase_bins`) — one set per thread.
# `wf = w .* (flux .- flux_mean)` is precomputed ONCE by the caller (it is
# period-independent), so the per-cadence fold — the hot loop, run
# n_period × n_obs times — does only a multiply, a floor, and two adds.
#
# `flank_frac > 0` switches from the GLOBAL two-level model (depth measured vs
# the global mean) to a LOCAL-baseline model: for each box position the
# out-of-transit level is taken from the `fw = round(flank_frac·box_bins)` bins
# immediately flanking the box on each side, not from the whole light curve.
# This is the generalisation a user asked for — robust to a wandering/under-
# detrended baseline, the way GLS generalises Lomb-Scargle. It costs SNR on a
# genuinely flat baseline (less out-of-transit data → looser baseline), so it
# is opt-in (`flank_frac=0.0` ⇒ the byte-identical global default).
@inline function _bls_one_period(t::Vector{Float64}, w::Vector{Float64},
                                  wf::Vector{Float64}, W::Float64,
                                  P::Float64, durations::Vector{Float64},
                                  n_phase_bins::Int,
                                  bin_w::Vector{Float64}, bin_wf::Vector{Float64};
                                  flank_frac::Float64 = 0.0)
    fill!(bin_w,  0.0)
    fill!(bin_wf, 0.0)
    n_obs = length(t)
    invP  = 1.0 / P                                      # one divide per period…
    nb    = n_phase_bins
    @inbounds for i in 1:n_obs
        x  = t[i] * invP                                 # …not per cadence
        ph = x - floor(x)                                # ∈ [0,1)
        b  = min(nb, 1 + unsafe_trunc(Int, ph * nb))     # ph ≥ 0 ⇒ trunc == floor
        bin_w[b]  += w[i]
        bin_wf[b] += wf[i]
    end

    best_snr   = 0.0
    best_depth = 0.0
    best_t0    = 0.0
    best_dur   = 0.0
    for q in durations
        box_bins = max(1, round(Int, q * n_phase_bins))
        box_bins >= n_phase_bins && continue

        sum_w  = 0.0
        sum_wf = 0.0
        @inbounds for b in 1:box_bins
            sum_w  += bin_w[b]
            sum_wf += bin_wf[b]
        end

        if flank_frac <= 0.0
            # --- GLOBAL baseline (default, unchanged) ---
            @inbounds for shift in 0:(n_phase_bins-1)
                if sum_w > 0 && (W - sum_w) > 0
                    μ_in  = sum_wf / sum_w
                    depth = -μ_in                        # positive depth = dip
                    se    = sqrt(1.0/sum_w + 1.0/(W - sum_w))
                    snr   = depth / se
                    if snr > best_snr
                        best_snr   = snr
                        best_depth = depth
                        best_dur   = q * P               # fractional q → days
                        # box covers 1-based bins (shift+1 … shift+box_bins);
                        # its center phase is (shift + box_bins/2)/N.
                        center_bin = mod(shift + box_bins/2, n_phase_bins)
                        best_t0    = (center_bin / n_phase_bins) * P
                    end
                end
                drop = mod(shift, n_phase_bins) + 1
                add  = mod(shift + box_bins, n_phase_bins) + 1
                sum_w  += bin_w[add]  - bin_w[drop]
                sum_wf += bin_wf[add] - bin_wf[drop]
            end
        else
            # --- LOCAL baseline: depth vs the flanking bins, not global mean ---
            fw = max(1, round(Int, flank_frac * box_bins))
            box_bins + 2 * fw >= n_phase_bins && continue
            ext_w  = 0.0
            ext_wf = 0.0
            @inbounds for b in (1 - fw):(box_bins + fw)   # box + both flanks
                bb = mod(b - 1, n_phase_bins) + 1
                ext_w  += bin_w[bb]
                ext_wf += bin_wf[bb]
            end
            @inbounds for shift in 0:(n_phase_bins-1)
                flank_w  = ext_w  - sum_w                 # flanks = ext − box
                flank_wf = ext_wf - sum_wf
                if sum_w > 0 && flank_w > 0
                    # global mean cancels (both are deviations from flux_mean):
                    depth = flank_wf / flank_w - sum_wf / sum_w
                    se    = sqrt(1.0/sum_w + 1.0/flank_w)
                    snr   = depth / se
                    if snr > best_snr
                        best_snr   = snr
                        best_depth = depth
                        best_dur   = q * P
                        center_bin = mod(shift + box_bins/2, n_phase_bins)
                        best_t0    = (center_bin / n_phase_bins) * P
                    end
                end
                # advance the box window by one bin
                bdrop = mod(shift, n_phase_bins) + 1
                badd  = mod(shift + box_bins, n_phase_bins) + 1
                sum_w  += bin_w[badd]  - bin_w[bdrop]
                sum_wf += bin_wf[badd] - bin_wf[bdrop]
                # advance the extended (box+flanks) window by one bin
                edrop = mod(shift - fw, n_phase_bins) + 1
                eadd  = mod(shift + box_bins + fw, n_phase_bins) + 1
                ext_w  += bin_w[eadd]  - bin_w[edrop]
                ext_wf += bin_wf[eadd] - bin_wf[edrop]
            end
        end
    end
    return best_snr, best_depth, best_t0, best_dur
end

"""
    _box_least_squares(t, flux, flux_err, periods;
                        durations=[0.01, 0.02, 0.04, 0.08],
                        n_phase_bins=100, n_peaks=10,
                        threaded=false, min_peak_spacing=nothing)
        -> (periods, depths, t0s, snrs, durations) for top peaks

`threaded=true` parallelises the period loop with `Threads.@threads
:static` (per-thread scratch bins) — use ONLY from `find_transits`,
never from inside a sampler's threaded region (nested `:static` is
illegal). The birth proposals call this serially (`threaded=false`).
`min_peak_spacing` overrides the index guard band used to keep adjacent
grid samples of one peak from all being returned.

Fast binned BLS. For each candidate period:
  1. Phase-fold and bin into `n_phase_bins` weighted bins
  2. For each candidate duration (given as a fraction of P):
     - Slide a box of that duration across the phase axis
     - Compute weighted depth and per-window SNR via running sum
     - Track the maximum SNR
  3. Record (period, best_depth, best_t0, best_snr)

The "binning then sliding window" trick (Hippke & Heller 2019 §2)
turns the naive O(N_period · N_obs · N_dur · N_phase) into
O(N_period · (N_obs + N_dur · N_phase)), which is what makes BLS
practical inside an MCMC proposal-cache loop.

Returns the top `n_peaks` periods by SNR.
"""
function _box_least_squares(
    t::Vector{Float64}, flux::Vector{Float64}, flux_err::Vector{Float64},
    periods::Vector{Float64};
    durations::Vector{Float64} = [0.01, 0.02, 0.04, 0.08],
    n_phase_bins::Int = 100,
    n_peaks::Int = 10,
    threaded::Bool = false,
    min_peak_spacing::Union{Int, Nothing} = nothing,
    rho_star::Union{Nothing, Float64} = nothing,
    duration_multipliers::Vector{Float64} = [0.3, 0.5, 0.7, 1.0, 1.3],
    flank_frac::Float64 = 0.0,
)
    n_obs = length(t)
    n_per = length(periods)
    n_obs > 0 && n_per > 0 ||
        return (Float64[], Float64[], Float64[], Float64[], Float64[])

    w = 1.0 ./ (flux_err .^ 2)            # inverse-variance weights
    W = sum(w)
    flux_mean = sum(w .* flux) / W
    wf = w .* (flux .- flux_mean)         # precompute once (period-independent)

    period_score = zeros(Float64, n_per)
    period_depth = zeros(Float64, n_per)
    period_t0    = zeros(Float64, n_per)
    period_dur   = zeros(Float64, n_per)   # best-fit transit duration [days]

    if threaded
        # Threaded path is ONLY for the top-level `find_transits` pre-fit
        # search (never called from inside a sampler's threaded region).
        # `:static` pins task→thread so `threadid()` is a stable index into
        # the per-thread scratch — do not relax to `:dynamic`. The serial
        # branch below is what the trans-dim birth proposals use, so adding
        # threading here would NOT nest illegally inside the sampler.
        nt  = Threads.nthreads()
        BW  = [zeros(Float64, n_phase_bins) for _ in 1:nt]
        BWF = [zeros(Float64, n_phase_bins) for _ in 1:nt]
        Threads.@threads :static for pi in 1:n_per
            tid = Threads.threadid()
            durs_pi = rho_star === nothing ? durations :
                      _duty_grid(rho_star, periods[pi], duration_multipliers)
            s, d, t0, du = _bls_one_period(t, w, wf, W,
                                            periods[pi], durs_pi, n_phase_bins,
                                            BW[tid], BWF[tid]; flank_frac=flank_frac)
            period_score[pi] = s; period_depth[pi] = d
            period_t0[pi]    = t0; period_dur[pi]   = du
        end
    else
        bin_w  = zeros(Float64, n_phase_bins)
        bin_wf = zeros(Float64, n_phase_bins)
        @inbounds for pi in 1:n_per
            durs_pi = rho_star === nothing ? durations :
                      _duty_grid(rho_star, periods[pi], duration_multipliers)
            s, d, t0, du = _bls_one_period(t, w, wf, W,
                                            periods[pi], durs_pi, n_phase_bins,
                                            bin_w, bin_wf; flank_frac=flank_frac)
            period_score[pi] = s; period_depth[pi] = d
            period_t0[pi]    = t0; period_dur[pi]   = du
        end
    end

    # Pick top peaks with a guard band so adjacent grid frequencies don't
    # all hit the same physical peak. The default scales with grid size
    # (fine for a linear grid); `find_transits` passes a frequency-peak-width
    # value for its uniform-in-frequency grid.
    min_spacing = min_peak_spacing === nothing ? max(3, div(n_per, 30)) :
                                                 max(1, min_peak_spacing)
    sorted_idx = sortperm(period_score; rev=true)
    peak_idx = Int[]
    for idx in sorted_idx
        too_close = any(abs(idx - p) < min_spacing for p in peak_idx)
        too_close && continue
        push!(peak_idx, idx)
        length(peak_idx) >= n_peaks && break
    end

    return (
        periods[peak_idx],
        period_depth[peak_idx],
        period_t0[peak_idx],
        period_score[peak_idx],
        period_dur[peak_idx],
    )
end

"""
    _refine_bls_period(t, flux, flux_err, P_coarse, t0_coarse, dur_coarse;
                        half_frac=0.006, n_fine=3000) -> (P_ref, depth, t0, dur, snr)

Refine a coarse BLS peak period to transit precision. The coarse
`_box_least_squares` grid resolves the period only to its grid spacing
(~0.2%). Over a multi-year baseline a transit needs ~`transit_dur/baseline`
(often ~1e-5) to stay phase-aligned across epochs, so 0.2% smears the
transits and a birth proposed there is rejected.

Uses a NON-BINNED, data-anchored fixed-phase box rather than the binned
sliding-box BLS — binning smears the fold and the box-width/SNR tradeoff
makes the binned peak land ~5e-4 off the true period (verified on
WASP-47). The box is anchored on a real transit time near the data start
(`t_anchor`, from the coarse P/t0); folding `(t − t_anchor)` keeps the
period lever-arm equal to the baseline (not the BJD origin), so the
in-box flux SNR de-coheres sharply off the true cross-epoch period and
the fine-grid argmax lands within ~1e-5. Fold the FULL light curve (NOT
the decimated detection LC — decimation's sparse in-transit sampling
can't resolve the period). `dur_coarse` sets the box half-width.
"""
function _refine_bls_period(t::Vector{Float64}, w::Vector{Float64},
                             wf::Vector{Float64}, W::Float64, P_coarse::Float64,
                             t0_coarse::Float64, dur_coarse::Float64;
                             half_frac::Float64 = 0.006,
                             n1::Int = 1000, n2::Int = 300)
    n = length(t)
    if !(n > 0 && P_coarse > 0 && dur_coarse > 0)
        t0_abs = (n > 0 && P_coarse > 0) ?
                 t[1] + mod(t0_coarse - t[1], P_coarse) : t0_coarse
        return (P_coarse, 0.0, t0_abs, dur_coarse, 0.0)
    end
    # Anchor the fold on a real transit near the data start (lever arm =
    # baseline, not the BJD origin). `t0_coarse` is the BLS folded transit
    # time in [0,P); shift it into the first epoch after t[1].
    t1 = t[1]
    t_anchor = t1 + mod(t0_coarse - t1, P_coarse)
    half_dur = 0.5 * dur_coarse
    # In-box flux SNR at one trial period (non-binned, fixed phase).
    snr_at = function (P)
        sw = 0.0; swf = 0.0
        @inbounds for i in 1:n
            d = t[i] - t_anchor
            r = d - P * round(d / P)        # circular distance ∈ [-P/2, P/2)
            if abs(r) < half_dur
                sw  += w[i]
                swf += wf[i]
            end
        end
        (sw > 0 && (W - sw) > 0) ? (-swf / sw) / sqrt(1.0/sw + 1.0/(W - sw)) : -Inf
    end
    # Stage 1: coarse sweep over ±half_frac (spacing < peak half-width
    # ~dur/baseline so the sharp SNR peak isn't stepped over).
    P_lo = P_coarse * (1 - half_frac); P_hi = P_coarse * (1 + half_frac)
    best_snr = -Inf; best_P = P_coarse
    for j in 0:(n1 - 1)
        P = P_lo + (P_hi - P_lo) * (j / (n1 - 1))
        s = snr_at(P)
        s > best_snr && (best_snr = s; best_P = P)
    end
    # Stage 2: zoom ±3 stage-1 spacings around the stage-1 max.
    span1 = (P_hi - P_lo) / (n1 - 1)
    P_lo2 = best_P - 3 * span1; P_hi2 = best_P + 3 * span1
    for j in 0:(n2 - 1)
        P = P_lo2 + (P_hi2 - P_lo2) * (j / (n2 - 1))
        s = snr_at(P)
        s > best_snr && (best_snr = s; best_P = P)
    end
    # Depth at the refined period.
    sw = 0.0; swf = 0.0
    @inbounds for i in 1:n
        d = t[i] - t_anchor
        r = d - best_P * round(d / best_P)
        if abs(r) < half_dur
            sw += w[i]; swf += wf[i]
        end
    end
    depth = sw > 0 ? -swf / sw : 0.0
    # Return the ABSOLUTE transit epoch (t_anchor — held fixed through the P
    # scan, so it IS the transit time near the data start at best_P), NOT the
    # BJD-0-anchored fold phase t0_coarse. Reconstructing an epoch from
    # t0_coarse downstream via t1 + mod(t0 − t1, P) has a t1/P ≈ million-cycle
    # lever arm: any P mismatch (even the proposal's own σ_P) randomizes the
    # epoch uniformly over the period — measured on WASP-47 e: |Tc offset|/P
    # was uniform and every birth missed the transit.
    return (best_P, depth, t_anchor, dur_coarse, best_snr)
end

"""
    _harmonic_dedup(peak_periods, peak_weights, active_periods; tol=0.03, max_order=5)
        -> (kept_periods, kept_weights)

Drop periods that are within `tol` (fractional) of m·P/n for any active
planet period P, with m, n ∈ 1:max_order. The point of this is to keep
BLS peaks at b's harmonics (8.31 = 2P_b, 12.48 = 3P_b, 2.08 = P_b/2,
1.39 = P_b/3, ...) from drowning out genuine non-harmonic signals
(d at 9.03, e at 0.79) in the proposal mixture.
"""
function _harmonic_dedup(peak_periods::Vector{Float64},
                          peak_weights::Vector{Float64},
                          active_periods::Vector{Float64};
                          tol::Float64 = 0.03,
                          max_order::Int = 5)
    isempty(active_periods) && return (peak_periods, peak_weights)
    keep = trues(length(peak_periods))
    @inbounds for i in eachindex(peak_periods)
        Pp = peak_periods[i]
        for Pa in active_periods
            matched = false
            for m in 1:max_order, n in 1:max_order
                target = m * Pa / n
                if abs(Pp - target) / target < tol
                    matched = true
                    break
                end
            end
            if matched
                keep[i] = false
                break
            end
        end
    end
    return (peak_periods[keep], peak_weights[keep])
end

"""
    _family_dedup_keep(peak_periods, peak_depths; tol=0.03, max_order=5)
        -> keep::BitVector

INTRA-list harmonic dedup: collapse each integer-ratio family of BLS peaks
to its fundamental. Without this, a strong unmodelled USP (WASP-47 e)
fills the whole proposal list with its own family (2×, 2.5×, 3×, 5×Pₑ
measured) and dilutes the fundamental's proposal mass ~5×.

BLS scores a transit at BOTH integer multiples m·P (all transits fold into
m clumps; the box catches one → ~FULL depth) and sub-harmonics P/n (every
n-th fold carries the transit; the bin averages it with n−1 flat folds →
depth DILUTED ~n×). So within a family the discriminator is depth, not
period order: (1) drop members with depth < 0.6× the family max
(sub-harmonics); (2) of the survivors — fundamental + multiples, all at
~full depth — keep the SHORTEST period (the singly-coherent fold).
A naive keep-shortest (no depth gate) is WRONG: it discards a real
fundamental in favour of its own diluted sub-harmonics — measured on
WASP-47 b: list {4.159@8700ppm, 2.08@~4400, 1.39@~2900} must keep 4.159.
"""
function _family_dedup_keep(peak_periods::Vector{Float64},
                             peak_depths::Vector{Float64};
                             tol::Float64 = 0.03, max_order::Int = 5)
    n = length(peak_periods)
    keep = trues(n)
    n <= 1 && return keep
    related(Pi, Pj) = begin                   # Pi ≈ (m/d)·Pj for small m, d?
        r = false
        for m in 1:max_order, d in 1:max_order
            m == d && continue
            target = m * Pj / d
            if abs(Pi - target) / target < tol
                r = true
                break
            end
        end
        r
    end
    # Greedy family clustering (transitive via the seed member is fine for a
    # proposal list), then the depth gate + keep-shortest within each family.
    assigned = falses(n)
    @inbounds for i in 1:n
        assigned[i] && continue
        fam = Int[i]
        assigned[i] = true
        for j in (i+1):n
            assigned[j] && continue
            if related(peak_periods[j], peak_periods[i])
                push!(fam, j)
                assigned[j] = true
            end
        end
        length(fam) == 1 && continue
        dmax = maximum(peak_depths[f] for f in fam)
        full = [f for f in fam if peak_depths[f] >= 0.6 * dmax]
        isempty(full) && (full = fam)
        keep_f = full[argmin(peak_periods[f] for f in full)]
        for f in fam
            keep[f] = (f == keep_f)
        end
    end
    return keep
end
