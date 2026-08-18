# Notch filter detrender (Rizzuto+ 2017, AJ 154 224).
#
# Designed for active stars where SG/GP detrenders silently absorb
# transit signals into the trend. At each cadence i, fit two models on
# a sliding window centered at t_i:
#
#   null:  flux ≈ a + b·t + c·t²
#   notch: flux ≈ (a + b·t + c·t²) · (1 − δ · 1[|t − t_i| < w/2])
#
# Compare via ΔBIC and pick the trend (poly part only) from the
# winning model. Notch shape is preserved as a residual, so a real
# transit shows up after detrending instead of being absorbed.
#
# Reference Python implementation: arizzuto/Notch_and_LOCoR core.py
# (sliding_window). This port uses an alternating linear-LSQ fit
# rather than Levenberg-Marquardt — the model is bilinear (linear in
# (a,b,c) for fixed δ; linear in δ for fixed (a,b,c)), so 5 alternations
# converge to the joint optimum without a nonlinear solver dep.

using Statistics: median, mean, std

"""
    detrend_notch(t, flux, flux_err; window=1.0,
                   durations=[1.0,2.0,3.0,4.0,6.0]./24,
                   ie_frac=0.1, delta_bic=-1.0, niter=5, gap_size=0.5)
        -> NamedTuple

Notch-filter detrending (Rizzuto+ 2017). Returns a NamedTuple with:

- `flux_detrended` — `flux ./ trend` (multiplicative convention; transit shape preserved as residual ≈ 1 − depth)
- `trend` — chosen polynomial trend at each cadence
- `depth` — best-fit notch depth at each cadence (0 when null model wins)
- `delta_bic` — `BIC_null − BIC_notch` per cadence; positive ⇒ notch wins
- `dur_best` — best-fit duration per cadence (days)
- `outlier` — `Bool` outlier flag (set during iterative σ-clipping)
- `segments` — segment ranges (gaps split via `find_segments`)

# Arguments

- `t, flux, flux_err` — light curve arrays (BJD, normalized flux, error)
- `window::Real=1.0` — sliding window width in days. ~1d is right for
   slow K-dwarf rotators (P_rot ≳ 5d); drop to 0.5d for fast rotators
   where rotational gradient over 1d violates the local poly2 assumption.
- `durations` — grid of trial transit duration *full-widths* (days).
   Default `[1, 2, 3, 4, 6] / 24` covers 1-6 hr durations including
   common short-period (~3 hr) and warm-Jupiter (~6 hr) regimes. The
   trapezoid (see `ie_frac`) means a wider trial doesn't dilute depth
   for shorter true transits — partial-coverage points are weighted by
   the trapezoid shape.
- `ie_frac::Real=0.1` — ingress/egress fraction of half-width. `0.0`
   recovers Rizzuto's box notch; `0.1` (default) is a trapezoid with
   ingress ≈ 10% of half-width on each side, matching the typical
   ingress duration of small-planet TESS transits and avoiding the
   width-mismatch dilution of a box.
- `delta_bic::Real=-1.0` — accept notch if `BIC_null − BIC_notch ≥
   delta_bic`. `-1.0` is permissive (Rizzuto default); set to `0.0` for
   stricter, `Inf` to force null everywhere.
- `niter::Int=5` — iterative σ-clip passes per window (3.5σ).
- `gap_size::Real=0.5` — segment break threshold (days), passed to
   [`find_segments`](@ref).

The detrended flux is a faithful preservation of transit dips: a real
transit appears at ≈ `1 − depth` in `flux_detrended`, and `delta_bic`
becomes a per-cadence "is there a transit-shaped feature here?"
statistic — feed it into TLS or BLS as a coarse pre-filter, or use
`flux_detrended` directly as the search input.
"""
function detrend_notch(t::Vector{Float64}, flux::Vector{Float64},
                        flux_err::Vector{Float64};
                        window::Real = 1.0,
                        durations::AbstractVector{<:Real} = [1.0, 2.0, 3.0, 4.0, 6.0] ./ 24,
                        ie_frac::Real = 0.1,
                        delta_bic::Real = -1.0,
                        niter::Int = 5,
                        refine::Bool = true,
                        gap_size::Real = 0.5,
                        show_progress::Bool = false,
                        sector_id::Union{Nothing, AbstractVector{<:Integer}} = nothing)
    n = length(t)
    n == length(flux) == length(flux_err) || throw(ArgumentError(
        "t, flux, flux_err must have the same length"))
    sector_id === nothing || length(sector_id) == n || throw(ArgumentError(
        "sector_id length $(length(sector_id)) != length(t) $n"))
    sid_out = sector_id === nothing ? nothing : Vector{Int}(sector_id)
    window > 0 || throw(ArgumentError("window must be positive"))
    isempty(durations) && throw(ArgumentError("durations must be non-empty"))

    segments = find_segments(t; gap_size = gap_size)

    flux_detrended = similar(flux)
    trend = similar(flux)
    depth = zeros(Float64, n)
    dbic = zeros(Float64, n)
    dur_best = zeros(Float64, n)
    outlier = falses(n)

    durs = collect(Float64, durations)
    half = window / 2
    ief = clamp(Float64(ie_frac), 0.0, 0.49)

    # Pass 1: detect the dips (no global masking yet). Accumulate every
    # accepted notch's in-trapezoid points into `dip_mask`.
    dip_mask = refine ? falses(n) : nothing
    for seg in segments
        _detrend_notch_segment!(flux_detrended, trend, depth, dbic,
                                dur_best, outlier,
                                t, flux, flux_err, seg, half, durs, ief,
                                Float64(delta_bic), niter, show_progress,
                                nothing, dip_mask, false)
    end

    # Pass 2: re-fit every window's trend with ALL pass-1 dips excluded, so a
    # transit sitting off-centre in a flanking window can no longer bend the
    # local poly2 — the source of the ±0.25% detrended "shoulders" around each
    # dip. Skipped when nothing was detected (pass 2 is then identical) or
    # refine=false. Costs a second fit pass; correctness over speed.
    if refine && any(dip_mask)
        for seg in segments
            _detrend_notch_segment!(flux_detrended, trend, depth, dbic,
                                    dur_best, outlier,
                                    t, flux, flux_err, seg, half, durs, ief,
                                    Float64(delta_bic), niter, show_progress,
                                    dip_mask, nothing, true)
        end
        # The per-cadence local poly2 trend still JUMPS across each detected
        # transit: every window extrapolates the dip-excluded fit across the
        # gap independently, so on a steep/curved baseline (a transit on a
        # rotation extremum) the trend rings into a V/overshoot. Carry a SMOOTH
        # baseline across each detected dip — interpolate the trend from the
        # clean cadences bracketing it, as the masked detrenders do — so the
        # transit survives cleanly in flux/trend with no trend artefact.
        _interp_trend_across_dips!(trend, flux_detrended, flux, t,
                                   dip_mask, segments)
    end

    return (; flux_detrended, trend, depth, delta_bic = dbic,
            dur_best, outlier, segments, sector_id = sid_out)
end

function _detrend_notch_segment!(flux_detrended, trend, depth, dbic,
                                 dur_best, outlier,
                                 t, flux, flux_err, seg, half, durs, ie_frac,
                                 delta_bic_thresh, niter, show_progress,
                                 dip_in, dip_out, trend_only)
    seg_t = view(t, seg)
    seg_y = view(flux, seg)
    seg_e = view(flux_err, seg)
    nseg = length(seg)

    # Each cadence i: pull window indices via two-pointer sweep. With
    # monotone t, the lower/upper window edges advance left-to-right.
    lo, hi = 1, 1
    for (k, i_glob) in enumerate(seg)
        ti = t[i_glob]
        while lo <= nseg && seg_t[lo] < ti - half
            lo += 1
        end
        while hi <= nseg && seg_t[hi] <= ti + half
            hi += 1
        end
        win = lo:(hi - 1)
        nwin = length(win)

        if nwin < 6  # too few points for poly2 + notch; fall back to identity
            trend[i_glob] = flux[i_glob]
            flux_detrended[i_glob] = 1.0
            continue
        end

        tw = collect(view(seg_t, win) .- ti)        # center on t_i
        yw = collect(view(seg_y, win))
        ew = collect(view(seg_e, win))

        # Window-local view of the global in-dip mask (pass 2 only). Excludes
        # off-centre transit points from this window's trend fit.
        dipw = dip_in === nothing ? nothing :
               Bool[dip_in[seg[w]] for w in win]

        # Iterative σ-clip + alternating linear LSQ for joint poly+notch.
        result = _notch_window_fit(tw, yw, ew, durs, ie_frac,
                                   delta_bic_thresh, niter, dipw)

        trend[i_glob] = result.trend_at_center
        flux_detrended[i_glob] = flux[i_glob] / result.trend_at_center
        # Detection products (depth, Δbic, duration, outliers) come ONLY from
        # pass 1, where the transit is still IN the trend fit so the notch-vs-
        # null BIC has full contrast. Pass 2 (trend_only) recomputes a cleaner
        # baseline but its dip-excluded trend σ-clips the transit out of the
        # detection χ² — which would silently destroy the very detections we
        # need — so we keep pass 1's here.
        if !trend_only
            depth[i_glob] = result.depth
            dbic[i_glob] = result.delta_bic
            dur_best[i_glob] = result.duration
            # Mark global outliers — translate window-local indices back.
            for j in result.outlier_idx
                outlier[seg[win[j]]] = true
            end
        end
        # Pass 1: record this window's detected dip (its in-trapezoid points)
        # into the global mask so pass 2 excludes the WHOLE transit from every
        # flanking window's trend, not just the one centred on it.
        if dip_out !== nothing && result.delta_bic >= delta_bic_thresh &&
           result.depth > 0
            hw = result.duration / 2 * (1 + ie_frac)
            @inbounds for w in win
                abs(seg_t[w] - ti) <= hw && (dip_out[seg[w]] = true)
            end
        end
    end
end

# Carry a SMOOTH baseline across each detected-dip run. The per-cadence local
# poly2 trend jumps across a transit (each window extrapolates the dip-excluded
# fit independently), so on a steep/curved baseline it rings into a V/overshoot.
# Within each contiguous `dip_mask` run we replace the trend with a straight
# line between the trend values at the clean cadences bracketing the run, then
# refresh flux_detrended = flux / trend there. Runs touching a segment edge use
# the one available bracket (flat carry). Operates per segment so it never
# interpolates across a data gap. The chord error vs the true curve is tiny —
# the run is short and the trend is locally near-extremal there anyway.
function _interp_trend_across_dips!(trend, flux_detrended, flux, t,
                                    dip_mask, segments)
    @inbounds for seg in segments
        s0, s1 = first(seg), last(seg)
        i = s0
        while i <= s1
            if dip_mask[i]
                j = i
                while j < s1 && dip_mask[j + 1]
                    j += 1
                end
                left  = i - 1 >= s0 ? i - 1 : 0      # clean bracket indices
                right = j + 1 <= s1 ? j + 1 : 0       # 0 = none (segment edge)
                if left != 0 && right != 0
                    tL, tR = t[left], t[right]
                    yL, yR = trend[left], trend[right]
                    dt = tR - tL
                    for k in i:j
                        w = dt > 0 ? (t[k] - tL) / dt : 0.0
                        trend[k] = yL + w * (yR - yL)
                        flux_detrended[k] = flux[k] / trend[k]
                    end
                elseif left != 0 || right != 0
                    yedge = trend[left != 0 ? left : right]
                    for k in i:j
                        trend[k] = yedge
                        flux_detrended[k] = flux[k] / trend[k]
                    end
                end
                i = j + 1
            else
                i += 1
            end
        end
    end
    return nothing
end

# Per-window joint fit. Returns trend_at_center, depth, delta_bic,
# duration, outlier_idx.
#
# Notch shape per point j is a trapezoid in `t`:
#   - 1.0 inside the flat region (|t| < half - ie),
#   - linear ramp 1→0 across the ingress/egress (|t| ∈ [half - ie, half + ie]),
#   - 0.0 outside the support.
# `ie_frac=0` recovers Rizzuto's box. The model is
#   y_j ≈ (a + b·t_j + c·t_j²) · (1 − δ · shape_j).
# Linear in (a,b,c) for fixed δ, linear in δ for fixed (a,b,c) — so
# alternating linear LSQ converges to the joint optimum without LM.
function _notch_window_fit(t::Vector{Float64}, y::Vector{Float64},
                            e::Vector{Float64},
                            durs::Vector{Float64},
                            ie_frac::Float64,
                            delta_bic_thresh::Float64,
                            niter::Int,
                            dipw::Union{Nothing, Vector{Bool}} = nothing)
    nw = length(t)
    # Seed the trend from points that are NOT global pass-1 dips. Fitting the
    # initial poly2 over a transit — especially one lying OFF-CENTRE in this
    # window — bends a0 and inflates the σ-clip scale, so the dip survives
    # clipping and biases the trend (the ±0.2% detrended "shoulders"). Masking
    # dips from the seed lets the σ-clip remove them properly.
    base = dipw === nothing ? trues(nw) : Bool[!d for d in dipw]
    a0, b0, c0 = _polyfit2(t, y, e, base)

    cliplim = 3.5
    best_chi2 = Inf
    best_dur = durs[1]
    best_a, best_b, best_c = a0, b0, c0
    best_depth = 0.0
    best_keep = trues(nw)
    shape_buf = Vector{Float64}(undef, nw)

    for w in durs
        # Build trapezoid shape for this trial width.
        _trap_shape!(shape_buf, t, w, ie_frac)
        n_full = count(>(0.99), shape_buf)
        n_oot  = count(==(0.0), shape_buf)
        # Need full-coverage in-notch points (>=1) and OOT points (>=4)
        # to identify both the depth and the polynomial trend.
        (n_full < 1 || n_oot < 4) && continue

        kk = trues(nw)
        a, b, c = a0, b0, c0
        d = 0.0
        chi2_w = Inf

        for iter in 1:niter
            # Step 1: fit poly2 to OOT-only kept points (shape == 0).
            # Partial-transit points (shape ∈ (0,1)) are excluded from the
            # trend fit because they carry transit signal; so are global
            # pass-1 dips (dipw) — a transit lying OFF-CENTRE in this window
            # would otherwise bend the poly2 and ring the detrended baseline.
            n_oot_kept = 0
            @inbounds for j in eachindex(t)
                (kk[j] && shape_buf[j] == 0 &&
                 (dipw === nothing || !dipw[j])) && (n_oot_kept += 1)
            end
            n_oot_kept < 4 && break
            mask_oot = falses(nw)
            @inbounds for j in eachindex(t)
                mask_oot[j] = kk[j] && shape_buf[j] == 0 &&
                              (dipw === nothing || !dipw[j])
            end
            a, b, c = _polyfit2(t, y, e, mask_oot)

            # Step 2: solve for depth from points with shape > 0,
            # weighted by shape · poly. For fixed (a,b,c):
            #   y_j = p_j (1 - d s_j)  ⇒  (p_j - y_j) = d · p_j s_j
            # → d_hat = Σ w_j p_j s_j (p_j - y_j) / Σ w_j (p_j s_j)²
            num = 0.0; den = 0.0
            @inbounds for j in eachindex(t)
                (kk[j] && shape_buf[j] > 0) || continue
                p = a + b * t[j] + c * t[j]^2
                ps = p * shape_buf[j]
                wj = 1.0 / e[j]^2
                num += wj * (p - y[j]) * ps
                den += wj * ps * ps
            end
            d = den > 0 ? clamp(num / den, 0.0, 1.0) : 0.0

            # χ² with current model
            chi2_w = 0.0
            @inbounds for j in eachindex(t)
                kk[j] || continue
                p = a + b * t[j] + c * t[j]^2
                model = p * (1 - d * shape_buf[j])
                r = (y[j] - model) / e[j]
                chi2_w += r * r
            end

            n_kept = count(kk)
            n_kept < 5 && break

            # σ-clip from full residual rms of currently-kept points.
            rms2 = 0.0
            @inbounds for j in eachindex(t)
                kk[j] || continue
                p = a + b * t[j] + c * t[j]^2
                model = p * (1 - d * shape_buf[j])
                rms2 += (y[j] - model)^2
            end
            σ_w = sqrt(rms2 / max(n_kept - 1, 1))
            σ_w == 0 && break

            any_new = false
            @inbounds for j in eachindex(t)
                kk[j] || continue
                p = a + b * t[j] + c * t[j]^2
                model = p * (1 - d * shape_buf[j])
                if abs(y[j] - model) > cliplim * σ_w
                    kk[j] = false
                    any_new = true
                end
            end
            any_new || break
        end

        if chi2_w < best_chi2
            best_chi2 = chi2_w
            best_dur = w
            best_a = a
            best_b = b
            best_c = c
            best_depth = d
            best_keep = copy(kk)
        end
    end

    # Null model: re-fit poly2 to currently-kept points (post-clip best mask).
    a_null, b_null, c_null = _polyfit2(t, y, e, best_keep)
    chi2_null = 0.0
    @inbounds for j in eachindex(t)
        best_keep[j] || continue
        p = a_null + b_null * t[j] + c_null * t[j]^2
        r = (y[j] - p) / e[j]
        chi2_null += r * r
    end

    n_eff = count(best_keep)
    # BIC: null has 3 params (a,b,c); notch has 5 (a,b,c,depth,duration).
    # Rizzuto uses (numpars - 2) for null vs numpars for notch ⇒ Δ = 2·log N.
    bic_null = chi2_null + 3 * log(max(n_eff, 2))
    bic_notch = best_chi2 + 5 * log(max(n_eff, 2))
    Δbic = bic_null - bic_notch

    accept_notch = Δbic >= delta_bic_thresh

    # Trend at center (t_local = 0) — poly part only.
    if accept_notch
        trend_at_center = best_a   # a + b*0 + c*0 = a
    else
        # Output trend at a NON-detected window must still ignore global pass-1
        # dips: an off-centre transit inflates the σ-clip scale, survives into
        # best_keep, and bends the null poly2 → the detrended "shoulders". The
        # BIC above deliberately used the dip-INCLUSIVE null, so detection is
        # unchanged; only the reported baseline is cleaned.
        nk = dipw === nothing ? best_keep :
             Bool[best_keep[j] && !dipw[j] for j in 1:nw]
        if dipw !== nothing && count(nk) >= 3 && count(nk) < count(best_keep)
            a_out, _, _ = _polyfit2(t, y, e, nk)
            trend_at_center = a_out
        else
            trend_at_center = a_null
        end
        best_depth = 0.0
    end

    outlier_idx = findall(!, best_keep)

    return (; trend_at_center,
            depth = best_depth,
            delta_bic = Δbic,
            duration = best_dur,
            outlier_idx)
end

"""Trapezoid shape on a window centered at t = 0.

`shape[j] = 1` inside `(-half + ie, half - ie)`, ramps linearly to 0 at
`±(half + ie)`, and is 0 outside. `ie_frac = 0` collapses to a box.
"""
function _trap_shape!(shape::Vector{Float64}, t::Vector{Float64},
                       width::Float64, ie_frac::Float64)
    half = width / 2
    ie = ie_frac * half
    flat = half - ie
    fall = half + ie
    @inbounds for j in eachindex(t)
        Δ = abs(t[j])
        if Δ <= flat
            shape[j] = 1.0
        elseif Δ < fall && fall > flat
            shape[j] = (fall - Δ) / (fall - flat)
        else
            shape[j] = 0.0
        end
    end
    return shape
end

"""Weighted least-squares poly2 fit `y ≈ a + b·t + c·t²` over `mask`-true points.
Returns `(a, b, c)`. Numerically robust via 3×3 normal equations on a
mean-zero `t` (the caller already centers t at t_i, so mean(t) ≈ 0).
"""
function _polyfit2(t::Vector{Float64}, y::Vector{Float64},
                    e::Vector{Float64}, mask::AbstractVector{Bool})
    # Build weighted normal equations Aᵀ W A · β = Aᵀ W y.
    s00 = 0.0; s10 = 0.0; s20 = 0.0; s30 = 0.0; s40 = 0.0
    sy0 = 0.0; sy1 = 0.0; sy2 = 0.0
    @inbounds for j in eachindex(t)
        mask[j] || continue
        w = 1.0 / e[j]^2
        tj = t[j]
        tj2 = tj * tj
        s00 += w
        s10 += w * tj
        s20 += w * tj2
        s30 += w * tj2 * tj
        s40 += w * tj2 * tj2
        sy0 += w * y[j]
        sy1 += w * tj * y[j]
        sy2 += w * tj2 * y[j]
    end
    # Symmetric 3×3 normal-equations system, solved by Cramer's rule.
    # The design matrix is well-conditioned since t is centered at 0 and
    # bounded by half the window width, so direct inversion is fine.
    return _solve_sym3(s00, s10, s20, s30, s40, sy0, sy1, sy2)
end

@inline function _solve_sym3(s00::Float64, s10::Float64, s20::Float64,
                              s30::Float64, s40::Float64,
                              b1::Float64, b2::Float64, b3::Float64)
    a11, a12, a13 = s00, s10, s20
    a21, a22, a23 = s10, s20, s30
    a31, a32, a33 = s20, s30, s40
    detM = a11 * (a22 * a33 - a23 * a32) -
           a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31)
    if abs(detM) < 1e-30
        return (0.0, 0.0, 0.0)
    end
    da = b1 * (a22 * a33 - a23 * a32) -
         a12 * (b2 * a33 - a23 * b3) +
         a13 * (b2 * a32 - a22 * b3)
    db = a11 * (b2 * a33 - a23 * b3) -
         b1 * (a21 * a33 - a23 * a31) +
         a13 * (a21 * b3 - b2 * a31)
    dc = a11 * (a22 * b3 - b2 * a32) -
         a12 * (a21 * b3 - b2 * a31) +
         b1 * (a21 * a32 - a22 * a31)
    return (da / detM, db / detM, dc / detM)
end
