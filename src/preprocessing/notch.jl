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
                   ie_frac=0.1, delta_bic=-1.0, niter=5, gap_size=0.5,
                   transit_mask=nothing, notch_search=true)
        -> NamedTuple

Notch-filter detrending (Rizzuto+ 2017). Returns a NamedTuple with:

- `flux_detrended` — `flux ./ trend` (multiplicative convention; transit shape preserved as residual ≈ 1 − depth)
- `trend` — chosen polynomial trend at each cadence
- `depth` — best-fit notch depth at each cadence (0 when null model wins)
- `delta_bic` — `BIC_null − BIC_notch` per cadence; positive ⇒ notch wins
- `dur_best` — best-fit duration per cadence (days)
- `outlier` — `Bool` outlier flag (set during iterative σ-clipping)
- `segments` — segment ranges (gaps split via `find_segments`)
- `transit_mask` — the mask used (passed through or `nothing`)

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
- `transit_mask::Union{Nothing, AbstractVector{Bool}}=nothing` — known
   transit cadences (`true` = in transit), same meaning as in
   [`detrend_savgol`](@ref). Masked cadences are excluded from every local
   window fit (the poly2 trend, the σ-clip and the notch search all see only
   unmasked points). The trend across each masked run (and across each dip
   the search found) is one σ-clipped poly2 fitted to the clean flux on both
   flanks, so `flux_detrended = flux ./ trend` keeps the transit depth.
   The returned `transit_mask` echoes the input. Build it from known `P`,
   `T0` and transit *durations*; these are unrelated to the notch search
   widths in `durations` above. With a mask, the BIC notch search runs only
   on the unmasked points, and at masked cadences it is not run at all:
   `depth`, `delta_bic` and `dur_best` are `0` and `outlier` is `false`
   there.
- `notch_search::Bool=true` — `false` skips the BIC notch search entirely: the
   trend in each window is a σ-clipped poly2 over the unmasked points, and
   `depth`, `delta_bic`, `dur_best` are all `0`. Use it with `transit_mask`
   when the transits are already known and only a clean local baseline is
   wanted. Without a mask it is a plain sliding-window poly2 detrender.

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
                        sector_id::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                        transit_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
                        notch_search::Bool = true)
    n = length(t)
    n == length(flux) == length(flux_err) || throw(ArgumentError(
        "t, flux, flux_err must have the same length"))
    sector_id === nothing || length(sector_id) == n || throw(ArgumentError(
        "sector_id length $(length(sector_id)) != length(t) $n"))
    sid_out = sector_id === nothing ? nothing : Vector{Int}(sector_id)
    transit_mask === nothing || length(transit_mask) == n || throw(ArgumentError(
        "transit_mask length $(length(transit_mask)) != length(t) $n"))
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
    dip_mask = (refine && notch_search) ? falses(n) : nothing
    for seg in segments
        _detrend_notch_segment!(flux_detrended, trend, depth, dbic,
                                dur_best, outlier,
                                t, flux, flux_err, seg, half, durs, ief,
                                Float64(delta_bic), niter, show_progress,
                                nothing, dip_mask, false,
                                transit_mask, notch_search)
    end

    # Pass 2: re-fit every window's trend with ALL pass-1 dips excluded, so a
    # transit sitting off-centre in a flanking window can no longer bend the
    # local poly2 — the source of the ±0.25% detrended "shoulders" around each
    # dip. Skipped when nothing was detected (pass 2 is then identical) or
    # refine=false. Costs a second fit pass; correctness over speed.
    if dip_mask !== nothing && any(dip_mask)
        for seg in segments
            _detrend_notch_segment!(flux_detrended, trend, depth, dbic,
                                    dur_best, outlier,
                                    t, flux, flux_err, seg, half, durs, ief,
                                    Float64(delta_bic), niter, show_progress,
                                    dip_mask, nothing, true,
                                    transit_mask, notch_search)
        end
    end

    # Excluded cadences (pass-1 dips, known transits) and the clean cadences
    # next to them get no trustworthy per-window trend: their windows hold
    # clean data on one side only, and the poly2 extrapolated across the gap
    # can be off by several ppt. Bridge each such run with one poly2 fitted
    # to the clean flux on both flanks (see `_bridge_trend!`).
    excl = dip_mask !== nothing && any(dip_mask) ?
           (transit_mask === nothing ? dip_mask : dip_mask .| transit_mask) :
           (transit_mask === nothing ? nothing : BitVector(transit_mask))
    if excl !== nothing && any(excl)
        _bridge_trend!(trend, flux_detrended, t, flux, flux_err,
                       excl, segments, half, niter)
    end

    return (; flux_detrended, trend, depth, delta_bic = dbic,
            dur_best, outlier, segments, transit_mask,
            sector_id = sid_out)
end

function _detrend_notch_segment!(flux_detrended, trend, depth, dbic,
                                 dur_best, outlier,
                                 t, flux, flux_err, seg, half, durs, ie_frac,
                                 delta_bic_thresh, niter, show_progress,
                                 dip_in, dip_out, trend_only,
                                 transit_mask = nothing, notch_search = true)
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

        # Window-local known-transit mask: excluded from every fit below.
        maskw = transit_mask === nothing ? nothing :
                Bool[transit_mask[seg[w]] for w in win]

        # Trend-only path: the notch search is switched off, or this cadence is
        # itself a masked (known-transit) one, so there is no search to run.
        if !notch_search || (maskw !== nothing && transit_mask[i_glob])
            excl = _union_excl(dipw, maskw, length(tw))
            tr = _clipped_poly2_trend(tw, yw, ew, excl, niter)
            isfinite(tr) || (tr = flux[i_glob])
            trend[i_glob] = tr
            flux_detrended[i_glob] = flux[i_glob] / tr
            if !trend_only
                depth[i_glob] = 0.0
                dbic[i_glob] = 0.0
                dur_best[i_glob] = 0.0
            end
            continue
        end

        # Iterative σ-clip + alternating linear LSQ for joint poly+notch.
        result = _notch_window_fit(tw, yw, ew, durs, ie_frac,
                                   delta_bic_thresh, niter, dipw, maskw)
        if maskw !== nothing && !(result.trend_at_center > 0 &&
                                  isfinite(result.trend_at_center))
            # Too few unmasked points survived to identify the poly2.
            tr = _clipped_poly2_trend(tw, yw, ew,
                                      _union_excl(dipw, maskw, length(tw)), niter)
            isfinite(tr) || (tr = flux[i_glob])
            result = merge(result, (; trend_at_center = tr))
        end

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

# Union of the optional exclusion masks (global dips, known transits) as a
# window-local `Vector{Bool}` (true = excluded).
function _union_excl(dipw, maskw, nw::Int)
    ex = falses(nw)
    dipw === nothing || (ex .|= dipw)
    maskw === nothing || (ex .|= maskw)
    return collect(ex)
end

# Weighted poly2 `y ≈ a + b·t + c·t²` over the `keep`-true points with
# iterative 3.5σ clipping. `keep` is updated in place (clipped points set
# false). Returns `(a, b, c)`, or `nothing` when fewer than 4 points remain.
function _clipped_poly2!(t::Vector{Float64}, y::Vector{Float64},
                         e::Vector{Float64}, kk::Vector{Bool},
                         niter::Int)
    count(kk) < 4 && return nothing
    a, b, c = NaN, NaN, NaN
    for _ in 1:max(niter, 1)
        a, b, c = _polyfit2(t, y, e, kk)
        rms2 = 0.0; nk = 0
        @inbounds for j in eachindex(t)
            kk[j] || continue
            rms2 += (y[j] - (a + b * t[j] + c * t[j]^2))^2
            nk += 1
        end
        σ = sqrt(rms2 / max(nk - 1, 1))
        σ == 0 && break
        any_new = false
        @inbounds for j in eachindex(t)
            kk[j] || continue
            if abs(y[j] - (a + b * t[j] + c * t[j]^2)) > 3.5 * σ
                kk[j] = false
                any_new = true
            end
        end
        (any_new && count(kk) >= 4) || break
    end
    return (a, b, c)
end

# Poly2 trend at the window centre (t = 0) from the non-excluded points, with
# iterative 3.5σ clipping. No notch search. Returns NaN when fewer than 4
# points are available (the caller falls back to the raw flux).
function _clipped_poly2_trend(t::Vector{Float64}, y::Vector{Float64},
                              e::Vector{Float64}, excl::AbstractVector{Bool},
                              niter::Int)
    coefs = _clipped_poly2!(t, y, e, Bool[!x for x in excl], niter)
    return coefs === nothing ? NaN : coefs[1]
end


# A flank needs at least this many clean cadences to anchor a bridge.
const _BRIDGE_MIN_FLANK = 5
# A clean cadence is "unsupported" when, on a side of its `±half` window that
# holds data, fewer than this fraction of that side's cadences are clean.
const _BRIDGE_MIN_SIDE_FRAC = 0.25

# Clean cadences whose `±half` window is mostly excluded on one side. Their
# per-window poly2 is fitted almost entirely to data on the other side and
# extrapolated to the centre (a short clean stretch wedged between two
# excluded runs, a cadence just outside a long run) and can be off by several
# ppt. A side with nothing excluded is never held against a cadence, however
# few cadences it holds (an ordinary window at a segment edge).
function _unsupported_cadences(t, excl::AbstractVector{Bool}, segments,
                               half::Float64)
    n = length(t)
    out = falses(n)
    cc = zeros(Int, n + 1)                    # nclean(a:b) = cc[b+1] - cc[a]
    @inbounds for i in 1:n
        cc[i + 1] = cc[i] + (excl[i] ? 0 : 1)
    end
    for seg in segments
        s1 = last(seg)
        lo, hi = first(seg), first(seg)
        @inbounds for i in seg
            while lo < i && t[lo] < t[i] - half
                lo += 1
            end
            hi = max(hi, i)
            while hi < s1 && t[hi + 1] <= t[i] + half
                hi += 1
            end
            excl[i] && continue
            nl_all, nr_all = i - lo, hi - i
            nl = cc[i] - cc[lo]
            nr = cc[hi + 1] - cc[i + 1]
            if (nl < nl_all && nl < _BRIDGE_MIN_SIDE_FRAC * nl_all) ||
               (nr < nr_all && nr < _BRIDGE_MIN_SIDE_FRAC * nr_all)
                out[i] = true
            end
        end
    end
    return out
end

# Collect clean cadence indices outward from `start` in direction `step`
# (±1) within `s0:s1`, until the collected clean cadences span `A` in time or
# the segment ends.
function _bridge_flank(t, excl, start::Int, step::Int, s0::Int, s1::Int, A::Float64)
    idx = Int[]
    k = start
    while s0 <= k <= s1
        if !excl[k]
            push!(idx, k)
            abs(t[k] - t[idx[1]]) >= A && break
        end
        k += step
    end
    return idx
end

# Replace the trend wherever the per-window poly2 is an extrapolation: at
# excluded cadences (`excl` = pass-1 dips ∪ known transits) and at the
# unsupported clean cadences next to them.
#
# The previous repair drew a chord between the trend values at the two
# cadences bracketing each excluded run, but those values come from windows
# with clean data on one side only. On TOI-4939 S103 (window 0.5 d) a false
# pass-1 dip right after egress left a 20-min clean island whose per-window
# trend was extrapolated across it; the chord anchored there turned a 6.5 ppt
# transit into 8.7 ppt.
#
# Per segment (never across a data gap), each run [i, j] of excluded ∪
# unsupported cadences, of duration L, gets ONE σ-clipped weighted poly2,
# evaluated at every cadence of the run. It is fitted to the RAW flux of
#  - the clean cadences inside the run (short stretches between excluded
#    runs: their flux is good data, only their own window trend is not), and
#  - the clean cadences on each flank, collected outward (skipping excluded
#    ones) until they span A = max(half − L/2, half/2): the whole fit then
#    spans about one `window`, the scale over which the notch already assumes
#    a poly2 baseline, and never less than half a window per flank.
# A run with fewer than `_BRIDGE_MIN_FLANK` clean cadences on one flank (a
# segment edge) gets a straight line fitted to the other flank, collected over
# a full `half`, continued across the run. A run with neither flank usable
# keeps its trend.
function _bridge_trend!(trend, flux_detrended, t, flux, flux_err,
                        excl::AbstractVector{Bool}, segments,
                        half::Float64, niter::Int)
    bridge = excl .| _unsupported_cadences(t, excl, segments, half)
    for seg in segments
        s0, s1 = first(seg), last(seg)
        i = s0
        while i <= s1
            if !bridge[i]
                i += 1
                continue
            end
            j = i
            while j < s1 && bridge[j + 1]
                j += 1
            end
            r = i:j
            i = j + 1

            L = t[last(r)] - t[first(r)]
            A = max(half - L / 2, half / 2)
            left = _bridge_flank(t, excl, first(r) - 1, -1, s0, s1, A)
            right = _bridge_flank(t, excl, last(r) + 1, 1, s0, s1, A)
            nL, nR = length(left), length(right)
            coefs = nothing
            tmid = (t[first(r)] + t[last(r)]) / 2
            inner = Int[q for q in r if !excl[q]]
            if nL >= _BRIDGE_MIN_FLANK && nR >= _BRIDGE_MIN_FLANK
                idx = vcat(left, inner, right)
                tt = Float64[t[q] - tmid for q in idx]
                yy = Float64[flux[q] for q in idx]
                ee = Float64[flux_err[q] for q in idx]
                coefs = _clipped_poly2!(tt, yy, ee, fill(true, length(idx)), niter)
                coefs === nothing && continue
            elseif max(nL, nR) >= _BRIDGE_MIN_FLANK
                # One flank (a segment edge): a straight line through that
                # flank, collected over a full `half`, continued across the run.
                side = nL >= nR ? -1 : 1
                start = side < 0 ? first(r) - 1 : last(r) + 1
                idx = vcat(_bridge_flank(t, excl, start, side, s0, s1, half), inner)
                tt = Float64[t[q] - tmid for q in idx]
                yy = Float64[flux[q] for q in idx]
                ee = Float64[flux_err[q] for q in idx]
                a, b = _clipped_line(tt, yy, ee, niter)
                (isfinite(a) && isfinite(b)) || continue
                coefs = (a, b, 0.0)
            else
                continue
            end
            for q in r
                x = t[q] - tmid
                tr = coefs[1] + coefs[2] * x + coefs[3] * x^2
                (isfinite(tr) && tr > 0) || continue
                trend[q] = tr
                flux_detrended[q] = flux[q] / tr
            end
        end
    end
    return nothing
end
# Weighted straight-line fit `y ≈ a + b·t` with 3.5σ clipping. Returns
# `(a, b)`; `(NaN, NaN)` with fewer than 2 points or a degenerate design.
function _clipped_line(t::Vector{Float64}, y::Vector{Float64},
                       e::Vector{Float64}, niter::Int)
    kk = fill(true, length(t))
    a, b = NaN, NaN
    for _ in 1:max(niter, 1)
        count(kk) < 2 && return (NaN, NaN)
        sw = 0.0; st = 0.0; stt = 0.0; sy = 0.0; sty = 0.0
        @inbounds for j in eachindex(t)
            kk[j] || continue
            w = 1.0 / e[j]^2
            sw += w; st += w * t[j]; stt += w * t[j]^2
            sy += w * y[j]; sty += w * t[j] * y[j]
        end
        det = sw * stt - st^2
        if !(abs(det) > 1e-300)
            # All points at one time: the flat weighted mean.
            return sw > 0 ? (sy / sw, 0.0) : (NaN, NaN)
        end
        a = (stt * sy - st * sty) / det
        b = (sw * sty - st * sy) / det
        rms2 = 0.0; nk = 0
        @inbounds for j in eachindex(t)
            kk[j] || continue
            rms2 += (y[j] - a - b * t[j])^2; nk += 1
        end
        σ = sqrt(rms2 / max(nk - 1, 1))
        σ == 0 && break
        any_new = false
        @inbounds for j in eachindex(t)
            if kk[j] && abs(y[j] - a - b * t[j]) > 3.5 * σ
                kk[j] = false; any_new = true
            end
        end
        (any_new && count(kk) >= 2) || break
    end
    return (a, b)
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
                            dipw::Union{Nothing, Vector{Bool}} = nothing,
                            maskw::Union{Nothing, Vector{Bool}} = nothing)
    nw = length(t)
    # Known-transit cadences never enter any fit; `keep0` is the starting kept
    # set (all true, bitwise the old behaviour, when no mask is given).
    keep0 = maskw === nothing ? trues(nw) : Bool[!m for m in maskw]
    # Seed the trend from points that are NOT global pass-1 dips. Fitting the
    # initial poly2 over a transit — especially one lying OFF-CENTRE in this
    # window — bends a0 and inflates the σ-clip scale, so the dip survives
    # clipping and biases the trend (the ±0.2% detrended "shoulders"). Masking
    # dips from the seed lets the σ-clip remove them properly.
    base = dipw === nothing ? copy(keep0) : Bool[keep0[j] && !dipw[j] for j in 1:nw]
    a0, b0, c0 = _polyfit2(t, y, e, base)

    cliplim = 3.5
    best_chi2 = Inf
    best_dur = durs[1]
    best_a, best_b, best_c = a0, b0, c0
    best_depth = 0.0
    best_keep = copy(keep0)
    shape_buf = Vector{Float64}(undef, nw)

    for w in durs
        # Build trapezoid shape for this trial width.
        _trap_shape!(shape_buf, t, w, ie_frac)
        n_full = 0; n_oot = 0
        @inbounds for j in 1:nw
            keep0[j] || continue
            shape_buf[j] > 0.99 && (n_full += 1)
            shape_buf[j] == 0.0 && (n_oot += 1)
        end
        # Need full-coverage in-notch points (>=1) and OOT points (>=4)
        # to identify both the depth and the polynomial trend.
        (n_full < 1 || n_oot < 4) && continue

        kk = copy(keep0)
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

    outlier_idx = maskw === nothing ? findall(!, best_keep) :
                  [j for j in 1:nw if !best_keep[j] && !maskw[j]]

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
