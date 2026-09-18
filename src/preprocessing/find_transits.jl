# Top-level transit-detection API.
#
# Chains detrending → BLS → intra-self harmonic dedup → physical
# estimates so users don't have to reach into private helpers
# (`_box_least_squares`, `_harmonic_dedup`, `_radius_from_depth`,
# `_mass_radius_chenkipping`, `_K_from_mass`).
#
# For active stars use `detrend = :notch` (Rizzuto+ 2017) — competes a
# local quadratic baseline against a quadratic × box-notch at every
# cadence and picks the BIC-winning model. Survives stellar rotation
# that kills naive savgol/GP detrending.

using Statistics: median
import TransitLeastSquares as _TLS

"""
    _run_tls(t, flux, flux_err; period_min, period_max, R_s, M_s,
              n_peaks, kwargs...) -> (peak_P, peak_depth, peak_t0, peak_score)

Adapter — runs `TransitLeastSquares.tls(…)` and returns the top
`n_peaks` SDE-ranked candidates in the same tuple format as
`_box_least_squares` so `find_transits` can dedup them with the same
pass. `kwargs` are forwarded into `TLSOptions`.
"""
function _run_tls(t::Vector{Float64}, flux::Vector{Float64},
                   flux_err::Vector{Float64};
                   period_min::Real, period_max::Real,
                   R_s::Real, M_s::Real, n_peaks::Int,
                   kwargs...)
    res = _TLS.tls(t, flux; flux_err = flux_err,
                    period_min = Float64(period_min),
                    period_max = Float64(period_max),
                    R_star = Float64(R_s),
                    M_star = Float64(M_s),
                    kwargs...)
    Ps = res.periods
    sde = res.power
    isempty(Ps) &&
        return (Float64[], Float64[], Float64[], Float64[], Float64[])

    # Take the top-`n_peaks` SDE points; the periodogram is broad-peaked
    # but the dedup downstream handles the duplicate sampling.
    order = sortperm(sde; rev = true)
    keep  = order[1:min(n_peaks, length(order))]
    peak_P     = Ps[keep]
    peak_score = sde[keep]
    # TLS only returns the global best (T0, depth, duration) — broadcast
    # it; the caller only uses the top peak's values, and that one matches.
    peak_t0    = fill(res.T0,    length(keep))
    peak_depth = fill(res.depth, length(keep))
    # `duration` is the TLS field name (days); guard in case the package
    # version differs, rather than erroring the whole search.
    dur_val    = hasproperty(res, :duration) ? Float64(res.duration) : NaN
    peak_dur   = fill(dur_val, length(keep))
    return (peak_P, peak_depth, peak_t0, peak_score, peak_dur)
end

"""
    find_transits(t, flux, flux_err; …) -> NamedTuple

Detect candidate transits in a light curve. Returns up to
`max_candidates` peaks ordered by BLS score (Kovács+ 2002), each with
estimated `R_p` [R⊕], `M_p` [M⊕], and `K` [m/s] from the Chen & Kipping
M-R relation.

# Arguments
- `t`, `flux`, `flux_err` — light curve (same length, time in days)

# Keyword arguments
- `detrend::Symbol = :savgol` — `:savgol`, `:gp`, `:notch`, `:locor`, or
  `:none`. Use `:notch` for active stars (Rizzuto+ 2017); `:locor` — LOCoR for
  fast rotators (Rizzuto+ 2017).
- `detrend_kwargs::NamedTuple = (;)` — forwarded to `detrend_*`.
- `trim_mask::Union{Nothing,AbstractVector{Bool}} = nothing` — cadences to
  TRIM. `trim_mask[i] == true` drops cadence `i` ENTIRELY before detrending and
  the period search, so it never appears anywhere in the output, including
  `flux_detrended` (not even in raw form). A per-cadence `transit_mask` in
  `detrend_kwargs` is trimmed in lockstep. Distinct from `transit_mask`, which
  keeps masked cadences but shields them from the detrender.
- `period_min::Real = 0.5`, `period_max::Real = 30.0` [days]
- `n_periods::Int = 10_000` — **floor** on the period-grid size. The grid is
  uniform in *frequency* with resolution sized to the baseline (Ofir 2014):
  `df = min(durations) / (oversample · baseline)`. The actual count is
  `max(n_periods, optimal)`, so long (e.g. multi-sector) baselines auto-refine
  instead of under-resolving the true period onto an alias/harmonic.
- `oversample::Real = 2.0` — frequency-grid oversampling factor (higher =
  finer = slower).
- `max_n_periods::Int = 2_000_000` — hard cap on the auto-sized grid; warns
  if hit.
- `durations::Vector{Float64} = [0.01,0.02,0.04,0.08]` — fractional
  duration grid (transit duration as fraction of period). Used as-is only
  when `scale_durations=false`.
- `scale_durations::Bool = true` — set the BLS box widths per-period from
  the expected duty cycle `q ∝ P^(-2/3)` (from `M_s`,`R_s` → ρ⋆ → a/R⋆),
  instead of a fixed fractional grid. Catches wide-duty-cycle USP transits
  (a fixed narrow grid aliases them to harmonics) without wasting wide
  boxes on long periods. Falls back to `durations` when `false`.
- `local_baseline::Bool = false` — measure transit depth against the bins
  flanking each candidate box (`flank_frac·box_bins` per side) rather than
  the global mean. Robust to a wandering/under-detrended baseline; costs
  SNR on a flat baseline, so opt-in.
- `flank_frac::Real = 1.0` — flank width as a fraction of the box width
  (only used when `local_baseline=true`).
- `n_phase_bins::Int = 100` — BLS phase bin count.
- `score_threshold::Real = 7.0` — minimum BLS score (SNR units) to
  keep.
- `max_candidates::Int = 5`
- `harmonic_tol::Real = 0.03` — fractional tolerance for marking a peak
  as an integer harmonic of a stronger peak (dropped from output).
- `harmonic_max_order::Int = 5` — check `m·P/n` for `m,n ∈ 1:order`.
- `R_s::Real = 1.0` [R_sun], `M_s::Real = 1.0` [M_sun] — stellar
  parameters used to convert BLS depth → R_p, M_p, K.

# Returns
`(periods, t0s, depths, durations, snr, R_p_earth, M_p_earth, K_ms,
   detrend_result, flux_detrended)` where

- `t0s` — mid-transit times as **real epochs** in the same time system as
  `t` (the transit nearest the baseline midpoint), NOT a phase.
- `durations` — best-fit transit durations [days].
- `snr` — detection significance (BLS SNR / TLS SDE). (Formerly `scores`.)
- `flux_detrended` — the detrended flux (NaN where a detrender left a cadence
  non-finite). With no `trim_mask` it is the **same length as the input `flux`**
  and index-aligns with the `t` you passed in. With a `trim_mask`, it spans only
  the KEPT cadences; realign via `t[result.kept]` (see `kept`).
- `kept` — boolean of length == the original input, `true` for cadences that
  survived `trim_mask` (all `true` when no `trim_mask`). `t[kept]` index-aligns
  with `flux_detrended` for phase-folding (see `plot_transit_phasefold`).
"""
function find_transits(
    t::AbstractVector{<:Real},
    flux::AbstractVector{<:Real},
    flux_err::AbstractVector{<:Real};
    method::Symbol = :bls,
    detrend::Symbol = :savgol,
    detrend_kwargs::NamedTuple = NamedTuple(),
    trim_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    sigma_clip::Union{Nothing, Real, NamedTuple} = nothing,
    period_min::Real = 0.5,
    period_max::Real = 30.0,
    n_periods::Int = 10_000,
    oversample::Real = 2.0,
    max_n_periods::Int = 2_000_000,
    durations::Vector{Float64} = [0.01, 0.02, 0.04, 0.08],
    n_phase_bins::Int = 100,
    n_peaks::Int = 20,
    score_threshold::Real = 7.0,
    max_candidates::Int = 5,
    harmonic_tol::Real = 0.03,
    harmonic_max_order::Int = 5,
    R_s::Real = 1.0,
    M_s::Real = 1.0,
    scale_durations::Bool = true,
    local_baseline::Bool = false,
    flank_frac::Real = 1.0,
    tls_options::NamedTuple = NamedTuple(),
)
    method in (:bls, :tls) ||
        throw(ArgumentError("method must be :bls or :tls (got $method)"))
    length(t) == length(flux) == length(flux_err) ||
        throw(ArgumentError("t, flux, flux_err must have matching lengths"))
    n_periods > 0 || throw(ArgumentError("n_periods must be > 0"))
    period_min > 0 && period_max > period_min ||
        throw(ArgumentError("require 0 < period_min < period_max"))

    tv = Vector{Float64}(t)
    fv = Vector{Float64}(flux)
    ev = Vector{Float64}(flux_err)

    # ---- Trim flagged cadences (removed entirely from the search) ------
    # trim_mask[i] == true ⇒ cadence i is dropped BEFORE detrend + BLS, so
    # nothing the search touches or returns (including `flux_detrended`) ever
    # contains it — not even in raw form. A per-cadence `transit_mask` passed
    # through `detrend_kwargs` is trimmed in lockstep so it stays aligned. The
    # returned `kept` boolean (length == original input) maps results back to
    # the cadences you passed in: `t[kept]` aligns with `flux_detrended`.
    # Combined trim = user trim_mask (time/cadence-based) ∪ sigma_clip
    # (statistical: extreme point-outliers via running-median residuals, MAD,
    # protecting any transit_mask in detrend_kwargs). Both removed entirely
    # before detrend + BLS; `kept` maps results back to the input cadences.
    combined = trim_mask === nothing ? falses(length(tv)) : collect(Bool, trim_mask)
    length(combined) == length(tv) || throw(ArgumentError(
        "trim_mask length $(length(combined)) != n cadences $(length(tv))"))
    if sigma_clip !== nothing
        scp  = sigma_clip isa Real ? (sigma = Float64(sigma_clip),) : sigma_clip
        prot = get(detrend_kwargs, :transit_mask, nothing)
        sc = sigma_clip_mask(tv, fv;
            sigma  = Float64(get(scp, :sigma, 5.0)),
            window = Int(get(scp, :window, 101)),
            iters  = Int(get(scp, :iters, 5)),
            protect_mask = prot === nothing ? nothing : Vector{Bool}(prot))
        combined = combined .| sc
    end
    kept = .!combined
    any(kept) || throw(ArgumentError("trim_mask/sigma_clip would remove ALL cadences"))
    if any(combined)
        tv = tv[kept]; fv = fv[kept]; ev = ev[kept]
        if haskey(detrend_kwargs, :transit_mask) && detrend_kwargs.transit_mask !== nothing
            detrend_kwargs = merge(detrend_kwargs,
                                   (; transit_mask = detrend_kwargs.transit_mask[kept]))
        end
    end

    # ---- Detrend -------------------------------------------------------
    dt_result, flux_dt = if detrend === :none
        (nothing, fv)
    elseif detrend === :savgol
        r = detrend_savgol(tv, fv, ev; detrend_kwargs...)
        (r, r.flux_detrended)
    elseif detrend === :gp
        r = detrend_gp(tv, fv, ev; detrend_kwargs...)
        (r, r.flux_detrended)
    elseif detrend === :notch
        r = detrend_notch(tv, fv, ev; detrend_kwargs...)
        (r, r.flux_detrended)
    elseif detrend === :locor
        r = detrend_locor(tv, fv, ev; detrend_kwargs...)
        (r, r.flux_detrended)
    else
        throw(ArgumentError("detrend ∈ (:savgol, :gp, :notch, :locor, :none); got $detrend"))
    end

    # ---- Per-segment leveling -----------------------------------------
    # Detrending is local (per gap-segment via `find_segments`). Before
    # stitching for BLS, re-level each segment's detrended flux to ≈1 and
    # convert its errors from absolute (e.g. raw PDCSAP e⁻/s) to fractional,
    # so per-segment flux-level differences don't bias the periodogram
    # (BLS SNR is sensitive to the error scale). This detrended flux is
    # internal scratch — persisting a clean, sector-aware LC is the job of
    # `clean_lightcurve`/`save_lightcurve`, not `find_transits`.
    if detrend !== :none
        flux_dt = copy(flux_dt)
        ev      = copy(ev)
        segs = (dt_result !== nothing && hasproperty(dt_result, :segments)) ?
               dt_result.segments : find_segments(tv)
        for seg in segs
            m_raw = median(@view fv[seg])                 # absolute flux level
            md    = median(filter(isfinite, @view flux_dt[seg]))
            (isfinite(md)    && md    != 0) && (flux_dt[seg] ./= md)     # → ≈1
            (isfinite(m_raw) && m_raw != 0) && (ev[seg]      ./= m_raw)  # abs → fractional
        end
    end

    # Keep the full-length, input-aligned detrended flux for the caller so
    # it folds against the same `t` they passed in (e.g. `plot_transit_
    # phasefold`) without needing the saved LC. `flux_dt[ok]` below
    # allocates a fresh filtered array, so this alias stays full-length.
    flux_detrended_full = flux_dt
    # Leveled, full-length fractional errors aligned 1:1 with
    # `flux_detrended_full` (same kept-space), so phase-fold plots can draw
    # per-point photometric error bars on the folded data.
    flux_err_detrended_full = ev

    # Drop cadences left non-finite by segment boundaries / GP gaps
    ok = isfinite.(flux_dt) .& isfinite.(ev)
    if !all(ok)
        tv      = tv[ok]
        flux_dt = flux_dt[ok]
        ev      = ev[ok]
    end

    # ---- Periodogram ---------------------------------------------------
    # Two estimators share the (P, depth, t0, score, dur) tuple convention
    # so the dedup pass below stays method-agnostic.
    peak_P, peak_depth, peak_t0, peak_score, peak_dur = if method === :bls
        # Uniform-in-frequency grid, resolution sized to the baseline
        # (Ofir 2014). A transit's phase must not drift more than a fraction
        # 1/oversample of the shortest searched duty cycle over the full
        # baseline T:  df ≤ min(durations) / (oversample · T). A fixed
        # linear-in-P grid silently under-resolves long baselines — the true
        # period falls between grid samples, smears, and loses to its
        # shorter-period aliases / harmonics (e.g. multi-sector TESS).
        # `n_periods` acts as a floor (keeps short baselines well sampled).
        baseline = maximum(tv) - minimum(tv)
        f_lo, f_hi = 1.0 / period_max, 1.0 / period_min
        df_opt = minimum(durations) / (oversample * max(baseline, eps()))
        n_opt  = df_opt > 0 ? ceil(Int, (f_hi - f_lo) / df_opt) : n_periods
        n_opt > max_n_periods && @warn(
            "find_transits: optimal BLS grid ($n_opt freqs) exceeds " *
            "max_n_periods=$max_n_periods; capping. Long-period resolution " *
            "may be degraded — raise max_n_periods or narrow [period_min, " *
            "period_max].")
        n_freq  = clamp(max(n_periods, n_opt), 2, max_n_periods)
        fgrid   = range(f_lo, f_hi; length = n_freq)
        periods = collect(1.0 ./ fgrid)              # descending P, uniform in f
        df      = step(fgrid)
        # peak half-width in frequency ≈ 1/baseline → guard ≈ 1/(T·df) samples
        peak_sp = max(3, round(Int, 1.0 / (baseline * df)))
        _box_least_squares(tv, flux_dt, ev, periods;
                            durations = durations,
                            n_phase_bins = n_phase_bins,
                            n_peaks = n_peaks,
                            threaded = true,
                            min_peak_spacing = peak_sp,
                            rho_star = scale_durations ?
                                       Float64(M_s) / Float64(R_s)^3 : nothing,
                            flank_frac = local_baseline ? Float64(flank_frac) : 0.0)
    else
        # TLS (Hippke & Heller 2019) — physically-modelled limb-darkened
        # transit template, vastly more sensitive than BLS at the cost
        # of ~10× wall-clock. Returns a single best fit + the full SDE
        # spectrum; we slice the top `n_peaks` SDE points and let the
        # downstream harmonic dedup do its job.
        _run_tls(tv, flux_dt, ev;
                  period_min = period_min, period_max = period_max,
                  R_s = R_s, M_s = M_s, n_peaks = n_peaks,
                  tls_options...)
    end

    # ---- Threshold + intra-self harmonic dedup ------------------------
    # `_harmonic_dedup` checks peaks against *external* known periods.
    # Here we want to drop, within the BLS output, any lower-scoring
    # peak that is an m/n harmonic of a higher-scoring one.
    order = sortperm(peak_score; rev = true)
    keep_idx = Int[]
    @inbounds for i in order
        peak_score[i] >= score_threshold || break
        Pp = peak_P[i]
        clash = false
        for k in keep_idx
            Pk = peak_P[k]
            for m in 1:harmonic_max_order, n in 1:harmonic_max_order
                target = m * Pk / n
                if abs(Pp - target) / target < harmonic_tol
                    clash = true; break
                end
            end
            clash && break
        end
        clash || push!(keep_idx, i)
        length(keep_idx) >= max_candidates && break
    end

    P_out   = peak_P[keep_idx]
    # BLS returns t0 as (transit-center phase)×P ∈ [0,P), referenced to
    # absolute time zero — a phase, not an epoch. Convert to a real
    # mid-transit time in the SAME time system as `t`, anchored to the
    # transit nearest the baseline midpoint (the best-constrained epoch).
    t_mid  = (tv[1] + tv[end]) / 2
    t0_out = Float64[t0p + round((t_mid - t0p) / Pp) * Pp
                     for (t0p, Pp) in zip(peak_t0[keep_idx], P_out)]
    d_out   = peak_depth[keep_idx]
    sc_out  = peak_score[keep_idx]
    dur_out = peak_dur[keep_idx]          # best-fit transit duration [days]

    # ---- Physical estimates -------------------------------------------
    R_p_E = [_radius_from_depth(d, Float64(R_s)) for d in d_out]
    M_p_E = [_mass_radius_chenkipping(r) for r in R_p_E]
    K_ms  = [_K_from_mass(M_p_E[i], P_out[i], Float64(M_s))
              for i in eachindex(P_out)]

    return (periods   = P_out,
             t0s       = t0_out,
             depths    = d_out,
             durations = dur_out,
             snr       = sc_out,
             R_p_earth = R_p_E,
             M_p_earth = M_p_E,
             K_ms      = K_ms,
             detrend_result = dt_result,
             flux_detrended = flux_detrended_full,
             flux_err_detrended = flux_err_detrended_full,
             kept           = kept)
end
