# Stellar rotation period from photometry — autocorrelation function
# (McQuillan+ 2013, 2014) plus partial autocorrelation for AR-noise
# disambiguation.
#
# Usage as a pre-step for `detrend_notch`:
#
#     P_rot = find_rotation_period(t, flux, σ).P_rot
#     dt    = detrend_notch(t, flux, σ; window = 1.2 * P_rot)
#     blsh  = find_transits(t, flux, σ; detrend = :none)

using Statistics: median, std, mean
using FFTW: rfft, irfft

# ----------------------------------------------------------------------
# Top-level API
# ----------------------------------------------------------------------

"""
    find_rotation_period(t, flux, flux_err; …) -> NamedTuple

Estimate stellar rotation period from a light curve via the
autocorrelation function (McQuillan+ 2013, 2014; Aigrain+ 2015).

# Keyword arguments
- `min_period = 0.1`, `max_period = nothing` — search window [days].
  `max_period` defaults to ¼ of the LC baseline.
- `smooth_sigma = nothing` — Gaussian kernel width [days] applied to
  the ACF before peak-finding. Defaults to one median cadence.
- `bin_cadence = nothing` — uniform-cadence bin width [days]. `nothing`
  uses the median observed cadence (correct for Kepler/TESS LCs).
- `method = :acf` — `:acf`, `:pacf`, or `:both` (computes both, peak-
  finding always done on the ACF).
- `n_peaks = 5` — number of ACF peaks to return for diagnostics. The
  first one is the rotation period; subsequent peaks should fall at
  2·P_rot, 3·P_rot, … for a coherent rotator.

# Returns

`NamedTuple` with fields:
- `P_rot::Float64` — best rotation period [days]
- `P_rot_err::Float64` — half-width at half-max of the peak [days]
- `lags::Vector{Float64}` — ACF lag grid [days]
- `acf::Vector{Float64}` — autocorrelation
- `pacf::Union{Nothing, Vector{Float64}}` — partial autocorrelation
  (when `method ∈ (:pacf, :both)`)
- `peaks::Vector{NamedTuple}` — top `n_peaks` peaks of the ACF
- `bin_cadence::Float64` — uniform cadence used [days]
- `n_bins::Int` — length of the binned series
"""
function find_rotation_period(t::AbstractVector{<:Real},
                                flux::AbstractVector{<:Real},
                                flux_err::AbstractVector{<:Real};
                                min_period::Real = 0.1,
                                max_period::Union{Nothing, Real} = nothing,
                                smooth_sigma::Union{Nothing, Real} = nothing,
                                bin_cadence::Union{Nothing, Real} = nothing,
                                method::Symbol = :acf,
                                n_peaks::Int = 5)
    length(t) == length(flux) == length(flux_err) ||
        throw(ArgumentError("t / flux / flux_err length mismatch"))
    method in (:acf, :pacf, :both) ||
        throw(ArgumentError("method must be :acf | :pacf | :both"))
    issorted(t) || throw(ArgumentError("`t` must be ascending"))

    tv  = collect(Float64, t)
    fv  = collect(Float64, flux)
    ev  = collect(Float64, flux_err)
    T   = tv[end] - tv[1]
    T > 0 || throw(ArgumentError("zero baseline"))

    # ---- Resample to uniform cadence -----------------------------------
    # Default to a COARSE grid (~ max_period/100), not the raw cadence. The
    # ACF/PACF need uniform lags and are a rotation-scale diagnostic; at TESS
    # 20-s cadence a native grid means ~10⁵ lags (unstemmable, and the PACF's
    # only non-zero term sits at lag 1 = 20 s, off-screen). Coarse binning is
    # the McQuillan/statsmodels-standard and leaves rotation (≫ bin) intact.
    # This is the diagnostic's internal grid only — not the science LC.
    Pmax_bin = max_period === nothing ? T / 4 : Float64(max_period)
    dt = bin_cadence === nothing ?
         max(median(diff(tv)), Pmax_bin / 100) : Float64(bin_cadence)
    dt > 0 || throw(ArgumentError("bin_cadence must be > 0"))

    t_lo, t_hi = tv[1], tv[end]
    n_bins = max(8, Int(round((t_hi - t_lo) / dt)) + 1)
    binned = _bin_lc(tv, fv, ev, t_lo, dt, n_bins)
    # Mean-subtract the binned series for autocovariance
    finite = isfinite.(binned)
    if !all(finite)
        # Linear interpolation across gaps (NaN bins)
        binned = _fill_nans_linear!(binned)
    end
    μ = mean(binned)
    y = binned .- μ

    # ---- ACF via FFT (Wiener-Khinchin) ---------------------------------
    n_pad = nextpow(2, 2 * length(y))
    pad   = zeros(Float64, n_pad)
    pad[1:length(y)] = y
    F = rfft(pad)
    @inbounds for k in eachindex(F)
        F[k] = F[k] * conj(F[k])
    end
    autocov = irfft(F, n_pad)[1:length(y)]
    autocov ./= autocov[1]       # ACF normalised to 1 at lag 0
    acf = autocov

    # ---- Gaussian smoothing -------------------------------------------
    # Default scales with the rotation search range (≈ Pmax/100), not the
    # raw cadence: at TESS 20-s cadence a one-cadence kernel does nothing,
    # leaving sub-resolution wiggles on the high-autocorrelation shoulder
    # that masquerade as peaks. Smoothing on the rotation scale removes them.
    Pmax_d = max_period === nothing ? T / 4 : Float64(max_period)
    σ = smooth_sigma === nothing ? max(dt, Pmax_d / 100) : Float64(smooth_sigma)
    if σ > 0
        sigma_bins = max(1.0, σ / dt)
        acf = _gaussian_smooth(acf, sigma_bins)
    end

    lags = collect(0:length(acf) - 1) .* dt

    # ---- Peak detection ------------------------------------------------
    lag_min_idx = max(2, Int(round(min_period / dt)))
    lag_max_idx = min(length(acf) - 1, Int(round(Pmax_d / dt)))
    lag_max_idx > lag_min_idx ||
        throw(ArgumentError("min_period $min_period >= max_period $Pmax_d"))

    peak_idx = Int[]
    @inbounds for i in lag_min_idx:lag_max_idx
        if acf[i] > acf[i-1] && acf[i] > acf[i+1] && acf[i] > 0
            push!(peak_idx, i)
        end
    end

    bg_med = median(view(acf, lag_min_idx:lag_max_idx))
    bg_std = std(view(acf, lag_min_idx:lag_max_idx))
    threshold = bg_med + 2 * bg_std

    # Skip the high-autocorrelation shoulder near lag 0: the ACF decays
    # from ~1 and carries sub-resolution wiggles there. McQuillan+ 2013
    # take the rotation peak as the first *significant peak after the ACF
    # first dips*. Require the first minimum to fall BELOW the background
    # (acf < bg_med) so shoulder wiggles — which sit at high ACF — are
    # skipped regardless of residual smoothing.
    first_min_idx = lag_min_idx
    @inbounds for i in (lag_min_idx + 1):(lag_max_idx - 1)
        if acf[i] < acf[i-1] && acf[i] < acf[i+1] && acf[i] < bg_med
            first_min_idx = i; break
        end
    end

    post = filter(i -> i > first_min_idx, peak_idx)
    sig  = filter(i -> acf[i] > threshold, post)
    # Prefer significant post-dip peaks; fall back to any post-dip peak.
    candidates = isempty(sig) ? post : sig
    if isempty(candidates)
        P_rot = NaN; P_rot_err = NaN
        peaks_out = NamedTuple[]
    else
        # Dominant (tallest-ACF) post-dip peak is the rotation period: the
        # biased ACF estimator decays with lag, so the fundamental P_rot
        # peak is the tallest and its 2P, 3P harmonics follow lower.
        first_idx = candidates[argmax(view(acf, candidates))]
        P_rot = lags[first_idx]
        # Half-width at half max for an error estimate.
        half = (acf[first_idx] + bg_med) / 2
        lo = first_idx
        while lo > lag_min_idx && acf[lo] > half; lo -= 1; end
        hi = first_idx
        while hi < lag_max_idx && acf[hi] > half; hi += 1; end
        P_rot_err = 0.5 * (lags[hi] - lags[lo])
        # Top N post-dip peaks (amplitude-sorted) for caller diagnostics;
        # for a coherent rotator they fall at P_rot, 2·P_rot, 3·P_rot, …
        order = sort(post; by = i -> acf[i], rev = true)
        npk = min(n_peaks, length(order))
        peaks_out = [(lag = lags[order[i]],
                       value = acf[order[i]]) for i in 1:npk]
    end

    # ---- PACF via Durbin-Levinson --------------------------------------
    pacf = nothing
    if method in (:pacf, :both)
        # Limit to lag_max_idx to keep compute O(L²) bounded.
        L = min(length(autocov) - 1, lag_max_idx)
        pacf = _pacf_durbin_levinson(view(autocov, 1:L+1))
    end

    return (P_rot = P_rot, P_rot_err = P_rot_err,
             lags = lags, acf = acf, pacf = pacf,
             peaks = peaks_out, bin_cadence = dt, n_bins = n_bins,
             threshold = isnan(P_rot) ? bg_med : threshold)
end


# ----------------------------------------------------------------------
# Per-sector wrapper (no stitched ACF — the ACF can't be combined across
# gaps without injecting an interpolation artifact, so each sector is
# analysed independently)
# ----------------------------------------------------------------------

"""
    find_rotation_period_sectors(t, flux, flux_err; sector_id=nothing,
        sector_names=nothing, gap_size=1.0, kwargs...) -> NamedTuple

Run [`find_rotation_period`](@ref) **independently on each sector** and
return the per-sector results. There is deliberately no stitched/combined
ACF: the autocorrelation requires uniform sampling, and concatenating
sectors across multi-day (or multi-year) gaps would force a linear
interpolation that dominates the ACF. Analyse per sector instead.

Sectors are defined by `sector_id` (per-cadence labels, same length as
`t`) when given; otherwise the LC is split into contiguous segments via
`find_segments` (gap `> gap_size` days — note this also splits at
intra-sector downlink gaps, so pass `sector_id` for true per-sector
grouping). `kwargs` are forwarded to `find_rotation_period`
(`min_period`, `max_period`, `method`, …); note `max_period` defaults to
¼ of each *sector's* baseline, so set it explicitly if `P_rot` may exceed
that.

Returns `(; names, results, P_rot)` where `results[i]` is the full
`find_rotation_period` NamedTuple for sector `i`, `names[i]` its label,
and `P_rot[i]` its rotation period (`NaN` if none significant).
"""
function find_rotation_period_sectors(t::AbstractVector{<:Real},
                                       flux::AbstractVector{<:Real},
                                       flux_err::AbstractVector{<:Real};
                                       sector_id::Union{Nothing, AbstractVector} = nothing,
                                       sector_names::Union{Nothing, AbstractVector} = nothing,
                                       gap_size::Real = 1.0,
                                       kwargs...)
    n = length(t)
    n == length(flux) == length(flux_err) ||
        throw(ArgumentError("t / flux / flux_err length mismatch"))
    issorted(t) || throw(ArgumentError("`t` must be ascending"))
    sector_id === nothing || length(sector_id) == n ||
        throw(ArgumentError("sector_id length $(length(sector_id)) != length(t) $n"))

    groups, labels = _sector_groups(collect(Float64, t), sector_id, gap_size)
    sector_names === nothing || length(sector_names) == length(groups) ||
        throw(ArgumentError("sector_names ($(length(sector_names))) must match " *
                            "the number of sectors ($(length(groups)))"))
    names = sector_names === nothing ? labels : string.(sector_names)

    results = Vector{NamedTuple}(undef, length(groups))
    P_rot   = Vector{Float64}(undef, length(groups))
    for (i, idx) in enumerate(groups)
        r = find_rotation_period(t[idx], flux[idx], flux_err[idx]; kwargs...)
        results[i] = r
        P_rot[i]   = r.P_rot
    end
    return (; names = names, results = results, P_rot = P_rot)
end

"Index groups per sector: by `sector_id` if given, else contiguous
`find_segments` segments (gap > `gap_size`)."
function _sector_groups(t::Vector{Float64},
                         sector_id::Union{Nothing, AbstractVector},
                         gap_size::Real)
    if sector_id !== nothing
        uniq = sort!(unique(sector_id))
        groups = [findall(==(s), sector_id) for s in uniq]
        return groups, string.(uniq)
    else
        segs = find_segments(t; gap_size = gap_size)
        groups = [collect(seg) for seg in segs]
        return groups, ["segment $i" for i in eachindex(segs)]
    end
end

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

"Bin flux onto a uniform `dt`-spaced grid via inverse-variance weighting.
Empty bins are filled with NaN and patched by `_fill_nans_linear!`."
function _bin_lc(t::Vector{Float64}, flux::Vector{Float64},
                  err::Vector{Float64}, t_lo::Float64, dt::Float64,
                  n_bins::Int)
    wsum   = zeros(Float64, n_bins)
    fsum   = zeros(Float64, n_bins)
    @inbounds for i in eachindex(t)
        b = Int(floor((t[i] - t_lo) / dt)) + 1
        (b < 1 || b > n_bins) && continue
        w = 1.0 / err[i]^2
        wsum[b] += w
        fsum[b] += w * flux[i]
    end
    out = Vector{Float64}(undef, n_bins)
    @inbounds for b in 1:n_bins
        out[b] = wsum[b] > 0 ? fsum[b] / wsum[b] : NaN
    end
    return out
end

"Linear-interpolate over NaN gaps in-place."
function _fill_nans_linear!(y::Vector{Float64})
    n = length(y)
    # Find first valid for left-edge extrapolation
    i = 1
    while i <= n && !isfinite(y[i]); i += 1; end
    i > n && return y    # all NaN
    fill_l = y[i]
    @inbounds for k in 1:i-1
        y[k] = fill_l
    end
    # Walk forward, interpolate
    while i <= n
        if isfinite(y[i])
            i += 1
        else
            j = i
            while j <= n && !isfinite(y[j]); j += 1; end
            if j > n
                @inbounds for k in i:n
                    y[k] = y[i-1]
                end
                return y
            end
            y_l = y[i-1]; y_r = y[j]
            @inbounds for k in i:j-1
                t_frac = (k - (i - 1)) / (j - (i - 1))
                y[k] = y_l + t_frac * (y_r - y_l)
            end
            i = j
        end
    end
    return y
end

"Reflect-padded Gaussian smoothing — handles edges without a step."
function _gaussian_smooth(y::Vector{Float64}, sigma::Float64)
    n = length(y)
    half = Int(ceil(3 * sigma))
    half == 0 && return copy(y)
    # Build the normalised kernel.
    kernel = Vector{Float64}(undef, 2 * half + 1)
    s2 = 2 * sigma^2
    @inbounds for k in -half:half
        kernel[k + half + 1] = exp(-k^2 / s2)
    end
    kernel ./= sum(kernel)
    out = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        acc = 0.0
        for k in -half:half
            j = i + k
            if j < 1
                j = 2 - j   # reflect at left edge
            elseif j > n
                j = 2 * n - j   # reflect at right edge
            end
            acc += kernel[k + half + 1] * y[j]
        end
        out[i] = acc
    end
    return out
end

"PACF via Durbin-Levinson recursion. `acov` is the autocovariance
sequence (need not be normalised). Returns PACF[1:L]."
function _pacf_durbin_levinson(acov::AbstractVector{<:Real})
    L = length(acov) - 1
    L <= 0 && return Float64[]
    φ = zeros(Float64, L, L)
    pacf = Vector{Float64}(undef, L)
    r0 = acov[1]
    r0 == 0 && return zeros(Float64, L)

    φ[1, 1] = acov[2] / r0
    pacf[1] = φ[1, 1]
    v = r0 * (1 - φ[1, 1]^2)
    @inbounds for k in 2:L
        sum_term = 0.0
        for j in 1:k-1
            sum_term += φ[k-1, j] * acov[k - j + 1]
        end
        φ[k, k] = (acov[k+1] - sum_term) / v
        pacf[k] = φ[k, k]
        for j in 1:k-1
            φ[k, j] = φ[k-1, j] - φ[k, k] * φ[k-1, k-j]
        end
        v *= (1 - φ[k, k]^2)
        v <= 0 && break
    end
    return pacf
end
