# GLS wrapper around LombScargle.jl.
#
# FAP thresholds via the Cumming (2004) analytic estimate:
#   FAP(z) ≈ M · exp(-z) · (1 - z)^((N-3)/2)  for the GLS power.
# We invert numerically — given a target FAP, root-find the power
# threshold. This matches what `wasp47_gls_check.jl` already does and is
# the standard for "exoplanet-paper" GLS plots.

# Qualified import only — keeps `power`/`freq` from polluting Nereus's
# top-level namespace where we define our own `power(::Periodogram)`.
import LombScargle as _LS

"""
    _cumming_fap(z, N_obs, M_indep)

Cumming 2004 analytic FAP for normalized GLS power `z` (in [0, 1]) at
`N_obs` samples and `M_indep` independent frequencies. Saturates at 1.
"""
@inline function _cumming_fap(z::Real, N_obs::Int, M_indep::Real)
    z = clamp(Float64(z), 0.0, 1.0 - 1e-12)
    p_single = (1.0 - z) ^ ((N_obs - 3) / 2)
    return clamp(1.0 - (1.0 - p_single) ^ M_indep, 0.0, 1.0)
end

"Invert Cumming 2004 numerically — find power z such that FAP(z) = fap."
function _cumming_power_for_fap(fap::Real, N_obs::Int, M_indep::Real)
    fap = clamp(Float64(fap), 1e-12, 1.0 - 1e-12)
    # Binary-search on z ∈ (0, 1)
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in 1:60
        mid = 0.5 * (lo + hi)
        f_mid = _cumming_fap(mid, N_obs, M_indep)
        if f_mid > fap
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

"""
    _gls_core(t, y, err, freqs) -> (power_vec, n_obs)

Run LombScargle.jl in `:standard` normalisation. Returns the power
spectrum on the supplied frequency grid (in `[0, 1]`).
"""
function _gls_core(t::AbstractVector{<:Real},
                    y::AbstractVector{<:Real},
                    err::AbstractVector{<:Real},
                    freqs::AbstractVector{<:Real})
    plan = _LS.plan(collect(Float64, t),
                     collect(Float64, y),
                     collect(Float64, err);
                     frequencies = collect(Float64, freqs),
                     normalization = :standard)
    pg = _LS.lombscargle(plan)
    return _LS.power(pg), length(t), pg, plan
end

"""
    gls_periodogram(t, y, err; period_min, period_max, n_freqs,
                     samples_per_peak, fap_levels=[0.1,0.05,0.01,0.001],
                     max_peaks=5) -> GLSPgram

GLS on `(t, y, err)`. Frequency grid is uniform between
`1/period_max` and `1/period_min` with at least
`samples_per_peak / baseline` resolution.
"""
function gls_periodogram(t::AbstractVector{<:Real},
                          y::AbstractVector{<:Real},
                          err::AbstractVector{<:Real};
                          period_min::Real = 1.0,
                          period_max::Union{Nothing, Real} = nothing,
                          n_freqs::Union{Nothing, Int} = nothing,
                          samples_per_peak::Int = 50,
                          fap_levels::Vector{<:Real} = [0.1, 0.05, 0.01, 0.001],
                          max_peaks::Int = 5,
                          min_log_sep::Real = 0.0,
                          fap_method::Symbol = :bootstrap,
                          n_bootstrap::Int = 1000,
                          kwargs...)
    length(t) == length(y) == length(err) ||
        throw(ArgumentError("t, y, err length mismatch"))
    isfinite(period_min) && period_min > 0 ||
        throw(ArgumentError("period_min must be > 0"))
    T = Float64(maximum(t) - minimum(t))
    T > 0 || throw(ArgumentError("time baseline must be > 0"))
    Pmax = period_max === nothing ? T : Float64(period_max)
    Pmax > period_min ||
        throw(ArgumentError("period_max must be > period_min"))

    f_min = 1.0 / Pmax
    f_max = 1.0 / period_min
    nf = if n_freqs !== nothing
        Int(n_freqs)
    else
        # samples_per_peak per intrinsic peak width Δf = 1/T
        max(64, ceil(Int, (f_max - f_min) * T * samples_per_peak))
    end
    freqs = collect(range(f_min, f_max; length = nf))
    pwr, n_obs, ls_pg, plan = _gls_core(t, y, err, freqs)
    P = 1.0 ./ freqs

    # FAP. :bootstrap (default) resamples the data n_bootstrap times and builds
    # the empirical null distribution of the periodogram MAXIMUM — exact, no
    # extreme-value assumptions, well-calibrated (validated). :analytic uses
    # LombScargle's 1−(1−prob)^M with M = T·Δf (frequency-range-aware); fast but
    # ~3× over-confident because the simple independent-frequency count
    # under-estimates the max-statistic distribution. The OLD homebrew used the
    # Horne & Baliunas M (a function of N only, ignoring the searched band),
    # which was ~10× over-confident — i.e. ~10× too many phantom detections.
    if fap_method === :bootstrap
        boot = _LS.bootstrap(n_bootstrap, plan)
        fap_func = z -> _LS.fap(boot, z)
        fap_thresh = [_LS.fapinv(boot, f) for f in fap_levels]
    else
        fap_func = z -> _LS.fap(ls_pg, z)
        fap_thresh = [_LS.fapinv(ls_pg, f) for f in fap_levels]
    end

    peaks = extract_peaks(freqs, P, pwr;
                           max_peaks = max_peaks,
                           min_log_sep = Float64(min_log_sep),
                           fap_func = fap_func)
    return GLSPgram(freqs, P, pwr, collect(Float64, fap_levels),
                     fap_thresh, n_obs, peaks)
end

"""
    gls_rotation(t, flux, flux_err; sector_id=nothing, sector_names=nothing,
                  period_min=0.5, period_max=nothing, samples_per_peak=10,
                  fap_levels=[0.1,0.05,0.01,0.001], max_peaks=5,
                  gap_size=1.0, window_def=:gls_tt) -> GLSSectorSet

Photometric rotation periodogram: Lomb-Scargle on **each sector's flux**
and on the **stitched** (concatenated, real-time) flux, plus the sampling
window. Unlike the ACF, LS needs no uniform binning and handles the
inter-sector gaps natively, so the stitched panel is valid (and has the
finest frequency resolution — the long baseline; its window carries the
gap aliases, which the per-sector panels help disambiguate).

All panels share one frequency grid (built from the stitched baseline at
`samples_per_peak` per peak width) so they stack with a common x-axis.
Sectors come from `sector_id` (per-cadence labels) or, if absent, from
`find_segments` (gap `> gap_size` d). `period_max` defaults to the
stitched baseline.
"""
function gls_rotation(t::AbstractVector{<:Real},
                       flux::AbstractVector{<:Real},
                       flux_err::AbstractVector{<:Real};
                       sector_id::Union{Nothing, AbstractVector} = nothing,
                       sector_names::Union{Nothing, AbstractVector} = nothing,
                       period_min::Real = 0.5,
                       period_max::Union{Nothing, Real} = nothing,
                       samples_per_peak::Int = 20,
                       fap_levels::Vector{<:Real} = [0.1, 0.05, 0.01, 0.001],
                       max_peaks::Int = 5,
                       min_log_sep::Real = 0.05,
                       gap_size::Real = 1.0,
                       window_def::Symbol = :gls_tt)
    tv = collect(Float64, t); fv = collect(Float64, flux); ev = collect(Float64, flux_err)
    length(tv) == length(fv) == length(ev) ||
        throw(ArgumentError("t / flux / flux_err length mismatch"))
    issorted(tv) || throw(ArgumentError("`t` must be ascending"))
    period_min > 0 || throw(ArgumentError("period_min must be > 0"))
    window_def in (:gls_tt, :unit_signal) ||
        throw(ArgumentError("window_def must be :gls_tt | :unit_signal"))
    sector_id === nothing || length(sector_id) == length(tv) ||
        throw(ArgumentError("sector_id length must match t"))

    T    = maximum(tv) - minimum(tv)
    Pmax = period_max === nothing ? T : Float64(period_max)
    Pmax > period_min || throw(ArgumentError("period_max must be > period_min"))
    # One shared grid (stitched baseline sets resolution) so all panels align.
    nf = max(64, ceil(Int, (1.0/period_min - 1.0/Pmax) * T * samples_per_peak))

    groups, labels = _sector_groups(tv, sector_id, gap_size)
    sector_names === nothing || length(sector_names) == length(groups) ||
        throw(ArgumentError("sector_names ($(length(sector_names))) must match " *
                            "the number of sectors ($(length(groups)))"))
    names = sector_names === nothing ? labels : string.(sector_names)

    # Per-sector median-normalize flux + errors so the STITCHED panel isn't
    # dominated by whichever sector has the largest raw level/amplitude.
    # (Per-sector panels are scale-invariant under :standard GLS, so this
    # only matters for the stitch — but it's cheap and correct everywhere.)
    fvn = copy(fv); evn = copy(ev)
    for idx in groups
        m = median(@view fv[idx])
        if isfinite(m) && m != 0
            @views fvn[idx] ./= m
            @views evn[idx] ./= m
        end
    end

    common = (; period_min = period_min, period_max = Pmax, n_freqs = nf,
                fap_levels = fap_levels, max_peaks = max_peaks,
                min_log_sep = min_log_sep)
    sec_pg = [gls_periodogram(tv[idx], fvn[idx], evn[idx]; common...) for idx in groups]
    stitched = gls_periodogram(tv, fvn, evn; common...)
    win_y, win_e = window_def === :gls_tt ? (copy(tv), ones(length(tv))) :
                                            (ones(length(tv)), ones(length(tv)))
    window = gls_periodogram(tv, win_y, win_e; common...)

    return GLSSectorSet(names, sec_pg, stitched, window, length(tv), T)
end
