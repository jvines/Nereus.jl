# GLS wrapper around LombScargle.jl.
#
# FAP thresholds come from LombScargle.jl's `fapinv`, via a BOOTSTRAP by
# default (`fap_method = :bootstrap`) with an `:analytic` opt-in.
#
# There used to be a homebrew Cumming-2004 inversion here. It was removed
# because it used the Horne & Baliunas independent-frequency count M — a
# function of N only, ignoring the searched band — which came out ~10x
# over-confident: a measured false-alarm rate of 0.46 at a nominal 0.05.
# Do not reintroduce it.

# Qualified import only — keeps `power`/`freq` from polluting Nereus's
# top-level namespace where we define our own `power(::Periodogram)`.
import LombScargle as _LS

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

# Frequency grid: uniform between 1/period_max (default: the baseline) and
# 1/period_min, `n_freqs` long or `samples_per_peak` per peak width 1/T.
function _gls_grid(t, period_min::Real, period_max, n_freqs, samples_per_peak::Int)
    isfinite(period_min) && period_min > 0 ||
        throw(ArgumentError("period_min must be > 0"))
    T = Float64(maximum(t) - minimum(t))
    T > 0 || throw(ArgumentError("time baseline must be > 0"))
    Pmax = period_max === nothing ? T : Float64(period_max)
    Pmax > period_min ||
        throw(ArgumentError("period_max must be > period_min"))
    f_min = 1.0 / Pmax
    f_max = 1.0 / period_min
    nf = n_freqs !== nothing ? Int(n_freqs) :
         max(64, ceil(Int, (f_max - f_min) * T * samples_per_peak))
    return collect(range(f_min, f_max; length = nf))
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
    freqs = _gls_grid(t, period_min, period_max, n_freqs, samples_per_peak)
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

# The spectral window of the sampling, W(f) = |Σⱼ exp(2πi f tⱼ)|² / N² (Roberts+
# 1987), on the grid `gls_periodogram` builds from the same arguments. GLS cannot
# give it: GLS fits and removes a mean, so a constant series has no power left
# (≈1e-11 everywhere). A window has no false-alarm probability, so the thresholds
# and the peaks' `fap` are NaN.
function _spectral_window_pgram(t::AbstractVector{<:Real};
                                period_min::Real = 1.0,
                                period_max::Union{Nothing, Real} = nothing,
                                n_freqs::Union{Nothing, Int} = nothing,
                                samples_per_peak::Int = 50,
                                fap_levels::Vector{<:Real} = [0.1, 0.05, 0.01, 0.001],
                                max_peaks::Int = 5,
                                min_log_sep::Real = 0.0)
    freqs = _gls_grid(t, period_min, period_max, n_freqs, samples_per_peak)
    pwr = _window_function(collect(Float64, t), freqs)
    P = 1.0 ./ freqs
    peaks = extract_peaks(freqs, P, pwr; max_peaks = max_peaks,
                          min_log_sep = Float64(min_log_sep))
    return GLSPgram(freqs, P, pwr, collect(Float64, fap_levels),
                    fill(NaN, length(fap_levels)), length(t), peaks)
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
    window = window_def === :gls_tt ?
             gls_periodogram(tv, copy(tv), ones(length(tv)); common...) :
             _spectral_window_pgram(tv; common...)

    return GLSSectorSet(names, sec_pg, stitched, window, length(tv), T)
end
