# transit_window.jl — keep only NEAR-TRANSIT photometry, at native cadence, to
# speed transit-model likelihoods on long / high-cadence light curves.
#
# This is MASKING, not binning: every kept cadence is untouched at its native
# sampling (so it honours the "never bin the LC" rule). The justification is
# purely about information content — on an already-detrended LC the out-of-transit
# baseline is flat and carries NO transit signal, so it only constrains the
# photometric jitter/zero-point. Dropping it preserves the full transit SNR
# (every in-transit cadence is kept) while shrinking the photometry block by the
# transit duty cycle (often 10–100×), since the transit likelihood is O(N_phot).
#
# Typical use:
#   tr = find_transits(t, flux, flux_err; ...)              # BLS/TLS ephemerides
#   w  = window_to_transits(t, flux, flux_err, tr; pad=3)   # or pass P/t0/dur
#   data = Data(; ..., t_phot=w.t, flux=w.flux, flux_err=w.flux_err)
# For a known transiting planet, pass its literature (P, t0, duration) directly.

"""
    transit_mask(t, periods, t0s, durations; pad=3.0) -> BitVector
    transit_mask(t, transits::NamedTuple; pad=3.0)

`true` where time `t[i]` falls within ±`pad`·`duration` of any transit centre
`t0 + n·P`, unioned over all planets in `(periods, t0s, durations)` (each in the
same time units as `t`). The `NamedTuple` form accepts a [`find_transits`](@ref)
result directly (`.periods`, `.t0s`, `.durations`). With no ephemerides the mask
is all-`true` (no-op). Native cadence — this is masking, NOT binning.
"""
function transit_mask(t::AbstractVector{<:Real},
                       periods::AbstractVector{<:Real},
                       t0s::AbstractVector{<:Real},
                       durations::AbstractVector{<:Real};
                       pad::Real = 3.0)
    (length(periods) == length(t0s) == length(durations)) ||
        throw(ArgumentError("periods, t0s, durations must be equal length"))
    pad > 0 || throw(ArgumentError("pad must be > 0"))
    isempty(periods) && return trues(length(t))
    keep = falses(length(t))
    @inbounds for (P, t0, dur) in zip(periods, t0s, durations)
        (P > 0 && dur > 0) || continue
        half = pad * dur
        for i in eachindex(t)
            keep[i] && continue
            φ = mod(t[i] - t0 + 0.5P, P) - 0.5P   # signed offset from nearest transit
            abs(φ) ≤ half && (keep[i] = true)
        end
    end
    keep
end
transit_mask(t::AbstractVector{<:Real}, tr::NamedTuple; pad::Real = 3.0) =
    transit_mask(t, tr.periods, tr.t0s, tr.durations; pad = pad)

"""
    window_to_transits(t, flux, flux_err, periods, t0s, durations; pad=3.0)
    window_to_transits(t, flux, flux_err, transits::NamedTuple; pad=3.0)
        -> (t=, flux=, flux_err=, mask=)

Subset photometry to near-transit windows (see [`transit_mask`](@ref)) and return
the masked `(t, flux, flux_err)` plus the `BitVector` `mask`. Drops the flat,
already-detrended out-of-transit baseline to make the transit likelihood ~duty-
cycle× cheaper WITHOUT binning — in-transit cadences (the only ones carrying
transit signal) are kept intact. Feed the result straight into
`Data(; t_phot=…, flux=…, flux_err=…)`.
"""
function window_to_transits(t::AbstractVector{<:Real}, flux::AbstractVector{<:Real},
                             flux_err::AbstractVector{<:Real},
                             periods::AbstractVector{<:Real},
                             t0s::AbstractVector{<:Real},
                             durations::AbstractVector{<:Real};
                             pad::Real = 3.0)
    m = transit_mask(t, periods, t0s, durations; pad = pad)
    (t = t[m], flux = flux[m], flux_err = flux_err[m], mask = m)
end
window_to_transits(t::AbstractVector{<:Real}, flux::AbstractVector{<:Real},
                    flux_err::AbstractVector{<:Real}, tr::NamedTuple; pad::Real = 3.0) =
    window_to_transits(t, flux, flux_err, tr.periods, tr.t0s, tr.durations; pad = pad)
