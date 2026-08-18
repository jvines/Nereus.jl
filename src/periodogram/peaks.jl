# Peak-extraction and harmonic-dedup helpers shared by every
# periodogram type. The same algorithm as `find_transits`'s intra-self
# dedup: take the top peaks by power, drop any that are m·P/n of a
# higher-scoring peak.

"""
    _local_maxima(p) -> Vector{Int}

Return the indices of strict local maxima of `p` (interior + boundaries
treated as edges). Used as the candidate pool for `extract_peaks`.
"""
function _local_maxima(p::AbstractVector{<:Real})
    n = length(p)
    n < 3 && return collect(1:n)
    idx = Int[]
    @inbounds for i in 2:(n - 1)
        if p[i] > p[i - 1] && p[i] >= p[i + 1]
            push!(idx, i)
        end
    end
    return idx
end

"""
    extract_peaks(freqs, periods, power; max_peaks=5,
                   harmonic_tol=0.03, harmonic_max_order=5,
                   fap_func=nothing) -> Vector{PgramPeak}

Find the top-`max_peaks` peaks in `power(freqs)` after:
  1. Restricting to strict local maxima (drops adjacent-grid duplicates
     within a single broad peak).
  2. Harmonic dedup — a candidate is rejected if it matches an
     already-kept peak at `m·P/n` within `harmonic_tol` (fractional).

`fap_func(power_value) -> fap_in_[0,1]` is called per accepted peak;
pass `nothing` for methods without an analytic FAP.
"""
function extract_peaks(freqs::AbstractVector{<:Real},
                        periods::AbstractVector{<:Real},
                        power::AbstractVector{<:Real};
                        max_peaks::Int = 5,
                        harmonic_tol::Float64 = 0.03,
                        harmonic_max_order::Int = 5,
                        min_log_sep::Float64 = 0.0,
                        fap_func::Union{Nothing, Function} = nothing)
    n = length(power)
    n > 0 || return PgramPeak[]
    # Candidate pool: strict local maxima only. Falls back to a
    # global-argmax single peak if the spectrum is monotonic.
    cand = _local_maxima(power)
    isempty(cand) && (cand = [argmax(power)])
    order = sort(cand; by = i -> power[i], rev = true)

    chosen = Int[]
    @inbounds for idx in order
        isfinite(power[idx]) && power[idx] > 0 || continue
        P_new = periods[idx]
        clash = false
        for c in chosen
            # Collapse an alias "forest": reject a candidate within
            # `min_log_sep` dex (in frequency) of an already-chosen, higher
            # peak so a dense comb of neighbouring spikes yields only its
            # tallest member (THE peak). `min_log_sep=0` disables (the
            # find_rv_planets default — unchanged behaviour).
            if min_log_sep > 0 &&
               abs(log10(freqs[idx]) - log10(freqs[c])) < min_log_sep
                clash = true; break
            end
            P_c = periods[c]
            # Reject only TRUE harmonics — integer sub-harmonic aliases (j·P_c)
            # and integer harmonics (P_c/j). The old check used every m/k ratio
            # with m,k∈1..n, which also collapses mean-motion RESONANCES (3:2,
            # 5:2, …) — those are genuinely DISTINCT planets, not aliases of one
            # signal, so deduping them silently drops real resonant systems
            # (validated: a P=31.6 planet at 5:2 with a P=12.3 planet vanished).
            for j in 2:harmonic_max_order
                if abs(P_new - j * P_c) / (j * P_c) < harmonic_tol ||
                   abs(P_new - P_c / j) / (P_c / j) < harmonic_tol
                    clash = true; break
                end
            end
            clash && break
        end
        clash || push!(chosen, idx)
        length(chosen) >= max_peaks && break
    end
    out = PgramPeak[]
    for c in chosen
        fap = fap_func === nothing ? NaN : fap_func(power[c])
        push!(out, PgramPeak(periods[c], freqs[c], power[c], fap))
    end
    return out
end
