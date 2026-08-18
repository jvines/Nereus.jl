# BGLS / SBGLS wrappers (Mortier+ 2015, Mortier & Collier Cameron 2017).
# Backed by the BGLS.jl package (jvines/BayesianGLS.jl). The raw output
# is a log-probability spectrum — we expose both the raw `log_probability`
# and the shifted `power = exp(logp - max(logp)) ∈ (0, 1]` so plots have
# a familiar 0..1 vertical scale.

import BayesianGLS as _BGLS

"""
    bgls_periodogram(t, y, err; period_min, period_max, n_freqs,
                      samples_per_peak, max_peaks=5) -> BGLSPgram
"""
function bgls_periodogram(t::AbstractVector{<:Real},
                           y::AbstractVector{<:Real},
                           err::AbstractVector{<:Real};
                           period_min::Real = 1.0,
                           period_max::Union{Nothing, Real} = nothing,
                           n_freqs::Union{Nothing, Int} = nothing,
                           samples_per_peak::Int = 50,
                           max_peaks::Int = 5,
                           kwargs...)
    T = Float64(maximum(t) - minimum(t))
    Pmax = period_max === nothing ? T : Float64(period_max)
    f_min = 1.0 / Pmax
    f_max = 1.0 / period_min
    nf = n_freqs === nothing ?
        max(64, ceil(Int, (f_max - f_min) * T * samples_per_peak)) :
        Int(n_freqs)
    freqs = collect(range(f_min, f_max; length = nf))
    res = _BGLS.bgls(collect(Float64, t),
                      collect(Float64, y),
                      collect(Float64, err),
                      freqs)
    logp = _BGLS.power(res)
    lp_max = maximum(filter(isfinite, logp))
    pwr = exp.(logp .- lp_max)   # 0..1
    P = 1.0 ./ freqs

    peaks = extract_peaks(freqs, P, pwr; max_peaks = max_peaks)
    return BGLSPgram(freqs, P, logp, pwr, length(t), peaks)
end

"""
    sbgls_periodogram(t, y, err; period_min, period_max, n_freqs,
                       samples_per_peak, n_min=10, max_peaks=5)
        -> SBGLSPgram

Stacked BGLS — runs BGLS on the first `k` observations for `k = n_min …
N`. The 2D `matrix[k, f]` is row-normalised to `[0, 1]`. Real planets
brighten monotonically; aliases/activity wax and wane.
"""
function sbgls_periodogram(t::AbstractVector{<:Real},
                            y::AbstractVector{<:Real},
                            err::AbstractVector{<:Real};
                            period_min::Real = 1.0,
                            period_max::Union{Nothing, Real} = nothing,
                            n_freqs::Union{Nothing, Int} = nothing,
                            samples_per_peak::Int = 50,
                            n_min::Int = 10,
                            max_peaks::Int = 5,
                            kwargs...)
    T = Float64(maximum(t) - minimum(t))
    Pmax = period_max === nothing ? T : Float64(period_max)
    f_min = 1.0 / Pmax
    f_max = 1.0 / period_min
    nf = n_freqs === nothing ?
        max(64, ceil(Int, (f_max - f_min) * T * samples_per_peak)) :
        Int(n_freqs)
    freqs = collect(range(f_min, f_max; length = nf))
    res = _BGLS.sbgls(collect(Float64, t),
                       collect(Float64, y),
                       collect(Float64, err),
                       freqs; n_min = n_min)
    P = 1.0 ./ freqs

    # Peaks from the final (full-data) row
    final_row = vec(res.matrix[end, :])
    peaks = extract_peaks(freqs, P, final_row; max_peaks = max_peaks)
    return SBGLSPgram(freqs, P, res.obs_times, res.matrix,
                       length(t), peaks)
end
