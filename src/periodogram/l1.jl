# L1 (LASSO-based) sparse periodogram wrapper (Hara+ 2017). Backed by
# the L1Periodograms.jl package (jvines/L1Periodograms.jl).
#
# Note: per-frequency amplitudes are the y-axis quantity. Most entries
# are exactly zero (that's the sparsity guarantee) so the plot is a
# stick spectrum — peaks are unambiguous. There's no analytic FAP.

import L1Periodograms as _L1

"""
    l1_periodogram_compute(t, y, err; period_min, period_max, n_freqs,
                            samples_per_peak, alpha=0.01, max_iter=1000,
                            tol=1e-6, max_peaks=5) -> L1Pgram

`alpha` is the LASSO regularisation strength as a fraction of `λ_max`
(the smallest λ that zeroes every penalised coefficient). Smaller α
keeps more peaks; larger α is sparser.
"""
function l1_periodogram_compute(t::AbstractVector{<:Real},
                                 y::AbstractVector{<:Real},
                                 err::AbstractVector{<:Real};
                                 period_min::Real = 1.0,
                                 period_max::Union{Nothing, Real} = nothing,
                                 n_freqs::Union{Nothing, Int} = nothing,
                                 samples_per_peak::Int = 50,
                                 alpha::Real = 0.01,
                                 max_iter::Int = 1000,
                                 tol::Real = 1e-6,
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
    res = _L1.l1_periodogram(
        collect(Float64, t),
        collect(Float64, y),
        collect(Float64, err),
        freqs;
        alpha = Float64(alpha),
        max_iter = Int(max_iter),
        tol = Float64(tol),
    )
    amps = _L1.amplitudes(res)
    P = 1.0 ./ freqs

    peaks = extract_peaks(freqs, P, amps; max_peaks = max_peaks)
    return L1Pgram(freqs, P, amps, amps, res.converged, res.iterations,
                    length(t), peaks)
end
