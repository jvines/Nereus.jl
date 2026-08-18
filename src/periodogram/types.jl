# Periodogram result types — common surface across GLS / BGLS / SBGLS /
# L1 / TLS so downstream plotting and peak-extraction code stays
# method-agnostic.
#
# All concrete types expose:
#   .frequencies :: Vector{Float64}    [1/day]
#   .periods     :: Vector{Float64}    [day]
#   .power_array(p) -> Vector{Float64}    (the y-axis quantity for plotting)
#   .peaks       :: Vector{Periodogram peak namedtuple}
#
# Method-specific extras (FAP thresholds for GLS, convergence flags
# for L1, the 2D matrix for SBGLS, etc.) live on the concrete type.

abstract type Periodogram end

"A single periodogram peak. Period in days, power in the method's
native units, fap in [0, 1] (NaN for methods without an analytic FAP)."
struct PgramPeak
    period::Float64
    frequency::Float64
    power::Float64
    fap::Float64
end

"""
    GLSPgram

Generalized Lomb-Scargle periodogram (Zechmeister & Kürster 2009).
FAP thresholds via the Cumming 2004 analytic estimate.
"""
struct GLSPgram <: Periodogram
    frequencies::Vector{Float64}
    periods::Vector{Float64}
    power::Vector{Float64}
    fap_levels::Vector{Float64}
    fap_thresholds::Vector{Float64}
    n_obs::Int
    peaks::Vector{PgramPeak}
end

"""
    BGLSPgram

Bayesian Generalized Lomb-Scargle log-probability spectrum
(Mortier+ 2015). `power` here is `exp(logp - max(logp))` ∈ (0, 1] so
the plot reads like a probability density. The raw log-probability is
in `log_probability`.
"""
struct BGLSPgram <: Periodogram
    frequencies::Vector{Float64}
    periods::Vector{Float64}
    log_probability::Vector{Float64}
    power::Vector{Float64}   # exp(logp - max(logp))
    n_obs::Int
    peaks::Vector{PgramPeak}
end

"""
    SBGLSPgram

Stacked Bayesian GLS (Mortier & Collier Cameron 2017). 2D map of
row-normalized log-probability vs frequency, where each row is BGLS on
the first `k` observations (`k = n_min, n_min+1, …, N`). Real planets
brighten monotonically as data accumulates; aliases/activity wax and
wane.
"""
struct SBGLSPgram <: Periodogram
    frequencies::Vector{Float64}
    periods::Vector{Float64}
    obs_times::Vector{Float64}
    matrix::Matrix{Float64}
    n_obs::Int
    peaks::Vector{PgramPeak}    # peaks at the *final* row (full data)
end

"""
    L1Pgram

L1 (LASSO-based) sparse periodogram (Hara+ 2017). `power` is the
amplitude of each frequency's sin/cos pair in the LASSO solution; most
entries are exactly 0. `converged` / `iterations` carry the coordinate-
descent state.
"""
struct L1Pgram <: Periodogram
    frequencies::Vector{Float64}
    periods::Vector{Float64}
    amplitudes::Vector{Float64}
    power::Vector{Float64}   # alias of amplitudes for plot symmetry
    converged::Bool
    iterations::Int
    n_obs::Int
    peaks::Vector{PgramPeak}
end

"""
    PgramSet

The full output of `find_rv_planets`: one periodogram per channel
(RV + indicators + window function) all computed on the same frequency
grid. `method` records which estimator was used (`:gls`, `:bgls`,
`:sbgls`, `:l1`).
"""
struct PgramSet
    method::Symbol
    rv::Periodogram
    indicators::Vector{Pair{String, <:Periodogram}}   # ordered
    window::Periodogram
    n_obs::Int
    baseline_days::Float64
end

"""
    GLSSectorSet

Photometric rotation GLS computed per sector plus on the stitched series
(Lomb-Scargle handles the inter-sector gaps natively, so the stitched
panel is valid and gives the finest frequency resolution). All panels —
each sector, the stitch, and the sampling window — share one frequency
grid so they stack with a common x-axis. Output of `gls_rotation`.
"""
struct GLSSectorSet
    names::Vector{String}         # per-sector labels, ordered
    sectors::Vector{GLSPgram}     # per-sector GLS (shared freq grid)
    stitched::GLSPgram            # GLS of the concatenated series
    window::GLSPgram              # sampling window of the stitched times
    n_obs::Int
    baseline_days::Float64
end

# --- Common accessors --------------------------------------------------

frequencies(p::Periodogram) = p.frequencies
periods(p::Periodogram)     = p.periods

"Per-method y-axis quantity used by the plot stack."
power(p::GLSPgram)   = p.power
power(p::BGLSPgram)  = p.power
power(p::L1Pgram)    = p.amplitudes
power(p::SBGLSPgram) = vec(p.matrix[end, :])  # last (full-data) row

best_period(p::Periodogram) = isempty(p.peaks) ? NaN : p.peaks[1].period
best_power(p::Periodogram)  = isempty(p.peaks) ? NaN : p.peaks[1].power

# A short type tag used in axis labels / file names
method_tag(::GLSPgram)   = "gls"
method_tag(::BGLSPgram)  = "bgls"
method_tag(::SBGLSPgram) = "sbgls"
method_tag(::L1Pgram)    = "l1"
