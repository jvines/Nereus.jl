# Photometry detrending — preprocessing for noisy light curves.
#
# OPTIONAL preprocessing step. A user may instead model the stellar
# variability in-fit via a noise model (CeleriteSHO, CeleriteRotation,
# CeleriteRotationFM17, etc.). This module is for users who prefer to
# detrend first and then run a clean transit fit.
#
# Methods:
#   - Savitzky-Golay (lightkurve.flatten-style)
#   - GP detrending with SHO kernel
#   - GP detrending with celerite2 rotation kernel
#   - GP detrending with FM17 rotation kernel
#
# Transit masking: optional, given (period, T0, window).
#
# Per-segment handling: TESS data has gaps (downlinks, sectors). The
# detrender splits the LC into segments separated by gaps > `gap_size`
# days, processes each segment independently, then concatenates.

using SavitzkyGolay
using Optim
using Statistics: mean, median, std

# =====================================================================
# Transit masking
# =====================================================================

"""
    mask_transits(t, periods, T0s; window=0.025) -> Vector{Bool}

Boolean mask of length `length(t)` — `true` for in-transit points.
Multiple transit signals supported via `periods`/`T0s` vectors.

# Arguments
- `t::Vector{Float64}` — time array (days, BJD)
- `periods` — orbital periods (days). Scalar or Vector.
- `T0s` — transit centers (days, BJD). Scalar or Vector.
- `window` — half-width in phase (e.g. 0.025 = ±2.5% of period).

Phase is computed as `mod((t - T0) / P + 0.5, 1) - 0.5` so transit is
centered at phase 0.
"""
function mask_transits(t::AbstractVector{<:Real}, periods, T0s;
                        window::Real=0.025)
    Ps = periods isa Real ? [Float64(periods)] : Float64.(periods)
    T0V = T0s isa Real ? [Float64(T0s)] : Float64.(T0s)
    length(Ps) == length(T0V) || throw(ArgumentError(
        "periods and T0s must have the same length"))
    mask = falses(length(t))
    for (P, T0) in zip(Ps, T0V)
        for i in eachindex(t)
            φ = mod((t[i] - T0) / P + 0.5, 1.0) - 0.5
            if abs(φ) < window
                mask[i] = true
            end
        end
    end
    return mask
end

# =====================================================================
# Segment splitting (handle gaps)
# =====================================================================

"""
    find_segments(t; gap_size=0.5) -> Vector{UnitRange{Int}}

Split a time array into contiguous segments separated by gaps larger
than `gap_size` days. Returns ranges of indices into `t`.

For TESS data, `gap_size=0.5` (12 hours) catches sector boundaries
and downlink interruptions.
"""
function find_segments(t::AbstractVector{<:Real}; gap_size::Real=0.5)
    n = length(t)
    n == 0 && return UnitRange{Int}[]
    segs = UnitRange{Int}[]
    start = 1
    @inbounds for i in 2:n
        if t[i] - t[i - 1] > gap_size
            push!(segs, start:(i - 1))
            start = i
        end
    end
    push!(segs, start:n)
    return segs
end

# =====================================================================
# Savitzky-Golay detrending
# =====================================================================

"""
    detrend_savgol(t, flux, flux_err; window_length, polyorder=3,
                    transit_mask=nothing, gap_size=0.5) -> NamedTuple

Savitzky-Golay polynomial filter detrending (lightkurve.flatten-style).

Returns a NamedTuple with:
- `flux_detrended` — flux divided by the smoothed trend (median ≈ 1)
- `trend` — the smoothed trend at each observation
- `segments` — segment ranges (indices into `t`)
- `transit_mask` — the mask used (passed through or `nothing`)

# Arguments
- `t, flux, flux_err` — light curve arrays
- `window_length::Int` — odd integer, length of SG window in points.
   For TESS 2-min cadence, ~301 points ≈ 10 hours; ~1501 ≈ 50 hours.
- `polyorder::Int=3` — polynomial order for SG fit
- `transit_mask::Union{AbstractVector{Bool}, Nothing}=nothing` — if provided,
   masked points are interpolated over before SG smoothing, then
   replaced with their original values. This prevents the filter from
   absorbing transit shape into the trend.
- `gap_size::Real=0.5` — segment splitting threshold (days)

The data is processed per-segment to avoid SG artifacts at gaps.
"""
function detrend_savgol(t::Vector{Float64}, flux::Vector{Float64},
                         flux_err::Vector{Float64};
                         window_length::Int,
                         polyorder::Int=3,
                         transit_mask::Union{AbstractVector{Bool}, Nothing}=nothing,
                         gap_size::Real=0.5,
                         sector_id::Union{Nothing, AbstractVector{<:Integer}}=nothing)
    isodd(window_length) || throw(ArgumentError("window_length must be odd"))
    n = length(t)
    n == length(flux) == length(flux_err) || throw(ArgumentError(
        "t, flux, flux_err must have the same length"))
    sector_id === nothing || length(sector_id) == n || throw(ArgumentError(
        "sector_id length $(length(sector_id)) != length(t) $n"))
    sid_out = sector_id === nothing ? nothing : Vector{Int}(sector_id)

    segments = find_segments(t; gap_size=gap_size)
    trend = Vector{Float64}(undef, n)
    flux_detrended = Vector{Float64}(undef, n)

    for seg in segments
        flux_seg = flux[seg]
        n_seg = length(seg)

        if n_seg < window_length
            # Segment too short for SG — fit a polynomial directly
            trend_seg = _fit_poly_trend(t[seg], flux_seg, polyorder)
        else
            # Mask transits if provided: replace masked values with
            # linear interpolation, smooth, then restore originals.
            if transit_mask !== nothing
                mask_seg = transit_mask[seg]
                flux_for_sg = _interp_masked(flux_seg, mask_seg)
            else
                flux_for_sg = flux_seg
            end
            sg = savitzky_golay(flux_for_sg, window_length, polyorder)
            trend_seg = sg.y
        end
        trend[seg] = trend_seg
        flux_detrended[seg] = flux_seg ./ trend_seg
    end

    return (; flux_detrended, trend, segments, transit_mask, sector_id = sid_out)
end

"""Linear interpolation across masked points."""
function _interp_masked(y::AbstractVector{<:Real}, mask::AbstractVector{Bool})
    out = Vector{Float64}(y)
    n = length(out)
    i = 1
    while i <= n
        if mask[i]
            # Find run of masked points
            j = i
            while j <= n && mask[j]
                j += 1
            end
            # Interpolate from i-1 to j (boundaries)
            left  = i > 1 ? out[i - 1] : (j <= n ? out[j] : zero(Float64))
            right = j <= n ? out[j] : left
            for k in i:(j - 1)
                if i > 1 && j <= n
                    α = (k - (i - 1)) / (j - (i - 1))
                    out[k] = (1 - α) * left + α * right
                else
                    out[k] = (i > 1) ? left : right
                end
            end
            i = j
        else
            i += 1
        end
    end
    return out
end

"""Fit a low-order polynomial to (t, y) and return values at t."""
function _fit_poly_trend(t::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                         order::Int)
    n = length(t)
    order = min(order, n - 1)
    order < 0 && return fill(mean(y), n)
    t0 = mean(t)
    # Build design matrix
    X = Matrix{Float64}(undef, n, order + 1)
    for k in 0:order
        X[:, k + 1] = (t .- t0) .^ k
    end
    coefs = X \ Float64.(y)
    return X * coefs
end

# =====================================================================
# GP detrending (celerite kernels)
# =====================================================================

"""
    detrend_gp(t, flux, flux_err, kernel; transit_mask=nothing,
                init_params=nothing, gap_size=0.5) -> NamedTuple

GP detrending using a celerite kernel. Hyperparameters are fitted by
maximizing the marginal likelihood (per-segment) via `Optim.LBFGS()`.

Returns a NamedTuple:
- `flux_detrended` — flux − GP_mean (additive convention)
- `trend` — the GP mean at each observation
- `gp_params` — fitted hyperparameters. Vector of Vectors; with
   `joint_segments=true` (default) this is length 1 (one shared set
   across all segments); with `joint_segments=false` it has one entry
   per segment.
- `segments` — segment ranges (indices into `t`)
- `transit_mask` — passed through

# Arguments
- `kernel` — one of `CeleriteSHO()`, `CeleriteRotation()`,
   `CeleriteRotationFM17()`
- `init_params::Union{Nothing, Vector{Float64}}` — starting values for
   hyperparameters (in the order expected by the kernel). If `nothing`,
   sensible defaults are used.
- `joint_segments::Bool=true` — fit ONE set of hyperparameters across
   all sectors simultaneously. With celerite's exponential kernels and
   multi-day τ_decay, cross-sector correlation across TESS gaps
   (~700 d) is numerically zero, so a single celerite_loglike on the
   concatenated data is mathematically equivalent to summing per-sector
   likelihoods — and it gives the optimizer N×more rotation cycles of
   leverage, breaking the per-sector multimodality.
- `sector_id::Union{Nothing, AbstractVector{<:Integer}}=nothing` —
   per-cadence sector tag (1-indexed). Cadences sharing a sector ID
   share an `offset_S<k>` parameter, fit jointly with the GP. Mirrors
   Nereus's `Data.phot_inst` pattern, so the user builds it naturally
   during loading. Defaults to all-ones ⇒ one sector.

The detrending is **additive** (flux − GP) rather than multiplicative,
matching the GP-as-noise-model convention used elsewhere in Nereus.
For multiplicative detrending, divide flux by `1 + trend`.
"""
function detrend_gp(t::Vector{Float64}, flux::Vector{Float64},
                     flux_err::Vector{Float64},
                     kernel::CovarianceNoise;
                     transit_mask::Union{AbstractVector{Bool}, Nothing}=nothing,
                     init_params::Union{Nothing, Vector{Float64}}=nothing,
                     gp_priors::Dict{String, <:PriorSpec}=Dict{String, PriorSpec}(),
                     joint_segments::Bool=true,
                     sector_id::Union{Nothing, AbstractVector{<:Integer}}=nothing)
    n = length(t)
    n == length(flux) == length(flux_err) || throw(ArgumentError(
        "t, flux, flux_err must have the same length"))
    sector_id === nothing || length(sector_id) == n || throw(ArgumentError(
        "sector_id length $(length(sector_id)) != length(t) $n"))
    # Echo the USER's input verbatim (Vector{Int} if given, else nothing).
    # Do NOT fabricate the internal all-ones default.
    sid_out = sector_id === nothing ? nothing : Vector{Int}(sector_id)

    # Sector identity is a structural property of the data the caller
    # knows about (TESS sectors, K2 campaigns) — pass a per-cadence
    # `sector_id` vector (1-indexed; same shape as `t`/`flux`). The
    # cleaner does NOT try to detect sectors from time-gap heuristics.
    # Default (`sector_id=nothing`) treats the whole input as a single
    # sector, which is right for a single-sector LC.
    segments = sector_id === nothing ?
               UnitRange{Int}[1:n] :
               _segments_from_sector_id(sector_id)
    trend = zeros(Float64, n)
    flux_detrended = Vector{Float64}(undef, n)
    gp_params = Vector{Vector{Float64}}()

    # Build the masked-variance vector. 1e8× multiplier is needed to
    # truly downweight transit cadences when offsets are also free
    # parameters — at 1e4× the residual influence of transit-shaped
    # features on the joint (offset, GP) optimum is enough to let the
    # GP track partial transit shape into the trend (depth recovery
    # collapses from ~1000 ppm to ~70 ppm). Earlier celerite LDL
    # blowups at 1e8 came from extreme NelderMead hyperparameter
    # drift; with sample_map + bounded priors that doesn't happen.
    var_seg_all = similar(flux)
    @inbounds for i in eachindex(flux)
        var_seg_all[i] = flux_err[i]^2
        if transit_mask !== nothing && transit_mask[i]
            var_seg_all[i] *= 1e8
        end
    end

    # Joint fit: one shared GP hyperparameter set + per-segment offsets,
    # all fit simultaneously by `_fit_gp_hyperparams` via Nereus's
    # `sample_map`. Each segment is a phot "instrument" so the framework
    # gives it its own free `offset_S<k>` slot.
    offsets = Float64[]
    if joint_segments
        joint_params, joint_offsets = _fit_gp_hyperparams(
            t, flux, var_seg_all, segments, kernel, init_params, gp_priors)
        push!(gp_params, joint_params)
        offsets = joint_offsets
    else
        # Per-segment fallback: fit each segment independently (one
        # instrument, one offset, kernel, free-fit per segment). Less
        # statistical leverage but useful when segments differ
        # qualitatively (e.g. different active/quiet states).
        offsets = zeros(length(segments))
        for (k, seg) in enumerate(segments)
            p, o = _fit_gp_hyperparams(t[seg], flux[seg],
                                        var_seg_all[seg], [1:length(seg)],
                                        kernel, init_params, gp_priors)
            push!(gp_params, p)
            offsets[k] = o[1]
        end
    end

    # Per-segment prediction using the (joint or per-seg) hyperparameters.
    for (k, seg) in enumerate(segments)
        t_seg     = t[seg]
        var_seg   = var_seg_all[seg]
        flux_seg  = flux[seg]

        params = joint_segments ? gp_params[1] : gp_params[k]
        offset = offsets[k]
        # GP runs on (flux - 1 - offset) so it's zero-mean.
        y_seg = flux_seg .- 1.0 .- offset

        ar, cr, ac, bc, cc, dc = _kernel_coefs(kernel, params)
        μ = celerite_predict_mean(t_seg, y_seg, var_seg,
                                   ar, cr, ac, bc, cc, dc)
        # `trend` is the full multiplicative-style baseline (GP rotation
        # + fitted per-sector DC offset). Cleaned flux subtracts BOTH
        # so all sectors land at 1.0 ± transit dips — the offset is a
        # nuisance parameter that absorbs per-sector pipeline-level
        # median drift; users want a flat continuum at unity, not a
        # staircase across sector boundaries.
        trend[seg]          = μ .+ 1.0 .+ offset
        flux_detrended[seg] = flux_seg .- μ .- offset
    end

    return (; flux_detrended, trend, gp_params, offsets,
            segments, transit_mask, sector_id = sid_out)
end

# Kernel coefficient dispatch
function _kernel_coefs(::CeleriteSHO, p::Vector{Float64})
    log_S0, log_Q, log_w0 = p
    return sho_coefficients(exp(log_S0), exp(log_Q), exp(log_w0))
end
function _kernel_coefs(::CeleriteRotation, p::Vector{Float64})
    σ, period, Q0, dQ, frac = p
    return rotation_coefficients(σ, period, Q0, dQ, frac)
end
function _kernel_coefs(::CeleriteRotationFM17, p::Vector{Float64})
    log_amp, log_tau, log_P, log_f = p
    return rotation_fm17_coefficients(exp(log_amp), exp(log_tau),
                                       exp(log_P), exp(log_f))
end

# Default initial parameters per kernel
_default_init(::CeleriteSHO, y, t) = [log(var(y)), log(0.5), log(2π / 5.0)]
_default_init(::CeleriteRotation, y, t) = [std(y), 8.0, 1.0, 1.0, 0.5]
_default_init(::CeleriteRotationFM17, y, t) = [log(std(y)), log(20.0), log(8.0), log(0.5)]

"""Convert a per-cadence `sector_id` vector into the contiguous
`UnitRange{Int}` segments used internally. Cadences are assumed to be
already grouped by sector (loading order); we just walk the array and
split where the ID changes."""
function _segments_from_sector_id(sector_id::AbstractVector{<:Integer})
    n = length(sector_id)
    n == 0 && return UnitRange{Int}[]
    segs = UnitRange{Int}[]
    start = 1
    @inbounds for i in 2:n
        if sector_id[i] != sector_id[i - 1]
            push!(segs, start:(i - 1))
            start = i
        end
    end
    push!(segs, start:n)
    return segs
end

"""Tag a CovarianceNoise kernel as targeting the photometry channel,
preserving the type so `_kernel_coefs` and `_default_init` dispatch
correctly. Used to wire `detrend_gp`'s segment fit through the
Nereus prior+layout+target machinery, which keys on `:phot`."""
_phot_channel_kernel(::CeleriteSHO)            = CeleriteSHO(channel=:phot)
_phot_channel_kernel(::CeleriteRotation)       = CeleriteRotation(channel=:phot)
_phot_channel_kernel(::CeleriteRotationFM17)   = CeleriteRotationFM17(channel=:phot)

"""Fit GP hyperparameters AND per-segment photometric offsets jointly
by routing through Nereus's `sample_map` on a noise-only target with
**one phot instrument per segment**. Each segment gets its own
`offset_S<k>` parameter (NormalPrior centered at 0); the supplied GP
kernel (channel `:phot`) is shared across all instruments — i.e. the
GP shape is global, the DC level is per-segment, all fit jointly by
L-BFGS + ForwardDiff.

`flux` is the RAW per-cadence flux (NOT mean-subtracted — the offset
parameters absorb the per-segment median drift). `variances` already
includes the masked-high-variance trick at transit-masked cadences.
`segments` defines the per-instrument grouping: cadences in
`segments[k]` are tagged `phot_inst[i] = k` and share `offset_S<k>`.

Returns `(gp_params, offsets)`."""
function _fit_gp_hyperparams(t::Vector{Float64}, flux::Vector{Float64},
                              variances::Vector{Float64},
                              segments::Vector{<:UnitRange{<:Integer}},
                              kernel::CovarianceNoise,
                              init_params::Union{Nothing, Vector{Float64}},
                              user_gp_priors::Dict{String, <:PriorSpec} =
                                  Dict{String, PriorSpec}())
    flux_err_in = sqrt.(variances)
    nm_phot = _phot_channel_kernel(kernel)

    n_seg = length(segments)
    inst_names = ["S$k" for k in 1:n_seg]
    phot_inst = Vector{Int}(undef, length(t))
    for (k, seg) in enumerate(segments)
        phot_inst[seg] .= k
    end

    data_seg = Data(; t_phot = t, flux = flux, flux_err = flux_err_in,
                      phot_inst = phot_inst)
    ic = InstrumentConfig(rv = String[], pm = inst_names)

    # Per-instrument offset (Normal centered at 0; ±5% bound is wide
    # enough for any sane TESS pipeline normalisation drift). Jitter
    # is FROZEN at zero — letting it float here is degenerate with the
    # GP amplitude in a noise-only fit (jitter saturates and the GP
    # collapses to amp≈0 as a result), so it goes back to the user to
    # set in a downstream transit fit if they want it.
    fixed = Dict{String, PriorSpec}()
    for name in inst_names
        fixed["offset_$name"] = NormalPrior(0.0, 1e-2, -0.05, 0.05)
        fixed["jitter_$name"] = FixedPrior(0.0)
    end
    for (k, v) in user_gp_priors
        fixed[k] = v
    end

    params = Params(;
        max_kplanet  = 0,
        planet_modes = PlanetDataSources[],
        instruments  = ic,
        data         = data_seg,
        M_s          = 1.0,
        R_s          = 1.0,
        stability    = :none,
        noise_models = NoiseModel[nm_phot],
        priors       = fixed,
    )
    target = NereusTarget(params, data_seg)

    init_uncon = nothing
    if init_params !== nothing
        gp_names = noise_param_names(nm_phot, ic)
        unfrozen = params.layout.unfrozen_names
        x_bounded = zeros(length(unfrozen))
        for (n, v) in zip(gp_names, init_params)
            i = findfirst(==(n), unfrozen)
            i === nothing && continue
            x_bounded[i] = v
        end
        init_uncon = target.transform === nothing ? x_bounded :
                     transform_forward(x_bounded, target.transform)
    end

    # Bayesian inference over GP hyperparameters via Pathfinder
    # (multi-path variational MAP+Laplace). 8 L-BFGS runs from prior
    # draws cover basin diversity; PSIS-reweighted draws form a real
    # posterior — robust to the multimodality / bound-pinning that MAP
    # alone gets stuck in. Returns posterior medians as the point
    # estimate used by the trend prediction below.
    pf = pathfinder_init(target;
                          n_runs = 3,
                          n_draws = 200,
                          seed   = 42,
                          quiet  = false)

    n_dim = size(pf.draws, 1)
    n_draws = size(pf.draws, 2)
    bounded = Matrix{Float64}(undef, n_dim, n_draws)
    @inbounds for j in 1:n_draws
        col = @view pf.draws[:, j]
        bounded[:, j] = target.transform === nothing ?
                        col :
                        transform_inverse(Vector(col), target.transform)
    end

    param_names = params.layout.unfrozen_names
    name_to_idx = Dict(n => i for (i, n) in enumerate(param_names))
    medians = Float64[median(@view bounded[i, :]) for i in 1:n_dim]

    gp_params = Float64[medians[name_to_idx[n]]
                        for n in noise_param_names(nm_phot, ic)]
    offsets   = Float64[medians[name_to_idx["offset_$name"]]
                        for name in inst_names]
    return gp_params, offsets
end
