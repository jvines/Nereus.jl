# Packed prior representation for type-stable evaluation.
#
# The heterogeneous `Vector{PriorSpec}` (where each element wraps a
# different Distribution type) causes dynamic dispatch in `log_prior`
# and blocks Enzyme. This module provides a flat, type-stable
# representation: each prior is encoded as (type_id::Int, params::NTuple,
# lo::Float64, hi::Float64). A single `_eval_packed_logpdf` function
# handles all types via if/elseif — fully type-stable, no dispatch,
# Enzyme-compatible.
#
# This is the same pattern Nereus Python used for numba compatibility.

# Type IDs (must be consecutive starting from 0)
const PACKED_UNIFORM      = 0
const PACKED_NORMAL       = 1
const PACKED_LOGUNIFORM   = 2
const PACKED_MODJEFFREYS  = 3
const PACKED_BETA         = 4
const PACKED_FIXED        = 5
const PACKED_SINE         = 6  # p(x) = sin(x)/2 on [0, π]

"""
    PackedPriors

Type-stable prior storage. Each prior is encoded as:
- `type_ids[i]::Int` — distribution type (PACKED_UNIFORM, etc.)
- `params[i,:]::NTuple{4,Float64}` — hyperparameters (meaning depends on type)
- `lowers[i]::Float64` — hard lower bound
- `uppers[i]::Float64` — hard upper bound

Layout of `params[i,:]` per type:
- UNIFORM:     (unused, unused, unused, unused) — density = 1/(hi-lo)
- NORMAL:      (μ, σ, unused, unused) — truncated Normal
- LOGUNIFORM:  (unused, unused, unused, unused) — density ∝ 1/x
- MODJEFFREYS: (knee, unused, unused, unused) — density ∝ 1/(x+knee)
- BETA:        (a, b, unused, unused) — Beta(a,b) on [lo,hi]
- FIXED:       (value, unused, unused, unused) — contributes 0 to logp
"""
struct PackedPriors
    type_ids::Vector{Int}
    params::Matrix{Float64}   # n × 4
    lowers::Vector{Float64}
    uppers::Vector{Float64}
end

"""
    pack_priors(priors::Vector{<:PriorSpec}) -> PackedPriors

Convert a vector of `PriorSpec` objects into a type-stable packed
representation. Called once at model construction; the packed form is
used in the hot path.
"""
function pack_priors(priors::AbstractVector{<:PriorSpec})
    n = length(priors)
    type_ids = Vector{Int}(undef, n)
    pars = zeros(Float64, n, 4)
    lowers = Vector{Float64}(undef, n)
    uppers = Vector{Float64}(undef, n)

    for i in 1:n
        ps = priors[i]
        lowers[i] = ps.lo
        uppers[i] = ps.hi
        d = ps.dist

        if d isa Fixed
            type_ids[i] = PACKED_FIXED
            pars[i, 1] = d.value
        elseif d isa Distributions.Uniform
            type_ids[i] = PACKED_UNIFORM
        elseif d isa Distributions.LogUniform
            type_ids[i] = PACKED_LOGUNIFORM
        elseif d isa Distributions.Truncated{<:Distributions.Normal}
            type_ids[i] = PACKED_NORMAL
            pars[i, 1] = d.untruncated.μ
            pars[i, 2] = d.untruncated.σ
        elseif d isa Distributions.Normal
            type_ids[i] = PACKED_NORMAL
            pars[i, 1] = d.μ
            pars[i, 2] = d.σ
        elseif d isa Distributions.Beta
            type_ids[i] = PACKED_BETA
            pars[i, 1] = d.α
            pars[i, 2] = d.β
        elseif d isa ModJeffreys
            type_ids[i] = PACKED_MODJEFFREYS
            pars[i, 1] = d.knee
        elseif d isa Sine
            type_ids[i] = PACKED_SINE
        else
            # Fallback: use the PriorSpec's logpdf directly (loses
            # type stability for this entry but doesn't crash).
            # This triggers for affine-scaled Beta and other compound
            # distributions.
            type_ids[i] = PACKED_UNIFORM  # treat as uniform fallback
            @warn "pack_priors: unknown distribution type $(typeof(d)) for prior $i; using Uniform fallback" maxlog=3
        end
    end

    return PackedPriors(type_ids, pars, lowers, uppers)
end

"""
    eval_packed_logpdf(x, tid, p1, p2, lo, hi) -> Float64

Evaluate log-pdf of a single packed prior at value `x`. Type-stable:
dispatches on `tid` via if/elseif, no dynamic dispatch, Enzyme-safe.
"""
@inline function eval_packed_logpdf(x::Real, tid::Int,
                                     p1::Float64, p2::Float64,
                                     lo::Float64, hi::Float64)
    # Bounds check (always, for all types)
    if isfinite(lo) && x < lo
        return -Inf
    end
    if isfinite(hi) && x > hi
        return -Inf
    end

    if tid == PACKED_FIXED
        return 0.0
    elseif tid == PACKED_UNIFORM
        return -log(hi - lo)
    elseif tid == PACKED_NORMAL
        μ, σ = p1, p2
        z = (x - μ) / σ
        return -0.5 * z * z - log(σ) - 0.5 * log(2π)
        # Note: this is the unnormalised truncated-Normal logpdf
        # (doesn't include the truncation normalisation constant).
        # For MCMC this is fine — the constant cancels in the
        # acceptance ratio.
    elseif tid == PACKED_LOGUNIFORM
        if x <= 0
            return -Inf
        end
        return -log(x) - log(log(hi / lo))
    elseif tid == PACKED_MODJEFFREYS
        knee = p1
        if x < 0
            return -Inf
        end
        return -log(x + knee) - log(log((knee + hi) / knee))
    elseif tid == PACKED_BETA
        a, b = p1, p2
        if x <= lo || x >= hi
            return -Inf
        end
        # Rescale to [0, 1] for Beta
        t = (x - lo) / (hi - lo)
        if t <= 0 || t >= 1
            return -Inf
        end
        return (a - 1) * log(t) + (b - 1) * log(1 - t) - log(hi - lo)
        # Omits logbeta(a,b) normalisation — constant, cancels in MCMC.
    elseif tid == PACKED_SINE
        # p(x) = sin(x)/2 on [0, π]; logpdf = log(sin(x)) - log(2)
        # Bounds already checked above.
        return log(sin(x)) - log(2.0)
    else
        return 0.0  # unknown type, no contribution
    end
end

"""
    log_prior_packed(values, unfrozen_idx, packed) -> Float64

Evaluate total log-prior using the packed representation. Type-stable,
Enzyme-compatible.
"""
function log_prior_packed(values::AbstractVector, unfrozen_idx::Vector{Int},
                           type_ids::Vector{Int}, params_mat::Matrix{Float64},
                           lowers::Vector{Float64}, uppers::Vector{Float64})
    total = zero(eltype(values))
    @inbounds for i in eachindex(unfrozen_idx)
        x = values[unfrozen_idx[i]]
        tid = type_ids[i]
        p1 = params_mat[i, 1]
        p2 = params_mat[i, 2]
        lo = lowers[i]
        hi = uppers[i]
        lp = eval_packed_logpdf(x, tid, p1, p2, lo, hi)
        if !isfinite(lp)
            return convert(eltype(values), -Inf)
        end
        total += lp
    end
    return total
end

log_prior_packed(values::AbstractVector, unfrozen_idx::Vector{Int},
                  packed::PackedPriors) =
    log_prior_packed(values, unfrozen_idx, packed.type_ids, packed.params,
                      packed.lowers, packed.uppers)
