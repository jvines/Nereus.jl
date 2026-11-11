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
const PACKED_WRAPPED      = 7  # uniform on a circle, chart [lo, hi] (walled like UNIFORM)
const PACKED_WRAPPED_SINE = 8  # |sin x|/4 on the chart [-π/2, 3π/2] (cyclic inclination)
const PACKED_POLAR        = 9  # p(fold(x))/2 on [-π/2, 3π/2]; base tid in p3 (polar angle)
const PACKED_WRAPPED_NORMAL = 10 # wrapped Normal(p1, p2) truncated to [p3, p4], log mass p5
const PACKED_WRAPPED_STEP = 11 # Uniform(p3, p4) wrapped onto one period

"""
    PackedPriors

Type-stable prior storage. Each prior is encoded as:
- `type_ids[i]::Int` — distribution type (PACKED_UNIFORM, etc.)
- `params[i,:]` — five hyperparameters (meaning depends on type)
- `lowers[i]::Float64` — hard lower bound
- `uppers[i]::Float64` — hard upper bound

Layout of `params[i,:]` per type:
- UNIFORM:     (unused, unused, unused, unused) — density = 1/(hi-lo)
- NORMAL:      (μ, σ, unused, unused) — truncated Normal
- LOGUNIFORM:  (unused, unused, unused, unused) — density ∝ 1/x
- MODJEFFREYS: (knee, unused, unused, unused) — density ∝ 1/(x+knee)
- BETA:        (a, b, unused, unused) — Beta(a,b) on [lo,hi]
- FIXED:       (value, unused, unused, unused) — contributes 0 to logp
- POLAR:       (base p1, base p2, base type id, unused, base log mass) — the
               base prior on [0, π] read at fold(x), halved
- WRAPPED_NORMAL: (μ, σ, blo, bhi, log mass on [blo, bhi])
- WRAPPED_STEP:   (unused, unused, blo, bhi, unused)
"""
struct PackedPriors
    type_ids::Vector{Int}
    params::Matrix{Float64}   # n × 5
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
    pars = zeros(Float64, n, 5)
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
        elseif d isa WrappedUniform
            type_ids[i] = PACKED_WRAPPED
        elseif d isa WrappedSine
            type_ids[i] = PACKED_WRAPPED_SINE
        elseif d isa PolarFold
            # The base prior on [0, π], packed as it would be on its own.
            b = pack_priors([d.base])
            type_ids[i] = PACKED_POLAR
            pars[i, 1], pars[i, 2] = b.params[1, 1], b.params[1, 2]
            pars[i, 3] = b.type_ids[1]
            bd = d.base.dist
            pars[i, 5] = bd isa Distributions.Truncated{<:Distributions.Normal} ?
                Distributions.logdiffcdf(bd.untruncated, d.base.hi, d.base.lo) : 0.0
        elseif d isa WrappedPushforward
            type_ids[i] = d.kind == _PUSH_NORMAL ? PACKED_WRAPPED_NORMAL : PACKED_WRAPPED_STEP
            pars[i, 1], pars[i, 2] = d.μ, d.σ
            pars[i, 3], pars[i, 4], pars[i, 5] = d.blo, d.bhi, d.logZ
        elseif d isa Distributions.LogUniform
            type_ids[i] = PACKED_LOGUNIFORM
        elseif d isa Distributions.Truncated{<:Distributions.Normal}
            type_ids[i] = PACKED_NORMAL
            pars[i, 1] = d.untruncated.μ
            pars[i, 2] = d.untruncated.σ
            pars[i, 5] = _normal_logmass(d.untruncated, ps.lo, ps.hi)
        elseif d isa Distributions.Normal
            type_ids[i] = PACKED_NORMAL
            pars[i, 1] = d.μ
            pars[i, 2] = d.σ
        elseif d isa Distributions.Beta
            type_ids[i] = PACKED_BETA
            pars[i, 1] = d.α
            pars[i, 2] = d.β
            pars[i, 5] = _beta_lognorm(d.α, d.β)
        elseif d isa Distributions.AffineDistribution && d.ρ isa Distributions.Beta
            # BetaPrior(a, b; lo, hi) off [0, 1]: lo + (hi - lo) Beta(a, b). This
            # fell through to the Uniform fallback below, so the samplers saw
            # a flat prior where the user wrote a Beta.
            type_ids[i] = PACKED_BETA
            pars[i, 1] = d.ρ.α
            pars[i, 2] = d.ρ.β
            pars[i, 5] = _beta_lognorm(d.ρ.α, d.ρ.β)
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
    eval_packed_logpdf(x, pp::PackedPriors, i) -> Float64
    eval_packed_logpdf(x, tid, p1, p2, lo, hi[, p3, p4, p5]) -> Float64

Evaluate log-pdf of a single packed prior at value `x`: prior `i` of `pp`, or
one given by its fields. Type-stable: dispatches on `tid` via if/elseif, no
dynamic dispatch, Enzyme-safe. The six-field form leaves `p3`-`p5` at zero,
which is right for every type but the polar and wrapped pushforward ones
(`PACKED_POLAR`, `PACKED_WRAPPED_NORMAL`, `PACKED_WRAPPED_STEP`): read those
through `pp` and `i`.
"""
@inline eval_packed_logpdf(x::Real, pp::PackedPriors, i::Int) =
    eval_packed_logpdf(x, pp.type_ids[i], pp.params[i, 1], pp.params[i, 2],
                       pp.lowers[i], pp.uppers[i],
                       pp.params[i, 3], pp.params[i, 4], pp.params[i, 5])

@inline eval_packed_logpdf(x::Real, tid::Int, p1::Float64, p2::Float64,
                           lo::Float64, hi::Float64) =
    eval_packed_logpdf(x, tid, p1, p2, lo, hi, 0.0, 0.0,
                       _packed_lognorm(tid, p1, p2, lo, hi))

# The log normaliser `p5` of a truncated Normal (its log mass on [lo, hi]) or a
# Beta (log B(a, b)), which `pack_priors` stores and the evaluation subtracts.
_packed_lognorm(tid::Int, p1::Float64, p2::Float64, lo::Float64, hi::Float64) =
    tid == PACKED_NORMAL ? _normal_logmass(Distributions.Normal(p1, p2), lo, hi) :
    tid == PACKED_BETA ? _beta_lognorm(p1, p2) : 0.0

"""
    _normal_logmass(N, lo, hi) -> Float64

`log P(lo ≤ X ≤ hi)` for `X ~ N`: the truncation constant of a Normal prior,
`0` when it is not truncated.
"""
function _normal_logmass(N::Distributions.Normal, lo::Real, hi::Real)
    (isfinite(lo) || isfinite(hi)) || return 0.0
    isfinite(lo) || return Float64(Distributions.logcdf(N, hi))
    isfinite(hi) || return Float64(Distributions.logccdf(N, lo))
    return Float64(Distributions.logdiffcdf(N, hi, lo))
end

# log B(a, b), from the normalised Beta logpdf at 1/2.
_beta_lognorm(a::Real, b::Real) =
    Float64((a - 1) * log(0.5) + (b - 1) * log(0.5) -
            Distributions.logpdf(Distributions.Beta(a, b), 0.5))

@inline function eval_packed_logpdf(x::Real, tid::Int,
                                     p1::Float64, p2::Float64,
                                     lo::Float64, hi::Float64,
                                     p3::Float64, p4::Float64, p5::Float64)
    # A polar angle (an inclination) on its circle: the base prior on [0, π]
    # read at the fold of x, halved, on the chart [lo, hi] = [-π/2, 3π/2]
    # (see `PolarFold`). `p5` is the base's log mass on [0, π] (a truncated
    # Normal's), so the lifted density is normalised.
    if tid == PACKED_POLAR
        lo <= x <= hi || return -Inf
        return _eval_packed_base(fold_inclination(x), round(Int, p3), p1, p2,
                                 0.0, Float64(π)) - p5 - log(2.0)
    end
    # A full-circle angle's prior pushed forward onto one period
    # (`WrappedPushforward`): periodic, walled at its chart.
    if tid == PACKED_WRAPPED_NORMAL || tid == PACKED_WRAPPED_STEP
        lo <= x <= hi || return -Inf
        return wrapped_push_logpdf(x, tid == PACKED_WRAPPED_NORMAL ? _PUSH_NORMAL :
                                      _PUSH_UNIFORM, p1, p2, p3, p4, p5)
    end
    # A truncated Normal and a Beta less their log normaliser `p5`, so the
    # packed density is the normalised one the Distributions path evaluates.
    # The constant does not cancel everywhere: trans-dim birth and death ratios
    # evaluate the born parameters' priors, and the evidence carries it.
    v = _eval_packed_base(x, tid, p1, p2, lo, hi)
    return (tid == PACKED_NORMAL || tid == PACKED_BETA) ? v - p5 : v
end

@inline function _eval_packed_base(x::Real, tid::Int,
                                   p1::Float64, p2::Float64,
                                   lo::Float64, hi::Float64)
    # The cyclic inclination: sin(i)/2 on [0, π] lifted to the circle, density
    # |sin x|/4 on its chart [lo, hi] = [-π/2, 3π/2], where 0 and π are
    # interior points. Zero outside the chart, which keeps the target proper
    # (see `WrappedSine`). Same Float64 arithmetic as `logpdf(::WrappedSine)`.
    if tid == PACKED_WRAPPED_SINE
        return lo <= x <= hi ? log(abs(sin(x))) - log(4.0) : -Inf
    end
    # Bounds check (always, for all other types)
    if isfinite(lo) && x < lo
        return -Inf
    end
    if isfinite(hi) && x > hi
        return -Inf
    end

    if tid == PACKED_FIXED
        return 0.0
    elseif tid == PACKED_UNIFORM || tid == PACKED_WRAPPED   # WrappedUniform: walled
        return -log(hi - lo)
    elseif tid == PACKED_NORMAL
        μ, σ = p1, p2
        z = (x - μ) / σ
        # The untruncated Normal; the truncation constant (p5) is subtracted
        # by `eval_packed_logpdf`.
        return -0.5 * z * z - log(σ) - 0.5 * log(2π)
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
        # log B(a, b) (p5) is subtracted by `eval_packed_logpdf`.
        return (a - 1) * log(t) + (b - 1) * log(1 - t) - log(hi - lo)
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
        lp = eval_packed_logpdf(x, tid, p1, p2, lo, hi,
                                params_mat[i, 3], params_mat[i, 4], params_mat[i, 5])
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
