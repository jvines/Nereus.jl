# Unconstrained-space parameter transforms — Enzyme-compatible.
#
# Replaces Bijectors.Stacked with a flat, type-stable implementation
# that Enzyme can trace through. Each transform is encoded as an
# integer type ID + bounds. The forward/inverse/Jacobian are computed
# in scalar loops with if/elseif — no Vector{Any}, no map, no dynamic
# dispatch.
#
# Three transform types cover all our priors:
#   IDENTITY — unbounded params (Normal without bounds)
#   LOGIT    — bounded [lo, hi] (Uniform, Truncated Normal, Beta, etc.)
#   LOG_SHIFT — lower-bounded [lo, ∞) (would be used for half-bounded;
#              currently not needed since all our priors are fully bounded)
#   LOGLOGIT — bounded [lo, hi], 0 < lo, logit of the position in log x: for
#              a log-uniform box spanning so many decades that the position
#              in x underflows the logit's clamp (`rho_circ_k`, src/density.jl)

const TRANSFORM_IDENTITY = 0
const TRANSFORM_LOGIT    = 1
const TRANSFORM_LOG      = 2   # x = lo + exp(y), for [lo, ∞)
const TRANSFORM_LOGLOGIT = 3   # x = exp(log lo + log(hi/lo) logistic(y)), 0 < lo < hi


"""
    PackedTransforms

Flat, type-stable representation of per-parameter transforms from
bounded to unconstrained space. Enzyme-compatible (no dynamic dispatch,
no Vector{Any}).

# Fields
- `type_ids::Vector{Int}` — transform type per parameter
- `lowers::Vector{Float64}` — lower bound
- `uppers::Vector{Float64}` — upper bound
"""
struct PackedTransforms
    type_ids::Vector{Int}
    lowers::Vector{Float64}
    uppers::Vector{Float64}
end

"""
    build_transform(params::Params) -> PackedTransforms

Build element-wise transforms from each unfrozen prior's bounds.
Bounded `[lo, hi]` → logit. Unbounded → identity.
"""
function build_transform(params::Params)
    layout = params.layout
    n = length(layout.unfrozen_priors)
    type_ids = Vector{Int}(undef, n)
    lowers = Vector{Float64}(undef, n)
    uppers = Vector{Float64}(undef, n)

    for i in 1:n
        lo, hi = bounds(layout.unfrozen_priors[i])
        lowers[i] = isfinite(lo) ? lo : 0.0
        uppers[i] = isfinite(hi) ? hi : 0.0

        if is_wrapped(layout.unfrozen_priors[i])
            # A wrapped angle has no boundary to map away: the sampler moves
            # on the line and the density is periodic. A logit here would put
            # back exactly the wall the wrapped prior exists to remove.
            type_ids[i] = TRANSFORM_IDENTITY
        elseif is_density_reparam_name(layout.unfrozen_names[i]) &&
               layout.unfrozen_priors[i].dist isa LogUniform
            # ρ_circ's box spans tens of decades (`DENSITY_G`): the logit of
            # its position in x would clamp at 1e-10 from `lo`.
            type_ids[i] = TRANSFORM_LOGLOGIT
        elseif isfinite(lo) && isfinite(hi)
            type_ids[i] = TRANSFORM_LOGIT
        elseif isfinite(lo)
            type_ids[i] = TRANSFORM_LOG
        else
            type_ids[i] = TRANSFORM_IDENTITY
        end
    end
    return PackedTransforms(type_ids, lowers, uppers)
end

# The log-logit map of one coordinate, its inverse and log|dx/dy|:
# u = log x on [log lo, log hi], y = logit of u's position there.
@inline function _loglogit_forward(x, lo::Float64, hi::Float64)
    a, b = log(lo), log(hi)
    t = clamp((log(max(x, lo)) - a) / (b - a), 1e-10, 1.0 - 1e-10)
    return log(t / (1 - t))
end
@inline function _loglogit_inverse(y, lo::Float64, hi::Float64)
    a, b = log(lo), log(hi)
    return exp(a + (b - a) / (1 + exp(-y)))
end
@inline function _loglogit_logJ(y, lo::Float64, hi::Float64)
    a, b = log(lo), log(hi)
    s = 1 / (1 + exp(-y))
    return a + (b - a) * s + log(b - a) + log(s) + log(1 - s)
end

# =====================================================================
# Forward: bounded x → unconstrained y
# =====================================================================

"""
    transform_forward(x, pt) -> y

Map from bounded (prior support) space to unconstrained ℝⁿ.
"""
transform_forward(x::AbstractVector, type_ids::Vector{Int},
                  lowers::Vector{Float64}, uppers::Vector{Float64}) =
    transform_forward!(similar(x, Float64), x, type_ids, lowers, uppers)

"""
    transform_forward!(y, x, pt) -> y

`transform_forward` into `y`, which may be `x` itself: each `y[i]` depends on
`x[i]` alone. Same arithmetic as the allocating method, which calls this.
"""
transform_forward!(y::AbstractVector, x::AbstractVector, pt::PackedTransforms) =
    transform_forward!(y, x, pt.type_ids, pt.lowers, pt.uppers)

function transform_forward!(y::AbstractVector, x::AbstractVector, type_ids::Vector{Int},
                            lowers::Vector{Float64}, uppers::Vector{Float64})
    @inbounds for i in eachindex(x)
        tid = type_ids[i]
        if tid == TRANSFORM_IDENTITY
            y[i] = x[i]
        elseif tid == TRANSFORM_LOGIT
            lo, hi = lowers[i], uppers[i]
            t = (x[i] - lo) / (hi - lo)
            # Clamp to avoid log(0) or log(negative)
            t = clamp(t, 1e-10, 1.0 - 1e-10)
            y[i] = log(t / (1 - t))
        elseif tid == TRANSFORM_LOGLOGIT
            y[i] = _loglogit_forward(x[i], lowers[i], uppers[i])
        else  # TRANSFORM_LOG
            lo = lowers[i]
            y[i] = log(max(x[i] - lo, 1e-10))
        end
    end
    return y
end

transform_forward(x::AbstractVector, pt::PackedTransforms) =
    transform_forward(x, pt.type_ids, pt.lowers, pt.uppers)

# Callable syntax: pt(x) == transform_forward(x, pt)
(pt::PackedTransforms)(x::AbstractVector) = transform_forward(x, pt)

# =====================================================================
# Inverse: unconstrained y → bounded x
# =====================================================================

"""
    transform_inverse(y, pt) -> x

Map from unconstrained ℝⁿ back to bounded (prior support) space.
Uses numerically stable logistic: `1 / (1 + exp(-y))`.
"""
function transform_inverse(y::AbstractVector, type_ids::Vector{Int},
                            lowers::Vector{Float64}, uppers::Vector{Float64})
    n = length(y)
    x = similar(y)
    @inbounds for i in 1:n
        tid = type_ids[i]
        if tid == TRANSFORM_IDENTITY
            x[i] = y[i]
        elseif tid == TRANSFORM_LOGIT
            lo, hi = lowers[i], uppers[i]
            s = 1 / (1 + exp(-y[i]))
            x[i] = lo + (hi - lo) * s
        elseif tid == TRANSFORM_LOGLOGIT
            x[i] = _loglogit_inverse(y[i], lowers[i], uppers[i])
        else  # TRANSFORM_LOG
            lo = lowers[i]
            x[i] = lo + exp(y[i])
        end
    end
    return x
end

transform_inverse(y::AbstractVector, pt::PackedTransforms) =
    transform_inverse(y, pt.type_ids, pt.lowers, pt.uppers)

"""
    transform_inverse_and_logJ(y, pt) -> (x, lj)

Fused inverse transform + log-Jacobian. Avoids recomputing sigmoid
for each element (T1 optimization).
"""
function transform_inverse_and_logJ(y::AbstractVector, type_ids::Vector{Int},
                                     lowers::Vector{Float64}, uppers::Vector{Float64})
    n = length(y)
    x = similar(y)
    lj = zero(eltype(y))
    @inbounds for i in 1:n
        tid = type_ids[i]
        if tid == TRANSFORM_IDENTITY
            x[i] = y[i]
        elseif tid == TRANSFORM_LOGIT
            lo, hi = lowers[i], uppers[i]
            s = 1 / (1 + exp(-y[i]))
            x[i] = lo + (hi - lo) * s
            lj += log(hi - lo) + log(s) + log(1 - s)
        elseif tid == TRANSFORM_LOGLOGIT
            x[i] = _loglogit_inverse(y[i], lowers[i], uppers[i])
            lj += _loglogit_logJ(y[i], lowers[i], uppers[i])
        else  # TRANSFORM_LOG
            lo = lowers[i]
            x[i] = lo + exp(y[i])
            lj += y[i]
        end
    end
    return x, lj
end

transform_inverse_and_logJ(y::AbstractVector, pt::PackedTransforms) =
    transform_inverse_and_logJ(y, pt.type_ids, pt.lowers, pt.uppers)

# =====================================================================
# Log-Jacobian of the inverse transform
# =====================================================================

"""
    transform_logabsdetjac_inv(y, pt) -> scalar

Log absolute determinant of the Jacobian of the inverse transform
(unconstrained → bounded). Added to the log-posterior to preserve
the correct density under the change of variables.

For logit: log|dx/dy| = log(hi - lo) + log(s) + log(1 - s)
where s = logistic(y).

For log-shift: log|dx/dy| = y[i].
"""
function transform_logabsdetjac_inv(y::AbstractVector, type_ids::Vector{Int},
                                     lowers::Vector{Float64}, uppers::Vector{Float64})
    lj = zero(eltype(y))
    @inbounds for i in eachindex(y)
        tid = type_ids[i]
        if tid == TRANSFORM_IDENTITY
            # log|dx/dy| = 0
        elseif tid == TRANSFORM_LOGIT
            lo, hi = lowers[i], uppers[i]
            s = 1 / (1 + exp(-y[i]))
            lj += log(hi - lo) + log(s) + log(1 - s)
        elseif tid == TRANSFORM_LOGLOGIT
            lj += _loglogit_logJ(y[i], lowers[i], uppers[i])
        else  # TRANSFORM_LOG
            lj += y[i]
        end
    end
    return lj
end

transform_logabsdetjac_inv(y::AbstractVector, pt::PackedTransforms) =
    transform_logabsdetjac_inv(y, pt.type_ids, pt.lowers, pt.uppers)
