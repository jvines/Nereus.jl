# Sampler target: LogDensityProblems interface.
#
# NereusTarget wraps (Params, Data, PackedTransforms) and provides the
# standard LogDensityProblems interface. The transforms map bounded
# parameters to unconstrained space so HMC/NUTS has no boundaries to
# hit. All internals are type-stable and Enzyme-compatible.

using LogDensityProblems
import Enzyme

"""
    NereusTarget(params, data; unconstrained=true)

Bayesian target implementing `LogDensityProblems.jl`.

When `unconstrained=true` (default), the sampler operates in ℝⁿ and
the target internally maps to bounded space via the packed transforms.
When `false`, the sampler operates directly in bounded space (for
nested sampling, MH, or trans-dim).
"""
struct NereusTarget{T}
    params::Params
    data::Data
    transform::T   # PackedTransforms or Nothing
end

function NereusTarget(params::Params, data::Data; unconstrained::Bool=true)
    if unconstrained
        t = build_transform(params)
        return NereusTarget(params, data, t)
    else
        return NereusTarget(params, data, nothing)
    end
end

# --- logdensity: direct bounded-space path (no transform) -------------

function LogDensityProblems.logdensity(target::NereusTarget{Nothing},
                                        x::AbstractVector)
    T = eltype(x)
    # NaN-in-x guard: Distributions.jl's `logpdf(Uniform, NaN)` doesn't
    # cleanly reject — NaN comparisons return false, so the prior
    # silently emits a finite-looking value, and NaN propagates
    # through to the RV likelihood where it might or might not be
    # caught. Bail at the boundary instead.
    @inbounds for i in eachindex(x)
        isfinite(x[i]) || return convert(T, -Inf)
    end
    theta = Theta{T}(target.params)
    set_unfrozen!(theta, x)
    lp = log_prior(theta)
    isfinite(lp) || return convert(T, -Inf)
    ll = rv_log_likelihood(theta, target.data)
    isfinite(ll) || return convert(T, -Inf)
    # `transit_log_likelihood` can return NaN (e.g. degenerate b/a_R
    # combos, a Cholesky failure that returns NaN instead of -Inf,
    # numerical edge in batman / NbodyGradient). Without this guard,
    # any sampler that uses `logp(draw)` directly — Pathfinder's IS
    # resample table, NUTS's accept/reject, ptemcee's swap proposals
    # — hits `sum(weights)/sum(logp) == NaN` and aborts. Bound the
    # final result the same way `lp` / `rv_log_likelihood` are
    # bounded above.
    lt = transit_log_likelihood(theta, target.data)
    # Doppler tomography. Zero-cost when data.tomo is empty; shares lambda and
    # v sin i with the RM velocities through the layout, which is what makes a
    # joint obliquity fit joint.
    ltomo = tomogram_log_likelihood(theta, target.data)
    isfinite(ltomo) || return convert(T, -Inf)
    isfinite(lt) || return convert(T, -Inf)
    return lp + ll + lt + ltomo
end

# --- logdensity: unconstrained path (packed transforms) ---------------

function LogDensityProblems.logdensity(target::NereusTarget{PackedTransforms},
                                        y::AbstractVector)
    pt = target.transform

    # Inverse transform + log-Jacobian. Both can produce non-finite
    # values for extreme `y` — `transform_inverse` saturates at hard
    # boundaries (e.g. `tanh(y) → 1.0` numerically for |y| ≳ 20, then
    # the back-mapping pinned to the boundary), and
    # `transform_logabsdetjac_inv` evaluates `log(d/dy transform)`
    # which often involves `log(1 - tanh²(y))` — that's `-Inf` at
    # saturation and `NaN` once the loss-of-precision tips over.
    #
    # Pathfinder's IS-resample step under an unconstrained target
    # (the 7a18999 fix path) routinely samples MVN draws far from the
    # LBFGS basin in unconstrained space, so it hits this regime
    # often. Without these guards the final `lp + ll + lt + lj` is
    # NaN, PSIS's weights vector contains NaN, AliasTable's
    # `checked_sum` throws `sum(weights) == NaN which is not finite`,
    # the whole job dies. This was the actual NaN source for the
    # NOI-106823 / FEROS RV-only `pt_warm` run; the earlier guard on
    # `transit_log_likelihood` (da2df6a) was correct defensive
    # coding but not this bug.
    x = transform_inverse(y, pt)
    T = eltype(x)
    @inbounds for i in eachindex(x)
        isfinite(x[i]) || return convert(T, -Inf)
    end
    lj = transform_logabsdetjac_inv(y, pt)
    isfinite(lj) || return convert(T, -Inf)

    # Build Theta and evaluate posterior in bounded space
    theta = Theta{T}(target.params)
    set_unfrozen!(theta, x)

    lp = log_prior(theta)
    isfinite(lp) || return convert(T, -Inf)

    ll = rv_log_likelihood(theta, target.data)
    isfinite(ll) || return convert(T, -Inf)
    # See the bounded-space path above — `transit_log_likelihood` can
    # return NaN at degenerate transit-shape configurations and we
    # must -Inf it out before downstream samplers (Pathfinder's IS
    # resample in particular) consume the value.
    lt = transit_log_likelihood(theta, target.data)
    # Doppler tomography. Zero-cost when data.tomo is empty; shares lambda and
    # v sin i with the RM velocities through the layout, which is what makes a
    # joint obliquity fit joint.
    ltomo = tomogram_log_likelihood(theta, target.data)
    isfinite(ltomo) || return convert(T, -Inf)
    isfinite(lt) || return convert(T, -Inf)
    return lp + ll + lt + ltomo + lj
end

# --- tempered components (for Hamiltonian parallel tempering) ----------
#
# Parallel tempering tempers the LIKELIHOOD only: π_β ∝ L^β · prior. HMC-PT
# therefore needs the prior(+Jacobian) and log-likelihood SEPARATELY so the
# tempered target is `(prior+jac) + β·loglike`. These mirror the logdensity
# paths above but return the two pieces instead of their sum.

"""
    _logdensity_parts(target, y) -> (prior_plus_jac, loglike)

Split the log-posterior into (log-prior + log-Jacobian, log-likelihood) for
tempering. `(-Inf, -Inf)` on any non-finite component (same guards as
`logdensity`). Unconstrained (transform) and bounded paths both provided.
"""
function _logdensity_parts(target::NereusTarget{PackedTransforms},
                           y::AbstractVector)
    pt = target.transform
    x = transform_inverse(y, pt)
    T = eltype(x)
    @inbounds for i in eachindex(x)
        isfinite(x[i]) || return (convert(T, -Inf), convert(T, -Inf))
    end
    lj = transform_logabsdetjac_inv(y, pt)
    isfinite(lj) || return (convert(T, -Inf), convert(T, -Inf))
    theta = Theta{T}(target.params)
    set_unfrozen!(theta, x)
    lp = log_prior(theta)
    isfinite(lp) || return (convert(T, -Inf), convert(T, -Inf))
    ll = rv_log_likelihood(theta, target.data)
    isfinite(ll) || return (convert(T, -Inf), convert(T, -Inf))
    lt = transit_log_likelihood(theta, target.data)
    # Doppler tomography. Zero-cost when data.tomo is empty; shares lambda and
    # v sin i with the RM velocities through the layout, which is what makes a
    # joint obliquity fit joint.
    ltomo = tomogram_log_likelihood(theta, target.data)
    isfinite(ltomo) || return (convert(T, -Inf), convert(T, -Inf))
    isfinite(lt) || return (convert(T, -Inf), convert(T, -Inf))
    # ltomo joins the LIKELIHOOD half: it is data, so it must be tempered with
    # the rest. Putting it in the prior half would leave the tomogram at full
    # strength on every rung of the ladder and break the thermodynamic path.
    return (lp + lj, ll + lt + ltomo)
end

function _logdensity_parts(target::NereusTarget{Nothing}, x::AbstractVector)
    T = eltype(x)
    @inbounds for i in eachindex(x)
        isfinite(x[i]) || return (convert(T, -Inf), convert(T, -Inf))
    end
    theta = Theta{T}(target.params)
    set_unfrozen!(theta, x)
    lp = log_prior(theta)
    isfinite(lp) || return (convert(T, -Inf), convert(T, -Inf))
    ll = rv_log_likelihood(theta, target.data)
    isfinite(ll) || return (convert(T, -Inf), convert(T, -Inf))
    lt = transit_log_likelihood(theta, target.data)
    # Doppler tomography. Zero-cost when data.tomo is empty; shares lambda and
    # v sin i with the RM velocities through the layout, which is what makes a
    # joint obliquity fit joint.
    ltomo = tomogram_log_likelihood(theta, target.data)
    isfinite(ltomo) || return (convert(T, -Inf), convert(T, -Inf))
    isfinite(lt) || return (convert(T, -Inf), convert(T, -Inf))
    return (lp, ll + lt + ltomo)
end

"""
    TemperedTarget(base::NereusTarget, beta)

LogDensityProblems target for `(prior + Jacobian) + β·loglike` — the tempered
posterior at inverse-temperature `β`. Wraps a `NereusTarget`; differentiable
via the same AD backends (ForwardDiff). Used by `sample_pt_hmc`.
"""
struct TemperedTarget{T}
    base::NereusTarget{T}
    beta::Float64
end

LogDensityProblems.dimension(t::TemperedTarget) =
    LogDensityProblems.dimension(t.base)
LogDensityProblems.capabilities(::Type{<:TemperedTarget}) =
    LogDensityProblems.LogDensityOrder{0}()

function LogDensityProblems.logdensity(t::TemperedTarget, y::AbstractVector)
    pj, ll = _logdensity_parts(t.base, y)
    T = eltype(y)
    (isfinite(pj) && isfinite(ll)) || return convert(T, -Inf)
    return pj + convert(T, t.beta) * ll
end

# --- dimension + capabilities -----------------------------------------

LogDensityProblems.dimension(target::NereusTarget) =
    n_unfrozen(target.params)

LogDensityProblems.capabilities(::Type{<:NereusTarget}) =
    LogDensityProblems.LogDensityOrder{0}()

# --- Enzyme-safe gradient (explicit Const arguments) -------------------
#
# Enzyme's reverse pass zeros mutable Vector{Float64} fields on Const
# structs. The 5 affected arrays (PackedTransforms.lowers/uppers,
# PackedPriors.params/lowers/uppers) are passed as separate Const
# arguments so Enzyme reads them from explicitly-Const top-level args
# rather than from nested struct fields.
#
# If Enzyme still zeros the arrays inside target despite not reading
# them from the struct path, a save/restore fallback activates
# automatically on the first gradient call.

"""
    _logdensity_for_enzyme(y, t_lowers, t_uppers, p_params, p_lowers, p_uppers, target)

Inner logdensity for Enzyme. Reads transform/prior Float64 arrays from
explicit arguments (not from target struct fields) so they can be
passed as separate `Enzyme.Const` to avoid shadow zeroing.
"""
function _logdensity_for_enzyme(
    y::Vector{Float64},
    t_lowers::Vector{Float64}, t_uppers::Vector{Float64},
    p_params::Matrix{Float64}, p_lowers::Vector{Float64},
    p_uppers::Vector{Float64},
    target::NereusTarget{PackedTransforms},
)
    pt = target.transform

    # Transform using explicit args (NOT pt.lowers/pt.uppers)
    x = transform_inverse(y, pt.type_ids, t_lowers, t_uppers)
    lj = transform_logabsdetjac_inv(y, pt.type_ids, t_lowers, t_uppers)

    # Build Theta (reads frozen_idx/frozen_values from target.params — safe)
    T = eltype(x)
    theta = Theta{T}(target.params)
    set_unfrozen!(theta, x)

    # Prior using explicit args (NOT packed_priors struct fields)
    lp = log_prior_packed(theta.values, target.params.layout.unfrozen_idx,
                           target.params.layout.packed_priors.type_ids,
                           p_params, p_lowers, p_uppers)
    isfinite(lp) || return convert(T, -Inf)

    # Likelihood — calls Enzyme-safe no-noise path directly.
    # Stability, external priors, noise models, and transit are NOT
    # included (use ForwardDiff for models with those features).
    ll = _rv_ll_no_noise(theta, target.data)
    return lp + ll + lj
end

"""
    EnzymeGradientConfig

Configuration for Enzyme gradient with automatic fallback to
save/restore if Enzyme zeros arrays inside `Const(target)`.

On the first gradient call, detects whether Enzyme zeroed the 5
Float64 arrays despite the explicit Const args. If so, activates
save/restore for all subsequent calls (same cost as previous approach).
If not, subsequent calls have zero overhead.
"""
mutable struct EnzymeGradientConfig
    dx::Vector{Float64}
    needs_restore::Bool
    checked::Bool
    # Saved copies for restore fallback
    pp_lowers::Vector{Float64}
    pp_uppers::Vector{Float64}
    pp_params::Matrix{Float64}
    pt_lowers::Vector{Float64}
    pt_uppers::Vector{Float64}
end

function EnzymeGradientConfig(target::NereusTarget{PackedTransforms})
    pp = target.params.layout.packed_priors
    pt = target.transform
    dim = n_unfrozen(target.params)
    return EnzymeGradientConfig(
        zeros(Float64, dim),
        false, false,
        copy(pp.lowers), copy(pp.uppers), copy(pp.params),
        copy(pt.lowers), copy(pt.uppers),
    )
end

"""
    enzyme_logdensity_and_gradient!(cfg, target, y) -> (lp, grad)

Enzyme gradient with explicit Const arguments for the 5 Float64 arrays
that Enzyme's reverse pass zeroes on Const structs. Falls back to
save/restore automatically if needed.
"""
function enzyme_logdensity_and_gradient!(cfg::EnzymeGradientConfig,
                                          target::NereusTarget{PackedTransforms},
                                          y::Vector{Float64})
    fill!(cfg.dx, 0.0)
    pp = target.params.layout.packed_priors
    pt = target.transform

    _, lp = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        Enzyme.Const(_logdensity_for_enzyme),
        Enzyme.Active,
        Enzyme.Duplicated(y, cfg.dx),
        Enzyme.Const(pt.lowers), Enzyme.Const(pt.uppers),
        Enzyme.Const(pp.params), Enzyme.Const(pp.lowers), Enzyme.Const(pp.uppers),
        Enzyme.Const(target),
    )

    # Auto-detect zeroing on first call
    if !cfg.checked
        cfg.checked = true
        cfg.needs_restore = (pt.lowers[1] != cfg.pt_lowers[1])
        if cfg.needs_restore
            @info "Enzyme: Const struct field zeroing detected, save/restore fallback active"
        else
            @info "Enzyme: explicit Const args working, zero overhead"
        end
    end

    # Restore if needed (same cost as previous EnzymeGradientState approach)
    if cfg.needs_restore
        pp.lowers .= cfg.pp_lowers
        pp.uppers .= cfg.pp_uppers
        pp.params .= cfg.pp_params
        pt.lowers .= cfg.pt_lowers
        pt.uppers .= cfg.pt_uppers
    end

    return lp, cfg.dx
end
