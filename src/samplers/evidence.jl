# Reddemcee-style evidence estimators for PT.
#
# Three estimators, all driven by per-temperature accumulators populated
# by `_sample_pt_transdim`. Refs:
#
#   Peña & Jenkins 2026, A&A 706 A323 ("Closing the evidence gap:
#   reddemcee"), arXiv:2509.24870. Equations cited below.
#
# - ti_plus     : Curvature-aware TI. PCHIP-interpolate ⟨log L⟩(β) on the
#                 PT inverse-temperature ladder and integrate the smooth
#                 curve analytically over each Hermite segment. Estimate
#                 discretization error by Richardson-extrapolating on a
#                 coarse ladder (every other β).
#
# - ss_plus     : Geometric-bridge stepping stones (Eq. 22). At pair
#                 (k, k+1) with Δβ = β_{k+1} - β_k:
#                   r_k = E_{β_k}[L^{Δβ/2}] / E_{β_{k+1}}[L^{-Δβ/2}]
#                 Both expectations are streamed via logsumexp; halving
#                 the exponent reduces variance vs the one-leg standard SS.
#
# - hybrid      : H+ (Eq. 23-24). Find β* = first interior knot where
#                 the integrand satisfies 2 U_k ≥ U_{k+1} (rectangular
#                 area dominates triangular). Use TI+ on [0, β*] and
#                 SS+ on [β*, 1] and add.
#
# All three operate on an `EvidenceAccumulator` that the sampler updates
# once per (post-warmup) iteration. Memory cost: O(n_chains), independent
# of n_iter.

"""
    EvidenceAccumulator

Per-inverse-temperature accumulators for reddemcee-style log-Z estimators.
Memory is O(n_chains), independent of iteration count.

Fields:
- `n`           : per-β count of accumulated samples
- `sum_logL`    : per-β running sum of `log L` → mean from this and n
- `sum_logL2`   : per-β running sum of `(log L)^2` → variance from these
- `ss_lse_fwd`  : per-pair running logsumexp of  Δβ_k/2 · log L_k (chain k)
- `ss_lse_bwd`  : per-pair running logsumexp of −Δβ_k/2 · log L_{k+1} (chain k+1)
- `started`     : whether any update has been made (prevents 0/0)
"""
mutable struct EvidenceAccumulator
    n::Vector{Int}
    sum_logL::Vector{Float64}
    sum_logL2::Vector{Float64}
    ss_lse_fwd::Vector{Float64}
    ss_lse_bwd::Vector{Float64}
    started::Bool
    n_nonfinite::Vector{Int}      # dropped draws per rung — see update_evidence!
end

function EvidenceAccumulator(n_chains::Int)
    n_pairs = max(0, n_chains - 1)
    return EvidenceAccumulator(
        zeros(Int, n_chains),
        zeros(Float64, n_chains),
        zeros(Float64, n_chains),
        fill(-Inf, n_pairs),
        fill(-Inf, n_pairs),
        false,
        zeros(Int, n_chains),
    )
end

"""Numerically stable log(exp(a) + exp(b)) — defensive against -Inf."""
@inline function _logaddexp(a::Float64, b::Float64)
    a == -Inf && return b
    b == -Inf && return a
    m = max(a, b)
    return m + log1p(exp(-abs(a - b)))
end

"""
    update_evidence!(acc, log_L_per_chain, betas)

Streaming update. `log_L_per_chain[k]` is the current log-likelihood of
the replica at temperature `betas[k]`. Call once per iteration after
warmup.
"""
function update_evidence!(acc::EvidenceAccumulator,
                          log_L_per_chain::AbstractVector{<:Real},
                          betas::AbstractVector{<:Real})
    n_chains = length(betas)
    @assert length(log_L_per_chain) == n_chains
    acc.started = true
    @inbounds for k in 1:n_chains
        L = Float64(log_L_per_chain[k])
        # SKIP non-finite draws. A single -Inf (a replica seeded or pushed out of
        # support) otherwise lands in sum_logL and makes that rung's mean -Inf for
        # the rest of the run, which propagates to TI/TI+/SS+/H+ as -Inf or NaN.
        # Counting it in acc.n as well would bias the mean toward the rejected
        # region, so the draw is dropped from both and tallied separately.
        if isfinite(L)
            acc.n[k]         += 1
            acc.sum_logL[k]  += L
            acc.sum_logL2[k] += L * L
        else
            acc.n_nonfinite[k] += 1
        end
    end
    @inbounds for k in 1:(n_chains - 1)
        dβ_half = (betas[k+1] - betas[k]) / 2
        L_k     = Float64(log_L_per_chain[k])
        L_k1    = Float64(log_L_per_chain[k+1])
        acc.ss_lse_fwd[k] = _logaddexp(acc.ss_lse_fwd[k],  dβ_half * L_k)
        acc.ss_lse_bwd[k] = _logaddexp(acc.ss_lse_bwd[k], -dβ_half * L_k1)
    end
    return acc
end

# ---------------------------------------------------------------------
# Standard TI (trapezoidal) — kept as a reference / sanity check.
# ---------------------------------------------------------------------

"""
    ti_trapezoidal(mean_logL, betas) -> log_Z

Standard thermodynamic integration with the trapezoidal rule (reddemcee
Eq. 5). Used as the baseline and as the building block for H+ when
applied to a sub-range.
"""
function ti_trapezoidal(mean_logL::AbstractVector{<:Real},
                          betas::AbstractVector{<:Real})
    n = length(betas)
    n < 2 && return 0.0
    @assert length(mean_logL) == n
    # See _pchip_integral: integrate ascending in β so logZ has the right
    # sign for the DESCENDING PT ladder (β: 1→0).
    b = betas; m = mean_logL
    if betas[1] > betas[end]
        p = sortperm(collect(Float64, betas))
        b = collect(Float64, betas)[p]; m = collect(Float64, mean_logL)[p]
    end
    log_z = 0.0
    @inbounds for k in 1:(n - 1)
        dβ = b[k+1] - b[k]
        log_z += dβ * (m[k] + m[k+1]) / 2
    end
    return log_z
end

# ---------------------------------------------------------------------
# PCHIP — monotonicity-preserving cubic Hermite (Fritsch & Carlson 1980).
# We need the analytic integral over each segment, not pointwise values,
# so we implement the slope rule and the closed-form Hermite integral
# directly instead of pulling in Interpolations.jl. ~30 LOC.
# ---------------------------------------------------------------------

"""
    _pchip_derivatives(x, y) -> derivs

Compute Fritsch-Carlson monotonic Hermite derivatives at each knot.
Equivalent to scipy's `PchipInterpolator`. Output has same length as x.
"""
function _pchip_derivatives(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    @assert n == length(y) && n ≥ 2
    d = zeros(Float64, n)
    if n == 2
        d[1] = d[2] = (y[2] - y[1]) / (x[2] - x[1])
        return d
    end
    h = [x[k+1] - x[k]      for k in 1:(n-1)]
    Δ = [(y[k+1] - y[k]) / h[k] for k in 1:(n-1)]
    # Interior knots: Brodlie / Fritsch-Carlson weighted harmonic mean.
    for k in 2:(n-1)
        if Δ[k-1] * Δ[k] > 0
            w1 = 2*h[k] + h[k-1]
            w2 = h[k]   + 2*h[k-1]
            d[k] = (w1 + w2) / (w1/Δ[k-1] + w2/Δ[k])
        else
            d[k] = 0.0  # local extremum: zero slope preserves monotonicity
        end
    end
    # Endpoints: 3-point one-sided formula, clipped for monotonicity.
    d[1] = ((2*h[1] + h[2]) * Δ[1] - h[1] * Δ[2]) / (h[1] + h[2])
    if sign(d[1]) != sign(Δ[1])
        d[1] = 0.0
    elseif sign(Δ[1]) != sign(Δ[2]) && abs(d[1]) > 3*abs(Δ[1])
        d[1] = 3 * Δ[1]
    end
    d[n] = ((2*h[n-1] + h[n-2]) * Δ[n-1] - h[n-1] * Δ[n-2]) / (h[n-1] + h[n-2])
    if sign(d[n]) != sign(Δ[n-1])
        d[n] = 0.0
    elseif sign(Δ[n-2]) != sign(Δ[n-1]) && abs(d[n]) > 3*abs(Δ[n-1])
        d[n] = 3 * Δ[n-1]
    end
    return d
end

"""
    _pchip_integral(x, y) -> ∫ y dx

Closed-form integral of the PCHIP interpolant from x[1] to x[end].
Each cubic Hermite piece on [x_k, x_{k+1}] with values y_k, y_{k+1}
and derivatives d_k, d_{k+1} integrates to:

    ∫ p(s) ds = (h/2)(y_k + y_{k+1}) + (h²/12)(d_k − d_{k+1})

where h = x_{k+1} − x_k. This is exact for the cubic Hermite.
"""
function _pchip_integral(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    n < 2 && return 0.0
    # ∫⟨logL⟩dβ is defined from β=0 to β=1. The PT ladder is stored
    # DESCENDING (β: 1→0, cold→hot), so integrating in array order yields
    # ∫₁⁰ = −logZ. Sort ascending so the result is +logZ regardless of
    # ladder direction — the sign-flip bug: sample_ptemcee reported −logZ,
    # inverting every model-comparison Bayes factor (caught by toy-Z gate
    # vs nested sampling, 2026-06-13).
    if x[1] > x[end]
        p = sortperm(collect(Float64, x))
        x = collect(Float64, x)[p]; y = collect(Float64, y)[p]
    end
    d = _pchip_derivatives(x, y)
    I = 0.0
    @inbounds for k in 1:(n-1)
        h = x[k+1] - x[k]
        I += (h / 2) * (y[k] + y[k+1]) + (h^2 / 12) * (d[k] - d[k+1])
    end
    return I
end

# ---------------------------------------------------------------------
# TI+ — curvature-aware TI with discretization error estimate.
# ---------------------------------------------------------------------

"""
    ti_plus(acc, betas) -> (log_Z, σ_Z)

Curvature-aware TI (reddemcee §2; arXiv:2509.24870).

- Interpolate ⟨log L⟩(β) with PCHIP and integrate analytically over each
  segment. This is the "TI+" estimator.
- Discretization error σ_D: redo on a coarse ladder (every other β) and
  take |Z_fine − Z_coarse| as the systematic component.
- Statistical error σ_S: trapezoidal-weighted sum of per-β chain SE on
  ⟨log L⟩_β. Assumes independence between chains (conservative).
- Total: σ_Z = √(σ_D² + σ_S²).
"""
function ti_plus(acc::EvidenceAccumulator, betas::AbstractVector{<:Real})
    n = length(betas)
    n < 2 && return (0.0, 0.0)
    acc.started || return (0.0, Inf)

    mean_logL = [acc.sum_logL[k] / max(1, acc.n[k]) for k in 1:n]
    var_logL  = [max(0.0,
                     acc.sum_logL2[k] / max(1, acc.n[k]) - mean_logL[k]^2)
                  for k in 1:n]

    # Fine-grid PCHIP integral
    log_Z_fine = _pchip_integral(collect(Float64, betas), mean_logL)

    # Coarse-grid PCHIP integral (every other knot, anchored at endpoints)
    if n ≥ 5
        coarse_idx = 1:2:n
        if last(coarse_idx) != n
            coarse_idx = vcat(collect(coarse_idx), [n])
        end
        β_c = [Float64(betas[i]) for i in coarse_idx]
        m_c = [mean_logL[i] for i in coarse_idx]
        log_Z_coarse = _pchip_integral(β_c, m_c)
        σ_D = abs(log_Z_fine - log_Z_coarse)
    else
        σ_D = 0.0
    end

    # Per-β chain SE on ⟨log L⟩_β: σ²_⟨L⟩ ≈ Var(log L) / n.
    # Propagate through trapezoidal weights (PCHIP is local so the
    # trapezoidal-weight bound is a good approximation).
    σ_S_sq = 0.0
    @inbounds for k in 1:n
        w  = 0.0
        if k == 1
            w = (betas[2] - betas[1]) / 2
        elseif k == n
            w = (betas[n] - betas[n-1]) / 2
        else
            w = (betas[k+1] - betas[k-1]) / 2
        end
        σ_S_sq += w^2 * (var_logL[k] / max(1, acc.n[k]))
    end
    σ_S = sqrt(σ_S_sq)

    return (log_Z_fine, sqrt(σ_D^2 + σ_S^2))
end

# ---------------------------------------------------------------------
# SS+ — geometric-bridge stepping stones (Eq. 22).
# ---------------------------------------------------------------------

"""
    ss_plus(acc, betas) -> (log_Z, σ_Z)

Geometric-bridge stepping-stones estimator (reddemcee Eq. 22).

For each adjacent pair (β_k, β_{k+1}) with Δβ = β_{k+1} − β_k:

    log r_k = log E_{β_k}[L^{Δβ/2}]  −  log E_{β_{k+1}}[L^{−Δβ/2}]

Each expectation is a logsumexp/n which the accumulator already streams.
Halving the exponent (vs the one-leg standard SS) reduces variance when
adjacent temperatures are well separated.

Uncertainty: per-pair MC error on the two logsumexp estimates is hard
to estimate without retaining samples; we return a coarse jackknife-
style bound by dropping each pair and bracketing.
"""
function ss_plus(acc::EvidenceAccumulator, betas::AbstractVector{<:Real})
    n = length(betas)
    n < 2 && return (0.0, 0.0)
    acc.started || return (0.0, Inf)
    log_Z = 0.0
    @inbounds for k in 1:(n-1)
        log_n_k  = log(max(1, acc.n[k]))
        log_n_k1 = log(max(1, acc.n[k+1]))
        log_r_k  = (acc.ss_lse_fwd[k] - log_n_k) -
                   (acc.ss_lse_bwd[k] - log_n_k1)
        log_Z += log_r_k
    end
    # Σ log r_k = log[Z(β_n)/Z(β_1)]. For the DESCENDING PT ladder
    # (β_1=1, β_n→0) that is log[1/Z] = −logZ; negate to recover +logZ
    # (the same sign-flip the TI integrators carry — see _pchip_integral).
    betas[1] > betas[end] && (log_Z = -log_Z)
    # Conservative uncertainty: dominant term from the smallest-n pair.
    # UNCERTAINTY. This used to return 1/sqrt(n_min), which is not an error on
    # log_Z at all -- it ignores the integrand entirely and produced values like
    # "+/- 0.00" next to an estimate that was wrong by hundreds of nats, i.e. the
    # most misleading output the estimator can emit.
    #
    # There is no cheap unbiased SE for a stepping-stone sum from streamed
    # logsumexps, so report an HONEST one: the spread of the per-pair bridge
    # terms scaled as an independent-sum error, which at least grows when the
    # rungs are poorly overlapped, plus NaN when it cannot be formed. A caller
    # that needs a real error should bootstrap retained samples.
    if n < 3
        return (log_Z, NaN)
    end
    terms = Float64[]
    @inbounds for k in 1:(n-1)
        lr = (acc.ss_lse_fwd[k] - log(max(1, acc.n[k]))) -
             (acc.ss_lse_bwd[k] - log(max(1, acc.n[k+1])))
        isfinite(lr) && push!(terms, lr)
    end
    σ_Z = length(terms) < 2 ? NaN :
          sqrt(length(terms)) * std(terms) / sqrt(max(1, minimum(acc.n)))
    return (log_Z, σ_Z)
end

# ---------------------------------------------------------------------
# H+ — hybrid TI on [0, β*] + SS on [β*, 1] (Eq. 23-24).
# ---------------------------------------------------------------------

"""
    hybrid_evidence(acc, betas) -> (log_Z, σ_Z, β_star)

Hybrid TI+ ⊕ SS+ estimator (reddemcee Eq. 23-24).

Selection rule for β*: walk the temperatures k = 1 … n−1 and stop at
the first interior k where the integrand satisfies

    2 ⟨log L⟩_k ≥ ⟨log L⟩_{k+1}

(rectangle dominates triangle — TI's quadrature error explodes past
this knot, so switch to SS for the high-β tail). If no such k exists,
fall back to pure TI+.
"""
function hybrid_evidence(acc::EvidenceAccumulator,
                          betas::AbstractVector{<:Real})
    n = length(betas)
    n < 3 && return (ti_plus(acc, betas)..., 0.0)
    mean_logL = [acc.sum_logL[k] / max(1, acc.n[k]) for k in 1:n]

    k_star = nothing
    for k in 1:(n-1)
        if 2 * mean_logL[k] ≥ mean_logL[k+1]
            k_star = k
            break
        end
    end
    if k_star === nothing
        # No crossover knot found: H+ is not a hybrid here, it IS ti_plus. It is
        # returned with beta_star = 1.0 so a caller can tell. The mirror case --
        # k_star == 1, where H+ degenerates to pure ss_plus and comes back
        # numerically identical to SS+ -- is handled below; when H+ and SS+ print
        # the same number to the last digit that is what happened, and they are
        # one estimator, not two agreeing ones.
        Z, σ = ti_plus(acc, betas)
        return (Z, σ, 1.0)
    end

    # TI+ on [β_1, β_{k_star}] using a sliced accumulator copy
    if k_star == 1
        log_Z_ti = 0.0
        σ_ti     = 0.0
    else
        acc_lo = EvidenceAccumulator(k_star)
        acc_lo.n           = acc.n[1:k_star]
        acc_lo.sum_logL    = acc.sum_logL[1:k_star]
        acc_lo.sum_logL2   = acc.sum_logL2[1:k_star]
        acc_lo.ss_lse_fwd  = acc.ss_lse_fwd[1:(k_star-1)]
        acc_lo.ss_lse_bwd  = acc.ss_lse_bwd[1:(k_star-1)]
        acc_lo.started     = acc.started
        log_Z_ti, σ_ti = ti_plus(acc_lo, betas[1:k_star])
    end

    # SS+ on [β_{k_star}, β_n]
    n_hi = n - k_star + 1
    acc_hi = EvidenceAccumulator(n_hi)
    acc_hi.n          = acc.n[k_star:n]
    acc_hi.sum_logL   = acc.sum_logL[k_star:n]
    acc_hi.sum_logL2  = acc.sum_logL2[k_star:n]
    acc_hi.ss_lse_fwd = acc.ss_lse_fwd[k_star:(n-1)]
    acc_hi.ss_lse_bwd = acc.ss_lse_bwd[k_star:(n-1)]
    acc_hi.started    = acc.started
    log_Z_ss, σ_ss = ss_plus(acc_hi, betas[k_star:n])

    log_Z = log_Z_ti + log_Z_ss
    σ_Z   = sqrt(σ_ti^2 + σ_ss^2)
    return (log_Z, σ_Z, Float64(betas[k_star]))
end

"""
    EvidenceReport

Bundle of all three reddemcee estimators returned alongside the chain.
Fields hold `(log_Z, σ_Z)` pairs except `hybrid_beta_star` which is
the H+ knot location.
"""
struct EvidenceReport
    ti::Tuple{Float64, Float64}         # trapezoidal TI baseline
    ti_plus::Tuple{Float64, Float64}    # PCHIP TI with Richardson error
    ss_plus::Tuple{Float64, Float64}    # geometric-bridge SS
    hybrid::Tuple{Float64, Float64}     # H+
    hybrid_beta_star::Float64
end

"""
    evidence_report(acc::EvidenceAccumulator, betas) -> EvidenceReport

Bundle the four β-path evidence estimators accumulated over a tempering
ladder into one result: `ti`, `ti_plus`, `ss_plus` and `hybrid`, each a
`(log_Z, sigma)` tuple, plus `hybrid_beta_star`, the knot where H+ switches
from TI+ to SS+.

`acc` is the [`EvidenceAccumulator`](@ref) filled during the run; `betas`
is the inverse-temperature ladder it was filled against.

!!! warning "Agreement between these four is not a check"
    All four read the same shared `mean_logL` array, so a bias in
    `<log L>` at any rung appears identically in every one of them. On a
    measured run the four agreed to 2.6 nats while all being 147 nats
    wrong. Corroborate against an estimator that does not temper --
    [`bridge_evidence`](@ref), [`reference_path_evidence`](@ref) or
    [`mode_laplace_evidence`](@ref) -- before quoting a number.

See the evidence guide for which estimator to report per sampler.
"""
function evidence_report(acc::EvidenceAccumulator, betas::AbstractVector{<:Real})
    n = length(betas)
    mean_logL = [acc.sum_logL[k] / max(1, acc.n[k]) for k in 1:n]
    ti0 = (ti_trapezoidal(mean_logL, betas), 0.0)
    tip = ti_plus(acc, betas)
    ssp = ss_plus(acc, betas)
    h_Z, h_σ, β_star = hybrid_evidence(acc, betas)
    return EvidenceReport(ti0, tip, ssp, (h_Z, h_σ), β_star)
end

function Base.show(io::IO, r::EvidenceReport)
    @printf(io, "EvidenceReport:\n")
    @printf(io, "  TI  (trapezoidal) = %12.4f\n", r.ti[1])
    @printf(io, "  TI+ (PCHIP)       = %12.4f ± %.4f\n", r.ti_plus[1], r.ti_plus[2])
    @printf(io, "  SS+ (geom-bridge) = %12.4f ± %.4f\n", r.ss_plus[1], r.ss_plus[2])
    @printf(io, "  H+  (β* = %.3f)   = %12.4f ± %.4f\n",
             r.hybrid_beta_star, r.hybrid[1], r.hybrid[2])
end
