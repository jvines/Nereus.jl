# Polynomial Stein Discrepancy (PSD) — Srinivasan, Sutton, Drovandi, South
# 2024 (arXiv:2412.05135).
#
# A linear-time goodness-of-fit and convergence diagnostic for posterior
# samples that detects discrepancies in the first r moments of the
# target. Cheaper than the kernel Stein discrepancy (KSD) and immune to
# the dimensional power-decay that afflicts KSD and its random-feature
# approximations.
#
# Theory. With the second-order Langevin-Stein operator
#
#     A g(x) = Δ g(x) + ∇g(x) · ∇log p(x)
#
# and polynomial test functions P_α(x) = ∏_j x_j^{α_j} indexed by
# multi-indices α with Σ α_j ≤ r and α ≠ 0, the discrepancy between a
# sample distribution Q and target P is
#
#     PSD² = Σ_k z̄_k²,    z̄_k = E_Q[A P_k(X)].
#
# Proposition 1 of the paper: for Gaussian P with positive-definite
# covariance, PSD = 0 iff the multi-index moments of P and Q match up to
# order r. Used here as a moment-convergence diagnostic complementing
# R-hat (cross-chain) and ESS (within-chain) — neither of which detect
# moment bias for asymptotically biased samplers (e.g. SG-MCMC) the way
# Stein discrepancies do.
#
# We implement the V-statistic and U-statistic estimators (Eq. 9 / Eq.
# 10 of the paper) plus the multinomial-bootstrap goodness-of-fit test
# (Eq. 11). Cost is O(n J) for n samples and J = (d+r choose d) - 1
# polynomials; the diagonal-only basis (no cross-monomials) reduces J
# to d × r and is recommended for d ≳ 20.

using Distributions: Multinomial
using LinearAlgebra: dot
using Random: AbstractRNG, default_rng
using Statistics: mean
using ForwardDiff

export polynomial_stein_discrepancy, psd_bootstrap_test, print_psd

# =====================================================================
# Multi-index basis
# =====================================================================

"""
    _generate_multiindices(d, r; diagonal_only=false) -> Vector{Vector{Int}}

Enumerate non-zero multi-indices α ∈ ℕ^d with Σα ≤ r. Each multi-index
labels a monomial test function P_α(x) = ∏_j x_j^{α_j}.

`diagonal_only=true` restricts the basis to single-variable monomials
(x_i, x_i², …, x_i^r); cost is O(d r) instead of O((d+r choose d)).
For Gaussian targets with diagonal covariance, the diagonal-only basis
already detects all marginal-moment differences (paper §3.1).
"""
function _generate_multiindices(d::Int, r::Int; diagonal_only::Bool=false)
    indices = Vector{Vector{Int}}()
    if diagonal_only
        for j in 1:d, k in 1:r
            α = zeros(Int, d)
            α[j] = k
            push!(indices, α)
        end
        return indices
    end
    α = zeros(Int, d)
    function _enumerate!(j::Int, remaining::Int)
        j == d + 1 && return
        for k in 0:remaining
            α[j] = k
            if j == d
                any(!iszero, α) && push!(indices, copy(α))
            else
                _enumerate!(j + 1, remaining - k)
            end
        end
        α[j] = 0
        return
    end
    _enumerate!(1, r)
    return indices
end

# =====================================================================
# Stein operator on monomials
# =====================================================================

"""
    _stein_apply_monomial(α, x, ∇log_p_x) -> A P_α(x)

Compute the Langevin-Stein operator applied to the monomial
P_α(x) = ∏_j x_j^{α_j} at point `x` with gradient of log-target
`∇log_p_x` of length `d`:

    A P_α(x) = Δ P_α(x) + ∇P_α(x) · ∇log p(x).

The Laplacian and gradient are computed analytically (cheap for
monomials — no AD needed). Returns a scalar in the element type of `x`.
"""
function _stein_apply_monomial(α::AbstractVector{Int},
                                x::AbstractVector{T},
                                ∇log_p_x::AbstractVector{T}) where {T}
    d = length(x)
    laplacian = zero(T)
    grad_dot_glogp = zero(T)
    @inbounds for j in 1:d
        αj = α[j]
        αj == 0 && continue
        # ∂P_α/∂x_j = α_j × x_j^{α_j-1} × ∏_{k≠j} x_k^{α_k}
        gj = T(αj)
        for k in 1:d
            xk = x[k]
            if k == j
                gj *= xk^(αj - 1)
            else
                αk = α[k]
                αk == 0 || (gj *= xk^αk)
            end
        end
        grad_dot_glogp += gj * ∇log_p_x[j]
        # ∂²P_α/∂x_j² = α_j(α_j - 1) × x_j^{α_j-2} × ∏_{k≠j} x_k^{α_k}
        if αj >= 2
            lj = T(αj * (αj - 1))
            for k in 1:d
                xk = x[k]
                if k == j
                    lj *= xk^(αj - 2)
                else
                    αk = α[k]
                    αk == 0 || (lj *= xk^αk)
                end
            end
            laplacian += lj
        end
    end
    return laplacian + grad_dot_glogp
end

# =====================================================================
# Sample-level PSD evaluation
# =====================================================================

"""
    _build_tau(samples, grad_log_p, basis) -> Matrix{T}  (J × n)

Evaluate τ(xᵢ) = (A P_1(xᵢ), …, A P_J(xᵢ)) for each sample. Each row of
`samples` is one draw; `grad_log_p` is a function of a d-vector
returning a d-vector.
"""
function _build_tau(samples::AbstractMatrix{T},
                     grad_log_p,
                     basis::Vector{Vector{Int}}) where {T}
    n, d = size(samples)
    J = length(basis)
    τ = Matrix{T}(undef, J, n)
    for i in 1:n
        x = view(samples, i, :)
        glogp = grad_log_p(Vector(x))
        length(glogp) == d || error(
            "grad_log_p returned length $(length(glogp)), expected $d")
        for k in 1:J
            τ[k, i] = _stein_apply_monomial(basis[k], x, glogp)
        end
    end
    return τ
end

"""
    polynomial_stein_discrepancy(samples, grad_log_p; degree=2,
                                  diagonal_only=false, statistic=:V) -> Float64

Squared Polynomial Stein Discrepancy between the empirical distribution
of `samples` (n × d matrix, one draw per row) and the target whose
log-density gradient is `grad_log_p(x)`.

`statistic = :V` returns the V-statistic Σ_k z̄_k², where
z̄_k = (1/n) Σᵢ A P_k(xᵢ). The V-statistic is non-negative.

`statistic = :U` returns the U-statistic estimator (Eq. 10 of Srinivasan
et al. 2024); it is unbiased and asymptotically normal under the
alternative, which makes it the right choice for the bootstrap
goodness-of-fit test (`psd_bootstrap_test`). The U-statistic can be
slightly negative on finite samples, even under H₀ — that is not a bug.

`degree` is the maximum polynomial order r. For Gaussian P, PSD = 0 iff
the first r marginal-moment-tensors of P and Q match.

`diagonal_only=true` uses the cheap d × r basis of single-variable
monomials. Recommended for d ≳ 20 or whenever cross-moment bias is not
a concern (Gaussian targets with diagonal covariance fall in this
regime).
"""
function polynomial_stein_discrepancy(
    samples::AbstractMatrix{T},
    grad_log_p;
    degree::Int = 2,
    diagonal_only::Bool = false,
    statistic::Symbol = :V,
) where {T <: Real}
    n, d = size(samples)
    n >= 2 || throw(ArgumentError("PSD requires n ≥ 2 samples; got $n"))
    degree >= 1 || throw(ArgumentError("PSD requires degree ≥ 1; got $degree"))
    statistic in (:V, :U) || throw(ArgumentError("statistic must be :V or :U"))

    basis = _generate_multiindices(d, degree; diagonal_only=diagonal_only)
    τ = _build_tau(samples, grad_log_p, basis)
    z̄ = vec(mean(τ; dims=2))
    sum_zbar2 = sum(abs2, z̄)
    if statistic === :V
        return sum_zbar2
    end
    # U-statistic: removes the i = j diagonal of (1/n²) ΣᵢΣⱼ Δ(xᵢ, xⱼ)
    sum_τ2 = sum(abs2, τ)               # Σ_k Σᵢ (A P_k(xᵢ))²
    return (n * n * sum_zbar2 - sum_τ2) / (n * (n - 1))
end

# =====================================================================
# Multinomial-bootstrap goodness-of-fit test
# =====================================================================

"""
    psd_bootstrap_test(samples, grad_log_p; degree=2, diagonal_only=false,
                        n_bootstrap=1000, rng=default_rng())
        -> (psd2 = …, psd2_boot = …, p_value = …, basis_size = …)

Multinomial-bootstrap goodness-of-fit test for H₀: samples are drawn
from the target with `grad_log_p`. Returns the observed squared U-stat
PSD², the bootstrap distribution, an upper-tail p-value with
Hodges-Lehmann correction, and the polynomial basis size J.

A small p-value indicates that the first `degree` moments of the
sample distribution disagree with the target; in the
biased-sampler setting (e.g. SG-MCMC, asymptotically biased adaptive
schemes) this is the diagnostic to watch.

The bootstrap uses centered multinomial weights ε_i = c_i/n − 1/n with
c ∼ Multinomial(n; 1/n,…,1/n) (Eq. 11 of Srinivasan et al. 2024); the
weights center to zero by construction.
"""
function psd_bootstrap_test(
    samples::AbstractMatrix{T},
    grad_log_p;
    degree::Int = 2,
    diagonal_only::Bool = false,
    n_bootstrap::Int = 1000,
    rng::AbstractRNG = default_rng(),
) where {T <: Real}
    n, d = size(samples)
    n >= 2 || throw(ArgumentError("PSD requires n ≥ 2 samples"))
    n_bootstrap >= 1 || throw(ArgumentError("n_bootstrap must be ≥ 1"))

    basis = _generate_multiindices(d, degree; diagonal_only=diagonal_only)
    J = length(basis)
    τ = _build_tau(samples, grad_log_p, basis)        # J × n
    z̄ = vec(mean(τ; dims=2))
    psd2_obs = (n * n * sum(abs2, z̄) - sum(abs2, τ)) / (n * (n - 1))

    boot = Vector{Float64}(undef, n_bootstrap)
    weights_dist = Multinomial(n, fill(1 / n, n))
    @inbounds for b in 1:n_bootstrap
        c  = rand(rng, weights_dist)
        ε  = (c ./ n) .- (1 / n)                       # centered weights
        wτ = τ * ε                                     # J-vector
        cross   = sum(abs2, wτ)
        diagsq  = sum(abs2, τ .* ε')                   # Σ_k Σᵢ (ε_i τ_{k,i})²
        boot[b] = cross - diagsq
    end

    p_value = (count(boot .>= psd2_obs) + 1) / (n_bootstrap + 1)
    return (psd2 = psd2_obs, psd2_boot = boot, p_value = p_value,
             basis_size = J)
end

# =====================================================================
# NereusTarget convenience wrappers
# =====================================================================

"""
    psd_bootstrap_test(target::NereusTarget, samples; …)

Convenience wrapper that constructs the gradient via ForwardDiff on
`target` and forwards the call. The samples must be expressed in the
parameter space the `target` operates in (typically `unconstrained=true`
for HMC/NUTS chains, since that is the space where ∇log p is defined
without boundary effects).
"""
function psd_bootstrap_test(target::NereusTarget, samples::AbstractMatrix;
                             kwargs...)
    grad_log_p = x -> ForwardDiff.gradient(target, x)
    return psd_bootstrap_test(samples, grad_log_p; kwargs...)
end

function polynomial_stein_discrepancy(target::NereusTarget,
                                       samples::AbstractMatrix; kwargs...)
    grad_log_p = x -> ForwardDiff.gradient(target, x)
    return polynomial_stein_discrepancy(samples, grad_log_p; kwargs...)
end

"""
    print_psd(result; io=stdout)

Print a PSD bootstrap-test result block.
"""
function print_psd(result::NamedTuple; io::IO = stdout)
    println(io, "\nPolynomial Stein Discrepancy (Srinivasan+ 2024):")
    @printf(io, "  PSD²              = %+.4e\n", result.psd2)
    @printf(io, "  bootstrap p-value = %.4f  (n_boot = %d, basis size J = %d)\n",
            result.p_value, length(result.psd2_boot), result.basis_size)
    if result.p_value < 0.01
        println(io, "  → samples FAIL the moment goodness-of-fit test")
    elseif result.p_value < 0.05
        println(io, "  → marginal moment fit; consider longer chains or smaller step size")
    else
        println(io, "  → moment fit consistent with target")
    end
end
