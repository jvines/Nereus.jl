# Bridge sampling: evidence from POSTERIOR SAMPLES ONLY.
#
# WHY THIS EXISTS. Every β-path estimator in evidence.jl (TI, TI+, SS+, H+)
# integrates from the prior to the posterior, so all of them depend on ⟨log L⟩
# at the hot end. On a joint RV+transit fit that expectation is
# E_prior[log L], and it is tail-dominated: prior draws on HD 18599 give a
# median log L of -4.9e4 with a minimum of -1.5e7. No ladder resolves a
# quantity whose variance is that large, and measurement bears it out — TI+
# sat 118 nats from the truth with swap acceptance 0.23 and a self-reported
# error of 0.48 nats, and adding rungs, cooling beta_min or enabling the
# adaptive ladder all moved it FURTHER away. That is a bias that does not
# shrink with computation, and it cannot be tuned out.
#
# The four estimators also all read one shared `mean_logL[k]` array, so a bias
# in ⟨log L⟩ per rung appears identically in every one of them. Their mutual
# agreement is therefore not evidence of correctness — on the run above they
# agreed with each other to 2.6 nats while all being 147 nats wrong.
#
# Bridge sampling never touches the prior end. It needs only samples already
# drawn from the posterior plus a proposal we can both sample and evaluate,
# and it costs `n_proposal` likelihood evaluations rather than the ~1.3e7 a
# nested-sampling run spends on the same model.
#
# METHOD. With p*(θ) the unnormalised posterior, q(θ) a normalised proposal,
# {θ_i} ~ p (N1 of them) and {φ_j} ~ q (N2), the optimal bridge (Meng & Wong
# 1996) solves the fixed point
#
#            (1/N2) Σ_j  l2_j / (s1·l2_j + s2·r)
#     r  =   ───────────────────────────────────  ,   l = p*/q,  s = N/(N1+N2)
#            (1/N1) Σ_i    1   / (s1·l1_i + s2·r)
#
# and r → Z. Iterated in logs so nothing overflows. Unlike the harmonic-mean
# estimator it has finite variance, and unlike Laplace it makes no Gaussianity
# assumption about the posterior — q only has to overlap it.

using Statistics: mean, std, cov
using LinearAlgebra: cholesky, logdet, Symmetric, I
using Random: MersenneTwister, randn, rand
using SpecialFunctions: loggamma

export bridge_evidence

"""
    bridge_evidence(target, chains; n_proposal, seed, proposal, max_iter, tol,
                    n_bootstrap) -> NamedTuple

Log-evidence by optimal bridge sampling from an existing posterior sample.

Returns `(; log_z, se, n_post, n_prop, iters, converged, overlap, proposal)`.

`chains` must be the β=1 chain in the target's own parameter space (that is what
every sampler here stores). The target must be `unconstrained=true`: the
proposal is Gaussian on ℝⁿ, so on a bounded parametrisation most proposal draws
would land outside support and be wasted.

`proposal` is `:gaussian` (mean/covariance of the posterior draws) or
`:student` (the same moments with heavier tails, ν=4). Prefer `:student` when
the posterior is skewed or heavy-tailed — an over-narrow proposal is the one
failure mode that biases bridge sampling badly, because the region where q has
no mass never gets sampled.

`overlap` is the fraction of proposal draws with finite posterior density. It is
the diagnostic that matters: below ~0.1 the proposal is a poor match and the
estimate should not be trusted, however tight `se` looks.

`se` is a bootstrap standard error over both sample sets. It is a real sampling
error, not a placeholder — but it says nothing about proposal mismatch, which is
what `overlap` is for.
"""
function bridge_evidence(target::NereusTarget, chains::MCMCChains.Chains;
                         n_proposal::Int = 20_000,
                         seed::Int = 1,
                         proposal::Symbol = :student,
                         max_iter::Int = 1_000,
                         tol::Real = 1e-10,
                         n_bootstrap::Int = 200)
    names = target.params.layout.unfrozen_names
    d = length(names)
    d >= 1 || return (; log_z = NaN, se = NaN, n_post = 0, n_prop = 0,
                        iters = 0, converged = false, overlap = NaN,
                        proposal = proposal)

    # --- posterior draws, in the target's own space --------------------------
    # Chains are stored in BOUNDED space; the target's log-density expects the
    # UNCONSTRAINED parametrisation (and folds in the Jacobian). Map forward, as
    # mode_laplace_evidence does. Skipping this is not a small error: a Gaussian
    # fitted to bounded samples escapes the box on essentially every draw in 22
    # dimensions, and the estimator returns overlap = 0 with no usable output.
    cols = [vec(Array(chains[:, Symbol(nm), :])) for nm in names]
    N = length(cols[1])
    pt = target.transform
    Y = Matrix{Float64}(undef, d, N)
    @inbounds for i in 1:N
        xi = Float64[cols[j][i] for j in 1:d]
        Y[:, i] = pt === nothing ? xi : transform_forward(xi, pt)
    end
    keep = vec(all(isfinite, Y; dims = 1))
    Y = Y[:, keep]
    N1 = size(Y, 2)
    N1 >= 2d + 2 || return (; log_z = NaN, se = NaN, n_post = N1, n_prop = 0,
                              iters = 0, converged = false, overlap = NaN,
                              proposal = proposal)

    logp(y) = (a = _logdensity_parts(target, y); Float64(a[1]) + Float64(a[2]))

    μ = vec(mean(Y; dims = 2))
    Σ = cov(Y; dims = 2) + 1e-10 * I(d)
    Lc = cholesky(Symmetric(Σ)).L
    ν  = 4.0

    # log q for the two proposal families, both normalised.
    function logq(y)
        z = Lc \ (y .- μ)
        q = sum(abs2, z)
        ld = logdet(Lc)
        if proposal === :gaussian
            return -0.5 * q - 0.5 * d * log(2π) - ld
        else
            return loggamma((ν + d) / 2) - loggamma(ν / 2) - 0.5 * d * log(ν * π) -
                   ld - 0.5 * (ν + d) * log1p(q / ν)
        end
    end

    rng = MersenneTwister(seed)
    # Student-t draw = Gaussian scaled by sqrt(nu / chi2_nu); chi2 with integer
    # nu is a sum of nu squared standard normals, so no extra dependency.
    nu_i = Int(ν)
    draw() = proposal === :gaussian ? μ .+ Lc * randn(rng, d) :
             μ .+ Lc * (randn(rng, d) / sqrt(sum(abs2, randn(rng, nu_i)) / ν))

    # --- log ratios l = log p* - log q on both sample sets --------------------
    l1 = Float64[]
    @inbounds for i in 1:N1
        y = @view Y[:, i]
        v = logp(y) - logq(y)
        isfinite(v) && push!(l1, v)
    end
    l2 = Float64[]
    n_finite = 0
    for _ in 1:n_proposal
        y = draw()
        v = logp(y)
        if isfinite(v)
            n_finite += 1
            push!(l2, v - logq(y))
        end
    end
    overlap = n_finite / n_proposal
    (isempty(l1) || isempty(l2)) &&
        return (; log_z = NaN, se = NaN, n_post = N1, n_prop = length(l2),
                  iters = 0, converged = false, overlap = overlap,
                  proposal = proposal)

    log_r, iters, converged = _bridge_iterate(l1, l2, max_iter, Float64(tol))

    # --- bootstrap standard error -------------------------------------------
    ses = Float64[]
    if n_bootstrap > 0
        rb = MersenneTwister(seed + 1)
        for _ in 1:n_bootstrap
            b1 = l1[rand(rb, 1:length(l1), length(l1))]
            b2 = l2[rand(rb, 1:length(l2), length(l2))]
            r, _, ok = _bridge_iterate(b1, b2, max_iter, Float64(tol))
            ok && isfinite(r) && push!(ses, r)
        end
    end
    se = length(ses) >= 2 ? std(ses) : NaN

    return (; log_z = log_r, se = se, n_post = length(l1), n_prop = length(l2),
              iters = iters, converged = converged, overlap = overlap,
              proposal = proposal)
end

# Fixed-point iteration in log space. l1 are log(p*/q) at posterior draws,
# l2 the same at proposal draws.
function _bridge_iterate(l1::Vector{Float64}, l2::Vector{Float64},
                         max_iter::Int, tol::Float64)
    n1, n2 = length(l1), length(l2)
    # Empty input is a caller error upstream (no finite draws on one side); return
    # a non-answer rather than throwing from inside a reduction.
    (n1 == 0 || n2 == 0) && return (NaN, 0, false)
    ls1 = log(n1 / (n1 + n2))
    ls2 = log(n2 / (n1 + n2))
    log_r = 0.5 * (mean(l1) + mean(l2))          # sane start, in the right decade
    iters = 0
    converged = false
    for it in 1:max_iter
        iters = it
        num = _logsumexp(Float64[l2[j] - _logaddexp(ls1 + l2[j], ls2 + log_r)
                                 for j in 1:n2]) - log(n2)
        den = _logsumexp(Float64[-_logaddexp(ls1 + l1[i], ls2 + log_r)
                                 for i in 1:n1]) - log(n1)
        new = num - den
        isfinite(new) || break
        if abs(new - log_r) < tol
            log_r = new; converged = true; break
        end
        log_r = new
    end
    return log_r, iters, converged
end

function _logsumexp(v::AbstractVector{<:Real})
    isempty(v) && return -Inf
    m = maximum(v)
    isfinite(m) || return m
    return m + log(sum(x -> exp(x - m), v))
end
