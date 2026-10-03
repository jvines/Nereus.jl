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
# and it costs `N1 + n_proposal` likelihood evaluations rather than the ~1.3e7 a
# nested-sampling run spends on the same model -- where N1 is the number of
# posterior draws handed in, NOT just `n_proposal`. The estimator re-evaluates
# the posterior at every draw of the chain, so its cost grows with the chain:
# 170_000 evaluations for a 100-walker × 1500-step pt_emcee run. Both evaluation
# loops are threaded (see below), which is what keeps that affordable.
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
    Y_all = Matrix{Float64}(undef, d, N)
    @inbounds for i in 1:N
        xi = Float64[cols[j][i] for j in 1:d]
        Y_all[:, i] = pt === nothing ? xi : transform_forward(xi, pt)
    end
    keep = vec(all(isfinite, Y_all; dims = 1))
    # Bound once: the threaded evaluation below captures `Y`, and a captured
    # variable that is assigned twice is boxed.
    Y = Y_all[:, keep]
    N1 = size(Y, 2)
    N1 >= 2d + 2 || return (; log_z = NaN, se = NaN, n_post = N1, n_prop = 0,
                              iters = 0, converged = false, overlap = NaN,
                              proposal = proposal)

    # The posterior through the workspace likelihoods, one evaluator (Theta +
    # PTWorkspace) per evaluation task. See `_bridge_logdensity!`.
    evs = [_BridgeEvaluator(target) for _ in 1:_bridge_ntasks(max(N1, n_proposal))]

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
    # Each term is the full posterior, so this costs N1 + `n_proposal` LIKELIHOOD
    # EVALUATIONS -- not `n_proposal`, which is what this function's docstring
    # and pt_emcee's call site both used to claim. N1 is the ENTIRE kept chain:
    # 150_000 draws for a 100-walker × 1500-post-burnin-step run, 525_000 for a
    # 150-walker × 3500-step one. That made this the most expensive single thing
    # in a fit -- 170_000 evals, 42 s measured on a Gaia DR4 IAD posterior,
    # against 89 s for the whole 3.3M-evaluation MCMC that produced it, because
    # the MCMC has 12 threads and this loop had one.
    #
    # Each task evaluates with its own Theta and workspace and touches no shared
    # state, so the evaluations thread. The PROPOSAL DRAWS do not: they come off
    # a single MersenneTwister, and drawing them out of order would change the
    # numbers.
    # So they are drawn serially up front and only the evaluations are spread
    # out, each writing its own slot, and both arrays are filtered in index
    # order afterwards -- so what reaches `_bridge_iterate` is bit-identical to
    # the serial version's, at any thread count. See `_bridge_eval!` for how
    # the threads are used.
    v1 = Vector{Float64}(undef, N1)
    _bridge_eval!(v1) do c, i
        y = @view Y[:, i]
        _bridge_logdensity!(evs[c], y) - logq(y)
    end
    l1 = Float64[v for v in v1 if isfinite(v)]

    Yp = Matrix{Float64}(undef, d, n_proposal)
    @inbounds for i in 1:n_proposal
        Yp[:, i] = draw()
    end
    v2 = Vector{Float64}(undef, n_proposal)
    _bridge_eval!(v2) do c, i
        _bridge_logdensity!(evs[c], @view Yp[:, i])
    end
    l2 = Float64[]
    n_finite = 0
    @inbounds for i in 1:n_proposal
        y = @view Yp[:, i]
        v = v2[i]
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
    # Only when the primary estimate CONVERGED. `se` is the sampling error OF
    # `log_r`, and every caller discards `log_r` outright when `converged` is
    # false (pt_emcee.jl gates on `b.converged && isfinite(b.log_z)`), so
    # bootstrapping a non-converged estimate prices 200 extra fixed-point solves
    # into an answer nobody reads -- and it is exactly the case where each of
    # them runs the full `max_iter`, because a fixed point that is not there at
    # 1000 iterations on the real data is not there on a resample of it either.
    # Measured on the HD 114762 joint fit (525k posterior draws, no fixed point):
    # 885 s of the 1064 s the sampler took, all of it thrown away. Convergence
    # either happens in a handful of iterations or not at all.
    ses = Float64[]
    if n_bootstrap > 0 && converged && isfinite(log_r)
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

# Block of indices a thread takes from the shared counter in `_bridge_eval!`.
# Small enough that the last blocks even out the finish across threads (a
# joint-fit evaluation is ~1 ms), large enough that the counter is never
# contended.
const _BRIDGE_EVAL_BLOCK = 16

"""
    _bridge_ntasks(n) -> Int

Number of tasks `_bridge_eval!` runs for `n` evaluations: one per thread, but
never more than there are blocks of work.
"""
_bridge_ntasks(n::Integer) = n <= 0 ? 0 : min(_nthread_chunks(), cld(n, _BRIDGE_EVAL_BLOCK))

"""
    _bridge_eval!(f, out) -> out

`out[i] = f(c, i)` for every index of `out`, on all threads, where `c` in
`1:_bridge_ntasks(length(out))` numbers the task making the call, so that `f`
can keep scratch per task.

One task per thread, each taking blocks of `_BRIDGE_EVAL_BLOCK` indices from a
shared counter until none are left, so a thread on a slow core or one that
draws cheap points (a proposal outside the support returns at the prior) does
not hold up the rest. Every `out[i]` is computed by one call and written once,
so the result does not depend on which thread computed it or on the thread
count.

When there is a task for every thread, each one is marked with
`_serial_inner_loops!`: every thread already has evaluations of its own, and
the photometry's threaded reduction (`transit_likelihood.jl`) would otherwise
spawn one task per thread on every call, which only queue behind the other
threads' evaluations and allocate. That reduction sums fixed chunks combined
in order, so running it serially gives the same bits.
"""
function _bridge_eval!(f::F, out::AbstractVector{Float64}) where {F}
    n = length(out)
    n_tasks = _bridge_ntasks(n)
    n_tasks == 0 && return out
    fills_threads = n_tasks >= Threads.nthreads()
    next = Threads.Atomic{Int}(1)
    Threads.@threads for c in 1:n_tasks
        fills_threads && _serial_inner_loops!()
        while true
            lo = Threads.atomic_add!(next, _BRIDGE_EVAL_BLOCK)
            lo > n && break
            for i in lo:min(lo + _BRIDGE_EVAL_BLOCK - 1, n)
                @inbounds out[i] = f(c, i)
            end
        end
    end
    return out
end

"""
    _BridgeEvaluator(target)

One evaluation task's scratch for `_bridge_logdensity!`: a `Theta` and a
`PTWorkspace`, as each slot of `pt_emcee` has, and whether the photometry can
go through the workspace method (`_bridge_phot_ws`).
"""
struct _BridgeEvaluator{TT<:NereusTarget}
    target::TT
    theta::Theta{Float64}
    ws::PTWorkspace
    phot_ws::Bool
end

function _BridgeEvaluator(target::NereusTarget)
    params, data = target.params, target.data
    ws = PTWorkspace(params, params.config.max_kplanet, length(params.config.noise_models);
                     n_obs = length(data.t_rv), n_phot = length(data.t_phot))
    return _BridgeEvaluator(target, Theta{Float64}(params), ws, _bridge_phot_ws(params))
end

"""
    _bridge_phot_ws(params) -> Bool

Whether `transit_log_likelihood(theta, data, ws)` computes the same model as
`transit_log_likelihood(theta, data)` for this fit. It does not when a planet
has gravity darkening (`:GD`): the workspace method has no gravity-darkened
transit, and it hands over to the other method only for TTVs, exposures
longer than 2 min and photometric noise models. A gravity-darkened fit with
2-min or shorter cadences therefore took the plain transit there, and its log
density did not depend on i_star or lambda.
"""
_bridge_phot_ws(params) = !any(has_gd, params.config.planet_modes)

"""
    _bridge_logdensity!(ev, y) -> Float64

The log-posterior at `y` in the target's own space, `sum(_logdensity_parts(target,
y))` with the same guards, but through the workspace likelihoods the samplers
draw with (`rv_log_likelihood(theta, data, ws)`, `transit_log_likelihood(theta,
data, ws)`) and the evaluator's own `Theta`, so that a call does not allocate
the likelihood's buffers afresh.

It is the same function as `_logdensity_parts` up to rounding, not to the bit:

- RV without noise models: the workspace method (`_rv_ll_no_noise(theta, data,
  ws)`) caches each planet's velocity curve and computes cos(f+ω) by the
  angle-sum identity, so its log L differs in the last bits at most points.
  Measured up to 2.4e-9 nats on the HD 18599 white-noise fit without the
  floor, and 2.4e-7 nats on prior draws of an RV + astrometry fit. With noise
  models it evaluates each cadence as the other method does, and gave the same
  bits on every fit measured (GP rotation, activity GP, activity
  decorrelation, error scale, white noise with the floor, the
  trans-dimensional job, RM with tomography).
- Photometry: the workspace method computes each cadence's sky separation by
  the same angle-sum route and sums every cadence in one pass, where the other
  sums fixed chunks and then the chunk totals. On the HD 18599 joint fits with
  a 20k-point light curve the whole log density differs by at most 5.4e-9 nats
  near the reference point, and on prior draws by at most 1e-11 relative
  (3.1e-13 with the GP rotation, 9.7e-12 with activity decorrelation).
- Gravity darkening: the workspace method has none, so a fit with a :GD planet
  takes the allocating photometry, as `_logdensity_parts` does
  (`_bridge_phot_ws`). So do TTV fits, exposures longer than 2 min and
  photometric noise models, which the workspace method hands over itself. In
  those cases the photometry gives the same bits as in `_logdensity_parts`, as
  the tomogram always does.
"""
function _bridge_logdensity!(ev::_BridgeEvaluator, y::AbstractVector)
    target = ev.target
    data = target.data
    theta = ev.theta
    pt = target.transform
    lj = 0.0
    if pt === nothing
        @inbounds for i in eachindex(y)
            isfinite(y[i]) || return -Inf
        end
        set_unfrozen!(theta, y)
    else
        x = transform_inverse(y, pt)
        @inbounds for i in eachindex(x)
            isfinite(x[i]) || return -Inf
        end
        lj = Float64(transform_logabsdetjac_inv(y, pt))
        isfinite(lj) || return -Inf
        set_unfrozen!(theta, x)
    end
    lp = log_prior(theta)
    isfinite(lp) || return -Inf
    ll = rv_log_likelihood(theta, data, ev.ws)
    isfinite(ll) || return -Inf
    lt = ev.phot_ws ? transit_log_likelihood(theta, data, ev.ws) :
                      transit_log_likelihood(theta, data)
    ltomo = tomogram_log_likelihood(theta, data)
    isfinite(ltomo) || return -Inf
    isfinite(lt) || return -Inf
    # Grouped as `_logdensity_parts` groups them: (prior + Jacobian) + likelihood.
    pj = pt === nothing ? lp : lp + lj
    return Float64(pj) + Float64(ll + lt + ltomo)
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
