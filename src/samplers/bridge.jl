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

Whether the evaluator's photometry goes through the workspace method
`transit_log_likelihood(theta, data, ws)`. Not when a planet has gravity
darkening (`:GD`): those fits take `transit_log_likelihood(theta, data)`.

The workspace method had no gravity-darkened transit when the bridge first
used it, and it handed over to the other method only for TTVs, supersampled
exposures (then longer than 2 min, now longer than 3 min) and photometric noise
models. A gravity-darkened fit with no supersampled cadence therefore took the
plain transit there, and its log density did not depend on i_star or lambda. The workspace method now computes
each cadence's gravity-darkened flux with the same calls as the allocating
one. Gravity-darkened fits still take the allocating method here, which keeps
their photometry at the bits of `_logdensity_parts`; through the workspace it
would match to rounding, not always to the bit (the summation order on light
curves longer than 4096 cadences).
"""
_bridge_phot_ws(params) = !any(has_gd, params.config.planet_modes)

"""
    _bridge_e_clamped(theta) -> Bool

Whether `true_anomaly` (orbit.jl) clamps the eccentricity of any active planet,
that is whether some planet's e lies outside [0, 0.9999] (or is NaN). The
allocating RV and transit methods take the true anomaly from `true_anomaly`;
the workspace photometry, and the workspace RV of a fit with no noise models,
compute cos f and sin f from E with the e they are given. Outside that interval
the two compute different models, not the same model with different rounding.

The planet blocks are abstractly typed, so `planet_e_w` is not inferred and its
result is boxed. A call allocates what `planet_e_w(theta, k)::Tuple{T, T}` does
for each planet, and nothing for the comparison: 64 bytes per planet. With
:sesinw that is all `planet_e_w` costs; with :ew `planet_e_w` costs 32 bytes
and the assertion boxes the tuple for the other 32.
"""
@inline function _bridge_e_clamped(theta::Theta{T}) where {T}
    for k in planet_indices(theta)
        # The planet blocks are abstractly typed, so `planet_e_w` is not
        # inferred; the assertion keeps that from spreading to the test below.
        e, _ = planet_e_w(theta, k)::Tuple{T, T}
        # `true_anomaly`'s own clamp: true exactly when it changes e, and for NaN.
        _anomaly_e(e) == e || return true
    end
    return false
end

"""
    _bridge_logdensity!(ev, y) -> Float64

The log-posterior at `y` in the target's own space, `sum(_logdensity_parts(target,
y))` with the same guards, but through the workspace likelihoods the samplers
draw with (`rv_log_likelihood(theta, data, ws)`, `transit_log_likelihood(theta,
data, ws)`) and the evaluator's own `Theta`, so that a call does not allocate
the likelihood's buffers afresh.

It takes the allocating methods where the workspace methods compute a
different model, and for gravity-darkened photometry:

- Gravity darkening: a fit with a :GD planet takes the allocating photometry
  (`_bridge_phot_ws`). The workspace transit had no gravity darkening when
  the bridge first used it; it has now, and the allocating method keeps these
  fits at the bits of `_logdensity_parts`. TTV fits, exposures longer than
  3 min and photometric noise models are handed over by the workspace method
  itself.
- Eccentricity outside [0, 0.9999] (`_bridge_e_clamped`): the allocating
  methods take the true anomaly from `true_anomaly`, which clamps e to 0.9999;
  the workspace photometry, and the workspace RV of a fit with no noise models,
  use the given e. Above 0.9999 the two differ by much more than rounding: on an RV +
  transit fit with a 20k-point light curve, up to 8.6 nats in the RV and
  8.1e4 nats in the photometry; in the RV, 5.8 nats on an SB2 fit and 2.5 on
  the HD 18599 white-noise fit without the floor. At those points both the RV
  and the photometry take the allocating methods. This follows
  `_logdensity_parts`, as the bridge did before it used the workspace;
  pt_emcee's own likelihood (`eval_bounded!`) keeps the given e there.

With that it is the same function as `_logdensity_parts` up to rounding at
every point, but not to the bit:

- RV of a fit with no noise models at all: both RV methods choose their branch
  on `config.noise_models` as a whole, whatever channel each model is on. With
  none, the workspace method (`_rv_ll_no_noise(theta, data, ws)`) caches each
  planet's velocity curve and computes cos(f+ω) by the angle-sum identity, so
  its log L differs in the last bits at most points. Measured up to 2.4e-9
  nats on the HD 18599 white-noise fit without the floor, and 2.4e-7 nats
  (3e-15 relative) on prior draws of an RV + astrometry fit. With any noise
  model, on the RV or only on the photometry, the RV goes through
  `_rv_ll_with_noise`, which evaluates each cadence as the other method does,
  in the same order. That gives the same bits except where an active
  IndicatorFloor with `kernel = :qp` scores a channel, that is a floor
  channel that no active joint ActivityGP covers and that has indicator data
  and its amplitude and jitter slots. The workspace path builds that floor's
  kernel sines by the angle-difference identity, so the floor term, and with
  it the RV log L, differs in the last bits at most points. On 300 prior
  draws of each HD 18599 fit with such a floor the RV matched at 31 to 113 of
  them. With RV and photometry it differed by up to 1.5e-6 nats with activity
  decorrelation, 5.0e-7 with white noise, 4.5e-7 with the GP rotation and
  1.3e-7 with the error scale; with RV only, by up to 6.0e-7 with the GP
  rotation, 3.0e-7 with white noise, 2.8e-7 with the error scale and 1.8e-7
  with activity decorrelation. The floor term alone differed by the same
  amounts. At 200 points close to a high-posterior point of each fit the
  largest difference was 1.1e-10 nats. A floor whose every channel an active
  ActivityGP covers scores nothing in either method, so the RV gives the same
  bits: on the HD 18599 activity-GP fit, whose floor is active and has the
  ActivityGP's four channels, at every one of those 300 draws and 200
  points, and in the trans-dimensional job, whose menu always carries a `:qp`
  floor on the ActivityGP's channels, in every state with the ActivityGP on.
  In a state with the ActivityGP off the floor scores its channels, and the
  RV matched at 42 to 111 of 300 prior draws in each state the job can reach
  (floor alone, or with one of ErrorScale, CeleriteRotation or activity
  decorrelation), up to 5.0e-7 nats apart. With no `:qp` floor that scores a channel the RV gave the same bits
  on every fit measured (RM with tomography, and a CeleriteSHO on the
  photometry only, with and without gravity darkening).
- Photometry: the workspace method sums every cadence in one pass, where the
  other sums fixed chunks and then the chunk totals. Both take each cadence's
  sky separation from the same per-call orbit constants (`_sky_orbit`); when
  the figures here were measured the allocating method still took it from
  `sky_separation`, a different route. On the HD 18599 joint fits with a
  20k-point light curve the whole log density then differed by up to 5.4e-9
  nats near the reference point.

In the whole density, on 300 prior draws per HD 18599 joint fit (floor on,
RV and photometry together), the largest relative differences were 1.9e-11
on the white-noise fit, 2.1e-11 with the GP rotation, 1.8e-12 with the
activity GP and 4.7e-11 with activity decorrelation. On the three fits whose
floor scores its channels the RV floor term set those: at the draw that gave
each, the RV differed by 8.0e-8 nats (white noise), 6.8e-9 (GP rotation) and
1.9e-8 (activity decorrelation), the photometry by 7.3e-11, 1.3e-10 and
6.5e-10. With the activity GP, whose RV matches, the whole difference is the
photometry's.

All of these are sample maxima, not bounds. Relative figures are |difference|
/ max(1, |log p|), and rounding does not bound that ratio: where |log p| is
below 1, terms of thousands of nats cancel and leave their rounding. At such
points between the posterior and the prior of the HD 18599 fits it reached
3.6e-7. Relative to |log prior| + |log L_RV| + |log L_phot|, the
largest difference measured was 2.9e-11. On a two-planet RV + transit fit
(10k-point light curve, ρ⋆ and the mean anomaly as parameters, the Gladman
stability check) it was 1.9e-10 nats at a point near the best one where
log p = -18.2, and 1.2e-10 nats on a prior draw where log p = 2.46 (1.03e-11
and 4.7e-11 relative to max(1, |log p|)).

The same bits as `_logdensity_parts` come out for the tomogram, for the
photometry wherever it takes the allocating method, for the RV of a fit with
a noise model and no active `:qp` IndicatorFloor that scores a channel, and
for the whole density at a clamped e. A gravity-darkened fit, whose
photometry always takes the allocating method, is therefore bit-identical
whenever it has no RV data, or has a noise model on either channel and no
such floor. Only a gravity-darkened fit with RV data and either no noise
model at all or a `:qp` floor that scores a channel carries the workspace RV
rounding, like any other such fit: on a one-planet RVPM_GD fit the RV log L
matched at 72 of 420 points (up to 5.1e-11 nats apart); on a two-band,
two-planet one the whole density matched at 410 of 500 (up to 4.7e-10 nats,
1.4e-15 relative), and one bridge_evidence of five moved log Z by 4.5e-13.
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
    # Where `true_anomaly` clamps an eccentricity the workspace methods compute
    # another model, so both likelihoods take the allocating methods there.
    ws_ok = !_bridge_e_clamped(theta)
    ll = ws_ok ? rv_log_likelihood(theta, data, ev.ws) : rv_log_likelihood(theta, data)
    isfinite(ll) || return -Inf
    lt = ws_ok && ev.phot_ws ? transit_log_likelihood(theta, data, ev.ws) :
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
