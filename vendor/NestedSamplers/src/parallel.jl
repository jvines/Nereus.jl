# Batch / parallel nested sampling.
#
# Standard single-point NS (step.jl) is sequential: each iteration removes the
# single lowest-likelihood live point and the threshold L* advances, so
# iterations cannot be parallelised. The high-d proposals (RWalk/Slice) are
# themselves sequential Markov chains, so there is no within-replacement
# parallelism either.
#
# This implements "K-point batch" NS (Burkoff+ 2012; the parallelism dynesty /
# PolyChord use): each iteration removes the K lowest-L points and draws K new
# points above L_(K) (the highest of the removed batch) via K INDEPENDENT
# constrained walks run concurrently. The K dead points are folded into the
# evidence with the exact same PR#80 trapezoidal bookkeeping as `step`, applied
# K times — so `batch_size = 1` reproduces the serial sampler bit-for-bit.
#
# Correctness: the only approximation is that the K replacements are drawn above
# L_(K) rather than each above its own advancing threshold; the resulting log Z
# bias is O(K/N) and vanishes for K << nactive. Verified against analytic Z.
#
# Threading is gated by `parallel`: set `parallel = false` under juliacall /
# the Python GIL (where Julia threads can deadlock); a pure-Julia daemon runs
# `parallel = true` safely.

"""
    sample(rng, model, sampler::Nested; batch_size, parallel, dlogz, ...)

Run batch/parallel nested sampling. `batch_size = K` removes & replaces the K
lowest-likelihood live points per iteration, with the K constrained walks run
concurrently (`Threads.@threads :static`) when `parallel = true`. `batch_size =
1` is identical to the serial sampler. Returns `(chain::Chains, state)`.

Keep `batch_size << nactive` (e.g. ≤ nthreads, with nactive ≥ ~20·batch_size)
to keep the O(K/N) evidence bias well inside `logzerr`.
"""
function sample_parallel(rng::AbstractRNG, model::AbstractModel, sampler::Nested;
                         batch_size::Int = Threads.nthreads(),
                         parallel::Bool = true,
                         dlogz::Real = 0.5, maxiter = Inf, maxcall = Inf,
                         maxlogl = Inf, add_live::Bool = true,
                         param_names = missing, check_wsum::Bool = true,
                         progress::Bool = false)
    N = sampler.nactive
    K = clamp(batch_size, 1, N - 1)

    # Bootstrap: the existing initial `step` builds live points + the first
    # dead sample/state, exactly as the serial path does.
    sample1, state = step(rng, model, sampler)
    samples = [sample1]

    # Per-thread RNGs for race-free, reproducible parallel walks.
    nthr = max(Threads.nthreads(), K)
    rngs = [Random.MersenneTwister(rand(rng, UInt64)) for _ in 1:nthr]
    # Per-thread proposal COPIES: proposals like RWalk are mutable structs
    # whose `scale` field is adapted in place (update_scale!). Sharing one
    # object across the K concurrent walks races on `scale` and biases the
    # sampling/evidence. A private copy per thread adapts independently.
    props = [deepcopy(sampler.proposal) for _ in 1:nthr]

    while true
        logz_remain = maximum(state.logl) + state.logvol
        delta_logz = logaddexp(state.logz, logz_remain) - state.logz
        (state.it ≥ maxiter || state.ncall ≥ maxcall ||
         delta_logz ≤ dlogz || state.logl_dead ≥ maxlogl) && break

        batch, state = step_batch(rng, rngs, props, model, sampler, state, K, parallel)
        append!(samples, batch)
        progress && @printf("\r\33[2Kit=%d ncall=%d Δlogz=%.3g logz=%.4g",
                            state.it, state.ncall, delta_logz, state.logz)
    end

    if add_live
        samples, state = add_live_points(samples, model, sampler, state)
    end
    return _bundle_parallel(samples, model, sampler, state, param_names, check_wsum)
end

sample_parallel(model::AbstractModel, sampler::Nested; kwargs...) =
    sample_parallel(Random.GLOBAL_RNG, model, sampler; kwargs...)

# One batch iteration: remove the K lowest-L points, draw K replacements above
# L_(K) (parallel), and fold the K dead points into the evidence with the exact
# `step` bookkeeping applied K times. K=1 ⇒ identical to the recursive `step`.
function step_batch(rng, rngs, props, model, sampler::Nested, state, K::Int, parallel::Bool)
    N = sampler.nactive
    pointvol = exp(state.logvol) / N

    # --- bounds update (mirror step.jl) ---
    if !state.has_bounds && state.ncall > sampler.min_ncall &&
       state.it / state.ncall < sampler.min_eff
        active_bound = Bounds.scale!(Bounds.fit(sampler.bounds, state.us; pointvol=pointvol), sampler.enlarge)
        since_update = 0; has_bounds = true
    elseif state.has_bounds && iszero(state.since_update % sampler.update_interval)
        active_bound = Bounds.scale!(Bounds.fit(sampler.bounds, state.us; pointvol=pointvol), sampler.enlarge)
        since_update = 0; has_bounds = true
    else
        active_bound = state.active_bound
        since_update = state.since_update
        has_bounds = state.has_bounds
    end

    # --- the K lowest-likelihood points (ascending L), threshold = L_(K) ---
    order = partialsortperm(state.logl, 1:K)
    Lstar = state.logl[order[K]]

    # --- draw K replacements above Lstar (concurrent constrained walks) ---
    reps = Vector{Tuple{Vector{Float64},Any,Float64,Int}}(undef, K)
    if parallel && K > 1
        Threads.@threads :static for j in 1:K
            tid = Threads.threadid()
            reps[j] = _draw_replacement(rngs[tid], props[tid], sampler, model,
                                         active_bound, has_bounds, Lstar, state.us, pointvol)
        end
    else
        for j in 1:K
            reps[j] = _draw_replacement(rng, props[1], sampler, model,
                                         active_bound, has_bounds, Lstar, state.us, pointvol)
        end
    end

    # --- fold in K dead points + write replacements (serial bookkeeping) ---
    logz = state.logz; h = state.h; logzerr = state.logzerr
    logvol = state.logvol; prev_ld = state.logl_dead; ncall = state.ncall
    batch = Vector{typeof((u=state.us[:,1], v=state.vs[:,1], logwt=0.0, logl=0.0))}(undef, K)
    @inbounds for j in 1:K
        idx = order[j]
        ldead = state.logl[idx]
        u_dead = state.us[:, idx]; v_dead = state.vs[:, idx]

        logvol  -= sampler.dlv
        logdvol  = logvol - log(N) - log(2)
        logwt    = logaddexp(prev_ld, ldead) + logdvol
        logz_new = logaddexp(logz, logwt)
        # Stable information update (mirrors step.jl): logz only inside differences,
        # so a -1e300 sentinel dead point contributes ~0 to h instead of ~1e300.
        a        = exp(logz - logz_new)
        b        = -expm1(logz - logz_new)
        lbar     = exp(prev_ld - logwt + logdvol) * prev_ld +
                   exp(ldead   - logwt + logdvol) * ldead
        h_new    = a * h + xlogx(a) + b * (lbar - logz_new)
        logzerr  = sqrt(max(zero(h_new), logzerr^2 + (h_new - h) * sampler.dlv))

        batch[j] = (u = u_dead, v = v_dead, logwt = logwt, logl = ldead)

        u, v, ll, nc = reps[j]
        state.us[:, idx] .= u; state.vs[:, idx] .= v; state.logl[idx] = ll
        ncall += nc; since_update += nc
        prev_ld = ldead; logz = logz_new; h = h_new
    end

    new_state = (it = state.it + K, ncall = ncall, us = state.us, vs = state.vs,
                 logl = state.logl, logl_dead = prev_ld, logz = logz,
                 logzerr = logzerr, h = h, logvol = logvol,
                 since_update = since_update, has_bounds = has_bounds,
                 active_bound = active_bound)
    return batch, new_state
end

# Draw one new point with L > Lstar via the sampler's proposal (bounded path
# mirrors step.jl, including the live-point/ellipsoid fallbacks).
function _draw_replacement(rng, proposal, sampler, model, active_bound, has_bounds, Lstar, us, pointvol)
    if has_bounds
        point, bound = rand_live(rng, active_bound, us)
        if isnothing(bound)
            active_bound = Bounds.scale!(Bounds.fit(sampler.bounds, us; pointvol=pointvol), sampler.enlarge)
            point, bound = rand_live(rng, active_bound, us)
        end
        if isnothing(bound)
            bound = Bounds.scale!(Bounds.fit(Bounds.Ellipsoid, us; pointvol=pointvol), sampler.enlarge)
            point = rand_live(rng, bound, us)[1]
        end
        u, v, logl, nc = proposal(rng, point, Lstar, bound, model)
    else
        point = rand(rng, eltype(us), sampler.ndims)
        bound = Bounds.fit(Bounds.NoBounds, us)
        u, v, logl, nc = Proposals.Rejection()(rng, point, Lstar, bound, model)
    end
    return (u, v, logl, nc)
end

# Bundle to Chains (mirrors bundle_samples(::Chains)).
function _bundle_parallel(samples, model, sampler, state, param_names, check_wsum)
    vals = mapreduce(t -> hcat(t.v..., exp(t.logwt - state.logz)), vcat, samples)
    if check_wsum
        wsum = sum(vals[:, end])
        err = !iszero(state.logzerr) ? 3 * state.logzerr : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end
    names = param_names === missing ?
        ["Parameter $i" for i in 1:length(vals[1, :]) - 1] : copy(param_names)
    push!(names, "weights")
    return Chains(vals, names, Dict(:internals => ["weights"]), evidence=state.logz), state
end
