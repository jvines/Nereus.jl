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
# Correctness: removing the K lowest of N points uniform in the prior volume X
# below L*, the j-th of them is the lowest of the N - j + 1 left once the j - 1
# below it are gone, so its volume is the previous one shrunk as by N - j + 1
# live points, and the K replacements drawn uniformly above L_(K) leave N points
# uniform in X_(K) again. Batch NS is exact with that count. It used to shrink
# each by N, as `step` does when every removal is replaced before the next:
# that credits K(K-1)/(2N^2) too little shrinkage per batch, and log Z came out
# about (K - 1) H / (2N) high, H the information.
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

The K dead points of a batch are credited the volume the K lowest of `nactive`
points take, so the evidence carries no bias from the batch size.

`callback`, if given, is called as AbstractMCMC calls it in the serial sampler,
`callback(rng, model, sampler, sample, state, i)`: once after the bootstrap step
and once after each batch, with `sample` the batch's dead points and
`i = state.it`.
"""
function sample_parallel(rng::AbstractRNG, model::AbstractModel, sampler::Nested;
                         batch_size::Int = Threads.nthreads(),
                         parallel::Bool = true,
                         dlogz::Real = 0.5, maxiter = Inf, maxcall = Inf,
                         maxlogl = Inf, add_live::Bool = true,
                         param_names = missing, check_wsum::Bool = true,
                         progress::Bool = false, callback = nothing)
    N = sampler.nactive
    K = clamp(batch_size, 1, N - 1)

    # Bootstrap: the existing initial `step` builds live points + the first
    # dead sample/state, exactly as the serial path does.
    sample1, state = step(rng, model, sampler)
    samples = [sample1]
    callback === nothing || callback(rng, model, sampler, sample1, state, state.it)

    # Per-thread RNGs for race-free, reproducible parallel walks. Sized by
    # `maxthreadid()`, NOT `nthreads()`: the walks below index by
    # `threadid()`, which is a global id spanning every threadpool, while
    # `nthreads()` counts the default pool only. Julia >= 1.12 starts an
    # interactive thread and numbers it first, so the workers run on ids
    # 2..nthreads+1 and an `nthreads()`-long vector is short by one — the
    # BoundsError was swallowed by Nereus's try/catch around
    # `sample_parallel`, leaving batch NS permanently and silently serial.
    nthr = max(Threads.maxthreadid(), K)
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
        callback === nothing || callback(rng, model, sampler, batch, state, state.it)
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
# L_(K) (parallel), and fold the K dead points into the evidence with `step`'s
# bookkeeping, the j-th shrinking the volume as N - j + 1 live points do (see
# the header). K=1 ⇒ identical to the recursive `step`.
function step_batch(rng, rngs, props, model, sampler::Nested, state, K::Int, parallel::Bool)
    N = sampler.nactive
    pointvol = exp(state.logvol) / N

    # --- bounds update (mirror step.jl) ---
    if !state.has_bounds && state.ncall > sampler.min_ncall &&
       state.it / state.ncall < sampler.min_eff
        active_bound = fit_bounds(rng, sampler, sampler.bounds, state.us, pointvol)
        since_update = 0; has_bounds = true
    elseif state.has_bounds && state.since_update >= sampler.update_interval  # see step.jl
        active_bound = fit_bounds(rng, sampler, sampler.bounds, state.us, pointvol)
        since_update = 0; has_bounds = true
    else
        active_bound = state.active_bound
        since_update = state.since_update
        has_bounds = state.has_bounds
    end

    # --- the K lowest-likelihood points (ascending L), threshold = L_(K) ---
    order = partialsortperm(state.logl, 1:K)
    Lstar = state.logl[order[K]]

    # --- where the K walks start, chosen here, serially ---
    # A start outside the bound refits the bound, the refit is kept, as `step`
    # keeps it, and the start stays. Each walk used to refit for itself alone
    # and draw another start, which chose the live points outside the bound
    # less often than the rest for as long as that bound lasted; when it
    # lasted the whole run, log Z came out 1.5-2.7 below pt_emcee's on an RM
    # obliquity fit. The walks start at live points above the threshold L_(K)
    # (`start_index`), not at the K dead points at or below it.
    starts = Vector{Any}(undef, K)
    if has_bounds
        for j in 1:K
            point = state.us[:, start_index(rng, state.logl, Lstar)]
            bound = Bounds.bound_containing(rng, active_bound, point)
            if isnothing(bound)
                active_bound = fit_bounds(rng, sampler, sampler.bounds, state.us, pointvol)
                since_update = 0
                bound = Bounds.bound_containing(rng, active_bound, point)
            end
            if isnothing(bound)
                bound = fit_bounds(rng, sampler, Bounds.Ellipsoid, state.us, pointvol)
            end
            starts[j] = (point, proposal_bound(sampler.proposal, active_bound, bound))  # see step.jl
        end
    end

    # --- draw K replacements above Lstar (concurrent constrained walks) ---
    reps = Vector{Tuple{Vector{Float64},Any,Float64,Int}}(undef, K)
    if parallel && K > 1
        Threads.@threads :static for j in 1:K
            tid = Threads.threadid()
            reps[j] = _draw_replacement(rngs[tid], props[tid], sampler, model,
                                         has_bounds ? starts[j] : nothing, Lstar, state.us, pointvol)
        end
    else
        for j in 1:K
            reps[j] = _draw_replacement(rng, props[1], sampler, model,
                                         has_bounds ? starts[j] : nothing, Lstar, state.us, pointvol)
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

        n_j      = N - j + 1                 # live points this one is the lowest of
        dlv      = log(n_j + 1) - log(n_j)   # sampler.dlv for j = 1
        logvol  -= dlv
        logdvol  = logvol - log(n_j) - log(2)
        logwt    = logaddexp(prev_ld, ldead) + logdvol
        logz_new = logaddexp(logz, logwt)
        # Stable information update (mirrors step.jl): logz only inside differences,
        # so a -1e300 sentinel dead point contributes ~0 to h instead of ~1e300.
        a        = exp(logz - logz_new)
        b        = -expm1(logz - logz_new)
        lbar     = exp(prev_ld - logwt + logdvol) * prev_ld +
                   exp(ldead   - logwt + logdvol) * ldead
        h_new    = a * h + xlogx(a) + b * (lbar - logz_new)
        logzerr  = sqrt(max(zero(h_new), logzerr^2 + (h_new - h) * dlv))

        batch[j] = (u = u_dead, v = v_dead, logwt = logwt, logl = ldead)

        u, v, ll, nc = reps[j]
        state.us[:, idx] .= u; state.vs[:, idx] .= v; state.logl[idx] = ll
        trace_remove!(state.trace, ldead, idx, Lstar)
        ncall += nc; since_update += nc
        prev_ld = ldead; logz = logz_new; h = h_new
    end

    new_state = (it = state.it + K, ncall = ncall, us = state.us, vs = state.vs,
                 logl = state.logl, logl_dead = prev_ld, logz = logz,
                 logzerr = logzerr, h = h, logvol = logvol,
                 since_update = since_update, has_bounds = has_bounds,
                 active_bound = active_bound, trace = state.trace)
    return batch, new_state
end

# Draw one new point with L ≥ Lstar via the sampler's proposal, from the start
# point and bound chosen in `step_batch`; uniform rejection from the cube before
# the first bound (`start === nothing`).
function _draw_replacement(rng, proposal, sampler, model, start, Lstar, us, pointvol)
    if start !== nothing
        point, bound = start
        if proposal isa Proposals.Rejection
            # refits during the search (see step.jl) stay with this draw
            u, v, logl, nc = draw_uniform(rng, sampler, proposal, Lstar, bound, model, us,
                                          pointvol, 0)
        else
            u, v, logl, nc = proposal(rng, point, Lstar, bound, model)
        end
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
