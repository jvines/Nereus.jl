
# Guarded x·log(x): IEEE gives 0*log(0)=0*(-Inf)=NaN, but the information update
# needs the limit xlogx(0)=0 (a zero-weight / -Inf-loglikelihood point carries no
# information). Used by the numerically-stable evidence/information recursion below
# (and mirrored in parallel.jl step_batch).
@inline xlogx(x) = iszero(x) ? zero(x) : x * log(x)

function step(rng, model, sampler::Nested; kwargs...)
    # Initialize particles
    # us are in unit space, vs are in prior space
    us, vs, logl = init_particles(rng, model, sampler)

    # Find least likely point
    logl_dead, idx_dead = findmin(logl)
    u_dead = us[:, idx_dead]
    v_dead = vs[:, idx_dead]

    # update weight using quadratic trapezoidal rule (PR #80)
    logvol = -sampler.dlv
    logdvol = logvol - log(sampler.nactive) - log(2)
    logwt = logl_dead + logdvol

    # sample a new live point without bounds
    bound = Bounds.fit(Bounds.NoBounds, us)
    proposal = Proposals.Rejection()
    u, v, ll, nc = proposal(rng, v_dead, logl_dead, bound, model)

    us[:, idx_dead] .= u
    vs[:, idx_dead] .= v
    logl[idx_dead] = ll

    ncall = since_update = nc

    # update evidence and information
    # Stable bootstrap: Z_old = 0 ⇒ a = 0, b = 1, lbar = logl_dead, so h = logl_dead - logz.
    # (logz enters only as a difference, so a -1e300 sentinel dead point gives h ≈ 0, not ~1e300.)
    logz = logwt
    h = logl_dead - logz
    logzerr = sqrt(max(zero(h), h * sampler.dlv))

    sample = (u = u_dead, v = v_dead, logwt = logwt, logl = logl_dead)
    state = (it = 1, ncall = ncall, us = us, vs = vs, logl = logl,
             logl_dead = logl_dead, logz = logz, logzerr = logzerr, h = h, logvol = logvol,
             since_update = since_update, has_bounds = false, active_bound = nothing)

    return sample, state
end

# Fit bounds of type `B` to the live points `us`, enlarged by the sampler's
# `enlarge` or its bootstrap expansion (`Bounds.fit`).
fit_bounds(rng, sampler::Nested, B, us, pointvol) =
    Bounds.fit(rng, B, us; pointvol = pointvol, enlarge = sampler.enlarge,
               bootstrap = sampler.bootstrap)

# The region a proposal is handed. `Rejection` draws uniformly from the whole
# bound: every ellipsoid weighted by its volume, a point in k of them kept with
# probability 1/k (`rand(::MultiEllipsoid)`). Handed the one ellipsoid around a
# random live point, as the walks are, it drew each ellipsoid as often as live
# points sit in it rather than by its volume, so a small ellipsoid holding many
# live points was drawn denser than the rest of the constrained region. The
# walks start at that live point and take only their steps' shape from it.
# The live point a walk starts from: uniformly among those above the threshold
# `logl_star`, as dynesty picks it. The dead point lies on the threshold, not
# inside the region a new point is drawn from, and a walk from it ends nearer
# the contour than such a draw; it used to be picked as often as any other.
# Any live point if none is above the threshold.
function start_index(rng::AbstractRNG, logl, logl_star)
    n = count(>(logl_star), logl)
    n == 0 && return rand(rng, eachindex(logl))
    k = rand(rng, 1:n)
    for i in eachindex(logl)
        if logl[i] > logl_star
            k -= 1
            k == 0 && return i
        end
    end
end

proposal_bound(::Proposals.Rejection, active_bound, bound) = active_bound
proposal_bound(_, active_bound, bound) = bound

# `Rejection`'s draw. dynesty refits the bound every `update_interval`
# likelihood calls while it searches for a point, not only between points,
# each refit with a fresh bootstrap expansion. Drawing from one bound until
# `maxiter` ran out instead: on an RM obliquity fit an expansion came out so
# large that 100 000 calls found no point. The refits are fitted to `us`, the
# live points this draw replaces one of. Returns the point, its calls, the
# bound it ended on and the calls since that bound was fitted, counting
# `since_update` made before this draw if there was no refit.
function draw_uniform(rng, sampler, prop::Proposals.Rejection, logl_star, bound, model, us,
                      pointvol, since_update)
    ncall = nout = 0
    since = since_update
    while true
        budget = clamp(sampler.update_interval - since, 1, prop.maxiter - ncall)
        u, v, logl, nc, no = Proposals.draw_rejection(rng, logl_star, bound, model, budget)
        ncall += nc; nout += no; since += nc
        u === nothing || return u, v, logl, ncall, bound, since
        (ncall ≥ prop.maxiter || nout ≥ 100 * prop.maxiter) &&
            throw(ErrorException("Couldn't generate a proper point after $ncall likelihood calls and $nout draws outside the unit cube, refitting the bounds every $(sampler.update_interval) calls: ellipsoids that cover this constrained region are too loose for uniform draws, so use a slice or walk proposal. Bounds=$bound, logl_star=$logl_star."))
        bound = fit_bounds(rng, sampler, sampler.bounds, us, pointvol)
        since = 0
    end
end

function step(rng, model, sampler, state; kwargs...)
    ## Update bounds
    pointvol = exp(state.logvol) / sampler.nactive
    # check if ready for first update
    if !state.has_bounds && state.ncall > sampler.min_ncall && state.it / state.ncall < sampler.min_eff
        @debug "First update: it=$(state.it), ncall=$(state.ncall), eff=$(state.it / state.ncall)"
        active_bound = fit_bounds(rng, sampler, sampler.bounds, state.us, pointvol)
        since_update = 0
        has_bounds = true
    # if accepted first update, is it time to update again? `since_update`
    # counts likelihood calls and grows by a whole proposal's worth at a time,
    # so it is compared, as in dynesty: testing it for an exact multiple of
    # `update_interval` refit the bounds almost never, and every proposal kept
    # drawing its steps and directions from the first bound.
    elseif state.has_bounds && state.since_update >= sampler.update_interval
        @debug "Updating bounds: it=$(state.it), ncall=$(state.ncall), eff=$(state.it / state.ncall)"
        active_bound = fit_bounds(rng, sampler, sampler.bounds, state.us, pointvol)
        since_update = 0
        has_bounds = true
    else
        active_bound = state.active_bound
        since_update = state.since_update
        has_bounds = state.has_bounds
    end

    ## Replace least-likely active point
    # Find least likely point
    logl_dead, idx_dead = findmin(state.logl)
    u_dead = state.us[:, idx_dead]
    v_dead = state.vs[:, idx_dead]

    # sample a new live point using bounds and proposal
    if has_bounds
        point = state.us[:, start_index(rng, state.logl, logl_dead)]
        bound = Bounds.bound_containing(rng, active_bound, point)
        if isnothing(bound)
            # live point not inside active bounds: refit them to the live
            # points, this one among them, and keep it. Drawing another point
            # instead chose the points a stale bound misses less often than
            # the rest, and a walk ends near where it starts.
            active_bound = fit_bounds(rng, sampler, sampler.bounds, state.us, pointvol)
            since_update = 0
            bound = Bounds.bound_containing(rng, active_bound, point)
        end
        if isnothing(bound)
            # still not bounded: fall back to single ellipsoid around all live points
            bound = fit_bounds(rng, sampler, Bounds.Ellipsoid, state.us, pointvol)
        end
        if sampler.proposal isa Proposals.Rejection
            u, v, logl, nc, active_bound, since_update =
                draw_uniform(rng, sampler, sampler.proposal, logl_dead, active_bound, model,
                             state.us, pointvol, since_update)
        else
            u, v, logl, nc = sampler.proposal(rng, point, logl_dead, bound, model)
            since_update += nc
        end
    else
        point = rand(rng, eltype(state.us), sampler.ndims)
        bound = Bounds.fit(Bounds.NoBounds, state.us)
        proposal = Proposals.Rejection()
        u, v, logl, nc = proposal(rng, point, logl_dead, bound, model)
        since_update += nc
    end

    state.us[:, idx_dead] .= u
    state.vs[:, idx_dead] .= v
    state.logl[idx_dead] = logl

    it = state.it + 1
    ncall = state.ncall + nc

    # update weight using quadratic trapezoidal rule (PR #80)
    logvol = state.logvol - sampler.dlv
    logdvol = logvol - log(sampler.nactive) - log(2)
    logwt = logaddexp(state.logl_dead, logl_dead) + logdvol

    # update evidence and information
    # Stable information update: reorganized so the running logz appears ONLY inside
    # differences before being scaled (shift-invariant in floating point; a -1e300
    # sentinel dead point contributes ~0 to h instead of inflating it to ~1e300).
    #   a = Z_old/Z_new ∈ [0,1],  b = 1-a = exp(logwt-logz),  lbar = trapezoid-avg dead logL.
    logz = logaddexp(state.logz, logwt)
    a = exp(state.logz - logz)
    b = -expm1(state.logz - logz)
    lbar = exp(state.logl_dead - logwt + logdvol) * state.logl_dead +
           exp(logl_dead - logwt + logdvol) * logl_dead
    h = a * state.h + xlogx(a) + b * (lbar - logz)
    logzerr = sqrt(max(zero(h), state.logzerr^2 + (h - state.h) * sampler.dlv))

    ## prepare returns
    sample = (u = u_dead, v = v_dead, logwt = logwt, logl = logl_dead)
    state = (it = it, ncall = ncall, us = state.us, vs = state.vs, logl = state.logl,
             logl_dead = logl_dead, logz = logz, logzerr = logzerr, h = h, logvol = logvol,
             since_update = since_update, has_bounds = has_bounds, active_bound = active_bound)

    return sample, state
end

function bundle_samples(samples,
        model::AbstractModel,
        sampler::Nested,
        state,
        ::Type{Chains};
        add_live=true,
        param_names=missing,
        check_wsum=true,
        kwargs...)

    if add_live
        samples, state = add_live_points(samples, model, sampler, state)
    end
    vals = mapreduce(t -> hcat(t.v..., exp(t.logwt - state.logz)), vcat, samples)

    if check_wsum
        wsum = sum(vals[:, end, 1])
        err = !iszero(state.logzerr) ? 3 * state.logzerr : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end

    # Parameter names
    if param_names === missing
        param_names = ["Parameter $i" for i in 1:length(vals[1, :]) - 1]
    end
    push!(param_names, "weights")

    return Chains(vals, param_names, Dict(:internals => ["weights"]), evidence=state.logz), state
end

function bundle_samples(samples,
        model::AbstractModel,
        sampler::Nested,
        state,
        ::Type{Array};
        add_live=true,
        check_wsum=true,
        kwargs...)

    if add_live
        samples, state = add_live_points(samples, model, sampler, state)
    end

    vals = mapreduce(t -> hcat(t.v..., exp(t.logwt - state.logz)), vcat, samples)

    if check_wsum
        wsum = sum(vals[:, end])
        err = !iszero(state.logzerr) ? 3 * state.logzerr : 1e-3
        isapprox(wsum, 1, atol=err) || @warn "Weights sum to $wsum instead of 1; possible bug"
    end


    return vals, state
end

## Helpers

init_particles(rng, ndims, nactive, model) =
    init_particles(rng, Float64, ndims, nactive, model)

init_particles(rng, model, sampler) =
    init_particles(rng, sampler.ndims, sampler.nactive, model)

# loop and fill arrays, checking validity of points
# will retry 100 times before erroring
function init_particles(rng, T, ndims, nactive, model)
    us = rand(rng, T, ndims, nactive)
    vs_and_logl = mapslices(
        Base.Fix1(prior_transform_and_loglikelihood, model), us;
        dims=1
    )
    vs = mapreduce(first, hcat, vs_and_logl)
    logl = dropdims(map(last, vs_and_logl), dims=1)

    ntries = 1
    while true
        any(isfinite, logl) && break
        rand!(rng, us)
        vs_and_logl .= mapslices(
            Base.Fix1(prior_transform_and_loglikelihood, model), us;
            dims=1
        )
        vs .= mapreduce(first, hcat, vs_and_logl)
        map!(last, logl, vs_and_logl)
        ntries += 1
        ntries > 100 && error("After 100 attempts, could not initialize any live points with finite loglikelihood. Please check your prior transform and loglikelihood methods.")
    end

    # force -Inf to be a finite but small number to keep estimators from breaking
    @. logl[logl == -Inf] = -1e300

    return us, vs, logl
end


# add remaining live points to `samples` (quadratic trapezoidal rule, PR #80)
function add_live_points(samples, model, sampler, state)
    prev_logvol = state.logvol
    prev_logz = state.logz
    prev_h = state.h
    prev_logzerr = state.logzerr
    prev_logl_dead = state.logl_dead

    local logz, h, logzerr, logl_dead, logvol
    N = length(samples)

    @inbounds for (i, idx) in enumerate(eachindex(state.logl))
        # get new point
        u = state.us[:, idx]
        v = state.vs[:, idx]
        logl_dead = state.logl[idx]

        # update weight using trapezoidal rule
        logvol = state.logvol + log1p(-i / (sampler.nactive + 1))
        dlv = prev_logvol - logvol
        logdvol = logvol + log(exp(dlv) - 1) - log(2)
        logwt = logaddexp(prev_logl_dead, logl_dead) + logdvol

        # update evidence and information (stable form; see step-with-state above)
        logz = logaddexp(prev_logz, logwt)
        a = exp(prev_logz - logz)
        b = -expm1(prev_logz - logz)
        lbar = exp(prev_logl_dead - logwt + logdvol) * prev_logl_dead +
               exp(logl_dead - logwt + logdvol) * logl_dead
        h = a * prev_h + xlogx(a) + b * (lbar - logz)
        logzerr = sqrt(max(zero(h), prev_logzerr^2 + (h - prev_h) * dlv))

        prev_logvol = logvol
        prev_logz = logz
        prev_h = h
        prev_logzerr = logzerr
        prev_logl_dead = logl_dead

        sample = (u = u, v = v, logwt = logwt, logl = logl_dead)
        save!!(samples, sample, N + i, model, sampler)
    end

    state = (it = state.it + sampler.nactive, us = state.us, vs = state.vs, logl = state.logl,
             logl_dead = logl_dead, logz = logz, logzerr = logzerr, h = h, logvol = logvol,
             since_update = state.since_update, has_bounds = state.has_bounds, active_bound = state.active_bound)
    return samples, state
end
