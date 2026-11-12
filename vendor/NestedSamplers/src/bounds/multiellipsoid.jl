
"""
    Bounds.MultiEllipsoid([T=Float64], ndims)
    Bounds.MultiEllipsoid(::AbstractVector{Ellipsoid})

Use multiple [`Ellipsoid`](@ref)s in an optimal clustering to bound prior space. This implementation follows the MultiNest implementation outlined in Feroz et al. (2008,2009).[^3][^4] For more details about the bounding algorithm, see the extended help (`??Bounds.MultiEllipsoid`)

[^3]: Feroz and Hobson, 2008, MNRAS 384, 2 ["Multimodal nested sampling: an efficient and robust alternative to Markov Chain Monte Carlo methods for astronomical data analyses"](https://academic.oup.com/mnras/article/384/2/449/1023701)
[^4]: Feroz et al., 2009, MNRAS 398, 4 ["MultiNest: an efficient and robust Bayesian inference tool for cosmology and particle physics"](https://academic.oup.com/mnras/article/398/4/1601/981502)

## Extended help

The multiple-ellipsoidal implementation is defined as follows:

1. Fit a [`Bounds.Ellipsoid`](@ref) to the sample.
2. Perform K-means clustering (here using [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)) centered at the endpoints of the bounding ellipsoid. This defines two clusters within the sample.
3. If either cluster has fewer than two points, consider it ill-defined and end any recursion.
4. Fit [`Bounds.Ellipsoid`](@ref) to each of the clusters assigned in (2).
5. Split when the children's volume is below the parent's by more than the BIC
   penalty for an ellipsoid's `d(d+3)/2` parameters, `exp(-d(d+3)/2 · log(n)/n)`
   for `n` points in `d` dimensions, or when recursing into the children brings
   `k` ellipsoids below `exp(-(k-1) d(d+3)/2 · log(n)/n)` of it (dynesty's rule;
   Pelleg & Moore 2000; Feroz et al. 2008), and recurse (1-5) to each child.

`bic = false` gives the rule this package had: split when the children hold
less than half the parent's volume, or when the parent is more than twice the
volume `2n · pointvol` expected of the points. That rule splits a connected
region into many ellipsoids of 2d+ points each. The bootstrap expansion cannot
make those cover the region: on a 10-d ball with 100 points, 1.0% of it stayed
outside after a 5-resample expansion (13% in the worst of 20 fits), against
0.04% (0.6%) split by dynesty's rule. And a walk or slice takes its metric from
the ellipsoid that holds its start, so on a region cut into many it depends on
where it starts and no longer leaves the uniform distribution in the region
unchanged: on a 10-d RM obliquity fit the region came in 15 ellipsoids against
1–5 by dynesty's rule, and log Z came out 0.35–0.5 nats above pt_emcee's with
`RSlice(slices = 50)` or `RWalk(walks = 200)`, against 0.04–0.17 by dynesty's
rule.

To sample from this distribution, a random ellipsoid is selected, and a random sample is sampled from that ellipsoid. We then reverse this and find all of the ellipsoids which enclose the sampled point, and select one of those randomly for the enclosing bound.
"""
struct MultiEllipsoid{T} <: AbstractBoundingSpace{T}
    ellipsoids::Vector{Ellipsoid{T}}
end
MultiEllipsoid(ndims::Integer) = MultiEllipsoid(Float64, ndims)
MultiEllipsoid(T::Type, ndims::Integer) = MultiEllipsoid([Ellipsoid(T, ndims)])

Base.eltype(::MultiEllipsoid{T}) where {T} = T
Base.broadcastable(me::MultiEllipsoid) = (me,)
Base.length(me::MultiEllipsoid) = length(me.ellipsoids)
Base.ndims(me::MultiEllipsoid) = ndims(me.ellipsoids[1])

volume(me::MultiEllipsoid) = sum(volume, me.ellipsoids)

function scale!(me::MultiEllipsoid, factor)
    scale!.(me.ellipsoids, factor)
    return me
end

# `rng`: k-means draws a new centre for a cluster left empty. From the global
# generator, a nested run seeded the same gave another run in a process that
# had drawn from it since.
function fit(::Type{<:MultiEllipsoid}, x::AbstractMatrix; pointvol = 0, bic = true,
             rng::AbstractRNG = GLOBAL_RNG)
    parent = fit(Ellipsoid, x, pointvol = pointvol)
    ells = fit(MultiEllipsoid, x, parent, pointvol = pointvol, depth = 0, bic = bic, rng = rng)
    return MultiEllipsoid(ells)
end

const _MAX_SPLIT_DEPTH = 32

function fit(::Type{<:MultiEllipsoid}, x::AbstractMatrix, parent::Ellipsoid; pointvol = 0, depth = 0,
             bic = true, rng::AbstractRNG = GLOBAL_RNG)
    ndim, npoints = size(x)

    # Clustering will fail with fewer than k=2 points
    npoints ≤ 2 && return [parent]

    # Cap recursion depth to prevent stack overflow in high dimensions
    depth ≥ _MAX_SPLIT_DEPTH && return [parent]

    p1, p2 = endpoints(parent)
    starting_points = hcat(p1, p2)
    R = kmeans!(x, starting_points; maxiter = 10, rng = rng)
    labels = assignments(R)
    x1 = x[:, labels .== 1]
    x2 = x[:, labels .== 2]

    # if either cluster has fewer than ndim points, it is ill-defined
    if size(x1, 2) < 2ndim || size(x2, 2) < 2ndim
        return [parent]
    end

    # Getting bounding ellipsoid for each cluster
    ell1, ell2 = fit.(Ellipsoid, (x1, x2), pointvol = pointvol)

    if bic   # dynesty's rule, see the extended help
        nparam = (ndim^2 + 3ndim) ÷ 2
        children() = vcat(fit(MultiEllipsoid, x1, ell1, pointvol = pointvol, depth = depth + 1, bic = true, rng = rng),
                          fit(MultiEllipsoid, x2, ell2, pointvol = pointvol, depth = depth + 1, bic = true, rng = rng))
        volume(ell1) + volume(ell2) < exp(-nparam * log(npoints) / npoints) * volume(parent) &&
            return children()
        out = children()
        sum(volume, out) < exp(-nparam * (length(out) - 1) * log(npoints) / npoints) * volume(parent) &&
            return out
        return [parent]
    end

    # If total volume decreased by over half, recurse
    if volume(ell1) + volume(ell2) < 0.5volume(parent)
        return vcat(fit(MultiEllipsoid, x1, ell1, pointvol = pointvol, depth = depth + 1, bic = false, rng = rng),
                    fit(MultiEllipsoid, x2, ell2, pointvol = pointvol, depth = depth + 1, bic = false, rng = rng))
    end

    # Otherwise see if total volume is much larger than expected
    # and split into more than 2 clusters
    if volume(parent) > 2npoints * pointvol
        out = vcat(fit(MultiEllipsoid, x1, ell1, pointvol = pointvol, depth = depth + 1, bic = false, rng = rng),
                    fit(MultiEllipsoid, x2, ell2, pointvol = pointvol, depth = depth + 1, bic = false, rng = rng))
        sum(volume, out) < 0.5volume(parent) && return out
    end

    # Otherwise, return single bounding ellipse
    return [parent]
end

Base.in(x::AbstractVector, me::MultiEllipsoid) = any(ell->x ∈ ell, me.ellipsoids)

function Base.rand(rng::AbstractRNG, me::MultiEllipsoid)
    length(me) == 1 && return rand(rng, me.ellipsoids[1])

    vols = volume.(me.ellipsoids)
    weights = vols ./ sum(vols)
    local x

    while true
        # Select random ellipsoid
        idx = rand(rng, Categorical(weights))
        ell = me.ellipsoids[idx]

        # Select point
        x = rand(rng, ell)

        # How many ellipsoids is the sample in
        n = count(ell -> x ∈ ell, me.ellipsoids)

        # Only accept with probability 1/n
        (n == 1 || rand(rng) < 1 / n) && break
    end

    return x
end

"""
Returns a random live point and a bounding ellipsoid, since MultiEllipsoid doesn't have
valid transformation axes
"""
function rand_live(rng::AbstractRNG, me::MultiEllipsoid, us)
    idx = rand(rng, Base.axes(us, 2))
    u = us[:, idx]
    return u, bound_containing(rng, me, u)
end

# A random one of the ellipsoids that contain `u`; `nothing` if none does, and
# the caller (step.jl) refits the bounds.
function bound_containing(rng::AbstractRNG, me::MultiEllipsoid, u)
    idxs = findall(ell -> u ∈ ell, me.ellipsoids)
    isempty(idxs) && return nothing
    return me.ellipsoids[rand(rng, idxs)]
end
