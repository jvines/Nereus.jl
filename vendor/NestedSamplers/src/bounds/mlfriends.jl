"""
    Bounds.MLFriends([T=Float64], N)

MLFriends region bound (Buchner 2019).[^mlf] The bounded region is the **union
of `N` balls** of a single, bootstrap-calibrated radius centred on the live
points, measured in a whitened (Mahalanobis) metric, **intersected** with a
bootstrap-enlarged wrapping ellipsoid (and the unit cube). Unlike a single or
multi-ellipsoid, the union of balls conforms to arbitrarily curved or
multimodal iso-likelihood contours, so points drawn from it are uniform on the
constrained prior even when that region is a curved ridge (e.g. the
eccentricity `e–ω` banana of an RV posterior). That uniformity is what removes
the log-evidence bias ellipsoidal bounds incur on such posteriors.

Pair with `proposal = :unif` (the `Rejection` proposal): `Base.rand` performs
overlap-corrected uniform sampling from the union, and `Rejection` then applies
the likelihood constraint.

Both the friends radius (`maxradiussq`, in whitened space) and the ellipsoid
threshold (`enlarge`, a squared-Mahalanobis radius in cube space) are derived by
leave-out bootstrap cross-validation — they self-calibrate to be conservative,
so MLFriends needs no hand-tuned enlargement pad (the sampler's `enlarge` factor
still applies multiplicatively on top, via `scale!`).

[^mlf]: J. Buchner 2019, PASP 131, 108005, ["Collaborative Nested Sampling"](https://arxiv.org/abs/1707.04476); reference implementation UltraNest `mlfriends.pyx`.
"""
mutable struct MLFriends{T} <: AbstractBoundingSpace{T}
    u::Matrix{T}                        # ndim × npoints — live points (cube space)
    unormed::Matrix{T}                  # ndim × npoints — whitened live points
    tree::KDTree                        # spatial index on `unormed` for O(log N) neighbour queries
    ctr::Vector{T}                      # whitening centre (cube space)
    W::Matrix{T}                        # whitening: unormed = W * (u .- ctr)
    Winv::Matrix{T}                     # untransform: u = Winv * unormed .+ ctr
    maxradiussq::T                      # friends radius² (whitened space)
    ell_ctr::Vector{T}                  # wrapping-ellipsoid centre (cube space)
    ell_invcov::Symmetric{T,Matrix{T}}  # wrapping-ellipsoid inverse covariance
    enlarge::T                          # ellipsoid squared-Mahalanobis threshold
    volume::T                           # estimated region volume (cube space)
end

Base.ndims(b::MLFriends) = size(b.u, 1)
volume(b::MLFriends) = b.volume
# A representative linear map (cube-space offset of the friends scale) for the
# proposals/randoffset fallbacks. Rejection (the intended pairing) does not use it.
axes(b::MLFriends) = b.Winv .* sqrt(b.maxradiussq)
randoffset(rng::AbstractRNG, b::MLFriends{T}) where {T} = axes(b) * randball(rng, T, ndims(b))

# ---- whitening metric (AffineLayer) -----------------------------------------
# Returns (ctr, W, Winv) with unormed = W*(u .- ctr) whitening cov*(ndim+2) → I.
function _affine_layer(u::AbstractMatrix{T}) where {T}
    ndim, np = size(u)
    ctr = vec(sum(u, dims = 2) ./ np)
    delta = u .- ctr
    cov = Symmetric((delta * delta') ./ max(np - 1, 1) .* (ndim + 2))
    E = eigen(cov)
    evmin = maximum(E.values) * 1e-40
    ev = max.(E.values, evmin)
    Q = E.vectors
    W = Diagonal(ev .^ (-one(T) / 2)) * Q'        # whitens: cov(W δ) ≈ I
    Winv = Q * Diagonal(ev .^ (one(T) / 2))
    return ctr, Matrix(W), Matrix(Winv)
end

# Wrapping ellipsoid in cube space: centre + conditioned inverse covariance.
function _bounding_ellipsoid(u::AbstractMatrix{T}) where {T}
    ndim, np = size(u)
    ctr = vec(sum(u, dims = 2) ./ np)
    delta = u .- ctr
    cov = Symmetric((delta * delta') ./ max(np - 1, 1) .* (ndim + 2))
    _, A = make_eigvals_positive(cov)             # A = conditioned inverse cov
    return ctr, A
end

# max over `bpts` columns of (min over `apts` columns of squared distance)
function _maxradiussq(apts::AbstractMatrix, bpts::AbstractMatrix)
    na = size(apts, 2); nb = size(bpts, 2); d = size(apts, 1)
    maxd = 0.0
    @inbounds for j in 1:nb
        mind = Inf
        for i in 1:na
            s = 0.0
            for k in 1:d
                Δ = apts[k, i] - bpts[k, j]
                s += Δ * Δ
            end
            s < mind && (mind = s)
        end
        mind > maxd && (maxd = mind)
    end
    return maxd
end

# ---- fit --------------------------------------------------------------------
fit(B::Type{<:MLFriends}, x::AbstractMatrix{S}; pointvol = 0, nbootstraps::Int = 30) where {S} =
    fit(B{float(S)}, x; pointvol = pointvol, nbootstraps = nbootstraps)

function fit(::Type{<:MLFriends{R}}, x::AbstractMatrix{S};
             pointvol = 0, nbootstraps::Int = 30) where {R,S}
    T = float(promote_type(R, S))
    u = Matrix{T}(x)
    ndim, np = size(u)

    ctr, W, Winv = _affine_layer(u)
    unormed = W * (u .- ctr)
    ell_ctr, ell_A = _bounding_ellipsoid(u)

    # Leave-out bootstrap cross-validation for BOTH the friends radius
    # (whitened space) and the ellipsoid threshold (cube space): max over rounds.
    maxd = zero(T); maxf = zero(T)
    sel = falses(np)
    if np > ndim + 1
        for _ in 1:nbootstraps
            fill!(sel, false)
            @inbounds for _ in 1:np
                sel[rand(1:np)] = true
            end
            (all(sel) || !any(sel)) && continue
            inb = findall(sel); oob = findall(.!sel)
            isempty(oob) && continue
            # friends radius in whitened space: each left-out point's nearest
            # in-bag neighbour, max over left-out — via a KDTree (O(N log N), so
            # the frequent refit-on-miss in the sampler loop stays cheap).
            tinb = KDTree(Matrix(view(unormed, :, inb)))
            _, dists = nn(tinb, Matrix(view(unormed, :, oob)))
            maxd = max(maxd, T(maximum(dists))^2)
            # wrapping-ellipsoid enlargement in cube space
            length(inb) > ndim + 1 || continue
            cb, Ab = _bounding_ellipsoid(view(u, :, inb))
            @inbounds for j in oob
                f = zero(T)
                dvec = view(u, :, j) .- cb
                f = dot(dvec, Ab * dvec)
                f > maxf && (maxf = f)
            end
        end
    end
    # Degenerate fallbacks (too few points to bootstrap): cover all points.
    if !(maxd > 0)
        maxd = zero(T)
        @inbounds for i in 1:np, j in 1:np
            i == j && continue
            s = zero(T)
            for k in 1:ndim
                Δ = unormed[k, i] - unormed[k, j]; s += Δ * Δ
            end
            s > maxd && (maxd = s)
        end
        maxd = max(maxd, eps(T))
    end
    if !(maxf > 0)
        maxf = zero(T)
        @inbounds for j in 1:np
            dvec = view(u, :, j) .- ell_ctr
            f = dot(dvec, ell_A * dvec)
            f > maxf && (maxf = f)
        end
        maxf = max(maxf, one(T))
    end

    tree = KDTree(unormed)
    b = MLFriends{T}(u, unormed, tree, ctr, W, Winv, maxd, ell_ctr,
                     Symmetric(Matrix{T}(ell_A)), maxf, zero(T))
    b.volume = _estimate_volume(b)

    # minvol guard (parity with Ellipsoid.fit): scale up if the region is
    # smaller than npoints·pointvol.
    if pointvol > 0
        minvol = np * pointvol
        b.volume < minvol && scale!(b, minvol / b.volume)
    end
    return b
end

# Union-of-balls volume estimate (cube space): N · V_ball / mean_overlap.
function _estimate_volume(b::MLFriends{T}) where {T}
    ndim, np = size(b.u)
    rsq = b.maxradiussq
    Vball_white = T(volume_prefactor(ndim)) * rsq^(ndim / 2)
    detWinv = abs(det(b.Winv))                 # whitened → cube volume factor
    # mean number of balls covering a live point (clustering / overlap measure)
    r = sqrt(rsq)
    total = 0
    @inbounds for i in 1:np
        total += inrangecount(b.tree, view(b.unormed, :, i), r)
    end
    mean_overlap = max(total / np, one(T))
    return np * Vball_white * detWinv / mean_overlap
end

# ---- membership -------------------------------------------------------------
function Base.in(x::AbstractVector, b::MLFriends)
    # gate 1: wrapping ellipsoid (cube space)
    de = x .- b.ell_ctr
    dot(de, b.ell_invcov * de) ≤ b.enlarge || return false
    # gate 2: within friends radius of any live point (whitened space) — O(log N)
    t = b.W * (x .- b.ctr)
    return inrangecount(b.tree, t, sqrt(b.maxradiussq)) > 0
end

# ---- sampling (overlap-corrected uniform draw from the union) ---------------
function Base.rand(rng::AbstractRNG, b::MLFriends{T}) where {T}
    ndim, np = size(b.unormed)
    r = sqrt(b.maxradiussq)
    maxtries = 100_000
    for _ in 1:maxtries
        i = rand(rng, 1:np)
        # uniform point in the unit ball, scaled to the friends radius
        v = randn(rng, T, ndim)
        v .*= (rand(rng, T)^(one(T) / ndim) * r) / sqrt(sum(abs2, v))
        @inbounds for k in 1:ndim
            v[k] += b.unormed[k, i]
        end
        # overlap correction: accept with probability 1/(# balls covering v) — O(log N)
        cnt = inrangecount(b.tree, v, r)
        (cnt == 0 || rand(rng, T) * cnt ≥ 1) && continue
        # untransform to cube space, enforce cube + ellipsoid gates
        u = b.Winv * v .+ b.ctr
        ok = true
        @inbounds for k in 1:ndim
            (zero(T) < u[k] < one(T)) || (ok = false; break)
        end
        ok || continue
        de = u .- b.ell_ctr
        dot(de, b.ell_invcov * de) ≤ b.enlarge || continue
        return u
    end
    error("MLFriends: could not draw a region point in $maxtries tries " *
          "(maxradiussq=$(b.maxradiussq), enlarge=$(b.enlarge))")
end

# ---- scale ------------------------------------------------------------------
# `factor` scales the region VOLUME (same convention as Ellipsoid). Volume ∝ r^d,
# so the (squared) friends radius and ellipsoid threshold scale by factor^(2/d).
function scale!(b::MLFriends, factor)
    f2 = factor^(2 / ndims(b))
    b.maxradiussq *= f2
    b.enlarge *= f2
    b.volume *= factor
    return b
end
