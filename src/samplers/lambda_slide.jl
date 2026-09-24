# The λ slide: (Mo, ω) → (Mo + δ, ω − δ).
#
# At low eccentricity an RV orbit constrains the mean longitude λ = Mo + ω, not
# Mo and ω separately. K[cos(ν + ω) + e cos ω] with ν → M as e → 0 is
# K cos(Mo + ω + n(t − t₀)), so at e = 0 only the sum enters; at finite e the
# split is pinned only by the O(e) harmonic. Measured on the easy 12.3 d target
# (e = 0.1, 60 RVs, 2 m/s errors): circular sd 0.056 rad for λ against 0.66 for
# each of Mo and ω -- a 12:1 ridge.
#
# Under the default (:Mo, :sesinw) parametrization that ridge is a HELIX in the
# sampler's own coordinates: Mo advances while (sesinw, secosw) = √e(sin ω,
# cos ω) rotates. The stretch move interpolates LINEARLY over all coordinates
# at once, so a chord between two walkers Δω apart along the helix leaves it by
# √e(1 − cos(Δω/2)) -- at √e = 0.26 and Δω = 1.3 rad that is 0.053, comparable
# to the ridge's own transverse width -- and is rejected. The walkers then
# cannot trade places along the degenerate direction. Measured on a failing
# run: between-walker circular sd 0.52 for Mo against within-walker 0.39, while
# λ mixed exactly as well as in a passing run (0.039 against 0.040). The ridge
# is the whole of what fails; the well-determined combination is fine.
#
# Tempering is the sampler's existing answer -- hot rungs cross the ridge freely
# and swaps carry that down -- but it only works when the cold rung is coupled,
# and it leaves β = 1 dependent on the ladder. This move addresses the geometry
# directly, which is what lets the cold chain traverse the ridge unaided.
#
# The move proposes the image of a walker under a rotation ALONG the ridge:
#     Mo → Mo + δ   (wrapped by the physical period into the sampled window)
#     ω  → ω − δ,  i.e. (sesinw, secosw) → R(−δ)·(sesinw, secosw)
# so λ = Mo + ω and e = sesinw² + secosw² are BOTH preserved exactly, every
# other coordinate is untouched, and it is accepted with the tempered ratio
#     α = min{1, π_β(T_δ θ) / π_β(θ)}.
#
# Correctness. δ is drawn from a symmetric density (a mixture of a local
# Gaussian and a global uniform on [−π, π), each symmetric), so q(δ) = q(−δ)
# Lebesgue-a.e.; the half-open endpoint is a two-point set of measure zero.
# T_δ is a translation in Mo composed with a rotation in the (sesinw, secosw)
# plane, so |det ∂T_δ| = 1, and T_{−δ} ∘ T_δ = id on a full-circle window (a
# chart of the circle -- src/circular.jl). Symmetric proposal, unit Jacobian,
# involutive pairing ⇒ the plain Metropolis ratio above is detailed-balance
# correct with respect to π_β at every rung, with NO assumption that the
# degeneracy is exact. Where it is exact α = 1; where the O(e) harmonic breaks
# it the ratio says so and the move is rejected more often. Composed with the
# stretch, trans-dim and swap kernels at a fixed per-walker probability, so the
# chain stays reversible and irreducible (Green 1995; Tierney 1998).
#
# The wrap uses `mod` by CIRCULAR_PERIOD and NOT `circular_relabel`, which pins
# a value past `hi` to `hi` (src/circular.jl). That clamp is right for
# relabelling a chart, but inside a move it would map an interval onto the
# single point `hi`, put an atom there and destroy reversibility. A window a
# hair short of 2π (`_CIRCULAR_SPAN_RTOL`) therefore lets a proposal land
# outside the support, where the prior is −Inf and the move is simply rejected
# -- which is the reversible thing to do.
#
# Offered only where it is defined, which saves the likelihood evaluation
# elsewhere (correctness does not depend on the choice):
#   * the planet exposes Mo and the sesinw/secosw pair as sampled parameters,
#     i.e. the default (:Mo, :sesinw) parametrization. Under :ew, ω is itself a
#     coordinate and the slide would be a pair of translations -- not wired up,
#     because :ew is not the default and the (e, ω) prior is not rotationally
#     symmetric in the same way;
#   * Mo is sampled on a full-circle window, so Mo + δ has a representative.
#
# The (sesinw, secosw) prior is UniformPrior(-1, 1) on each coordinate -- a
# square -- while the rotation preserves the radius √(sesinw² + secosw²) = √e.
# Every state with a finite likelihood has e < 1, so it lies in the unit DISC,
# which is inscribed in that square: the rotation can never carry a valid
# walker out of the prior's support, and on the support the prior is flat, so
# this factor cancels from the ratio exactly. (Were it otherwise the ratio
# would simply reject, which is still correct -- but it does not arise.)

"""
    LambdaSlide(planet, mo, se, sc)

Unfrozen-layout indices of one planet's `Mo_k`, `sesinw_k` and `secosw_k`
coordinates, for [`_lambda_slide_sweep!`](@ref).
"""
struct LambdaSlide
    planet::Int
    mo::Int
    se::Int
    sc::Int
end

"""
    lambda_slides(params::Params) -> Vector{LambdaSlide}

The planets whose (Mo, ω) degeneracy the λ slide can act on: those sampling
`Mo_k`, `sesinw_k` and `secosw_k`, with `Mo_k` on a full-circle window.
"""
function lambda_slides(params::Params)
    L = params.layout
    out = LambdaSlide[]
    for k in 1:params.config.max_kplanet
        mo = findfirst(==("Mo_k$k"), L.unfrozen_names)
        se = findfirst(==("sesinw_k$k"), L.unfrozen_names)
        sc = findfirst(==("secosw_k$k"), L.unfrozen_names)
        (mo === nothing || se === nothing || sc === nothing) && continue
        # `is_circular`, not the span alone: it refuses Mo when TTVs are modelled
        # or `obs_prior` is on (src/circular.jl), and in both cases Mo is NOT
        # 2π-periodic -- the O'Neil Jacobian carries the mean anomaly linearly,
        # and `Tp = t_ref - Mo·P/2π` renumbers every transit. The move would be
        # proposed and then rejected on the ratio, which is correct but wastes a
        # quarter of a likelihood sweep per planet.
        is_circular(L.unfrozen_names[mo], L.unfrozen_priors[mo],
                    params.config) || continue
        lo, hi = bounds(L.unfrozen_priors[mo])
        # The span gate is ONE-SIDED, and the eps slop is the point of it.
        #
        # A window LONGER than a period would be fatal: the wrap below has image
        # [lo, lo+2π), so the sliver past it would be in the support but never
        # proposed into, T_{−δ} ∘ T_δ would carry it to x − 2π, and detailed
        # balance would fail there. `is_circular` alone would admit one -- it
        # tests |span − 2π| against `_CIRCULAR_SPAN_RTOL`, which is two-sided.
        # In practice the model builder caps Mo at exactly 2π and refuses a
        # wider prior outright, so this is belt-and-braces rather than a live
        # hole; it is written down because the builder's cap is a long way from
        # here and nothing else would catch it if that changed.
        #
        # What IS reachable is a re-cut window over by ULPS: src/circular.jl
        # stores `hi = lo + span`, and `(lo + 2π) − lo` is not bit-exact -- at
        # lo = 4.21 it reads back 2π + 8.9e-16. Hence the slop: a flat `<=`
        # would silently drop the move on a legitimate full-circle window, which
        # is the worse failure of the two.
        #
        # Windows SHORT of a period are admitted and are provably reversible:
        # `mod` is injective on any interval shorter than a period, and a
        # proposal landing in the gap is out of support, gets lp = −Inf and is
        # rejected, which IS the reversible outcome.
        hi - lo <= CIRCULAR_PERIOD + 8 * eps(CIRCULAR_PERIOD) || continue
        push!(out, LambdaSlide(k, mo, se, sc))
    end
    return out
end

"""
    lambda_slide!(buf, s::LambdaSlide, δ, lo, hi)

Write `T_δ` of the walker already copied into `buf`: `Mo += δ` wrapped into the
window, `(sesinw, secosw)` rotated by `−δ`. λ and e are preserved exactly.
"""
@inline function lambda_slide!(buf::AbstractVector{Float64}, s::LambdaSlide,
                                δ::Float64, lo::Float64)
    c, sn = cos(δ), sin(δ)
    se, sc = buf[s.se], buf[s.sc]
    buf[s.mo] = lo + mod(buf[s.mo] + δ - lo, CIRCULAR_PERIOD)
    buf[s.se] = se * c - sc * sn
    buf[s.sc] = sc * c + se * sn
    return nothing
end

# Symmetric in δ by construction: an even mixture of a local Gaussian (matched
# to the measured ridge extent, ≈ 0.66 rad) and a global uniform over the whole
# circle, so the move both refines along the ridge and jumps across it.
@inline function _lambda_slide_draw(rng, sigma::Float64)
    return rand(rng) < 0.5 ? sigma * randn(rng) :
                             (2 * rand(rng) - 1) * Float64(π)
end

"""
    _lambda_slide_sweep!(eval!, state, logπ_arr, logL_arr, βs, slides, params,
                         prob, sigma, rngs, bufs, proposed, accepted;
                         eligible) -> n_evals

One λ-slide attempt per (rung, walker, planet) with probability `prob`.
Threaded over chunks, like every other sweep in the samplers; `eval!(buf, t, w,
slot)` returns `(logπ, logL)` for the proposal.
"""
function _lambda_slide_sweep!(eval!::F, state::Array{Float64,3},
                              logπ_arr::Matrix{Float64}, logL_arr::Matrix{Float64},
                              βs::AbstractVector{Float64},
                              slides::Vector{LambdaSlide}, params::Params,
                              prob::Float64, sigma::Float64,
                              rngs::Vector{MersenneTwister},
                              bufs::Vector{Vector{Float64}},
                              proposed::Vector{Threads.Atomic{Int}},
                              accepted::Vector{Threads.Atomic{Int}};
                              eligible::G = (t, w, k) -> true) where {F,G}
    n_t, n_w, n_dim = size(state)
    n_evals = Threads.Atomic{Int}(0)
    isempty(slides) && return n_evals[]
    L = params.layout
    # Chunked against `length(bufs)`, never `Threads.threadid()` (see
    # src/threading.jl).
    chunks = _chunk_ranges(n_t * n_w, length(bufs))
    Threads.@threads :static for slot in 1:length(chunks)
        for task_idx in chunks[slot]
            t = (task_idx - 1) ÷ n_w + 1
            w = (task_idx - 1) % n_w + 1
            rng = rngs[task_idx]
            buf = bufs[slot]
            for s in slides
                rand(rng) < prob || continue
                eligible(t, w, s.planet) || continue
                δ = _lambda_slide_draw(rng, sigma)
                @inbounds for d in 1:n_dim
                    buf[d] = state[t, w, d]
                end
                lo, _ = bounds(L.unfrozen_priors[s.mo])
                lambda_slide!(buf, s, δ, Float64(lo))
                lp, ll = eval!(buf, t, w, slot)
                Threads.atomic_add!(n_evals, 1)
                Threads.atomic_add!(proposed[t], 1)
                # A proposal landing outside a window short of a full circle has
                # lp = -Inf here and is rejected -- the reversible outcome.
                if isfinite(ll) && log(rand(rng)) <
                        βs[t] * (ll - logL_arr[t, w]) + (lp - logπ_arr[t, w])
                    @inbounds for d in 1:n_dim
                        state[t, w, d] = buf[d]
                    end
                    logπ_arr[t, w] = lp
                    logL_arr[t, w] = ll
                    Threads.atomic_add!(accepted[t], 1)
                end
            end
        end
    end
    return n_evals[]
end

# One line per run, after the progress bar. The β = 1 acceptance is the check
# that the ridge is real: high when λ is well determined and the split is not,
# falling as e grows and the O(e) harmonic starts to pin ω on its own.
function _lambda_slide_info(sampler::AbstractString, slides::Vector{LambdaSlide},
                            params::Params, proposed::Vector{Int},
                            accepted::Vector{Int})
    (isempty(slides) || sum(proposed) == 0) && return nothing
    nms = [params.layout.unfrozen_names[s.mo] for s in slides]
    cold = proposed[1] > 0 ? round(accepted[1] / proposed[1]; digits = 3) : NaN
    all_ = round(sum(accepted) / sum(proposed); digits = 3)
    @info "$sampler: λ slide (Mo, ω) → (Mo + δ, ω − δ) on $(join(nms, ", ")): " *
          "$(sum(proposed)) proposals, acceptance $cold at β = 1 ($all_ over all " *
          "rungs). A reversible move along the mean-longitude ridge, which the " *
          "linear stretch can only cut chords across."
    return nothing
end
