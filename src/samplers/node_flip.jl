# The astrometric node flip: (Ω, ω) → (Ω + π, ω + π).
#
# Astrometry alone cannot tell the ascending node from the descending one. The
# photocentre moves as A·X + F·Y, B·X + G·Y, with X, Y functions of (E, e) only
# and every Thiele-Innes constant a product of one trig function of ω and one of
# Ω (A = cos ω cos Ω − sin ω sin Ω cos i, ...). Shifting both angles by π flips
# the sign of both factors, so A, B, F, G -- and the likelihood -- are unchanged.
# A planet constrained only by astrometry therefore has two modes of exactly
# equal mass, 180° apart in Ω, with ω flipped along. RVs break the tie
# (K[cos(ν + ω) + e cos ω] changes sign).
#
# The stretch move cannot cross between them: its proposal lies on the line
# through two walkers, never less than halfway from the partner, so a walker in
# one mode is never proposed into the other, and only temperature swaps carry
# walkers across. On Gaia-4 (astrometry only, pt_emcee, 100 walkers × 1500
# recorded steps) that left the cold rung at 61/39 where the symmetry demands
# 50/50. Every walker spent 54-69% of its steps in the heavier mode, so the
# shared bias passed R-hat (1.003) and fit_health.
#
# The move proposes the image directly. T maps a walker to its mirror,
#     Ω → Ω + π (relabelled into the current window),
#     (√e sin ω, √e cos ω) → −(√e sin ω, √e cos ω)   [(e sin ω, e cos ω) alike],
#     ω → ω + π under `:ew`,
# every other coordinate unchanged, and is accepted with the tempered ratio
#     α = min{1, π_β(Tθ) / π_β(θ)}.
# T is an involution (T∘T = id: two reflections, and Ω + 2π ≡ Ω on a
# full-circle window, which is a chart of the circle -- src/circular.jl) with
# |det ∂T| = 1 (a translation and reflections), so for any sets A, B the flow
#     ∫_A 1{Tθ ∈ B} min{π(θ), π(Tθ)} dθ
# equals, after substituting θ = Tφ, the flow from B to A: detailed balance
# with respect to π_β at every rung, with NO assumption that the symmetry is
# exact. When it is exact α = 1; when something breaks it (an asymmetric user
# prior, the O'Neil obs_prior) the ratio says so and the move is simply
# rejected more often. Attempted with a fixed probability per walker and step,
# composed with the stretch and swap kernels, so the chain stays reversible and
# irreducible (Green 1995; Tierney 1998; Neklyudov et al. 2020).
#
# The walkers move in bounded space, and on log(x + s) for LogUniform /
# ModJeffreys parameters (see sample_pt_emcee). T only touches Uniform-scale
# coordinates, so its Jacobian is 1 on the walkers' own scale too, and the
# log-scale terms in the cached prior cancel in the ratio.
#
# Offered only where it can be accepted, which saves the likelihood evaluation
# elsewhere (correctness does not depend on this choice):
#   * the planet is constrained by astrometry and nothing else. Judged on the
#     DATA, not the mode alone: `build_target` gives an astrometry-only planet
#     the RVAS mode, so an `:RV` / `:PM` source counts only when the fit has RV
#     / photometric data for it to act on. Any other source (RM, gravity
#     darkening, TTVs, an SB2 orbit) rules the planet out;
#   * Ω is sampled with a full-circle Uniform prior, so Ω + π has a
#     representative in the window;
#   * the ω pair is sampled. Under `:ew`, ω must be full-circle too. With the
#     pair frozen at e = 0, the in-plane phase at fixed t is ω + M, and the same
#     degeneracy is realised as (Ω + π, Mo + π), so Mo is flipped instead,
#     when it is the sampled, full-circle time anchor;
#   * the time anchor is Mo or Tp, which T holds fixed, so M(t) is unchanged.
#     Tc is defined through ω, and holding it fixed moves Tp: the image of a
#     walker would not be its mirror, and the move would only be rejected.
#     An astrometry-only planet with a Tc anchor is not offered the move.

"""
    NodeFlip

One planet's node flip: the unfrozen positions `shift` move by π around their
circle (Ω, plus ω under `:ew` or Mo for a frozen e = 0 orbit), and the positions
in `negate` change sign (the eccentricity-vector pair). See the header of
`src/samplers/node_flip.jl`.
"""
struct NodeFlip
    planet::Int
    shift::Vector{Int}
    negate::Vector{Int}
end

"""
    node_flips(params, data; flat_shift = nothing) -> Vector{NodeFlip}

The planets whose likelihood is invariant under (Ω, ω) → (Ω + π, ω + π), as
[`NodeFlip`](@ref)s. Empty for any target without an astrometry-only planet.
`flat_shift` (the samplers' per-dimension log-scale shift) drops a flip that
would touch a coordinate the walkers move on a log scale, where T would not be
unit-Jacobian; no default prior puts one there.
"""
function node_flips(params::Params, data::Data; flat_shift = nothing)
    L, cfg = params.layout, params.config
    pc = cfg.parametrization
    flips = NodeFlip[]
    # Sources a planet may carry and still be astrometry-only: RV and PM do
    # nothing without data to act on (`build_target` makes an astrometry-only
    # planet RVAS). Everything else reads ω or Ω on its own.
    function astrometry_only(m)
        AS_SOURCE in m || return false
        all(s -> s in (AS_SOURCE, RV_SOURCE, PM_SOURCE), m.sources) || return false
        RV_SOURCE in m && !isempty(data.t_rv)   && return false
        PM_SOURCE in m && !isempty(data.t_phot) && return false
        return true
    end
    free(nm) = findfirst(==(nm), L.unfrozen_names)
    function frozen(nm)
        i = get(L.name_to_idx, nm, 0)
        j = findfirst(==(i), L.frozen_idx)
        return j === nothing ? nothing : L.frozen_values[j]
    end
    circ(d) = d !== nothing &&
              is_circular(L.unfrozen_names[d], L.unfrozen_priors[d], cfg)
    for (k, modes) in enumerate(cfg.planet_modes)
        astrometry_only(modes) || continue
        pc.time in (:Mo, :Tp) || continue
        sfx  = "_k$k"
        node = free("Omega" * sfx)
        circ(node) || continue
        f = nothing
        if pc.ew === :ew
            w = free("w" * sfx)
            if circ(w)
                f = NodeFlip(k, [node, w], Int[])
            elseif frozen("ecc" * sfx) == 0 && w === nothing
                f = _circular_orbit_flip(k, node, free("Mo" * sfx), pc, circ)
            end
        else
            n1, n2 = pc.ew === :sesinw ? ("sesinw", "secosw") : ("esinw", "ecosw")
            d1, d2 = free(n1 * sfx), free(n2 * sfx)
            if d1 !== nothing && d2 !== nothing
                f = NodeFlip(k, [node], [d1, d2])
            elseif d1 === nothing && d2 === nothing &&
                   frozen(n1 * sfx) == 0 && frozen(n2 * sfx) == 0
                f = _circular_orbit_flip(k, node, free("Mo" * sfx), pc, circ)
            end
        end
        f === nothing && continue
        if flat_shift !== nothing
            any(d -> flat_shift[d] !== nothing, vcat(f.shift, f.negate)) && continue
        end
        push!(flips, f)
    end
    return flips
end

_circular_orbit_flip(k, node, mo, pc, circ) =
    pc.time === :Mo && circ(mo) ? NodeFlip(k, [node, mo], Int[]) : nothing

"""
    node_flip!(x, f::NodeFlip, params) -> x

Map the unfrozen-vector point `x` to its mirror under `f`, in place. Shifted
angles are relabelled into the layout's CURRENT windows, so a window moved by a
circular re-cut is respected. An involution up to rounding.
"""
@inline function node_flip!(x::AbstractVector, f::NodeFlip, params::Params)
    L = params.layout
    @inbounds for d in f.shift
        lo, hi = bounds(L.unfrozen_priors[d])
        x[d] = circular_relabel(x[d] + π, lo, hi)
    end
    @inbounds for d in f.negate
        x[d] = -x[d]
    end
    return x
end

"""
    _node_flip_sweep!(eval!, state, logπ_arr, logL_arr, βs, flips, params,
                      prob, rngs, bufs, proposed, accepted; eligible) -> n_evals

One node-flip pass over every (rung, walker): for each flip, with probability
`prob`, propose the walker's mirror and accept it with the tempered ratio
β·Δlog L + Δlog π. `eval!(buf, t, w, tid)` returns the sampler's own
`(log π, log L)` split, the one `logπ_arr` / `logL_arr` cache. `eligible(t, w,
k)` gates planet `k` per walker (a trans-dim walker without the planet) and
must be invariant under the flip. `rngs` holds one stream per (rung, walker),
indexed `(t - 1) · n_walkers + w`, so the chain does not depend on the thread
count. Returns the number of likelihood evaluations.
"""
function _node_flip_sweep!(eval!::F, state::Array{Float64,3},
                           logπ_arr::Matrix{Float64}, logL_arr::Matrix{Float64},
                           βs::AbstractVector{Float64}, flips::Vector{NodeFlip},
                           params::Params, prob::Float64,
                           rngs::Vector{MersenneTwister},
                           bufs::Vector{Vector{Float64}},
                           proposed::Vector{Threads.Atomic{Int}},
                           accepted::Vector{Threads.Atomic{Int}};
                           eligible::G = (t, w, k) -> true) where {F,G}
    n_t, n_w, n_dim = size(state)
    n_evals = Threads.Atomic{Int}(0)
    Threads.@threads :static for task_idx in 1:(n_t * n_w)
        tid = Threads.threadid()
        t = (task_idx - 1) ÷ n_w + 1
        w = (task_idx - 1) % n_w + 1
        rng = rngs[task_idx]
        buf = bufs[tid]
        for f in flips
            rand(rng) < prob || continue
            eligible(t, w, f.planet) || continue
            @inbounds for d in 1:n_dim
                buf[d] = state[t, w, d]
            end
            node_flip!(buf, f, params)
            lp, ll = eval!(buf, t, w, tid)
            Threads.atomic_add!(n_evals, 1)
            Threads.atomic_add!(proposed[t], 1)
            log_ratio = βs[t] * (ll - logL_arr[t, w]) + (lp - logπ_arr[t, w])
            if log(rand(rng)) < log_ratio
                @inbounds for d in 1:n_dim
                    state[t, w, d] = buf[d]
                end
                logπ_arr[t, w] = lp
                logL_arr[t, w] = ll
                Threads.atomic_add!(accepted[t], 1)
            end
        end
    end
    return n_evals[]
end

# One line per run, after the progress bar, like the circular re-cut report. The
# β = 1 acceptance is the check that the symmetry held: ≈ 1 when the planet
# really is astrometry-only, lower when a prior or term breaks it.
function _node_flip_info(sampler::AbstractString, flips::Vector{NodeFlip},
                         params::Params, proposed::Vector{Int},
                         accepted::Vector{Int})
    (isempty(flips) || sum(proposed) == 0) && return nothing
    nms = [params.layout.unfrozen_names[first(f.shift)] for f in flips]
    cold = proposed[1] > 0 ? round(accepted[1] / proposed[1]; digits = 3) : NaN
    all_ = round(sum(accepted) / sum(proposed); digits = 3)
    @info "$sampler: node flip (Ω, ω) → (Ω + π, ω + π) on $(join(nms, ", ")): " *
          "$(sum(proposed)) proposals, acceptance $cold at β = 1 ($all_ over all " *
          "rungs). A reversible move between the two exactly degenerate " *
          "astrometric solutions; it weights them by the posterior instead of by " *
          "how often swaps happened to carry walkers across."
    return nothing
end
