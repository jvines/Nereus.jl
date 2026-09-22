# Circular parameters — the 0/2π seam is a coordinate choice, not a wall.
#
# Mo, Ω, λ and (under the `:ew` parametrization) ω reach every likelihood only
# through sin/cos or Kepler's equation. With a Uniform prior spanning exactly one
# period, the prior window [lo, lo + 2π) is a CHART of the circle, and where its
# seam sits is arbitrary. The samplers nevertheless treat `lo` / `hi` as hard
# walls: eval_packed_logpdf rejects anything outside, the logit transform sends
# the seam to y = ±∞, and the nested samplers wall the unit cube. A posterior
# centred on the seam is then cut in two -- one mode shows up as two lobes at
# opposite ends of the interval with an empty interior between them. Nothing is
# truncated, since the window covers the circle exactly once, but it becomes a
# two-mode mixing problem, and every linear summary of that parameter (median,
# quantiles, R̂ / ESS, the prior-rail check) is meaningless. That is what
# "Mo_k1: median railed at upper bound (median 6.264, bound 6.283)" was.
#
# The fix MOVES the seam instead of removing it. A uniform full-circle prior is
# rotation-invariant, so relabelling every value into a window whose seam sits in
# the emptiest arc of the posterior changes neither the prior density (1/2π) nor
# the likelihood: log-posteriors, the evidence and the Markov kernel are
# untouched, only the chart moves. Bounded-space samplers re-cut at the end of
# burn-in (the chart is then fixed for the production run), and `run_job` /
# `fit_*` re-cut once more on whatever the engine returned, so every downstream
# consumer sees one contiguous posterior with its median inside the window the
# user wrote.
#
# `config.priors` keeps each prior exactly as the user wrote it; only the layout
# -- the prior the samplers evaluate -- carries the moved window.
#
# Deliberately NOT circular:
#   * priors that are not Uniform over exactly one period. A narrow arc, a
#     truncated Normal on Mo, a two-period U(-2π, 2π): those walls are the
#     user's, and a density that is not periodic cannot be relabelled exactly.
#   * inclination-like angles. i ∈ [0, π] is not periodic.
#   * Mo whenever TTVs are modelled. TTV epoch numbering hangs off
#     Tp = t_ref − Mo·P/2π, so Mo → Mo + 2π relabels every transit and the
#     likelihood is not periodic in Mo there.
#   * Mo under the O'Neil observation-based prior (`obs_prior`). Its Jacobian
#     carries the mean anomaly at each relative-astrometry epoch LINEARLY
#     (the 3M terms of |2(e²−2)sinE + e(3M + sin2E) + 3M cosE|), so shifting
#     Mo by 2π changes the density.

const CIRCULAR_PERIOD = 2π

# Per-planet angles only. Anchored, so `gp_act_lambda_e` and the ARMA `omega_*`
# coefficients can never match; `w_k` exists only under the `:ew`
# parametrization, where it is the argument of periastron.
const _CIRCULAR_NAME = r"^(Mo|Omega|lambda|w)_k\d+$"

# Wide enough for 2π typed to four decimals (6.2832 is off by 2.4e-6
# relative). The window keeps its own span when it moves, so the prior density
# never changes; a span a hair short of 2π only means a relabelled value can
# land in the sliver past `hi`, where `circular_relabel` pins it to `hi`.
const _CIRCULAR_SPAN_RTOL = 1e-4

_ttv_modelled(config) =
    any(m -> has_ttv(m) || has_ttv_nb(m), config.planet_modes)

"""
    is_circular(name, ps::PriorSpec, config::ParamsConfig) -> Bool

`true` when the parameter is an angle whose prior is Uniform over exactly one
period, so its prior window is a chart of the circle and the seam may be moved.
See the header of `src/circular.jl` for what is excluded and why.
"""
function is_circular(name::AbstractString, ps::PriorSpec, config)
    m = match(_CIRCULAR_NAME, name)
    m === nothing && return false
    ps.dist isa Distributions.Uniform || return false
    abs((ps.hi - ps.lo) - CIRCULAR_PERIOD) <=
        _CIRCULAR_SPAN_RTOL * CIRCULAR_PERIOD || return false
    if m.captures[1] == "Mo"
        _ttv_modelled(config) && return false
        _obs_prior_on(config) && return false
    end
    return true
end

_obs_prior_on(config) = hasproperty(config, :parametrization) &&
                        config.parametrization.obs_prior

"""
    circular_indices(params) -> Vector{Int}

Positions, in the unfrozen vector, of the parameters [`is_circular`](@ref)
accepts.
"""
function circular_indices(params::Params)
    L = params.layout
    return Int[i for i in eachindex(L.unfrozen_names)
               if is_circular(L.unfrozen_names[i], L.unfrozen_priors[i],
                              params.config)]
end

"""
    circular_names(params) -> Set{String}
"""
circular_names(params::Params) =
    Set{String}(params.layout.unfrozen_names[i] for i in circular_indices(params))

"""
    circular_windows(params) -> Dict{String, Tuple{Float64, Float64}}

The layout's current window of every circular parameter. Taken right after an
engine returns and BEFORE [`recenter_circular!`](@ref), it is the chart the
engine actually sampled in -- the one place a seam it could not cross still
shows (see `_run_fit_health!`).
"""
circular_windows(params::Params) =
    Dict{String, Tuple{Float64, Float64}}(
        params.layout.unfrozen_names[i] => bounds(params.layout.unfrozen_priors[i])
        for i in circular_indices(params))

"""
    circular_relabel(x, lo, hi; period = 2π) -> Real

The representative of angle `x` in the window `[lo, hi]` (`hi = lo + period`,
up to the span tolerance). A value already inside is returned untouched, bit
for bit, so relabelling an already-contiguous sample is a true no-op.
"""
@inline function circular_relabel(x::Real, lo::Real, hi::Real;
                                  period::Real = CIRCULAR_PERIOD)
    (lo <= x <= hi || !isfinite(x)) && return x
    y = lo + mod(x - lo, period)
    return y > hi ? oftype(y, hi) : y
end

"""
    circular_cut(x, lo; period = 2π, nbins = 72) -> Float64

Lower edge of the one-period window whose seam sits in the emptiest arc of the
draws `x` (any representative; non-finite entries are ignored). The circle is
binned into `nbins` arcs starting at `lo`; a candidate seam sits on every arc
edge and costs the draws in the two arcs beside it. The current seam (`lo`) is
kept unless it sits in a significantly DENSE arc: more than 3 Poisson σ above
the typical (median) edge and more than twice the emptiest one. Comparing with
the minimum alone is not enough -- the least of 72 noisy counts sits well below
the typical one, and a flat posterior's window then moved on noise in most
runs. Otherwise the new seam goes in the middle of the longest run of cheapest
edges, as far from the posterior mass as it can get. The result may exceed
`lo + period`; windows are only defined modulo `period`.
"""
function circular_cut(x::AbstractVector{<:Real}, lo::Real;
                      period::Real = CIRCULAR_PERIOD, nbins::Int = 72)
    w = period / nbins
    counts = zeros(Int, nbins)
    n = 0
    for xi in x
        isfinite(xi) || continue
        b = floor(Int, mod(xi - lo, period) / w) + 1
        counts[clamp(b, 1, nbins)] += 1
        n += 1
    end
    n == 0 && return Float64(lo)
    # Edge e sits between arc e and arc e+1, circularly; edge `nbins` is the
    # current seam.
    cost = [counts[e] + counts[mod1(e + 1, nbins)] for e in 1:nbins]
    cmin = minimum(cost)
    cmed = median(cost)
    cs = cost[nbins]
    (cs <= cmed + 3 * sqrt(cmed + 1) || cs <= 2 * cmin + 1) && return Float64(lo)
    best_start, best_len = 0, 0
    for s in 1:nbins
        cost[s] == cmin || continue
        cost[mod1(s - 1, nbins)] == cmin && continue   # not the start of a run
        len = 0
        while len < nbins && cost[mod1(s + len, nbins)] == cmin
            len += 1
        end
        len > best_len && ((best_start, best_len) = (s, len))
    end
    best_len == 0 && return Float64(lo)
    return Float64(lo + (best_start + (best_len - 1) / 2) * w)
end

"""
    circular_window(x, lo, lo_ref; period = 2π, nbins = 72) -> Float64

Lower edge of the window to relabel draws `x` (currently charted from `lo`)
into: seam in the emptiest arc ([`circular_cut`](@ref)), then shifted by whole
periods so the median lands in `[lo_ref, lo_ref + period)` -- the window the
user wrote -- so reported numbers keep the user's convention. Returns `lo`
exactly when nothing needs to move.
"""
function circular_window(x::AbstractVector{<:Real}, lo::Real, lo_ref::Real;
                         period::Real = CIRCULAR_PERIOD, nbins::Int = 72)
    c = circular_cut(x, lo; period, nbins)
    xs = Float64[circular_relabel(xi, c, c + period; period)
                 for xi in x if isfinite(xi)]
    isempty(xs) && return Float64(c)
    k = floor((median(xs) - lo_ref) / period)
    return k == 0 ? Float64(c) : Float64(c - k * period)
end

"""
    set_circular_window!(params, i, lo; transforms = ()) -> (lo, hi)

Move circular unfrozen parameter `i` to the window `[lo, lo + span)` everywhere
the samplers read its bounds: the layout's `PriorSpec`, the packed priors, and
every `PackedTransforms` in `transforms`. `config.priors` keeps what the user
wrote. Exact: the prior is still Uniform over one period, so its density does
not change.

The caller must then move every stored value of the parameter into the new
window with [`circular_relabel`](@ref) -- hot rungs, ring buffers, cached
proposals -- before the next log-density evaluation, or those values sit
outside the support. An `EnzymeGradientConfig` built before the move snapshots
the old bounds and must be rebuilt.
"""
function set_circular_window!(params::Params, i::Int, lo::Real; transforms = ())
    L = params.layout
    ps = L.unfrozen_priors[i]
    span = ps.hi - ps.lo
    lo = Float64(lo)
    hi = lo + span
    L.unfrozen_priors[i] = UniformPrior(lo, hi)
    L.packed_priors.lowers[i] = lo
    L.packed_priors.uppers[i] = hi
    for pt in transforms
        pt isa PackedTransforms || continue
        pt.lowers[i] = lo
        pt.uppers[i] = hi
    end
    return (lo, hi)
end

"""
    recut_circular_param!(params, i, draws; transforms = ())
        -> Union{Nothing, Tuple{Float64, Float64}}

Pick a new window for circular unfrozen parameter `i` from `draws` (its current
posterior draws; exclude inactive trans-dim slots, non-finite entries are
ignored) and move it there with [`set_circular_window!`](@ref). Returns the new
`(lo, hi)`, or `nothing` when the window stays put. After a move the caller must
relabel all of its own state into `(lo, hi)`.
"""
function recut_circular_param!(params::Params, i::Int,
                               draws::AbstractVector{<:Real}; transforms = ())
    ps = params.layout.unfrozen_priors[i]
    lo_ref = params.config.priors[params.layout.unfrozen_names[i]].lo
    new_lo = circular_window(draws, ps.lo, lo_ref)
    new_lo == ps.lo && return nothing
    return set_circular_window!(params, i, new_lo; transforms)
end

"""
    circular_relabel_point!(x, params) -> x

Relabel the circular entries of an unfrozen-vector point `x` into the layout's
CURRENT windows. For a caller's `init`: it is written in the user's window, and
a target that has already been through a fit carries moved windows, so an
unrelabelled `Mo = 6.2` can sit outside `[-3.05, 3.23)` and start every walker
at -Inf (or, through the logit, pinned to the wall).
"""
function circular_relabel_point!(x::AbstractVector, params::Params)
    L = params.layout
    for i in circular_indices(params)
        lo, hi = bounds(L.unfrozen_priors[i])
        x[i] = circular_relabel(x[i], lo, hi)
    end
    return x
end

# ------------------------------------------------------------------
# Trans-dimensional samplers: same-mode slots share ONE window.
#
# Every birth ends in `_sort_group_periods!`, which swaps whole parameter blocks
# between same-mode slots, so Mo_k2's value can land in slot 1 (DonorBirth and
# the AD↔planet swap copy between slots too). With a window per slot it would
# land outside slot 1's window and the birth would die on the prior for no
# posterior reason; relabelled into it, the MoMS birth/death would still
# disagree, since their Gaussian is a density on the line, not the circle.
# Either way the labelling invariance the sort relies on ("slot priors within a
# group are identical") is gone. One shared window keeps it exact. The price: the
# seam is cut from the pooled draws of every planet in the group, and stays put
# when their phases leave no clearly empty arc.
#
# λ is not in `planet_slot_indices` and is never swapped; with no planet births
# (`permutable = false`) nothing is. Both get a window per parameter. A group
# with a member that is not circular (a user who wrote different priors for
# exchangeable slots) was already label-variant and is left alone.
# ------------------------------------------------------------------

"""
    circular_groups(params; permutable) -> Vector{Vector{Tuple{Int,Int}}}

The circular parameters grouped by the window they must share, as
`(unfrozen position, owning planet)` pairs. `permutable = true` for a sampler
that births planets or swaps planet blocks between slots: same-mode slots then
share one window per angle. Otherwise every parameter is its own group.
"""
function circular_groups(params::Params; permutable::Bool)
    L = params.layout
    circ = circular_indices(params)
    groups = Vector{Vector{Tuple{Int,Int}}}()
    isempty(circ) && return groups
    modes  = params.config.planet_modes
    blocks = L.planet_blocks
    done = falses(length(L.unfrozen_idx))
    for d in circ
        done[d] && continue
        # `is_circular` only accepts `<angle>_k<k>`, so the owner is in the name.
        k = parse(Int, match(r"_k(\d+)$", L.unfrozen_names[d]).captures[1])
        q = permutable && k <= length(blocks) ?
            findfirst(==(L.unfrozen_idx[d]), planet_slot_indices(blocks[k])) :
            nothing
        if q === nothing
            done[d] = true
            push!(groups, [(d, k)])
            continue
        end
        grp = Tuple{Int,Int}[]
        ok = true
        for k2 in eachindex(blocks)
            modes[k2] == modes[k] || continue
            sl = planet_slot_indices(blocks[k2])
            d2 = q <= length(sl) ? findfirst(==(sl[q]), L.unfrozen_idx) : nothing
            if d2 === nothing || !(d2 in circ)
                ok = false
                continue
            end
            done[d2] = true
            push!(grp, (d2, k2))
        end
        ok && push!(groups, grp)
    end
    return groups
end

"""
    unify_circular_groups!(params, groups; transforms = ())

Put every member of each group into ONE window -- the first member's, as the
user wrote it. Call it when a trans-dim sampler starts, BEFORE it draws its
first states (it relabels nothing): a target reused after an earlier fit
carries a window per column (`recenter_circular!` moves each on its own), and
same-mode slots in different windows break the block swaps births rely on.
Exact: every member is Uniform over one period either way.
"""
function unify_circular_groups!(params::Params, groups; transforms = ())
    L = params.layout
    for grp in groups
        length(grp) > 1 || continue
        lo = params.config.priors[L.unfrozen_names[first(first(grp))]].lo
        for (d, _) in grp
            L.unfrozen_priors[d].lo == lo ||
                set_circular_window!(params, d, lo; transforms)
        end
    end
    return params
end

"""
    recut_circular_group!(params, grp, draws; transforms = ())
        -> Union{Nothing, Tuple{Float64, Float64}}

[`recut_circular_param!`](@ref) for a group that shares one window: the cut is
chosen from `draws` (the pooled ACTIVE draws of every member) and every member
moves to it. Returns the first member's new `(lo, hi)`, or `nothing`. The
caller relabels each member's state with that member's `bounds` afterwards.
"""
function recut_circular_group!(params::Params, grp, draws::AbstractVector{<:Real};
                               transforms = ())
    d1 = first(first(grp))
    win = recut_circular_param!(params, d1, draws; transforms)
    win === nothing && return nothing
    for (d, _) in grp
        d == d1 || set_circular_window!(params, d, first(win); transforms)
    end
    return win
end

"""
    recenter_circular!(chains, params; transforms = ()) -> chains

Output-side re-cut, run by `run_job` and `fit_*` on whatever the engine
returned, before anything reads the draws. Every circular parameter is moved to a
window whose seam sits in the emptiest arc of its posterior (inactive trans-dim
slots do not vote), with the median inside the user's own window, and every
draw is relabelled into it. Summaries, science tables, fit_health, plots,
`chains.nc` and the bridge evidence then see one contiguous posterior. Mutates
`chains` and the layout in place; a no-op for an engine that already re-cut.
"""
function recenter_circular!(chains::MCMCChains.Chains, params::Params;
                            transforms = ())
    idx = circular_indices(params)
    isempty(idx) && return chains
    A = chains.value.data
    vars = names(chains)
    for i in idx
        name = params.layout.unfrozen_names[i]
        j = findfirst(==(Symbol(name)), vars)
        j === nothing && continue
        col = view(A, :, j, :)
        v = Float64[x isa Real ? Float64(x) : NaN for x in vec(col)]
        mask = sci_active_mask(chains, params, name)
        draws = mask === nothing ? v : v[mask]
        moved = recut_circular_param!(params, i, draws; transforms)
        lo, hi = moved === nothing ? bounds(params.layout.unfrozen_priors[i]) : moved
        # Relabel whether or not the window moved: an engine may hand back
        # representatives outside its own window (OFTI folds Ω into [0, 2π)
        # whatever the prior; inactive trans-dim slots drift).
        for k in eachindex(col)
            x = col[k]
            x isa Real && (col[k] = circular_relabel(x, lo, hi))
        end
    end
    return chains
end

"""
    circular_contiguous(v; period = 360.0, lo_ref = 0.0, mask = nothing) -> Vector{Float64}

Copy of angle draws `v` (units set by `period`) relabelled into the window whose
seam sits in the emptiest arc of the draws (masked, finite), median inside
`[lo_ref, lo_ref + period)`. For reporting an angle that was not sampled through
a moved window: derived ω and λ in degrees, or a sampled angle whose prior is not
a full circle. Non-finite entries pass through unchanged.
"""
function circular_contiguous(v::AbstractVector{<:Real};
                             period::Real = 360.0, lo_ref::Real = 0.0,
                             mask = nothing)
    out = Float64.(v)
    draws = mask === nothing ? out : out[mask]
    lo = circular_window(draws, lo_ref, lo_ref; period)
    @inbounds for k in eachindex(out)
        out[k] = circular_relabel(out[k], lo, lo + period; period)
    end
    return out
end
