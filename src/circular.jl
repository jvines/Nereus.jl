# Circular parameters — the 0/2π seam is a coordinate choice, not a wall.
#
# Mo, the transit phase φ_c (`phic`), Ω, λ and (under the `:ew` parametrization)
# ω reach every likelihood only through sin/cos or Kepler's equation. With a
# Uniform prior spanning exactly one period, the prior window [lo, lo + 2π) is a
# CHART of the circle, and where its seam sits is arbitrary. The samplers
# nevertheless treat `lo` / `hi` as hard walls: eval_packed_logpdf rejects
# anything outside, the logit transform sends the seam to y = ±∞, and the
# nested samplers wall the unit cube. A posterior centred on the seam is then
# cut in two -- one mode shows up as two lobes at
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
# The orbital inclination is cyclic in a different way. i ∈ [0, π] is not an
# angle on a circle, but every likelihood reads it through `planet_inc`, which
# returns |x| reduced through x ≡ x + 2π (`fold_inclination`): RVs see |K| (or
# M_sec·|sin x|), transits cos i, and astrometry the sky projection, so x, -x
# and x + 2π are one orbit. A Sine prior on `inc_k<N>` is therefore sampled on
# a circle of circumference 2π (`WrappedSinePrior`: density |sin x|/4, identity
# transform), which is exactly sin(i)/2 once folded. Under the bounded Sine the
# samplers met a wall at i = π: the logit sent it to y = ±∞, walkers piled
# against it, and a near-face-on retrograde orbit was reported as railed at its
# upper bound. On the circle 0 and π are points a walker passes through, into
# the mirror representative 2π - i (or -i).
#
# The circle still needs a chart, [c, c + 2π], with walls at its seam, and the
# density is zero past them because on the whole line it would be improper
# (see `WrappedSine`). The posterior on the circle is the physical one twice
# over, at x and at its mirror -x, so a seam can cut the mirror copy as surely
# as the original. The first chart, [-π/2, 3π/2], has its seam on the mirror
# image of i = 90°: every edge-on posterior -- an astrometric orbit of a
# transiting planet, inc ≈ 90° from RVs and astrometry, a gravity-darkened star
# seen equator-on -- had its mirror mode cut in two by the walls, and walkers
# that reached it through a pole, or that swaps carried down from a hot rung,
# sat against them. The density is 2π-periodic, so the chart is moved exactly
# as a full-circle angle's window is, by the same re-cuts at the same points
# of every sampler: the seam goes where the pooled draws and their mirror
# images are thinnest (`recut_circular_param!`), which is a pole for an
# edge-on posterior and stays on the equator's image for a face-on or inclined
# one. Of the two mirror-equivalent places the cut may pick, it takes the one
# in [π, 2π], so the chart always holds the physical half [0, π].
#
# Each representative carries the whole physical posterior, so the draws are
# also folded to [0, π]: by `recenter_circular!` before anything reads them,
# and in the samplers wherever a chart may move (prior draws,
# `circular_relabel_point!`, the burn-in re-cuts), which keeps an ensemble in
# one representative. Both are exact: the likelihood is the same at x and at
# fold(x), to the bit, and the prior up to the rounding of sin(2π - x) against
# sin(x). With the chart holding [0, π] the folded value is always inside it.
#
# Every polar angle is cyclic this way, whatever prior covers [0, π]: the
# orbital inclination and the stellar inclination i★ (which gravity darkening
# reads through `system_i_star`, folded like `planet_inc`). A Sine prior becomes
# `WrappedSinePrior`; a Uniform(0, π) or a Normal truncated to [0, π] becomes
# `PolarFoldPrior` -- the user's density reflected at both poles onto the same
# chart, `p(fold(x))/2`, so it folds back to exactly what was written
# (`polar_lift` in `_build_layout`). A prior on part of [0, π] keeps its walls.
#
# Gravity darkening has one more exact symmetry: (i★, λ) and (π - i★, λ + π),
# the spin vector and its reverse, are one light curve. Without RM velocities
# or a tomogram to measure the sense of rotation, both representatives are
# sampled and the draws put in one by `spin_flip` (see there), at the same
# places the inclinations are folded.
#
# A full-circle angle under a prior that is not a uniform over one period is
# still a full circle: the likelihood is periodic, so a Normal on the line, or a
# Uniform over several periods, induces on the circle its pushforward
# p̃(θ) = Σ_n p(θ + 2πn) -- the same posterior over orbits, the same evidence.
# `circular_lift` replaces such a prior with `WrappedPushforward`, that density
# on a one-period chart, and from then on it is charted and re-cut like the
# uniform. `config.priors` keeps the prior as written. WrappedUniform, once a
# chart with no walls, is now the circular Uniform: on the whole line its
# density is improper, and the affine stretch moves spread the ensemble
# geometrically over it (measured: every cold walker of an RM fit outside
# [-π, π] by the end of the run, hot rungs at |λ| ~ 1e12 rad). A wall at the
# chart edge and a re-cut that moves the seam off the posterior is exact; a
# walker re-wrapped mid-run is not a valid stretch move.
#
# Deliberately NOT circular:
#   * priors that cover only an arc of the circle. A narrow arc, a Normal
#     truncated inside one period: those walls are the user's, and the
#     reporters say a rail against them is the prior's (`restricted_angles`).
#   * the inclinations, which are polar, not full-circle (above): their
#     charts move through the same re-cuts (`seam_indices`), but
#     `is_circular` and everything that reports a full-circle angle
#     leave them out.
#   * Mo and φ_c whenever TTVs are modelled. TTV epoch numbering hangs off the
#     transit time Tc = tp_to_tc(Tp) with Tp = t_ref − Mo·P/2π (under φ_c,
#     Tc = t_ref_c + φ_c·P/2π), so a 2π shift relabels every transit and the
#     likelihood is not periodic in the anchor there.
#   * Mo and φ_c under the O'Neil observation-based prior (`obs_prior`). Its
#     Jacobian carries the mean anomaly at each relative-astrometry epoch
#     LINEARLY (the 3M terms of |2(e²−2)sinE + e(3M + sin2E) + 3M cosE|), so
#     shifting the anchor by 2π (Tp by P) changes the density.

const CIRCULAR_PERIOD = 2π

# Per-planet angles only. Anchored, so `gp_act_lambda_e` and the ARMA `omega_*`
# coefficients can never match; `w_k` exists only under the `:ew`
# parametrization, where it is the argument of periastron.
const _CIRCULAR_NAME = r"^(Mo|phic|Omega|lambda|w)_k\d+$"

# Wide enough for 2π typed to four decimals (6.2832 is off by 2.4e-6
# relative). The window keeps its own span when it moves, so the prior density
# never changes; a span a hair short of 2π only means a relabelled value can
# land in the sliver past `hi`, where `circular_relabel` pins it to `hi`.
const _CIRCULAR_SPAN_RTOL = 1e-4

_ttv_modelled(config) =
    any(m -> has_ttv(m) || has_ttv_nb(m), config.planet_modes)

"""
    periodic_angle(name, config::ParamsConfig) -> Bool

`true` when the likelihood reads the parameter `name` only periodically, so
`x` and `x + 2π` are one state: Mo, φ_c, Ω, λ and (under `:ew`) ω of a planet,
except Mo and φ_c under TTVs or the O'Neil observation-based prior (see the
header of `src/circular.jl`).
"""
function periodic_angle(name::AbstractString, config)
    m = match(_CIRCULAR_NAME, name)
    m === nothing && return false
    if m.captures[1] == "Mo" || m.captures[1] == "phic"
        _ttv_modelled(config) && return false
        _obs_prior_on(config) && return false
    end
    return true
end

"""
    is_circular(name, ps::PriorSpec, config::ParamsConfig) -> Bool

`true` when the parameter is a [`periodic_angle`](@ref) whose prior is a chart
of exactly one period -- a Uniform over one period, a `WrappedUniformPrior`,
or the [`WrappedPushforward`](@ref) the layout builds from any other prior
covering a period or more (`circular_lift`) -- so the seam may be moved.
See the header of `src/circular.jl` for what is excluded and why.
"""
function is_circular(name::AbstractString, ps::PriorSpec, config)
    periodic_angle(name, config) || return false
    ps.dist isa WrappedPushforward && return true
    (ps.dist isa Distributions.Uniform || ps.dist isa WrappedUniform) || return false
    return abs((ps.hi - ps.lo) - CIRCULAR_PERIOD) <=
           _CIRCULAR_SPAN_RTOL * CIRCULAR_PERIOD
end

"""
    circular_ref_lo(params, i) -> Float64

Lower edge of the window the user's convention puts circular unfrozen
parameter `i` in: its prior's own `lo` when that prior was one period, else
the reference edge its [`WrappedPushforward`](@ref) carries (`μ - π` for a
Normal, `blo` for a Uniform over several periods). Reported medians land in
`[ref, ref + 2π)`. For a polar angle, whose draws are reported folded, the
first chart's `-π/2`.
"""
function circular_ref_lo(params::Params, i::Int)
    ps = params.layout.unfrozen_priors[i]
    # A polar angle's first chart, centred on [0, π]: it has no user window.
    is_cyclic_inclination(ps) && return _WRAPPED_SINE_LO
    ps.dist isa WrappedPushforward && return ps.dist.ref
    return params.config.priors[params.layout.unfrozen_names[i]].lo
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
    seam_indices(params) -> Vector{Int}

Positions, in the unfrozen vector, of every angle whose chart the samplers
move: the full-circle angles ([`circular_indices`](@ref)) and the polar angles
on their circles ([`cyclic_inclination_indices`](@ref)), in layout order. The
re-cuts, the windows several chains share and the checkpoints go over these;
the reports keep the two kinds apart.
"""
seam_indices(params::Params) =
    sort!(vcat(circular_indices(params), cyclic_inclination_indices(params)))

"""
    polar_chart_key(params) -> NamedTuple

The checkpoint-fingerprint entry of a model with polar angles, whose charts
move ([`polar_window`](@ref)): a checkpoint written before they did holds a
re-cut history and windows without them, which a sampler that indexes its
saved history by group, or by seam angle, would misread. Empty for a model
without one, whose checkpoints stay valid.
"""
polar_chart_key(params::Params) =
    isempty(cyclic_inclination_indices(params)) ? (;) : (; polar_charts = :moving)

"""
    circular_names(params) -> Set{String}
"""
circular_names(params::Params) =
    Set{String}(params.layout.unfrozen_names[i] for i in circular_indices(params))

"""
    circular_windows(params) -> Dict{String, Tuple{Float64, Float64}}

The layout's current window of every angle whose chart moves
([`seam_indices`](@ref): full-circle and polar). Taken right after an engine
returns and BEFORE [`recenter_circular!`](@ref), it is the chart the engine
actually sampled in -- the one place a seam it could not cross still shows
(see `_run_fit_health!`). The checkpoints save it, and a resume puts it back.
"""
circular_windows(params::Params) =
    Dict{String, Tuple{Float64, Float64}}(
        params.layout.unfrozen_names[i] => bounds(params.layout.unfrozen_priors[i])
        for i in seam_indices(params))

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

# The same prior over a moved chart: Uniform stays Uniform, a WrappedUniform
# stays one, a pushforward keeps its density (it is periodic) on the new chart,
# and so does a polar angle's lifted prior. Only ever called for a
# `seam_indices` parameter, whose prior is one of these.
function _rewindow(ps::PriorSpec, lo::Real, hi::Real)
    ps.dist isa WrappedSine && return PriorSpec(WrappedSine(Float64(lo), Float64(hi)), lo, hi)
    ps.dist isa PolarFold &&
        return PriorSpec(PolarFold(ps.dist.base, Float64(lo), Float64(hi)), lo, hi)
    ps.dist isa WrappedUniform && return PriorSpec(WrappedUniform(lo, hi), lo, hi)
    ps.dist isa WrappedPushforward && return PriorSpec(_rechart(ps.dist, lo, hi), lo, hi)
    return UniformPrior(lo, hi)
end

"""
    set_circular_window!(params, i, lo; transforms = (), hi = lo + span) -> (lo, hi)

Move unfrozen parameter `i` (a full-circle or polar angle, [`seam_indices`](@ref))
to the window `[lo, lo + span)` (`[lo, hi)` given `hi`, to the bit) everywhere
the samplers read its bounds: the
layout's `PriorSpec`, the packed priors, and every `PackedTransforms` in
`transforms`. `config.priors` keeps what the user wrote. Exact: the prior is
periodic and still covers one period (Uniform, a pushforward evaluated on the
circle, or a polar angle's lifted prior), so its density does not change.

The caller must then move every stored value of the parameter into the new
window with [`circular_relabel`](@ref) -- hot rungs, ring buffers, cached
proposals -- before the next log-density evaluation, or those values sit
outside the support.
"""
function set_circular_window!(params::Params, i::Int, lo::Real; transforms = (),
                              hi::Union{Nothing,Real} = nothing)
    L = params.layout
    ps = L.unfrozen_priors[i]
    span = ps.hi - ps.lo
    lo = Float64(lo)
    hi = hi === nothing ? lo + span : Float64(hi)
    L.unfrozen_priors[i] = _rewindow(ps, lo, hi)
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
relabel all of its own state into `(lo, hi)`. A polar angle's chart is cut by
[`polar_window`](@ref), from the draws and their mirror images.
"""
function recut_circular_param!(params::Params, i::Int,
                               draws::AbstractVector{<:Real}; transforms = ())
    ps = params.layout.unfrozen_priors[i]
    new_lo = is_cyclic_inclination(ps) ? polar_window(draws, ps.lo) :
             circular_window(draws, ps.lo, circular_ref_lo(params, i))
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
    for i in seam_indices(params)
        lo, hi = bounds(L.unfrozen_priors[i])
        x[i] = circular_relabel(x[i], lo, hi)
    end
    # A cyclic inclination is folded to [0, π] (its own representative): an
    # `init` written as 3.3 or -0.4 starts at 2π - 3.3 or 0.4.
    fold_cyclic_inclinations!(x, params)
    return x
end

# ------------------------------------------------------------------
# Polar angles (cyclic inclinations): re-charted, and folded (see the header).
# ------------------------------------------------------------------

"""
    polar_window(draws, lo) -> Float64

Lower edge of the chart for a polar angle currently charted from `lo`, cut from
`draws` (any representatives; non-finite ones are ignored). The posterior on
the circle is symmetric under `x → -x`, and a sampler can reach either copy,
so the seam is placed by [`circular_cut`](@ref) on the folded draws AND their
mirror images: a pole for an edge-on posterior (its copies at ±90°), and the
current seam kept for a face-on or inclined one, where it is already clear of
both copies. A cut in `(0, π)` is replaced by its mirror in `(π, 2π)`, which is
just as empty, and the edge is returned in `[-π, 0]`: the chart then always
holds the physical half `[0, π]`, where folded values land. Returns `lo`
exactly when the seam stays.
"""
function polar_window(draws::AbstractVector{<:Real}, lo::Real)
    xs = Float64[]
    for x in draws
        isfinite(x) || continue
        f = fold_inclination(Float64(x))
        push!(xs, f, -f)
    end
    c = circular_cut(xs, lo)
    c == lo && return Float64(lo)
    s = mod(c, CIRCULAR_PERIOD)
    0 < s < π && (s = CIRCULAR_PERIOD - s)
    new_lo = s == 0 ? 0.0 : s - CIRCULAR_PERIOD
    # The same seam as `lo`, written another period over: nothing moves.
    rem2pi(new_lo - lo, RoundNearest) == 0 && return Float64(lo)
    return new_lo
end

"""
    polar_centre_lo(x) -> Float64

The chart edge a single polar value `x` (a caller's `init`, an optimum)
places the seam by: as far as it can be from `fold(x)` and its mirror, which is
the pole farther from them -- `π` (chart `[-π, π]`) for an inclination below
90°, `0` (chart `[0, 2π]`) from 90° up. Not the antipode of `x`, which is the
mirror image of 90° for an edge-on value.
"""
polar_centre_lo(x::Real) = fold_inclination(Float64(x)) < π / 2 ? -Float64(π) : 0.0

"""
    prior_centre(ps) -> Float64

A point at the middle of prior `ps` for code that parks or starts a walker
there: its median, or for a polar angle the median on its first chart
`[-π/2, 3π/2]` (`π/2` for a Sine prior, always in `[0, π]`, which every chart
holds). A moved chart's own median can be a pole, where a Sine prior's density
is zero.
"""
prior_centre(ps::PriorSpec) = is_cyclic_inclination(ps) ?
    circular_relabel(quantile(_rewindow(ps, _WRAPPED_SINE_LO, _WRAPPED_SINE_HI), 0.5),
                     ps.lo, ps.hi) :
    quantile(ps, 0.5)

"""
    moved_off_value(off, ps, shift) -> Float64

A MoMS off-value (the parked value and birth centre of an inactive planet's
slot) after its parameter's window moved by `shift` to `ps`'s: a full-circle
angle's moves with the window, keeping its place in it; a polar angle's is
relabelled into the new chart, keeping its inclination, since a fixed place in
the chart can be a pole.
"""
moved_off_value(off::Real, ps::PriorSpec, shift::Real) =
    is_cyclic_inclination(ps) ? circular_relabel(Float64(off), ps.lo, ps.hi) :
                                off + shift

"""
    cyclic_inclination_indices(params) -> Vector{Int}

Positions, in the unfrozen vector, of the polar angles sampled on a circle --
orbital and stellar inclinations under [`WrappedSinePrior`](@ref) or
[`PolarFoldPrior`](@ref) (see [`is_cyclic_inclination`](@ref)).
"""
cyclic_inclination_indices(params::Params) =
    Int[i for (i, ps) in enumerate(params.layout.unfrozen_priors)
        if is_cyclic_inclination(ps)]

"""
    cyclic_inclination_names(params) -> Set{String}
"""
cyclic_inclination_names(params::Params) =
    Set{String}(params.layout.unfrozen_names[i]
                for i in cyclic_inclination_indices(params))

"""
    fold_cyclic_inclinations!(x, params; idx = cyclic_inclination_indices(params),
                              sf = spin_flip(params), decide_from = nothing) -> x

Fold the cyclic inclinations of `x` to `[0, π]` with [`fold_inclination`](@ref),
then put every point in one representative of the stellar spin flip when the
model has that symmetry ([`spin_flip`](@ref); `sf`): `(i★, λ)` or
`(π - i★, λ + π)`, whichever lies on the positive side of the layout's spin
axis. `x` is an unfrozen-vector point, or any array whose LAST dimension runs
over the unfrozen parameters (a samples matrix, a sampler's `state[t, w, d]`
or ring buffer). With `decide_from` (an array of points in the same layout,
e.g. the cold walkers) the spin axis is first re-chosen from those points
([`decide_spin_axis!`](@ref)).

Exact: the log-density is the same at `x` and at its image (the likelihood bit
for bit for the fold, to rounding in sin/cos for the flip; the prior up to
rounding), so cached log-densities stay valid. A no-op without either.
"""
function fold_cyclic_inclinations!(x::AbstractArray, params::Params;
                                   idx = cyclic_inclination_indices(params),
                                   sf = spin_flip(params), decide_from = nothing)
    nd = ndims(x)
    for d in idx
        v = selectdim(x, nd, d)
        @inbounds for k in eachindex(v)
            v[k] = fold_inclination(v[k])
        end
    end
    sf === nothing && return x
    if decide_from !== nothing
        P = decide_from
        npd = ndims(P)
        decide_spin_axis!(params, vec(selectdim(P, npd, sf.istar)),
                          vec(selectdim(P, npd, sf.lams[1])))
    end
    _spin_canonical!(x, params, sf)
    return x
end

"""
    fold_cyclic_inclinations!(chains, params) -> chains

Fold every draw of each cyclic inclination in `chains` to `[0, π]`, in place,
and -- under the stellar spin-flip symmetry ([`spin_flip`](@ref)) -- choose the
spin axis from the draws and put every draw in its representative. Run by
[`recenter_circular!`](@ref), so `run_job` and `fit_*` hand on folded draws
only. Columns missing from `chains` are skipped.
"""
function fold_cyclic_inclinations!(chains::MCMCChains.Chains, params::Params)
    idx = cyclic_inclination_indices(params)
    isempty(idx) && return chains
    A = chains.value.data
    vars = names(chains)
    col_of(i) = findfirst(==(Symbol(params.layout.unfrozen_names[i])), vars)
    for i in idx
        j = col_of(i)
        j === nothing && continue
        col = view(A, :, j, :)
        for k in eachindex(col)
            x = col[k]
            x isa Real && (col[k] = fold_inclination(x))
        end
    end
    sf = spin_flip(params)
    sf === nothing && return chains
    ji = col_of(sf.istar)
    jl = [col_of(d) for d in sf.lams]
    (ji === nothing || any(isnothing, jl)) && return chains
    decide_spin_axis!(params, Float64.(vec(view(A, :, ji, :))),
                      Float64.(vec(view(A, :, jl[1], :))))
    n = params.layout.spin_axis
    wins = [bounds(params.layout.unfrozen_priors[d]) for d in sf.lams]
    for c in axes(A, 3), r in axes(A, 1)
        i_s = Float64(A[r, ji, c]); λ1 = Float64(A[r, jl[1], c])
        spin_canonical(i_s, λ1, n) && continue
        A[r, ji, c] = π - i_s
        for (q, j) in enumerate(jl)
            A[r, j, c] = circular_relabel(Float64(A[r, j, c]) + π, wins[q]...)
        end
    end
    return chains
end

# ------------------------------------------------------------------
# The stellar spin flip.
#
# Gravity darkening cannot tell a spin vector from its reverse: the brightness
# map depends on the colatitude through cos²θ only, and the disc mean and
# v_eq = v sin i★ / sin i★ through sin² i★ and cos² i★ (src/gravity_darkening.jl),
# so (i★, λ_k) and (π - i★, λ_k + π) for every λ_k are one light curve, exactly.
# The spin vector s = (sin i★ sin λ, sin i★ cos λ, cos i★) goes to -s.
# Velocities -- an RM anomaly or a Doppler shadow -- measure the sense of the
# rotation and break it.
#
# A bounded i★ on [0, π/2] once held one representative of each spin: a wall
# at i★ = 90°, against which an equator-on star railed, and which split its
# posterior into two λ lobes π apart with nothing to carry walkers between
# them. Now i★ is a polar angle on the whole of [0, π] (`polar_lift`), the
# samplers roam both representatives, and the draws are put in ONE: the one on
# the positive side of an axis n, s·n ≥ 0. Whatever n is, that is exact -- each
# draw is replaced by its own image under a symmetry of the posterior. n is
# chosen so the cut, the great circle s·n = 0, sits where the posterior is
# thinnest: the principal axis of the draws' scatter matrix E[s sᵀ] (the same
# for s and -s). n = (0, 0, 1) is the old convention i★ ≤ 90°, right for a
# star far from equator-on; for an equator-on star n lies in the sky plane, and
# the cut is a λ, not i★ = 90°.
# ------------------------------------------------------------------

"""
    SpinFlip(istar, lams)

Unfrozen positions of the stellar inclination and of every λ_k the spin flip
moves with it (see [`spin_flip`](@ref)).
"""
struct SpinFlip
    istar::Int
    lams::Vector{Int}
end

"""
    spin_flip(params) -> Union{Nothing, SpinFlip}

The stellar spin-flip symmetry `(i★, λ_k) → (π - i★, λ_k + π)` when the
posterior has it exactly, else `nothing`. It needs:

* gravity darkening, the only reader of `i_star`, and no measurement of the
  sense of rotation: no RM anomaly, no Doppler tomography (mode or residual
  maps);
* `i_star` a polar angle on its circle (`is_cyclic_inclination`) whose prior is
  symmetric about 90° (Uniform or Sine on `[0, π]`);
* every planet with an obliquity carries its λ_k as a full-circle angle under a
  uniform prior over one period, so `λ + π` costs nothing.

A Normal prior on `i_star` off 90°, a narrowed λ prior or a fixed one breaks
the symmetry, and the representatives are then not equally probable: no
canonicalisation, the sampler reports both.
"""
function spin_flip(params::Params)
    L = params.layout
    sys = L.systemic
    sys.i_star > 0 || return nothing
    modes = params.config.planet_modes
    any(m -> has_any_rm(m) || has_dt(m), modes) && return nothing
    isempty(sys.tomo_sigma_line) || return nothing
    di = findfirst(==(sys.i_star), L.unfrozen_idx)
    di === nothing && return nothing
    ps = L.unfrozen_priors[di]
    is_cyclic_inclination(ps) || return nothing
    if ps.dist isa PolarFold
        ps.dist.base.dist isa Distributions.Uniform || return nothing
    end
    lams = Int[]
    for k in eachindex(modes)
        has_obliquity(modes[k]) || continue
        name = "lambda_k$k"
        d = findfirst(==(name), L.unfrozen_names)
        d === nothing && return nothing
        pl = L.unfrozen_priors[d]
        (is_circular(name, pl, params.config) &&
         (pl.dist isa Distributions.Uniform || pl.dist isa WrappedUniform)) ||
            return nothing
        push!(lams, d)
    end
    isempty(lams) && return nothing
    return SpinFlip(di, lams)
end

"""
    SPIN_SENSE_NOTE

The note a report carries when the data measure the stellar spin axis but not
its sense ([`spin_flip`](@ref): gravity darkening without an RM anomaly or
Doppler tomography, under priors that do not pick one). The representatives
fit alike, so ψ is reported as the angle between the spin axis and the orbit
normal, folded to [0°, 90°] (`compute_derived`); i★ and λ stay in the
representative the sampler kept.
"""
const SPIN_SENSE_NOTE = "rotation sense not constrained by the data: " *
    "(i★, λ) ≡ (180°−i★, λ+180°), ψ ≡ 180°−ψ"

# The spin vector of (i★, λ), λ measured as in src/gravity_darkening.jl.
@inline _spin_vec(i::Real, λ::Real) =
    (sin(i) * sin(λ), sin(i) * cos(λ), cos(i))

"""
    spin_canonical(i_star, λ, n) -> Bool

Whether `(i_star, λ)` is the reported representative of its spin: `s·n ≥ 0`.
"""
@inline function spin_canonical(i::Real, λ::Real, n::AbstractVector)
    s = _spin_vec(i, λ)
    return s[1] * n[1] + s[2] * n[2] + s[3] * n[3] >= 0
end

"""
    decide_spin_axis!(params, i_star, λ) -> n

Choose the spin-flip axis from draws of the (folded) stellar inclination and
the first λ: the principal axis of `E[s sᵀ]`, so the cut `s·n = 0` sits where
the draws are thinnest. The current axis is kept unless the new one cuts
significantly fewer draws (3 Poisson σ), so a posterior with no preferred axis
does not move it on noise.
"""
function decide_spin_axis!(params::Params, istar::AbstractVector, lam::AbstractVector)
    T = zeros(3, 3)
    S = NTuple{3,Float64}[]
    for (i, λ) in zip(istar, lam)
        (isfinite(i) && isfinite(λ)) || continue
        s = _spin_vec(fold_inclination(Float64(i)), Float64(λ))
        push!(S, s)
        for a in 1:3, b in 1:3
            T[a, b] += s[a] * s[b]
        end
    end
    n_cur = params.layout.spin_axis
    isempty(S) && return n_cur
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(T))
    n_new = E.vectors[:, argmax(E.values)]
    near(n) = count(s -> abs(s[1] * n[1] + s[2] * n[2] + s[3] * n[3]) < sind(5), S)
    c_cur, c_new = near(n_cur), near(n_new)
    if c_new + 3 * sqrt(c_new + 1) < c_cur
        n_cur .= n_new ./ LinearAlgebra.norm(n_new)
    end
    return n_cur
end

# Put every point of `x` (last dimension = the unfrozen vector) in the
# representative on the positive side of the layout's spin axis. λ's are
# relabelled into their current windows after the + π.
function _spin_canonical!(x::AbstractArray, params::Params, sf::SpinFlip)
    n = params.layout.spin_axis
    wins = [bounds(params.layout.unfrozen_priors[d]) for d in sf.lams]
    nd = ndims(x)
    if nd == 1
        _spin_canonical_point!(x, sf, n, wins)
        return x
    end
    for I in CartesianIndices(axes(x)[1:nd-1])
        _spin_canonical_point!(view(x, I, :), sf, n, wins)
    end
    return x
end

@inline function _spin_canonical_point!(v::AbstractVector, sf::SpinFlip, n, wins)
    i = fold_inclination(v[sf.istar])
    spin_canonical(i, v[sf.lams[1]], n) && return v
    v[sf.istar] = oftype(i, π) - i
    for (q, d) in enumerate(sf.lams)
        v[d] = circular_relabel(v[d] + π, wins[q]...)
    end
    return v
end

"""
    n_symmetry_folds(params) -> Int

How many two-to-one symmetries the reported draws are folded through: one per
cyclic inclination, one more for the stellar spin flip. An evidence estimator
that integrates the folded posterior with the unfolded density finds
`Z / 2^n` and adds `n log 2`.
"""
n_symmetry_folds(params::Params) =
    length(cyclic_inclination_indices(params)) + (spin_flip(params) === nothing ? 0 : 1)

"""
    in_folded_domain(x, params, sf = spin_flip(params)) -> Bool

Whether the unfrozen-vector point `x` (bounded space) is a reported
representative: every cyclic inclination in `[0, π]` and, under the spin flip,
on the positive side of the spin axis.
"""
function in_folded_domain(x::AbstractVector, params::Params, sf = spin_flip(params);
                          cinc = cyclic_inclination_indices(params))
    for j in cinc
        0 <= x[j] <= π || return false
    end
    sf === nothing && return true
    return spin_canonical(x[sf.istar], x[sf.lams[1]], params.layout.spin_axis)
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

The angles whose charts move ([`seam_indices`](@ref): full-circle and polar)
grouped by the window they must share, as `(unfrozen position, owning planet)`
pairs; the owner is 0 for the stellar inclination, which no planet owns.
`permutable = true` for a sampler that births planets or swaps planet blocks
between slots: same-mode slots then share one window per angle. Otherwise
every parameter is its own group.
"""
function circular_groups(params::Params; permutable::Bool)
    L = params.layout
    circ = seam_indices(params)
    groups = Vector{Vector{Tuple{Int,Int}}}()
    isempty(circ) && return groups
    modes  = params.config.planet_modes
    blocks = L.planet_blocks
    done = falses(length(L.unfrozen_idx))
    for d in circ
        done[d] && continue
        # `is_circular` only accepts `<angle>_k<k>`, and a polar angle is a
        # planet's `inc_k<k>` or the star's `i_star`, owner 0: always present.
        m = match(r"_k(\d+)$", L.unfrozen_names[d])
        if m === nothing
            done[d] = true
            push!(groups, [(d, 0)])
            continue
        end
        k = parse(Int, m.captures[1])
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
        lo = circular_ref_lo(params, first(first(grp)))
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

Every cyclic inclination is folded to `[0, π]` first
([`fold_cyclic_inclinations!`](@ref)): nothing downstream sees a representative
outside it.
"""
function recenter_circular!(chains::MCMCChains.Chains, params::Params;
                            transforms = ())
    fold_cyclic_inclinations!(chains, params)
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

# ------------------------------------------------------------------
# Live convergence diagnostics read every angle on its circle.
#
# The live R-hat / ESS (`_pt_convergence`, `_td_convergence`) and the end-of-run
# `convergence_report` treat each walker as a chain. Two walkers in different
# representatives of one state -- a cyclic inclination either side of a pole,
# a spin either side of its flip, a full-circle angle either side of its seam
# -- would count as separate modes and hold R-hat up on a converged run. Each
# diagnostic row is put in its reported representative first (folded,
# spin-flipped by the layout's axis), and each full-circle angle's column is
# relabelled into the window whose seam sits in the emptiest arc of the draws.
# ------------------------------------------------------------------

"""
    AngleCanon(params)

What the live convergence diagnostics need to read the angles of a samples
buffer (columns = the unfrozen vector) in their reported representatives:
the cyclic inclinations to fold, the spin flip and its axis, the circular
angles to unwrap.
"""
struct AngleCanon
    fold::Vector{Int}
    sf::Union{Nothing, SpinFlip}
    axis::Vector{Float64}
    circ::Vector{Int}
end

AngleCanon(params::Params) =
    AngleCanon(cyclic_inclination_indices(params), spin_flip(params),
               copy(params.layout.spin_axis), circular_indices(params))

"""
    canon_row_flips(ac, samples, src) -> Bool

Whether row `src` of `samples` is the mirror representative of its spin (the
spin-flip image of the reported one).
"""
@inline canon_row_flips(ac::AngleCanon, samples::AbstractMatrix, src::Int) =
    ac.sf !== nothing &&
    !spin_canonical(fold_inclination(samples[src, ac.sf.istar]),
                    samples[src, ac.sf.lams[1]], ac.axis)

"""
    canon_value(ac, samples, src, c, flip) -> Float64

`samples[src, c]` in its reported representative: folded if `c` is a cyclic
inclination, moved through the spin flip if `flip` (see [`canon_row_flips`](@ref)).
A λ is shifted by π without relabelling: [`unwrap_circular!`](@ref) does that.
"""
@inline function canon_value(ac::AngleCanon, samples::AbstractMatrix, src::Int,
                             c::Int, flip::Bool)
    v = Float64(samples[src, c])
    c in ac.fold && (v = fold_inclination(v))
    if flip
        c == ac.sf.istar && return π - v
        c in ac.sf.lams && return v + π
    end
    return v
end

"""
    unwrap_circular!(v) -> v

Relabel the angle draws `v` (radians) into the one-period window whose seam
sits in their emptiest arc ([`circular_contiguous`](@ref)), in place.
"""
function unwrap_circular!(v::AbstractArray)
    w = circular_contiguous(vec(v); period = CIRCULAR_PERIOD, lo_ref = 0.0)
    @inbounds for (k, i) in enumerate(eachindex(v))
        v[i] = w[k]
    end
    return v
end

# ------------------------------------------------------------------
# Evidence estimators integrate the FOLDED posterior: every draw put in its
# reported representative, the density 2^n p* there (n = `n_symmetry_folds`)
# and nothing outside. `SpinCanon` freezes the spin flip, its axis and the λ
# windows for one such pass, so every draw and every proposal is judged by the
# same cut.
# ------------------------------------------------------------------

"""
    SpinCanon(params) -> Union{Nothing, SpinCanon}

The stellar spin flip of `params` with a copy of its current axis and λ
windows, or `nothing` without the symmetry ([`spin_flip`](@ref)).
"""
struct SpinCanon
    sf::SpinFlip
    axis::Vector{Float64}
    wins::Vector{NTuple{2,Float64}}
end

function spin_canon(params::Params)
    sf = spin_flip(params)
    sf === nothing && return nothing
    return SpinCanon(sf, copy(params.layout.spin_axis),
                     NTuple{2,Float64}[Float64.(bounds(params.layout.unfrozen_priors[d]))
                                       for d in sf.lams])
end

"""
    spin_canon!(x, sc) -> x

Put the bounded-space point `x` (its cyclic inclinations already folded) in
the reported representative of its spin. A no-op for `sc === nothing`.
"""
spin_canon!(x::AbstractVector, ::Nothing) = x
spin_canon!(x::AbstractVector, sc::SpinCanon) =
    _spin_canonical_point!(x, sc.sf, sc.axis, sc.wins)

"""
    spin_canon_ok(y, sc, pt) -> Bool

Whether the point `y` -- in the target's space, `pt` its transform or
`nothing` -- is the reported representative of its spin. The stellar
inclination is untransformed (identity transform on a cyclic inclination);
λ is mapped back to bounded space for the test.
"""
spin_canon_ok(y::AbstractVector, ::Nothing, pt) = true
function spin_canon_ok(y::AbstractVector, sc::SpinCanon, pt)
    j = sc.sf.lams[1]
    λ = pt === nothing ? y[j] : _bounded_coord(y[j], j, pt)
    return spin_canonical(y[sc.sf.istar], λ, sc.axis)
end

# One coordinate of `transform_inverse`.
@inline function _bounded_coord(yj::Real, j::Int, pt)
    tid = pt.type_ids[j]
    tid == TRANSFORM_IDENTITY && return yj
    lo = pt.lowers[j]
    tid == TRANSFORM_LOGIT && return lo + (pt.uppers[j] - lo) / (1 + exp(-yj))
    tid == TRANSFORM_LOGLOGIT && return _loglogit_inverse(yj, lo, pt.uppers[j])
    return lo + exp(yj)
end

"""
    n_folds(sc, cinc) -> Int

`length(cinc)` plus one under the spin flip: the `log 2`s an estimator that
integrates the folded posterior adds back.
"""
n_folds(sc, cinc) = length(cinc) + (sc === nothing ? 0 : 1)

# ------------------------------------------------------------------
# Burn-in re-cut for samplers whose walkers live in the TARGET's space.
# ------------------------------------------------------------------

"""
    recut_walkers!(X, params, transform) -> Bool

The burn-in re-cut of `sample_pt_emcee`, for an ensemble `X` (n_dim × n_walkers)
held in the target's own coordinates (`transform` a `PackedTransforms`, or
`nothing` for a bounded target): cyclic inclinations folded to `[0, π]`, the
stellar spin put in one representative ([`spin_flip`](@ref), axis chosen from
the walkers), and each full-circle or polar angle's window moved so its seam
sits in the emptiest arc of the walkers (and, for a polar one, of their mirror
images; [`recut_circular_param!`](@ref), which also moves the transform's
bounds). Returns whether any walker moved. Through a logit the
move changes the walker's unconstrained coordinate and its Jacobian, so the
caller re-evaluates its cached log-densities when this returns `true`.
"""
function recut_walkers!(X::AbstractMatrix{Float64}, params::Params, transform)
    cinc = cyclic_inclination_indices(params)
    circ = seam_indices(params)
    (isempty(cinc) && isempty(circ)) && return false
    W = Matrix{Float64}(undef, size(X, 2), size(X, 1))
    for j in axes(X, 2)
        W[j, :] .= transform === nothing ? view(X, :, j) :
                   transform_inverse(X[:, j], transform)
    end
    W0 = copy(W)
    fold_cyclic_inclinations!(W, params; idx = cinc, decide_from = W)
    tr = transform === nothing ? () : (transform,)
    for d in circ
        win = recut_circular_param!(params, d, view(W, :, d); transforms = tr)
        win === nothing && continue
        lo, hi = win
        @inbounds for j in axes(W, 1)
            W[j, d] = circular_relabel(W[j, d], lo, hi)
        end
    end
    W == W0 && return false
    for j in axes(X, 2)
        X[:, j] .= transform === nothing ? view(W, j, :) :
                   transform_forward(W[j, :], transform)
    end
    return true
end

"""
    restricted_angles(params) -> Dict{String, String}

The unfrozen angles whose prior covers only part of their range, each mapped
to that full range: a full-circle angle ([`periodic_angle`](@ref)) whose prior
is not a full period (`"[0, 2π)"`), or an inclination whose prior does not
span `[0, π]` (`"[0, π]"`). A user's deliberate sub-range is kept as written;
the reporters use this to say that a rail there is the prior's restriction
(`assess_fit`'s `angle_range`, the science tables' rail note).
"""
function restricted_angles(params::Params)
    L = params.layout
    out = Dict{String, String}()
    for (nm, ps) in zip(L.unfrozen_names, L.unfrozen_priors)
        if periodic_angle(nm, params.config)
            is_circular(nm, ps, params.config) || (out[nm] = "[0, 2π)")
        elseif occursin(r"^inc_k\d+$", nm) || nm == "i_star"
            is_cyclic_inclination(ps) || (out[nm] = "[0, π]")
        end
    end
    return out
end

# ------------------------------------------------------------------
# Several chains over one shared Params: where they all stand still.
# ------------------------------------------------------------------
#
# A circular window lives in the layout, which every chain of a multi-chain
# run shares. A chain cannot move it alone: the others would find their states
# outside the new support mid-sweep. The re-cut is therefore taken at a
# rendezvous: each chain arrives with its state at the same point of its own
# warmup (the samplers' schedules depend only on the step count, which every
# chain shares), the last to arrive runs the re-cut over the pooled states of
# all of them while the rest wait, and every chain then carries on with the
# result. No chain evaluates the target while the windows move.

"""
    _Rendezvous(n)

A meeting point for `n` chains. See [`_rendezvous!`](@ref),
[`_rendezvous_leave!`](@ref) and [`_rendezvous_abort!`](@ref).
"""
mutable struct _Rendezvous
    cond::Threads.Condition
    n::Int                    # chains still taking part
    payloads::Vector{Any}     # this round's arrivals
    gen::Int                  # rounds completed
    f::Any                    # this round's leader function
    result::Any
    err::Any
    aborted::Bool             # no round runs any more (`_rendezvous_abort!`)
end
_Rendezvous(n::Integer) = _Rendezvous(Threads.Condition(), Int(n), Any[], 0,
                                      nothing, nothing, nothing, false)

"""What `_rendezvous!` returns once the rendezvous has been aborted."""
struct _RendezvousAborted end
const _RV_ABORTED = _RendezvousAborted()

# Run the round: the leader function over every arrival's payload, its result
# (or error) left for all of them. Called with the lock held.
function _rendezvous_run!(rv::_Rendezvous)
    rv.result, rv.err = nothing, nothing
    try
        rv.result = rv.f(rv.payloads)
    catch e
        rv.err = e
    end
    rv.payloads = Any[]
    rv.gen += 1
    notify(rv.cond)
    return nothing
end

"""
    _rendezvous!(f, rv, payload) -> result

Arrive at the rendezvous with `payload`. The last chain to arrive calls
`f(payloads)` (one payload per arrival, in arrival order) while the others
wait; every chain then returns `f`'s result, or rethrows its error. `f` may
touch only the payloads and what the chains share, never the caller's locals:
whichever chain arrives last runs it. Once the rendezvous is aborted no round runs:
every chain waiting, or arriving later, gets `_RV_ABORTED` instead.
"""
function _rendezvous!(f, rv::_Rendezvous, payload)
    lock(rv.cond)
    try
        rv.aborted && return _RV_ABORTED
        push!(rv.payloads, payload)
        rv.f = f
        g = rv.gen
        if length(rv.payloads) >= rv.n
            _rendezvous_run!(rv)
        else
            while rv.gen == g && !rv.aborted
                wait(rv.cond)
            end
            rv.gen == g && return _RV_ABORTED
        end
        rv.err === nothing || throw(rv.err)
        return rv.result
    finally
        unlock(rv.cond)
    end
end

"""
    _rendezvous_leave!(rv)

A chain stops taking part (it finished, or threw): the round in progress, if
every remaining chain has arrived, runs now rather than waiting for it.
"""
function _rendezvous_leave!(rv::_Rendezvous)
    lock(rv.cond)
    try
        rv.n -= 1
        !isempty(rv.payloads) && length(rv.payloads) >= rv.n && _rendezvous_run!(rv)
    finally
        unlock(rv.cond)
    end
    return nothing
end

"""
    _rendezvous_abort!(rv)

A chain stops taking part, and no round may run without it: the round in
progress and every later one end with `_RV_ABORTED` for the chains that wait at
them. For a re-cut that must pool every chain or none (`_hmc_recut!`), where a
round of the others alone would move the windows differently from the run that
waits for all of them.
"""
function _rendezvous_abort!(rv::_Rendezvous)
    lock(rv.cond)
    try
        rv.aborted = true
        notify(rv.cond)
    finally
        unlock(rv.cond)
    end
    return nothing
end

"""
    _restore_shared_windows!(params, windows, transforms) -> Bool

Put the layout's circular windows back as a checkpoint saved them (`windows`, a
`circular_windows` Dict), before the chains of a resumed multi-chain run start.
Returns whether any moved.
"""
function _restore_shared_windows!(params::Params, windows, transforms)
    L = params.layout
    moved = false
    for (nm, (lo, _)) in windows
        i = findfirst(==(nm), L.unfrozen_names)
        i === nothing && continue
        L.unfrozen_priors[i].lo == lo && continue
        set_circular_window!(params, i, lo; transforms)
        moved = true
    end
    return moved
end

"""
    _relabel_into_windows!(values, params, saved_windows) -> Vector{(pos, shift)}

A resumed chain's slot-vector `values`, saved while the layout's windows were
`saved_windows`, relabelled by whole periods into the windows the layout holds
now. Returns each moved window's unfrozen position and `shift = lo_now -
lo_saved` (for MoMS off-values). The log-density is unchanged: a circular
angle's prior and likelihood are periodic.
"""
function _relabel_into_windows!(values::AbstractVector, params::Params, saved_windows)
    L = params.layout
    shifts = Tuple{Int,Float64}[]
    for (nm, (lo_s, _)) in saved_windows
        d = findfirst(==(nm), L.unfrozen_names)
        d === nothing && continue
        lo, hi = bounds(L.unfrozen_priors[d])
        lo == lo_s && continue
        s = L.unfrozen_idx[d]
        values[s] = circular_relabel(values[s], lo, hi)
        push!(shifts, (d, lo - lo_s))
    end
    return shifts
end

"""
    _fold_polar_slots!(values, params) -> Bool

Fold every polar angle (cyclic inclination) of a slot-vector `values` -- a
`Theta`'s, inactive planet slots included -- to `[0, π]`, its representative,
which every chart holds ([`polar_window`](@ref)): a local sampler then works in
one copy of the posterior, the one its chart was cut around. Exact for the
log-density (see [`fold_cyclic_inclinations!`](@ref)). Returns whether any
value changed.
"""
function _fold_polar_slots!(values::AbstractVector, params::Params)
    L = params.layout
    changed = false
    for d in cyclic_inclination_indices(params)
        s = L.unfrozen_idx[d]
        v = fold_inclination(values[s])
        v == values[s] && continue
        values[s] = v
        changed = true
    end
    return changed
end

# ------------------------------------------------------------------
# Nested sampling: angles in the unit cube.
# ------------------------------------------------------------------
#
# A nested sampler replaces a dead point by a walk from a random live point,
# so pieces of the constrained region that no walk crosses keep whatever share
# of the live points chance leaves them: the share random-walks, and a piece
# can die. A seam is such a crossing. Measured on an RM fit with λ = 180° on
# the seam of [-π, π]: 23 % of the mass above 180° (pt_emcee), 8 % from the
# INS sampler and 31 % from NestedSamplers' rslice.
#
# A full-circle angle has no chart without a seam, so the seam is moved: in the
# loops this package owns, to the emptiest arc of the live points at each bound
# refit (`_ns_recut!`); for NestedSamplers.jl, which owns its loop, before the
# run from a pilot. A polar angle is charted on its circle as in every other
# sampler and re-cut the same way, from the live points and their mirror
# images. Its base prior on [0, π] has walls at the poles, which cut a face-on
# or retrograde posterior; on the circle the two copies of the posterior carry
# identical likelihoods, so a copy the live points abandon does not bias the
# evidence (the volume each live point stands for is the same in both).

"""
    _ns_prior(ps) -> PriorSpec

The prior a nested sampler's unit cube maps onto: the layout's own, a polar
angle on its current chart included (see the section above).
"""
_ns_prior(ps::PriorSpec) = ps

"""
    _ns_recut!(live_u, dead_sets, params; transforms = ()) -> Bool

Nested sampling's seam re-cut, in the unit cube. Each full-circle or polar
angle's window moves to the emptiest arc of the live points (and, for a polar
one, of their mirror images; [`recut_circular_param!`](@ref)), and every live
and dead point's coordinate
`u` rotates with it: on the circle the prior's CDF from the new seam is the
old one less its value there, so `u' = mod(u - F_old(lo_new), 1)` is the same
angle. The constrained region, the live points' uniform distribution over it
and every likelihood are unchanged; the caller refits its bound. `dead_sets`
is a collection of vectors of dead points' `u`.
"""
function _ns_recut!(live_u::AbstractMatrix, dead_sets, params::Params;
                    transforms = ())
    L = params.layout
    moved = false
    for d in seam_indices(params)
        ps = L.unfrozen_priors[d]
        xs = Float64[quantile(ps, live_u[d, i]) for i in axes(live_u, 2)]
        lo0, hi0 = bounds(ps)
        win = recut_circular_param!(params, d, xs; transforms)
        win === nothing && continue
        c = Float64(cdf(ps, circular_relabel(first(win), lo0, hi0)))
        rot(u) = (v = u - c; v < 0 ? v + 1 : v)
        for i in axes(live_u, 2)
            live_u[d, i] = rot(live_u[d, i])
        end
        for set in dead_sets, du in set
            du[d] = rot(du[d])
        end
        moved = true
    end
    return moved
end

"""
    _ns_home_charts!(params; transforms = ()) -> params

Every full-circle and polar angle back on the chart the layout gave it, before
a nested run: its bounds as lifted from the user's prior (`circular_lift`,
`polar_lift`), to the bit. A run moves the charts as it goes (`_ns_recut!`) and
leaves them where it ended, with the draws it returns in them. A second run on
the same target started from those charts, so the unit cube mapped onto other
angles and the same seed gave another run; so did a chart moved back by its
lower edge alone, whose upper edge, `lo + span` from a moved window, was a bit
off.
"""
function _ns_home_charts!(params::Params; transforms = ())
    L = params.layout
    for d in seam_indices(params)
        ps = L.unfrozen_priors[d]
        ps0 = params.config.priors[L.unfrozen_names[d]]
        lo, hi = bounds(is_cyclic_inclination(ps) ? polar_lift(ps0) : circular_lift(ps0))
        bounds(ps) == (lo, hi) && continue
        set_circular_window!(params, d, lo; transforms, hi)
    end
    return params
end
