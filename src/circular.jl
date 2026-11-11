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
# The orbital inclination is cyclic in a different way, and is FOLDED rather
# than re-charted. i ∈ [0, π] is not an angle on a circle, but every likelihood
# reads it through `planet_inc`, which returns |x| reduced through x ≡ x + 2π
# (`fold_inclination`): RVs see |K| (or M_sec·|sin x|), transits cos i, and
# astrometry the sky projection, so x, -x and x + 2π are one orbit. A Sine
# prior on `inc_k<N>` is therefore sampled on a circle of circumference 2π
# (`WrappedSinePrior`: density |sin x|/4 on the chart [-π/2, 3π/2], identity
# transform), which is exactly sin(i)/2 once folded. Under the bounded Sine the
# samplers met a wall at i = π: the logit sent it to y = ±∞, walkers piled
# against it, and a near-face-on retrograde orbit was reported as railed at its
# upper bound. On the chart 0 and π are interior points, and a walker passes
# through π into the mirror representative 2π - i; the chart's edges, the
# mirror image of i = 90°, are its only walls, and the density is zero past
# them because on the whole line it would be improper (see `WrappedSine`).
# Each representative carries the whole physical posterior (x and -x are
# separated by the zeros of |sin x| at 0 and π), so there is no window to move:
# the draws are folded to [0, π] instead, by `recenter_circular!` before
# anything reads them, and the walkers are folded wherever this file's windows
# re-chart them (prior draws, `circular_relabel_point!`, the samplers' burn-in
# re-cuts), which keeps an ensemble in one representative. Both are exact: the
# likelihood is the same at x and at fold(x), to the bit, and the prior up to
# the rounding of sin(2π - x) against sin(x).
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
#   * the inclinations, which are folded instead (above).
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
`[ref, ref + 2π)`.
"""
function circular_ref_lo(params::Params, i::Int)
    ps = params.layout.unfrozen_priors[i]
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

# The same prior over a moved chart: Uniform stays Uniform, a WrappedUniform
# stays one, a pushforward keeps its density (it is periodic) on the new chart.
# Only ever called for an `is_circular` parameter, whose prior is one of these.
function _rewindow(ps::PriorSpec, lo::Real, hi::Real)
    ps.dist isa WrappedUniform && return PriorSpec(WrappedUniform(lo, hi), lo, hi)
    ps.dist isa WrappedPushforward && return PriorSpec(_rechart(ps.dist, lo, hi), lo, hi)
    return UniformPrior(lo, hi)
end

"""
    set_circular_window!(params, i, lo; transforms = ()) -> (lo, hi)

Move circular unfrozen parameter `i` to the window `[lo, lo + span)` everywhere
the samplers read its bounds: the layout's `PriorSpec`, the packed priors, and
every `PackedTransforms` in `transforms`. `config.priors` keeps what the user
wrote. Exact: the prior is periodic and still covers one period (Uniform, or a
pushforward evaluated on the circle), so its density does not change.

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
relabel all of its own state into `(lo, hi)`.
"""
function recut_circular_param!(params::Params, i::Int,
                               draws::AbstractVector{<:Real}; transforms = ())
    ps = params.layout.unfrozen_priors[i]
    lo_ref = circular_ref_lo(params, i)
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
    # A cyclic inclination is folded to [0, π] (its own representative): an
    # `init` written as 3.3 or -0.4 starts at 2π - 3.3 or 0.4.
    fold_cyclic_inclinations!(x, params)
    return x
end

# ------------------------------------------------------------------
# Polar angles (cyclic inclinations): folded, not re-charted (see the header).
# ------------------------------------------------------------------

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
the walkers), and each full-circle angle's window moved so its seam sits in the
emptiest arc of the walkers ([`recut_circular_param!`](@ref), which also moves
the transform's bounds). Returns whether any walker moved. Through a logit the
move changes the walker's unconstrained coordinate and its Jacobian, so the
caller re-evaluates its cached log-densities when this returns `true`.
"""
function recut_walkers!(X::AbstractMatrix{Float64}, params::Params, transform)
    cinc = cyclic_inclination_indices(params)
    circ = circular_indices(params)
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
