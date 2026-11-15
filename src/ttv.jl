# Per-transit free time offsets (TTV-A).
#
# Each planet with `:TTV` in its PlanetDataSources can carry N free
# parameters `ttv_kK_tI` (I = 1..N) — additive offsets applied to the
# transit centres. Slot I is the (I − 1)-th transit after the planet's
# FIRST OBSERVED transit: data-point time `t` near transit i gets
# corrected as
#
#   t_eff = t − ttv_kK_t(i − n0 + 1)
#
# where i = round((t − Tc) / P) is the integer transit number counted
# from Tc, and n0 = round((t_first − Tc) / P) is the number of the
# planet's first observed transit, t_first the earliest time among the
# planet's data: the photometry's cadences and its measured transit times
# (`ttv_first_time`). n0 is recomputed at every evaluation, so a transit
# keeps its slot whatever Tc a draw has: the cycle count between two
# observed times does not depend on where Tc sits. The measured transit
# times use the same mapping (src/transit_times_likelihood.jl), so a
# transit gets one slot from both kinds of data.
#
# The slots used to be numbered from Tc itself (slot I = transit I − 1
# from Tc). Under the default anchor, φ_c, Tc is the transit nearest the
# middle of the data, and every transit before it fell outside the slots
# and was fitted with no offset at all.
#
# Tc is the time of mid-transit: the `Tc` parameter itself under the :Tc
# time parametrization, and under :Tp and :Mo the transit within half a
# period of periastron passage, `tp_to_tc(Tp, P, e, ω)`. The likelihood used
# to number from Tp. Tp and that Tc are up to half a period apart (e.g. for
# a transit at apoastron, ω = −π/2), and then a transit sat across the
# boundary round((t − Tp)/P) = i + 1/2: the cadences before its centre took
# one offset and those after it the next one's.
#
# Slot indices outside [1, N] are treated as δt = 0 (no correction).
# This lets users specify N at Params construction time without
# tracking the exact data span; extra slots simply stay at their
# NormalPrior(0, σ_ttv) tight-around-zero default.

export _decode_ttv_state, ttv_effective_time

"""
    _has_active_ttv(theta, p_idx) -> Bool

Cheap predicate: does any active planet in `p_idx` have `:TTV` with
`ttv_n_transits[k] > 0`? Used by `transit_log_likelihood` to bypass
the orbit-only workspace flux-cache when TTV offsets are in play.
"""
@inline function _has_active_ttv(theta::Theta, p_idx)
    config = theta.params.config
    @inbounds for k in p_idx
        modes = config.planet_modes[k]
        if has_ttv(modes) && get(config.ttv_n_transits, k, 0) > 0
            return true
        end
        has_ttv_nb(modes) && return true
    end
    return false
end

"""
    _decode_ttv_state(theta, p_idx) -> (n_ttv, state) | (0, nothing)

Pre-decode TTV state for each TTV-enabled active planet. Returns
`(0, nothing)` when no planet has `:TTV`. Otherwise:

- `n_ttv::Int` — number of TTV planets
- `state.j_active::Vector{Int}` — index `j ∈ 1..n_transit` into the
  pre-decoded orbit arrays (`Ps`, `Tps`, …)
- `state.k_planet::Vector{Int}` — original planet index `k` (for
  layout name lookup)
- `state.δts::Vector{Vector{T}}` — per-planet δt values, length =
  `config.ttv_n_transits[k]`
- `state.n0s::Vector{Int}` — the transit number (from Tc) of `δts[r][1]`.
  For a `:TTV` row, the planet's first observed transit,
  `round((t_first − Tc)/P)` with `t_first = ttv_first_time(data, k)`, so slot
  I is the (I − 1)-th transit after it; 0 (slot I = transit I − 1 from Tc)
  without `data` or when the planet has no data. For a `:TTV_NB` row, set by
  `_apply_ttv_nb!`, the first transit of the predicted grid, which reaches
  back before Tc when the photometry does.

Pass `data` wherever a likelihood reads the state: without it the free
offsets are numbered from Tc.
"""
function _decode_ttv_state(theta::Theta{T}, p_idx, data = nothing) where {T}
    config = theta.params.config
    layout = theta.params.layout

    # Count rows: TTV-A planets (with n_ttv > 0) AND TTV-NB planets.
    n_ttv = 0
    j = 0
    for k in p_idx
        has_pm(config.planet_modes[k]) || continue
        j += 1
        is_a  = has_ttv(config.planet_modes[k]) &&
                  get(config.ttv_n_transits, k, 0) > 0
        is_nb = has_ttv_nb(config.planet_modes[k])
        (is_a || is_nb) || continue
        n_ttv += 1
    end
    n_ttv == 0 && return (0, nothing)

    j_active = Vector{Int}(undef, n_ttv)
    k_planet = Vector{Int}(undef, n_ttv)
    δts      = Vector{Vector{T}}(undef, n_ttv)
    is_nb    = Vector{Bool}(undef, n_ttv)
    n0s      = zeros(Int, n_ttv)

    r = 0
    j = 0
    for k in p_idx
        has_pm(config.planet_modes[k]) || continue
        j += 1
        is_a  = has_ttv(config.planet_modes[k]) &&
                  get(config.ttv_n_transits, k, 0) > 0
        is_nb_k = has_ttv_nb(config.planet_modes[k])
        (is_a || is_nb_k) || continue
        r += 1
        j_active[r] = j
        k_planet[r] = k
        is_nb[r]    = is_nb_k
        if is_a
            N  = get(config.ttv_n_transits, k, 0)
            vs = Vector{T}(undef, N)
            @inbounds for i in 1:N
                vs[i] = theta.values[layout.name_to_idx["ttv_k$(k)_t$i"]]
            end
            δts[r] = vs
            data === nothing || (n0s[r] = _ttv_first_transit(theta, data, k))
        else
            # TTV-NB row: leave δts empty until _apply_ttv_nb! fills it.
            δts[r] = T[]
        end
    end
    return (n_ttv, (; j_active, k_planet, δts, is_nb, n0s))
end

"""
    ttv_first_time(data, k) -> Float64 or nothing

The earliest time among planet `k`'s data, which the free TTV offsets are
numbered from: the first photometric cadence and planet `k`'s first measured
transit time in the likelihood, whichever is earlier. `nothing` with neither.
"""
function ttv_first_time(data, k::Int)
    t = isempty(data.t_phot) ? Inf : Float64(minimum(data.t_phot))
    for s in data.transit_times
        s.planet == k || continue
        for i in s.used
            t = min(t, s.tc[i])
        end
    end
    return isfinite(t) ? t : nothing
end

# The transit number, counted from planet k's Tc at `theta`, of its first
# observed transit: round((t_first − Tc)/P). 0 without data, or for an
# unphysical orbit (which the likelihood refuses anyway). An integer decision,
# taken on the values of a Dual.
function _ttv_first_transit(theta::Theta, data, k::Int)
    t_first = ttv_first_time(data, k)
    t_first === nothing && return 0
    pc = theta.params.config.parametrization
    P = ForwardDiff_value(planet_P(theta, k))
    e, w = planet_e_w(theta, k)
    e, w = ForwardDiff_value(e), ForwardDiff_value(w)
    (0 <= e < 1 && P > 0 && isfinite(P)) || return 0
    Tp = anchor_to_tp(pc.time, ForwardDiff_value(planet_time_anchor(theta, k)), P, e, w,
                      data.t_ref, pc.t_ref_c)
    x = (t_first - tp_to_tc(Tp, P, e, w)) / P
    (isfinite(x) && abs(x) < 1e9) || return 0
    return round(Int, x)
end

"""
    ttv_effective_time(t, P, Tc, δts, n0 = 0) -> t_eff

Return the TTV-corrected time at which to evaluate `sky_separation`.
`i = round((t − Tc) / P)` is the integer transit number counted from `Tc`,
the time of mid-transit (not of periastron). `δts[1]` is the offset of
transit `n0`: for the free offsets, the planet's first observed transit
(`_decode_ttv_state` with `data`). If `i − n0 + 1` is within
`[1, length(δts)]`, returns `t − δts[i − n0 + 1]`; otherwise returns `t`
unchanged.
"""
@inline function ttv_effective_time(t::Real, P::Real, Tc::Real,
                                      δts::AbstractVector{<:Real}, n0::Int = 0)
    isempty(δts) && return t
    i = Int(round((t - Tc) / P))     # transit number (can be negative)
    slot = i - n0 + 1                 # 1-based slot
    1 <= slot <= length(δts) || return t
    return t - δts[slot]
end

"""
    ttv_effective_time_r(t, r, ttv_state, Ps, Tcs) -> t_eff

Lookup form for the transit-likelihood inner loop. Given that planet
index `j` (1..n_transit) is a TTV planet at position `r` in
`ttv_state.j_active`, returns the TTV-corrected time. If planet `j`
has no TTV, returns `t` unchanged. `Tcs[j]` is planet `j`'s time of
mid-transit (`tp_to_tc`), the epoch its transits are numbered from.

Caller must pre-find `r` (or pass `r = findfirst(==(j), j_active)`).
For speed, the inner loop should iterate over `r` in `1:n_ttv` and
extract `j_active[r]` instead of doing the `findfirst` per obs.
"""
@inline function ttv_effective_time_r(t::Real, r::Int, ttv_state,
                                         Ps::AbstractVector,
                                         Tcs::AbstractVector)
    j = ttv_state.j_active[r]
    return ttv_effective_time(t, Ps[j], Tcs[j], ttv_state.δts[r], ttv_state.n0s[r])
end
