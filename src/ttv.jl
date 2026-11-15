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
# from Tc. Slot indices outside [1, N] are treated as δt = 0 (no
# correction).
#
# n0 is the transit number (from Tc) of the earliest point of the
# planet's data that lies within a quarter period of a predicted centre,
# |frac((t − Tc)/P)| < 1/4: the photometry's cadences in time order and
# the planet's measured transit times, which sit at centres and qualify at
# once (`_ttv_first_transit`). It is recomputed at every evaluation, so a
# transit keeps its slot whatever Tc a draw has: the cycle count between
# two observed times does not depend on where Tc sits. The measured
# transit times use the same mapping (src/transit_times_likelihood.jl), so
# a transit gets one slot from both kinds of data.
#
# The quarter-period window is what makes n0 robust. It used to be the
# number of the earliest time overall, round((t_first − Tc)/P), and a light
# curve cropped to ±P/2 around its transits starts half a period from a
# centre: the rounding then tied, and draws on either side of the tie
# numbered every transit one apart. A point P/4 from a centre that fails
# the test is followed by points of the same transit window or by none.
# Before that, the slots were numbered from Tc itself, and under the
# default anchor, φ_c, every transit before the one nearest the middle of
# the data fell outside them and was fitted with no offset.
#
# Tc is the time of mid-transit: the `Tc` parameter itself under the :Tc
# time parametrization, and under :Tp and :Mo the transit within half a
# period of periastron passage, `tp_to_tc(Tp, P, e, ω)`. The likelihood used
# to number from Tp. Tp and that Tc are up to half a period apart (e.g. for
# a transit at apoastron, ω = −π/2), and then a transit sat across the
# boundary round((t − Tp)/P) = i + 1/2: the cadences before its centre took
# one offset and those after it the next one's.

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
  For a `:TTV` row, the planet's first observed transit
  (`_ttv_first_transit`), so slot I is the (I − 1)-th transit after it; 0
  (slot I = transit I − 1 from Tc) without `data` or when the planet has no
  data. For a `:TTV_NB` row, set by
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

# Planet k's earliest measured transit time in the likelihood; Inf without.
function _ttv_first_tt(data, k::Int)
    t = Inf
    for s in data.transit_times
        s.planet == k || continue
        for i in s.used
            t = min(t, s.tc[i])
        end
    end
    return t
end

"""
    ttv_first_time(data, k) -> Float64 or nothing

The earliest time among planet `k`'s data: its first photometric cadence or
its first measured transit time in the likelihood, whichever is earlier.
`nothing` with neither.
"""
function ttv_first_time(data, k::Int)
    t = min(isempty(data.t_phot_sorted) ? Inf : data.t_phot_sorted[1],
            _ttv_first_tt(data, k))
    return isfinite(t) ? t : nothing
end

"""
    ttv_first_transit(ts, t_tt, Tc, P) -> Int or nothing

The transit number, counted from `Tc`, of the first observed transit: that of
the earliest point within a quarter period of a predicted centre,
`|frac((t − Tc)/P)| < 1/4`, among the cadences `ts` (sorted) and the measured
transit time `t_tt` (`Inf` for none), which sits at a centre and qualifies at
once. `nothing` when no point qualifies.

A cadence that fails the test sends the scan straight to the first cadence of
the next transit window (a binary search), so it costs a few comparisons
whatever the photometry's length.
"""
function ttv_first_transit(ts::AbstractVector{Float64}, t_tt::Float64,
                           Tc::Float64, P::Float64)
    n = length(ts)
    i = 1
    while i <= n
        t = @inbounds ts[i]
        t < t_tt || break
        x = (t - Tc) / P
        m = round(x)
        abs(x - m) < 0.25 && return Int(m)
        # Outside every window: the next one is transit m when x sits before
        # m's window, m + 1 when after it. `max` keeps the scan moving when a
        # cadence sits exactly on the window's edge.
        nxt = x < m ? m : m + 1
        i = max(searchsortedfirst(ts, Tc + (nxt - 0.25) * P), i + 1)
    end
    isfinite(t_tt) && return round(Int, (t_tt - Tc) / P)
    return nothing
end

# The transit number, counted from planet k's Tc at `theta`, of its first
# observed transit (`ttv_first_transit` over the photometry and the planet's
# transit times). Without a point within a quarter period of a centre, that
# of the earliest point; 0 without data, or for an unphysical orbit (which the
# likelihood refuses anyway). An integer decision, taken on the values of a
# Dual.
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
    Tc = Float64(tp_to_tc(Tp, P, e, w))
    ts = data.t_phot_sorted
    t_tt = _ttv_first_tt(data, k)
    t_last = max(isempty(ts) ? t_first : ts[end], isfinite(t_tt) ? t_tt : t_first)
    x0, x1 = (t_first - Tc) / P, (t_last - Tc) / P
    (isfinite(x0) && isfinite(x1) && abs(x0) < 1e9 && abs(x1) < 1e9) || return 0
    n0 = ttv_first_transit(ts, t_tt, Tc, Float64(P))
    return n0 === nothing ? round(Int, x0) : n0
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
