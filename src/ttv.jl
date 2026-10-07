# Per-transit free time offsets (TTV-A).
#
# Each planet with `:TTV` in its PlanetDataSources can carry N free
# parameters `ttv_kK_tI` (I = 1..N) — additive offsets applied to the
# i-th transit centre. Mapping: data-point time `t` near the i-th
# transit gets corrected as
#
#   t_eff = t − ttv_kK_tI(i+1)
#
# where i = round((t − Tc) / P) is the integer transit number relative
# to the FIRST transit (Tc). The 1-based slot index is `i+1` (Julia
# arrays are 1-based; transit 0 → slot 1, transit 1 → slot 2, …).
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
- `state.n0s::Vector{Int}` — the transit number (from Tc) of `δts[r][1]`:
  0 for a `:TTV` row, whose slot I is transit I − 1; for a `:TTV_NB` row,
  set by `_apply_ttv_nb!`, the first transit of the predicted grid, which
  reaches back before Tc when the photometry does
"""
function _decode_ttv_state(theta::Theta{T}, p_idx) where {T}
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
        else
            # TTV-NB row: leave δts empty until _apply_ttv_nb! fills it.
            δts[r] = T[]
        end
    end
    return (n_ttv, (; j_active, k_planet, δts, is_nb, n0s))
end

"""
    ttv_effective_time(t, P, Tc, δts, n0 = 0) -> t_eff

Return the TTV-corrected time at which to evaluate `sky_separation`.
`i = round((t − Tc) / P)` is the integer transit number relative to
the first transit at `Tc`, the time of mid-transit (not of periastron).
`δts[1]` is the offset of transit `n0` (0, transit Tc itself, for the free
offsets). If `i − n0 + 1` is within `[1, length(δts)]`, returns
`t − δts[i − n0 + 1]`; otherwise returns `t` unchanged.
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
