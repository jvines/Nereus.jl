# Likelihood of measured mid-transit times (`data.transit_times`).
#
# An independent Gaussian term per planet p:
#
#   ln L_tt = -½ Σ_i [ (tc_i − T(n_i))² / s_i² + ln(2π s_i²) ],
#   s_i² = tc_err_i² + σ_tt,p²,
#
# σ_tt,p the planet's `sigma_tt_k<p>` jitter, 0 unless the planet has one.
#
# The model's mid-transit times
# -----------------------------
# The mid-transit time is the time of inferior conjunction, true anomaly
# f = π/2 − ω: Nereus measures ω so that the transit is at f + ω = π/2
# (`tc_to_tp`). The model's Tc is computed as the photometric likelihood
# computes it (`_transit_ll_direct`): Tp = anchor_to_tp(time, anchor, P, e, ω,
# t_ref, t_ref_c), Tc = tp_to_tc(Tp, P, e, ω), through Kepler's equation at that
# f. It is not t0 + P·epoch with the anchor read as t0: under :Mo and :Tp the
# anchor is a mean anomaly or a periastron time, and for an eccentric orbit Tc
# is (P/2π)(E_c − e sin E_c) after Tp, E_c the eccentric anomaly at f = π/2 − ω.
# A Keplerian orbit is strictly periodic, so transit n is at
#
#   T_lin(n) = Tc + n·P
#
# under every anchor and at any eccentricity; the anchor changes how Tc is
# parametrized, never this relation. A secular period change dP/dN would add
# ½·n²·P·(dP/dN) (T(n) = Tc + n·P + ½ n² P dP/dN, the orbital-decay
# ephemeris); Nereus has no such parameter, so there is no quadratic term. On top
# of T_lin:
#
#   * :TTV (PM_TTV, RVPM_TTV, ...): T(n) = T_lin(n) + δt_{n−n0+1}, the free
#     offset `ttv_k<p>_t<n−n0+1>`, n0 the planet's first observed transit
#     (src/ttv.jl), zero outside the configured slots -- the numbering
#     `ttv_effective_time` gives the photometry, so a transit has one slot in
#     both terms.
#   * :TTV_NB (RVPM_TTV_NB, RVPMAS_TTV_NB): T(n) = T_lin(n) + δt_n, the N-body
#     transit n minus T_lin(n), from the same solver the photometry uses
#     (`_apply_ttv_nb!`: TTVFaster, or NbodyGradient under ttv_backend =
#     :nbody), integrated over the span of the photometry and the transit times
#     together (`_ttv_nb_span`), so the two terms see the same integration.
#
# Matching epochs
# ---------------
# The epochs are labels: cycle counts from someone else's zero, made with an
# ephemeris the likelihood never reads. Only their differences are used. Point i
# is transit n_i = epoch_i + Δ of the model, with one integer Δ per planet: the
# point nearest the model's Tc is matched to its nearest predicted conjunction
# (T(n) of the three transits about round((tc − Tc)/P)), and Δ is that
# transit's number minus the point's epoch.
#
# Guard: a point more than P/4 from its transit T(n_i) means the labels and the
# model disagree on which transit it is -- the assignment is ambiguous -- and
# ln L = −Inf. Wherever ln L is finite, every point is therefore matched to its
# nearest predicted conjunction AND the labels agree with that matching. A
# period that counts a different number of cycles between two points than their
# labels do (P/2, an alias across a gap) is refused rather than fitted.
#
# A planet with transit times that is not active (an empty trans-dim slot)
# leaves them unexplained: ln L = −Inf.

"""
    _ttv_nb_span(T, data) -> (t_min, t_max)

The time span the N-body TTV prediction covers: the photometry and the transit
times in the likelihood together, `(0, 0)` with neither. Without transit times
it is the photometric span, as it always was.
"""
function _ttv_nb_span(::Type{T}, data::Data) where {T}
    lo, hi = Inf, -Inf
    if !isempty(data.t_phot)
        lo, hi = minimum(data.t_phot), maximum(data.t_phot)
    end
    for s in data.transit_times, i in s.used
        lo = min(lo, s.tc[i]); hi = max(hi, s.tc[i])
    end
    isfinite(lo) || return (zero(T), zero(T))
    return (T(lo), T(hi))
end

# Planet k's model mid-transit time Tc and period P, as the photometric
# likelihood computes them; `nothing` for an unphysical orbit.
@inline function _tt_tc_P(theta::Theta{T}, data::Data, k::Int) where {T}
    pc = theta.params.config.parametrization
    P = planet_P(theta, k)
    e, w = planet_e_w(theta, k)
    (e < 0 || e >= 1 || !(P > 0) || !isfinite(P)) && return nothing
    Tp = anchor_to_tp(pc.time, planet_time_anchor(theta, k), P, e, w, data.t_ref,
                      pc.t_ref_c)
    Tc = tp_to_tc(Tp, P, e, w)
    isfinite(Tc) || return nothing
    return Tc, P
end

# Model time of transit n: T_lin(n) plus the offset of the slot it falls in.
@inline function _tt_model_time(Tc, P, n::Int, δts, n0::Int)
    slot = n - n0 + 1
    off = (δts === nothing || slot < 1 || slot > length(δts)) ? zero(Tc) : δts[slot]
    return Tc + n * P + off
end

# The integer Δ (n_i = epoch_i + Δ) of a set at (Tc, P, offsets); `nothing`
# when Tc or P is unusable.
function _tt_match(s::TransitTimeSet, Tc, P, δts, n0::Int)
    Tcv, Pv = ForwardDiff_value(Tc), ForwardDiff_value(P)
    (isfinite(Tcv) && isfinite(Pv) && Pv > 0) || return nothing
    iref = s.used[1]
    dref = abs(s.tc[iref] - Tcv)
    for i in s.used
        d = abs(s.tc[i] - Tcv)
        d < dref && (iref = i; dref = d)
    end
    x = (s.tc[iref] - Tcv) / Pv
    abs(x) < 1e12 || return nothing
    nc = round(Int, x)
    best, bd = nc, Inf
    for n in (nc - 1, nc, nc + 1)
        d = abs(s.tc[iref] - ForwardDiff_value(_tt_model_time(Tc, P, n, δts, n0)))
        d < bd && (best = n; bd = d)
    end
    return best - s.epoch[iref]
end

# Planet k's transit offsets and the transit number of the first, from the TTV
# state; `(nothing, 0)` for a planet without TTVs.
function _tt_offsets(ttv_state, k::Int)
    ttv_state === nothing && return (nothing, 0)
    r = findfirst(==(k), ttv_state.k_planet)
    r === nothing && return (nothing, 0)
    return ttv_state.δts[r], ttv_state.n0s[r]
end

# The orbits of the transiting planets in `p_idx` (`has_pm`), in the order
# `_decode_ttv_state` numbers them: what `_apply_ttv_nb!` reads. `nothing` for
# an unphysical eccentricity.
function _tt_transit_orbits(theta::Theta{T}, data::Data, p_idx) where {T}
    config = theta.params.config
    pc = config.parametrization
    ks = [k for k in p_idx if has_pm(config.planet_modes[k])]
    n = length(ks)
    Ps, es, ws, Tps, bs, a_Rs = (Vector{T}(undef, n) for _ in 1:6)
    for (j, k) in enumerate(ks)
        Ps[j] = planet_P(theta, k)
        e, w = planet_e_w(theta, k)
        (e < 0 || e >= 1) && return nothing
        es[j], ws[j] = e, w
        Tps[j] = anchor_to_tp(pc.time, planet_time_anchor(theta, k), Ps[j], e, w,
                              data.t_ref, pc.t_ref_c)
        b_raw, rr_raw = planet_b_rr(theta, k)
        bs[j] = pc.geom === :r1r2 ? b_raw * (1 - rr_raw) / 2 : b_raw
        a_Rs[j] = planet_a_Rs(theta, k, Ps[j])
    end
    return Ps, es, ws, Tps, bs, a_Rs
end

# The TTV state the transit times read (free offsets decoded, N-body offsets
# predicted), or `nothing` when no planet with transit times has TTVs. `false`
# for an unphysical orbit.
function _tt_ttv_state(theta::Theta{T}, data::Data, p_idx) where {T}
    modes = theta.params.config.planet_modes
    any(s -> n_tt(s) > 0 && s.planet <= length(modes) &&
             (has_ttv(modes[s.planet]) || has_ttv_nb(modes[s.planet])),
        data.transit_times) || return nothing
    n_ttv, state = _decode_ttv_state(theta, p_idx, data)
    n_ttv == 0 && return nothing
    if any(state.is_nb)
        orb = _tt_transit_orbits(theta, data, p_idx)
        orb === nothing && return false
        t_min, t_max = _ttv_nb_span(T, data)
        _apply_ttv_nb!(state, theta, p_idx, orb..., t_max; t_phot_min = t_min)
    end
    return state
end

"""
    transit_times_log_likelihood(theta, data) -> Real

Gaussian log-likelihood of the measured mid-transit times in
`data.transit_times`: per planet, `-½ Σ [(tc − T)²/s² + ln(2π s²)]` with
`s² = tc_err² + σ_tt²` and `T` the model's own time of the transit each point is
matched to (the linear ephemeris of the orbit, plus the free offsets of a
`:TTV` planet or the N-body offsets of a `:TTV_NB` one). `-Inf` when a point is
more than a quarter period from its transit (the epoch assignment is ambiguous)
or when a planet with transit times is not active. Exactly zero without transit
times. Added to the total by `rv_log_likelihood`, beside the astrometry. See
the comment at the top of src/transit_times_likelihood.jl for the derivation.
"""
function transit_times_log_likelihood(theta::Theta{T}, data::Data) where {T}
    sets = data.transit_times
    isempty(sets) && return zero(T)
    p_idx = planet_indices(theta)
    state = _tt_ttv_state(theta, data, p_idx)
    state === false && return convert(T, -Inf)
    two_pi = T(2π)
    ll = zero(T)
    for s in sets
        n_tt(s) == 0 && continue
        k = s.planet
        k in p_idx || return convert(T, -Inf)
        tp = _tt_tc_P(theta, data, k)
        tp === nothing && return convert(T, -Inf)
        Tc, P = tp
        δts, n0 = _tt_offsets(state, k)
        Δ = _tt_match(s, Tc, P, δts, n0)
        Δ === nothing && return convert(T, -Inf)
        σj2 = tt_jitter_value(theta, k)^2
        quarter = P / 4
        @inbounds for i in s.used
            r = s.tc[i] - _tt_model_time(Tc, P, s.epoch[i] + Δ, δts, n0)
            abs(r) > quarter && return convert(T, -Inf)
            v = s.tc_err[i]^2 + σj2
            ll -= (r * r / v + log(two_pi * v)) / 2
        end
    end
    return ll
end

"""
    transit_times_model(theta, data) -> Vector{NamedTuple}

Per set of `data.transit_times`, the model at `theta`: `(planet, model,
residual, sigma, delta, t0, P, ambiguous)`. `model[i]` is the model time of the
transit point i is matched to (every point, excluded ones too), `residual = tc −
model`, `sigma` the error the likelihood weights it by (jitter included),
`delta` the offset between the labels and the model's transit numbers
(transit = epoch + delta), `t0 = Tc + delta·P` the model's linear-ephemeris
time at epoch 0 of the labels, and `ambiguous` whether some point in the
likelihood is more than P/4 from its transit (the guard that makes ln L −Inf).
"""
function transit_times_model(theta::Theta{Float64}, data::Data)
    p_idx = planet_indices(theta)
    state = _tt_ttv_state(theta, data, p_idx)
    out = NamedTuple[]
    for s in data.transit_times
        k = s.planet
        n = length(s.tc)
        tp = (k in p_idx && state !== false) ? _tt_tc_P(theta, data, k) : nothing
        if tp === nothing || n_tt(s) == 0
            push!(out, (planet = k, model = fill(NaN, n), residual = fill(NaN, n),
                        sigma = copy(s.tc_err), delta = nothing, t0 = NaN, P = NaN,
                        ambiguous = true))
            continue
        end
        Tc, P = tp
        δts, n0 = _tt_offsets(state === false ? nothing : state, k)
        Δ = _tt_match(s, Tc, P, δts, n0)
        model = [_tt_model_time(Tc, P, s.epoch[i] + Δ, δts, n0) for i in 1:n]
        res = s.tc .- model
        σ = sqrt.(s.tc_err .^ 2 .+ tt_jitter_value(theta, k)^2)
        amb = any(i -> abs(res[i]) > P / 4, s.used)
        push!(out, (planet = k, model = model, residual = res, sigma = σ, delta = Δ,
                    t0 = Tc + Δ * P, P = P, ambiguous = amb))
    end
    return out
end
