# TTV-C: N-body-predicted transit timing variations via TTVFaster
# (Agol & Deck 2016, 1st order in eccentricity). Planets with the
# `:TTV_NB` source flag get their per-transit time offsets PREDICTED at
# each likelihood evaluation from mutual gravitational interactions —
# no free per-transit parameters. Needs ≥ 2 planets carrying `:TTV_NB`
# for any non-zero signal (pairwise interactions); a single TTV-NB
# planet collapses to zero δts.
#
# Plumbing reuses the TTV-A state shape (`state.δts[r]`) and the same
# `r_for_j` / `ttv_effective_time_r` machinery in transit_likelihood,
# so downstream call sites need zero awareness of TTV-A vs TTV-NB.

using TTVFaster: Planet_plane_hk, compute_ttv!
using ForwardDiff

# Conversion constant: M_Jup / M_Sun (NIST/CODATA-ish).
const _MJ_PER_MS = 9.545942339693249e-4

# The eccentricity vector in each backend's own convention. Nereus measures ω
# from the sky plane, with the transit at f + ω = π/2 (`tc_to_tp`).
#
# TTVFaster measures the longitude of periastron ϖ from the line of sight, with
# the transit at true longitude f + ϖ = 0 (its λ₀ = -n t₀ + 2 e sin ϖ is the
# mean longitude at a transit, to first order in e): ϖ = ω - π/2, and
# (e cos ϖ, e sin ϖ) = (e sin ω, -e cos ω).
#
# NbodyGradient puts the observer on -z and the transit at f + ω' = 3π/2
# (`kepler_init`): ω' = ω + π, and (e cos ω', e sin ω') = -(e cos ω, e sin ω).
#
# Both used to be handed (e cos ω, e sin ω): orbits turned by 90 and 180
# degrees, whose periastra were not those of the light curve's orbit, and on
# an eccentric pair near 2:1 the two backends' TTVs then anticorrelated
# (r = -0.4) where they agree (r = 0.99) with the right vectors.
@inline _ttvfaster_hk(e, ω) = (e * sin(ω), -e * cos(ω))
@inline _nbodygradient_hk(e, ω) = (-e * cos(ω), -e * sin(ω))

"""
    _apply_ttv_nb!(state, theta, p_idx, Ps, es, ws, Tps, bs, a_Rs, t_phot_max;
                   jmax = 5)

For each row `r` in `state` flagged as TTV-NB (i.e. `state.is_nb[r] ==
true`), overwrite `state.δts[r]` with the TTVFaster-predicted offsets
at the linear ephemeris `Tc(r) + (i-1)·P(r)` for `i = 1..n`, where
`Tc(r) = tp_to_tc(Tp, P, e, ω)` is the time of mid-transit the likelihood
numbers transits from. Sums pairwise contributions across all NB planets
(1st-order valid).

Mass ratios derive from K via `msini` divided by `sin(inc)` where
`inc = acos(b / (a/R*))`. Requires `M_s` to be set in
`theta.params.config`; if missing, leaves the NB rows at zero δts.
"""
function _apply_ttv_nb!(state, theta::Theta{T}, p_idx,
                          Ps::AbstractVector{T},
                          es::AbstractVector{T},
                          ws::AbstractVector{T},
                          Tps::AbstractVector{T},
                          bs::AbstractVector{T},
                          a_Rs::AbstractVector{T},
                          t_phot_max::T;
                          jmax::Int = 5) where {T<:Real}
    # Dispatch on user-selected TTV backend. Default is the perturbative
    # TTVFaster series (this body); :nbody routes through NbodyGradient.jl
    # for full ODE integration (see ttv_nbody_full.jl).
    if theta.params.config.ttv_backend === :nbody
        return _apply_ttv_nb_full!(state, theta, p_idx, Ps, es, ws,
                                     Tps, bs, a_Rs, t_phot_max)
    end
    M_s = theta.params.config.M_s
    isnan(M_s) && return
    n = length(state.j_active)
    nb_rows = Int[]
    @inbounds for r in 1:n
        state.is_nb[r] && push!(nb_rows, r)
    end
    n_nb = length(nb_rows)
    n_nb < 2 && return

    # Build Planet_plane_hk + linear-ephemeris time grids per NB planet.
    # Keep everything in `T` (allows ForwardDiff Duals → NUTS can backprop
    # through the N-body prediction directly into M_p/e/ω posterior).
    planets = Vector{Planet_plane_hk{T}}(undef, n_nb)
    times   = Vector{Vector{T}}(undef, n_nb)
    sums    = Vector{Vector{T}}(undef, n_nb)
    @inbounds for q in 1:n_nb
        r = nb_rows[q]
        j = state.j_active[r]
        k = state.k_planet[r]
        P  = Ps[j]; e = es[j]; w = ws[j]
        # The linear ephemeris starts at the time of mid-transit, the epoch the
        # likelihood numbers transits from (`ttv_effective_time`). It was
        # Tp + P/4, which is that only for a circular orbit with ω = 0;
        # otherwise TTVFaster put the planet at another orbital phase.
        Tc1 = tp_to_tc(Tps[j], P, e, w)
        # Grid size from data span. The conditional on ntr uses scalar
        # comparison which works for both Float64 and Dual (Dual <: Real).
        ntr_raw = max(1, Int(ceil(ForwardDiff_value(t_phot_max - Tc1) /
                                    ForwardDiff_value(P))) + 1)
        ntr = min(ntr_raw, 10_000)
        times[q] = T[Tc1 + (i - 1) * P for i in 1:ntr]
        sums[q]  = zeros(T, ntr)

        # Mass ratio = M_p / M_s via Kepler + inclination from b, a/R*.
        K = planet_K(theta, k)
        inc_deg = acosd(bs[j] / a_Rs[j])
        mp_mjup = msini(M_s, K, P, e) / sind(inc_deg)
        mp_msun = mp_mjup * _MJ_PER_MS
        mass_ratio = mp_msun / M_s
        # TTVFaster 1st-order series has a 0/0 at exactly e=0 (formula
        # divides by e at one step). Floor the eccentricity vector at
        # 1e-4 — well below any detectable TTV amplitude at this order.
        # TTVFaster's ϖ is measured from the line of sight (`_ttvfaster_hk`).
        ecosw, esinw = _ttvfaster_hk(e, w)
        if e < 1e-4
            # Bias the e-vector along ω = 0 with tiny magnitude. Signal
            # contribution is O(e), so ≲1e-4 × max_amp ≪ 1 ms.
            ecosw = oftype(e, 1e-4)
            esinw = zero(e)
        end
        planets[q] = Planet_plane_hk(mass_ratio, P, Tc1, ecosw, esinw)
    end

    # Pairwise TTVFaster. Inner planet (shorter P) must be p1.
    for a in 1:(n_nb - 1), b in (a + 1):n_nb
        inner, outer = ForwardDiff_value(planets[a].period) <
                         ForwardDiff_value(planets[b].period) ?
                         (a, b) : (b, a)
        n_in  = length(times[inner])
        n_out = length(times[outer])
        ttv_in  = zeros(T, n_in)
        ttv_out = zeros(T, n_out)
        try
            compute_ttv!(jmax, planets[inner], planets[outer],
                          times[inner], times[outer], ttv_in, ttv_out)
        catch err
            err isa DomainError || err isa ArgumentError ||
                err isa AssertionError || rethrow()
            continue
        end
        # NaN-guard the series result — TTVFaster occasionally NaNs at
        # the boundary of validity (e.g. very small e mixed with very
        # different period ratios). Treat that pair as TTV-free for
        # this draw rather than poisoning the likelihood with NaN.
        (any(isnan, ttv_in) || any(isnan, ttv_out)) && continue
        @inbounds for i in 1:n_in
            sums[inner][i] += ttv_in[i]
        end
        @inbounds for i in 1:n_out
            sums[outer][i] += ttv_out[i]
        end
    end

    @inbounds for q in 1:n_nb
        r = nb_rows[q]
        state.δts[r] = sums[q]
    end
    return
end

# Strip a ForwardDiff Dual back to its Float64 partial — used only for
# decisions that must be scalar Int (array sizes, branch selection).
@inline ForwardDiff_value(x::Real) = x
@inline ForwardDiff_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)

"""
    _has_active_ttv_nb(theta, p_idx) -> Bool

Cheap predicate mirroring `_has_active_ttv` — does any active planet
carry `:TTV_NB`? Used to decide whether the transit-likelihood ws
fast-path can be taken (it cannot — TTV-NB δts depend on every other
NB planet's orbit, so the orbit-only hash key is wrong).
"""
@inline function _has_active_ttv_nb(theta::Theta, p_idx)
    config = theta.params.config
    @inbounds for k in p_idx
        has_ttv_nb(config.planet_modes[k]) && return true
    end
    return false
end
