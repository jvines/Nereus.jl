# Full N-body TTV backend via NbodyGradient.jl (Agol+ port of TTVFast
# with autodiff). Activated when `params.config.ttv_backend === :nbody`.
# Used in place of the perturbative TTVFaster path (`ttv_nbody.jl`)
# when the user wants exact N-body integration: high masses, high
# eccentricities, or strongly-resonant interactions where the 1st-
# order TTVFaster series breaks down.
#
# Drops into the same `state.δts[r]` shape that `_apply_ttv_nb!`
# (TTVFaster) populates — predicted transit times minus the linear
# ephemeris `Tc_1 + (i-1) · P` — so downstream `transit_likelihood`
# code is backend-agnostic.

using NbodyGradient: ElementsIC, Integrator, TransitTiming, State

# Same conversion constant as ttv_nbody.jl
const _MJ_PER_MS_NBG = _MJ_PER_MS

"""
    _apply_ttv_nb_full!(state, theta, p_idx, Ps, es, ws, Tps, bs, a_Rs, t_phot_max;
                        t_phot_min = nothing)

`:nbody` backend for TTV-NB: integrates the full N-body system via
NbodyGradient.jl and writes per-transit time offsets (relative to the
linear ephemeris) into `state.δts[r]` for every `:TTV_NB` row.

Drops in for `_apply_ttv_nb!` (TTVFaster) when
`theta.params.config.ttv_backend === :nbody`. Same input/output
contract: `δts[r][i]` is transit `state.n0s[r] + i - 1`, counted from the
time of mid-transit. The elements are osculating at the start of the
integration, a quarter of the shortest period before the first transit
needed (the transit Tc, or the first one of the photometry if that is
earlier).
"""
function _apply_ttv_nb_full!(state, theta::Theta{T}, p_idx,
                              Ps::AbstractVector{T},
                              es::AbstractVector{T},
                              ws::AbstractVector{T},
                              Tps::AbstractVector{T},
                              bs::AbstractVector{T},
                              a_Rs::AbstractVector{T},
                              t_phot_max::T;
                              t_phot_min::Union{Nothing, Real} = nothing) where {T<:Real}
    M_s = theta.params.config.M_s
    isnan(M_s) && return
    n = length(state.j_active)
    nb_rows = Int[]
    @inbounds for r in 1:n
        state.is_nb[r] && push!(nb_rows, r)
    end
    n_nb = length(nb_rows)
    n_nb < 2 && return

    # NbodyGradient is typed `T <: AbstractFloat` and rejects
    # ForwardDiff Duals at the ElementsIC constructor. If a Dual θ
    # gets here it means a gradient sampler (NUTS / Pathfinder) is
    # attempting to backprop through the N-body block. Silently
    # stripping to Float64 would produce **zero** gradient
    # contribution from the N-body component — the sampler would
    # think the N-body-coupled params have no likelihood effect,
    # which is a silent bug, not a graceful degradation.
    #
    # Until proper NbodyGradient.jl Jacobian-passthrough is wired up
    # (chain-rule `tt.dtdelements` back into the Dual partials), throw
    # a clear ArgumentError so the user knows to either:
    #   1. Use a gradient-free sampler (pt_emcee / nested / MoMS).
    #   2. Switch to `ttv_backend=:ttvfaster` (analytic, autodiff-clean).
    if T <: ForwardDiff.Dual
        throw(ArgumentError(
            "ttv_backend=:nbody is incompatible with gradient samplers " *
            "(NUTS, Pathfinder, pathfinder-seeded sample_pt). NbodyGradient.jl is " *
            "typed `T <: AbstractFloat` and does not propagate Dual " *
            "partials, so the N-body block would silently contribute " *
            "zero gradient. Either use a gradient-free sampler " *
            "(sample_pt_emcee / sample_nested / sample_daedalus / " *
            "sample_pa) or switch to `ttv_backend=:ttvfaster` for the " *
            "analytic perturbative N-body model (gradient-clean)."))
    end
    M_s_f = Float64(ForwardDiff_value(M_s))
    t_max_f = Float64(ForwardDiff_value(t_phot_max))

    # Build per-planet element rows + per-planet (Tc1, P) for the
    # linear ephemeris subtraction.
    elems = zeros(Float64, n_nb + 1, 7)
    elems[1, 1] = M_s_f
    Tc1s = zeros(Float64, n_nb)
    Ps_f = zeros(Float64, n_nb)
    n0s  = zeros(Int, n_nb)
    t_min_f = t_phot_min === nothing ? nothing : Float64(ForwardDiff_value(t_phot_min))
    for q in 1:n_nb
        r = nb_rows[q]
        j = state.j_active[r]
        k = state.k_planet[r]
        P  = Float64(ForwardDiff_value(Ps[j]))
        e  = Float64(ForwardDiff_value(es[j]))
        w  = Float64(ForwardDiff_value(ws[j]))
        Tp = Float64(ForwardDiff_value(Tps[j]))
        # The time of mid-transit (it was Tp + P/4, right only for a circular
        # orbit with ω = 0; see `_apply_ttv_nb!`).
        Tc1 = tp_to_tc(Tp, P, e, w)
        # Mass ratio from K, inc.
        K = Float64(ForwardDiff_value(planet_K(theta, k)))
        b = Float64(ForwardDiff_value(bs[j]))
        aR = Float64(ForwardDiff_value(a_Rs[j]))
        inc_rad = acos(b / aR)
        mp_mjup = msini(M_s_f, K, P, e) / sin(inc_rad)
        mp_msun = mp_mjup * _MJ_PER_MS_NBG

        elems[q+1, 1] = mp_msun
        elems[q+1, 2] = P
        elems[q+1, 3] = Tc1
        # NbodyGradient's ω is Nereus's + π (`_nbodygradient_hk`).
        elems[q+1, 4], elems[q+1, 5] = _nbodygradient_hk(e, w)
        elems[q+1, 6] = inc_rad
        elems[q+1, 7] = 0.0   # Ω = 0 — coplanar assumption
        Tc1s[q] = Tc1
        Ps_f[q] = P
        # The first transit to predict: Tc, or the one nearest the first
        # cadence when the photometry starts earlier (see `_apply_ttv_nb!`).
        n0s[q] = t_min_f === nothing ? 0 : min(0, floor(Int, (t_min_f - Tc1) / P))
    end

    t_start = minimum(Tc1s[q] + n0s[q] * Ps_f[q] for q in 1:n_nb) - minimum(Ps_f) / 4
    # On to the transit nearest the last cadence, which may come after it.
    duration = t_max_f + maximum(Ps_f) / 2 - t_start
    duration > 0 || return

    # Choose integration step: 1/40 of the shortest period, clamped to
    # a reasonable lower bound. Smaller step → more accurate, slower.
    h_step = max(min(minimum(Ps_f) / 40, 1.0), 1e-3)

    local ic, tt
    try
        ic = ElementsIC(t_start, n_nb + 1, elems)
        # NbodyGradient uses DURATION (relative to ic.t0) for both the
        # Integrator's tmax and the TransitTiming preallocation — not
        # absolute time.
        intr = Integrator(h_step, duration)
        tt = TransitTiming(duration, ic)
        s = State(ic)
        intr(s, tt)
    catch err
        @warn "NbodyGradient integration failed; leaving δts at zero" exception = err
        return
    end

    # Convert raw transit times to (δt relative to linear ephemeris).
    # Body index in tt is 1-based with star = row 1; planet q = row q+1.
    for q in 1:n_nb
        r = nb_rows[q]
        n_tr = min(Int(tt.count[q + 1]), size(tt.tt, 2))
        n_tr == 0 && continue
        raw = tt.tt[q + 1, 1:n_tr]
        # Each time goes to the slot of the transit it is, the nearest
        # Tc1 + n·P, not to its rank among the transits found: the first one
        # after t_start need not be transit n0 (a planet whose Tc is more
        # than a period after the earliest one's has transits before it).
        n0 = n0s[q]
        ns = [round(Int, (raw[i] - Tc1s[q]) / Ps_f[q]) for i in 1:n_tr]
        δts = zeros(T, max(0, maximum(ns) - n0 + 1))
        @inbounds for i in 1:n_tr
            ns[i] < n0 && continue
            δts[ns[i] - n0 + 1] = convert(T, raw[i] - (Tc1s[q] + ns[i] * Ps_f[q]))
        end
        state.δts[r] = δts
        state.n0s[r] = n0
    end
    return
end
