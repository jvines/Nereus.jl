# Transit (photometric) log-likelihood.
#
# Architecture mirrors rv_log_likelihood:
#   - Planets decoded ONCE
#   - Per-observation: compute z(t), evaluate transit flux, Gaussian LL
#   - Geometry gate: only planets with b < 1 + rr contribute
#
# Limb darkening: per-instrument Kipping q1, q2 parameters.

# Fixed partition size for the photometry log-likelihood reduction. A CONSTANT,
# deliberately: making it a function of nthreads() would make the summation
# order -- and therefore the last bits of the result -- machine-dependent, so a
# run on 4 threads and a run on 8 would disagree at exactly the level that
# flips trans-dimensional accept/reject decisions.
const _PHOT_REDUCE_CHUNK = 4096

# Per-task chunk of the photometry-Gaussian log-likelihood used by the
# chunked-@spawn path in the ws-aware transit_log_likelihood. Avoids
# the per-iteration Threads.threadid() lookup and thread-sync overhead
# that the old @threads pattern incurred (~19% of CPU time per profile).
# Each task accumulates its chunk's contribution to a local Float64
# and returns it; the caller sums fetched results.
#
# Per-task chunk for SPARSE refresh: iterates the supplied in-transit
# index list `idx[a:b]` and computes flux_cache[j, idx[k]] for each k.
# Used by the chunked-@spawn refresh path when the in-transit set is
# large enough (>2000) to benefit from threading.
@inline function _phot_sparse_refresh(a::Int, b::Int, j::Int,
                                       data::Data,
                                       idx::AbstractVector{Int},
                                       P::Float64, e::Float64, ω::Float64,
                                       Tp::Float64, b_imp::Float64,
                                       a_Rs::Float64, rr::Float64,
                                       lds::AbstractVector,
                                       flux_cache::AbstractMatrix{Float64})
    a > b && return nothing
    # Orbit constants once per chunk; the true anomaly takes e unclamped here.
    o = _sky_orbit(P, e, ω, Tp, b_imp, a_Rs, e)
    @inbounds for k in a:b
        i = idx[k]
        z, sin_wf = _sky_separation_signed(o, data.t_phot[i])
        ld_i = lds[data.phot_inst[i]]
        # Behind the star (sin(ω+f) <= 0) it is an occultation: no dip.
        flux_cache[j, i] = sin_wf > 0 ? transit_flux(ld_i, z, rr) : 1.0
    end
    return nothing
end

# Per-task chunk for refreshing one row of `flux_cache` for planet `j`.
# Several layers of optimization vs the obvious sky_separation+transit_flux loop:
#   1. Orbit constants (e_factor, cos_i, sin_i_sq, 1-e², sin/cos(ω),
#      2π/P) computed ONCE per chunk — invariant across the 13k-point loop.
#   2. Direct (cos f, sin f) from (cos E, sin E) via half-angle identity:
#        cos f = (cos E − e) / (1 − e cos E)
#        sin f = √(1−e²) sin E / (1 − e cos E)
#      This skips the atan in true_anomaly() AND the second sincos(f) we
#      had after — saves ~60-90 ns per phot point (one transcendental
#      call instead of three).
#   3. Reuses (sin E, cos E) from the converged Kepler iteration —
#      they're already computed inside kepler_solve's last iteration but
#      not exposed; we recompute once at the end.
@inline function _phot_refresh_chunk(i_lo::Int, i_hi::Int, j::Int,
                                       data::Data,
                                       P::Float64, e::Float64, ω::Float64,
                                       Tp::Float64, b_imp::Float64,
                                       a_Rs::Float64, rr::Float64,
                                       lds::AbstractVector,
                                       flux_cache::AbstractMatrix{Float64})
    i_lo > i_hi && return nothing
    # Orbit-constant precomputes
    one_minus_e2  = 1 - e * e
    sqrt_1_me2    = sqrt(one_minus_e2)
    sinω, cosω    = sincos(ω)
    e_factor      = (1 + e * sinω) / one_minus_e2
    cos_i         = b_imp * e_factor / a_Rs
    sin_i_sq      = 1 - cos_i * cos_i
    two_pi_over_P = 2π / P
    @inbounds for i in i_lo:i_hi
        t = data.t_phot[i]
        M = two_pi_over_P * (t - Tp)
        E = kepler_solve(M, e)
        sinE, cosE = sincos(E)
        # cos f, sin f without atan / second sincos — half-angle identity
        denom = 1 - e * cosE
        cosf  = (cosE - e) / denom
        sinf  = sqrt_1_me2 * sinE / denom
        r_over_a = one_minus_e2 / (1 + e * cosf)
        sin_wf   = sinω * cosf + cosω * sinf
        z = a_Rs * r_over_a *
            sqrt(max(1 - sin_i_sq * sin_wf * sin_wf, 0.0))
        ld_i = lds[data.phot_inst[i]]
        # Behind the star (sin(ω+f) <= 0) it is an occultation: no dip.
        flux_cache[j, i] = sin_wf > 0 ? transit_flux(ld_i, z, rr) : 1.0
    end
    return nothing
end

@inline function _phot_chunk_loglik(i_lo::Int, i_hi::Int, data::Data,
                                      n_transit::Int,
                                      transits::AbstractVector{Bool},
                                      flux_cache::AbstractMatrix{Float64},
                                      offsets::AbstractVector{Float64},
                                      jitters::AbstractVector{Float64},
                                      dilutions::AbstractVector{Float64},
                                      t_ref::Float64, inv_th::Float64,
                                      trends::Vector{Vector{Float64}},
                                      two_pi::Float64)
    s = 0.0
    i_lo > i_hi && return s
    @inbounds for i in i_lo:i_hi
        obs     = data.flux[i]
        obs_err = data.flux_err[i]
        ins_idx = data.phot_inst[i]
        jitter  = jitters[ins_idx]
        dil     = dilutions[ins_idx]

        tprod = 1.0
        for j in 1:n_transit
            transits[j] || continue
            tprod *= flux_cache[j, i]
        end
        # continuum (1+offset+trend) × dilution (D=0 ⇒ identity (1−D)·transit + D)
        cont = _phot_continuum(data.t_phot[i], t_ref, inv_th,
                               offsets[ins_idx], trends[ins_idx])
        model_flux = cont * ((1.0 - dil) * tprod + dil)

        var_i = obs_err * obs_err + jitter * jitter
        resid = obs - model_flux
        s += -(log(two_pi * var_i) + resid * resid / var_i) / 2
    end
    return s
end

# Supersampling factor for finite-exposure integration (Kipping 2010). Targets a
# ≈1-min sub-cadence; returns 1 (instantaneous — no supersampling, fast/cached
# paths used unchanged) when exposures are absent or short (≤3 min, e.g. TESS
# 2-min / Kepler short-cadence). Capped at 30 to bound cost on long cadence.
function _phot_n_super(data::Data)
    isempty(data.exposure_times) && return 1
    mx = 0.0                                # max FINITE positive exposure (days)
    @inbounds for e in data.exposure_times
        (isfinite(e) && e > mx) && (mx = e)
    end
    return _phot_n_super_point(mx)
end

# Quantities that depend on the photometry alone, cached per workspace and filled
# on first use. A workspace is already bound to one dataset (its transit flux
# cache is indexed by cadence), so nothing here needs a data key. Not part of a
# checkpoint (`_WS_NOT_SAVED`): it is rebuilt from the data on the first call
# after a resume, and the data itself is in the run fingerprint.
mutable struct PhotDataCache
    n_super::Int        # _phot_n_super(data); 0 = not computed yet (it is >= 1)
    # Time order of the cadences, for `_phot_window_indices!`. `t_phot` is not
    # sorted across instruments. Built on first use (`order_built`).
    order_built::Bool
    sortable::Bool              # every t_phot finite
    perm::Vector{Int}           # t_phot[perm] == t_sorted
    t_sorted::Vector{Float64}
end
PhotDataCache() = PhotDataCache(0, false, false, Int[], Float64[])

function _phot_time_order!(data::Data, cache::PhotDataCache)
    if !cache.order_built
        t = data.t_phot
        cache.sortable = all(isfinite, t)
        if cache.sortable
            cache.perm = sortperm(t)
            cache.t_sorted = t[cache.perm]
        end
        cache.order_built = true
    end
    return cache
end

# Append to `idx` every cadence i with |Δt| <= hw, where Δt = t_i - Tc folded
# by P: the same set, from the same expression, as testing all n_obs cadences,
# but only the cadences within hw + δ of a transit centre are tested, found by
# binary search in the time order. δ is far above the rounding of the folded Δt
# and of Tc + mP, so no cadence the test accepts is missed. Indices come in time
# order, not index order; each one's flux is computed on its own, so the order
# changes nothing. Falls back to the full scan when the window is not narrow
# (hw >= P/4, NaN), or an input or a cadence time is not finite.
function _phot_window_indices!(idx::Vector{Int}, data::Data, cache::PhotDataCache,
                               Tc::Float64, P::Float64, hw::Float64)
    t_phot = data.t_phot
    n = length(t_phot)
    n == 0 && return idx
    _phot_time_order!(data, cache)
    ts = cache.t_sorted
    δ = cache.sortable ? 1e-6 + 1e-12 * (abs(ts[1]) + abs(ts[end]) + abs(Tc) + abs(P)) : 0.0
    w = max(hw, 0.0) + δ
    if !(cache.sortable && isfinite(Tc) && P > 0 && isfinite(P) && w < P / 4)
        @inbounds for i in 1:n
            Δt = t_phot[i] - Tc
            Δt -= P * round(Δt / P)
            abs(Δt) <= hw && push!(idx, i)
        end
        return idx
    end
    perm = cache.perm
    fwd = Base.Order.Forward
    r = 1
    @inbounds while r <= n
        c = Tc + round((ts[r] - Tc) / P) * P     # nearest transit centre
        if ts[r] < c - w                          # before its window: jump in
            r = searchsortedfirst(ts, c - w, r, n, fwd)
        elseif ts[r] > c + w                      # after it: jump to the next
            r = searchsortedfirst(ts, c + P - w, r, n, fwd)
        else
            hi = c + w
            while r <= n && ts[r] <= hi
                i = perm[r]
                Δt = t_phot[i] - Tc
                Δt -= P * round(Δt / P)
                abs(Δt) <= hw && push!(idx, i)
                r += 1
            end
        end
    end
    return idx
end

# `_phot_n_super(data)` reads every exposure time, about 20 us on 20k cadences,
# and the workspace likelihood asked for it on every call.
@inline function _phot_n_super(data::Data, cache::PhotDataCache)
    n = cache.n_super
    if n == 0
        n = _phot_n_super(data)
        cache.n_super = n
    end
    return n
end

# =====================================================================
# Transit window: which cadences get a flux evaluation
# =====================================================================
#
# A cadence's flux is computed only when it lies within ±T_dur_safe of the
# transit centre Tc, folded by P; everywhere else it is taken as exactly 1.
# Transits.compute returns exactly one(T) when z >= 1 + k, and a cadence with
# the planet behind the star has flux 1, so a window that holds every cadence
# with the planet in front and z < 1 + k changes no bits.
#
# The half-width used to be 2P(1+k)/(π a_R), about four circular
# half-durations. That is not an upper bound: at high e it cut real transits
# short (P = 4.137 d, k = 0.031, a_R = 12.7, b = 0.3, ω = -π/2 lost 4.4 % of
# the in-transit cadences at e = 0.90 and 21 % at e = 0.93; at a_R = 3 the
# loss starts near e = 0.8). The window is now `_transit_window_halfwidth`, a
# bound from the orbit itself. With u = ω + f,
#
#     z² = (r/R*)² (1 - sin²i sin²u) = (r/R*)² (cos²i + sin²i cos²u)
#
# and r >= r_min, so z < 1 + k needs
#
#     |cos u| < sqrt((s² - cos²i) / sin²i),     s = (1 + k) / r_min,
#
# i.e. f within an angle Δf of the transit conjunction f_c = π/2 - ω (only there
# is the planet in front; behind the star the flux is 1). r_min starts at
# a_R(1-e); for e > 0 it is then raised to the smallest r inside the f-window
# just found (at its end nearer periastron, or a_R(1-e) when periastron is
# inside), and Δf recomputed, three times. The transit lies inside each window,
# and r there is at least that window's r_min, so every step keeps every
# in-contact cadence. For a circular orbit the result is the transit itself,
# T14/2 plus the margins below. The bound used to drop the cos i term and keep
# r_min = a_R(1-e), which at b = 0.75 made the window 1.35 x T14/2 for a
# circular orbit, and wider at high e.
#
# The half-width is the longer of the two mean-anomaly spans f_c -> f_c ± Δf,
# through Kepler's equation, in time. The spans are taken on an unwrapped M(f)
# (`_mean_anomaly_unwrapped`): the abs(rem2pi(span)) used before folded a span
# above π back below it, and a span of π or more means the window is the whole
# orbit, so the bound is Inf. cos i is built from b as the z kernels build it:
# `planet_sky_position` and `sky_separation`, and `_sky_orbit` for the
# likelihood, all satisfy the identity above.
#
# Margins. s is inflated and cos i shrunk by 1e-12 relative (a larger s and a
# smaller cos i only widen the window), far above the relative rounding of r and
# cos i in the kernels, also in a nearly face-on orbit, where 1/sin²i amplifies
# it. δ = 32 eps max(1, a_R/r_min) is added to s² - cos²i for the cancellation
# in the ordinary kernels' 1 - sin²i sin²u near conjunction, an absolute error
# measured at up to 15 eps for r >= a and 15 eps a/r below (sin(ω+f) from the
# half-angle identity divides by 1 - e cos E = r/a). At a grazing b, where the
# window shrinks towards nothing, δ is the larger margin once cos²i is small
# (a_R above ~100); without it a few cadences with z a few ulp below 1 + k were
# skipped. In mean anomaly, ×(1 + 1e-9) and +1e-8 rad, which covers the Kepler
# solver's 1e-10 tolerance and the rounding of M = 2π(t - Tp)/P for |t - Tp| up
# to ~10⁶ periods; plus 8 ulp of the epoch for the rounding of Tc itself.
#
# Inf (no bound: every cadence is evaluated) when s >= 1 at r_min = a_R(1-e)
# (periastron within 1 + k stellar radii), when e is outside [0, 0.9999) (the
# range the bound is tested over; it used to be where `true_anomaly` clamped e,
# and above it periastron lies within 1 + k of the star anyway unless
# a_R > 10⁴), or when an input is not a finite positive number. A b that is not finite is
# taken as 0, the bound without the cos i term.

# Plain Float64 of a likelihood input (strips ForwardDiff duals). A type it
# cannot strip gives NaN, and with it no bound: every cadence is evaluated.
@inline _gate_float(x::Float64) = x
@inline _gate_float(x::ForwardDiff.Dual) = _gate_float(ForwardDiff.value(x))
@inline _gate_float(x::Real) = x isa AbstractFloat ? Float64(x) : NaN

# M(f), continuous and increasing in f, with no branch cut: E/2 lies in f/2's
# quadrant and |E - f| < π, so E is put on f's turn. q = sqrt((1-e)/(1+e)).
@inline function _mean_anomaly_unwrapped(f::Float64, e::Float64, q::Float64)
    sh, ch = sincos(f / 2)
    E = 2 * atan(q * sh, ch)
    E += 2π * round((f - E) / (2π))
    return E - e * sin(E)
end

function _transit_window_halfwidth(P::Float64, e::Float64, ω::Float64, k::Float64,
                                   aR::Float64, tscale::Float64, b::Float64 = 0.0)
    (0.0 <= e < 0.9999) || return Inf
    (P > 0 && aR > 0 && k >= 0 && isfinite(P) && isfinite(aR) && isfinite(k) &&
     isfinite(ω) && isfinite(tscale)) || return Inf
    rmin = aR * (1 - e)
    (1 + k) / rmin < 1 || return Inf
    # cos i as the kernels build it, shrunk; without the term if it is not below 1
    # (b >= 1 + k then, and the planet does not transit)
    ci = isfinite(b) ? abs(b) * (1 + e * sin(ω)) / (aR * (1 - e * e)) * (1 - 1e-12) : 0.0
    ci2 = ci * ci < 1 ? ci * ci : 0.0
    fc = π / 2 - ω
    Δf = π / 2
    for pass in 1:4
        s = (1 + k) * (1 + 1e-12) / rmin
        δ = 32 * eps() * max(1.0, aR / rmin)
        Δf = min(Δf, asin(sqrt(clamp((s * s - ci2 + δ) / (1 - ci2), 0.0, 1.0))))
        (pass == 4 || e == 0) && break
        abs(rem2pi(fc, RoundNearest)) <= Δf && break        # periastron inside
        rmin = aR * (1 - e * e) / (1 + e * max(cos(fc - Δf), cos(fc + Δf)))
    end
    q  = sqrt((1 - e) / (1 + e))
    Mc = _mean_anomaly_unwrapped(fc, e, q)
    d1 = _mean_anomaly_unwrapped(fc + Δf, e, q) - Mc
    d2 = Mc - _mean_anomaly_unwrapped(fc - Δf, e, q)
    hw_M = max(d1, d2) * (1 + 1e-9) + 1e-8
    (d1 >= 0 && d2 >= 0 && hw_M < π) || return Inf
    return hw_M * P / (2π) + 8 * eps(tscale)
end

# Every transiting planet's window (non-workspace paths): the bound, with its cos i
# term from `bs` (the looser b = 0 bound without it), plus `pad`, plus, for a TTV
# planet, the largest |δt| of its transits, since its flux is evaluated at
# t - δt. The likelihood and prediction paths pass pad = 0: how far a supersampled
# cadence's sub-samples reach, half its own exposure, is added per cadence in
# `_phot_transit_product`, so a 2-min cadence is no longer widened by the longest
# exposure in the fit. A NaN anywhere gives a NaN window, which skips no cadence.
function _set_transit_windows!(T_dur_safe, n_transit::Int, transits, r_for_j, ttv_state,
                               Ps, es, ws, rrs, a_Rs, Tc_centers, Tps, pad::Float64,
                               bs = nothing)
    @inbounds for j in 1:n_transit
        transits[j] || continue
        tscale = max(abs(_gate_float(Tc_centers[j])), abs(_gate_float(Tps[j])))
        bj = bs === nothing ? 0.0 : _gate_float(bs[j])
        hw = _transit_window_halfwidth(_gate_float(Ps[j]), _gate_float(es[j]),
                                       _gate_float(ws[j]), _gate_float(rrs[j]),
                                       _gate_float(a_Rs[j]), tscale, bj) + pad
        if r_for_j[j] > 0
            shift = 0.0
            for δ in ttv_state.δts[r_for_j[j]]
                shift = max(shift, abs(_gate_float(δ)))
            end
            hw += shift
        end
        T_dur_safe[j] = hw
    end
    return T_dur_safe
end

# Supersampling factor for ONE cadence, from that cadence's own exposure (days).
# Same rule as above: ≈1-min sub-cadence, none at or below 3 min, capped at 30.
# `_phot_n_super(data)` is the dataset maximum and only says whether any cadence
# needs integrating; the per-point count must come from here. Applying the maximum
# to every point integrated a 2-min or 5-s cadence as finely as the longest
# exposure in the fit (10 sub-samples whenever a 10-min band was present), which
# costs time and changes nothing for the short cadences.
#
# The limit was 2 min, which integrated a 121.5-s cadence over 3 sub-samples and
# a 120-s one not at all. It is 3 min: cadences up to that are evaluated at their
# mid-time, and longer ones keep the counts they had (200 s: 4, 600 s: 10,
# 1800 s: 30).
@inline function _phot_n_super_point(texp::Real)
    m = texp * 1440.0                       # days → minutes
    (isfinite(m) && m > 3.0) || return 1
    return clamp(ceil(Int, m), 1, 30)
end

# Per-cadence transit product Π_j fluxⱼ(t) with optional finite-exposure
# supersampling. With n_super>1 each transit's flux is the mean of the model over
# `n_super` sub-samples at the midpoints of equal sub-intervals spanning
# [t−texp/2, t+texp/2] (Kipping 2010 numerical resampling). With n_super==1 it is
# a single instantaneous evaluation — IDENTICAL to the pre-existing inline loops.
# Returns the PURE transit signal (OOT = 1); the caller applies dilution + offset.
# `gd` is `nothing` for the standard model (every existing call path), or a
# NamedTuple (i_star, λs, ω_frac, on) enabling the gravity-darkened
# transit for the planets flagged in `on`; `gd_ctxs` is then the per-call context
# matrix from `_gd_contexts` and `ins_idx` the cadence's instrument column.
#
# All three are POSITIONAL, defaulting to `nothing`, so the ordinary model is
# bit-for-bit unchanged. They used to be keywords, with the caller passing
# `view(gd_ctxs, :, ins_idx)`: per cadence that built a SubArray and a keyword
# NamedTuple and left this function unspecialised, so the limb-darkening struct
# and the returned flux were boxed too. About 400 bytes per cadence, 7.5 MB per
# likelihood call on a 18 000-cadence fit, and the garbage collector, which stops
# every thread, then took 29 per cent of wall time at 8 threads and most of it at
# 90. Indexing the matrix in place allocates nothing.

# One brightness context per (transit planet, photometric instrument), built once
# per likelihood evaluation. `β` is per band, so the context is too. Returns
# `nothing` when gravity darkening is off, which keeps the ordinary model on a
# path with no extra work at all.
function _gd_contexts(theta, gd, n_transit::Int, n_pm::Int)
    gd === nothing && return nothing
    # The disc average depends on i★, ω and β, not on λ, and its β-free part (log g
    # at the quadrature nodes) on ω alone: that is computed once here, and the
    # average once per distinct β, shared by every planet's context in the band.
    ω = _gd_omega(gd.ω_frac)
    lg = ω < 1e-6 ? nothing : _gd_node_logs(ω)        # nothing: no rotation, flat map
    β1 = system_gd_beta(theta, 1)
    mean1 = lg === nothing ? nothing : _gd_disc_mean_nodes(lg, gd.i_star, β1)
    ctx1 = _gd_context(gd.i_star, gd.λs[1], ω, β1, mean1)
    out = Matrix{typeof(ctx1)}(undef, n_transit, n_pm)
    @inbounds for ix in 1:n_pm
        β = system_gd_beta(theta, ix)
        # Bands that share β (several cadences of one survey, say) share the
        # contexts.
        prev = 0
        for jx in 1:(ix - 1)
            system_gd_beta(theta, jx) == β && (prev = jx; break)
        end
        mean = (prev > 0 || ix == 1 || lg === nothing) ? mean1 :
               _gd_disc_mean_nodes(lg, gd.i_star, β)
        for j in 1:n_transit
            out[j, ix] = !gd.on[j] ? ctx1 :
                prev > 0 ? out[j, prev] :
                (j == 1 && ix == 1) ? ctx1 :
                _gd_context(gd.i_star, gd.λs[j], ω, β, mean)
        end
    end
    return out
end

@inline function _phot_transit_product(t::Float64, texp::Float64, ld,
        n_transit::Int, transits, Ps::AbstractVector{T}, es, ws, Tps, bs, a_Rs,
        rrs, Tc_centers, T_dur_safe, r_for_j, ttv_state, n_super::Int,
        gd = nothing, gd_ctxs = nothing, ins_idx::Int = 0, sky = nothing) where {T}
    tprod = one(T)
    # `n_super` (dataset maximum) only switches integration on; the number of
    # sub-samples is this cadence's own, so short cadences stay instantaneous.
    ns = n_super > 1 ? _phot_n_super_point(texp) : 1
    integ = ns > 1 && texp > 0
    # The window T_dur_safe holds the transit; a supersampled cadence reaches
    # half its own exposure beyond it, and only that far.
    reach = integ ? texp / 2 : 0.0
    @inbounds for j in 1:n_transit
        transits[j] || continue
        Δt = t - Tc_centers[j]
        Δt -= Ps[j] * round(Δt / Ps[j])
        abs(Δt) > T_dur_safe[j] + reach && continue
        # A TTV planet's transits are numbered from Tc (the transit nearest
        # the cadence, as the window above folds it), not from Tp.
        t_eff = r_for_j[j] > 0 ?
            ttv_effective_time_r(t, r_for_j[j], ttv_state, Ps, Tc_centers) : t
        use_gd = gd !== nothing && gd.on[j]
        # The callers pass this call's orbit constants (`_sky_orbits`); built
        # here when they do not.
        orb = sky === nothing ? _sky_orbit(Ps[j], es[j], ws[j], Tps[j], bs[j], a_Rs[j]) :
                                sky[j]
        if integ
            fsum = zero(T)
            for s in 1:ns
                off = ((2s - ns - 1) / (2 * ns)) * texp
                # A sub-sample outside the window has flux exactly 1: add it
                # without computing it. The cadence's own test, at the
                # sub-sample's time (TTV shift included in T_dur_safe), so the
                # sum is the one computing every sub-sample gives.
                d = Δt + off
                d -= Ps[j] * round(d / Ps[j])
                if abs(d) > T_dur_safe[j]
                    fsum += one(T)
                    continue
                end
                tsub = t_eff + off
                if use_gd
                    x, y, zl = _sky_position(orb, tsub)
                    fsum += transit_flux_gd(ld, gd_ctxs[j, ins_idx], x, y, zl, rrs[j])
                else
                    zs, sw = _sky_separation_signed(orb, tsub)
                    # Behind the star (sin(ω+f) <= 0): an occultation, no dip.
                    fsum += sw > 0 ? transit_flux(ld, zs, rrs[j]) : one(T)
                end
            end
            tprod *= fsum / ns
        elseif use_gd
            x, y, zl = _sky_position(orb, t_eff)
            tprod *= transit_flux_gd(ld, gd_ctxs[j, ins_idx], x, y, zl, rrs[j])
        else
            z, sw = _sky_separation_signed(orb, t_eff)
            tprod *= sw > 0 ? transit_flux(ld, z, rrs[j]) : one(T)
        end
    end
    return tprod
end

# Every transit planet's orbit constants for this call (`_sky_orbit`, transit.jl).
_sky_orbits(n_transit::Int, Ps, es, ws, Tps, bs, a_Rs) =
    [_sky_orbit(Ps[j], es[j], ws[j], Tps[j], bs[j], a_Rs[j]) for j in 1:n_transit]

# Per-instrument flux dilution / third light: observed = (1−D)·transit + D, with
# D = contaminant flux fraction ∈ [0,1). D=0 (the default FixedPrior) ⇒ identity.
@inline _apply_dilution(tprod, D) = (one(D) - D) * tprod + D

# Per-instrument photometric baseline continuum: 1 + offset + Σ_p c_p·x^p, with
# x = (t − t_ref)·inv_th the time normalized to the half-baseline span (keeps the
# coefficients O(1) and span-independent). `trend` is the coefficient vector;
# EMPTY ⇒ identity continuum 1 + offset (the phot_trend_order==0 default, so the
# common no-trend path is unchanged). Horner-evaluated.
@inline function _phot_continuum(t::Float64, t_ref::Float64, inv_th::Float64,
                                  offset::T, trend::AbstractVector{T}) where {T}
    isempty(trend) && return one(T) + offset
    x = (t - t_ref) * inv_th
    c = zero(T)
    @inbounds for p in length(trend):-1:1
        c = c * x + trend[p]
    end
    return one(T) + offset + c * x
end

# 1 / (half the photometric baseline span), in 1/day. 0 when no phot data. Only
# consulted when a baseline trend is active.
function _phot_inv_t_half(data::Data)
    isempty(data.t_phot) && return 0.0
    span = maximum(data.t_phot) - minimum(data.t_phot)
    span > 0 ? 2.0 / span : 0.0
end

# Gather a Float64/Dual vector of trend coefficients per PM instrument (empty when
# phot_trend_order==0). Mirrors the offsets/jitters/dilutions per-call caches.
@inline function _phot_trend_cache(theta::Theta{T}, n_pm::Int) where {T}
    out = Vector{Vector{T}}(undef, n_pm)
    sys = theta.params.layout.systemic
    @inbounds for ix in 1:n_pm
        slots = sys.pm_trend[ix]
        out[ix] = T[theta.values[s] for s in slots]
    end
    return out
end

"""
    transit_log_likelihood(theta::Theta{T}, data::Data) -> T

Gaussian log-likelihood of photometric (transit) observations.
Only planets with transit geometry (PMOnlyBlock or RVPMBlock) and
passing the geometry gate (b < 1 + Rp/Rs) contribute.
"""
transit_log_likelihood(theta::Theta, data::Data) =
    _transit_ll_direct(theta, data, _phot_n_super(data))

# The body of `transit_log_likelihood(theta, data)`, with the dataset's
# supersampling factor passed in so the workspace method, which caches it, can
# fall back here without scanning every exposure time again.
function _transit_ll_direct(theta::Theta{T}, data::Data, n_super_data::Int) where {T}
    n_obs = length(data.t_phot)
    n_obs > 0 || return zero(T)

    parametrization = theta.params.config.parametrization
    t_ref = data.t_ref
    two_pi = T(2π)

    # --- Decode transit planets (geometry gate) ----------------------
    # Only planets with b and r fields that pass b < 1 + rr transit.
    p_idx = planet_indices(theta)
    # One assignment, not a running count: a variable assigned more than once
    # and captured by the threaded loops below is put in a Core.Box, and every
    # read of it in the loop body is then untyped (see `gd_s` below).
    n_transit = count(k -> has_geometry(theta.params.layout.planet_blocks[k]), p_idx)

    if n_transit == 0
        return _phot_ll_no_transit(theta, data)
    end

    Ps   = Vector{T}(undef, n_transit)
    es   = Vector{T}(undef, n_transit)
    ws   = Vector{T}(undef, n_transit)
    Tps  = Vector{T}(undef, n_transit)
    rrs  = Vector{T}(undef, n_transit)
    bs   = Vector{T}(undef, n_transit)
    a_Rs = Vector{T}(undef, n_transit)
    transits = Vector{Bool}(undef, n_transit)  # geometry gate result
    # Phase-distance early-out cache (skips kepler_solve for points
    # outside the transit window). Tc_centers[j] = the transit centre
    # (tp_to_tc); T_dur_safe[j] = the window half-width from
    # `_set_transit_windows!`, a bound from the orbit that holds every
    # cadence with the planet in front of the star and z < 1 + rr. A point
    # with mod(|t - Tc|, P) > T_dur_safe (plus half its exposure when it is
    # supersampled) has flux exactly 1 without solving Kepler, and so does a
    # sub-sample that far out. Cadences inside it with the planet behind the
    # star are occultations, given no dip by the sign of sin(ω + f).
    Tc_centers = Vector{T}(undef, n_transit)
    T_dur_safe = Vector{T}(undef, n_transit)

    # Gravity darkening: which transit rows use it, and each row's λ. Built in
    # the same j-loop as the geometry so the indexing cannot drift apart.
    any_gd  = any(has_gd(m) for m in theta.params.config.planet_modes)
    gd_on   = any_gd ? fill(false, n_transit) : Bool[]
    gd_lams = any_gd ? zeros(T, n_transit)    : T[]

    # Get a/R* from rho_s or per-planet
    use_rho = parametrization.use_rho_s
    rho_val = use_rho ? rho_s(theta) : zero(T)

    j = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_geometry(block) || continue
        j += 1
        Ps[j] = planet_P(theta, k)
        e, w = planet_e_w(theta, k)
        (e < 0 || e >= 1) && return convert(T, -Inf)
        es[j] = e
        ws[j] = w

        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tps[j] = t_ref - ta * Ps[j] / two_pi
        elseif parametrization.time === :Tp
            Tps[j] = ta
        else  # :Tc
            Tps[j] = tc_to_tp(ta, Ps[j], e, w)
        end

        b_raw, rr_raw = planet_b_rr(theta, k)
        # Decode r1r2 if needed
        if parametrization.geom === :r1r2
            # Espinoza 2018: r1 = (rr + b), r2 = (rr - b) / (rr + b)
            # => rr = r1 * (1 + r2) / 2, b = r1 * (1 - r2) / 2
            rrs[j] = b_raw * (1 + rr_raw) / 2
            bs[j]  = b_raw * (1 - rr_raw) / 2
        else  # :b_rr
            bs[j]  = b_raw
            rrs[j] = rr_raw
        end

        # a/R* from rho_s or from M_s + R_s via Kepler's third law
        if (ai = a_Rs_slot(theta, k)) > 0
            a_Rs[j] = theta.values[ai]
        elseif use_rho
            a_Rs[j] = rho_s_to_a_Rs(rho_val, Ps[j])
        else
            M_s = theta.params.config.M_s
            R_s = theta.params.config.R_s
            if !isnan(M_s) && !isnan(R_s) && R_s > 0
                # a/R* = (G M_s P² / 4π²)^(1/3) / R_s
                P_s = Ps[j] * T(86400.0)
                GM = T(GM_SUN_CGS) * M_s  # GM_sun * M_s [cm³/s²]
                a_cm = cbrt(GM * P_s^2 / (4 * T(π)^2))
                a_Rs[j] = a_cm / (R_s * T(R_SUN_CM))
            else
                # No stellar params available — cannot compute transit geometry
                return zero(T)
            end
        end

        # Geometry gate: planet transits if b < 1 + rr
        transits[j] = bs[j] < 1 + rrs[j]

        if any_gd && has_gd(theta.params.config.planet_modes[k])
            gd_on[j]   = true
            gd_lams[j] = planet_lambda(theta, k)
        end

        # Phase-distance early-out cache: the window half-width T_dur_safe
        # is set after the TTV decode (`_set_transit_windows!`), from a bound
        # that holds every cadence with the planet in front and z < 1 + rr.
        # Tc_centers must be the *true* transit center (mid-conjunction).
        # `Tps + P/4` is the CIRCULAR-orbit approximation — wrong by up
        # to ~e·P for ω near ±π/2, large enough that the T_dur_safe
        # phase gate folds the wrong window and skips the actual transit
        # entirely. tp_to_tc recovers Tc exactly via the Kepler inverse.
        Tc_centers[j] = tp_to_tc(Tps[j], Ps[j], es[j], ws[j])
        T_dur_safe[j] = T(Inf)        # set by _set_transit_windows!
    end

    any(transits) || return _phot_ll_no_transit(theta, data)

    # --- Gravity-darkening state -------------------------------------
    # ω is NOT free: v_eq = v_sin_i_star / sin(i★) and ω = v_eq / v_crit, so the
    # oblateness -- and hence the amplitude of the light-curve asymmetry -- is
    # fixed by i★. That is what turns the asymmetry into a measurement of i★
    # rather than a degenerate scale factor.
    gd_state = nothing
    if any_gd && any(gd_on)
        i_star_v = system_i_star(theta)
        # Open at both ends because sin(i★) = 0 has no finite v_eq. NOT capped at
        # π/2: past it the spin vector points away from the observer, which the
        # brightness map handles, and a cap here overrode whatever range the
        # prior on i_star gave.
        (i_star_v <= 0 || i_star_v >= T(π)) && return convert(T, -Inf)
        M_s_gd = theta.params.config.M_s
        R_s_gd = theta.params.config.R_s
        (isnan(M_s_gd) || isnan(R_s_gd) || R_s_gd <= 0) &&
            throw(ArgumentError("a planet uses :GD but M_s / R_s are not set; " *
                                "gravity darkening needs both to convert " *
                                "v_sin_i_star into a fraction of break-up"))
        v_eq_kms = system_vsini(theta) / (sin(i_star_v) * 1000)
        gd_state = (i_star = i_star_v,
                    λs     = gd_lams,
                    ω_frac = omega_frac_from_veq(v_eq_kms, M_s_gd, R_s_gd),
                    on     = gd_on)
    end
    gd_ctxs = _gd_contexts(theta, gd_state, n_transit,
                           length(theta.params.config.instruments.pm_names))
    # Single-assignment copies for the threaded loops below. `gd_state` is assigned
    # twice above, and a variable assigned more than once and captured by a
    # `Threads.@threads` closure is put in a Core.Box: every use inside the loop is
    # then untyped, and the per-cadence call boxed its arguments and its result.
    gd_s, gd_c = gd_state, gd_ctxs

    # --- TTV decode ---------------------------------------------------
    # Per-planet timing offsets (TTV-A). `r_for_j[j]` is the row in
    # `ttv_state.δts` for transit-array index `j`, or 0 if planet `j`
    # is not a TTV planet. Both `_decode_ttv_state` and the j-loop
    # above iterate `p_idx` skipping non-photometric planets, so the
    # j indexing matches.
    n_ttv, ttv_state = _decode_ttv_state(theta, p_idx)
    r_for_j = zeros(Int, n_transit)
    if n_ttv > 0
        @inbounds for r in 1:n_ttv
            r_for_j[ttv_state.j_active[r]] = r
        end
        # TTV-C: fill δts for `:TTV_NB` rows from TTVFaster. Pairwise
        # N-body prediction; rows are no-ops for `:TTV` (free-offset)
        # planets, which keep their layout-decoded δts.
        if any(ttv_state.is_nb)
            t_max = isempty(data.t_phot) ? zero(T) : T(maximum(data.t_phot))
            t_min = isempty(data.t_phot) ? zero(T) : T(minimum(data.t_phot))
            _apply_ttv_nb!(ttv_state, theta, p_idx,
                            Ps, es, ws, Tps, bs, a_Rs, t_max; t_phot_min = t_min)
        end
    end
    _set_transit_windows!(T_dur_safe, n_transit, transits, r_for_j, ttv_state, Ps, es, ws,
                          rrs, a_Rs, Tc_centers, Tps, 0.0, bs)

    # --- Limb darkening per instrument (via precomputed indices) ------
    systemic = theta.params.layout.systemic

    # Detect active photometry-channel noise models: AR/MA, a GP, an
    # additive covariance (HarmonicBlock / NightlyOffset), Student-t --
    # anything on :phot. With any of these the path is serial: AR is
    # applied to model flux before residual computation, MA after, and the
    # covariance / likelihood family is handled by `_eval_channel_likelihood`.
    # With no phot noise models we keep the thread-parallel white-noise sum,
    # which knows nothing else.
    noise_models = theta.params.config.noise_models
    needs_serial = _phot_noise_active(theta, noise_models)

    if !needs_serial
        # White-noise path — thread-parallel Gaussian sum. Per-instrument
        # offset (additive DC drift) and jitter (added to variance in
        # quadrature) come from the layout slots; the same accessors
        # are used by the serial path below.
        #
        # OPTIMIZATION: pre-build per-instrument QuadLimbDark, offset,
        # and jitter once per likelihood call instead of once per phot
        # point per planet. For 13k photometric points × 4 active
        # planets that was ~52k QuadLimbDark constructions per likelihood
        # eval — the dominant per-eval allocation cost. Looking up by
        # `ins_idx` inside the loop is a single load instead of a struct
        # build that calls `compute_gn` and a normalization divide.
        n_pm = length(theta.params.config.instruments.pm_names)
        # Build the first LD up-front so we can declare a concrete type
        # for the vector — this keeps `lds[ins_idx]` lookup type-stable
        # in the threaded inner loop, avoiding dynamic dispatch overhead.
        q1_1 = theta.values[systemic.ld_q1[1]]
        q2_1 = theta.values[systemic.ld_q2[1]]
        u1_1, u2_1 = kipping_q_to_u(q1_1, q2_1)
        ld1 = QuadLimbDark([u1_1, u2_1])
        lds      = Vector{typeof(ld1)}(undef, n_pm)
        offsets  = Vector{T}(undef, n_pm)
        jitters  = Vector{T}(undef, n_pm)
        dilutions = Vector{T}(undef, n_pm)
        lds[1]     = ld1
        offsets[1] = pm_offset(theta, 1)
        jitters[1] = pm_jitter(theta, 1)
        dilutions[1] = pm_dilution(theta, 1)
        for ix in 2:n_pm
            q1 = theta.values[systemic.ld_q1[ix]]
            q2 = theta.values[systemic.ld_q2[ix]]
            u1, u2 = kipping_q_to_u(q1, q2)
            lds[ix]     = QuadLimbDark([u1, u2])
            offsets[ix] = pm_offset(theta, ix)
            jitters[ix] = pm_jitter(theta, ix)
            dilutions[ix] = pm_dilution(theta, ix)
        end
        n_super = n_super_data
        has_exp = !isempty(data.exposure_times)
        trends_c = _phot_trend_cache(theta, n_pm)
        inv_th = theta.params.config.phot_trend_order > 0 ? _phot_inv_t_half(data) : 0.0
        sky_c = _sky_orbits(n_transit, Ps, es, ws, Tps, bs, a_Rs)

        # DETERMINISTIC REDUCTION. This accumulator used to be
        # `local_total[Threads.threadid()] += ...` under `Threads.@threads`
        # (:dynamic), which had two defects:
        #
        #   * a genuine lost-update race — a task may migrate between threads
        #     BETWEEN the read and the write of `+=`, so the update is dropped
        #     outright, not merely reordered;
        #   * non-determinism — which thread accumulates which `i` varies per
        #     run, so the floating-point summation order varies, and the total
        #     moves by ~1e-3 log units between identical runs.
        #
        # 1e-3 is far below MCMC noise for a fixed-dimension fit, which is why
        # it was tolerated. It is NOT below the threshold that matters for
        # trans-dimensional sampling: a 1e-3 perturbation flips birth/death
        # accept-reject decisions, and the resulting occupancies are not
        # reproducible at fixed seed.
        #
        # The fix is to partition into FIXED-SIZE chunks and give each chunk
        # its own slot, indexed by CHUNK rather than by thread. Each chunk sums
        # sequentially, and the partials are combined in index order, so the
        # result is bit-identical regardless of how tasks are scheduled AND
        # regardless of nthreads(). Chunk size is a constant, not a function of
        # nthreads(), or the partition itself would depend on the machine.
        #
        # `:static` remains unavailable here (this runs nested inside
        # sample_pt's chain-parallel @spawn), but it is no longer needed: the
        # correctness now comes from the indexing, not from the schedule. In a
        # task of pt_emcee's walker loop the chunks run serially (see
        # `@_threads_unless_nested`), with the same partition and the same bits.
        nchunks  = cld(n_obs, _PHOT_REDUCE_CHUNK)
        partials = fill(zero(T), nchunks)
        @inbounds @_threads_unless_nested for c in 1:nchunks
            lo = (c - 1) * _PHOT_REDUCE_CHUNK + 1
            hi = min(c * _PHOT_REDUCE_CHUNK, n_obs)
            acc = zero(T)
            for i in lo:hi
            t       = data.t_phot[i]
            obs     = data.flux[i]
            obs_err = data.flux_err[i]
            ins_idx = data.phot_inst[i]

            ld     = lds[ins_idx]
            offset = offsets[ins_idx]
            jitter = jitters[ins_idx]

            texp = has_exp ? data.exposure_times[i] : 0.0
            tprod = _phot_transit_product(t, texp, ld, n_transit, transits,
                        Ps, es, ws, Tps, bs, a_Rs, rrs, Tc_centers, T_dur_safe,
                        r_for_j, ttv_state, n_super,
                        gd_s, gd_c, ins_idx, sky_c)
            model_flux = _phot_continuum(t, t_ref, inv_th, offset, trends_c[ins_idx]) *
                         _apply_dilution(tprod, dilutions[ins_idx])

            var_i = obs_err * obs_err + jitter * jitter
            resid = obs - model_flux
            acc += -(log(two_pi * var_i) + resid * resid / var_i) / 2
            end
            partials[c] = acc
        end

        return sum(partials)
    end

    # Serial path — needed for Stage 2 (AR/MA) sequential evaluation
    # and/or Stage 3 (celerite GP) which both demand time-ordered
    # buffers. Auto-diff propagates through (T-typed buffers below).
    # Same per-instrument LD/offset/jitter caching as the threaded path.
    n_pm_s = length(theta.params.config.instruments.pm_names)
    q1_1s = theta.values[systemic.ld_q1[1]]
    q2_1s = theta.values[systemic.ld_q2[1]]
    u1_1s, u2_1s = kipping_q_to_u(q1_1s, q2_1s)
    ld1s = QuadLimbDark([u1_1s, u2_1s])
    lds_s     = Vector{typeof(ld1s)}(undef, n_pm_s)
    offsets_s = Vector{T}(undef, n_pm_s)
    jitters_s = Vector{T}(undef, n_pm_s)
    dilutions_s = Vector{T}(undef, n_pm_s)
    lds_s[1]     = ld1s
    offsets_s[1] = pm_offset(theta, 1)
    jitters_s[1] = pm_jitter(theta, 1)
    dilutions_s[1] = pm_dilution(theta, 1)
    for ix in 2:n_pm_s
        q1 = theta.values[systemic.ld_q1[ix]]
        q2 = theta.values[systemic.ld_q2[ix]]
        u1, u2 = kipping_q_to_u(q1, q2)
        lds_s[ix]     = QuadLimbDark([u1, u2])
        offsets_s[ix] = pm_offset(theta, ix)
        jitters_s[ix] = pm_jitter(theta, ix)
        dilutions_s[ix] = pm_dilution(theta, ix)
    end
    n_super_s = n_super_data
    has_exp_s = !isempty(data.exposure_times)
    trends_cs = _phot_trend_cache(theta, n_pm_s)
    inv_th_s = theta.params.config.phot_trend_order > 0 ? _phot_inv_t_half(data) : 0.0
    sky_cs = _sky_orbits(n_transit, Ps, es, ws, Tps, bs, a_Rs)

    predictions = Vector{T}(undef, n_obs)
    variances   = Vector{T}(undef, n_obs)
    # Threaded vs serial prediction build:
    # - Float64 (the PT chain itself, RWM/slice — no gradients): threaded.
    # - ForwardDiff.Dual (Pathfinder warmup gradients): serial — Duals
    #   allocate per-iteration scratch and Threads.@threads on a 152k loop
    #   creates 8× allocator pressure that GC-thrashes for hours.
    # The AR/MA/GP stages below stay serial regardless (they need
    # sequential access).
    if T <: AbstractFloat
        @_threads_unless_nested for i in 1:n_obs
            @inbounds begin
                t       = data.t_phot[i]
                obs_err = data.flux_err[i]
                ins_idx = data.phot_inst[i]

                ld     = lds_s[ins_idx]
                offset = offsets_s[ins_idx]
                jitter = jitters_s[ins_idx]

                texp = has_exp_s ? data.exposure_times[i] : 0.0
                tprod = _phot_transit_product(t, texp, ld, n_transit, transits,
                            Ps, es, ws, Tps, bs, a_Rs, rrs, Tc_centers,
                            T_dur_safe, r_for_j, ttv_state, n_super_s,
                        gd_s, gd_c, ins_idx, sky_cs)
                predictions[i] = _phot_continuum(t, t_ref, inv_th_s, offset,
                                     trends_cs[ins_idx]) *
                                 _apply_dilution(tprod, dilutions_s[ins_idx])
                variances[i]   = obs_err * obs_err + jitter * jitter
            end
        end
    else
        @inbounds for i in 1:n_obs
            t       = data.t_phot[i]
            obs_err = data.flux_err[i]
            ins_idx = data.phot_inst[i]

            ld     = lds_s[ins_idx]
            offset = offsets_s[ins_idx]
            jitter = jitters_s[ins_idx]

            texp = has_exp_s ? data.exposure_times[i] : 0.0
            tprod = _phot_transit_product(t, texp, ld, n_transit, transits,
                        Ps, es, ws, Tps, bs, a_Rs, rrs, Tc_centers,
                        T_dur_safe, r_for_j, ttv_state, n_super_s,
                        gd_s, gd_c, ins_idx, sky_cs)
            predictions[i] = _phot_continuum(t, t_ref, inv_th_s, offset,
                                 trends_cs[ins_idx]) *
                             _apply_dilution(tprod, dilutions_s[ins_idx])
            variances[i]   = obs_err * obs_err + jitter * jitter
        end
    end

    # Stage 2 — AR on the model flux (Tuomi convention)
    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa ARModel && noise_channel(nm) === :phot
            apply_ar!(predictions, data.t_phot, data.phot_inst, theta, nm)
        end
    end

    residuals = Vector{T}(undef, n_obs)
    @inbounds for i in 1:n_obs
        residuals[i] = data.flux[i] - predictions[i]
    end

    # Stage 2 — MA on residuals
    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa MAModel && noise_channel(nm) === :phot
            apply_ma!(residuals, data.t_phot, data.phot_inst, theta, nm)
        end
    end

    return _eval_channel_likelihood(theta, residuals, variances, data.t_phot,
                                     data.phot_inst, :phot, two_pi)
end

"""
    phot_predictions(theta, data) -> (predictions, variances)

Forward-model the photometric flux (transits + per-instrument offsets)
and per-cadence variances under the current `theta`. Mirrors
`rv_predictions` for the photometry channel.

Returns `(predictions::Vector{T}, variances::Vector{T})` of length
`length(data.t_phot)`. `predictions` is the noise-free transit model
`(1 + offset_inst) × Π_k transit_flux_k(t)`; `variances` is the
white-noise floor `flux_err² + jitter_inst²`.

Only includes Stage 1 (transits + offset) and white noise. Does NOT
apply Stage 2 (AR/MA) or Stage 3 (GP) noise models — those are noise
models added on top of the mean prediction. For residual-periodogram
diagnostics, this is what you want: per-cadence "what would the model
predict if there were no AR/MA/GP" — subtract from data to get raw
residuals.
"""
function phot_predictions(theta::Theta{T}, data::Data) where {T}
    n_obs = length(data.t_phot)
    if n_obs == 0
        return Vector{T}(), Vector{T}()
    end

    parametrization = theta.params.config.parametrization
    t_ref  = data.t_ref
    two_pi = T(2π)

    # Decode transit planets (geometry gate)
    p_idx = planet_indices(theta)
    # One assignment, not a running count: a variable assigned more than once
    # and captured by the threaded loops below is put in a Core.Box, and every
    # read of it in the loop body is then untyped (see `gd_s` below).
    n_transit = count(k -> has_geometry(theta.params.layout.planet_blocks[k]), p_idx)

    systemic = theta.params.layout.systemic
    n_pm     = length(theta.params.config.instruments.pm_names)

    # Per-instrument offset / jitter
    offsets  = Vector{T}(undef, n_pm)
    jitters  = Vector{T}(undef, n_pm)
    for ix in 1:n_pm
        offsets[ix] = pm_offset(theta, ix)
        jitters[ix] = pm_jitter(theta, ix)
    end
    trends_cp = _phot_trend_cache(theta, n_pm)
    inv_th_p = theta.params.config.phot_trend_order > 0 ? _phot_inv_t_half(data) : 0.0
    t_ref_p  = data.t_ref

    predictions = Vector{T}(undef, n_obs)
    variances   = Vector{T}(undef, n_obs)

    # Fast path: no active transiting planets — model is just the
    # per-instrument offset. It reads neither limb darkening nor dilution:
    # a fit with no transiting planet (a noise-only light curve) has no
    # slots for them, as `_phot_ll_no_transit` knows.
    if n_transit == 0
        @inbounds for i in 1:n_obs
            ins_idx = data.phot_inst[i]
            predictions[i] = _phot_continuum(data.t_phot[i], t_ref_p, inv_th_p,
                                             offsets[ins_idx], trends_cp[ins_idx])
            σ²            = data.flux_err[i]^2 + jitters[ins_idx]^2
            variances[i]   = σ²
        end
        return predictions, variances
    end

    # Per-instrument LD / dilution
    q1_1 = theta.values[systemic.ld_q1[1]]
    q2_1 = theta.values[systemic.ld_q2[1]]
    u1_1, u2_1 = kipping_q_to_u(q1_1, q2_1)
    ld1 = QuadLimbDark([u1_1, u2_1])
    lds      = Vector{typeof(ld1)}(undef, n_pm)
    dilutions = Vector{T}(undef, n_pm)
    lds[1]     = ld1
    dilutions[1] = pm_dilution(theta, 1)
    for ix in 2:n_pm
        q1 = theta.values[systemic.ld_q1[ix]]
        q2 = theta.values[systemic.ld_q2[ix]]
        u1, u2 = kipping_q_to_u(q1, q2)
        lds[ix]     = QuadLimbDark([u1, u2])
        dilutions[ix] = pm_dilution(theta, ix)
    end
    n_super_p = _phot_n_super(data)
    has_exp_p = !isempty(data.exposure_times)

    # Decode orbit parameters per active transiting planet
    Ps   = Vector{T}(undef, n_transit)
    es   = Vector{T}(undef, n_transit)
    ws   = Vector{T}(undef, n_transit)
    Tps  = Vector{T}(undef, n_transit)
    rrs  = Vector{T}(undef, n_transit)
    bs   = Vector{T}(undef, n_transit)
    a_Rs = Vector{T}(undef, n_transit)
    transits   = Vector{Bool}(undef, n_transit)
    Tc_centers = Vector{T}(undef, n_transit)
    T_dur_safe = Vector{T}(undef, n_transit)

    any_gd  = any(has_gd(m) for m in theta.params.config.planet_modes)
    gd_on   = any_gd ? fill(false, n_transit) : Bool[]
    gd_lams = any_gd ? zeros(T, n_transit)    : T[]

    use_rho = parametrization.use_rho_s
    rho_val = use_rho ? rho_s(theta) : zero(T)

    j = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_geometry(block) || continue
        j += 1
        Ps[j] = planet_P(theta, k)
        e, w  = planet_e_w(theta, k)
        es[j] = e; ws[j] = w
        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tps[j] = t_ref - ta * Ps[j] / two_pi
        elseif parametrization.time === :Tp
            Tps[j] = ta
        else
            Tps[j] = tc_to_tp(ta, Ps[j], e, w)
        end
        b_raw, rr_raw = planet_b_rr(theta, k)
        if parametrization.geom === :r1r2
            rrs[j] = b_raw * (1 + rr_raw) / 2
            bs[j]  = b_raw * (1 - rr_raw) / 2
        else
            bs[j]  = b_raw
            rrs[j] = rr_raw
        end
        if (ai = a_Rs_slot(theta, k)) > 0
            a_Rs[j] = theta.values[ai]
        elseif use_rho
            a_Rs[j] = rho_s_to_a_Rs(rho_val, Ps[j])
        else
            M_s = theta.params.config.M_s
            R_s = theta.params.config.R_s
            if !isnan(M_s) && !isnan(R_s) && R_s > 0
                P_s = Ps[j] * T(86400.0)
                GM = T(GM_SUN_CGS) * M_s
                a_cm = cbrt(GM * P_s^2 / (4 * T(π)^2))
                a_Rs[j] = a_cm / (R_s * T(R_SUN_CM))
            else
                a_Rs[j] = zero(T)
            end
        end
        transits[j] = bs[j] < 1 + rrs[j]
        if any_gd && has_gd(theta.params.config.planet_modes[k])
            gd_on[j]   = true
            gd_lams[j] = planet_lambda(theta, k)
        end
        # Tc_centers must be the *true* transit center (mid-conjunction).
        # `Tps + P/4` is the CIRCULAR-orbit approximation — wrong by up
        # to ~e·P for ω near ±π/2, large enough that the T_dur_safe
        # phase gate folds the wrong window and skips the actual transit
        # entirely. tp_to_tc recovers Tc exactly via the Kepler inverse.
        Tc_centers[j] = tp_to_tc(Tps[j], Ps[j], es[j], ws[j])
        T_dur_safe[j] = T(Inf)        # set by _set_transit_windows!
    end

    gd_state = nothing
    if any_gd && any(gd_on)
        i_star_v = system_i_star(theta)
        M_s_gd = theta.params.config.M_s
        R_s_gd = theta.params.config.R_s
        if 0 < i_star_v < T(π) && !isnan(M_s_gd) && !isnan(R_s_gd) && R_s_gd > 0
            v_eq_kms = system_vsini(theta) / (sin(i_star_v) * 1000)
            gd_state = (i_star = i_star_v, λs = gd_lams,
                        ω_frac = omega_frac_from_veq(v_eq_kms, M_s_gd, R_s_gd),
                        on = gd_on)
        end
    end
    gd_ctxs = _gd_contexts(theta, gd_state, n_transit,
                           length(theta.params.config.instruments.pm_names))
    # Single-assignment copies for the threaded loops below. `gd_state` is assigned
    # twice above, and a variable assigned more than once and captured by a
    # `Threads.@threads` closure is put in a Core.Box: every use inside the loop is
    # then untyped, and the per-cadence call boxed its arguments and its result.
    gd_s, gd_c = gd_state, gd_ctxs

    # TTV decode (TTV-A + TTV-C) — see transit_log_likelihood for rationale.
    n_ttv, ttv_state = _decode_ttv_state(theta, p_idx)
    r_for_j = zeros(Int, n_transit)
    if n_ttv > 0
        @inbounds for r in 1:n_ttv
            r_for_j[ttv_state.j_active[r]] = r
        end
        if any(ttv_state.is_nb)
            t_max = isempty(data.t_phot) ? zero(T) : T(maximum(data.t_phot))
            t_min = isempty(data.t_phot) ? zero(T) : T(minimum(data.t_phot))
            _apply_ttv_nb!(ttv_state, theta, p_idx,
                            Ps, es, ws, Tps, bs, a_Rs, t_max; t_phot_min = t_min)
        end
    end
    _set_transit_windows!(T_dur_safe, n_transit, transits, r_for_j, ttv_state, Ps, es, ws,
                          rrs, a_Rs, Tc_centers, Tps, 0.0, bs)
    sky_p = _sky_orbits(n_transit, Ps, es, ws, Tps, bs, a_Rs)

    # Threaded for Float64 (no Duals = no GC thrashing), serial otherwise.
    if T <: AbstractFloat
        @_threads_unless_nested for i in 1:n_obs
            @inbounds begin
                t       = data.t_phot[i]
                ins_idx = data.phot_inst[i]
                ld      = lds[ins_idx]
                offset  = offsets[ins_idx]

                texp = has_exp_p ? data.exposure_times[i] : 0.0
                tprod = _phot_transit_product(t, texp, ld, n_transit, transits,
                            Ps, es, ws, Tps, bs, a_Rs, rrs, Tc_centers,
                            T_dur_safe, r_for_j, ttv_state, n_super_p,
                        gd_s, gd_c, ins_idx, sky_p)
                predictions[i] = _phot_continuum(t, t_ref_p, inv_th_p, offset,
                                     trends_cp[ins_idx]) *
                                 _apply_dilution(tprod, dilutions[ins_idx])
                variances[i]   = data.flux_err[i]^2 + jitters[ins_idx]^2
            end
        end
    else
        @inbounds for i in 1:n_obs
            t       = data.t_phot[i]
            ins_idx = data.phot_inst[i]
            ld      = lds[ins_idx]
            offset  = offsets[ins_idx]

            texp = has_exp_p ? data.exposure_times[i] : 0.0
            tprod = _phot_transit_product(t, texp, ld, n_transit, transits,
                        Ps, es, ws, Tps, bs, a_Rs, rrs, Tc_centers,
                        T_dur_safe, r_for_j, ttv_state, n_super_p,
                        gd_s, gd_c, ins_idx, sky_p)
            predictions[i] = _phot_continuum(t, t_ref_p, inv_th_p, offset,
                                 trends_cp[ins_idx]) *
                             _apply_dilution(tprod, dilutions[ins_idx])
            variances[i]   = data.flux_err[i]^2 + jitters[ins_idx]^2
        end
    end

    return predictions, variances
end

# Whether any noise model on the :phot channel is active. The white-noise
# sums of the transit likelihood score independent Gaussians and nothing
# else, so any of them -- not only AR/MA and GPs, but an additive covariance
# or a Student-t likelihood too -- has to take the serial path through
# `_eval_channel_likelihood`, or it is silently dropped.
function _phot_noise_active(theta::Theta, noise_models)
    for (nm_idx, nm) in enumerate(noise_models)
        noise_channel(nm) === :phot || continue
        is_noise_model_active(theta, nm_idx) && return true
    end
    return false
end

"""
    _phot_ll_no_transit(theta, data) -> T

Fast path: no transiting planets — flux is modelled as 1.0. If a
photometry-channel `CovarianceNoise` is active, the residual `flux − 1`
is handed to `gp_log_likelihood` (so this path is also the one used by
`detrend_gp`'s noise-only NereusTarget when fitting GP hyperparameters
to a transit-masked LC). Otherwise the white-noise Gaussian sum is
evaluated.
"""
function _phot_ll_no_transit(theta::Theta{T}, data::Data) where {T}
    n_obs = length(data.t_phot)
    n_obs > 0 || return zero(T)

    two_pi = T(2π)
    noise_models = theta.params.config.noise_models

    # Build the (transit-free) mean model and per-cadence variances. The baseline
    # continuum (offset + any trend) still applies out of transit.
    n_pm = length(theta.params.config.instruments.pm_names)
    trends_c = _phot_trend_cache(theta, n_pm)
    inv_th = theta.params.config.phot_trend_order > 0 ? _phot_inv_t_half(data) : 0.0
    t_ref = data.t_ref
    predictions = Vector{T}(undef, n_obs)
    variances   = Vector{T}(undef, n_obs)
    @inbounds for i in 1:n_obs
        ins_idx = data.phot_inst[i]
        jitter  = pm_jitter(theta, ins_idx)
        predictions[i] = _phot_continuum(data.t_phot[i], t_ref, inv_th,
                                         pm_offset(theta, ins_idx), trends_c[ins_idx])
        variances[i]   = data.flux_err[i] * data.flux_err[i] + jitter * jitter
    end

    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa ARModel && noise_channel(nm) === :phot
            apply_ar!(predictions, data.t_phot, data.phot_inst, theta, nm)
        end
    end

    residuals = Vector{T}(undef, n_obs)
    @inbounds for i in 1:n_obs
        residuals[i] = data.flux[i] - predictions[i]
    end

    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa MAModel && noise_channel(nm) === :phot
            apply_ma!(residuals, data.t_phot, data.phot_inst, theta, nm)
        end
    end

    return _eval_channel_likelihood(theta, residuals, variances, data.t_phot,
                                     data.phot_inst, :phot, two_pi)
end

# Workspace-aware no-transit phot log-L (Fix B). The no-transit photometry
# likelihood reads only per-instrument offset/jitter (the flat 1+offset model),
# so we cache it keyed on hash(offset, jitter). Valid ONLY when no phot AR/MA/GP
# is active — those read extra params not in the hash, so fall back to the
# non-cached variant. Shares ws.phot_ll_total_{cache,hash} with the transit
# path; the "NOTRANS" sentinel keeps the two key-spaces disjoint.
@inline function _phot_ll_no_transit(theta::Theta{T}, data::Data, ws) where {T}
    @inbounds for nm_idx in 1:length(theta.params.config.noise_models)
        nm = theta.params.config.noise_models[nm_idx]
        is_noise_model_active(theta, nm_idx) || continue
        noise_channel(nm) === :phot && return _phot_ll_no_transit(theta, data)
    end
    n_pm = length(theta.params.config.instruments.pm_names)
    h = hash(0x4e4f5452414e53)   # "NOTRANS" sentinel
    for ix in 1:n_pm
        h = hash(pm_offset(theta, ix), hash(pm_jitter(theta, ix), h))
        for s in theta.params.layout.systemic.pm_trend[ix]
            h = hash(theta.values[s], h)
        end
    end
    if ws.phot_ll_total_hash == h && ws.phot_ll_total_hash != zero(UInt)
        return ws.phot_ll_total_cache
    end
    result = _phot_ll_no_transit(theta, data)
    ws.phot_ll_total_cache = result
    ws.phot_ll_total_hash = h
    return result
end

# `_phot_sparse_refresh` for the workspace flux cache with one more argument:
# `nothing` for an ordinary row, which is the method above unchanged, or the
# `_gd_contexts` matrix for a row whose planet uses :GD. A :GD row's flux at
# each cadence comes from the same `_sky_orbit`, `_sky_position` and
# `transit_flux_gd` calls, on the same arguments, as the instantaneous
# gravity-darkened branch of `_phot_transit_product`, so it is bit-identical to
# the non-workspace one.
@inline _phot_sparse_refresh(a::Int, b::Int, j::Int, data::Data,
                             idx::AbstractVector{Int}, P::Float64, e::Float64,
                             ω::Float64, Tp::Float64, b_imp::Float64, a_Rs::Float64,
                             rr::Float64, lds::AbstractVector,
                             flux_cache::AbstractMatrix{Float64}, ::Nothing) =
    _phot_sparse_refresh(a, b, j, data, idx, P, e, ω, Tp, b_imp, a_Rs, rr, lds,
                         flux_cache)

@inline function _phot_sparse_refresh(a::Int, b::Int, j::Int, data::Data,
                                       idx::AbstractVector{Int},
                                       P::Float64, e::Float64, ω::Float64,
                                       Tp::Float64, b_imp::Float64,
                                       a_Rs::Float64, rr::Float64,
                                       lds::AbstractVector,
                                       flux_cache::AbstractMatrix{Float64},
                                       gd_ctxs::AbstractMatrix)
    a > b && return nothing
    o = _sky_orbit(P, e, ω, Tp, b_imp, a_Rs)
    @inbounds for k in a:b
        i = idx[k]
        ins_idx = data.phot_inst[i]
        x, y, zl = _sky_position(o, data.t_phot[i])
        flux_cache[j, i] = transit_flux_gd(lds[ins_idx], gd_ctxs[j, ins_idx],
                                           x, y, zl, rr)
    end
    return nothing
end

# =====================================================================
# Workspace-aware variant (used by `_eval_ll(theta, data, ctr, ws)`
# in trans-dim PT). Reuses pre-allocated planet-decode buffers from
# `ws` instead of allocating Vector{T}(undef, n_transit) per call.
# `ws` is left untyped to dodge the forward-reference to PTWorkspace
# (defined later in the include order — same pattern as
# `rv_log_likelihood(..., ws)` in likelihood.jl).
# =====================================================================

"""
    transit_log_likelihood(theta::Theta{T}, data::Data, ws) -> T

Workspace-aware transit log-likelihood. Equivalent to the non-`ws`
method but reuses `ws.transit_*` buffers for the per-planet decode,
avoiding ~9 small `Vector{T}` allocations per call. Only the
threaded white-noise path uses `ws`; the serial path (phot ARMA/GP
active) falls back to the original allocating implementation.
"""
function transit_log_likelihood(theta::Theta{T}, data::Data, ws) where {T}
    n_obs = length(data.t_phot)
    n_obs > 0 || return zero(T)

    parametrization = theta.params.config.parametrization
    t_ref = data.t_ref
    two_pi = T(2π)

    p_idx = planet_indices(theta)
    n_transit = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_geometry(block) || continue
        n_transit += 1
    end

    if n_transit == 0
        return _phot_ll_no_transit(theta, data, ws)
    end

    # TTV bypass: the workspace flux-cache hashes orbit parameters only
    # (no δts), so it cannot detect cache invalidation when TTV offsets
    # change. For TTV fits, route through the non-cached variant which
    # recomputes per-point sky_separation every call.
    n_super_data = _phot_n_super(data, ws.phot_data)
    if _has_active_ttv(theta, p_idx)
        return _transit_ll_direct(theta, data, n_super_data)
    end

    # Finite-exposure bypass: the analytic flux-cache evaluates the transit at
    # cadence MIDPOINTS only. When long-cadence supersampling is required
    # (Kipping 2010), route through the non-cached variant which integrates the
    # model over each exposure. Short/absent exposures (n_super==1) keep the cache.
    if n_super_data > 1
        return _transit_ll_direct(theta, data, n_super_data)
    end

    Ps   = ws.transit_Ps
    es   = ws.transit_es
    ws_  = ws.transit_ws        # shadow to avoid name collision with `ws`
    Tps  = ws.transit_Tps
    rrs  = ws.transit_rrs
    bs   = ws.transit_bs
    a_Rs = ws.transit_a_Rs
    transits = ws.transit_active
    # Gravity darkening: which rows use it, and each row's λ, filled in the
    # j-loop below as in the non-workspace method. `nothing` without a :GD
    # planet, so an ordinary fit allocates nothing more.
    gd_rows = any(has_gd(m) for m in theta.params.config.planet_modes) ?
        (on = fill(false, n_transit), λs = zeros(T, n_transit)) : nothing

    use_rho = parametrization.use_rho_s
    rho_val = use_rho ? rho_s(theta) : zero(T)

    j = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_geometry(block) || continue
        j += 1
        Ps[j] = planet_P(theta, k)
        e, w = planet_e_w(theta, k)
        (e < 0 || e >= 1) && return convert(T, -Inf)
        es[j] = e
        ws_[j] = w

        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tps[j] = t_ref - ta * Ps[j] / two_pi
        elseif parametrization.time === :Tp
            Tps[j] = ta
        else
            Tps[j] = tc_to_tp(ta, Ps[j], e, w)
        end

        b_raw, rr_raw = planet_b_rr(theta, k)
        if parametrization.geom === :r1r2
            rrs[j] = b_raw * (1 + rr_raw) / 2
            bs[j]  = b_raw * (1 - rr_raw) / 2
        else
            bs[j]  = b_raw
            rrs[j] = rr_raw
        end

        if (ai = a_Rs_slot(theta, k)) > 0
            a_Rs[j] = theta.values[ai]
        elseif use_rho
            a_Rs[j] = rho_s_to_a_Rs(rho_val, Ps[j])
        else
            M_s = theta.params.config.M_s
            R_s = theta.params.config.R_s
            if !isnan(M_s) && !isnan(R_s) && R_s > 0
                P_s = Ps[j] * T(86400.0)
                GM = T(GM_SUN_CGS) * M_s
                a_cm = cbrt(GM * P_s^2 / (4 * T(π)^2))
                a_Rs[j] = a_cm / (R_s * T(R_SUN_CM))
            else
                return zero(T)
            end
        end

        transits[j] = bs[j] < 1 + rrs[j]

        if gd_rows !== nothing && has_gd(theta.params.config.planet_modes[k])
            gd_rows.on[j] = true
            gd_rows.λs[j] = planet_lambda(theta, k)
        end
    end

    any_transit = false
    for jj in 1:n_transit
        if transits[jj]
            any_transit = true
            break
        end
    end
    any_transit || return _phot_ll_no_transit(theta, data, ws)

    systemic = theta.params.layout.systemic
    noise_models = theta.params.config.noise_models
    needs_serial = _phot_noise_active(theta, noise_models)   # see the non-ws method

    if !needs_serial
        # Same per-instrument LD/offset/jitter caching as the non-ws
        # variant — see comments there.
        n_pm = length(theta.params.config.instruments.pm_names)
        q1_1 = theta.values[systemic.ld_q1[1]]
        q2_1 = theta.values[systemic.ld_q2[1]]
        u1_1, u2_1 = kipping_q_to_u(q1_1, q2_1)
        ld1 = QuadLimbDark([u1_1, u2_1])
        lds      = Vector{typeof(ld1)}(undef, n_pm)
        offsets  = Vector{T}(undef, n_pm)
        jitters  = Vector{T}(undef, n_pm)
        dilutions = Vector{T}(undef, n_pm)
        lds[1]     = ld1
        offsets[1] = pm_offset(theta, 1)
        jitters[1] = pm_jitter(theta, 1)
        dilutions[1] = pm_dilution(theta, 1)
        for ix in 2:n_pm
            q1 = theta.values[systemic.ld_q1[ix]]
            q2 = theta.values[systemic.ld_q2[ix]]
            u1, u2 = kipping_q_to_u(q1, q2)
            lds[ix]     = QuadLimbDark([u1, u2])
            offsets[ix] = pm_offset(theta, ix)
            jitters[ix] = pm_jitter(theta, ix)
            dilutions[ix] = pm_dilution(theta, ix)
        end
        trends_c = _phot_trend_cache(theta, n_pm)
        inv_th = theta.params.config.phot_trend_order > 0 ? _phot_inv_t_half(data) : 0.0

        # Gravity-darkening state, built as the non-workspace method builds it
        # (see there for the physics), with the same -Inf at the ends of i★ and
        # the same error without M_s / R_s. `gd_h` keys a :GD row's cached flux
        # on what the brightness map reads besides the orbit and LD: i★, ω
        # (from v sin i★ and i★) and every band's β. The row's own λ is added
        # per row below.
        gd_state = nothing
        gd_h = zero(UInt)
        if gd_rows !== nothing && any(gd_rows.on)
            i_star_v = system_i_star(theta)
            (i_star_v <= 0 || i_star_v >= T(π)) && return convert(T, -Inf)
            M_s_gd = theta.params.config.M_s
            R_s_gd = theta.params.config.R_s
            (isnan(M_s_gd) || isnan(R_s_gd) || R_s_gd <= 0) &&
                throw(ArgumentError("a planet uses :GD but M_s / R_s are not set; " *
                                    "gravity darkening needs both to convert " *
                                    "v_sin_i_star into a fraction of break-up"))
            v_eq_kms = system_vsini(theta) / (sin(i_star_v) * 1000)
            gd_state = (i_star = i_star_v,
                        λs     = gd_rows.λs,
                        ω_frac = omega_frac_from_veq(v_eq_kms, M_s_gd, R_s_gd),
                        on     = gd_rows.on)
            gd_h = hash(gd_state.i_star, hash(gd_state.ω_frac, hash(0x4744)))  # "GD"
            for ix in 1:n_pm
                gd_h = hash(system_gd_beta(theta, ix), gd_h)
            end
        end
        # Brightness contexts: built on the first :GD row that needs a refresh.
        gd_ctxs = nothing

        # ============================================================
        # Per-planet transit-flux cache. Each cached row holds the
        # full Mandel-Agol flux contribution per phot point for one
        # planet. The cache key combines the planet's orbit
        # (P,e,ω,Tp,b,a/R*), its rr, and ALL instrument LDs (because
        # phot points across instruments share a row, and a change to
        # any instrument's q1/q2 invalidates the whole row). A :GD
        # planet's key adds `gd_h` and its λ, so its cached flux is never
        # reused across a different i★, v sin i★, λ or β.
        # ============================================================
        flux_cache = ws.transit_flux_cache
        flux_hash  = ws.transit_flux_hash

        # Aggregate LD hash across all instruments — a change to any
        # instrument's q1/q2 invalidates every active planet's row.
        ld_h = zero(UInt)
        for ix in 1:n_pm
            u_n = lds[ix].u_n
            ld_h = hash(u_n[2], hash(u_n[3], ld_h))
        end

        # Chunk geometry shared between cache-refresh and main-pass
        # @spawn loops. Profile showed `Threads.@threads + threadid()`
        # was costing ~19% in thread sync overhead — chunked @spawn
        # eliminates it.
        nT    = Threads.nthreads()
        chunk = cld(n_obs, nT)

        # Refresh stale rows — SPARSE PATH. Only points within the
        # transit window contribute non-trivial flux. We:
        #   1. Reset previously non-trivial entries to 1.0
        #   2. Find the new in-transit index set
        #   3. Compute z + transit_flux only at those indices
        # This is ~50× fewer kepler_solve calls vs the dense path
        # at typical 2% transit fractions.
        in_idx_list = ws.transit_in_idx
        # Fix B: accumulate a total hash over every transiting planet's transit
        # key (full_h, which already folds in orbit+geom+LD). Folded with
        # offset/jitter below to key the total phot-ll cache. The "TRANSIT"
        # sentinel + n_transit keeps it disjoint from the no-transit key-space.
        total_h = hash(UInt(n_transit), hash(0x5452414e534954))
        for j in 1:n_transit
            transits[j] || continue
            full_h = hash(Ps[j], hash(es[j], hash(ws_[j],
                  hash(Tps[j], hash(bs[j], hash(a_Rs[j],
                  hash(rrs[j], ld_h)))))))
            is_gd = gd_state !== nothing && gd_state.on[j]
            is_gd && (full_h = hash(gd_state.λs[j], hash(gd_h, full_h)))
            total_h = hash(full_h, total_h)
            if flux_hash[j] != full_h
                # Brightness contexts for a :GD row (`nothing` for the others),
                # built once per call, and only when such a row is refreshed.
                row_gd = !is_gd ? nothing : gd_ctxs !== nothing ? gd_ctxs :
                    (gd_ctxs = _gd_contexts(theta, gd_state, n_transit, n_pm))

                # 1. Reset previous in-transit positions to 1.0
                old_idx = in_idx_list[j]
                @inbounds for i in old_idx
                    flux_cache[j, i] = 1.0
                end
                empty!(old_idx)

                # 2. Find new in-transit indices via phase-distance check
                Pj, ej, wj, Tpj, bj, aj, rrj = Ps[j], es[j], ws_[j],
                    Tps[j], bs[j], a_Rs[j], rrs[j]
                # True transit center via the Kepler inverse (NOT the circular
                # Tpj + Pj/4, which is wrong by ~e·P for ω near ±π/2 and makes
                # the phase gate fold the wrong window → high-e/grazing transits
                # silently vanish). Mirrors the non-ws path (tp_to_tc).
                Tc_center = tp_to_tc(Tpj, Pj, ej, wj)
                # Every cadence with the planet in front and z < 1 + rr (Inf,
                # i.e. all cadences, where the orbit gives no bound).
                T_dur_safe = _transit_window_halfwidth(Pj, ej, wj, rrj, aj,
                                                       max(abs(Tc_center), abs(Tpj)), bj)
                _phot_window_indices!(old_idx, data, ws.phot_data, Tc_center, Pj,
                                      T_dur_safe)

                # 3. Compute flux at the in-transit indices. Threaded
                # only when N is large enough to amortize spawn overhead;
                # most planets have <500 in-transit points, single-thread
                # is faster. Never in a task marked by `_serial_inner_loops!`,
                # whose sampler has every thread busy already.
                n_tx = length(old_idx)
                if n_tx > 2000 && !_inner_loops_serial()
                    tx_chunk = cld(n_tx, nT)
                    refresh_tasks = Vector{Task}(undef, nT)
                    for c in 1:nT
                        a = (c - 1) * tx_chunk + 1
                        b = min(c * tx_chunk, n_tx)
                        refresh_tasks[c] = Threads.@spawn _phot_sparse_refresh(
                            a, b, j, data, old_idx, Pj, ej, wj, Tpj,
                            bj, aj, rrj, lds, flux_cache, row_gd)
                    end
                    for c in 1:nT
                        wait(refresh_tasks[c])
                    end
                else
                    _phot_sparse_refresh(1, n_tx, j, data, old_idx,
                        Pj, ej, wj, Tpj, bj, aj, rrj, lds, flux_cache, row_gd)
                end
                flux_hash[j] = full_h
            end
        end

        # Fix B: fold per-instrument offset/jitter into the total hash, then
        # short-circuit the O(n_phot) assembly+sum if the entire phot state is
        # unchanged since the last eval (e.g. a within-model move on an RV-only
        # coordinate). flux_cache is already up to date from the refresh loop.
        for ix in 1:n_pm
            total_h = hash(offsets[ix], hash(jitters[ix],
                           hash(dilutions[ix], total_h)))
            for c in trends_c[ix]
                total_h = hash(c, total_h)
            end
        end
        if ws.phot_ll_total_hash == total_h && ws.phot_ll_total_hash != zero(UInt)
            return ws.phot_ll_total_cache
        end

        # Main pass is cheap (cached fluxes → array reads + multiplies +
        # one log per point). Threading 13k items across 8 spawns adds
        # ~80 μs of scheduler overhead vs ~250 μs of useful work — net
        # 25% overhead per call. Single-threaded with @inbounds + SIMD-
        # friendly loop is faster end-to-end for this size.
        result = _phot_chunk_loglik(1, n_obs, data, n_transit, transits,
                                     flux_cache, offsets, jitters, dilutions,
                                     t_ref, inv_th, trends_c, two_pi)
        ws.phot_ll_total_cache = result
        ws.phot_ll_total_hash = total_h
        return result
    end

    # Serial path (phot ARMA / GP) — fall back to the allocating
    # variant. The per-cadence n_obs buffers there have a different
    # lifetime than `ws.transit_*` and aren't worth threading through.
    return _transit_ll_direct(theta, data, n_super_data)
end
