# Likelihood functions.
#
# Architecture — 3-stage pipeline:
#   Stage 1: Build mean prediction (Keplerian + gamma + trend + activity)
#   Stage 2: Apply sequential noise (MA/AR) to residuals
#   Stage 3: Evaluate log-likelihood (white noise or GP covariance)
#
# Planets decoded ONCE, Kepler solve per-planet per-obs.
# All time anchors (Mo/Tp/Tc) unified to Tp in decode-once.

"""
    _comp_rv(geom, comp, KA, cB) -> contribution

Per-observation RV contribution of ONE decoded orbital block to a point
of stellar component `comp` (1 = primary A, 2 = secondary B), where
`geom = cos(f + ω) + e·cos(ω)` is the unit-amplitude Keplerian factor.

- `KA` — amplitude seen by the primary (component A): the planet's `K`
  for a circumprimary planet, or the binary's `K_A` for the SB2 orbit.
- `cB` — coefficient for the secondary (component B): `0` for a
  circumprimary planet (invisible in the secondary's deblended lines),
  or `−K_B` for the SB2 binary orbit (the secondary's anti-phase reflex).

THE single source of truth for the SB2 two-channel gating. Every RV
summation site (all `_rv_ll_*` variants, `rv_predictions`, and the
plotting model grid) funnels the per-block reflex through this so the
component physics can never drift between loops.

For a normal single-star fit every point has `comp == 1` and every
block has `cB == 0`, so this reduces to `KA * geom` — a bit-for-bit
no-op relative to the pre-SB2 code path.
"""
@inline _comp_rv(geom, comp::Integer, KA, cB) = (comp == 1 ? KA : cB) * geom

# A block subject to the planet-only hard priors — period ordering,
# dynamical stability, the per-planet eccentricity prior. The SB2 binary
# is EXCLUDED: a stellar companion is not label-interchangeable with the
# circumprimary planets, doesn't obey a planet ecc prior, and its
# planet–planet "stability" is not the relevant regime. It also has no
# single `K` (planet_K throws), so these gates must skip it.
@inline _is_ordered_planet(block::PlanetBlock) = has_K(block) && !is_sb2(block)

"""
    rv_log_likelihood(theta::Theta{T}, data::Data) -> T

Gaussian log-likelihood of RV observations with composable noise models.
Includes stability check and external priors (not Enzyme-safe).
"""
function rv_log_likelihood(theta::Theta{T}, data::Data) where {T}
    ll = _rv_log_likelihood_core(theta, data)
    isfinite(ll) || return ll

    # --- Period ordering hard prior (canonical labeling) -------------
    # Fixed-dim: global P_k1 < P_k2 < ... (Brewer & Donovan 2015, Faria+
    # 2018 KIMA). Trans-dim: per MODE-GROUP over ACTIVE slots, paired
    # with `_sort_group_periods!` insertion births — see the ws-variant
    # for the full rationale.
    p_idx = planet_indices(theta)
    # Planet-only gates count/iterate real planets — the SB2 binary is
    # excluded via `_is_ordered_planet`.
    n_rv_planets = count(k -> _is_ordered_planet(theta.params.layout.planet_blocks[k]), p_idx)
    if theta.td === nothing
        if n_rv_planets >= 2
            prev_P = -Inf
            for k in p_idx
                _is_ordered_planet(theta.params.layout.planet_blocks[k]) || continue
                P_k = planet_P(theta, k)
                P_k > prev_P || return convert(T, -Inf)
                prev_P = P_k
            end
        end
    elseif theta.td.n_planets_active >= 2
        modes = theta.params.config.planet_modes
        for (ii, k) in enumerate(p_idx)
            _is_ordered_planet(theta.params.layout.planet_blocks[k]) || continue
            P_k = planet_P(theta, k)
            for jj in 1:(ii - 1)
                j = p_idx[jj]
                _is_ordered_planet(theta.params.layout.planet_blocks[j]) || continue
                modes[j] == modes[k] || continue
                planet_P(theta, j) < P_k || return convert(T, -Inf)
            end
        end
    end

    # --- Stability hard prior (uses sortperm — not Enzyme-safe) ------
    stab = theta.params.config.stability
    if n_rv_planets >= 2 && stab !== :none && !isnan(theta.params.config.M_s)
        Ps = [planet_P(theta, k) for k in p_idx if _is_ordered_planet(theta.params.layout.planet_blocks[k])]
        Ks = [planet_K(theta, k) for k in p_idx if _is_ordered_planet(theta.params.layout.planet_blocks[k])]
        # `T[]` (not `Float64[]`) so `push!` accepts ForwardDiff.Dual
        # values when the chain is being differentiated by Pathfinder /
        # NUTS. With T = Float64 in the trans-dim PT path this is the
        # same shape as before.
        es_arr = T[]
        for k in p_idx
            _is_ordered_planet(theta.params.layout.planet_blocks[k]) || continue
            e, _ = planet_e_w(theta, k)
            push!(es_arr, e)
        end
        if !check_stability(Ps, Ks, es_arr, theta.params.config.M_s, stab)
            return convert(T, -Inf)
        end
    end

    # --- External priors (dynamic dispatch — not Enzyme-safe) --------
    lp_ext = zero(T)
    for ep in theta.params.config.external_priors
        if ep.quantity === :ecc && ep.per_planet
            for k in p_idx
                _is_ordered_planet(theta.params.layout.planet_blocks[k]) || continue
                e, _ = planet_e_w(theta, k)
                lp_ext += logpdf(ep.prior, e)
            end
        elseif ep.quantity === :rho_s && !ep.per_planet
            lp_ext += logpdf(ep.prior, rho_s(theta))
        end
        isfinite(lp_ext) || return convert(T, -Inf)
    end

    # --- Astrometry log-likelihoods (relative + HGCA) ----------------
    # No-op when data.relastrom and data.hgca are both nothing.
    ll_astrom = astrom_log_likelihood(theta, data)
    isfinite(ll_astrom) || return convert(T, -Inf)

    # Indicator-floor (no-op unless an IndicatorFloor noise model is active
    # — i.e. only in a trans-dim noise selection that includes ActivityGP).
    ll_ifloor = indicator_floor_log_likelihood(theta, data)
    isfinite(ll_ifloor) || return convert(T, -Inf)

    return ll + lp_ext + ll_astrom + ll_ifloor
end

"""
    _rv_log_likelihood_core(theta::Theta{T}, data::Data) -> T

Core RV log-likelihood (Enzyme-safe). No stability check, no external priors.
Called directly by `_logdensity_for_enzyme`.
"""
# Enzyme-safe entry: checks for noise models, dispatches accordingly.
function _rv_log_likelihood_core(theta::Theta{T}, data::Data) where {T}
    nm = theta.params.config.noise_models
    if isempty(nm)
        return _rv_ll_no_noise(theta, data)
    else
        return _rv_ll_with_noise(theta, data, nm)
    end
end

# RV log-likelihood without noise models. Enzyme-safe: no abstract iteration.
function _rv_ll_no_noise(theta::Theta{T}, data::Data) where {T}
    parametrization = theta.params.config.parametrization
    kplanet = n_p(theta)
    t_ref   = data.t_ref
    n_obs   = length(data.t_rv)
    two_pi  = T(2π)

    # --- L6: Decode all planets ONCE ---------------------------------
    p_idx = planet_indices(theta)
    n_rv_planets = 0
    for k in p_idx
        if has_K(theta.params.layout.planet_blocks[k])
            n_rv_planets += 1
        end
    end

    Ps  = Vector{T}(undef, n_rv_planets)
    Ks  = Vector{T}(undef, n_rv_planets)   # amplitude seen by component A (K, or K_A)
    cBs = Vector{T}(undef, n_rv_planets)   # component-B coefficient (0, or −K_B)
    es  = Vector{T}(undef, n_rv_planets)
    ws  = Vector{T}(undef, n_rv_planets)
    Tps = Vector{T}(undef, n_rv_planets)

    j = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_K(block) || continue
        j += 1
        Ps[j]  = planet_P(theta, k)
        if block isa SB2Block
            Ks[j]  = planet_K_A(theta, k)
            cBs[j] = -planet_K_B(theta, k)
        else
            Ks[j]  = planet_K(theta, k)
            cBs[j] = zero(T)
        end
        e, w   = planet_e_w(theta, k)
        (e < 0 || e >= 1) && return convert(T, -Inf)
        es[j]  = e
        ws[j]  = w
        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tps[j] = t_ref - ta * Ps[j] / two_pi
        elseif parametrization.time === :Tp
            Tps[j] = ta
        else  # :Tc
            Tps[j] = tc_to_tp(ta, Ps[j], e, w)
        end
    end

    # --- Decode trend (once) -----------------------------------------
    trend_order = theta.params.config.trend_order
    trend_dvdt = trend_order >= 1 ? rv_dvdt(theta) : zero(T)
    trend_curv = trend_order >= 2 ? rv_d2vdt2(theta) : zero(T)

    # --- Decode Rossiter-McLaughlin state (once) ---------------------
    n_rm, rm_state = _decode_rm_state(theta, p_idx, Ps)
    n_rm == -1 && return convert(T, -Inf)   # RM enabled but M_s/R_s missing

    # --- Stage 1: Build predictions + variances -----------------------
    # When γ-marginalization is on, the systemic offset γ is integrated
    # analytically (see `_rv_ll_gamma_marginalized`), so it is OMITTED
    # from the per-point mean here.
    marg_γ = parametrization.marginalize_gamma
    predictions = Vector{T}(undef, n_obs)
    variances   = Vector{T}(undef, n_obs)

    @inbounds for i in 1:n_obs
        t       = data.t_rv[i]
        obs_err = data.rv_err[i]
        ins_idx = data.rv_inst[i]

        dt_i  = t - t_ref
        pred  = trend_dvdt * dt_i + trend_curv * dt_i * dt_i
        marg_γ || (pred += rv_gamma(theta, ins_idx))

        # No noise models in this path — skip activity decorrelation

        # Keplerian contributions (SB2-gated via _comp_rv: primary sees
        # K_A / planet K; secondary sees −K_B / nothing)
        comp = data.rv_comp[i]
        for j in 1:n_rv_planets
            M = two_pi * (t - Tps[j]) / Ps[j]
            E = kepler_solve(M, es[j])
            f = true_anomaly(E, es[j])
            geom = cos(f + ws[j]) + es[j] * cos(ws[j])
            pred += _comp_rv(geom, comp, Ks[j], cBs[j])
        end

        # Rossiter-McLaughlin during in-transit windows — component A only
        # (the RM anomaly lives in the primary's deblended lines)
        (n_rm > 0 && comp == 1) && (pred += rm_contribution(t, n_rm, rm_state,
                                                Ps, es, ws, Tps))

        sigma = rv_sigma(theta, ins_idx)
        predictions[i] = pred
        var_i = obs_err * obs_err + sigma * sigma

        # No noise models — no activity jitter

        variances[i] = var_i
    end

    # No noise: straight to white-noise likelihood
    residuals = Vector{T}(undef, n_obs)
    @inbounds for i in 1:n_obs
        residuals[i] = data.rv[i] - predictions[i]
    end

    if marg_γ
        return _rv_ll_gamma_marginalized(residuals, variances, data.rv_inst,
                                          theta.params.layout.systemic.rv_gamma,
                                          n_obs, two_pi)
    end

    return _white_noise_ll(residuals, variances, two_pi)
end

# =====================================================================
# Workspace-aware paths (zero-allocation for PT hot loop)
# =====================================================================

# Forward-declare the workspace type (defined in samplers/rjmcmc.jl,
# included after this file).  We only need the name for dispatch; the
# struct fields are accessed generically.
# NOTE: PTWorkspace is defined later in the include order.  Julia
# allows adding methods to a function after the struct is defined as
# long as the method is called after both are loaded.  We therefore
# gate these methods behind `@eval` at module init time — but that is
# ugly.  Instead, we just forward-reference the struct: the method
# bodies will be compiled when first called, at which point the struct
# is fully defined.

"""
    _rv_ll_no_noise(theta, data, ws::PTWorkspace) -> T

Workspace-aware RV log-likelihood without noise models.  Reuses
pre-allocated buffers from `ws` instead of heap-allocating per call.
Functionally identical to `_rv_ll_no_noise(theta, data)`.
"""
function _rv_ll_no_noise(theta::Theta{T}, data::Data, ws) where {T}
    parametrization = theta.params.config.parametrization
    t_ref   = data.t_ref
    n_obs   = length(data.t_rv)
    two_pi  = T(2π)

    # --- Decode all planets ONCE (into pre-allocated buffers) --------
    p_idx = planet_indices(theta)
    n_rv_planets = 0
    for k in p_idx
        if has_K(theta.params.layout.planet_blocks[k])
            n_rv_planets += 1
        end
    end

    Ps  = ws.planet_Ps
    Ks  = ws.planet_Ks
    cBs = ws.planet_cBs   # component-B coefficient (0, or −K_B for SB2)
    es  = ws.planet_es
    ws_buf = ws.planet_ws
    Tps = ws.planet_Tps

    j = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_K(block) || continue
        j += 1
        Ps[j]  = planet_P(theta, k)
        if block isa SB2Block
            Ks[j]  = planet_K_A(theta, k)
            cBs[j] = -planet_K_B(theta, k)
        else
            Ks[j]  = planet_K(theta, k)
            cBs[j] = zero(T)
        end
        e, w   = planet_e_w(theta, k)
        (e < 0 || e >= 1) && return convert(T, -Inf)
        es[j]  = e
        ws_buf[j] = w
        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tps[j] = t_ref - ta * Ps[j] / two_pi
        elseif parametrization.time === :Tp
            Tps[j] = ta
        else  # :Tc
            Tps[j] = tc_to_tp(ta, Ps[j], e, w)
        end
    end

    # --- Per-planet RV-velocity cache. ------------------------------
    # Refresh stale rows only. For each planet, hoist the orbit-
    # constant precomputes (sin/cos(ω), e·cos(ω), 2π/P, √(1-e²)) out of
    # the per-time loop, and use the half-angle identity to skip the
    # atan in true_anomaly + the second cos(f+ω) call:
    #   cos(f+ω) = cos f · cos ω − sin f · sin ω,
    #   cos f = (cos E − e) / (1 − e cos E),
    #   sin f = √(1−e²) · sin E / (1 − e cos E).
    # Saves ~70 ns/point/planet on cache miss; 100% saved on cache hit.
    rv_cache = ws.rv_vel_cache
    rv_hash  = ws.rv_vel_hash
    @inbounds for j in 1:n_rv_planets
        Pj, Kj, ej, ωj, Tpj = Ps[j], Ks[j], es[j], ws_buf[j], Tps[j]
        cBj = cBs[j]
        # cBj MUST enter the hash: for an SB2 block the cached per-point
        # contribution depends on K_B (via cBj = −K_B), so a K_B-only move
        # has to invalidate the row. `data.rv_comp` is a fixed data
        # property (never changes between calls) so it stays out of the key.
        h = hash(Pj, hash(Kj, hash(cBj, hash(ej, hash(ωj, hash(Tpj))))))
        if rv_hash[j] != h
            one_minus_e2  = 1 - ej * ej
            sqrt_1_me2    = sqrt(one_minus_e2)
            sinω, cosω    = sincos(ωj)
            e_cos_ω       = ej * cosω
            two_pi_over_P = two_pi / Pj
            for i in 1:n_obs
                t = data.t_rv[i]
                M = two_pi_over_P * (t - Tpj)
                E = kepler_solve(M, ej)
                sinE, cosE = sincos(E)
                denom = 1 - ej * cosE
                cosf  = (cosE - ej) / denom
                sinf  = sqrt_1_me2 * sinE / denom
                cos_fpω = cosf * cosω - sinf * sinω
                # Bake the component sign/amplitude into the cache HERE — the
                # accumulation loop below just sums rows, so the physics must
                # be resolved at fill time.
                rv_cache[j, i] = _comp_rv(cos_fpω + e_cos_ω, data.rv_comp[i], Kj, cBj)
            end
            rv_hash[j] = h
        end
    end

    # --- Decode trend (once) -----------------------------------------
    trend_order = theta.params.config.trend_order
    trend_dvdt = trend_order >= 1 ? rv_dvdt(theta) : zero(T)
    trend_curv = trend_order >= 2 ? rv_d2vdt2(theta) : zero(T)

    # --- Decode RM state (once) --------------------------------------
    n_rm, rm_state = _decode_rm_state(theta, p_idx, Ps)
    n_rm == -1 && return convert(T, -Inf)

    # --- Stage 1: Build predictions + variances (reuse buffers) ------
    # γ-marginalization: omit the systemic offset from the per-point
    # mean (integrated analytically in `_rv_ll_gamma_marginalized`).
    marg_γ = parametrization.marginalize_gamma
    predictions = ws.predictions
    variances   = ws.variances

    @inbounds for i in 1:n_obs
        t       = data.t_rv[i]
        obs_err = data.rv_err[i]
        ins_idx = data.rv_inst[i]

        dt_i  = t - t_ref
        pred  = trend_dvdt * dt_i + trend_curv * dt_i * dt_i
        marg_γ || (pred += rv_gamma(theta, ins_idx))

        # Cache rows already carry the SB2 component gating (baked at fill).
        for j in 1:n_rv_planets
            pred += rv_cache[j, i]
        end

        # RM — component A only.
        (n_rm > 0 && data.rv_comp[i] == 1) &&
            (pred += rm_contribution(t, n_rm, rm_state,
                                     Ps, es, view(ws_buf, 1:n_rv_planets),
                                     Tps))

        sigma = rv_sigma(theta, ins_idx)
        predictions[i] = pred
        var_i = obs_err * obs_err + sigma * sigma
        variances[i] = var_i
    end

    # No noise: straight to white-noise likelihood (reuse residuals buffer)
    residuals = ws.residuals
    @inbounds for i in 1:n_obs
        residuals[i] = data.rv[i] - predictions[i]
    end

    if marg_γ
        return _rv_ll_gamma_marginalized(view(residuals, 1:n_obs),
                                          view(variances, 1:n_obs),
                                          data.rv_inst,
                                          theta.params.layout.systemic.rv_gamma,
                                          n_obs, two_pi)
    end

    return _white_noise_ll_views(residuals, variances, n_obs, two_pi)
end

"""
    _rv_ll_with_noise(theta, data, noise_models, ws) -> T

Workspace-aware RV log-likelihood WITH noise models.  Reuses
pre-allocated buffers from `ws`.
"""
function _rv_ll_with_noise(theta::Theta{T}, data::Data,
                            noise_models::Vector{<:NoiseModel}, ws) where {T}
    parametrization = theta.params.config.parametrization
    t_ref = data.t_ref
    n_obs = length(data.t_rv)
    two_pi = T(2π)

    p_idx = planet_indices(theta)
    n_rv_planets = 0
    for k in p_idx
        has_K(theta.params.layout.planet_blocks[k]) && (n_rv_planets += 1)
    end

    Ps  = ws.planet_Ps
    Ks  = ws.planet_Ks
    cBs = ws.planet_cBs   # component-B coefficient (0, or −K_B for SB2)
    es  = ws.planet_es
    ws_buf = ws.planet_ws
    Tps = ws.planet_Tps

    j = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_K(block) || continue
        j += 1
        Ps[j] = planet_P(theta, k)
        if block isa SB2Block
            Ks[j] = planet_K_A(theta, k); cBs[j] = -planet_K_B(theta, k)
        else
            Ks[j] = planet_K(theta, k); cBs[j] = zero(T)
        end
        e, w = planet_e_w(theta, k)
        (e < 0 || e >= 1) && return convert(T, -Inf)
        es[j] = e; ws_buf[j] = w
        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tps[j] = t_ref - ta * Ps[j] / two_pi
        elseif parametrization.time === :Tp
            Tps[j] = ta
        else
            Tps[j] = tc_to_tp(ta, Ps[j], e, w)
        end
    end

    trend_order = theta.params.config.trend_order
    trend_dvdt = trend_order >= 1 ? rv_dvdt(theta) : zero(T)
    trend_curv = trend_order >= 2 ? rv_d2vdt2(theta) : zero(T)

    # --- RM state (once) ---------------------------------------------
    n_rm, rm_state = _decode_rm_state(theta, p_idx, Ps)
    n_rm == -1 && return convert(T, -Inf)

    predictions = ws.predictions
    variances = ws.variances

    @inbounds for i in 1:n_obs
        t = data.t_rv[i]; obs_err = data.rv_err[i]; ins_idx = data.rv_inst[i]
        gamma = rv_gamma(theta, ins_idx)
        dt_i = t - t_ref
        pred = gamma + trend_dvdt * dt_i + trend_curv * dt_i * dt_i

        for (nm_idx, nm) in enumerate(noise_models)
            is_noise_model_active(theta, nm_idx) || continue
            if nm isa ActivityDecorrelation
                pred = apply_activity_decorrelation(pred, theta, data, nm, ins_idx, i)
            end
        end

        comp = data.rv_comp[i]
        for jj in 1:n_rv_planets
            M = two_pi * (t - Tps[jj]) / Ps[jj]
            E = kepler_solve(M, es[jj])
            f = true_anomaly(E, es[jj])
            geom = cos(f + ws_buf[jj]) + es[jj] * cos(ws_buf[jj])
            pred += _comp_rv(geom, comp, Ks[jj], cBs[jj])
        end

        (n_rm > 0 && comp == 1) &&
            (pred += rm_contribution(t, n_rm, rm_state, Ps, es,
                                     view(ws_buf, 1:n_rv_planets), Tps))

        sigma = rv_sigma(theta, ins_idx)
        predictions[i] = pred
        var_i = obs_err * obs_err + sigma * sigma

        for (nm_idx, nm) in enumerate(noise_models)
            is_noise_model_active(theta, nm_idx) || continue
            if nm isa ActivityJitter
                var_i = apply_activity_jitter(obs_err * obs_err, theta, data, nm, ins_idx, i)
            elseif nm isa ErrorScale && _errorscale_covers(theta, nm, ins_idx)
                # Multiplicative error-scale REPLACES additive jitter: f²·σ_formal²
                # (whenever it covers this instrument — independent of the drawn f).
                f2 = error_scale_factor(theta, nm, ins_idx)
                var_i = f2 * obs_err * obs_err
            end
        end
        variances[i] = var_i
    end

    # Compute residuals in-place
    residuals = ws.residuals
    @inbounds for i in 1:n_obs
        residuals[i] = data.rv[i] - predictions[i]
    end

    return _apply_noise_and_eval(theta, data, predictions, residuals,
                                  variances, noise_models, two_pi)
end

"""
    _rv_log_likelihood_core(theta, data, ws) -> T

Workspace-aware core RV log-likelihood dispatcher.
"""
function _rv_log_likelihood_core(theta::Theta{T}, data::Data, ws) where {T}
    nm = theta.params.config.noise_models
    if isempty(nm)
        return _rv_ll_no_noise(theta, data, ws)
    else
        return _rv_ll_with_noise(theta, data, nm, ws)
    end
end

"""
    rv_log_likelihood(theta, data, ws) -> T

Workspace-aware top-level RV log-likelihood.  Includes stability
check and external priors, using pre-allocated buffers from `ws`.
"""
function rv_log_likelihood(theta::Theta{T}, data::Data, ws) where {T}
    ll = _rv_log_likelihood_core(theta, data, ws)
    isfinite(ll) || return ll

    # --- Period ordering hard prior (canonical labeling) -------------
    # Fixed-dim: global P_k1 < P_k2 < ... over RV planets (label-switching
    # breaker). Trans-dim: per MODE-GROUP ordering over ACTIVE slots —
    # births insert into period order (`_sort_group_periods!`), deaths
    # can't disorder, and this gate rejects within-model moves that
    # would, so the chain stays on one canonical labeling and the k!
    # permutation degeneracy is broken without inflating the evidence.
    # A naive global slot-order gate under trans-dim hard-rejected any
    # short-period planet born after longer ones (WASP-47 e after b+d:
    # every birth was -Inf regardless of fit; Np=4 pinned at 0.00).
    p_idx = planet_indices(theta)
    # `n_rv_buf` counts every has_K block — it matches the length of the
    # decode buffers filled by the core (SB2 binary INCLUDED). `n_planet_gate`
    # counts real planets only (SB2 excluded) and drives the planet-only
    # hard priors below.
    n_rv_buf = count(k -> has_K(theta.params.layout.planet_blocks[k]), p_idx)
    n_planet_gate = count(k -> _is_ordered_planet(theta.params.layout.planet_blocks[k]), p_idx)
    if theta.td === nothing
        if n_planet_gate >= 2
            prev_P = -Inf
            for k in p_idx
                _is_ordered_planet(theta.params.layout.planet_blocks[k]) || continue
                P_k = planet_P(theta, k)
                P_k > prev_P || return convert(T, -Inf)
                prev_P = P_k
            end
        end
    elseif theta.td.n_planets_active >= 2
        modes = theta.params.config.planet_modes
        for (ii, k) in enumerate(p_idx)
            _is_ordered_planet(theta.params.layout.planet_blocks[k]) || continue
            P_k = planet_P(theta, k)
            for jj in 1:(ii - 1)
                j = p_idx[jj]
                _is_ordered_planet(theta.params.layout.planet_blocks[j]) || continue
                modes[j] == modes[k] || continue
                planet_P(theta, j) < P_k || return convert(T, -Inf)
            end
        end
    end

    # --- Stability hard prior (workspace-aware) ----------------------
    stab = theta.params.config.stability
    if n_planet_gate >= 2 && stab !== :none && !isnan(theta.params.config.M_s)
        if n_planet_gate == n_rv_buf
            # Fast path (no SB2): the pre-filled buffers are all planets.
            Ps_v  = @view ws.planet_Ps[1:n_rv_buf]
            Ks_v  = @view ws.planet_Ks[1:n_rv_buf]
            es_v  = @view ws.planet_es[1:n_rv_buf]
            ord_v = @view ws.stab_order[1:n_rv_buf]
            mas_v = @view ws.stab_masses[1:n_rv_buf]
            sma_v = @view ws.stab_sma[1:n_rv_buf]
            if !check_stability(Ps_v, Ks_v, es_v, theta.params.config.M_s, stab,
                                ord_v, mas_v, sma_v)
                return convert(T, -Inf)
            end
        else
            # SB2 present: the buffers interleave the binary (with K_A in
            # the K slot) among the planets — filter it out into fresh
            # arrays. Rare path (stability + SB2), small alloc is fine.
            Ps_s = [planet_P(theta, k) for k in p_idx if _is_ordered_planet(theta.params.layout.planet_blocks[k])]
            Ks_s = [planet_K(theta, k) for k in p_idx if _is_ordered_planet(theta.params.layout.planet_blocks[k])]
            es_s = T[]
            for k in p_idx
                _is_ordered_planet(theta.params.layout.planet_blocks[k]) || continue
                e, _ = planet_e_w(theta, k)
                push!(es_s, e)
            end
            if !check_stability(Ps_s, Ks_s, es_s, theta.params.config.M_s, stab)
                return convert(T, -Inf)
            end
        end
    end

    # --- External priors (dynamic dispatch) --------------------------
    lp_ext = zero(T)
    for ep in theta.params.config.external_priors
        if ep.quantity === :ecc && ep.per_planet
            for k in p_idx
                _is_ordered_planet(theta.params.layout.planet_blocks[k]) || continue
                e, _ = planet_e_w(theta, k)
                lp_ext += logpdf(ep.prior, e)
            end
        elseif ep.quantity === :rho_s && !ep.per_planet
            lp_ext += logpdf(ep.prior, rho_s(theta))
        end
        isfinite(lp_ext) || return convert(T, -Inf)
    end

    # --- Astrometry log-likelihoods (relative + HGCA) ----------------
    # No-op when data.relastrom and data.hgca are both nothing.
    ll_astrom = astrom_log_likelihood(theta, data)
    isfinite(ll_astrom) || return convert(T, -Inf)

    # Indicator-floor (no-op unless an IndicatorFloor noise model is active
    # — i.e. only in a trans-dim noise selection that includes ActivityGP).
    ll_ifloor = indicator_floor_log_likelihood(theta, data)
    isfinite(ll_ifloor) || return convert(T, -Inf)

    return ll + lp_ext + ll_astrom + ll_ifloor
end

"""
    _white_noise_ll_views(residuals, variances, n, two_pi) -> T

White-noise log-likelihood operating on views/sub-arrays.
Uses an explicit length `n` rather than `eachindex`.
"""
function _white_noise_ll_views(residuals, variances, n::Int, two_pi)
    total = zero(eltype(residuals))
    @inbounds for i in 1:n
        total += -(log(two_pi * variances[i]) + residuals[i]^2 / variances[i]) / 2
    end
    return total
end


# RV log-likelihood WITH noise models. Not Enzyme-safe.
function _rv_ll_with_noise(theta::Theta{T}, data::Data,
                            noise_models::Vector{<:NoiseModel}) where {T}
    parametrization = theta.params.config.parametrization
    t_ref = data.t_ref
    n_obs = length(data.t_rv)
    two_pi = T(2π)

    p_idx = planet_indices(theta)
    n_rv_planets = 0
    for k in p_idx
        has_K(theta.params.layout.planet_blocks[k]) && (n_rv_planets += 1)
    end

    Ps  = Vector{T}(undef, n_rv_planets)
    Ks  = Vector{T}(undef, n_rv_planets)
    cBs = Vector{T}(undef, n_rv_planets)   # component-B coefficient (0, or −K_B for SB2)
    es  = Vector{T}(undef, n_rv_planets)
    ws  = Vector{T}(undef, n_rv_planets)
    Tps = Vector{T}(undef, n_rv_planets)

    j = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_K(block) || continue
        j += 1
        Ps[j] = planet_P(theta, k)
        if block isa SB2Block
            Ks[j] = planet_K_A(theta, k); cBs[j] = -planet_K_B(theta, k)
        else
            Ks[j] = planet_K(theta, k); cBs[j] = zero(T)
        end
        e, w = planet_e_w(theta, k)
        (e < 0 || e >= 1) && return convert(T, -Inf)
        es[j] = e; ws[j] = w
        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tps[j] = t_ref - ta * Ps[j] / two_pi
        elseif parametrization.time === :Tp
            Tps[j] = ta
        else
            Tps[j] = tc_to_tp(ta, Ps[j], e, w)
        end
    end

    trend_order = theta.params.config.trend_order
    trend_dvdt = trend_order >= 1 ? rv_dvdt(theta) : zero(T)
    trend_curv = trend_order >= 2 ? rv_d2vdt2(theta) : zero(T)

    # --- RM state (once) ---------------------------------------------
    n_rm, rm_state = _decode_rm_state(theta, p_idx, Ps)
    n_rm == -1 && return convert(T, -Inf)

    predictions = Vector{T}(undef, n_obs)
    variances = Vector{T}(undef, n_obs)

    @inbounds for i in 1:n_obs
        t = data.t_rv[i]; obs_err = data.rv_err[i]; ins_idx = data.rv_inst[i]
        gamma = rv_gamma(theta, ins_idx)
        dt_i = t - t_ref
        pred = gamma + trend_dvdt * dt_i + trend_curv * dt_i * dt_i

        for (nm_idx, nm) in enumerate(noise_models)
            is_noise_model_active(theta, nm_idx) || continue
            if nm isa ActivityDecorrelation
                pred = apply_activity_decorrelation(pred, theta, data, nm, ins_idx, i)
            end
        end

        comp = data.rv_comp[i]
        for jj in 1:n_rv_planets
            M = two_pi * (t - Tps[jj]) / Ps[jj]
            E = kepler_solve(M, es[jj])
            f = true_anomaly(E, es[jj])
            geom = cos(f + ws[jj]) + es[jj] * cos(ws[jj])
            pred += _comp_rv(geom, comp, Ks[jj], cBs[jj])
        end

        (n_rm > 0 && comp == 1) && (pred += rm_contribution(t, n_rm, rm_state,
                                                Ps, es, ws, Tps))

        sigma = rv_sigma(theta, ins_idx)
        predictions[i] = pred
        var_i = obs_err * obs_err + sigma * sigma

        for (nm_idx, nm) in enumerate(noise_models)
            is_noise_model_active(theta, nm_idx) || continue
            if nm isa ActivityJitter
                var_i = apply_activity_jitter(obs_err * obs_err, theta, data, nm, ins_idx, i)
            elseif nm isa ErrorScale && _errorscale_covers(theta, nm, ins_idx)
                # Multiplicative error-scale REPLACES additive jitter: f²·σ_formal²
                # (whenever it covers this instrument — independent of the drawn f).
                f2 = error_scale_factor(theta, nm, ins_idx)
                var_i = f2 * obs_err * obs_err
            end
        end
        variances[i] = var_i
    end

    return _apply_noise_and_eval(theta, data, predictions, residuals_from(data, predictions),
                                  variances, noise_models, two_pi)
end

function residuals_from(data::Data, predictions::Vector{T}) where {T}
    r = Vector{T}(undef, length(predictions))
    @inbounds for i in eachindex(r)
        r[i] = data.rv[i] - predictions[i]
    end
    return r
end

# Apply noise models and evaluate likelihood.
# RV channel only — phot-channel noise models (AR/MA/GP) are routed
# through `transit_log_likelihood`. AR/MA dispatch on `noise_channel`,
# so we only apply the `:rv`-channel ones here. GP routing (global vs
# per-instrument) is handled by `_eval_channel_likelihood`.
#
# When an `ActivityGP` (multivariate Rajpaul) is active, we route the
# RV+indicator joint covariance through a separate path that produces
# one joint Gaussian log-likelihood over all channels at once. The
# AR step (Stage-1 prediction adjustment) still runs in that case;
# MA / single-channel GP do not (composition rule documented in
# `validate_noise_models`).
function _apply_noise_and_eval(theta::Theta{T}, data, predictions, residuals,
                                variances, noise_models, two_pi) where {T}
    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa ARModel && noise_channel(nm) === :rv
            apply_ar!(predictions, data.t_rv, data.rv_inst, theta, nm)
        end
    end

    @inbounds for i in eachindex(residuals)
        residuals[i] = data.rv[i] - predictions[i]
    end

    # Multivariate-GP (Rajpaul) routing. Two cases supported:
    #
    # (1) One active ActivityGP with `instruments = []` (global) →
    #     joint LL over ALL RV + the configured indicators.
    # (2) Any number of active ActivityGPs with non-empty, pairwise-
    #     disjoint `instruments` lists → per-instrument-group joint LL,
    #     summed. RV instruments NOT covered by any AGP fall back to
    #     the standard white-noise / celerite path.
    agp_list = _active_activity_gps(theta, noise_models)
    # indicators_only AGPs score ONLY the indicator block log p(y_I|θ);
    # the RV path continues below (white noise / celerite) as if no GP
    # were registered, and the indicator term is ADDED to whichever
    # branch returns. Used for the chain-rule evidence
    # log Z_cond = log Z(joint) − log Z(indicators_only).
    ind_only_ll = zero(T)
    if !isempty(agp_list)
        kept = ActivityGP[]
        for agp in agp_list
            if agp.indicators_only
                lli = _activity_gp_joint_ll(theta, data, predictions,
                                              variances, agp)
                isfinite(lli) || return convert(T, -Inf)
                ind_only_ll += lli
            else
                push!(kept, agp)
            end
        end
        agp_list = kept
    end
    if !isempty(agp_list)
        # The ActivityGP joint path does not compose with AdditiveCovariance
        # (NightlyOffset / HarmonicBlock) — those terms would be silently
        # dropped. Fail loud instead (−Inf) so no state is silently wrong; the
        # default menu excludes this combo, so it only bites hand-built configs.
        for (j, nm2) in enumerate(noise_models)
            nm2 isa AdditiveCovariance && is_noise_model_active(theta, j) &&
                return convert(T, -Inf)
        end
        # Detect mode.
        global_agp = nothing
        scoped_agps = ActivityGP[]
        for agp in agp_list
            if isempty(agp.instruments)
                global_agp = agp
            else
                push!(scoped_agps, agp)
            end
        end
        if global_agp !== nothing && isempty(scoped_agps)
            return ind_only_ll +
                   _activity_gp_joint_ll(theta, data, predictions,
                                            variances, global_agp)
        elseif global_agp === nothing && !isempty(scoped_agps)
            return ind_only_ll +
                   _activity_gp_scoped_ll(theta, data, predictions,
                                            residuals, variances, scoped_agps,
                                            noise_models, two_pi)
        else
            # Both global AND scoped AGP active — composition rule
            # already rejected by `validate_noise_models`, but be safe.
            return convert(T, -Inf)
        end
    end

    for (nm_idx, nm) in enumerate(noise_models)
        is_noise_model_active(theta, nm_idx) || continue
        if nm isa MAModel && noise_channel(nm) === :rv
            apply_ma!(residuals, data.t_rv, data.rv_inst, theta, nm)
        end
    end

    return ind_only_ll +
           _eval_channel_likelihood(theta, residuals, variances, data.t_rv,
                                     data.rv_inst, :rv, two_pi)
end

# Return the first active `ActivityGP` in `noise_models`, or `nothing`.
@inline function _active_activity_gp(theta::Theta, noise_models)
    for (nm_idx, nm) in enumerate(noise_models)
        nm isa ActivityGP || continue
        is_noise_model_active(theta, nm_idx) || continue
        return nm
    end
    return nothing
end

# All active ActivityGPs.
function _active_activity_gps(theta::Theta, noise_models)
    out = ActivityGP[]
    for (nm_idx, nm) in enumerate(noise_models)
        nm isa ActivityGP || continue
        is_noise_model_active(theta, nm_idx) || continue
        push!(out, nm)
    end
    return out
end

# Indicator-floor white likelihood for a trans-dim noise SELECTION that
# includes ActivityGP (see the IndicatorFloor docstring). Scores each
# floor channel as iid N(0, σ_floor² + err²), SKIPPING channels already
# covered by an active joint ActivityGP (the AGP scores those). Returns
# 0 when no active IndicatorFloor is configured — i.e. for every run
# without an AGP-toggle, so this is a no-op on the entire existing suite.
# Quasi-periodic GP log-likelihood for ONE indicator channel (dense Cholesky,
# mirrors ActivityGP's k_GG block — VALUE term only, no derivative coupling).
#   k(τ) = amp²·exp(−τ²/2λe² − sin²(πτ/P)/2λp²),  diag += err² + jit²
# y must be ~zero-mean (indicators are standardized upstream). Generic over T.
function _qp_floor_channel_loglike(t::Vector{Float64}, y::AbstractVector{T},
                                    errs, amp::T, jit::T,
                                    P::T, λ_e::T, λ_p::T) where {T}
    n = length(t)
    amp2     = amp * amp
    inv_2λe2 = 1 / (2 * λ_e * λ_e)
    inv_2λp2 = 1 / (2 * λ_p * λ_p)
    π_P      = T(π) / P
    Σ = Matrix{T}(undef, n, n)
    @inbounds for j in 1:n
        tj = t[j]
        for i in 1:j
            τ = t[i] - tj
            s = sin(π_P * τ)
            val = amp2 * exp(-τ * τ * inv_2λe2 - s * s * inv_2λp2)
            Σ[i, j] = val
            i == j || (Σ[j, i] = val)
        end
    end
    jit2 = jit * jit
    @inbounds for i in 1:n
        e2 = errs === nothing ? zero(T) : convert(T, errs[i])^2
        Σ[i, i] += e2 + jit2
    end
    F = cholesky(Symmetric(Σ); check = false)
    issuccess(F) || return convert(T, -Inf)
    α = F \ y
    return -0.5 * (dot(y, α) + logdet(F) + n * log(2π))
end

function indicator_floor_log_likelihood(theta::Theta{T}, data::Data) where {T}
    nm_list = theta.params.config.noise_models
    floor = nothing
    @inbounds for i in eachindex(nm_list)
        nm = nm_list[i]
        nm isa IndicatorFloor || continue
        is_noise_model_active(theta, i) || continue
        floor = nm; break
    end
    floor === nothing && return zero(T)
    covered = Set{Symbol}()
    @inbounds for i in eachindex(nm_list)
        nm = nm_list[i]
        nm isa ActivityGP || continue
        nm.indicators_only && continue
        is_noise_model_active(theta, i) || continue
        for ch in nm.channels
            ch === :rv || push!(covered, ch)
        end
    end
    layout = theta.params.layout
    total = zero(T)

    if floor.kernel === :qp
        # Quasi-periodic GP floor: shared kernel hyperparams + per-channel
        # amplitude/jitter. Gives the floor the same correlation-capturing
        # power as AGP's indicator block (see IndicatorFloor docstring).
        P   = theta.values[get(layout.name_to_idx, "ind_floor_period", 0)]
        λ_e = theta.values[get(layout.name_to_idx, "ind_floor_lambda_e", 0)]
        λ_p = theta.values[get(layout.name_to_idx, "ind_floor_lambda_p", 0)]
        (P > 0 && λ_e > 0 && λ_p > 0) || return convert(T, -Inf)
        t = data.t_rv isa Vector{Float64} ? data.t_rv : Float64.(data.t_rv)
        for ch in floor.channels
            ch in covered && continue
            name = String(ch)
            haskey(data.indicators, name) || continue
            ai = get(layout.name_to_idx, "ind_floor_$(ch)_amp", 0)
            ji = get(layout.name_to_idx, "ind_floor_$(ch)_jit", 0)
            (ai == 0 || ji == 0) && continue
            amp = theta.values[ai]; jit = theta.values[ji]
            (amp > 0 && jit > 0) || return convert(T, -Inf)
            errs = get(data.indicator_errs, name, nothing)
            yv = T[convert(T, v) for v in data.indicators[name]]
            ll = _qp_floor_channel_loglike(t, yv, errs, amp, jit, P, λ_e, λ_p)
            isfinite(ll) || return convert(T, -Inf)
            total += ll
        end
        return total
    end

    for ch in floor.channels
        ch in covered && continue
        name = String(ch)
        haskey(data.indicators, name) || continue
        vals = data.indicators[name]
        errs = get(data.indicator_errs, name, nothing)
        idx = get(layout.name_to_idx, "ind_floor_$(ch)", 0)
        idx == 0 && continue
        σf = theta.values[idx]
        σf > 0 || return convert(T, -Inf)
        σf2 = σf * σf
        @inbounds for k in eachindex(vals)
            e2 = errs === nothing ? zero(T) : convert(T, errs[k])^2
            v2 = σf2 + e2
            total += -0.5 * (convert(T, vals[k])^2 / v2 + log(2π * v2))
        end
    end
    return total
end

# Per-instrument-scoped ActivityGP path. Each AGP covers an instrument
# subset; remaining RV obs go through the standard channel-likelihood
# (white noise / celerite). Total log L is the sum.
function _activity_gp_scoped_ll(theta::Theta{T}, data::Data,
                                  predictions::Vector{T},
                                  residuals::Vector{T},
                                  variances::Vector{T},
                                  agps::Vector{ActivityGP},
                                  noise_models, two_pi) where {T}
    inst_names = theta.params.config.instruments.rv_names
    n_rv_obs = length(data.t_rv)

    total = zero(T)
    covered = falses(n_rv_obs)

    for agp in agps
        # Map instrument names → integer ids.
        inst_ids = Int[]
        for name in agp.instruments
            k = findfirst(==(name), inst_names)
            k === nothing && continue
            push!(inst_ids, k)
        end
        isempty(inst_ids) && continue

        # Build the masked subset of RV observations.
        mask = falses(n_rv_obs)
        @inbounds for i in 1:n_rv_obs
            if data.rv_inst[i] in inst_ids
                mask[i] = true
                covered[i] && return convert(T, -Inf)  # overlap not allowed
                covered[i] = true
            end
        end
        any(mask) || continue

        ll = _activity_gp_joint_ll_scoped(theta, data, predictions,
                                            variances, agp, mask)
        isfinite(ll) || return convert(T, -Inf)
        total += ll
    end

    # Uncovered RV obs → standard channel likelihood. MA still applies
    # to those residuals, celerite GP routing is handled inside
    # `_eval_channel_likelihood` (and any global GP is already
    # forbidden when scoped AGPs are present, so MA is the only Stage-2
    # noise that could be active here).
    if !all(covered)
        for (nm_idx, nm) in enumerate(noise_models)
            is_noise_model_active(theta, nm_idx) || continue
            if nm isa MAModel && noise_channel(nm) === :rv
                apply_ma!(residuals, data.t_rv, data.rv_inst, theta, nm)
            end
        end
        uncov_idx = findall(.!covered)
        total += _eval_channel_likelihood(theta,
                                            view(residuals, uncov_idx),
                                            view(variances, uncov_idx),
                                            view(data.t_rv, uncov_idx),
                                            view(data.rv_inst, uncov_idx),
                                            :rv, two_pi)
    end

    return total
end

# Joint Rajpaul log-LL over a SUBSET of RV obs (defined by `mask`) +
# the corresponding indicator subset (indicators are stored per-RV-
# time so the mask transfers directly). Mirrors `_activity_gp_joint_ll`
# but allocates the joint matrices on the masked subset only.
function _activity_gp_joint_ll_scoped(theta::Theta{T}, data::Data,
                                        predictions::Vector{T},
                                        variances::Vector{T},
                                        agp::ActivityGP,
                                        mask::AbstractVector{Bool}) where {T}
    layout = theta.params.layout
    s = _gp_suffix(agp)
    # Unit-variance G(t) (Rajpaul+ 2015): amp ≡ 1; the guarded lookup
    # only serves legacy layouts/chains that still carry gp_act_amp.
    amp_idx = get(layout.name_to_idx, "gp_act_amp$s", 0)
    amp = amp_idx == 0 ? one(T) : theta.values[amp_idx]
    P   = theta.values[layout.name_to_idx["gp_act_period$s"]]
    λe  = theta.values[layout.name_to_idx["gp_act_lambda_e$s"]]
    λp  = theta.values[layout.name_to_idx["gp_act_lambda_p$s"]]
    (amp > 0 && P > 0 && λe > 0 && λp > 0) || return convert(T, -Inf)

    # Derivative couplings are SAMPLED as amplitudes in the channel's
    # data units (unit-variance G ⇒ the G coefficient already is); the
    # physical Ġ coefficient is amplitude / std(Ġ). Sampling raw Ġ
    # coefficients rides the unpenalized (λe, λp, Vr) ridge: Var(Ġ)→0
    # as λ_p rails, Vr→∞ compensating at fixed RV power.
    inv_sdG = 1 / sqrt(1 / (λe * λe) + π * π / (P * P * λp * λp))

    Vc = theta.values[layout.name_to_idx["Vc$s"]]
    Vr = agp.use_derivative ?
         theta.values[layout.name_to_idx["Vr$s"]] * inv_sdG : zero(T)

    rv_idx = findall(mask)
    n_rv_obs = length(rv_idx)

    ind_meta = Tuple{Symbol, Vector{Float64}, Vector{Float64}, T, T, T}[]
    n_obs_total = n_rv_obs
    for ch in agp.channels
        ch === :rv && continue
        name = String(ch)
        haskey(data.indicators, name) || throw(ArgumentError(
            "ActivityGP requires data.indicators[\"$name\"] for channel :$ch"))
        haskey(data.indicator_errs, name) || throw(ArgumentError(
            "ActivityGP requires data.indicator_errs[\"$name\"] for channel :$ch"))
        vals = data.indicators[name]; errs = data.indicator_errs[name]
        # Data-model invariant: indicators are parallel to the RV data (one
        # value per RV epoch, Rajpaul-standard same-spectra measurement). The
        # `vals[rv_idx]` slice below — and the shared-epoch covariance — are
        # only valid then. Fail clearly instead of OOB / silently-wrong LL.
        length(vals) == length(data.rv) && length(errs) == length(data.rv) ||
            throw(ArgumentError(
                "ActivityGP indicator :$ch has $(length(vals)) values (errs " *
                "$(length(errs))) but there are $(length(data.rv)) RV " *
                "observations — indicators must be parallel to the RV data " *
                "(one value per RV epoch)."))
        cg, cd = _ACTIVITY_GP_COEFFS[ch]
        a_coef = theta.values[layout.name_to_idx[string(cg, s)]]
        b_coef = (cd === nothing || !agp.use_derivative) ? zero(T) :
                  theta.values[layout.name_to_idx[string(cd, s)]] * inv_sdG
        # Fitted per-channel jitter floor on Σ_II's diagonal (guarded
        # lookup: legacy layouts without jit params evaluate at 0).
        jit_idx = get(layout.name_to_idx, "gp_act_jit_$(ch)$s", 0)
        jit² = jit_idx == 0 ? zero(T) : theta.values[jit_idx]^2
        push!(ind_meta, (ch, vals[rv_idx], errs[rv_idx], a_coef, b_coef, jit²))
        n_obs_total += n_rv_obs
    end

    # Only y / σ² are needed downstream (the solve + diagonal noise); the
    # covariance comes from the block-factored builder, not a flat per-point
    # array, so t/a/b/channel flats are no longer built here.
    y_flat  = Vector{T}(undef, n_obs_total)
    σ²_flat = Vector{T}(undef, n_obs_total)
    @inbounds for (k, i) in enumerate(rv_idx)
        y_flat[k]  = data.rv[i] - predictions[i]
        σ²_flat[k] = variances[i]
    end
    offset = n_rv_obs
    for (ch, vals, errs, a_coef, b_coef, jit²) in ind_meta
        @inbounds for k in 1:n_rv_obs
            y_flat[offset + k]  = convert(T, vals[k])
            σ²_flat[offset + k] = convert(T, errs[k]^2) + jit²
        end
        offset += n_rv_obs
    end

    # Block-factored assembly: every channel shares the RV epochs (the
    # conditional path places indicators at data.t_rv[rv_idx]) and has a
    # per-channel-constant (a, b), so the kernel transcendentals are evaluated
    # once on the N×N epoch grid instead of C²× over the flat (N·C)² pairs.
    # Numerically identical to the dense builder (machine precision).
    chan_a = T[Vc, (m[4] for m in ind_meta)...]
    chan_b = T[Vr, (m[5] for m in ind_meta)...]
    Σ = activity_gp_covariance_blocked(view(data.t_rv, rv_idx),
                                        chan_a, chan_b, amp, P, λe, λp)
    @inbounds for i in 1:n_obs_total
        Σ[i, i] += σ²_flat[i]
    end
    if agp.marginalize_indicators && n_obs_total > n_rv_obs
        Σ_RR = view(Σ, 1:n_rv_obs, 1:n_rv_obs)
        Σ_RI = view(Σ, 1:n_rv_obs, (n_rv_obs + 1):n_obs_total)
        Σ_II = view(Σ, (n_rv_obs + 1):n_obs_total, (n_rv_obs + 1):n_obs_total)
        y_R  = view(y_flat, 1:n_rv_obs)
        y_I  = view(y_flat, (n_rv_obs + 1):n_obs_total)
        F_II = cholesky(Symmetric(Matrix(Σ_II)); check = false)
        issuccess(F_II) || return convert(T, -Inf)
        μ_cond = Σ_RI * (F_II \ y_I)
        Σ_cond = Symmetric(Matrix(Σ_RR) .- Σ_RI * (F_II \ Matrix(transpose(Σ_RI))))
        F_R = cholesky(Σ_cond; check = false)
        issuccess(F_R) || return convert(T, -Inf)
        r = y_R .- μ_cond
        return convert(T,
            -0.5 * (dot(r, F_R \ r) + logdet(F_R) +
                     n_rv_obs * log(2π)))
    end
    F = cholesky(Symmetric(Σ); check = false)
    issuccess(F) || return convert(T, -Inf)
    α = F \ y_flat
    return convert(T,
        -0.5 * (dot(y_flat, α) + logdet(F) + n_obs_total * log(2π)))
end

# Joint Rajpaul-GP log-likelihood across RV and the indicator channels
# declared by `agp.channels`. RV residuals come from the caller
# (already AR-corrected); indicator residuals are taken to be the
# indicator values themselves (Rajpaul-standard: indicators are
# centred per-instrument upstream of the fit). Variances are the
# per-point measurement uncertainties from `data.rv_err` / instrument
# jitter (RV) and `data.indicator_errs` (indicators).
function _activity_gp_joint_ll(theta::Theta{T}, data::Data,
                                 predictions::Vector{T},
                                 variances::Vector{T},
                                 agp::ActivityGP) where {T}
    layout = theta.params.layout
    s = _gp_suffix(agp)

    # Unit-variance G(t) (Rajpaul+ 2015); guarded lookup for legacy chains.
    amp_idx = get(layout.name_to_idx, "gp_act_amp$s", 0)
    amp = amp_idx == 0 ? one(T) : theta.values[amp_idx]
    P   = theta.values[layout.name_to_idx["gp_act_period$s"]]
    λe  = theta.values[layout.name_to_idx["gp_act_lambda_e$s"]]
    λp  = theta.values[layout.name_to_idx["gp_act_lambda_p$s"]]
    (amp > 0 && P > 0 && λe > 0 && λp > 0) || return convert(T, -Inf)

    # Derivative couplings sampled as amplitudes (see the scoped twin):
    # physical Ġ coefficient = amplitude / std(Ġ).
    inv_sdG = 1 / sqrt(1 / (λe * λe) + π * π / (P * P * λp * λp))

    # indicators_only mode scores no RV channel — couplings absent.
    Vc = agp.indicators_only ? zero(T) :
         theta.values[layout.name_to_idx["Vc$s"]]
    Vr = (agp.use_derivative && !agp.indicators_only) ?
         theta.values[layout.name_to_idx["Vr$s"]] * inv_sdG : zero(T)

    n_rv_obs = length(data.rv)

    # Indicator channels — verify data is present.
    ind_meta = Tuple{Symbol, Vector{Float64}, Vector{Float64}, T, T, T}[]
    n_total = n_rv_obs
    for ch in agp.channels
        ch === :rv && continue
        name = String(ch)
        haskey(data.indicators, name) || throw(ArgumentError(
            "ActivityGP requires data.indicators[\"$name\"] for channel :$ch"))
        haskey(data.indicator_errs, name) || throw(ArgumentError(
            "ActivityGP requires data.indicator_errs[\"$name\"] for channel :$ch"))
        vals = data.indicators[name]
        errs = data.indicator_errs[name]
        # Data-model invariant: indicators are parallel to the RV data (one
        # value per RV epoch). The joint path places indicator k at epoch
        # data.t_rv[k]; only valid when lengths match. Fail clearly rather
        # than OOB (n_ind > n_rv) or silently-wrong epochs (n_ind < n_rv).
        length(vals) == n_rv_obs && length(errs) == n_rv_obs ||
            throw(ArgumentError(
                "ActivityGP indicator :$ch has $(length(vals)) values (errs " *
                "$(length(errs))) but there are $n_rv_obs RV observations — " *
                "indicators must be parallel to the RV data (one value per " *
                "RV epoch)."))
        cg, cd = _ACTIVITY_GP_COEFFS[ch]
        a_coef = theta.values[layout.name_to_idx[string(cg, s)]]
        b_coef = (cd === nothing || !agp.use_derivative) ? zero(T) :
                  theta.values[layout.name_to_idx[string(cd, s)]] * inv_sdG
        jit_idx = get(layout.name_to_idx, "gp_act_jit_$(ch)$s", 0)
        jit² = jit_idx == 0 ? zero(T) : theta.values[jit_idx]^2
        push!(ind_meta, (ch, vals, errs, a_coef, b_coef, jit²))
        n_total += length(vals)
    end

    if agp.indicators_only
        # Score ONLY the indicator block: log p(y_I | θ) under the
        # latent-G Rajpaul covariance over the indicator channels. The
        # caller continues the standard RV path (white noise / celerite)
        # and ADDS this term — see the routing in rv_log_likelihood.
        # This is the second term of the chain-rule evidence
        #   log Z_cond = log Z(joint) − log Z(indicators_only),
        # the honest p(y_R | y_I) score for model comparison.
        n_ind_total = n_total - n_rv_obs
        n_ind_total > 0 || return zero(T)
        y_I  = Vector{T}(undef, n_ind_total)
        σ²_I = Vector{T}(undef, n_ind_total)
        off = 0
        for (ch, vals, errs, a_coef, b_coef, jit²) in ind_meta
            @inbounds for i in 1:length(vals)
                y_I[off + i]  = convert(T, vals[i])
                σ²_I[off + i] = convert(T, errs[i]^2) + jit²
            end
            off += length(vals)
        end
        chan_aI = T[m[4] for m in ind_meta]
        chan_bI = T[m[5] for m in ind_meta]
        Σ_I = activity_gp_covariance_blocked(view(data.t_rv, 1:n_rv_obs),
                                              chan_aI, chan_bI,
                                              amp, P, λe, λp)
        @inbounds for i in 1:n_ind_total
            Σ_I[i, i] += σ²_I[i]
        end
        F_I = cholesky(Symmetric(Σ_I); check = false)
        issuccess(F_I) || return convert(T, -Inf)
        αI = F_I \ y_I
        return convert(T, -0.5 * (dot(y_I, αI) + logdet(F_I) +
                                   n_ind_total * log(2π)))
    end

    t_flat       = Vector{Float64}(undef, n_total)
    y_flat       = Vector{T}(undef, n_total)
    σ²_flat      = Vector{T}(undef, n_total)
    a_flat       = Vector{T}(undef, n_total)
    b_flat       = Vector{T}(undef, n_total)
    channel_flat = Vector{Symbol}(undef, n_total)

    @inbounds for i in 1:n_rv_obs
        t_flat[i]       = data.t_rv[i]
        y_flat[i]       = data.rv[i] - predictions[i]
        σ²_flat[i]      = variances[i]
        a_flat[i]       = Vc
        b_flat[i]       = Vr
        channel_flat[i] = :rv
    end
    offset = n_rv_obs
    for (ch, vals, errs, a_coef, b_coef, jit²) in ind_meta
        n_ch = length(vals)
        @inbounds for i in 1:n_ch
            t_flat[offset + i]       = data.t_rv[i]
            y_flat[offset + i]       = convert(T, vals[i])
            σ²_flat[offset + i]      = convert(T, errs[i]^2) + jit²
            a_flat[offset + i]       = a_coef
            b_flat[offset + i]       = b_coef
            channel_flat[offset + i] = ch
        end
        offset += n_ch
    end

    # Opt-in O(N·r²) SEMISEPARABLE latent kernel (Matérn-3/2 / SHO / ES / MEP /
    # ESP). The default `:qp_dense` leaves the dense/low-rank QP path below
    # untouched. Same Rajpaul layout — every channel observes aⱼ·G + bⱼ·Ġ of the
    # shared latent — but the joint marginal is factored in linear time via the
    # S+LEAF LDLᵀ solver (src/noise/multiseries_gp.jl), the only tractable route
    # at decade baselines with 10³⁺ epochs. Scores the JOINT only; the
    # marginalize/conditional diagnostic keeps the explicit dense partition.
    if agp.latent_kernel !== :qp_dense && !agp.marginalize_indicators
        kern = _ss_latent_kernel(agp.latent_kernel, amp, P, λe, λp)
        series_id = Vector{Int}(undef, n_total)
        @inbounds for i in 1:n_rv_obs
            series_id[i] = 1
        end
        soff = n_rv_obs
        for (s, m) in enumerate(ind_meta)
            n_ch = length(m[2])
            @inbounds for i in 1:n_ch
                series_id[soff + i] = s + 1
            end
            soff += n_ch
        end
        α_ss = T[Vc, (m[4] for m in ind_meta)...]
        β_ss = T[Vr, (m[5] for m in ind_meta)...]
        # σ²_flat already carries the per-channel measurement variance + jitter.
        return multiseries_loglike(t_flat, y_flat, σ²_flat, series_id,
                                    α_ss, β_ss, kern)
    end

    # Build covariance + diagonal measurement noise. Block-factored assembly
    # when every channel shares the RV epochs and has per-channel-constant
    # (a, b) — the standard Rajpaul layout (indicators 1:1 with RVs at the same
    # epochs). Numerically identical to the dense builder; transcendentals
    # evaluated C²× fewer times. Falls back to the dense builder if an
    # indicator vector isn't aligned 1:1 with the RVs (unequal length).
    if all(m -> length(m[2]) == n_rv_obs, ind_meta)
        chan_a = T[Vc, (m[4] for m in ind_meta)...]
        chan_b = T[Vr, (m[5] for m in ind_meta)...]
        # LOW-RANK joint: the C channels observe linear combos of a 2N
        # latent (G, Ġ), so Σ = M·K_g·Mᵀ + D and the dominant factorization
        # drops from (C·N)² to (2N)² — ~4× faster at C=5 (machine-precision
        # identical; validated). Only when it pays (C ≥ 4 ⇒ 2N < C·N) and
        # we score the joint (the marginalize/conditional path needs the
        # explicit block partition, so it keeps the dense build).
        if !agp.marginalize_indicators && n_total > n_rv_obs &&
           length(chan_a) >= 4
            return activity_gp_joint_logpdf_lowrank(
                view(data.t_rv, 1:n_rv_obs), chan_a, chan_b,
                amp, P, λe, λp, y_flat, σ²_flat)
        end
        Σ = activity_gp_covariance_blocked(view(data.t_rv, 1:n_rv_obs),
                                            chan_a, chan_b, amp, P, λe, λp)
    else
        Σ = activity_gp_covariance(t_flat, channel_flat, a_flat, b_flat,
                                     amp, P, λe, λp)
    end
    @inbounds for i in 1:n_total
        Σ[i, i] += σ²_flat[i]
    end

    if agp.marginalize_indicators && n_total > n_rv_obs
        # Conditional Gaussian: log p(RV | indicators) — marginalises
        # the indicator block out of the joint Rajpaul Gaussian.
        #
        # Partition Σ = [Σ_RR Σ_RI; Σ_IR Σ_II], with indicator block I
        # spanning rows (n_rv+1):n_total. The conditional is
        #   μ_R|I = Σ_RI · Σ_II⁻¹ · (y_I − μ_I)     (μ_I = 0 here)
        #   Σ_R|I = Σ_RR − Σ_RI · Σ_II⁻¹ · Σ_IR
        # and the log-density of (y_R − μ_R|I) under N(0, Σ_R|I)
        # gives `log p(RV | I, θ)` directly comparable to AD's log L.
        Σ_RR = view(Σ, 1:n_rv_obs, 1:n_rv_obs)
        Σ_RI = view(Σ, 1:n_rv_obs, (n_rv_obs + 1):n_total)
        Σ_II = view(Σ, (n_rv_obs + 1):n_total, (n_rv_obs + 1):n_total)
        y_R  = view(y_flat, 1:n_rv_obs)
        y_I  = view(y_flat, (n_rv_obs + 1):n_total)
        F_II = cholesky(Symmetric(Matrix(Σ_II)); check = false)
        issuccess(F_II) || return convert(T, -Inf)
        rhs = F_II \ Matrix(Σ_IR_view(Σ_RI))   # n_ind × n_rv
        μ_cond = Σ_RI * (F_II \ y_I)
        Σ_cond = Symmetric(Matrix(Σ_RR) .- Σ_RI * rhs)
        F_R = cholesky(Σ_cond; check = false)
        issuccess(F_R) || return convert(T, -Inf)
        r = y_R .- μ_cond
        return convert(T,
            -0.5 * (dot(r, F_R \ r) + logdet(F_R) +
                     n_rv_obs * log(2π)))
    end

    # Joint Gaussian log p(RV, indicators).
    F = cholesky(Symmetric(Σ); check = false)
    issuccess(F) || return convert(T, -Inf)
    α = F \ y_flat
    return convert(T,
        -0.5 * (dot(y_flat, α) + logdet(F) + n_total * log(2π)))
end

# Helper to transpose a SubArray-aware view (Σ_IR is the transpose
# of Σ_RI by symmetry; rebuilding via `Σ_RI'` keeps the Symmetric
# block PD).
@inline _Σ_IR_view(Σ_RI) = transpose(Matrix(Σ_RI))
const Σ_IR_view = _Σ_IR_view


"""
    rv_predictions(theta::Theta, data::Data) -> (predictions, variances)

Compute RV mean model predictions and per-observation variances.
Returns Stage 1 output (Keplerian + gamma + trend + activity) without
sequential noise (AR/MA) or GP covariance.

Used by `compute_model_stats` for residual diagnostics.
"""
function rv_predictions(theta::Theta{T}, data::Data) where {T}
    parametrization = theta.params.config.parametrization
    t_ref   = data.t_ref
    n_obs   = length(data.t_rv)
    two_pi  = T(2π)
    noise_models = theta.params.config.noise_models

    # Decode planets
    p_idx = planet_indices(theta)
    n_rv_planets = 0
    for k in p_idx
        if has_K(theta.params.layout.planet_blocks[k])
            n_rv_planets += 1
        end
    end

    Ps  = Vector{T}(undef, n_rv_planets)
    Ks  = Vector{T}(undef, n_rv_planets)
    cBs = Vector{T}(undef, n_rv_planets)   # component-B coefficient (0, or −K_B for SB2)
    es  = Vector{T}(undef, n_rv_planets)
    ws  = Vector{T}(undef, n_rv_planets)
    Tps = Vector{T}(undef, n_rv_planets)

    j = 0
    for k in p_idx
        block = theta.params.layout.planet_blocks[k]
        has_K(block) || continue
        j += 1
        Ps[j]  = planet_P(theta, k)
        if block isa SB2Block
            Ks[j]  = planet_K_A(theta, k)
            cBs[j] = -planet_K_B(theta, k)
        else
            Ks[j]  = planet_K(theta, k)
            cBs[j] = zero(T)
        end
        e, w   = planet_e_w(theta, k)
        es[j]  = e
        ws[j]  = w
        ta = planet_time_anchor(theta, k)
        if parametrization.time === :Mo
            Tps[j] = t_ref - ta * Ps[j] / two_pi
        elseif parametrization.time === :Tp
            Tps[j] = ta
        else
            Tps[j] = tc_to_tp(ta, Ps[j], e, w)
        end
    end

    # Decode trend
    trend_order = theta.params.config.trend_order
    trend_dvdt = trend_order >= 1 ? rv_dvdt(theta) : zero(T)
    trend_curv = trend_order >= 2 ? rv_d2vdt2(theta) : zero(T)

    # Decode Rossiter-McLaughlin state (mirrors the likelihood) so the predicted
    # RV — used by PPC, residuals, fit-health and all RV plots — includes the
    # in-transit RM anomaly. No-op for non-RM fits (n_rm == 0); a missing M_s/R_s
    # (n_rm == -1) simply skips RM here rather than poisoning predictions.
    n_rm, rm_state = _decode_rm_state(theta, p_idx, Ps)
    n_rm == -1 && (n_rm = 0)

    # Build predictions + variances
    predictions = Vector{T}(undef, n_obs)
    variances   = Vector{T}(undef, n_obs)

    @inbounds for i in 1:n_obs
        t       = data.t_rv[i]
        obs_err = data.rv_err[i]
        ins_idx = data.rv_inst[i]

        gamma = rv_gamma(theta, ins_idx)
        dt_i  = t - t_ref
        pred  = gamma + trend_dvdt * dt_i + trend_curv * dt_i * dt_i

        for (nm_idx, nm) in enumerate(noise_models)
            is_noise_model_active(theta, nm_idx) || continue
            if nm isa ActivityDecorrelation
                pred = apply_activity_decorrelation(pred, theta, data, nm, ins_idx, i)
            end
        end

        comp = data.rv_comp[i]
        for j in 1:n_rv_planets
            M = two_pi * (t - Tps[j]) / Ps[j]
            E = kepler_solve(M, es[j])
            f = true_anomaly(E, es[j])
            geom = cos(f + ws[j]) + es[j] * cos(ws[j])
            pred += _comp_rv(geom, comp, Ks[j], cBs[j])
        end

        (n_rm > 0 && comp == 1) &&
            (pred += rm_contribution(t, n_rm, rm_state, Ps, es, ws, Tps))

        sigma = rv_sigma(theta, ins_idx)
        predictions[i] = pred
        var_i = obs_err * obs_err + sigma * sigma

        for (nm_idx, nm) in enumerate(noise_models)
            is_noise_model_active(theta, nm_idx) || continue
            if nm isa ActivityJitter
                var_i = apply_activity_jitter(obs_err * obs_err, theta, data, nm, ins_idx, i)
            elseif nm isa ErrorScale && _errorscale_covers(theta, nm, ins_idx)
                # Multiplicative error-scale REPLACES additive jitter: f²·σ_formal²
                # (whenever it covers this instrument — independent of the drawn f).
                f2 = error_scale_factor(theta, nm, ins_idx)
                var_i = f2 * obs_err * obs_err
            end
        end
        variances[i] = var_i
    end

    return predictions, variances
end


"""
    _white_noise_ll(residuals, variances, two_pi) -> T

Diagonal Gaussian log-likelihood.
"""
function _white_noise_ll(residuals::Vector{T}, variances::Vector{T},
                          two_pi::T) where {T}
    total = zero(T)
    @inbounds for i in eachindex(residuals)
        total += -(log(two_pi * variances[i]) + residuals[i]^2 / variances[i]) / 2
    end
    return total
end

# =====================================================================
# Analytic γ-marginalized white-noise RV log-likelihood
# =====================================================================
#
# The per-instrument RV systemic offset γ_g enters the model LINEARLY:
#
#     pred_i = γ_{g(i)} + μ_i          (μ_i = Keplerian + trend + RM)
#
# so, conditional on the orbit and the (sampled) jitter σ, the
# white-noise Gaussian likelihood is Gaussian in each γ_g and can be
# integrated in closed form under a FLAT (improper) prior — orvara's
# approach. This mirrors the linear-parameter marginalization in
# `iad_log_likelihood` (χ²_min = rᵀWr − vᵀA⁻¹v with a −½ log det A
# Jacobian); here each instrument group is a scalar linear parameter so
# the 5×5 normal equations collapse to per-group scalars.
#
# For group g define d_i = rv_i − μ_i, w_i = 1/v_i (v_i = obs_err_i² +
# σ_g², σ_g the sampled jitter for the group's instruments) and
#
#     A_g = Σ_i w_i,  B_g = Σ_i w_i d_i,  C_g = Σ_i w_i d_i².
#
# Then Σ_i w_i (d_i − γ)² = A_g γ² − 2 B_g γ + C_g, and
#
#     ∫ exp(−½(A_g γ² − 2 B_g γ + C_g)) dγ
#       = exp(−½ (C_g − B_g²/A_g)) · √(2π / A_g).
#
# The γ-marginalized log-likelihood for the group is therefore
#
#     ln L_g = −½ Σ_i log(2π v_i)              # per-point Gaussian norm
#              −½ (C_g − B_g²/A_g)             # χ²_min after profiling γ
#              +½ log(2π / A_g)                # Gaussian-integral Jacobian
#
# The conditional MAP point estimate is γ̂_g = B_g / A_g (reported via
# `conditional_gamma`).
#
# KEY SUBTLETY: jitter σ makes v_i (hence w_i) depend on σ, so γ is
# marginalized at FIXED σ — σ stays a sampled parameter. The γ-marginal
# is Gaussian only conditional on σ, which is exactly what this does.

"""
    _rv_ll_gamma_marginalized(residuals_no_gamma, variances, rv_inst,
                              gamma_slot, n_obs, two_pi) -> T

White-noise RV log-likelihood with the per-instrument-group systemic
offset γ analytically marginalized (flat prior). `residuals_no_gamma[i]
= rv_i − μ_i` are the data minus the γ-free mean model; `variances[i]`
are the per-point variances (formal error² + jitter²). `gamma_slot[i]`
is the shared γ layout-slot index for observation `i` — observations
that share a slot (the standard per-instrument case, or a `:gamma`
sharing group) are pooled into one Gaussian integral.

Accumulates the per-group sufficient statistics (A, B, C) in a single
pass, then adds the closed-form marginal. Allocation is one small
`Dict` keyed by γ-slot (number of distinct RV instrument groups, ≪
n_obs); the hot per-point loop is allocation-free.
"""
function _rv_ll_gamma_marginalized(residuals_no_gamma::AbstractVector{T},
                                    variances::AbstractVector{T},
                                    rv_inst::AbstractVector{<:Integer},
                                    gamma_slot::Vector{Int},
                                    n_obs::Int, two_pi::T) where {T}
    # Per-point Gaussian normalization Σ −½ log(2π v_i), plus per-group
    # accumulation of (A_g, B_g, C_g) keyed by the shared γ slot.
    log_norm = zero(T)
    # Group sufficient statistics. Distinct γ slots are few (one per RV
    # instrument group), so a Dict keyed by slot is cheap and robust to
    # non-contiguous / shared slot indices.
    A = Dict{Int, T}()
    B = Dict{Int, T}()
    C = Dict{Int, T}()

    @inbounds for i in 1:n_obs
        v = variances[i]
        w = inv(v)
        d = residuals_no_gamma[i]
        log_norm += -0.5 * log(two_pi * v)
        g = gamma_slot[rv_inst[i]]
        A[g] = get(A, g, zero(T)) + w
        B[g] = get(B, g, zero(T)) + w * d
        C[g] = get(C, g, zero(T)) + w * d * d
    end

    # Per-group marginal:  −½ (C − B²/A) + ½ log(2π / A).
    marg = zero(T)
    for (g, Ag) in A
        Bg = B[g]
        Cg = C[g]
        # A_g > 0 whenever the group has ≥1 finite-variance observation.
        chi2_min = Cg - Bg * Bg / Ag
        marg += -0.5 * chi2_min + 0.5 * log(two_pi / Ag)
    end

    return log_norm + marg
end
