# DB-correct swaps: planet <-> ActivityDecorrelation, and within-group noise.
#
# Birth/death cannot cross the "none" valley between two mutually exclusive
# members at beta = 1; these go straight across.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# DB-correct planet ↔ ActivityDecorrelation SWAP (Green 1995 RJMCMC).
# Trades a planet for the AD model that absorbs its signal (and reverse)
# in ONE accept/reject, so a redundant planet can be killed at β≈1 — the
# move the standard birth/death can't make (the path crosses a likelihood
# valley). Unlike the burn-in DB-free swap, this carries the exact swap
# Hastings, so it runs POST-burn-in and makes the occupancy a true P(M|D).
#
# Proposals are INFORMED and density-evaluable in both directions:
#   * AD birth: coefficients ~ N(C_ols, inflate²(XᵀWX)⁻¹)  → _ad_mvn_logq
#   * planet birth: log-P ~ informed mixture (_informed_log_q); K/e/ω/Mo
#     from their priors. Both seed on `_compute_rv_residuals` (the same
#     state-independent residual the standard InformedBirth uses), so the
#     peaks — hence q — are identical forward and reverse. Jacobian = 1
#     (drawn values ARE the new params).
#
# log_q_ratio = log q(reverse) − log q(forward), where each q factors as
#   [direction-select] × [which-component-to-kill] × [informed birth density].
# =====================================================================
const _SWAP_AD_INFLATE = 2.0
const _SWAP_PSIG_K = 1.0    # log-K proposal width for the swap planet birth
const _SWAP_PSIG_E = 0.3
const _SWAP_NFREQ  = 3000   # periodogram-grid cap for the swap's _find_peaks
                            # (locate peaks cheaply; per-move, so cost matters)

# Find the (single) ActivityDecorrelation among toggleable models; its
# global noise index, or nothing.
function _swap_ad_index(toggleable::Vector{NoiseModel}, noise_models::Vector{NoiseModel})
    for nm in toggleable
        nm isa ActivityDecorrelation || continue
        j = findfirst(==(nm), noise_models)
        j === nothing || return (nm, j)
    end
    return (nothing, nothing)
end

# Planet-birth proposal for the swap: informed log-P + prior K/e1/e2/t.
# mode=:draw  → sample params into new_values, return log_q (over the drawn
#               coords: log-P + the prior coords).
# mode=:eval  → evaluate log_q at the params ALREADY in `vals` for slot k.
# Returns (log_q, P_lo, P_hi) or (-Inf,…) on degenerate setup.
function _swap_planet_q(theta::Theta{T}, k::Int, data::Data, rng,
                         vals::Vector{T}; mode::Symbol) where {T}
    layout = theta.params.layout
    block  = layout.planet_blocks[k]
    P_pos  = findfirst(==(block.P), layout.unfrozen_idx)
    P_pos === nothing && return (convert(T,-Inf),)
    P_lo, P_hi = bounds(layout.unfrozen_priors[P_pos])
    logPmin, logPmax = log(max(P_lo, 0.1)), log(P_hi)
    res = _compute_rv_residuals(theta, data)
    # Cap the periodogram grid: the swap fires per trans-dim move, so a full
    # baseline-resolution LS here dominates wall-time on real data (long
    # baseline + wide P range = tens of thousands of frequencies). A coarse
    # grid still LOCATES the peaks the proposal needs; draw and eval use the
    # same cap, so the informed density stays self-consistent (DB-safe).
    pk_P, pk_w, _, _ = _find_peaks(data.t_rv, res, P_lo, P_hi;
                                    t_ref = data.t_ref, σ = data.rv_err,
                                    max_nfreq = _SWAP_NFREQ)
    pk_logP = isempty(pk_P) ? Float64[] : log.(pk_P)
    α = isempty(pk_logP) ? 0.0 : INFORMED_ALPHA

    # --- period ---
    if mode === :draw
        if rand(rng) < α
            r = rand(rng); cum = 0.0; ci = 1
            for (i,wt) in enumerate(pk_w); cum += wt; (r < cum) && (ci = i; break); end
            logP = pk_logP[ci] + INFORMED_SIGMA_P * randn(rng)
        else
            logP = logPmin + (logPmax - logPmin) * rand(rng)
        end
        logP = clamp(logP, logPmin, logPmax)
        vals[block.P] = convert(T, exp(logP))
    else
        logP = clamp(log(vals[block.P]), logPmin, logPmax)
    end
    # Period density over P (Jacobian −logP), to match the LogUniform period
    # prior counted (over P) in Δlogπ — same convention as InformedBirth.
    log_q = _informed_log_q(logP, pk_logP, pk_w, logPmin, logPmax) - logP

    # --- remaining planet slots from prior (K/e1/e2/t/…) ---
    # K/e/Mo are drawn from (and evaluated at) their priors. The informed
    # alternative (centring on the LS estimate, like InformedBirth) was tried
    # and REVERTED: it cratered swap acceptance (15%→0.7%) — the circular-biased
    # e proposal N(0,0.15) is badly mismatched to the high-e Lucy-Sweeney
    # posterior under uniform sesinw/secosw — while leaving the stationary
    # unchanged (verified DB-correct either way by toy_swap_db_gate_proper.jl).
    # The swap only has to MOVE the shared signal between planet and AD; the
    # period is informed (that's what locates the trade), K/e/Mo from prior is
    # both DB-exact and the better mixer.
    for slot in planet_slot_indices(block)
        slot == block.P && continue
        uf = findfirst(==(slot), layout.unfrozen_idx); uf === nothing && continue
        lo, hi = bounds(layout.unfrozen_priors[uf])
        if mode === :draw
            v = clamp(rand(rng, layout.unfrozen_priors[uf].dist), lo, hi)
            vals[slot] = convert(T, v)
        end
        log_q += eval_packed_logpdf(vals[slot],
            layout.packed_priors.type_ids[uf], layout.packed_priors.params[uf,1],
            layout.packed_priors.params[uf,2], layout.packed_priors.lowers[uf],
            layout.packed_priors.uppers[uf])
    end
    return (convert(T, log_q), P_lo, P_hi)
end

# AD-birth proposal density for the swap. mode=:draw fills the AD coef slots
# of `vals` with N(C_ols, inflate²Σ) draws (C_ols fit on `theta`'s residual,
# which must have AD INACTIVE); mode=:eval reads the coefs from `vals`.
# Returns log_q (the Gaussian proposal density) or -Inf on degenerate fit.
function _swap_ad_q(theta::Theta{T}, ad::ActivityDecorrelation, data::Data, rng,
                     vals::Vector{T}; mode::Symbol) where {T}
    layout = theta.params.layout
    cnames, C_ols, F = ad_ols_fit(theta, data, ad)
    (F === nothing || isempty(cnames)) && return convert(T,-Inf)
    Ccur = Float64[vals[layout.name_to_idx[nm]] for nm in cnames]
    if mode === :draw
        z = randn(rng, length(C_ols))
        Cd = C_ols .+ _SWAP_AD_INFLATE .* (F.U \ z)
        @inbounds for (i,nm) in enumerate(cnames)
            slot = layout.name_to_idx[nm]
            uf = findfirst(==(slot), layout.unfrozen_idx); uf === nothing && continue
            lo, hi = bounds(layout.unfrozen_priors[uf])
            Cd[i] = clamp(Cd[i], lo, hi); vals[slot] = convert(T, Cd[i])
        end
        Ccur = Cd
    end
    return convert(T, _ad_mvn_logq(Ccur, C_ols, F, _SWAP_AD_INFLATE))
end

# Available swap directions from a state: :pn (kill planet, activate AD) needs
# ≥1 planet AND AD inactive; :np (deactivate AD, birth planet) needs AD active
# AND a free planet slot.
function _swap_dirs(td, adi::Int, modes, max_k::Int)
    dirs = Symbol[]
    (td.n_planets_active ≥ 1 && !is_noise_active(td, adi)) && push!(dirs, :pn)
    (is_noise_active(td, adi) && td.n_planets_active < max_k) && push!(dirs, :np)
    dirs
end

"""
    propose_planet_ad_swap(theta, rng, toggleable; data) -> (new_theta, log_q_ratio)

DB-correct planet↔ActivityDecorrelation swap (see header above ad_ols_fit).
Returns `(theta, -Inf)` if no AD model is toggleable or no swap direction is
available. The caller combines `log_q_ratio` with ΔlogL+Δlogπ in the standard
RJMCMC acceptance; valid POST-burn-in.
"""
function propose_planet_ad_swap(theta::Theta{T}, rng::AbstractRNG,
                                 toggleable::Vector{NoiseModel};
                                 data::Union{Data,Nothing}=nothing) where {T}
    data === nothing && return (theta, convert(T,-Inf))
    nm_models = theta.params.config.noise_models
    ad, adi = _swap_ad_index(toggleable, nm_models)
    ad === nothing && return (theta, convert(T,-Inf))
    modes = theta.params.config.planet_modes
    max_k = length(theta.td.planet_active)
    dirs = _swap_dirs(theta.td, adi, modes, max_k)
    isempty(dirs) && return (theta, convert(T,-Inf))
    d_x = length(dirs)
    dir = dirs[rand(rng, 1:d_x)]

    nv = copy(theta.values); ntd = copy(theta.td)
    nt = Theta{T}(theta.params, nv; td = ntd)

    if dir === :pn
        # forward: kill a random active planet, birth AD on the freed residual
        active = active_planets(theta.td)
        k = active[rand(rng, 1:length(active))]; n_p = theta.td.n_planets_active
        deactivate_planet!(nt.td, k)
        q_ad_fwd = _swap_ad_q(nt, ad, data, rng, nt.values; mode = :draw)   # planet k & AD both inactive here
        isfinite(q_ad_fwd) || return (theta, convert(T,-Inf))
        activate_noise!(nt.td, adi)
        # reverse (np from nt): deactivate AD, birth planet k. Its informed
        # density at the killed params (residual is state-independent → use nt).
        r = _swap_planet_q(nt, k, data, rng, nt.values; mode = :eval)
        q_pl_rev = r[1]; isfinite(q_pl_rev) || return (theta, convert(T,-Inf))
        d_xp = length(_swap_dirs(nt.td, adi, modes, max_k))
        Gp = length(group_first_inactive(nt.td, modes))      # reverse picks the slot among these
        log_q = (log(d_x) - log(d_xp)) + (log(n_p) - log(max(Gp,1))) + (q_pl_rev - q_ad_fwd)
        _sort_group_periods!(nt)
        return (nt, convert(T, log_q))
    else  # :np
        n_p = theta.td.n_planets_active
        deactivate_noise!(nt.td, adi)
        cand = group_first_inactive(nt.td, modes)
        isempty(cand) && return (theta, convert(T,-Inf))
        k = cand[rand(rng, 1:length(cand))]; G = length(cand)
        r = _swap_planet_q(nt, k, data, rng, nt.values; mode = :draw)
        q_pl_fwd = r[1]; isfinite(q_pl_fwd) || return (theta, convert(T,-Inf))
        activate_planet!(nt.td, k)
        # reverse (pn from nt): kill planet k, birth AD. AD-birth density at the
        # CURRENT coefs (still in nt.values), fit on nt-with-planet-k removed.
        tmp_td = copy(nt.td); deactivate_planet!(tmp_td, k)
        tmp = Theta{T}(theta.params, nt.values; td = tmp_td)
        q_ad_rev = _swap_ad_q(tmp, ad, data, rng, nt.values; mode = :eval)
        isfinite(q_ad_rev) || return (theta, convert(T,-Inf))
        d_xp = length(_swap_dirs(nt.td, adi, modes, max_k))
        n_p_after = n_p + 1                                   # reverse kills one of these
        log_q = (log(d_x) - log(d_xp)) + (log(G) - log(n_p_after)) + (q_ad_rev - q_pl_fwd)
        _sort_group_periods!(nt)
        return (nt, convert(T, log_q))
    end
end

# Density-evaluable noise-birth proposal for the within-group swap.
#   * ActivityDecorrelation → OLS-INFORMED draw/eval (_swap_ad_q): N(C_ols,
#     inflate²Σ). The informed escape is what lets a walker entrenched in a
#     flexible model jump to AD at β≈1 (a prior-drawn AD never fits, so a blind
#     swap would never be accepted in that direction).
#   * everything else → prior draw/eval, density = Σ logpdf(prior). This matches
#     propose_noise_birth's non-informed branch, so the −log_q_fwd cancels the
#     +Σ logpdf the activation adds to Δlogπ (occupancy stays P(M|D)).
# `theta` MUST have the target model INACTIVE so an AD OLS fit sees the
# model-free residual. mode=:draw fills `vals` at the model's slots and returns
# the forward density; mode=:eval reads them from `vals`.
function _noise_birth_q(theta::Theta{T}, nm::NoiseModel,
                         data::Union{Data,Nothing}, rng,
                         vals::Vector{T}; mode::Symbol) where {T}
    if nm isa ActivityDecorrelation && data !== nothing
        return _swap_ad_q(theta, nm, data, rng, vals; mode = mode)
    end
    layout = theta.params.layout
    instruments = theta.params.config.instruments
    nm_names = noise_param_names(nm, instruments)
    log_q = zero(T)
    for name in nm_names
        haskey(layout.name_to_idx, name) || continue
        slot = layout.name_to_idx[name]
        uf = findfirst(==(slot), layout.unfrozen_idx); uf === nothing && continue
        prior = layout.unfrozen_priors[uf]
        lo, hi = bounds(prior)
        if mode === :draw
            v = clamp(rand(rng, prior.dist), lo, hi)
            vals[slot] = convert(T, v)
        end
        log_q += eval_packed_logpdf(vals[slot],
            layout.packed_priors.type_ids[uf], layout.packed_priors.params[uf, 1],
            layout.packed_priors.params[uf, 2], layout.packed_priors.lowers[uf],
            layout.packed_priors.uppers[uf])
    end
    return convert(T, log_q)
end

"""
    _any_group_member_active(theta, groups) -> Bool

Is any member of any exclusion group currently active? A within-group swap has
nothing to swap from otherwise, and callers should fall through to birth/death.
"""
function _any_group_member_active(theta::Theta, groups)
    nms = theta.params.config.noise_models
    for grp in groups, m in grp
        i = findfirst(==(m), nms)
        i === nothing && continue
        is_noise_active(theta.td, i) && return true
    end
    return false
end

"Pick uniformly among the exclusion groups that have an active member."
function _pick_active_group(theta::Theta, groups, rng::AbstractRNG)
    nms = theta.params.config.noise_models
    live = Int[]
    for (gi, grp) in enumerate(groups)
        for m in grp
            i = findfirst(==(m), nms)
            i === nothing && continue
            if is_noise_active(theta.td, i); push!(live, gi); break; end
        end
    end
    isempty(live) && return groups[rand(rng, 1:length(groups))]
    return groups[live[rand(rng, 1:length(live))]]
end

"""
    propose_noise_swap(theta, rng, group, toggleable; data) -> (new_theta, log_q_ratio)

DB-correct WITHIN-EXCLUSION-GROUP noise swap: deactivate the active member A and
activate another member B in ONE move, so a walker entrenched in a flexible
disfavoured model (e.g. GP-Rot) can ESCAPE to the high-evidence model (e.g. AD)
WITHOUT crossing the "none" valley that birth/death can't at β≈1. Unlike the
burn-in DB-free climb-swap, this is density-evaluable in BOTH directions (AD via
the OLS-informed _swap_ad_q; every other model via its prior), so it carries the
exact swap Hastings and runs POST-burn-in — which is what keeps the noise
occupancy = P(M|D) when the modes are deep (the disfavoured mode otherwise
over-represents, measured at toy scale: GP-Rot 0.3% vs evidence ~0).

`group` is ONE exclusion group; exactly one member is active by construction.
Returns `(theta,-Inf)` if 0 or >1 members active, no inactive member, or a
degenerate fit. The caller combines `log_q_ratio` with β·ΔlogL+Δlogπ.
"""
function propose_noise_swap(theta::Theta{T}, rng::AbstractRNG,
                             group::Vector{NoiseModel},
                             toggleable::Vector{NoiseModel};
                             data::Union{Data,Nothing} = nothing) where {T}
    nm_models = theta.params.config.noise_models
    members = Tuple{NoiseModel,Int}[]
    for nm in group
        nm in toggleable || continue
        j = findfirst(==(nm), nm_models); j === nothing && continue
        push!(members, (nm, j))
    end
    length(members) ≥ 2 || return (theta, convert(T, -Inf))
    active = [(nm, j) for (nm, j) in members if is_noise_active(theta.td, j)]
    length(active) == 1 || return (theta, convert(T, -Inf))
    A, ai = active[1]
    inactive = [(nm, j) for (nm, j) in members if !is_noise_active(theta.td, j)]
    isempty(inactive) && return (theta, convert(T, -Inf))
    G = length(inactive)
    B, bi = inactive[rand(rng, 1:G)]

    nv = copy(theta.values); ntd = copy(theta.td)
    nt = Theta{T}(theta.params, nv; td = ntd)
    deactivate_noise!(nt.td, ai)                # A off; B still off → residual model-free for B's fit
    q_B_fwd = _noise_birth_q(nt, B, data, rng, nt.values; mode = :draw)
    isfinite(q_B_fwd) || return (theta, convert(T, -Inf))
    activate_noise!(nt.td, bi)
    # reverse: from nt, kill B, birth A at A's OLD params (its slots are untouched
    # in nt.values — A and B are distinct models with disjoint param names).
    tmp_td = copy(nt.td); deactivate_noise!(tmp_td, bi)
    tmp = Theta{T}(theta.params, nt.values; td = tmp_td)     # A & B both inactive
    q_A_rev = _noise_birth_q(tmp, A, data, rng, nt.values; mode = :eval)
    isfinite(q_A_rev) || return (theta, convert(T, -Inf))
    # selection: forward picks B among G inactive; reverse picks A among nt's
    # inactive members (same set size — B now active, A now inactive).
    G_rev = count(((nm, j),) -> !is_noise_active(nt.td, j), members)
    log_q = (log(G) - log(max(G_rev, 1))) + (q_A_rev - q_B_fwd)
    return (nt, convert(T, log_q))
end

# log N(x; μ, inflate²·prec⁻¹) given prec = LLᵀ (cholesky `F`).
function _ad_mvn_logq(x::AbstractVector, μ::AbstractVector, F, inflate::Float64)
    K = length(x)
    d = x .- μ
    Ld = F.L' * d                                   # Lᵀ d  (prec = L Lᵀ)
    quad = (Ld ⋅ Ld) / (inflate^2)
    logdet_prec = 2 * sum(log, @view F.factors[diagind(F.factors)])
    logdet_cov = -logdet_prec + 2 * K * log(inflate)
    return -0.5 * (K * log(2π) + logdet_cov + quad)
end

