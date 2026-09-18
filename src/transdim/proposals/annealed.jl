# Annealed (bridged) noise birth — Karagiannis & Andrieu (2013).
#
# Makes "birth, relax, then judge" reversible, so it is admissible during
# sampling and not only in burn-in. See the file body for the construction.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# Annealed (bridged) noise birth  —  Karagiannis & Andrieu (2013)
# =====================================================================
#
# WHY. A blind birth of a 5-12 parameter noise model is judged at the drawn
# configuration, which is nowhere near the mode, so it is rejected; and the
# matching death is penalised by how unlikely that draw was, so an entrenched
# model cannot leave either. Both doors jam, the chain freezes in whichever
# member of an exclusion group it entered first, and the reported occupancy
# reflects where it got stuck rather than the evidence.
#
# `climb_newborn!` already does the obvious thing — relax the newborn with a
# few MCMC steps BEFORE judging it — and it is deliberately restricted to
# burn-in, because done naively it is not reversible: you have optimised the
# candidate before the accept test, so births win more often than the posterior
# says they should and the chain drifts to higher dimension. Nicer numbers,
# wrong numbers.
#
# The annealed construction makes exactly that idea reversible. Bridge from
#
#   rho_0 = prior_A * L_A * q(u)      (model A, plus the auxiliary draw)
# to
#   rho_T = prior_B * L_B             (model B)
#
# geometrically, rho_t = rho_0^(1-b_t) * rho_T^(b_t), run a rho_t-invariant
# kernel on the newborn parameters at each stage, and accumulate
#
#   log W = sum_t (b_t - b_{t-1}) * [log rho_T(x_{t-1}) - log rho_0(x_{t-1})]
#
# The accept probability is exp(log W) times the combinatorial ratio. Two
# properties make this checkable rather than hopeful:
#
#   * with ONE bridge stage (b_1 = 1) no MCMC runs and log W collapses to
#     log rho_T(x_0) - log rho_0(x_0), i.e. the ORDINARY reversible-jump
#     ratio. That exact reduction is the primary unit test — it is what
#     catches a mis-accumulated weight.
#   * the relaxation kernel can be anything rho_t-invariant, so its quality
#     affects efficiency only, never the stationary distribution.
#
# It also crosses the empty-model valley without a swap: intermediate stages
# down-weight the new component's likelihood, so the path never has to accept
# a state where nothing explains the signal.

"Log prior density of a model's own parameters at their current values."
function _noise_block_logprior(theta::Theta, slots::Vector{Int})
    layout = theta.params.layout
    lp = 0.0
    @inbounds for slot in slots
        uf = findfirst(==(slot), layout.unfrozen_idx)
        uf === nothing && continue
        lp += eval_packed_logpdf(theta.values[slot],
                layout.packed_priors.type_ids[uf],
                layout.packed_priors.params[uf, 1],
                layout.packed_priors.params[uf, 2],
                layout.packed_priors.lowers[uf],
                layout.packed_priors.uppers[uf])
    end
    return lp
end

"Unfrozen parameter slots belonging to noise model `nm_idx`."
function _noise_model_slots(theta::Theta, nm_idx::Int)
    layout = theta.params.layout
    nm = theta.params.config.noise_models[nm_idx]
    slots = Int[]
    for name in noise_param_names(nm, theta.params.config.instruments)
        haskey(layout.name_to_idx, name) || continue
        slot = layout.name_to_idx[name]
        findfirst(==(slot), layout.unfrozen_idx) === nothing || push!(slots, slot)
    end
    return slots
end

"""
    propose_noise_birth_annealed(theta, rng, toggleable, loglik, logprior; ...)
        -> (cand, log_accept, ll_new)

Annealed reversible-jump birth of a toggleable noise model. `loglik(θ)` and `logprior(θ)` must be the SAME functions the caller uses for
its own acceptance bookkeeping (in `transdim_ptemcee` that means
`spike_slab_log_prior`, not `log_prior` — they differ by the planet-indicator
Bernoulli terms). `beta` is the chain's temperature.

Returns the FULL log acceptance exponent, not a proposal ratio — the bridge has
already consumed the prior and likelihood terms. `log_accept = -Inf` means the
move could not be constructed and the caller should do nothing.

`n_bridge = 1` reproduces the ordinary RJ birth exactly (see the note above).
"""
function propose_noise_birth_annealed(theta::Theta{T}, rng::AbstractRNG,
        toggleable::Vector{NoiseModel}, loglik, logprior;
        data::Union{Data,Nothing}=nothing,
        exclusion_groups::Vector{Vector{NoiseModel}}=Vector{NoiseModel}[],
        beta::Real=1.0, n_bridge::Int=8, n_relax::Int=2,
        relax_scale::Real=0.3,
        ll_current::Union{Nothing,Real}=nothing,
        lp_current::Union{Nothing,Real}=nothing) where {T}

    n_bridge >= 1 || throw(ArgumentError("n_bridge must be ≥ 1"))
    parts = Ref{Any}(nothing)
    cand, lqr = propose_noise_birth(theta, rng, toggleable; data=data,
                                     exclusion_groups=exclusion_groups,
                                     parts=parts)
    (isfinite(lqr) && parts[] !== nothing) || return (theta, convert(T,-Inf), NaN)
    p = parts[]
    slots = _noise_model_slots(cand, p.nm_idx)
    isempty(slots) && return (theta, convert(T,-Inf), NaN)
    # q != prior for the OLS-informed AD birth, so the relaxation target below
    # is not its rho_t. Fall back to a single stage (== ordinary RJ).
    if theta.params.config.noise_models[p.nm_idx] isa ActivityDecorrelation
        n_bridge = 1
    end

    ll_A = ll_current === nothing ? loglik(theta)  : Float64(ll_current)
    lp_A = lp_current === nothing ? logprior(theta) : Float64(lp_current)
    (isfinite(ll_A) && isfinite(lp_A)) || return (theta, convert(T,-Inf), NaN)

    ll_x = loglik(cand)
    lp_x = logprior(cand)
    (isfinite(ll_x) && isfinite(lp_x)) || return (theta, convert(T,-Inf), NaN)

    # D(x) = log rho_T(x) - log rho_0(x), with
    #   rho_0 = pi_A(theta) * q(u)        rho_T = pi_B(theta, u)
    # Everything shared between the models cancels inside the prior difference.
    Dof(lp, ll) = (lp + beta*ll) - (lp_A + beta*ll_A + p.log_q_fwd)

    logW = 0.0
    βprev = 0.0
    for t in 1:n_bridge
        βt = t / n_bridge
        logW += (βt - βprev) * Dof(lp_x, ll_x)
        βprev = βt
        t == n_bridge && break                  # nothing to relax after the last

        # Relaxation must leave rho_t invariant. In general
        #   rho_t(u) ∝ q(u)^(1-b_t) * [prior(u) * L^beta]^(b_t)
        # which needs q evaluable at arbitrary u. When the birth DRAWS FROM THE
        # PRIOR (the default path) q = prior and this collapses to
        #   rho_t(u) ∝ prior(u) * L(theta,u)^(beta*b_t)
        # i.e. plain tempered-likelihood M-H on the newborn — which is what
        # `climb_newborn!` already approximates, only now its work is counted
        # instead of discarded. ActivityDecorrelation births are OLS-informed
        # (q != prior), so they are left to the standard path; their occupancy
        # gate already passes.
        for _ in 1:n_relax
            for slot in slots
                uf = findfirst(==(slot), cand.params.layout.unfrozen_idx)
                uf === nothing && continue
                lo, hi = bounds(cand.params.layout.unfrozen_priors[uf])
                old = cand.values[slot]
                span = (isfinite(hi) && isfinite(lo)) ? (hi - lo) : abs(old) + 1
                nv = old + relax_scale * span * randn(rng)
                (lo <= nv <= hi) || continue
                cur = lp_x + βt * beta * ll_x
                cand.values[slot] = convert(T, nv)
                ll_try = loglik(cand); lp_try = logprior(cand)
                if isfinite(ll_try) && isfinite(lp_try) &&
                   log(rand(rng)) < (lp_try + βt*beta*ll_try) - cur
                    ll_x = ll_try; lp_x = lp_try
                else
                    cand.values[slot] = old
                end
            end
        end
    end

    log_accept = logW + p.log_comb
    return (cand, convert(T, log_accept), ll_x)
end

"""
    propose_noise_death(theta, rng, toggleable) -> (new_theta, log_q_ratio)

Deactivate a random active noise component.
"""
function propose_noise_death(theta::Theta{T}, rng::AbstractRNG,
                              toggleable::Vector{NoiseModel};
                              scratch::Union{Theta{T}, Nothing}=nothing,
                              data::Union{Data, Nothing}=nothing) where {T}
    td = theta.td
    noise_models = theta.params.config.noise_models

    # Find active toggleable components
    active_toggle = Int[]
    for nm in toggleable
        nm_idx = findfirst(==(nm), noise_models)
        nm_idx === nothing && continue
        if is_noise_active(td, nm_idx)
            push!(active_toggle, nm_idx)
        end
    end

    isempty(active_toggle) && return (theta, convert(T, -Inf))

    nm_idx = active_toggle[rand(rng, 1:length(active_toggle))]

    # Use scratch if available, otherwise allocate
    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        deactivate_noise!(scratch.td, nm_idx)
        new_td = scratch.td
        new_theta = scratch
    else
        new_td = copy(td)
        deactivate_noise!(new_td, nm_idx)
        new_values = copy(theta.values)
        new_theta = Theta{T}(theta.params, new_values; td=new_td)
    end

    n_active = length(active_toggle)
    # Count how many would be inactive after (for reverse proposal)
    n_inactive_after = count(.!new_td.noise_active[
        [findfirst(==(nm), noise_models) for nm in toggleable
         if findfirst(==(nm), noise_models) !== nothing]])
    n_inactive_after = max(n_inactive_after, 1)

    # Reverse-birth density of the killed params (prior pdf at their current
    # values) — mirrors the −log_q_fwd in `propose_noise_birth`: it cancels
    # the −Σ logpdf that Δlog_pi gains on deactivation, so the toggle samples
    # P(M|D) instead of a prior-tilted pseudo-posterior.
    layout = theta.params.layout
    instruments = theta.params.config.instruments
    killed = noise_models[nm_idx]
    # Reverse-birth density of the killed AD coefficients must MATCH the
    # OLS-informed forward birth: evaluate N(C_current; C_ols, inflate²
    # (XᵀWX)⁻¹) where the OLS is on the POST-death (AD-inactive) residual,
    # i.e. `new_theta` here (AD already deactivated above).
    log_q_rev = zero(T)
    ad_informed = false
    if AD_INFORMED_BIRTH[] && killed isa ActivityDecorrelation && data !== nothing
        cnames, C_ols, F = ad_ols_fit(new_theta, data, killed)
        if F !== nothing && length(C_ols) == length(cnames) && !isempty(cnames)
            Ccur = Float64[theta.values[layout.name_to_idx[nm]] for nm in cnames]
            log_q_rev = _ad_mvn_logq(Ccur, C_ols, F, 2.0)
            ad_informed = true
        end
    end
    # Mirror of the informed forward birth: the hint must be built on the
    # POST-death residual (`new_theta`, model already deactivated) so it is the
    # same hint the birth would have used.
    gp_hints = ad_informed ? NamedTuple[] :
               _gp_hints(new_theta, data, killed, layout, instruments)
    hint_of(sl) = (for h in gp_hints; h.slot == sl && return h; end; nothing)
    if !ad_informed
        for name in noise_param_names(killed, instruments)
            haskey(layout.name_to_idx, name) || continue
            slot = layout.name_to_idx[name]
            uf_pos = findfirst(==(slot), layout.unfrozen_idx)
            hh = hint_of(slot)
            if hh !== nothing
                lq = _gp_informed_logq(theta.values[slot], hh)
                isfinite(lq) || return (theta, convert(T, -Inf))
                log_q_rev += lq
            elseif uf_pos !== nothing
                log_q_rev += eval_packed_logpdf(theta.values[slot],
                    layout.packed_priors.type_ids[uf_pos],
                    layout.packed_priors.params[uf_pos, 1],
                    layout.packed_priors.params[uf_pos, 2],
                    layout.packed_priors.lowers[uf_pos],
                    layout.packed_priors.uppers[uf_pos])
            end
        end
    end

    log_q_ratio = log(n_inactive_after) - log(n_active) + log_q_rev

    return (new_theta, convert(T, log_q_ratio))
end

