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
        lp += eval_packed_logpdf(theta.values[slot], layout.packed_priors, uf)
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
its own acceptance bookkeeping (in `transdim_pt_emcee` that means
`spike_slab_log_prior`, not `log_prior` — they differ by the planet-indicator
Bernoulli terms). `beta` is the chain's temperature.

Returns the FULL log acceptance exponent, not a proposal ratio — the bridge has
already consumed the prior and likelihood terms. `log_accept = -Inf` means the
move could not be constructed and the caller should do nothing.

`n_bridge = 1` reproduces the ordinary RJ birth exactly (see the note above).
Its reverse is [`propose_noise_death_annealed`](@ref) with the same `n_bridge`,
`n_relax` and `relax_scale`: the annealed birth paired with the instantaneous
death is NOT reversible (the birth enjoys the bridge, the death does not, and
the chain over-occupies whatever the bridge makes easy to enter).
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
    n_bridge = _annealed_stages(theta, p.nm_idx, n_bridge)

    ll_A = ll_current === nothing ? loglik(theta)  : Float64(ll_current)
    lp_A = lp_current === nothing ? logprior(theta) : Float64(lp_current)
    (isfinite(ll_A) && isfinite(lp_A)) || return (theta, convert(T,-Inf), NaN)

    ll_x = loglik(cand)
    lp_x = logprior(cand)
    (isfinite(ll_x) && isfinite(lp_x)) || return (theta, convert(T,-Inf), NaN)

    # The birth proposal q(u) as a density (theta has the model inactive): the
    # relaxation targets rho_t, which carries q(u)^(1-b_t).
    qset = _noise_birth_q_setup(theta, p.nm_idx, data)
    lq_x = Float64(p.log_q_fwd)

    logW = 0.0
    for t in 1:n_bridge
        logW += (lp_x + beta * ll_x - lq_x - lp_A - beta * ll_A) / n_bridge
        t == n_bridge && break                  # nothing to relax after the last
        ll_x, lp_x, lq_x = _annealed_relax!(cand, slots, t / n_bridge, beta,
                                             ll_x, lp_x, lq_x, loglik, logprior,
                                             qset, p.nm_idx, n_relax, relax_scale, rng)
    end

    log_accept = logW + p.log_comb
    return (cand, convert(T, log_accept), ll_x)
end

# ActivityDecorrelation births are OLS-informed and already land near the
# mode; they keep the single-stage (ordinary RJ) move in both directions.
_annealed_stages(theta::Theta, nm_idx::Int, n_bridge::Int) =
    theta.params.config.noise_models[nm_idx] isa ActivityDecorrelation ? 1 : n_bridge

# One relaxation sweep block at bridge stage b of the geometric path
#   log rho_b(u) = (1 - b)·log q(u) + b·[log prior(θ,u) + beta·log L(θ,u)] + const,
# by coordinate random-walk Metropolis on the model's own slots. For a prior
# draw (q = the prior) this is prior(u)·L^(beta·b), the tempered posterior of
# the newborn; for an informed draw (GP hints) the q term is what keeps the
# kernel rho_b-invariant — without it the AIS weights do not match the path.
function _annealed_relax!(cand::Theta{T}, slots, b::Float64, beta, ll_x, lp_x, lq_x,
                           loglik, logprior, qset, nm_idx::Int, n_relax::Int,
                           relax_scale, rng) where {T}
    layout = cand.params.layout
    tgt(lp, ll, lq) = (1 - b) * lq + b * (lp + beta * ll)
    for _ in 1:n_relax
        for slot in slots
            uf = findfirst(==(slot), layout.unfrozen_idx)
            uf === nothing && continue
            lo, hi = bounds(layout.unfrozen_priors[uf])
            old = cand.values[slot]
            span = (isfinite(hi) && isfinite(lo)) ? (hi - lo) : abs(old) + 1
            nv = old + relax_scale * span * randn(rng)
            (lo <= nv <= hi) || continue
            cand.values[slot] = convert(T, nv)
            ll_try = loglik(cand); lp_try = logprior(cand)
            lq_try = (isfinite(ll_try) && isfinite(lp_try)) ?
                     _noise_birth_q_eval(qset, cand, nm_idx, cand.values) : -Inf
            if isfinite(lq_try) &&
               log(rand(rng)) < tgt(lp_try, ll_try, lq_try) - tgt(lp_x, ll_x, lq_x)
                ll_x = ll_try; lp_x = lp_try; lq_x = lq_try
            else
                cand.values[slot] = old
            end
        end
    end
    return (ll_x, lp_x, lq_x)
end

"""
    propose_noise_death_annealed(theta, rng, toggleable, loglik, logprior; ...)
        -> (cand, log_accept, ll_new)

The reverse of [`propose_noise_birth_annealed`](@ref): pick an active toggleable
model, run the SAME bridge backwards from the current state (stages
T−1, …, 1, each relaxing the model's parameters under the stage's
rho_t-invariant kernel) accumulating

    log W_rev = −Σ_t (b_t − b_{t−1}) · [log rho_T(x_t) − log rho_0(x_t)],

then switch the model off. On the path space this is the time reversal of the
birth (Karagiannis & Andrieu 2013), so the pair is reversible; with
`n_bridge = 1` it is exactly the ordinary death `propose_noise_death`.
Returns the FULL log acceptance exponent and the post-death log-likelihood.
"""
function propose_noise_death_annealed(theta::Theta{T}, rng::AbstractRNG,
        toggleable::Vector{NoiseModel}, loglik, logprior;
        data::Union{Data,Nothing}=nothing,
        exclusion_groups::Vector{Vector{NoiseModel}}=Vector{NoiseModel}[],
        beta::Real=1.0, n_bridge::Int=8, n_relax::Int=2,
        relax_scale::Real=0.3,
        ll_current::Union{Nothing,Real}=nothing,
        lp_current::Union{Nothing,Real}=nothing) where {T}

    n_bridge >= 1 || throw(ArgumentError("n_bridge must be ≥ 1"))
    td = theta.td
    noise_models = theta.params.config.noise_models
    active_toggle = Int[]
    for nm in toggleable
        j = findfirst(==(nm), noise_models)
        j === nothing && continue
        is_noise_active(td, j) && push!(active_toggle, j)
    end
    isempty(active_toggle) && return (theta, convert(T, -Inf), NaN)
    nm_idx = active_toggle[rand(rng, 1:length(active_toggle))]
    n_bridge = _annealed_stages(theta, nm_idx, n_bridge)

    off_td = copy(td); deactivate_noise!(off_td, nm_idx)
    off = Theta{T}(theta.params, copy(theta.values); td = off_td)
    ll_A = loglik(off); lp_A = logprior(off)
    (isfinite(ll_A) && isfinite(lp_A)) || return (theta, convert(T, -Inf), NaN)

    cand = Theta{T}(theta.params, copy(theta.values); td = copy(td))
    slots = _noise_model_slots(cand, nm_idx)
    ll_x = ll_current === nothing ? loglik(cand)  : Float64(ll_current)
    lp_x = lp_current === nothing ? logprior(cand) : Float64(lp_current)
    (isfinite(ll_x) && isfinite(lp_x)) || return (theta, convert(T, -Inf), NaN)
    qset = _noise_birth_q_setup(off, nm_idx, data)
    lq_x = _noise_birth_q_eval(qset, cand, nm_idx, cand.values)
    isfinite(lq_x) || return (theta, convert(T, -Inf), NaN)

    logW = -(lp_x + beta * ll_x - lq_x - lp_A - beta * ll_A) / n_bridge
    for t in (n_bridge - 1):-1:1
        ll_x, lp_x, lq_x = _annealed_relax!(cand, slots, t / n_bridge, beta,
                                             ll_x, lp_x, lq_x, loglik, logprior,
                                             qset, nm_idx, n_relax, relax_scale, rng)
        logW -= (lp_x + beta * ll_x - lq_x - lp_A - beta * ll_A) / n_bridge
    end

    # Reverse-birth selection among the models activatable after the death,
    # against this death's choice among the active ones.
    n_act_after = length(_noise_activatable(off_td, toggleable, noise_models,
                                            exclusion_groups))
    log_comb = log(max(n_act_after, 1)) - log(length(active_toggle))
    return (off, convert(T, logW + log_comb), ll_A)
end

"""
    propose_noise_death(theta, rng, toggleable; data, exclusion_groups) -> (new_theta, log_q_ratio)

Deactivate a random active noise component. The Hastings ratio carries the
reverse birth's density at the killed parameters (`_noise_birth_q_eval`, from
the post-death state) and its selection probability among the models
activatable after the death (`_noise_activatable`, which needs the run's
`exclusion_groups`).
"""
function propose_noise_death(theta::Theta{T}, rng::AbstractRNG,
                              toggleable::Vector{NoiseModel};
                              scratch::Union{Theta{T}, Nothing}=nothing,
                              data::Union{Data, Nothing}=nothing,
                              exclusion_groups::Vector{Vector{NoiseModel}}=
                                  Vector{NoiseModel}[]) where {T}
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
    # The reverse birth picks uniformly among the models activatable in the
    # post-death state — the same set `propose_noise_birth` builds.
    n_inactive_after = max(length(_noise_activatable(new_td, toggleable,
                                                     noise_models,
                                                     exclusion_groups)), 1)

    # Reverse-birth density of the killed params — mirrors the −log_q_fwd in
    # `propose_noise_birth` (OLS-informed AD, GP hints, else the prior), built
    # on the POST-death state (`new_theta`, model already off), the state the
    # reverse birth would start from.
    qset = _noise_birth_q_setup(new_theta, nm_idx, data)
    log_q_rev = _noise_birth_q_eval(qset, theta, nm_idx, theta.values)
    isfinite(log_q_rev) || return (theta, convert(T, -Inf))

    log_q_ratio = log(n_inactive_after) - log(n_active) + log_q_rev

    return (new_theta, convert(T, log_q_ratio))
end
