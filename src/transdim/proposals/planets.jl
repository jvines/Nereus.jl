# Planet birth and death: the trans-dimensional core.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# Planet birth/death
# =====================================================================

"""
    _birth_death_probs(n_active, max_k) -> (p_birth, p_death)

Compute birth/death selection probabilities at the boundaries.
At n=0, force birth. At n=max_k, force death. Otherwise 50/50.
"""
function _birth_death_probs(n_active::Int, max_k::Int)
    if n_active == 0
        return (1.0, 0.0)
    elseif n_active == max_k
        return (0.0, 1.0)
    else
        return (0.5, 0.5)
    end
end

# Swap the full parameter blocks of two SAME-MODE planet slots. Identical
# modes ⇒ identical slot layouts, so the positional zip is exact.
function _swap_planet_blocks!(theta::Theta, a::Int, b::Int)
    sa = planet_slot_indices(theta.params.layout.planet_blocks[a])
    sb = planet_slot_indices(theta.params.layout.planet_blocks[b])
    @inbounds for i in eachindex(sa)
        v = theta.values[sa[i]]
        theta.values[sa[i]] = theta.values[sb[i]]
        theta.values[sb[i]] = v
    end
    return nothing
end

"""
    _sort_group_periods!(theta) -> theta

Canonical labeling for trans-dim states: within each planet-mode group,
keep the ACTIVE slots period-sorted by swapping parameter blocks
(insertion sort; activation bits untouched). Called at the end of every
birth proposal — a freshly-born planet is inserted into period order
instead of being left wherever the free slot was. Deaths cannot disorder
a sorted state, and the per-group ordering gate in `rv_log_likelihood`
rejects within-model moves that would disorder, so the chain stays on the
single canonical labeling and the k! permutation degeneracy is broken
exactly as in fixed-dim (no evidence inflation). Detailed balance is
unaffected: slot priors within a group are identical, so the proposal
density is labeling-invariant, and the death move picks planets
slot-agnostically. No-op for fixed-dim states.
"""
function _sort_group_periods!(theta::Theta)
    td = theta.td
    td === nothing && return theta
    modes = theta.params.config.planet_modes
    K = length(td.planet_active)
    @inbounds for i in 2:K
        td.planet_active[i] || continue
        j = i
        while j > 1
            p = j - 1
            while p >= 1 && !(td.planet_active[p] && modes[p] == modes[j])
                p -= 1
            end
            p < 1 && break
            if planet_P(theta, p) > planet_P(theta, j)
                _swap_planet_blocks!(theta, p, j)
                j = p
            else
                break
            end
        end
    end
    return theta
end

"""
    propose_planet_birth(theta, rng, ::PriorBirth) -> (new_theta, log_q_ratio)

Birth move: activate an inactive planet slot and draw its parameters
from the prior. Returns the new theta (with planet activated) and the
log proposal ratio for the RJMCMC acceptance.

The convention is that callers compose
    log_α = (log_L_new − log_L_old) + (log_π_new − log_π_old) + log_q_ratio
where `log_π` uses `_log_prior_transdim` (skips inactive slots) so
`Δlog_π` for activating slot k equals `Σ logpdf(slab_prior_k, β_new)`.
The Hastings ratio `log(q_rev/q_fwd)` for prior-draw birth + deterministic
death is `−Σ logpdf(slab_prior_k, β_new) + log_combinatorial`. Subtracting
the slab-prior term in `log_q_ratio` is what makes the prior cancel in
log_α (a previous version omitted it; the bias was invisible in standard
RJMCMC because ΔL≫|log_prior(β_new)| but lethal in MoMS-NS where the
constrained M-H has no ΔL term).
"""
function propose_planet_birth(theta::Theta{T}, rng::AbstractRNG,
                               ::PriorBirth;
                               data::Union{Data, Nothing}=nothing,
                               population::Union{AbstractVector{<:Theta}, Nothing}=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active

    # Pick the target slot uniformly among each mode-group's first inactive
    # slot (see `group_first_inactive`); the 1/G choice factor enters the
    # combinatorial term below.
    cand_slots = group_first_inactive(td, theta.params.config.planet_modes)
    isempty(cand_slots) && return (theta, convert(T, -Inf))  # all slots full
    k = cand_slots[rand(rng, 1:length(cand_slots))]
    G = length(cand_slots)

    # Draw new planet params from prior
    layout = theta.params.layout
    block = layout.planet_blocks[k]

    # Use scratch if available, otherwise allocate
    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        new_values = scratch.values
        new_td = scratch.td
    else
        new_values = copy(theta.values)
        new_td = copy(td)
    end

    log_q_fwd = zero(T)   # Σ log q_fwd(β_new) over all newly-drawn slot params

    for slot in planet_slot_indices(block)
        # Find this slot's position in the unfrozen arrays
        uf_pos = findfirst(==(slot), layout.unfrozen_idx)
        if uf_pos !== nothing
            prior = layout.unfrozen_priors[uf_pos]
            v = rand(rng, prior.dist)
            lo, hi = bounds(prior)
            v = clamp(v, lo, hi)
            new_values[slot] = convert(T, v)
            # q_fwd density for this slot (= prior density since drawn from prior)
            lp = eval_packed_logpdf(v,
                layout.packed_priors.type_ids[uf_pos],
                layout.packed_priors.params[uf_pos, 1],
                layout.packed_priors.params[uf_pos, 2],
                layout.packed_priors.lowers[uf_pos],
                layout.packed_priors.uppers[uf_pos])
            log_q_fwd += lp
        end
        # If slot is frozen, it keeps its frozen value (already in new_values)
    end

    # Activate planet in td
    activate_planet!(new_td, k)

    # Build result theta
    if scratch !== nothing
        new_theta = scratch
    else
        new_theta = Theta{T}(theta.params, new_values; td=new_td)
    end

    p_birth, _ = _birth_death_probs(n_active, max_k)
    _, p_death_new = _birth_death_probs(n_active + 1, max_k)
    # +log(G): forward slot choice is uniform over the G mode-group slots.
    log_comb = log(p_death_new) - log(n_active + 1) - log(p_birth) + log(G)

    # log q_ratio = log_comb − log q_fwd(β_new)
    # The −log_q_fwd term cancels the slab-prior gain that Δlog_pi adds
    # via _log_prior_transdim, so log_α reduces to ΔL + log_comb (standard
    # RJMCMC) or just log_comb (MoMS-NS constrained M-H).
    log_q_ratio = log_comb - log_q_fwd
    _sort_group_periods!(new_theta)

    return (new_theta, convert(T, log_q_ratio))
end

"""
    propose_planet_death(theta, rng) -> (new_theta, log_q_ratio)

Death move: deactivate a randomly chosen active planet. Returns the
new theta (with planet deactivated) and the log proposal ratio.
"""
function propose_planet_death(theta::Theta{T}, rng::AbstractRNG;
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active

    n_active > 0 || return (theta, convert(T, -Inf))  # nothing to kill

    # Pick uniformly among active planets
    active = active_planets(td)
    k = active[rand(rng, 1:length(active))]

    # Use scratch if available, otherwise allocate
    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        deactivate_planet!(scratch.td, k)
        new_theta = scratch
    else
        new_td = copy(td)
        deactivate_planet!(new_td, k)
        new_values = copy(theta.values)
        new_theta = Theta{T}(theta.params, new_values; td=new_td)
    end

    # Proposal ratio (reverse of birth)
    _, p_death = _birth_death_probs(n_active, max_k)
    p_birth_new, _ = _birth_death_probs(n_active - 1, max_k)

    # log q_ratio = log(p_birth_new) + log(n_active) - log(p_death)
    # −log(G_after): the reverse birth picks the slot uniformly among the
    # post-death mode-group slots (mirrors the +log(G) in the births).
    G_after = length(group_first_inactive(new_theta.td,
                                           theta.params.config.planet_modes))
    log_q_ratio = log(p_birth_new) + log(n_active) - log(p_death) -
                  log(max(G_after, 1))

    return (new_theta, convert(T, log_q_ratio))
end

"""
    propose_planet_death(theta, rng, strategy::BirthStrategy; scratch)

Fallback for birth strategies with no dedicated death: use the
strategy-agnostic uniform-pick death. `InformedBirth`/`JointInformedBirth`
are documented as pairing with this one. Keeps the strategy-dispatched call
total so callers can pass whatever strategy they selected.
"""
propose_planet_death(theta::Theta{T}, rng::AbstractRNG, ::BirthStrategy;
                      scratch::Union{Theta{T}, Nothing}=nothing) where {T} =
    propose_planet_death(theta, rng; scratch=scratch)

"""
    propose_planet_death(theta, rng, ::PriorBirth; scratch) -> (new_theta, log_q_ratio)

Death paired with `PriorBirth`. The reverse move is a prior-draw birth, so the
Hastings ratio must carry that birth's density at the values being killed,
`Σ logpdf(slab_prior_k, β_killed)` — exactly mirroring the `−Σ logpdf(...)`
the forward birth subtracts, and mirroring how
`propose_planet_death(::MoMSBirth)` carries its own reverse density.

Omitting it does NOT cancel against Δlog_π: `log_prior` skips inactive slots,
so a death already loses `Σ logpdf(slab_prior, β_killed)` from the prior term,
and with a purely combinatorial ratio the round trip
`log_q_birth + log_q_death` comes out at `−Σ logpdf(...)` — order +10 nats,
i.e. deaths over-accepted by e^10, biasing occupancy toward low Np. The
strategy-agnostic 2-argument method is left as-is: it is the documented
reverse for the informed births, whose own log_q is computed against it.
"""
function propose_planet_death(theta::Theta{T}, rng::AbstractRNG,
                               ::PriorBirth;
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active

    n_active > 0 || return (theta, convert(T, -Inf))

    active = active_planets(td)
    k = active[rand(rng, 1:length(active))]

    layout = theta.params.layout
    block  = layout.planet_blocks[k]

    # Reverse-proposal density q_birth(β_killed): the prior-draw birth would
    # have to redraw exactly these values. Same packed-prior evaluation the
    # forward birth uses, over the same slots (frozen slots are skipped in
    # both directions, so they cancel).
    log_q_reverse = zero(T)
    for slot in planet_slot_indices(block)
        uf_pos = findfirst(==(slot), layout.unfrozen_idx)
        uf_pos === nothing && continue
        log_q_reverse += eval_packed_logpdf(Float64(theta.values[slot]),
            layout.packed_priors.type_ids[uf_pos],
            layout.packed_priors.params[uf_pos, 1],
            layout.packed_priors.params[uf_pos, 2],
            layout.packed_priors.lowers[uf_pos],
            layout.packed_priors.uppers[uf_pos])
    end

    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        deactivate_planet!(scratch.td, k)
        new_theta = scratch
    else
        new_td = copy(td)
        deactivate_planet!(new_td, k)
        new_theta = Theta{T}(theta.params, copy(theta.values); td=new_td)
    end

    _, p_death = _birth_death_probs(n_active, max_k)
    p_birth_new, _ = _birth_death_probs(n_active - 1, max_k)
    G_after = length(group_first_inactive(new_theta.td,
                                           theta.params.config.planet_modes))
    log_q_ratio = log(p_birth_new) + log(n_active) - log(p_death) +
                  log_q_reverse - log(max(G_after, 1))

    return (new_theta, convert(T, log_q_ratio))
end

