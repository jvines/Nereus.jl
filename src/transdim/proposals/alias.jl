# Within-model alias / harmonic period jumps.
#
# Not a dimension change: a planet born at 2P of the truth cannot migrate there
# by local steps, and cannot escape by death-then-rebirth either because the
# death has to be accepted first. Green's update move.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# Within-model alias / harmonic mode jump
# =====================================================================

"""
Multiplicative period ratios for `propose_planet_alias_jump`. The set is
CLOSED UNDER INVERSION (2↔1/2, 3↔1/3, 3/2↔2/3) — that is what makes the
proposal symmetric, so the Hastings ratio reduces to the Jacobian alone.
"""
const ALIAS_RATIOS = (2.0, 0.5, 3.0, 1/3, 1.5, 2/3)

"""
    propose_planet_alias_jump(theta, rng; scratch) -> (new_theta, log_q_ratio)

Within-model update move: pick an ACTIVE planet and multiply its period by a
ratio drawn from [`ALIAS_RATIOS`](@ref), leaving every other parameter alone.

Birth/death alone cannot repair a planet that was born at a harmonic or alias
of the true period: the within-model kernels are local, and the likelihood
valley between P and 2P is exactly what a local step cannot cross. Death-then-
rebirth is not an escape either — the death has to be accepted first, and an
entrenched (if aliased) planet is usually a big likelihood loss to drop. This
is Green's update move for the period coordinate.

The map P → rP is deterministic with Jacobian r, and the ratio draw is
symmetric (q(r) = q(1/r) by construction of the ratio set), so

    log_q_ratio = log(q_reverse / q_forward) + log|J| = 0 + log(r)

Returns `-Inf` when there is nothing to move, and under the `:a_driven`
parametrization, where the period slot holds a semi-major axis rather than a
period and the map above is not the intended one.

The planet keeps its slot — no canonical re-sort — so the move is its own
inverse family. If a period-ordering gate is active it will simply reject the
jumps that would violate ordering, which costs acceptance, not correctness.
"""
function propose_planet_alias_jump(theta::Theta{T}, rng::AbstractRNG;
                                    scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    td = theta.td
    td === nothing && return (theta, convert(T, -Inf))
    td.n_planets_active > 0 || return (theta, convert(T, -Inf))
    theta.params.config.parametrization.mass === :a_driven &&
        return (theta, convert(T, -Inf))

    active = active_planets(td)
    k = active[rand(rng, 1:length(active))]
    r = ALIAS_RATIOS[rand(rng, 1:length(ALIAS_RATIOS))]

    block = theta.params.layout.planet_blocks[k]

    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        new_theta = scratch
    else
        new_theta = Theta{T}(theta.params, copy(theta.values); td=copy(td))
    end
    new_theta.values[block.P] = convert(T, theta.values[block.P] * r)

    return (new_theta, convert(T, log(r)))
end

# DonorBirth: clone a planet from a donor theta with jitter.
# Requires a population (vector of Theta). Falls back to PriorBirth
# when called without one or when no donors have active planets.

"""
    propose_planet_birth(theta, rng, ::DonorBirth; data, population) -> (new_theta, log_q_ratio)

Clone a planet from a random donor in `population` and apply small
Gaussian jitter. Symmetric proposal → log_q_ratio is just combinatorial.
Falls back to PriorBirth if no population or no suitable donors.
"""
function propose_planet_birth(theta::Theta{T}, rng::AbstractRNG,
                               ::DonorBirth;
                               data::Union{Data, Nothing}=nothing,
                               population::Union{AbstractVector{<:Theta}, Nothing}=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    # No population available → fall back
    if population === nothing || isempty(population)
        return propose_planet_birth(theta, rng, PriorBirth(); data=data, scratch=scratch)
    end

    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active

    cand_slots = group_first_inactive(td, theta.params.config.planet_modes)
    isempty(cand_slots) && return (theta, convert(T, -Inf))
    k = cand_slots[rand(rng, 1:length(cand_slots))]
    G = length(cand_slots)

    # Find donors that have at least one active planet
    donors = [d for d in population if d.td !== nothing && d.td.n_planets_active > 0]
    if isempty(donors)
        return propose_planet_birth(theta, rng, PriorBirth(); data=data, scratch=scratch)
    end

    # Pick a random donor and a random active planet from it
    donor = donors[rand(rng, 1:length(donors))]
    donor_active = active_planets(donor.td)
    dk = donor_active[rand(rng, 1:length(donor_active))]

    # Copy donor's planet params into our inactive slot k, with jitter
    layout = theta.params.layout
    our_block = layout.planet_blocks[k]
    donor_block = layout.planet_blocks[dk]

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

    our_slots = planet_slot_indices(our_block)
    donor_slots = planet_slot_indices(donor_block)

    # Match slots by position (both blocks have same param ordering)
    n_match = min(length(our_slots), length(donor_slots))
    jitter_scale = 0.01

    for i in 1:n_match
        donor_val = Float64(donor.values[donor_slots[i]])
        # Apply multiplicative jitter for positive params, additive for others
        uf_pos = findfirst(==(our_slots[i]), layout.unfrozen_idx)
        if uf_pos !== nothing
            lo, hi = bounds(layout.unfrozen_priors[uf_pos])
            jittered = donor_val + jitter_scale * (hi - lo) * randn(rng)
            jittered = clamp(jittered, lo, hi)
            new_values[our_slots[i]] = convert(T, jittered)
        else
            new_values[our_slots[i]] = convert(T, donor_val)
        end
    end

    # Any remaining slots (block type mismatch) drawn from prior
    for i in (n_match + 1):length(our_slots)
        uf_pos = findfirst(==(our_slots[i]), layout.unfrozen_idx)
        if uf_pos !== nothing
            prior = layout.unfrozen_priors[uf_pos]
            v = rand(rng, prior.dist)
            lo, hi = bounds(prior)
            new_values[our_slots[i]] = convert(T, clamp(v, lo, hi))
        end
    end

    activate_planet!(new_td, k)

    if scratch !== nothing
        new_theta = scratch
    else
        new_theta = Theta{T}(theta.params, new_values; td=new_td)
    end

    # Symmetric proposal → log_q_ratio is just combinatorial
    # (+log(G): forward slot choice uniform over mode-group slots).
    p_birth, _ = _birth_death_probs(n_active, max_k)
    _, p_death_new = _birth_death_probs(n_active + 1, max_k)
    log_q_ratio = log(p_death_new) - log(n_active + 1) - log(p_birth) + log(G)

    _sort_group_periods!(new_theta)
    return (new_theta, convert(T, log_q_ratio))
end

