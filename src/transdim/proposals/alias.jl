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

# Donors: the population members with at least one active planet. The birth
# and its death both build this list from the same `population` (in
# `sample_pt` the other replicas, fixed during this replica's move).
_donor_pool(population) = population === nothing ? Theta[] :
    [d for d in population if d.td !== nothing && d.td.n_planets_active > 0]

const _DONOR_JITTER = 0.01      # jitter σ as a fraction of the prior width

_donor_lse(v) = (m = maximum(v); isfinite(m) ? m + log(sum(exp.(v .- m))) : m)

"""
    _donor_log_q(params, k, vals, donors) -> Float64

Density of the DonorBirth draw for slot `k` at the values in `vals`: an equal
mixture over donors, and over each donor's active planets, of the jitter
kernel about that planet's parameters (wrapped for circular coordinates,
Gaussian elsewhere; the draw rejects instead of clamping, so the plain
Gaussian density is exact inside the prior), times the prior for slots with
no donor counterpart. Frozen slots are not drawn and do not enter.
"""
function _donor_log_q(params::Params, k::Int, vals::AbstractVector, donors)
    layout = params.layout
    circ = circular_indices(params)
    our = planet_slot_indices(layout.planet_blocks[k])
    terms = Float64[]
    for d in donors
        act = active_planets(d.td)
        for dk in act
            ds = planet_slot_indices(layout.planet_blocks[dk])
            n_match = min(length(our), length(ds))
            l = -log(length(donors)) - log(length(act))
            for i in eachindex(our)
                uf = findfirst(==(our[i]), layout.unfrozen_idx)
                uf === nothing && continue
                ps = layout.unfrozen_priors[uf]
                lo, hi = bounds(ps)
                x = Float64(vals[our[i]])
                if i <= n_match
                    μ = Float64(d.values[ds[i]])
                    σ = _DONOR_JITTER * (hi - lo)
                    if uf in circ || is_cyclic_inclination(ps)
                        P = CIRCULAR_PERIOD
                        l += _donor_lse([-0.5 * ((x - μ + n * P) / σ)^2 for n in -3:3]) -
                             log(σ) - 0.5 * log(2π)
                    else
                        (lo <= x <= hi) || (l = -Inf; break)
                        l += -0.5 * ((x - μ) / σ)^2 - log(σ) - 0.5 * log(2π)
                    end
                else
                    l += eval_packed_logpdf(x, layout.packed_priors, uf)
                end
            end
            push!(terms, l)
        end
    end
    return isempty(terms) ? -Inf : _donor_lse(terms)
end

"""
    propose_planet_birth(theta, rng, ::DonorBirth; data, population) -> (new_theta, log_q_ratio)

Clone a planet from a random donor in `population` with small Gaussian jitter.
The proposal is a mixture over donors, not symmetric: the Hastings ratio
carries `−_donor_log_q` at the drawn values, and the paired death,
`propose_planet_death(theta, rng, ::DonorBirth; population)`, carries the same
density at the killed values. Falls back to PriorBirth (both directions) if
there are no donors.
"""
function propose_planet_birth(theta::Theta{T}, rng::AbstractRNG,
                               ::DonorBirth;
                               data::Union{Data, Nothing}=nothing,
                               population::Union{AbstractVector{<:Theta}, Nothing}=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    donors = _donor_pool(population)
    isempty(donors) &&
        return propose_planet_birth(theta, rng, PriorBirth(); data=data, scratch=scratch)

    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active

    cand_slots = group_first_inactive(td, birth_group_keys(theta.params))
    isempty(cand_slots) && return (theta, convert(T, -Inf))
    k = cand_slots[rand(rng, 1:length(cand_slots))]
    G = length(cand_slots)

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
    jitter_scale = _DONOR_JITTER
    circ = circular_indices(theta.params)

    for i in 1:n_match
        donor_val = Float64(donor.values[donor_slots[i]])
        # Apply multiplicative jitter for positive params, additive for others
        uf_pos = findfirst(==(our_slots[i]), layout.unfrozen_idx)
        if uf_pos !== nothing
            ps = layout.unfrozen_priors[uf_pos]
            lo, hi = bounds(ps)
            jittered = donor_val + jitter_scale * (hi - lo) * randn(rng)
            # A circular angle wraps instead: the clamp piles every jitter that
            # crosses the seam onto the window edge -- a point mass, and a phase
            # or node the donor never had. Wrapped, the jitter is a wrapped
            # Gaussian, a function of circular distance only, so it is symmetric
            # whichever window slot k (or the donor's slot) is charted in. So
            # does a cyclic inclination, whose chart is one period of a
            # periodic density (src/circular.jl).
            # Elsewhere a draw outside the prior is rejected rather than
            # clamped: a clamp piles mass on the wall that the density in
            # `_donor_log_q` cannot carry.
            if uf_pos in circ || is_cyclic_inclination(ps)
                jittered = circular_relabel(jittered, lo, hi)
            elseif !(lo <= jittered <= hi)
                return (theta, convert(T, -Inf))
            end
            new_values[our_slots[i]] = convert(T, jittered)
        end
        # A frozen slot keeps its own value: slots with their own priors may
        # freeze different values, and the donor's is not this slot's.
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

    # Selection (+log G: forward slot choice uniform over the birth groups;
    # −log(n+1): the reverse death's pick) and the draw's own density.
    p_birth, _ = _birth_death_probs(n_active, max_k)
    _, p_death_new = _birth_death_probs(n_active + 1, max_k)
    log_q_fwd = _donor_log_q(theta.params, k, new_values, donors)
    isfinite(log_q_fwd) || return (theta, convert(T, -Inf))
    log_q_ratio = log(p_death_new) - log(n_active + 1) - log(p_birth) + log(G) -
                  log_q_fwd

    _sort_group_periods!(new_theta)
    return (new_theta, convert(T, log_q_ratio))
end


"""
    propose_planet_death(theta, rng, ::DonorBirth; data, population, scratch)

Death paired with DonorBirth: kills a uniformly chosen active planet and
carries the donor mixture density (`_donor_log_q`, same `population`) at the
killed values. With no donors it is the PriorBirth death, as the birth is.
"""
function propose_planet_death(theta::Theta{T}, rng::AbstractRNG, ::DonorBirth;
                               data::Union{Data, Nothing}=nothing,
                               population::Union{AbstractVector{<:Theta}, Nothing}=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    donors = _donor_pool(population)
    isempty(donors) &&
        return propose_planet_death(theta, rng, PriorBirth(); data=data, scratch=scratch)
    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active
    n_active > 0 || return (theta, convert(T, -Inf))
    active = active_planets(td)
    k = active[rand(rng, 1:length(active))]
    log_q_reverse = _donor_log_q(theta.params, k, theta.values, donors)
    isfinite(log_q_reverse) || return (theta, convert(T, -Inf))
    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        deactivate_planet!(scratch.td, k)
        new_theta = scratch
    else
        new_td = copy(td)
        deactivate_planet!(new_td, k)
        new_theta = Theta{T}(theta.params, copy(theta.values); td = new_td)
    end
    _, p_death = _birth_death_probs(n_active, max_k)
    p_birth_new, _ = _birth_death_probs(n_active - 1, max_k)
    G_after = length(group_first_inactive(new_theta.td, birth_group_keys(theta.params)))
    log_q_ratio = log(p_birth_new) + log(n_active) - log(p_death) +
                  log_q_reverse - log(max(G_after, 1))
    return (new_theta, convert(T, log_q_ratio))
end

