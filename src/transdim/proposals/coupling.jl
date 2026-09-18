# Astrometric-coupling birth/death: the (K, S_1..S_K) model space.
#
# Flips WHICH OBSERVABLE constrains a companion, not whether it exists.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# Astrometric-coupling birth/death
# =====================================================================
#
# Flips whether companion k's reflex is fitted to the astrometry. Unlike the
# planet and noise toggles this is not a change in what EXISTS — the companion
# is there either way and the RV still sees it — but in which observable
# constrains it. That is the (K, S_1..S_K) model space: component count AND
# per-component data-source assignment.
#
# The move is a pure dimension extension, checked before it was written: mass in
# RVASBlock is DERIVED from (K, P, e, sin i, M_pri), so no parameter changes
# meaning when the flag flips and there is no Jacobian beyond the proposal
# density. Contrast the 9p<->Keplerian astrometric jump, which is a genuine
# reparametrisation.
#
# HOW MANY PARAMETERS APPEAR depends on what the planet already has, and the
# asymmetry is physical rather than bookkeeping:
#   RVAS    (no transit): (inc, Omega) — m sin i becomes a true mass
#   RVPMAS  (transiting): (Omega)      — the transit already fixed inc
# So coupling a transiting planet buys the NODE, not the mass. Omega is what
# mutual inclination is made of.
#
# The proposal is a blind prior draw. That is a deliberate choice, not laziness:
# the blind-birth bias measured against dimension (toy_blind_birth_dimension.jl)
# is 0.050 nats at K=1 and 0.148 at K=6, so at the one or two parameters this
# move adds it is ~0.05 nats — far below anything that matters here. A
# Thiele-Innes OLS-informed proposal would improve MIXING and is the natural
# next step, but it is an efficiency question, not a correctness one.

"Slots the astrometric coupling controls for planet `k`, or empty."
function _as_coupling_slots(theta::Theta, k::Int)
    block = theta.params.layout.planet_blocks[k]
    has_AS(block) || return Int[]
    s = Int[]
    hasproperty(block, :inc)   && block.inc   > 0 && push!(s, block.inc)
    hasproperty(block, :Omega) && block.Omega > 0 && push!(s, block.Omega)
    return s
end

"Planet indices whose coupling can be toggled: active, astrometry-bearing."
function _as_togglable(theta::Theta)
    out = Int[]
    for k in planet_indices(theta)
        isempty(_as_coupling_slots(theta, k)) && continue
        push!(out, k)
    end
    return out
end

"""
    propose_as_birth(theta, rng; scratch) -> (new_theta, log_q_ratio)

Couple a currently-decoupled companion to the astrometry, drawing the newly
meaningful orientation parameters from their priors.
"""
function propose_as_birth(theta::Theta{T}, rng::AbstractRNG;
                          scratch::Union{Theta{T},Nothing}=nothing) where {T}
    theta.td === nothing && return (theta, convert(T, -Inf))
    cand = [k for k in _as_togglable(theta) if !is_as_active(theta.td, k)]
    isempty(cand) && return (theta, convert(T, -Inf))
    n_off = length(cand)
    k = cand[rand(rng, 1:n_off)]

    layout = theta.params.layout
    new_values = scratch !== nothing ? (scratch.values .= theta.values; scratch.values) :
                                        copy(theta.values)
    new_td = scratch !== nothing ? (copy_into!(scratch.td, theta.td); scratch.td) :
                                    copy(theta.td)

    log_q_fwd = zero(T)
    for slot in _as_coupling_slots(theta, k)
        uf = findfirst(==(slot), layout.unfrozen_idx)
        uf === nothing && continue
        prior = layout.unfrozen_priors[uf]
        lo, hi = bounds(prior)
        v = clamp(rand(rng, prior.dist), lo, hi)
        new_values[slot] = convert(T, v)
        log_q_fwd += eval_packed_logpdf(v,
            layout.packed_priors.type_ids[uf],
            layout.packed_priors.params[uf, 1],
            layout.packed_priors.params[uf, 2],
            layout.packed_priors.lowers[uf],
            layout.packed_priors.uppers[uf])
    end
    activate_as!(new_td, k)

    new_theta = scratch !== nothing ? scratch :
                Theta{T}(theta.params, new_values; td=new_td)
    n_on_after = count(j -> is_as_active(new_td, j), _as_togglable(theta))
    log_q_ratio = log(n_on_after) - log(n_off) - log_q_fwd
    return (new_theta, convert(T, log_q_ratio))
end

"""
    propose_as_death(theta, rng; scratch) -> (new_theta, log_q_ratio)

Decouple a companion from the astrometry. Its orientation parameters revert to
carrying their priors, which integrate to 1 and so leave the marginal
likelihood untouched — the reverse-birth density here is what cancels the prior
term the deactivation gains, so the toggle samples P(coupled | data) rather
than a prior-tilted pseudo-posterior.
"""
function propose_as_death(theta::Theta{T}, rng::AbstractRNG;
                          scratch::Union{Theta{T},Nothing}=nothing) where {T}
    theta.td === nothing && return (theta, convert(T, -Inf))
    cand = [k for k in _as_togglable(theta) if is_as_active(theta.td, k)]
    isempty(cand) && return (theta, convert(T, -Inf))
    n_on = length(cand)
    k = cand[rand(rng, 1:n_on)]

    layout = theta.params.layout
    new_td = scratch !== nothing ? (copy_into!(scratch.td, theta.td); scratch.td) :
                                    copy(theta.td)
    scratch !== nothing && (scratch.values .= theta.values)
    deactivate_as!(new_td, k)
    new_theta = scratch !== nothing ? scratch :
                Theta{T}(theta.params, copy(theta.values); td=new_td)

    log_q_rev = zero(T)
    for slot in _as_coupling_slots(theta, k)
        uf = findfirst(==(slot), layout.unfrozen_idx)
        uf === nothing && continue
        log_q_rev += eval_packed_logpdf(theta.values[slot],
            layout.packed_priors.type_ids[uf],
            layout.packed_priors.params[uf, 1],
            layout.packed_priors.params[uf, 2],
            layout.packed_priors.lowers[uf],
            layout.packed_priors.uppers[uf])
    end
    n_off_after = length(_as_togglable(theta)) -
                  count(j -> is_as_active(new_td, j), _as_togglable(theta))
    log_q_ratio = log(n_off_after) - log(n_on) + log_q_rev
    return (new_theta, convert(T, log_q_ratio))
end

