# MoMS (Mixtures of Mutually Singular distributions) proposals.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# MoMS (Mixtures of Mutually Singular distributions) proposals.
# van den Bergh, Clyde, Raftery, Marsman (arXiv:2604.27791).
#
# Each inactive planet sits at a designated off-location β_off; an add
# move proposes new parameters via a Gaussian random walk centered on
# β_off; a delete move resets to β_off. All in fixed-dim parameter
# space — no Jacobian, no auxiliary variables.
#
# Equivalent to RJMCMC with identity dimension-map; van den Bergh+ note
# the equivalence at the M-H acceptance level (their §2.5).
# =====================================================================

"""
    spike_slab_log_prior(theta, strategy, inclusion_prior) -> Float64

Spike-and-slab log-prior for the MoMS parameterisation: the continuous
prior `π_β(β_k)` is integrated against the spike `δ(β_k − β_k_off)` when
γ_k = 0 (contributing 0 to the log-prior since the delta is normalised),
and against the slab `π_β` when γ_k = 1. The Bernoulli prior on each γ_k
contributes `log(p_inc)` or `log(1 − p_inc)`.

Nereus's `log_prior(theta)` always sums the *continuous* prior over
every unfrozen slot — including planet slots whose planet is inactive
(i.e. β at the off-location). That over-counts by `log π_β(β_off)` per
inactive planet, biasing trans-dim sampling and Bayes-factor comparisons
between dimensionalities. This helper returns the correctly-normalised
spike-and-slab log-prior built as a correction on top of `log_prior`,
so external priors / physical-bound enforcement / etc. are preserved.

Returns `-Inf` if `log_prior(theta)` does (out-of-bounds, etc.).
"""
function spike_slab_log_prior(theta::Theta, strategy::MoMSBirth,
                                inclusion_prior::Float64)
    base = log_prior(theta)
    isfinite(base) || return base

    # Spike-and-slab joint prior:
    #   π(γ, β) = Π_k [π_γ(γ_k) · (γ_k · π_slab(β_k) + (1-γ_k) · δ(β_k - off_k))]
    #
    # `log_prior` (via `_log_prior_transdim`) already SKIPS inactive
    # slots, so it does not contribute log π_slab(β_k) for γ_k=0. The
    # spike-and-slab structure adds the indicator factors π_γ on top.
    # An earlier version of this function additionally subtracted
    # logpdf(prior, off_value) for inactive slots — that was a phantom
    # correction (nothing to cancel) and biased Δlog_pi for any γ-flip
    # by an extra +log_prior(off), making MoMS-NS births effectively
    # impossible because the cancellation in the M-H ratio depends on
    # this not being there.
    correction = 0.0
    @inbounds for k in 1:length(theta.td.planet_active)
        correction += theta.td.planet_active[k] ? log(inclusion_prior) :
                                                  log(1.0 - inclusion_prior)
    end
    return base + correction
end

"""Estimate (off-value, scale) for a slot's prior by sampling.

Sample-based estimation auto-adapts to any prior shape:
  * For symmetric priors (Uniform, Normal) the mean is the midpoint
    and the stddev is a fraction of the support width.
  * For log-priors (LogUniform) the *median* of samples sits at the
    geometric mean and the sample stddev shrinks proportionally with
    the lower bound — exactly the multiplicative scale a random walk
    needs.
  * For ModJeffreys / Beta the same robustness applies.

We use the sample median (robust) for the off-value and the sample
inter-quartile-range divided by 1.349 (≈ Gaussian stddev for
non-Gaussian samples) for the initial scale, capped at half the
prior width to avoid runaway proposals out-of-bounds.
"""
function _moms_off_and_scale(prior::PriorSpec, init_scale::Float64,
                              rng::AbstractRNG)
    K = 101
    samples = Float64[]
    sizehint!(samples, K)
    for _ in 1:K
        v = rand(rng, prior.dist)
        isfinite(v) && push!(samples, v)
    end
    isempty(samples) && return (0.0, init_scale)
    sort!(samples)
    n = length(samples)
    med = samples[(n + 1) ÷ 2]
    q1  = samples[max(1, (n + 3) ÷ 4)]
    q3  = samples[min(n, (3n + 1) ÷ 4)]
    iqr_stddev = (q3 - q1) / 1.349
    iqr_stddev > 0 || (iqr_stddev = max(abs(med) * 0.1, 1e-3))
    lo, hi = bounds(prior)
    width = isfinite(lo) && isfinite(hi) ? (hi - lo) : Inf
    scale = clamp(init_scale * iqr_stddev, 0.0, isfinite(width) ? width / 2 : Inf)
    return (med, scale)
end

"""
    MoMSBirth(params; init_scale=1.0, rng=Random.default_rng())

Build a MoMSBirth strategy whose off-values are the sample medians of
the priors of each planet's unfrozen slots, and whose initial proposal
scales are derived from the prior's sample inter-quartile range
(≈ Gaussian stddev) — the scales are subsequently adapted by
`sample_moms` via Robbins-Monro during warmup.

`init_scale = 1.0` means "one IQR-stddev"; lower values give tighter
proposals (more conservative).
"""
function MoMSBirth(params::Params; init_scale::Float64 = 1.0,
                    rng::AbstractRNG = Random.default_rng())
    layout = params.layout
    max_k = params.config.max_kplanet
    off_values = Vector{Vector{Float64}}(undef, max_k)
    scales     = Vector{Vector{Float64}}(undef, max_k)
    slot_idx   = Vector{Vector{Int}}(undef, max_k)
    for k in 1:max_k
        block = layout.planet_blocks[k]
        offs = Float64[]
        scs  = Float64[]
        idxs = Int[]
        for slot in planet_slot_indices(block)
            uf_pos = findfirst(==(slot), layout.unfrozen_idx)
            uf_pos === nothing && continue   # frozen slot, not part of MoMS state
            prior = layout.unfrozen_priors[uf_pos]
            off, scale = _moms_off_and_scale(prior, init_scale, rng)
            push!(offs, off)
            push!(scs,  scale)
            push!(idxs, uf_pos)
        end
        off_values[k] = offs
        scales[k]     = scs
        slot_idx[k]   = idxs
    end
    return MoMSBirth(off_values, scales, slot_idx)
end

# log P(lo ≤ N(μ,σ) ≤ hi): the normalizer of the bounds-TRUNCATED Gaussian
# random-walk kernel the MoMS birth actually samples from (out-of-bounds draws
# are rejected/retried). Omitting it — using the raw Gaussian density — makes
# the birth too easy and the deterministic death too hard by |log Z| each, so
# the stationary P(Np) is inflated by ~2·Σ|log Z| nats (≈1.9 on the white-noise
# DB gate; the per-slot |log Z| are 0.2–0.3 for the default prior-IQR scales).
# Carrying it in BOTH the forward birth density and the reverse death density
# restores detailed balance.
@inline function _log_trunc_norm(μ::Float64, σ::Float64, lo, hi)
    σ > 0 || return 0.0
    zhi = isfinite(hi) ? cdf(Normal(), (hi - μ) / σ) : 1.0
    zlo = isfinite(lo) ? cdf(Normal(), (lo - μ) / σ) : 0.0
    Z = zhi - zlo
    Z > 0 ? log(Z) : -Inf
end

"""
    propose_planet_birth(theta, rng, ::MoMSBirth; data, scratch) -> (new_theta, log_q_ratio)

MoMS-flavored birth: activate an inactive planet slot and propose its
parameters via the Algorithm-2 Gaussian random walk kernel of van den
Bergh+ 2026 (arXiv:2604.27791, Eq. 6). The paper's kernel is
`q(β*|γ*=1, β) = N(β*; β, τ²)` — a Gaussian centered on the **current**
β, not on the off-value. Because Nereus's death move (see
`propose_planet_death(::MoMSBirth)`) deterministically resets β to
`strategy.off_values[k]` whenever γ_k = 0, the current β equals
`off_values[k]` at every birth attempt, so numerically the proposal
mean is `off_values[k]`. The two characterisations are equivalent for
the as-implemented kernel but become distinct if a future variant
allows multi-step or partial births — keep that in mind before
generalising.

The Gaussian density evaluated at the proposed β* is contributed to
`log_q_ratio`; the matching reverse density (death → off, deterministic)
contributes 0, so the only correction the M-H acceptance needs is the
forward Gaussian + the combinatorial p_birth/p_death factor.

Truncates proposals to prior support; out-of-support proposals are
rejected via `log_q_ratio = -Inf`.
"""
function propose_planet_birth(theta::Theta{T}, rng::AbstractRNG,
                               strategy::MoMSBirth;
                               data::Union{Data, Nothing}=nothing,
                               population::Union{AbstractVector{<:Theta}, Nothing}=nothing,
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active

    cand_slots = group_first_inactive(td, theta.params.config.planet_modes)
    isempty(cand_slots) && return (theta, convert(T, -Inf))
    k = cand_slots[rand(rng, 1:length(cand_slots))]
    G = length(cand_slots)

    layout = theta.params.layout
    offs   = strategy.off_values[k]
    scs    = strategy.scales[k]
    uf_idx = strategy.slot_indices[k]
    isempty(uf_idx) && return (theta, convert(T, -Inf))

    # Propose new values via random walk around off-location, with up
    # to 8 retries if a draw lands outside prior bounds. The
    # rejected-and-retried draws are not part of the M-H proposal —
    # the recorded log_q corresponds only to the accepted (in-bounds)
    # draw, which is the right density for the truncated proposal in
    # the limit where retries succeed (high acceptance regime).
    proposed = similar(offs)
    log_q_propose = 0.0
    for i in eachindex(uf_idx)
        prior = layout.unfrozen_priors[uf_idx[i]]
        lo, hi = bounds(prior)
        v = NaN
        for _ in 1:8
            v_try = offs[i] + scs[i] * randn(rng)
            if (!isfinite(lo) || v_try >= lo) && (!isfinite(hi) || v_try <= hi)
                v = v_try
                break
            end
        end
        isnan(v) && return (theta, convert(T, -Inf))
        proposed[i] = v
        log_q_propose += -0.5 * ((v - offs[i]) / scs[i])^2 - log(scs[i] * sqrt(2π)) -
                         _log_trunc_norm(offs[i], scs[i], lo, hi)
    end

    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        new_values = scratch.values
        new_td     = scratch.td
    else
        new_values = copy(theta.values)
        new_td     = copy(td)
    end

    for i in eachindex(uf_idx)
        slot = layout.unfrozen_idx[uf_idx[i]]
        new_values[slot] = convert(T, proposed[i])
    end
    activate_planet!(new_td, k)

    new_theta = scratch !== nothing ? scratch :
                Theta{T}(theta.params, new_values; td=new_td)

    # Combinatorial birth/death balance, plus the M-H proposal correction:
    #   log_q_ratio = log(p_death_new) - log(n_active+1) - log(p_birth)
    #               + log q(β | β*) [reverse, =1 since deterministic delete]
    #               - log q(β* | β_off) [forward random walk density]
    p_birth, _ = _birth_death_probs(n_active, max_k)
    _, p_death_new = _birth_death_probs(n_active + 1, max_k)
    # +log(G): forward slot choice uniform over mode-group slots.
    log_q_ratio = log(p_death_new) - log(n_active + 1) - log(p_birth) -
                  log_q_propose + log(G)

    _sort_group_periods!(new_theta)
    return (new_theta, convert(T, log_q_ratio))
end

"""
    propose_planet_death(theta, rng, ::MoMSBirth; scratch) -> (new_theta, log_q_ratio)

MoMS-flavored death: deactivate a uniformly chosen active planet and
deterministically reset its parameters to `strategy.off_values[k]`.
The reverse move is the random-walk add, whose density is included in
`log_q_ratio`.
"""
function propose_planet_death(theta::Theta{T}, rng::AbstractRNG,
                               strategy::MoMSBirth;
                               scratch::Union{Theta{T}, Nothing}=nothing) where {T}
    td = theta.td
    max_k = length(td.planet_active)
    n_active = td.n_planets_active
    n_active > 0 || return (theta, convert(T, -Inf))

    active = active_planets(td)
    k = active[rand(rng, 1:length(active))]

    layout = theta.params.layout
    offs   = strategy.off_values[k]
    scs    = strategy.scales[k]
    uf_idx = strategy.slot_indices[k]

    # Compute the reverse-proposal density q(β_old | β_off) before resetting,
    # including the bounds-truncation normalizer (MUST match the forward birth).
    log_q_reverse = 0.0
    for i in eachindex(uf_idx)
        slot = layout.unfrozen_idx[uf_idx[i]]
        v = Float64(theta.values[slot])
        lo, hi = bounds(layout.unfrozen_priors[uf_idx[i]])
        log_q_reverse += -0.5 * ((v - offs[i]) / scs[i])^2 - log(scs[i] * sqrt(2π)) -
                         _log_trunc_norm(offs[i], scs[i], lo, hi)
    end

    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        new_values = scratch.values
        new_td     = scratch.td
    else
        new_values = copy(theta.values)
        new_td     = copy(td)
    end

    for i in eachindex(uf_idx)
        slot = layout.unfrozen_idx[uf_idx[i]]
        new_values[slot] = convert(T, offs[i])
    end
    deactivate_planet!(new_td, k)

    new_theta = scratch !== nothing ? scratch :
                Theta{T}(theta.params, new_values; td=new_td)

    p_birth_new, _ = _birth_death_probs(n_active - 1, max_k)
    _, p_death     = _birth_death_probs(n_active, max_k)
    # −log(G_after): mirror of the births' uniform mode-group slot choice.
    G_after = length(group_first_inactive(new_td,
                                           theta.params.config.planet_modes))
    log_q_ratio = log(p_birth_new) + log(n_active) - log(p_death) +
                  log_q_reverse - log(max(G_after, 1))

    return (new_theta, convert(T, log_q_ratio))
end


