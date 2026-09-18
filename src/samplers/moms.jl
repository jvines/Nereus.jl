# MoMS sampler — Mixtures of Mutually Singular distributions for trans-
# dimensional Bayesian variable selection (van den Bergh, Clyde,
# Raftery, Marsman, arXiv:2604.27791).
#
# Conceptually equivalent to RJMCMC with an identity dimension-map: each
# inactive planet sits at a designated "off" location β_off, and an add
# move proposes new planet parameters via a Gaussian random walk
# centered on β_off. A delete move resets to β_off. The Metropolis-
# Hastings acceptance ratio uses the standard fixed-dim form — no
# Jacobian, no auxiliary variables, no dimension-matching transformation.
#
# Departures from the paper that need acknowledging in any write-up:
#   1. **Trans-dim move is paper Algorithm 2**, with Robbins-Monro
#      scale adaptation per Eq. D1 (φ=0.75, target α_*=0.234 for
#      vector groups — the paper's univariate α_*=0.44 is the right
#      choice only for scalar β_i flips, see RGG 1997).
#   2. **Within-model kernel is NOT MoMS Algorithm 2 line 11** — the
#      paper does Gibbs on σ² and coefficients (requires conjugacy);
#      Nereus does slice sampling (`:slice`) or random-walk
#      Metropolis (`:rwm`) on the active-subset coordinates, both
#      black-box and target-agnostic. Cite Skilling 2003 / Neal 2003
#      for these, not the MoMS paper. The fixed-γ marginal is left
#      invariant by either kernel, so detailed balance of the joint
#      chain w.r.t. the spike-and-slab prior is preserved.
#   3. **Vector-valued groups**: paper does single-coefficient β_i
#      flips, Nereus flips entire planet blocks (5–7 dims per add
#      move). The Eq. D1 derivation extends straightforwardly to
#      vector flips with the multivariate target acceptance 0.234
#      (Roberts, Gelman, Gilks 1997).
#   4. **Always-on parameters**: paper has none, Nereus treats system
#      systematics (γ_inst, σ_inst, q1, q2, ρ_⋆, ...) as always-active
#      with arbitrary continuous priors. Eq. 6's M-H ratio is
#      invariant under marginalising out always-on coordinates because
#      the proposal density factorises and they appear identically on
#      both sides of the ratio.
#   5. **Non-zero off-values**: paper fixes β_off = 0, Nereus uses
#      a per-group off-value vector (typically near the prior median).
#      The M-H ratio is invariant under translation of off-values.
#
# Cite: van den Bergh, Clyde, Raftery, Marsman 2026, "Reversible Jump
# MCMC With No Regrets" (arXiv:2604.27791).

using Random
using MCMCChains

export sample_moms

"""
    sample_moms(target, data; td, kwargs...) -> (chains::MCMCChains.Chains,
                                                  evals::Int,
                                                  strategy::MoMSBirth)

Run a MoMS sampler for trans-dimensional inference. Returns the chain,
the total number of likelihood evaluations, and the (mutated) MoMSBirth
strategy carrying the adapted per-planet proposal scales — useful for
warm-starting a follow-up run.

# Required
- `target::NereusTarget` — the Bayesian target (`unconstrained=false`)
- `data::Data` — observation data
- `td::TransDimConfig` — trans-dim configuration. Any `birth_strategies`
  passed to `td` are ignored — `sample_moms` always uses MoMS proposals.

# Keywords
- `n_samples::Int = 5000` — posterior samples to store
- `n_warmup::Int  = 5000` — warmup iterations (adapt MoMS scales + slice
  widths)
- `seed::Int = 1`         — RNG seed
- `init_scale::Float64 = 0.3` — initial proposal scale as a fraction of
  prior width per slot
- `target_birth_accept::Float64 = 0.234` — Robbins-Monro target during
  warmup adaptation
- `inclusion_prior::Float64 = 0.5` — Bernoulli prior P(γ_k = 1) per planet,
  used in the spike-and-slab log-prior (Bayes factors between N_p
  configurations are unbiased only when this is set deliberately by the
  user; default 0.5 corresponds to a uniform prior over models).
"""
function sample_moms(
    target::NereusTarget,
    data::Data;
    td::TransDimConfig,
    n_samples::Int = 5000,
    n_warmup::Int  = 5000,
    seed::Int      = 1,
    n_chains::Int = 1,
    init_scale::Real = 0.3,
    target_birth_accept::Real = 0.234,
    inclusion_prior::Real = 0.5,
    show_progress::Bool = true,
    progress_every::Int = max(1000, n_samples ÷ 20),
    within_model::Symbol = :slice,
    informed_birth_fraction::Real = 0.0,
)
    # JSON may deliver these floats as Int; normalize.
    init_scale              = Float64(init_scale)
    target_birth_accept     = Float64(target_birth_accept)
    inclusion_prior         = Float64(inclusion_prior)
    informed_birth_fraction = Float64(informed_birth_fraction)
    0.0 < inclusion_prior < 1.0 || throw(ArgumentError(
        "inclusion_prior must be in (0, 1)"))
    within_model in (:slice, :rwm) || throw(ArgumentError(
        "within_model must be :slice or :rwm; got :$within_model"))
    0.0 <= informed_birth_fraction <= 1.0 || throw(ArgumentError(
        "informed_birth_fraction must be in [0, 1]"))
    n_chains >= 1 || throw(ArgumentError("n_chains must be ≥ 1"))

    if n_chains > 1
        # Multi-chain: spawn n_chains independent samplers. Each runs
        # its own MoMSBirth strategy with its own warmup-adapted scales,
        # so the final MoMSBirth returned is from chain 1 only (kept for
        # backward compatibility — callers that need the strategies for
        # warm-starting a follow-up should call single-chain).
        per_chain_samples = max(1, cld(n_samples, n_chains))
        tasks = Vector{Task}(undef, n_chains)
        for c in 1:n_chains
            chain_seed = seed + 1000 * (c - 1)
            chain_show_progress = show_progress && (c == 1)
            tasks[c] = Threads.@spawn sample_moms(target, data;
                td = td, n_samples = per_chain_samples,
                n_warmup = n_warmup, seed = chain_seed,
                n_chains = 1,
                init_scale = init_scale,
                target_birth_accept = target_birth_accept,
                inclusion_prior = inclusion_prior,
                show_progress = chain_show_progress,
                progress_every = progress_every,
                within_model = within_model,
                informed_birth_fraction = informed_birth_fraction,
            )
        end
        results = [fetch(t) for t in tasks]
        chains = MCMCChains.chainscat((r[1] for r in results)...)
        total_evals = sum(r[2] for r in results)
        return chains, total_evals, results[1][3]   # strategy from chain 1
    end

    rng    = MersenneTwister(seed)
    params = target.params
    layout = params.layout

    # Build the MoMS strategy from the priors. All scales are mutable
    # and are adapted by Robbins-Monro during warmup (see below).
    strategy = MoMSBirth(params; init_scale = init_scale, rng = rng)

    # TransDim state: 0 planets, no TOGGLEABLE noise active. Mask sized to
    # the FULL noise-model list (consumers index by config.noise_models
    # position — a length(toggleable) mask left always-on models silently
    # inactive; see the rjmcmc/transdim_ptemcee fix), non-toggleables
    # forced active. With td.planets=false the planet set is FIXED: all
    # slots active (previously they started inactive and, with planet
    # moves gated off, a fixed-planet noise run fit ZERO planets forever).
    td_state = TransDimState(
        max_planets = td.max_kplanet,
        n_noise = length(params.config.noise_models),
    )
    for (nm_idx, nm) in enumerate(params.config.noise_models)
        if !(td.noise && nm in td.toggleable)
            td_state.noise_active[nm_idx] = true
        end
    end
    if !td.planets
        for k in 1:td.max_kplanet
            activate_planet!(td_state, k)
        end
    end
    theta = Theta{Float64}(params; td = td_state)
    _init_systemics_from_prior!(theta, rng)

    ctr = LikelihoodCounter()
    # Use the spike-and-slab log-prior so γ-flips have unbiased M-H
    # ratios. Within-model slice sampling internally uses Nereus's
    # continuous log_prior — that's fine because at fixed γ the
    # spike-and-slab correction is constant and the slice condition
    # is invariant under constant offsets in log_prior.
    log_pi = spike_slab_log_prior(theta, strategy, inclusion_prior)
    log_L  = _eval_ll(theta, data, ctr)
    widths = _prior_widths(layout)

    # PTWorkspace lets us call the fast RWM within-model move when
    # `within_model = :rwm`. Slice sampling spends 30-100 likelihood
    # evals per coordinate per outer iteration; on a 30+ dim trans-dim
    # target that's the dominant cost. The RWM path drops it to 1 eval
    # per coord with Robbins-Monro σ adaptation during warmup.
    n_obs_rv = length(data.t_rv)
    n_phot   = length(data.t_phot)
    n_noise_for_ws = length(params.config.noise_models)
    # Built for BOTH :rwm and ws-:slice. Slice benefits most from the Fix B
    # phot cache (one coord perturbed 30-100× with photometry held fixed →
    # RV-coord slices hit the cache repeatedly). n_phot REQUIRED (omitting it
    # segfaults). Name kept as rwm_ws to avoid touching downstream references.
    rwm_ws = PTWorkspace(params, td.max_kplanet, n_noise_for_ws;
                          n_obs = n_obs_rv, n_phot = n_phot)

    n_total_iter = n_warmup + n_samples
    n_unfrozen   = length(layout.unfrozen_idx)

    # Output column layout: unfrozen params + n_planets + noise_active_*
    noise_names  = Symbol[]
    n_noise_cols = 0
    if td.noise && !isempty(td.toggleable)
        for nm in td.toggleable
            nm_idx = findfirst(==(nm), params.config.noise_models)
            nm_idx === nothing && continue
            push!(noise_names, Symbol("noise_active_$(nm_idx)"))
            n_noise_cols += 1
        end
    end
    n_pl_cols = td.max_kplanet
    planet_names = Symbol[Symbol("planet_active_$(k)") for k in 1:n_pl_cols]
    param_names_out = vcat(Symbol.(layout.unfrozen_names),
                            [:n_planets], planet_names, noise_names)
    n_cols  = n_unfrozen + 1 + n_pl_cols + n_noise_cols
    samples = Matrix{Float64}(undef, n_samples, n_cols)
    sample_idx = 0
    t_start    = time()

    # Per-planet birth/death adaptation counters
    max_k = td.max_kplanet
    n_attempts_planet = zeros(Int, max_k)
    n_accepts_planet  = zeros(Int, max_k)
    adapt_window      = max(50, n_warmup ÷ 20)
    adapt_iter_counter = 0

    pb = ProgressBar("MoMS";
                       total = n_total_iter, enabled = show_progress)

    for iter in 1:n_total_iter
        if rand(rng) < td.transdim_fraction
            # Between-model move: planets via MoMS proposals, noise via
            # the existing toggle infrastructure.
            choice = rand(rng)
            if td.planets && choice < 0.5
                accepted, k_touched = _moms_planet_move!(theta, data, td,
                                                          strategy, rng,
                                                          log_pi, log_L,
                                                          inclusion_prior,
                                                          ctr;
                                                          informed_birth_fraction =
                                                              informed_birth_fraction
                                                          ) do new_lpi, new_lL
                    log_pi = new_lpi
                    log_L  = new_lL
                end
                if k_touched > 0
                    n_attempts_planet[k_touched] += 1
                    accepted && (n_accepts_planet[k_touched] += 1)
                end
            elseif td.noise && !isempty(td.toggleable)
                _noise_move!(theta, data, td, rng, log_pi, log_L;
                              ctr=ctr) do new_lpi, new_lL
                    log_pi = new_lpi
                    log_L  = new_lL
                end
                # _noise_move! returns log_pi in Nereus's log_prior units;
                # convert back to spike-and-slab so subsequent γ-flips use
                # consistent values.
                log_pi = spike_slab_log_prior(theta, strategy, inclusion_prior)
            elseif td.planets
                accepted, k_touched = _moms_planet_move!(theta, data, td,
                                                          strategy, rng,
                                                          log_pi, log_L,
                                                          inclusion_prior,
                                                          ctr;
                                                          informed_birth_fraction =
                                                              informed_birth_fraction
                                                          ) do new_lpi, new_lL
                    log_pi = new_lpi
                    log_L  = new_lL
                end
                if k_touched > 0
                    n_attempts_planet[k_touched] += 1
                    accepted && (n_accepts_planet[k_touched] += 1)
                end
            end
        else
            if within_model === :rwm
                # RWM path — single likelihood eval per coordinate, no
                # slice stepping out. Adapt σ during warmup only.
                adapt = iter <= n_warmup
                _within_model_move_rwm!(theta, data, rng, rwm_ws,
                                          log_pi, log_L;
                                          adapt = adapt,
                                          ctr = ctr) do new_lpi, new_lL
                    log_pi = new_lpi
                    log_L  = new_lL
                end
            else  # :slice — ws path so the Fix B phot cache applies
                _within_model_move!(theta, data, rng, 1.0, widths,
                                     log_pi, log_L, rwm_ws; ctr=ctr) do new_lpi, new_lL
                    log_pi = new_lpi
                    log_L  = new_lL
                end
            end
            # Both moves return log_pi in Nereus's log_prior units;
            # convert to spike-and-slab for between-model M-H consistency.
            log_pi = spike_slab_log_prior(theta, strategy, inclusion_prior)
        end

        # Robbins-Monro scale adaptation — only during warmup
        if iter <= n_warmup
            adapt_iter_counter += 1
            if adapt_iter_counter >= adapt_window
                _adapt_moms_scales!(strategy, n_attempts_planet,
                                     n_accepts_planet, target_birth_accept,
                                     iter, n_warmup)
                adapt_iter_counter = 0
                fill!(n_attempts_planet, 0)
                fill!(n_accepts_planet, 0)
            end
        end

        # Store sample
        if iter > n_warmup
            sample_idx += 1
            for (j, uf_idx) in enumerate(layout.unfrozen_idx)
                samples[sample_idx, j] = theta.values[uf_idx]
            end
            col = n_unfrozen + 1
            samples[sample_idx, col] = Float64(n_p(theta))
            for k in 1:n_pl_cols
                samples[sample_idx, col + k] =
                    theta.td !== nothing && theta.td.planet_active[k] ? 1.0 : 0.0
            end
            if n_noise_cols > 0
                ni = 0
                for nm in td.toggleable
                    nm_idx = findfirst(==(nm), params.config.noise_models)
                    nm_idx === nothing && continue
                    ni += 1
                    samples[sample_idx, col + n_pl_cols + ni] =
                        Float64(is_noise_active(theta.td, nm_idx))
                end
            end
        end

        if iter > n_warmup && sample_idx > 0
            np_running = view(samples, 1:sample_idx, n_unfrozen + 1)
            np_post = join([@sprintf("%2.0f", 100 * count(==(Float64(k)), np_running) / sample_idx)
                             for k in 0:td.max_kplanet], "/")
            update!(pb; n_done = iter,
                     fields = (:phase => "sample",
                               :Np => np_post * "%",
                               :logL => log_L))
        else
            update!(pb; n_done = iter,
                     fields = (:phase => "warmup",
                               :logL => log_L))
        end
    end
    finish!(pb)

    chains = MCMCChains.Chains(samples, param_names_out)
    return chains, ctr.count, strategy
end

"""
    _moms_planet_move!(callback, theta, data, td, strategy, rng,
                        log_pi, log_L, ctr) -> (accepted::Bool, k::Int)

Single planet birth-or-death attempt under the MoMS proposal.
Returns whether the move was accepted and which planet index was
touched (or 0 if the proposal was invalid before reaching M-H).
"""
function _moms_planet_move!(callback, theta::Theta, data::Data,
                              td::TransDimConfig, strategy::MoMSBirth,
                              rng::AbstractRNG, log_pi::Float64,
                              log_L::Float64, inclusion_prior::Float64,
                              ctr::LikelihoodCounter;
                              informed_birth_fraction::Float64 = 0.0)
    n_active = theta.td.n_planets_active
    max_k    = td.max_kplanet
    p_birth, _ = _birth_death_probs(n_active, max_k)
    do_birth = rand(rng) < p_birth

    # Pick the proposal strategy. With probability `informed_birth_fraction`
    # use Lomb-Scargle periodogram-driven proposals (period from
    # InformedBirth, other dims from prior — Hastings ratio computed
    # by `propose_planet_birth(::InformedBirth)`). Otherwise use the
    # native MoMSBirth Gaussian RW around `off_values`. Forward and
    # reverse moves use the same strategy so detailed balance holds.
    use_informed = informed_birth_fraction > 0 &&
                    rand(rng) < informed_birth_fraction
    chosen_strategy = use_informed ? InformedBirth() : strategy

    new_theta, log_q = if do_birth
        propose_planet_birth(theta, rng, chosen_strategy; data=data)
    elseif chosen_strategy isa MoMSBirth
        # MoMS death is deterministic (reset to `off_values`); its
        # M-H ratio uses the matching MoMSBirth-specific proposal
        # density.
        propose_planet_death(theta, rng, chosen_strategy)
    else
        # InformedBirth/PriorBirth deaths are uniform over active
        # planets; the Hastings ratio in `propose_planet_birth(::Informed)`
        # was computed against this strategy-agnostic death.
        propose_planet_death(theta, rng)
    end

    !isfinite(log_q) && return (false, 0)

    # Identify which planet slot was touched (for adaptation accounting)
    k_touched = 0
    for k in 1:max_k
        if theta.td.planet_active[k] != new_theta.td.planet_active[k]
            k_touched = k
            break
        end
    end

    new_log_pi = spike_slab_log_prior(new_theta, strategy, inclusion_prior)
    !isfinite(new_log_pi) && return (false, k_touched)

    new_log_L = _eval_ll(new_theta, data, ctr)
    !isfinite(new_log_L) && return (false, k_touched)

    if rjmcmc_accept(log_L, new_log_L, log_pi, new_log_pi, log_q, rng)
        theta.values .= new_theta.values
        theta.td.n_planets_active = new_theta.td.n_planets_active
        theta.td.planet_active .= new_theta.td.planet_active
        callback(new_log_pi, new_log_L)
        return (true, k_touched)
    end
    return (false, k_touched)
end

"""
    _adapt_moms_scales!(strategy, n_attempts, n_accepts, target,
                         iter, n_warmup)

Robbins-Monro adaptation of the per-planet proposal scales. After each
adaptation window, scales are updated multiplicatively by
`exp(γ_t × (rate − target))` with a decreasing step
`γ_t = 1 / (iter+1)^0.75`, matching MoMS Eq. D1 of van den Bergh+ 2026
(arXiv:2604.27791) — the φ=0.75 exponent is what gives Roberts-Rosenthal
containment + diminishing-adaptation. The paper's univariate target
α_* = 0.44 applies to scalar coordinate flips; for vector group flips
(Nereus's planet blocks have 5–7 dims) the appropriate target is the
multivariate limit α_* = 0.234 (Roberts, Gelman, Gilks 1997). The
default `target_birth_accept` kwarg in `sample_moms` is set
accordingly.
"""
function _adapt_moms_scales!(strategy::MoMSBirth,
                              n_attempts::AbstractVector{<:Integer},
                              n_accepts::AbstractVector{<:Integer},
                              target::Float64,
                              iter::Int, n_warmup::Int)
    γ = 1.0 / (iter + 1)^0.75
    for k in eachindex(strategy.scales)
        n_attempts[k] == 0 && continue
        rate = n_accepts[k] / n_attempts[k]
        bump = exp(γ * (rate - target))
        # Clamp to keep numerical sanity (avoid runaway shrinkage/inflation)
        bump = clamp(bump, 0.5, 2.0)
        for i in eachindex(strategy.scales[k])
            strategy.scales[k][i] *= bump
        end
    end
    return strategy
end
