# Standalone RJMCMC sampler for trans-dimensional inference.
#
# Two move classes:
#   - Within-model: Gaussian random walk on active parameters (MH accept)
#   - Between-model: planet/noise birth/death (RJMCMC accept)
#
# Move mix controlled by TransDimConfig.transdim_fraction.

using Random
using MCMCChains
using Statistics: median

"""
    sample_rjmcmc(target, data; td, kwargs...) -> (MCMCChains.Chains, n_evals::Int)

Run a standalone RJMCMC sampler for trans-dimensional inference.

# Required
- `target::NereusTarget` — the Bayesian target (use `unconstrained=false`)
- `data::Data` — observation data
- `td::TransDimConfig` — trans-dimensional configuration

# Keywords
- `n_samples::Int=10000` — total number of posterior samples to store across all chains
- `n_warmup::Int=5000` — warmup iterations per chain (adapt proposal scale)
- `n_chains::Int=1` — number of independent chains to run in parallel via
  `Threads.@spawn`. Each chain gets `n_samples ÷ n_chains` samples and a
  distinct seed. Returns a single `Chains` object with a populated `:chain`
  axis so MCMCChains diagnostics (Rhat, ESS) work out of the box.
- `seed::Int=1` — base random seed; chain `c` uses `seed + 1000*(c-1)`
- `initial_scale::Float64=0.01` — initial Gaussian proposal scale
- `target_accept::Float64=0.234` — target acceptance rate for within-model
"""
function sample_rjmcmc(
    target::NereusTarget,
    data::Data;
    td::TransDimConfig,
    n_samples::Int = 10000,
    n_warmup::Int = 5000,
    seed::Int = 1,
    n_chains::Int = 1,
    initial_scale::Real = 0.01,
    target_accept::Real = 0.234,
    within_model::Symbol = :slice,
    noise_swap_rate::Real = 0.5,
    show_progress::Bool = true,
    progress_every::Int = max(1000, (n_samples + n_warmup) ÷ 20),
)
    initial_scale = Float64(initial_scale)   # JSON may deliver Ints
    target_accept = Float64(target_accept)
    within_model in (:slice, :rwm) ||
        throw(ArgumentError("within_model must be :slice or :rwm (got :$within_model)"))
    n_chains >= 1 || throw(ArgumentError("n_chains must be ≥ 1"))
    noise_swap_rate = Float64(noise_swap_rate)
    0.0 <= noise_swap_rate <= 1.0 ||
        throw(ArgumentError("noise_swap_rate must be in [0, 1]"))
    if n_chains == 1
        return _sample_rjmcmc_one(target, data; td=td, n_samples=n_samples,
                                   n_warmup=n_warmup, seed=seed,
                                   initial_scale=initial_scale,
                                   target_accept=target_accept,
                                   within_model=within_model,
                                   noise_swap_rate=noise_swap_rate,
                                   show_progress=show_progress,
                                   progress_every=progress_every)
    end
    # Multi-chain: spawn n_chains independent chains. Total samples is
    # divided across chains; each chain runs full warmup independently
    # (warmup is per-chain to give each its own adapted proposal scale).
    per_chain_samples = max(1, cld(n_samples, n_chains))
    tasks = Vector{Task}(undef, n_chains)
    for c in 1:n_chains
        chain_seed = seed + 1000 * (c - 1)
        # Only the first chain shows the live progress bar; the others
        # would clobber the carriage-return line.
        chain_show_progress = show_progress && (c == 1)
        tasks[c] = Threads.@spawn _sample_rjmcmc_one(
            target, data;
            td = td,
            n_samples = per_chain_samples,
            n_warmup = n_warmup,
            seed = chain_seed,
            initial_scale = initial_scale,
            target_accept = target_accept,
            within_model = within_model,
            noise_swap_rate = noise_swap_rate,
            show_progress = chain_show_progress,
            progress_every = progress_every,
        )
    end
    results = [fetch(t) for t in tasks]
    # `MCMCChains.chainscat` concatenates along the chain axis so Rhat/
    # ESS work correctly. Inner chain has shape (n_samples, n_params, 1);
    # output has (n_samples, n_params, n_chains).
    chains = MCMCChains.chainscat((r[1] for r in results)...)
    total_evals = sum(r[2] for r in results)
    return chains, total_evals
end

function _sample_rjmcmc_one(
    target::NereusTarget,
    data::Data;
    td::TransDimConfig,
    n_samples::Int,
    n_warmup::Int,
    seed::Int,
    initial_scale::Float64,
    target_accept::Float64,
    within_model::Symbol = :slice,
    noise_swap_rate::Float64 = 0.5,
    show_progress::Bool,
    progress_every::Int,
)
    rng = MersenneTwister(seed)
    params = target.params
    layout = params.layout

    # Initialize theta with trans-dim state (0 planets, no toggleable noise
    # active). The noise mask MUST be sized to the FULL noise-model list:
    # every consumer (propose_noise_birth, _log_prior_transdim,
    # is_noise_model_active) indexes by position in config.noise_models, so
    # a mask sized to length(toggleable) left always-on models at low config
    # indices silently INACTIVE (likelihood spectators) and made births of a
    # toggleable model at config index > length(toggleable) a BoundsError.
    n_noise_slots = length(params.config.noise_models)
    td_state = TransDimState(
        max_planets = td.max_kplanet,
        n_noise = n_noise_slots,
    )
    # Non-toggleable noise models are ALWAYS active (mirrors
    # transdim_ptemcee); toggleables start off and enter via births.
    for (nm_idx, nm) in enumerate(params.config.noise_models)
        if !(td.noise && nm in td.toggleable)
            td_state.noise_active[nm_idx] = true
        end
    end
    # td.planets=false ⇒ the planet set is FIXED: all slots active. The
    # chain starts at 0 planets by design for trans-dim planet search,
    # but with planet moves gated off a fixed-planet noise-only run
    # previously stayed planet-less for the entire run.
    if !td.planets
        for k in 1:td.max_kplanet
            activate_planet!(td_state, k)
        end
    end
    theta = Theta{Float64}(params; td=td_state)

    # Set systemics to random valid values from prior
    _init_systemics_from_prior!(theta, rng)

    # Likelihood counter
    ctr = LikelihoodCounter()

    # Evaluate initial state
    log_pi = log_prior(theta)
    log_L = _eval_ll(theta, data, ctr)

    # Slice sampling widths (initial stepping-out size per param)
    widths = _prior_widths(layout)

    # Workspace for BOTH ws-aware within-model kernels (:rwm and ws-:slice).
    # It carries the per-planet RV/transit caches + the Fix B total-phot cache,
    # and Robbins-Monro σ for :rwm. Slice benefits MOST from the phot cache:
    # _slice_sample_coord! perturbs ONE coord 30-100× (step-out + shrink), so
    # for an RV coordinate the photometry is identical across every one of those
    # steps → the cache hits 30-100× in a row, skipping the O(n_phot) sum each
    # time. n_phot REQUIRED so transit_flux_cache is sized correctly (omitting
    # it segfaults — see moms.jl fix).
    within_ws = PTWorkspace(params, td.max_kplanet, n_noise_slots;
                            n_obs = length(data.t_rv), n_phot = length(data.t_phot))

    # Storage
    n_total_iter = n_warmup + n_samples
    n_unfrozen = length(layout.unfrozen_idx)
    # Columns: unfrozen params + n_planets + planet_active_* (one per slot;
    # without them consumers fall back to the order-assuming "slot k active
    # iff k ≤ n_planets") + noise_active_* (one per toggleable).
    n_pl_cols = td.max_kplanet
    planet_names = Symbol[Symbol("planet_active_$(k)") for k in 1:n_pl_cols]
    noise_names = Symbol[]
    n_noise_cols = 0
    if td.noise && !isempty(td.toggleable)
        for nm in td.toggleable
            nm_idx = findfirst(==(nm), params.config.noise_models)
            nm_idx === nothing && continue
            push!(noise_names, Symbol("noise_active_$(nm_idx)"))
            n_noise_cols += 1
        end
    end
    param_names_out = vcat(Symbol.(layout.unfrozen_names), [:n_planets],
                           planet_names, noise_names)
    n_cols = n_unfrozen + 1 + n_pl_cols + n_noise_cols
    samples = Matrix{Float64}(undef, n_samples, n_cols)

    sample_idx = 0
    t_start = time()

    pb = ProgressBar("RJMCMC";
                       total = n_total_iter, enabled = show_progress)

    for iter in 1:n_total_iter
        if rand(rng) < td.transdim_fraction
            # --- Between-model move (birth/death) -------------------------
            if td.planets && td.alias_jump_fraction > 0.0 &&
                    rand(rng) < td.alias_jump_fraction
                # Green's UPDATE move: re-propose an active planet's period at
                # a harmonic/alias. Within-model (dimension unchanged), so it
                # is a plain M-H accept, but it is the only move that can cross
                # the likelihood valley between P and 2P once a planet has
                # entrenched there.
                _alias_jump_move!(theta, data, rng,
                                   log_pi, log_L; ctr=ctr) do new_lpi, new_lL
                    log_pi = new_lpi
                    log_L = new_lL
                end
            elseif td.planets && rand(rng) < 0.5
                _planet_move!(theta, data, td, rng,
                              log_pi, log_L; ctr=ctr) do new_lpi, new_lL
                    log_pi = new_lpi
                    log_L = new_lL
                end
            elseif td.noise && !isempty(td.toggleable)
                _noise_move!(theta, data, td, rng, log_pi, log_L; ctr=ctr,
                             noise_swap_rate=noise_swap_rate) do new_lpi, new_lL
                    log_pi = new_lpi
                    log_L = new_lL
                end
            else
                if td.planets
                    _planet_move!(theta, data, td, rng,
                                  log_pi, log_L; ctr=ctr) do new_lpi, new_lL
                        log_pi = new_lpi
                        log_L = new_lL
                    end
                end
            end
        elseif within_model === :rwm
            _within_model_move_rwm!(theta, data, rng, within_ws,
                                    log_pi, log_L;
                                    beta = 1.0, adapt = (iter <= n_warmup),
                                    ctr = ctr) do new_lpi, new_lL
                log_pi = new_lpi
                log_L = new_lL
            end
        else  # :slice — ws-aware path so the Fix B phot cache applies
            _within_model_move!(theta, data, rng, 1.0, widths,
                                log_pi, log_L, within_ws; ctr=ctr) do new_lpi, new_lL
                log_pi = new_lpi
                log_L = new_lL
            end
        end

        # Store sample (after warmup)
        if iter > n_warmup
            sample_idx += 1
            for (j, uf_idx) in enumerate(layout.unfrozen_idx)
                samples[sample_idx, j] = theta.values[uf_idx]
            end
            col = n_unfrozen + 1
            samples[sample_idx, col] = Float64(n_p(theta))
            # Per-slot planet activity
            for k in 1:n_pl_cols
                samples[sample_idx, col + k] =
                    theta.td !== nothing && theta.td.planet_active[k] ? 1.0 : 0.0
            end
            # Noise active state
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

        # Single-line bar update — show within-warmup vs sampling phase
        # plus a running N_p posterior once samples are accumulating.
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
    return chains, ctr.count
end

# --- Likelihood evaluation --------------------------------------------

@inline function _eval_ll(theta::Theta, data::Data,
                           ctr::Union{LikelihoodCounter, Nothing}=nothing)
    if ctr !== nothing
        ctr.count += 1
    end
    ll = rv_log_likelihood(theta, data)
    isfinite(ll) || return -Inf
    ll += transit_log_likelihood(theta, data)
    return ll
end

"""
    _prior_widths(layout) -> Vector{Float64}

Precompute a natural scale for each unfrozen parameter from its prior.
For log-uniform priors, use a fraction of the log-range (perturbation
in log-space). For bounded priors, use a fraction of the width.
This gives sensible initial perturbation magnitudes before adaptation.
"""
function _prior_widths(layout::ParamsLayout)
    n = length(layout.unfrozen_idx)
    widths = Vector{Float64}(undef, n)
    packed = layout.packed_priors
    for i in 1:n
        lo = packed.lowers[i]
        hi = packed.uppers[i]
        tid = packed.type_ids[i]
        w = hi - lo
        if !isfinite(w) || w ≤ 0
            widths[i] = 1.0
        elseif tid == PACKED_LOGUNIFORM || tid == PACKED_MODJEFFREYS
            # Log-scale priors: perturb proportional to current value
            # Use geometric mean of bounds as reference scale
            mid = sqrt(max(lo, 1e-10) * hi)
            widths[i] = mid * 0.1  # ~10% of geometric mean
        else
            widths[i] = w
        end
    end
    return widths
end

"""
    PTWorkspace

Pre-allocated per-chain workspace to avoid allocations in the
Threads.@threads hot loop. One per chain.

Likelihood evaluation buffers (planet_Ps … residuals) eliminate the
~12 heap vectors that `_rv_ll_no_noise` / `_rv_ll_with_noise` would
otherwise allocate on every call.  Stability buffers (stab_order …
stab_sma) do the same for `check_stability`.
"""
mutable struct PTWorkspace
    active_idx::Vector{Int}     # scratch for _active_unfrozen!
    active_pos::Vector{Int}
    n_active::Int               # how many entries are valid
    skip_flags::Vector{Bool}    # length = n_total params, true = skip
    population::Vector{Theta{Float64}}  # scratch for donor birth
    n_pop::Int
    scratch_theta::Theta{Float64}  # pre-allocated scratch for proposals
    # --- Likelihood evaluation buffers (RV path) ---
    planet_Ps::Vector{Float64}    # length = max_kplanet
    planet_Ks::Vector{Float64}    # component-A amplitude (K, or K_A for SB2)
    planet_cBs::Vector{Float64}   # component-B coefficient (0, or −K_B for SB2)
    planet_es::Vector{Float64}
    planet_ws::Vector{Float64}
    planet_Tps::Vector{Float64}
    predictions::Vector{Float64}  # length = n_rv
    variances::Vector{Float64}
    residuals::Vector{Float64}
    # --- Likelihood evaluation buffers (transit / phot path) ---
    transit_Ps::Vector{Float64}     # length = max_kplanet
    transit_es::Vector{Float64}
    transit_ws::Vector{Float64}
    transit_Tps::Vector{Float64}
    transit_rrs::Vector{Float64}
    transit_bs::Vector{Float64}
    transit_a_Rs::Vector{Float64}
    transit_active::Vector{Bool}    # length = max_kplanet — geometry-gate result
    # --- Per-planet RV-velocity cache ---
    # Stores K·(cos(f+ω) + e·cos(ω)) per RV time per planet — the
    # additive Keplerian RV contribution. Keyed on planet's orbit + K
    # hash. Single-coord MCMC steps that don't change a given planet's
    # orbit/K reuse the cached column → save ~80% of kepler_solve calls
    # at typical RV cache-hit rates. Analogous to transit_flux_cache.
    rv_vel_cache::Matrix{Float64}        # max_kplanet × n_rv
    rv_vel_hash::Vector{UInt}            # max_kplanet, 0 = invalid
    # --- Per-planet transit-flux cache (sparse) ---
    # Stores the full Mandel-Agol contribution per phot point per planet.
    # SPARSE LAYOUT: the cache matrix is initialized to 1.0 everywhere,
    # and per-planet `in_transit_idx[j]` tracks the small set (~few %
    # of phot) of phot indices where flux differs from 1.0. On cache
    # refresh we (a) reset previously non-trivial entries to 1.0,
    # (b) compute the new in-transit set, (c) write fresh flux values at
    # those indices. For 13k phot × 4 transiting planets at 2% in-transit
    # fraction this means ~1k kepler_solve calls per refresh instead of
    # 52k — ~50× fewer.
    transit_flux_cache::Matrix{Float64}  # max_kplanet × n_phot, init=1.0
    transit_flux_hash::Vector{UInt}      # max_kplanet, 0 = invalid
    transit_in_idx::Vector{Vector{Int}}  # per planet: phot indices where flux≠1
    # --- Stability buffers ---
    stab_order::Vector{Int}       # length = max_kplanet
    stab_masses::Vector{Float64}
    stab_sma::Vector{Float64}
    # --- RWM within-model kernel state (Robbins-Monro adapted σ per coord) ---
    rwm_sigmas::Vector{Float64}   # length = n_uf — per-coord proposal stddev
    rwm_attempts::Vector{Int}      # length = n_uf
    rwm_accepts::Vector{Int}       # length = n_uf
    # --- Total photometry log-likelihood cache (Fix B) ---
    # Caches the full phot log-L keyed on a hash of EVERY param the phot
    # likelihood reads: transiting planets' orbit+geom+LD (via their
    # transit_flux_hash key), plus per-instrument offset/jitter — or
    # offset/jitter alone in the no-transit case. A within-model move on an
    # RV-only coordinate (K, γ, σ, astrometry) leaves all of these unchanged,
    # so the hash hits and the O(n_phot) assembly+sum is skipped. Only used on
    # the white-noise path (no active phot AR/MA/GP). hash==0 ⇒ invalid.
    phot_ll_total_cache::Float64
    phot_ll_total_hash::UInt
end

# `max_donors` sizes the donor-birth scratch buffer. It must be ≥ the number
# of donor replicas the caller will ever offer (n_chains − 1 for PT), because
# the population loop counts every eligible donor. The old hardcoded 20
# silently assumed "no more than 20 chains" and BoundsError'd above that.
function PTWorkspace(params::Params, max_kplanet::Int, n_noise::Int=0;
                     n_obs::Int=0, n_phot::Int=0, max_donors::Int=20)
    n_uf = length(params.layout.unfrozen_idx)
    nt = params.layout.n_total
    td_state = TransDimState(
        max_planets = max_kplanet,
        n_noise = n_noise,
    )
    scratch = Theta{Float64}(params; td=td_state)
    PTWorkspace(
        Vector{Int}(undef, n_uf),
        Vector{Int}(undef, n_uf),
        0,
        Vector{Bool}(undef, nt),
        Vector{Theta{Float64}}(undef, max_kplanet > 0 ? max_donors : 0),
        0,
        scratch,
        # RV likelihood buffers (Ps, Ks, cBs, es, ws, Tps)
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),   # planet_cBs (component-B coef)
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, n_obs),
        Vector{Float64}(undef, n_obs),
        Vector{Float64}(undef, n_obs),
        # Transit (phot) likelihood buffers — sized by max_kplanet
        # because per-planet decoded values, not per-cadence
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Bool}(undef, max_kplanet),
        # Per-planet RV-velocity cache (max_kplanet × n_rv). Initialised
        # to 0 (no contribution) so untouched rows add 0 to predictions.
        # Hashes seeded to 0 (invalid) so first call recomputes.
        zeros(Float64, max_kplanet, max(n_obs, 1)),
        zeros(UInt, max_kplanet),
        # Sparse per-planet transit-flux cache (max_kplanet × n_phot).
        # Initialised to 1.0 (no-transit flux) so untouched points
        # contribute model_flux *= 1 in the main pass. Hashes seeded
        # to 0 (invalid) so first call recomputes.
        ones(Float64, max_kplanet, max(n_phot, 1)),
        zeros(UInt, max_kplanet),
        # Per-planet in-transit index list. Empty initially; populated
        # by _phot_refresh_chunk when orbit changes. Pre-allocate
        # capacity for ~5% transit fraction (typical) to avoid
        # repeated reallocs as planets move slightly between MCMC steps.
        [Int[] for _ in 1:max_kplanet],
        # Stability buffers
        Vector{Int}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        Vector{Float64}(undef, max_kplanet),
        # RWM σ initialised from prior widths × 0.1 (one stddev = 10%
        # of prior support); adapted by Robbins-Monro to ~0.44 acceptance.
        let widths = _prior_widths(params.layout)
            σ0 = [w > 0 ? 0.1 * w : 0.1 for w in widths]
            σ0
        end,
        zeros(Int, n_uf),
        zeros(Int, n_uf),
        # Total phot-ll cache (Fix B): value + hash, seeded invalid (0).
        0.0,
        zero(UInt),
    )
end

"""Workspace-aware _eval_ll: threads pre-allocated buffers through."""
@inline function _eval_ll(theta::Theta, data::Data,
                           ctr::Union{LikelihoodCounter, Nothing},
                           ws::PTWorkspace)
    if ctr !== nothing
        ctr.count += 1
    end
    ll = rv_log_likelihood(theta, data, ws)
    isfinite(ll) || return -Inf
    ll += transit_log_likelihood(theta, data, ws)
    return ll
end

"""
    _active_unfrozen!(ws, params, td) -> n_active

In-place version: fills ws.active_idx and ws.active_pos, returns count.
"""
function _active_unfrozen!(ws::PTWorkspace, params::Params,
                            td::Union{TransDimState, Nothing})
    layout = params.layout
    n_total = length(ws.skip_flags)

    # Clear skip flags
    @inbounds for i in 1:n_total
        ws.skip_flags[i] = false
    end

    if td !== nothing
        for k in 1:length(td.planet_active)
            if !td.planet_active[k]
                for idx in planet_slot_indices(layout.planet_blocks[k])
                    idx ≤ n_total && (ws.skip_flags[idx] = true)
                end
            end
        end
        noise_models = params.config.noise_models
        instruments = params.config.instruments
        for (nm_idx, nm) in enumerate(noise_models)
            if nm_idx ≤ length(td.noise_active) && !td.noise_active[nm_idx]
                for name in noise_param_names(nm, instruments)
                    if haskey(layout.name_to_idx, name)
                        ws.skip_flags[layout.name_to_idx[name]] = true
                    end
                end
            end
        end
    end

    count = 0
    @inbounds for (pos, uf_idx) in enumerate(layout.unfrozen_idx)
        if !ws.skip_flags[uf_idx]
            count += 1
            ws.active_idx[count] = uf_idx
            ws.active_pos[count] = pos
        end
    end
    ws.n_active = count
    return count
end

"""
    _active_unfrozen(params, td) -> (active_uf_idx, active_uf_pos)

Return theta slot indices and their positions in the unfrozen vector
for parameters that are currently active (skipping inactive planets
AND inactive noise components).
"""
function _active_unfrozen(params::Params, td::Union{TransDimState, Nothing})
    layout = params.layout
    skip_slots = Set{Int}()
    if td !== nothing
        for k in 1:length(td.planet_active)
            if !td.planet_active[k]
                for idx in planet_slot_indices(layout.planet_blocks[k])
                    push!(skip_slots, idx)
                end
            end
        end
        noise_models = params.config.noise_models
        instruments = params.config.instruments
        for (nm_idx, nm) in enumerate(noise_models)
            if nm_idx ≤ length(td.noise_active) && !td.noise_active[nm_idx]
                for name in noise_param_names(nm, instruments)
                    if haskey(layout.name_to_idx, name)
                        push!(skip_slots, layout.name_to_idx[name])
                    end
                end
            end
        end
    end
    active_idx = Int[]
    active_pos = Int[]
    for (pos, uf_idx) in enumerate(layout.unfrozen_idx)
        if uf_idx ∉ skip_slots
            push!(active_idx, uf_idx)
            push!(active_pos, pos)
        end
    end
    return active_idx, active_pos
end

function _init_systemics_from_prior!(theta::Theta, rng::AbstractRNG)
    layout = theta.params.layout
    # Initialize all unfrozen systemic params from their priors
    for (i, idx) in enumerate(layout.unfrozen_idx)
        prior = layout.unfrozen_priors[i]
        v = rand(rng, prior.dist)
        lo, hi = bounds(prior)
        v = clamp(v, lo, hi)
        theta.values[idx] = v
    end
end

"""
    _seed_gammas_from_data!(theta, data) -> theta

Override the RV-instrument γ slots with per-instrument median(rv) instead
of the random prior draws produced by `_init_systemics_from_prior!`.
For multi-instrument fits the broad γ priors are far from the right
zero-points, and the resulting γ-noise dominates residuals enough to
swamp the planetary signals at the LS / log-L level. Seeding γ at the
per-instrument median is a near-zero-cost way to put each chain on a
sane starting calibration; subsequent within-model MCMC adjusts γ as
the data prefers.

If a γ slot is shared across multiple instruments (sharing config), the
seed value is the median of the *combined* RV samples in that group.
The resulting γ is clamped to its prior bounds so the move can't escape
the prior support.
"""
function _seed_gammas_from_data!(theta::Theta, data::Data)
    layout = theta.params.layout
    rv_gamma_slots = layout.systemic.rv_gamma
    isempty(rv_gamma_slots) && return theta
    isempty(data.t_rv)      && return theta

    # Group instruments by their shared γ slot
    by_slot = Dict{Int, Vector{Int}}()
    for (inst, slot) in enumerate(rv_gamma_slots)
        slot == 0 && continue
        push!(get!(() -> Int[], by_slot, slot), inst)
    end

    for (slot, inst_list) in by_slot
        mask = falses(length(data.rv_inst))
        for ins in inst_list
            mask .|= (data.rv_inst .== ins)
        end
        any(mask) || continue
        med = median(view(data.rv, mask))

        # Clamp to prior bounds so we don't seed outside the support.
        uf_pos = findfirst(==(slot), layout.unfrozen_idx)
        if uf_pos !== nothing
            prior = layout.unfrozen_priors[uf_pos]
            lo, hi = bounds(prior)
            med = clamp(med, lo, hi)
        end
        theta.values[slot] = med
    end
    return theta
end

function _planet_move!(callback, theta::Theta, data::Data, td::TransDimConfig,
                        rng::AbstractRNG, log_pi::Float64, log_L::Float64;
                        ctr::Union{LikelihoodCounter, Nothing}=nothing)
    n_active = theta.td.n_planets_active
    max_k = td.max_kplanet

    # Decide birth or death
    p_birth, _ = _birth_death_probs(n_active, max_k)
    do_birth = rand(rng) < p_birth

    # Select the strategy for BOTH directions. The reverse of a strategy-s
    # birth is a strategy-s death, so the selection weight w_s cancels and the
    # pair only has to be reversible within s. Selecting only inside the birth
    # branch left the death blind to which birth it was supposed to invert, so
    # it fell back to the combinatorial-only kernel and lost the reverse
    # proposal density (see test_birth_death_reversibility.jl).
    strategy = _select_strategy(td, rng)
    new_theta, log_q = do_birth ?
        propose_planet_birth(theta, rng, strategy; data=data) :
        propose_planet_death(theta, rng, strategy)

    !isfinite(log_q) && return

    new_log_pi = log_prior(new_theta)
    !isfinite(new_log_pi) && return

    new_log_L = _eval_ll(new_theta, data, ctr)
    !isfinite(new_log_L) && return

    if rjmcmc_accept(log_L, new_log_L, log_pi, new_log_pi, log_q, rng)
        theta.values .= new_theta.values
        theta.td.n_planets_active = new_theta.td.n_planets_active
        theta.td.planet_active .= new_theta.td.planet_active
        callback(new_log_pi, new_log_L)
    end
end

"""
    _alias_jump_move!(callback, theta, data, rng, log_pi, log_L; ctr)

Within-model alias/harmonic period jump on one active planet. Dimension is
unchanged, so this is an ordinary Metropolis-Hastings accept — the same
`rjmcmc_accept` composition works because `log_q` from
`propose_planet_alias_jump` already carries the map's log-Jacobian and the
proposal is symmetric in the ratio.
"""
function _alias_jump_move!(callback, theta::Theta, data::Data,
                            rng::AbstractRNG, log_pi::Float64, log_L::Float64;
                            ctr::Union{LikelihoodCounter, Nothing}=nothing)
    new_theta, log_q = propose_planet_alias_jump(theta, rng)
    !isfinite(log_q) && return

    new_log_pi = log_prior(new_theta)
    !isfinite(new_log_pi) && return

    new_log_L = _eval_ll(new_theta, data, ctr)
    !isfinite(new_log_L) && return

    if rjmcmc_accept(log_L, new_log_L, log_pi, new_log_pi, log_q, rng)
        theta.values .= new_theta.values
        callback(new_log_pi, new_log_L)
    end
end

function _noise_move!(callback, theta::Theta, data::Data, td::TransDimConfig,
                       rng::AbstractRNG, log_pi::Float64, log_L::Float64;
                       ctr::Union{LikelihoodCounter, Nothing}=nothing,
                       noise_swap_rate::Float64 = 0.5)
    # Within-group swap, BEFORE the birth/death branch.
    #
        # A swap moves BETWEEN two members, so it needs one already active.
        # Attempting it from the "none" state wastes the move — and at a high
        # swap rate it starves birth/death entirely, freezing the chain in
        # "none" forever. Fall through to birth/death instead of returning.
    #
    # rjmcmc runs at beta = 1 with no ladder, so unlike the tempered samplers it
    # has NO way to cross the "none" valley that sits between two members of an
    # exclusion group: going A -> none -> B requires accepting an intermediate
    # state that is a deep posterior minimum. transdim_ptemcee can do it on its
    # hot chains (which is why its swap is gated to beta > 0.3, cold side only);
    # here there are no hot chains, so without this move a walker that lands in
    # one member never leaves and the occupancy is NOT P(M|D).
    if !isempty(td.noise_exclusion_groups) && rand(rng) < noise_swap_rate &&
       _any_group_member_active(theta, td.noise_exclusion_groups)
        grp = _pick_active_group(theta, td.noise_exclusion_groups, rng)
        cand, lq = propose_noise_swap(theta, rng, grp, td.toggleable; data=data)
        isfinite(lq) || return
        cand_lp = log_prior(cand)
        isfinite(cand_lp) || return
        cand_ll = _eval_ll(cand, data, ctr)
        isfinite(cand_ll) || return
        if rjmcmc_accept(log_L, cand_ll, log_pi, cand_lp, lq, rng)
            theta.values .= cand.values
            theta.td.noise_active .= cand.td.noise_active
            callback(cand_lp, cand_ll)
        end
        return
    end

    # 50/50 birth/death. `data` MUST be forwarded: without it every informed
    # proposal (AD's OLS coefficients, the GP period/amplitude hints) silently
    # falls back to a blind prior draw, which is the difference between an
    # occupancy that tracks the evidence and one that does not.
    if rand(rng) < 0.5
        new_theta, log_q = propose_noise_birth(theta, rng, td.toggleable;
                                                exclusion_groups=td.noise_exclusion_groups,
                                                data=data)
    else
        new_theta, log_q = propose_noise_death(theta, rng, td.toggleable; data=data)
    end

    !isfinite(log_q) && return

    new_log_pi = log_prior(new_theta)
    !isfinite(new_log_pi) && return

    new_log_L = _eval_ll(new_theta, data, ctr)
    !isfinite(new_log_L) && return

    if rjmcmc_accept(log_L, new_log_L, log_pi, new_log_pi, log_q, rng)
        theta.values .= new_theta.values
        theta.td.noise_active .= new_theta.td.noise_active
        callback(new_log_pi, new_log_L)
    end
end

"""
    _within_model_move!(callback, theta, data, rng, scale, widths, log_pi, log_L; beta=1.0)

Slice sampling (Neal 2003) — one coordinate at a time with
stepping-out and shrinking. No tuning needed. Each coordinate
is updated exactly from its full conditional (up to the slice).

`widths` provides the initial stepping-out width per parameter.
`scale` is a global multiplier (set to 1.0 and forget about it).
"""
function _within_model_move!(callback, theta::Theta, data::Data,
                              rng::AbstractRNG, scale::Float64,
                              widths::Vector{Float64},
                              log_pi::Float64, log_L::Float64;
                              beta::Float64 = 1.0,
                              ctr::Union{LikelihoodCounter, Nothing}=nothing)
    layout = theta.params.layout

    active_idx, active_pos = _active_unfrozen(theta.params, theta.td)
    isempty(active_idx) && return false

    current_lp = beta * log_L + log_pi  # tempered log-posterior

    for (i, idx) in enumerate(active_idx)
        w = scale * widths[active_pos[i]]
        current_lp = _slice_sample_coord!(theta, data, idx, w, current_lp, beta, rng;
                                           ctr=ctr)
    end

    # Recompute decomposed log_pi and log_L from the final state
    new_log_pi = log_prior(theta)
    new_log_L = _eval_ll(theta, data, ctr)
    callback(new_log_pi, new_log_L)
    return true
end

"""In-place version using pre-allocated workspace (no allocations)."""
function _within_model_move!(callback, theta::Theta, data::Data,
                              rng::AbstractRNG, scale::Float64,
                              widths::Vector{Float64},
                              log_pi::Float64, log_L::Float64,
                              ws::PTWorkspace;
                              beta::Float64 = 1.0,
                              ctr::Union{LikelihoodCounter, Nothing}=nothing)
    n_act = _active_unfrozen!(ws, theta.params, theta.td)
    n_act == 0 && return false

    current_lp = beta * log_L + log_pi

    @inbounds for i in 1:n_act
        idx = ws.active_idx[i]
        w = scale * widths[ws.active_pos[i]]
        current_lp = _slice_sample_coord!(theta, data, idx, w, current_lp, beta, rng, ws;
                                           ctr=ctr)
    end

    new_log_pi = log_prior(theta)
    new_log_L = _eval_ll(theta, data, ctr, ws)
    callback(new_log_pi, new_log_L)
    return true
end

"""
    _slice_sample_coord!(theta, data, idx, w, current_lp, beta, rng) -> new_lp

Slice sample one coordinate of theta at slot `idx`.
Neal (2003) stepping-out + shrinking procedure.
`current_lp` = beta * log_L + log_pi (tempered log-posterior).
Returns the new tempered log-posterior.
"""
function _slice_sample_coord!(theta::Theta, data::Data, idx::Int,
                               w::Float64, current_lp::Float64,
                               beta::Float64, rng::AbstractRNG;
                               max_steps::Int = 32,
                               ctr::Union{LikelihoodCounter, Nothing}=nothing)
    x0 = theta.values[idx]

    # Draw the slice level
    log_y = current_lp + log(rand(rng))  # log_y < current_lp

    # Stepping out: find (L, R) bracket
    u = rand(rng)
    L = x0 - u * w
    R = L + w

    # Expand left
    for _ in 1:max_steps
        theta.values[idx] = L
        lp = _tempered_lp(theta, data, beta; ctr=ctr)
        lp < log_y && break
        L -= w
    end

    # Expand right
    for _ in 1:max_steps
        theta.values[idx] = R
        lp = _tempered_lp(theta, data, beta; ctr=ctr)
        lp < log_y && break
        R += w
    end

    # Shrinking: sample from [L, R] until we find a point above the slice
    for _ in 1:1024
        x1 = L + rand(rng) * (R - L)
        theta.values[idx] = x1
        new_lp = _tempered_lp(theta, data, beta; ctr=ctr)

        if new_lp > log_y
            return new_lp  # accept
        end

        # Shrink bracket
        if x1 < x0
            L = x1
        else
            R = x1
        end
    end

    # Exhausted iterations — restore original and return unchanged
    theta.values[idx] = x0
    return current_lp
end

"""Workspace-aware slice sampling — threads ws through _tempered_lp."""
function _slice_sample_coord!(theta::Theta, data::Data, idx::Int,
                               w::Float64, current_lp::Float64,
                               beta::Float64, rng::AbstractRNG,
                               ws::PTWorkspace;
                               max_steps::Int = 32,
                               ctr::Union{LikelihoodCounter, Nothing}=nothing)
    x0 = theta.values[idx]
    log_y = current_lp + log(rand(rng))

    u = rand(rng)
    L = x0 - u * w
    R = L + w

    for _ in 1:max_steps
        theta.values[idx] = L
        lp = _tempered_lp(theta, data, beta, ws; ctr=ctr)
        lp < log_y && break
        L -= w
    end

    for _ in 1:max_steps
        theta.values[idx] = R
        lp = _tempered_lp(theta, data, beta, ws; ctr=ctr)
        lp < log_y && break
        R += w
    end

    for _ in 1:1024
        x1 = L + rand(rng) * (R - L)
        theta.values[idx] = x1
        new_lp = _tempered_lp(theta, data, beta, ws; ctr=ctr)

        if new_lp > log_y
            return new_lp
        end

        if x1 < x0
            L = x1
        else
            R = x1
        end
    end

    theta.values[idx] = x0
    return current_lp
end

"""
    _tempered_lp(theta, data, beta; ctr) -> Float64

Tempered log-posterior: beta * log_L + log_pi.
"""
@inline function _tempered_lp(theta::Theta, data::Data, beta::Float64;
                               ctr::Union{LikelihoodCounter, Nothing}=nothing)
    lp = log_prior(theta)
    isfinite(lp) || return -Inf
    ll = _eval_ll(theta, data, ctr)
    isfinite(ll) || return -Inf
    return beta * ll + lp
end

"""Workspace-aware tempered log-posterior."""
@inline function _tempered_lp(theta::Theta, data::Data, beta::Float64,
                               ws::PTWorkspace;
                               ctr::Union{LikelihoodCounter, Nothing}=nothing)
    lp = log_prior(theta)
    isfinite(lp) || return -Inf
    ll = _eval_ll(theta, data, ctr, ws)
    isfinite(ll) || return -Inf
    return beta * ll + lp
end

# =====================================================================
# Random-walk Metropolis-within-Gibbs (RWM) within-model kernel
#
# Single-coord Gaussian random walk per active parameter, with per-coord
# σ adapted via Robbins-Monro to target ~0.44 acceptance (the optimum
# for univariate proposals; Roberts & Rosenthal 2001). Each coord costs
# ~1 likelihood eval; slice sampling costs ~5-10. On data-rich joint
# fits this is the difference between minutes and hours per round.
#
# Trade-offs vs slice:
#   * RWM mixes worse on heavy-tailed or strongly-correlated posteriors
#   * Slice handles multi-modality better (slice-along-coord can jump
#     across local barriers; RWM can't unless σ is large)
#   * RWM needs warmup adaptation; slice needs none
#
# Use case: joint RV+phot trans-dim PT where the per-iter likelihood
# cost is the bottleneck and the posterior is locally Gaussian. Falls
# back to slice for robustness when in doubt.
# =====================================================================

"""
    _within_model_move_rwm!(callback, theta, data, rng, ws, log_pi, log_L;
                              beta=1.0, adapt=true, ctr=nothing) -> Bool

Single-coord random-walk Metropolis-within-Gibbs with Robbins-Monro
per-coord σ adaptation. Updates `ws.rwm_sigmas[i]` toward the target
acceptance rate during the warmup window. Mutates `theta.values`,
`ws.rwm_attempts/accepts/sigmas`. Returns `true` if at least one
coord move was accepted.
"""
function _within_model_move_rwm!(callback, theta::Theta, data::Data,
                                   rng::AbstractRNG, ws::PTWorkspace,
                                   log_pi::Float64, log_L::Float64;
                                   beta::Float64 = 1.0,
                                   adapt::Bool = true,
                                   target_accept::Float64 = 0.44,
                                   ctr::Union{LikelihoodCounter, Nothing}=nothing)
    n_act = _active_unfrozen!(ws, theta.params, theta.td)
    n_act == 0 && return false

    layout = theta.params.layout
    current_lp = beta * log_L + log_pi

    any_accepted = false
    @inbounds for i in 1:n_act
        idx = ws.active_idx[i]
        pos = ws.active_pos[i]   # position in unfrozen vector

        old_val = theta.values[idx]
        σ = ws.rwm_sigmas[pos]
        prop = old_val + σ * randn(rng)

        # Reject out-of-bounds via prior — log_prior returns -Inf
        theta.values[idx] = prop
        new_lp = _tempered_lp(theta, data, beta, ws; ctr=ctr)
        ws.rwm_attempts[pos] += 1

        if log(rand(rng)) < new_lp - current_lp
            current_lp = new_lp
            ws.rwm_accepts[pos] += 1
            any_accepted = true
        else
            theta.values[idx] = old_val
        end
    end

    # Robbins-Monro adaptation of σ_i (only during warmup). Step size
    # 1/n^0.75 matches MoMS Eq. D1 (van den Bergh+ 2026) which is the
    # form Roberts-Rosenthal use to prove containment + diminishing
    # adaptation for adaptive Metropolis. Per-coord, so target_accept
    # = 0.44 (univariate optimum, RGG 1997).
    if adapt
        @inbounds for i in 1:n_act
            pos = ws.active_pos[i]
            n   = ws.rwm_attempts[pos]
            n >= 8 || continue   # need a few attempts before reacting
            rate = ws.rwm_accepts[pos] / n
            step = (rate - target_accept) / Float64(n)^0.75
            ws.rwm_sigmas[pos] *= exp(step)
            ws.rwm_sigmas[pos] = clamp(ws.rwm_sigmas[pos], 1e-6, 1e6)
        end
    end

    new_log_pi = log_prior(theta)
    new_log_L = _eval_ll(theta, data, ctr, ws)
    callback(new_log_pi, new_log_L)
    return any_accepted
end

function _select_strategy(td::TransDimConfig, rng::AbstractRNG)
    r = rand(rng)
    cumsum = 0.0
    for (i, w) in enumerate(td.birth_weights)
        cumsum += w
        if r < cumsum
            return td.birth_strategies[i]
        end
    end
    return td.birth_strategies[end]
end
