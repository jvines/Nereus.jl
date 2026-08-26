# Trans-dim parallel-tempered affine-invariant ensemble MCMC.
#
# `sample_transdim_ptemcee` extends `sample_ptemcee` with MoMS variable-
# selection moves on the planet count. Each walker maintains its own
# `TransDimState` (which planets are active); at each step we interleave
# the standard stretch ensemble move (operating on continuous params)
# with a MoMS birth/death move (operating on the planet activation
# bits, with a Gaussian random walk on the activated/deactivated
# planet's continuous params).
#
# Constant-dimension parameter space (à la van den Bergh+ 2026,
# arXiv:2604.27791): inactive planets sit at deterministic "off-values",
# so the parameter vector length is always max_kplanet × per-planet
# slots. Stretch moves on inactive slots are harmless (they propose
# new off-values which are then rejected via the spike-and-slab prior
# at the next MoMS evaluation).
#
# Supports planet trans-dim, noise-model trans-dim, or both: `td.planets`
# and `td.noise` are independent, and at least one must be set. Noise
# toggling has its own annealed birth (`n_noise_bridge`/`n_noise_relax`),
# its own swap move (`noise_swap`), and per-temperature accept/propose
# accounting in `noise_td_proposed`/`noise_td_accepted`.

using Random: AbstractRNG, MersenneTwister
using Statistics: median, quantile
import MCMCChains

export sample_transdim_ptemcee, TransDimPTemceeResult

"""
    TransDimPTemceeResult

Output of `sample_transdim_ptemcee`. Same fields as `PTemceeResult`,
plus per-sample planet activation state (concatenated as additional
columns of `chains` — `:n_planets` plus one column per planet slot).
"""
struct TransDimPTemceeResult
    chains::MCMCChains.Chains
    log_evidence::Float64
    evidence_report::Any
    acceptance_within::Vector{Float64}
    acceptance_swap::Vector{Float64}
    acceptance_transdim::Vector{Float64}      # per-planet
    betas::Vector{Float64}
    n_evals::Int
    # RAW trans-dim move counts — rates alone are not diagnostics: a
    # rounded rate of 0.000 (or an uninstrumented 0/0) reads as "frozen"
    # while the chain is in fact transitioning (HD 18599 post-mortem).
    td_proposed::Vector{Int}                  # per-planet, planet moves
    td_accepted::Vector{Int}                  # per-planet, planet moves
    noise_td_proposed::Vector{Int}            # per-temp, noise toggles
    noise_td_accepted::Vector{Int}            # per-temp, noise toggles
    # Per-temp planet moves split by direction. A death acceptance RATE per
    # rung is the only way to test whether the ladder is hot enough for PT to
    # rescue an entrenched model order: if even the hottest rung's death
    # acceptance is ~0, swaps have nothing to propagate down to β=1.
    planet_birth_proposed::Vector{Int}        # per-temp
    planet_birth_accepted::Vector{Int}        # per-temp
    planet_death_proposed::Vector{Int}        # per-temp
    planet_death_accepted::Vector{Int}        # per-temp
end

# Debug event log for trans-dim moves, gated by ENV["TD_DEBUG_BIRTHS"]="1".
# Each event: (step, temp, P, code, dll) with code 1=birth accepted, 2=death
# accepted (P of the killed planet), 3=multi-try selected a candidate
# (pre-accept), 4=multi-try candidate evaluated. dll = candidate logL − the
# walker's current logL (NaN where not applicable). Read
# Nereus._TD_DEBUG_EVENTS after a run to autopsy why a specific planet
# never survives. Cleared at sampler start.
const _TD_DEBUG_EVENTS = Vector{NTuple{5,Float64}}()
const _TD_DEBUG_LOCK = ReentrantLock()
@inline function _td_debug!(on::Bool, step::Int, t::Int, P::Float64, code::Int,
                             dll::Float64 = NaN)
    on || return nothing
    lock(_TD_DEBUG_LOCK) do
        push!(_TD_DEBUG_EVENTS, (Float64(step), Float64(t), P, Float64(code), dll))
    end
    nothing
end

# After a birth, `_sort_group_periods!` may have moved the NEWBORN's params
# to a different slot than the newly-activated bit (insertion ordering). Find
# the active slot of `new_theta` whose period matches no active slot of the
# pre-move `old_theta` — that's where the newborn lives (periods are exact
# copies under the sort, so == is safe). Refinement and the birth autopsy
# must target this slot, not the bit-flip slot.
function _newborn_slot(old_theta::Theta, new_theta::Theta, fallback::Int)
    tdn = new_theta.td; tdo = old_theta.td
    (tdn === nothing || tdo === nothing) && return fallback
    @inbounds for k in eachindex(tdn.planet_active)
        tdn.planet_active[k] || continue
        Pk = planet_P(new_theta, k)
        matched = false
        for j in eachindex(tdo.planet_active)
            tdo.planet_active[j] || continue
            if planet_P(old_theta, j) == Pk
                matched = true
                break
            end
        end
        matched || return k
    end
    return fallback
end

# Per-parameter RWM step size for post-birth refinement, in NATURAL units.
# Period is BLS-precise so it gets a tiny step; geometry/eccentricity get
# fixed moderate steps; K/rr scale with their value. `time_param` distinguishes
# Tc (days) from Mo (radians) for the time anchor.
@inline function _refine_sigma(blk, slot, val::Float64, time_param::Symbol)
    if slot == blk.P
        return max(1e-6, 1e-4 * abs(val))
    elseif has_geometry(blk) && slot == blk.b
        return 0.08
    elseif has_geometry(blk) && slot == blk.r
        return max(1e-4, 0.1 * abs(val))
    elseif slot == blk.t
        return (time_param === :Tc || time_param === :Tp) ? 0.01 : 0.05
    elseif slot == blk.e1 || slot == blk.e2
        return 0.05
    elseif has_K(blk) && slot == blk.K
        return max(0.5, 0.1 * abs(val))
    else
        return 0.05
    end
end

"""
    sample_transdim_ptemcee(target, data; td, kwargs...) -> TransDimPTemceeResult

Trans-dim parallel-tempered ensemble sampler with MoMS variable
selection on planet count.

# Required
- `target::NereusTarget`
- `data::Data`
- `td::TransDimConfig` — trans-dim configuration. Must have at least one
  of `planets = true` or `noise = true`; both together is the usual case
  (joint planet-count and noise-model selection).

# Trans-dim kwargs
- `inclusion_prior::Float64 = 0.5` — Bernoulli prior P(γ_k = 1).
- `moms_init_scale::Float64 = 1.0` — initial scale multiplier for the
  MoMS Gaussian RW proposal in active-component dimensions.
- `informed_birth_fraction::Float64 = 0.0` — fraction of births that use a
  data-informed proposal (`JointInformedBirth`: BLS photometry peaks + RV
  Lomb-Scargle, with depth→radius→mass→K and BLS-t0 anchoring; auto-falls back
  to RV-only `InformedBirth` when there is no photometry) instead of the native
  MoMS Gaussian RW. Forward/reverse use the same strategy family so detailed
  balance is preserved (same pattern as `sample_moms`). Critical for finding
  weak / arbitrary-period planets — `0.0` (the old default) gives blind RW
  births that rarely land on a real period.

# PT / ensemble kwargs (same as `sample_ptemcee`)
- `n_temps::Int = 5`
- `n_walkers::Int = 100`
- `n_steps::Int = 2000`
- `n_burnin::Int = 1000`
- `betas::Union{Nothing,Vector{Float64}} = nothing`
- `beta_min::Real = 1e-4` — hottest inverse-temperature. The default ladder is
  geometric across `[beta_min, 1]` (`β_i = beta_min^(i/(n_temps-1))`), so adding
  temps DENSIFIES the cold end (closing the cold-rung swap gap) rather than
  pushing T_max ever hotter. Ignored when `betas` is supplied.
- `stretch_a::Float64 = 2.0`
- `seed::Int = 1`
- `thin::Int = 1`
- `show_progress::Bool = true`

# Notes
- Within-step ordering: stretch on continuous params → MoMS planet
  move per (walker, temp) → swap between temperatures. The MoMS move
  fires with probability `td.transdim_fraction` per (walker, temp)
  per step.
- Tempered M-H: the MoMS acceptance uses `β · (log_L_new − log_L_old)`
  so hot chains explore more freely between models.
"""
function sample_transdim_ptemcee(
    target::NereusTarget,
    data::Data;
    td::TransDimConfig,
    inclusion_prior::Real = 0.5,
    moms_init_scale::Real = 1.0,
    informed_birth_fraction::Real = 0.0,
    target_birth_accept::Real = 0.234,
    n_birth_refine::Int = 0,
    n_birth_tries::Int = 1,
    # Annealed noise birth. 1 = the ordinary single-draw birth (default, so
    # behaviour is unchanged unless asked for); >1 bridges through that many
    # stages, relaxing the newborn `n_noise_relax` steps at each.
    n_noise_bridge::Int = 1,
    n_noise_relax::Int = 2,
    n_temps::Int = 5,
    n_walkers::Int = 100,
    n_steps::Int = 2000,
    n_burnin::Int = 1000,
    betas::Union{Nothing,AbstractVector} = nothing,
    beta_min::Real = 1e-4,
    stretch_a::Real = 2.0,
    seed::Int = 1,
    thin::Int = 1,
    show_progress::Bool = true,
    noise_swap::Bool = true,
    noise_swap_rate::Real = 0.5,
    adapt_ladder::Bool = false,
    ladder_adapt_window::Int = 50,
    ladder_adapt_ν0::Real = 10.0,
    ladder_adapt_K::Real = 1.0,
    untemper_transit::Bool = false,
)
    # JSON may deliver floats-as-Int and arrays-as-JSON3.Array; normalize.
    inclusion_prior         = Float64(inclusion_prior)
    moms_init_scale         = Float64(moms_init_scale)
    informed_birth_fraction = Float64(informed_birth_fraction)
    target_birth_accept     = Float64(target_birth_accept)
    stretch_a               = Float64(stretch_a)
    ladder_adapt_ν0         = Float64(ladder_adapt_ν0)
    ladder_adapt_K          = Float64(ladder_adapt_K)

    # UNTEMPERED TRANSIT — legitimate ONLY when the transit likelihood is
    # INVARIANT across the models being selected among. The noise menu is all
    # RV-side (ErrorScale / CeleriteRotation / AD / ActivityGP), so L_transit is
    # literally the same function for every member ⇒ it may live in the reference,
    # and Z_ref cancels exactly in every noise-model Bayes factor. Trans-dim
    # PLANETS change the transit model, so the same move would judge every newborn
    # against full-strength photometry at EVERY rung (births die everywhere) and
    # Z_ref would no longer cancel. Same invariance condition governs both.
    let has_phot = !isempty(data.t_phot)
        if untemper_transit
            (td.planets && has_phot) && throw(ArgumentError(
                "untemper_transit=true is invalid with trans-dim planets + photometry: " *
                "L_transit is not invariant across planet counts, so births are judged " *
                "against full-strength photometry at every temperature and die, and " *
                "Z_ref no longer cancels. Use RV-only trans-dim for planet search."))
            has_phot || @warn "untemper_transit=true but no photometry present — no effect."
            has_phot && @info "untemper_transit: transit likelihood held in the UNTEMPERED " *
                "reference, π_β ∝ prior·L_transit·L_RV^β. Transit is at full strength in " *
                "every rung; β=1 remains the exact joint posterior (nothing frozen or " *
                "pre-fit). NOTE: TI/SS/H+ now return logZ RELATIVE to the " *
                "transit-conditioned reference — cancels exactly in noise-model Bayes " *
                "factors; add the transit-only evidence for absolute logZ."
        end
    end
    betas = betas === nothing ? nothing : collect(Float64, betas)
    (td.planets || td.noise) || throw(ArgumentError(
        "sample_transdim_ptemcee requires td.planets or td.noise = true"))
    0 < inclusion_prior < 1 || throw(ArgumentError(
        "inclusion_prior must be in (0, 1)"))

    params  = target.params
    layout  = params.layout
    n_dim   = length(layout.unfrozen_idx)
    unfrozen_idx = layout.unfrozen_idx
    max_k   = td.max_kplanet
    td_debug = get(ENV, "TD_DEBUG_BIRTHS", "0") == "1"
    td_debug && empty!(_TD_DEBUG_EVENTS)

    # Jitter slots, refined JOINTLY with a freshly-birthed planet. At Np the
    # jitter inflates to absorb the unmodelled planet's dips, so the planet's
    # conditional gain at FIXED jitter is capped far below its true ΔlogL
    # (measured on WASP-47 e: +47 ceiling vs +210 true → births too weak to
    # lock). Multiplicative RWM (jitter is log-scale) with the Hastings
    # log-ratio correction in the refinement acceptance.
    jitter_refine_idx = Int[j for j in eachindex(layout.unfrozen_names) if
                             occursin("jitter", String(layout.unfrozen_names[j]))]

    # Goodman-Weare: each half needs ≥ n_dim+1 walkers
    n_walkers_eff = max(n_walkers, 2 * n_dim + 2)
    n_walkers_eff % 2 == 1 && (n_walkers_eff += 1)

    # Geometric ladder DENSIFIED at fixed β_min: β_i = β_min^(i/(N-1)), spanning
    # [β_min, 1]. Adding temps closes the cold-end gap (raises cold swap acceptance)
    # instead of pushing T_max to ever-hotter, wasted rungs — which is what a
    # FIXED per-rung ratio (e.g. (1/√5)^i) does (β_min = (1/√5)^(N-1) shrinks with N,
    # so the extra rungs land at β≈0 while the cold spacing 1→0.447→0.2 never changes).
    beta_min = Float64(beta_min)
    βs = if betas !== nothing
        copy(betas)
    elseif n_temps <= 1
        Float64[1.0]
    else
        Float64[beta_min^(i / (n_temps - 1)) for i in 0:(n_temps - 1)]
    end
    length(βs) == n_temps || throw(ArgumentError(
        "Provided `betas` has length $(length(βs)), expected $n_temps"))

    # --- Per-thread mutable state -------------------------------------
    n_thr = max(Threads.nthreads(), 1)
    n_noise_slots_init = length(params.config.noise_models)
    thread_theta  = [Theta{Float64}(params;
                       td = TransDimState(; max_planets = max_k,
                                            n_noise = n_noise_slots_init))
                       for _ in 1:n_thr]
    # Per-thread PTWorkspace → ws-aware likelihood: reuses preallocated buffers
    # (no per-call Vector allocs → no GC thrash over the millions of ensemble-PT
    # evals) + the per-planet flux cache + total phot-ll cache. Same proven path
    # as PT/rjmcmc/nested; one ws per thread, indexed by threadid() (:static).
    thread_ws = [PTWorkspace(params, max_k, n_noise_slots_init;
                             n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                 for _ in 1:n_thr]
    thread_proposal = [Vector{Float64}(undef, n_dim) for _ in 1:n_thr]
    thread_rngs   = [MersenneTwister(seed + 1_000_003 * t) for t in 1:n_thr]
    rng_master    = MersenneTwister(seed)

    # MoMS proposal strategy — shared across threads. Scales are tuned
    # during burn-in (Robbins-Monro); writes are protected by atomic
    # updates of per-planet acceptance counters then a single-threaded
    # adapt step at the end of each window.
    strategy = MoMSBirth(params; init_scale = moms_init_scale,
                          rng = MersenneTwister(seed + 7))

    # --- Walker state -------------------------------------------------
    state    = Array{Float64,3}(undef, n_temps, n_walkers_eff, n_dim)
    logL_arr = fill(-Inf, n_temps, n_walkers_eff)
    logπ_arr = fill(-Inf, n_temps, n_walkers_eff)
    # Per-walker TransDimState (planet activation bits + count)
    n_noise_slots = length(params.config.noise_models)
    # Indices into `params.config.noise_models` for toggleable models,
    # mirroring sample_moms' output column layout.
    noise_toggle_idx = Int[]
    if td.noise && !isempty(td.toggleable)
        for nm in td.toggleable
            nm_idx = findfirst(==(nm), params.config.noise_models)
            nm_idx === nothing && continue
            push!(noise_toggle_idx, nm_idx)
        end
    end
    do_noise = td.noise && !isempty(noise_toggle_idx)
    # Exclusion groups as noise-model indices, for init validity below.
    noise_excl_idx = Vector{Vector{Int}}()
    if do_noise
        for grp in td.noise_exclusion_groups
            idxs = Int[]
            for nm in grp
                j = findfirst(==(nm), params.config.noise_models)
                j === nothing || push!(idxs, j)
            end
            length(idxs) >= 2 && push!(noise_excl_idx, idxs)
        end
    end

    td_states = Matrix{TransDimState}(undef, n_temps, n_walkers_eff)
    for t in 1:n_temps, w in 1:n_walkers_eff
        td_states[t, w] = TransDimState(; max_planets = max_k,
                                          n_noise = n_noise_slots)
        # Bernoulli-init planet + (toggleable) noise activations, so
        # the prior over k and active noise components is well-sampled
        # at step 0. With td.planets=false the planet count is FIXED:
        # all slots active, no Bernoulli — previously the Bernoulli ran
        # unconditionally, so a noise-only model-selection run (planets=false)
        # started half its walkers with the planet(s) missing.
        for k in 1:max_k
            if !td.planets || rand(rng_master) < inclusion_prior
                activate_planet!(td_states[t, w], k)
            end
        end
        for nm_idx in noise_toggle_idx
            if rand(rng_master) < inclusion_prior
                td_states[t, w].noise_active[nm_idx] = true
            end
        end
        # Exclusion groups: the independent Bernoulli flips above can start
        # a walker with >1 member of a group active — a FORBIDDEN config
        # that birth/death moves can never repair (births check exclusions,
        # but the violating state already exists; measured on HD 18599:
        # 25% of walkers started AD+AGP both-active and the run burned its
        # early phase killing them off). Rejection-resample each group's
        # bits until valid — exact conditional-Bernoulli given ≤1 active.
        for grp in noise_excl_idx
            while count(i -> td_states[t, w].noise_active[i], grp) > 1
                for i in grp
                    td_states[t, w].noise_active[i] =
                        rand(rng_master) < inclusion_prior
                end
            end
        end
        # NON-toggleable noise models are ALWAYS active — the TransDimState
        # constructor defaults all noise bits to false and only toggleable
        # models were initialized above, so an always-on noise model (e.g.
        # a GP with transdim_noise=false) was silently INACTIVE in every
        # walker for the whole run: its hyperparameters became likelihood
        # spectators (measured on K2-138 — bitwise-identical runs under
        # different gp_Q0 priors). Mirrors `winning_td_state`'s convention.
        for nm_idx in 1:n_noise_slots
            if !(nm_idx in noise_toggle_idx)
                td_states[t, w].noise_active[nm_idx] = true
            end
        end
    end

    @inline function eval_bounded!(x::AbstractVector{Float64},
                                     tds::TransDimState, tid::Int)
        theta = thread_theta[tid]
        wb = thread_ws[tid]
        @inbounds for (j, idx) in enumerate(unfrozen_idx)
            theta.values[idx] = x[j]
        end
        copyto!(theta.td.planet_active, tds.planet_active)
        theta.td.n_planets_active = tds.n_planets_active
        # noise_active MUST be synced too: without this, every stretch/
        # refinement eval ran under whatever noise mask this thread's theta
        # last held (another walker's, after any td move on this thread) —
        # for noise-toggling runs the stored logL/logπ no longer matched the
        # walker's own noise config (HD 18599 noise-selection post-mortem, 2026-06-11).
        copyto!(theta.td.noise_active, tds.noise_active)
        lp = spike_slab_log_prior(theta, strategy, inclusion_prior)
        isfinite(lp) || return (-Inf, -Inf)
        ll_rv = rv_log_likelihood(theta, data, wb)
        ll_tr = transit_log_likelihood(theta, data, wb)
        # UNTEMPERED TRANSIT (see `untemper_transit`): fold the transit term into
        # the untempered reference, so the tempered path is
        #     π_β ∝ prior · L_transit · L_RV^β
        # with the transit at FULL strength in every rung (nothing frozen, nothing
        # pre-fit, β=1 is still the exact joint posterior). Returning it inside
        # `lp` makes all three downstream users correct BY CONSTRUCTION — the
        # within-chain acceptance β·Δll + Δlp, the swap ratio Δβ·Δll, and the TI
        # accumulator over ll. Patching only the swap ratio would leave the swap
        # inconsistent with the sampled targets and break detailed balance.
        return untemper_transit ? (lp + ll_tr, ll_rv) : (lp, ll_rv + ll_tr)
    end

    # Same reference split as `eval_bounded!`, for the birth/death/refinement
    # sites that build (log_prior, log_likelihood) by hand. EVERY such site must
    # use it: the stored logπ_arr/logL_arr follow this convention, so a site that
    # kept the transit inside the likelihood would compare a full-likelihood
    # candidate against a reference-split incumbent and silently corrupt the
    # acceptance ratio.
    @inline function split_ref(lp0::Float64, th, tid::Int)
        ll_rv = rv_log_likelihood(th, data, thread_ws[tid])
        ll_tr = transit_log_likelihood(th, data, thread_ws[tid])
        return untemper_transit ? (lp0 + ll_tr, ll_rv) : (lp0, ll_rv + ll_tr)
    end

    # Per-noise-model refinement table: (unfrozen pos, additive σ from the
    # prior's central width, positive-support flag). Built once; consumed
    # by refine_noise_birth! below.
    noise_refine_slots = Vector{Vector{Tuple{Int, Float64, Bool}}}(undef,
                                                                    n_noise_slots)
    for nm_idx in 1:n_noise_slots
        nm = params.config.noise_models[nm_idx]
        entries = Tuple{Int, Float64, Bool}[]
        for name in noise_param_names(nm, params.config.instruments)
            haskey(layout.name_to_idx, name) || continue
            slot = layout.name_to_idx[name]
            uf_pos = findfirst(==(slot), layout.unfrozen_idx)
            uf_pos === nothing && continue
            prior_ps = layout.unfrozen_priors[uf_pos]
            lo, _ = bounds(prior_ps)
            q16 = quantile(prior_ps, 0.16)
            q84 = quantile(prior_ps, 0.84)
            σ_add = 0.1 * max(q84 - q16, 1e-12)
            isfinite(σ_add) || (σ_add = 1e-3)
            push!(entries, (uf_pos, σ_add, isfinite(lo) && lo >= 0))
        end
        noise_refine_slots[nm_idx] = entries
    end

    # COST-AWARE refine budget. The post-birth climb does
    # refine × (model's param count) likelihood evals, and eval cost is
    # wildly model-dependent: a 4-channel ActivityGP eval is a dense
    # ~(n·C)² Cholesky, ~100× a white/MA eval (measured 2.3 ms vs ~20 µs
    # on HD 18599). A FLAT n_birth_refine therefore spends ~0.5 s on every
    # AGP birth-climb (18 × 13 params × 2.3 ms) while a cheap model's climb
    # costs ~nothing — the AGP climb dominates wall-clock. Measure each
    # toggleable model's eval cost once and scale its climb budget
    # inversely, floored at `min_refine` so even the expensive models get
    # a real climb. Cheap models keep the full budget.
    noise_refine_budget = fill(n_birth_refine, n_noise_slots)
    if n_birth_refine > 1 && do_noise
        min_refine = max(3, n_birth_refine ÷ 6)
        x_med = Float64[clamp(quantile(layout.unfrozen_priors[j], 0.5),
                              bounds(layout.unfrozen_priors[j])...)
                        for j in 1:n_dim]
        tds_c = TransDimState(; max_planets = max_k, n_noise = n_noise_slots)
        for k in 1:max_k; activate_planet!(tds_c, k); end
        costs = Dict{Int, Float64}()
        for nm_idx in noise_toggle_idx
            isempty(noise_refine_slots[nm_idx]) && continue
            fill!(tds_c.noise_active, false)
            for j in 1:n_noise_slots
                (j in noise_toggle_idx) || (tds_c.noise_active[j] = true)
            end
            tds_c.noise_active[nm_idx] = true
            eval_bounded!(x_med, tds_c, 1)                  # warm
            t = @elapsed for _ in 1:3; eval_bounded!(x_med, tds_c, 1); end
            costs[nm_idx] = t / 3
        end
        if !isempty(costs)
            tmin = minimum(values(costs))
            for (nm_idx, c) in costs
                noise_refine_budget[nm_idx] =
                    clamp(round(Int, n_birth_refine * tmin / c),
                          min_refine, n_birth_refine)
            end
        end
    end

    # Post-birth refinement for NOISE models — the same medicine that fixed
    # planet births. A noise birth draws ALL of the model's hyperparams
    # from their priors in one shot, so for multi-param models (4-channel
    # Rajpaul AGP: 12+, multi-indicator AD: 12) a competitive config is
    # never proposed and the model reads as "rejected by the data" (HD
    # 18599: trans-dim occupancy contradicted the fixed-run ΔlogZ by 230
    # nats). Coordinate M-H over the newborn model's params climbs the
    # drawn config toward its mode; positive-support params step
    # multiplicatively (Hastings-corrected), the rest additively at a
    # tenth of the prior's central width. Standard within-model M-H ⇒
    # detailed balance is safe wherever it runs.
    function refine_noise_birth!(t::Int, w::Int, nm_idx::Int, β::Float64,
                                  tid::Int, trng)
        entries = noise_refine_slots[nm_idx]
        isempty(entries) && return
        rbuf = thread_proposal[tid]
        for _ in 1:noise_refine_budget[nm_idx]
            for (jr, σ_add, ispos) in entries
                @inbounds for j in 1:n_dim
                    rbuf[j] = state[t, w, j]
                end
                hast = 0.0
                old = rbuf[jr]
                if ispos && old > 0
                    nv = old * exp(0.2 * randn(trng))
                    rbuf[jr] = nv
                    hast = log(nv) - log(old)
                else
                    rbuf[jr] = old + σ_add * randn(trng)
                end
                lp2, ll2 = eval_bounded!(rbuf, td_states[t, w], tid)
                if isfinite(ll2) && log(rand(trng)) <
                            β * (ll2 - logL_arr[t, w]) + (lp2 - logπ_arr[t, w]) + hast
                    @inbounds for j in 1:n_dim
                        state[t, w, j] = rbuf[j]
                    end
                    logL_arr[t, w] = ll2; logπ_arr[t, w] = lp2
                end
            end
        end
    end

    # (3) Post-birth refinement: climb a freshly-birthed planet toward its
    # mode with PER-SLOT (one-param-at-a-time) M-H sweeps over the new
    # planet's params + a multiplicative jitter step (Hastings-corrected).
    # Per-slot beats all-params-at-once RWM on sharp transit modes: a 7-D
    # joint Gaussian step is nearly always rejected near the needle, while
    # coordinate-wise moves accept independently and track the (P, Tc)
    # ridge. A lone fresh planet CANNOT be refined by the stretch move (its
    # only valid partners would need the same planet active), so this is
    # its only mobility until the ensemble populates. Standard within-model
    # M-H ⇒ detailed balance is safe.
    function refine_birth!(t::Int, w::Int, kb::Int, β::Float64,
                            tid::Int, trng)
        blk = params.layout.planet_blocks[kb]
        tparam = params.config.parametrization.time
        rbuf = thread_proposal[tid]
        for _ in 1:n_birth_refine
            for slot in planet_slot_indices(blk)
                jr = findfirst(==(slot), unfrozen_idx)
                jr === nothing && continue
                @inbounds for j in 1:n_dim
                    rbuf[j] = state[t, w, j]
                end
                rbuf[jr] += _refine_sigma(blk, slot, rbuf[jr], tparam) * randn(trng)
                lp2, ll2 = eval_bounded!(rbuf, td_states[t, w], tid)
                if isfinite(ll2) && log(rand(trng)) <
                            β * (ll2 - logL_arr[t, w]) + (lp2 - logπ_arr[t, w])
                    @inbounds for j in 1:n_dim
                        state[t, w, j] = rbuf[j]
                    end
                    logL_arr[t, w] = ll2; logπ_arr[t, w] = lp2
                end
            end
            for jj in jitter_refine_idx
                @inbounds for j in 1:n_dim
                    rbuf[j] = state[t, w, j]
                end
                old = rbuf[jj]
                old > 0 || continue
                nv = old * exp(0.15 * randn(trng))
                rbuf[jj] = nv
                hast = log(nv) - log(old)
                lp2, ll2 = eval_bounded!(rbuf, td_states[t, w], tid)
                if isfinite(ll2) && log(rand(trng)) <
                            β * (ll2 - logL_arr[t, w]) + (lp2 - logπ_arr[t, w]) + hast
                    @inbounds for j in 1:n_dim
                        state[t, w, j] = rbuf[j]
                    end
                    logL_arr[t, w] = ll2; logπ_arr[t, w] = lp2
                end
            end
        end
        nothing
    end

    # --- Initialize walkers from prior -------------------------------
    init_seeds = rand(rng_master, UInt64, n_temps * n_walkers_eff)
    Threads.@threads :static for task_idx in 1:(n_temps * n_walkers_eff)
        tid = Threads.threadid()
        t = (task_idx - 1) ÷ n_walkers_eff + 1
        w = (task_idx - 1) % n_walkers_eff + 1
        rng = MersenneTwister(init_seeds[task_idx])
        for _ in 1:200
            # allow_nonfinite: the bare target scores all noise active (−Inf for
            # eval-incompatible menu members); the real gate is eval_bounded!
            # against this walker's toggled td_state, below.
            x_b = _draw_from_prior(target, rng; allow_nonfinite=true)
            @inbounds for d in 1:n_dim
                state[t, w, d] = x_b[d]
            end
            lp, ll = eval_bounded!(@view(state[t, w, :]),
                                    td_states[t, w], tid)
            if isfinite(lp) && isfinite(ll)
                logπ_arr[t, w] = lp
                logL_arr[t, w] = ll
                break
            end
        end
    end

    # Prior-seeded CovarianceNoise hyperparams.
    # `_draw_from_prior` samples from broad priors (LogUniform spans
    # several decades on the AGP amp / λ_e), so walker init can land
    # an active GP at an extreme amp/λ that produces a poorly-
    # conditioned covariance. Subsequent trans-dim births draw from
    # the same broad prior, so once a walker drops out of the active
    # state it almost never recovers — observed e.g. on HD 18599 where
    # P(AGP active) settled to ~1.6% within the first 100 steps.
    #
    # Mitigation: post-init, re-seed every CovarianceNoise hyperparameter
    # slot from the CENTRAL HALF of its prior (per-walker staggered
    # quantiles, u ∈ [0.25, 0.75]). NOT the single prior median for all
    # walkers: the Goodman-Weare stretch proposes partner + z·(self −
    # partner), so a dimension where the whole ensemble holds one
    # identical value is a FIXED POINT — the stretch can never diversify
    # it and the hyperparams stay frozen at the seed for the entire run
    # (measured on K2-138: all five gp_* posteriors bitwise-equal to
    # their prior medians with zero variance; the GP never engaged).
    # Central-band seeding keeps the conditioning benefit (no extreme
    # amp/λ draws) while giving the ensemble the spread it needs to mix
    # out to the full prior.
    cov_seed_slots = Tuple{Int, Any}[]   # (unfrozen pos, prior)
    for nm in params.config.noise_models
        nm isa CovarianceNoise || continue
        for name in noise_param_names(nm, params.config.instruments)
            haskey(layout.name_to_idx, name) || continue
            slot = layout.name_to_idx[name]
            uf_pos = findfirst(==(slot), layout.unfrozen_idx)
            uf_pos === nothing && continue
            push!(cov_seed_slots, (uf_pos, layout.unfrozen_priors[uf_pos]))
        end
    end
    if !isempty(cov_seed_slots)
        Threads.@threads :static for task_idx in 1:(n_temps * n_walkers_eff)
            tid = Threads.threadid()
            t = (task_idx - 1) ÷ n_walkers_eff + 1
            w = (task_idx - 1) % n_walkers_eff + 1
            trng_s = thread_rngs[tid]
            @inbounds for (uf_pos, prior_ps) in cov_seed_slots
                v = quantile(prior_ps, 0.25 + 0.5 * rand(trng_s))
                isfinite(v) || continue
                state[t, w, uf_pos] = v
            end
            lp, ll = eval_bounded!(@view(state[t, w, :]),
                                    td_states[t, w], tid)
            if isfinite(lp) && isfinite(ll)
                logπ_arr[t, w] = lp
                logL_arr[t, w] = ll
            end
        end
    end

    # --- Sample storage (β=1, post-burnin, thinned) ------------------
    # Column layout: unfrozen params + n_planets + per-planet bits +
    #                per-toggleable-noise bits + lp
    n_keep_per_walker = max(0, cld(n_steps - n_burnin, thin))
    n_keep_total      = n_keep_per_walker * n_walkers_eff
    n_noise_cols      = length(noise_toggle_idx)
    n_extra_cols      = 1 + max_k + n_noise_cols
    samples = Matrix{Float64}(undef, max(n_keep_total, 1), n_dim + n_extra_cols)
    lp_samples = Vector{Float64}(undef, max(n_keep_total, 1))
    keep_idx = 0

    evidence_acc = EvidenceAccumulator(length(βs))

    accept_within  = zeros(Float64, n_temps)
    propose_within = zeros(Int,     n_temps)
    accept_swap    = zeros(Float64, n_temps - 1)
    propose_swap   = zeros(Int,     n_temps - 1)
    accept_transdim  = zeros(Int, max_k)
    propose_transdim = zeros(Int, max_k)
    # Noise-toggle moves get their OWN per-temp raw counters. They were
    # previously uninstrumented: accept_transdim/propose_transdim only ever
    # counted planet moves, so a noise-only run (td.planets=false) displayed
    # td_acc = 0/0 → "0.000" and a chain that was in fact toggling read as
    # frozen (HD 18599 noise-selection post-mortem, 2026-06-11). Raw counts, never just
    # a rounded rate.
    accept_noise_transdim  = zeros(Int, n_temps)
    propose_noise_transdim = zeros(Int, n_temps)

    # Per-TEMPERATURE planet moves, split by DIRECTION. The PT immunity
    # argument for trans-dim selection is that hot chains accept the deaths
    # the cold chain cannot, and swaps carry that down to β=1 — which only
    # holds if the ladder is actually hot enough. Testing it needs a death
    # acceptance RATE per rung, and neither existing counter can give one:
    # propose_transdim/accept_transdim are per-planet-SLOT and pool births
    # with deaths, while propose_ntd/accept_ntd are per-temp but pool noise
    # births, noise deaths and two burn-in swap classes. Raw counts, so a
    # 0/0 rung reads as "never proposed" rather than "frozen".
    planet_birth_proposed = zeros(Int, n_temps)
    planet_birth_accepted = zeros(Int, n_temps)
    planet_death_proposed = zeros(Int, n_temps)
    planet_death_accepted = zeros(Int, n_temps)

    # Robbins-Monro scale adaptation during burn-in. Mirrors
    # sample_moms: per-window aggregate (accept, attempt) counts → bump
    # the per-planet RW proposal scale by exp(γ·(rate − target)) with
    # γ = 1/(iter+1)^0.75. Clamped per-step.
    adapt_n_attempts = zeros(Int, max_k)
    adapt_n_accepts  = zeros(Int, max_k)
    adapt_window     = max(50, n_burnin ÷ 20)

    n_evals_atomic = Threads.Atomic{Int}(n_temps * n_walkers_eff)
    pswap_prop = Threads.Atomic{Int}(0); pswap_acc = Threads.Atomic{Int}(0)  # planet↔activity swap diag
    pn_prop = Threads.Atomic{Int}(0); pn_acc = Threads.Atomic{Int}(0)   # planet→AD direction
    np_prop = Threads.Atomic{Int}(0); np_acc = Threads.Atomic{Int}(0)   # AD→planet direction
    accept_temp  = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    propose_temp = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    accept_td    = [Threads.Atomic{Int}(0) for _ in 1:max_k]
    propose_td   = [Threads.Atomic{Int}(0) for _ in 1:max_k]
    accept_ntd   = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    propose_ntd  = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    accept_pb    = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    propose_pb   = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    accept_pd    = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    propose_pd   = [Threads.Atomic{Int}(0) for _ in 1:n_temps]

    half = n_walkers_eff ÷ 2
    tasks_h1 = Vector{Tuple{Int,Int}}(undef, n_temps * half)
    tasks_h2 = Vector{Tuple{Int,Int}}(undef, n_temps * half)
    let i1 = 0, i2 = 0
        for t in 1:n_temps
            for w in 1:half;                  i1 += 1; tasks_h1[i1] = (t, w); end
            for w in (half + 1):n_walkers_eff; i2 += 1; tasks_h2[i2] = (t, w); end
        end
    end

    # ---- Stretch move (continuous params only; same as fixed-dim) ----
    function do_half_step!(tasks::Vector{Tuple{Int,Int}}, active_half::Symbol)
        partner_lo = active_half === :h1 ? half + 1 : 1
        partner_hi = active_half === :h1 ? n_walkers_eff : half
        Threads.@threads :static for task_idx in 1:length(tasks)
            tid = Threads.threadid()
            t, w = tasks[task_idx]
            β = βs[t]
            trng = thread_rngs[tid]
            buf  = thread_proposal[tid]

            w_partner = rand(trng, partner_lo:partner_hi)
            u = rand(trng)
            z = ((stretch_a - 1) * u + 1)^2 / stretch_a

            @inbounds for d in 1:n_dim
                buf[d] = state[t, w_partner, d] +
                         z * (state[t, w, d] - state[t, w_partner, d])
            end

            lp_prop, ll_prop = eval_bounded!(buf, td_states[t, w], tid)
            log_ratio = (n_dim - 1) * log(z) +
                        β * (ll_prop - logL_arr[t, w]) +
                        (lp_prop - logπ_arr[t, w])
            Threads.atomic_add!(propose_temp[t], 1)
            if log(rand(trng)) < log_ratio
                @inbounds for d in 1:n_dim
                    state[t, w, d] = buf[d]
                end
                logπ_arr[t, w] = lp_prop
                logL_arr[t, w] = ll_prop
                Threads.atomic_add!(accept_temp[t], 1)
            end
        end
        Threads.atomic_add!(n_evals_atomic, length(tasks))
    end

    # ---- Trans-dim MoMS step (per walker, tempered M-H) --------------
    # Each (t, w) attempts one trans-dim move with probability
    # `td.transdim_fraction`. With `do_noise` active, half the moves
    # toggle a noise indicator; the rest do MoMS planet birth/death.
    # Tempered M-H acceptance: β·Δlog_L + Δlog_π + log_q.
    function do_transdim_step!(step::Int)
        in_burnin = step <= n_burnin
        Threads.@threads :static for task_idx in 1:(n_temps * n_walkers_eff)
            tid = Threads.threadid()
            t = (task_idx - 1) ÷ n_walkers_eff + 1
            w = (task_idx - 1) % n_walkers_eff + 1
            trng = thread_rngs[tid]
            rand(trng) < td.transdim_fraction || continue

            theta = thread_theta[tid]
            @inbounds for (j, idx) in enumerate(unfrozen_idx)
                theta.values[idx] = state[t, w, j]
            end
            copyto!(theta.td.planet_active, td_states[t, w].planet_active)
            theta.td.n_planets_active = td_states[t, w].n_planets_active
            copyto!(theta.td.noise_active, td_states[t, w].noise_active)

            # DB-correct planet↔AD swap (ALL phases): trade a redundant planet
            # for the AD model that absorbs its signal (or vice versa) in one
            # accept/reject with the exact swap Hastings. This is the move the
            # standard birth/death can't make across the likelihood valley, and
            # being DB-correct it runs post-burn-in → drives occupancy to P(M|D).
            # Cold-side only (β>0.3) and infrequent: the swap runs a periodogram
            # + OLS fit per call, so firing it on every temp at 0.3 dominated
            # wall-time (the entrenchment it fixes is a cold-chain pathology, and
            # over a long run a low rate still fires plenty to escape it).
            if td.planets && do_noise && βs[t] > 0.3 && rand(trng) < 0.05
                β = βs[t]
                cand_s, lq_s = propose_planet_ad_swap(theta, trng, td.toggleable; data=data)
                if isfinite(lq_s)
                    is_pn = cand_s.td.n_planets_active < td_states[t, w].n_planets_active
                    Threads.atomic_add!(pswap_prop, 1)
                    Threads.atomic_add!(is_pn ? pn_prop : np_prop, 1)
                    lpc = spike_slab_log_prior(cand_s, strategy, inclusion_prior)
                    if isfinite(lpc)
                        lpc, llc = split_ref(lpc, cand_s, tid)
                        Threads.atomic_add!(n_evals_atomic, 1)
                        if isfinite(llc) && log(rand(trng)) <
                                β * (llc - logL_arr[t, w]) + (lpc - logπ_arr[t, w]) + lq_s
                            @inbounds for (j, idx) in enumerate(unfrozen_idx)
                                state[t, w, j] = cand_s.values[idx]
                            end
                            copy_into!(td_states[t, w], cand_s.td)
                            logπ_arr[t, w] = lpc; logL_arr[t, w] = llc
                            Threads.atomic_add!(pswap_acc, 1)
                            Threads.atomic_add!(is_pn ? pn_acc : np_acc, 1)
                        end
                    end
                end
                continue
            end

            # DB-correct WITHIN-GROUP noise swap (ALL phases): let a walker
            # entrenched in a flexible disfavoured model (GP-Rot/AGP) ESCAPE to
            # the high-evidence model (AD) in ONE exact-Hastings move. Birth/death
            # can't cross the "none" valley at β≈1, so without this the
            # disfavoured mode over-represents and occupancy ≠ P(M|D) (measured at
            # toy scale: GP-Rot 0.3% at a 14.8-nat gap, ~4 orders above evidence;
            # severe on deep modes). Cold-side, infrequent (periodogram+OLS per
            # call) — same rationale as the planet↔AD swap above.
            if noise_swap && do_noise && !isempty(td.noise_exclusion_groups) &&
               βs[t] > 0.3 && rand(trng) < noise_swap_rate
                β = βs[t]
                grp = td.noise_exclusion_groups[
                          rand(trng, 1:length(td.noise_exclusion_groups))]
                cand_s, lq_s = propose_noise_swap(theta, trng, grp,
                                                   td.toggleable; data=data)
                if isfinite(lq_s)
                    Threads.atomic_add!(propose_ntd[t], 1)
                    lpc = spike_slab_log_prior(cand_s, strategy, inclusion_prior)
                    if isfinite(lpc)
                        lpc, llc = split_ref(lpc, cand_s, tid)
                        Threads.atomic_add!(n_evals_atomic, 1)
                        if isfinite(llc) && log(rand(trng)) <
                                β * (llc - logL_arr[t, w]) +
                                (lpc - logπ_arr[t, w]) + lq_s
                            @inbounds for (j, idx) in enumerate(unfrozen_idx)
                                state[t, w, j] = cand_s.values[idx]
                            end
                            copyto!(td_states[t, w].noise_active,
                                     cand_s.td.noise_active)
                            logπ_arr[t, w] = lpc; logL_arr[t, w] = llc
                            Threads.atomic_add!(accept_ntd[t], 1)
                        end
                    end
                end
                continue
            end

            # planets=false ⇒ every trans-dim move is a noise toggle; the
            # planet branch below must never fire (the planet set is fixed).
            attempt_noise = do_noise && (!td.planets || rand(trng) < 0.5)
            (attempt_noise || td.planets) || continue

            if attempt_noise
                β = βs[t]

                # Coordinate-M-H climb of a candidate's newborn noise
                # params (burn-in mode-assembly; see the multi-try block).
                # Mutates cand.values; returns the climbed (lp, ll).
                function climb_newborn!(cand, nb::Int, lp0::Float64,
                                         ll0::Float64)
                    rbuf = thread_proposal[tid]
                    @inbounds for (j, idx) in enumerate(unfrozen_idx)
                        rbuf[j] = cand.values[idx]
                    end
                    cand_ll = ll0; cand_lp = lp0
                    for _ in 1:noise_refine_budget[nb]
                        for (jr, σ_add, ispos) in noise_refine_slots[nb]
                            old = rbuf[jr]
                            hast = 0.0
                            if ispos && old > 0
                                nv = old * exp(0.2 * randn(trng))
                                rbuf[jr] = nv
                                hast = log(nv) - log(old)
                            else
                                rbuf[jr] = old + σ_add * randn(trng)
                            end
                            lp2, ll2 = eval_bounded!(rbuf, cand.td, tid)
                            if isfinite(ll2) && log(rand(trng)) <
                                    β * (ll2 - cand_ll) +
                                    (lp2 - cand_lp) + hast
                                cand_ll = ll2; cand_lp = lp2
                            else
                                rbuf[jr] = old
                            end
                        end
                    end
                    @inbounds for (j, idx) in enumerate(unfrozen_idx)
                        cand.values[idx] = rbuf[j]
                    end
                    return (cand_lp, cand_ll)
                end

                # Within-exclusion-group SWAP (burn-in only, DB-free):
                # kill the active group member and birth-climb another in
                # ONE move. Without this, exclusion groups force a
                # challenger model through the "none" valley — killing a
                # FITTED incumbent costs hundreds of nats at β ≈ 1, so
                # whichever member assembles first is entrenched for the
                # whole run regardless of which model the data prefer
                # (measured: AD-only=1.000 with AGP never evaluated
                # head-to-head). The swap judges challenger-at-fitted-
                # quality vs incumbent-at-fitted-quality directly.
                if in_burnin && β > 0.3 && n_birth_tries > 1 &&
                   rand(trng) < 0.5
                    dead, lq_death = propose_noise_death(theta, trng,
                                                          td.toggleable; data=data)
                    if isfinite(lq_death)
                        best_α = -Inf
                        best_nt = nothing
                        best_lp = -Inf; best_ll = -Inf
                        for _ in 1:n_birth_tries
                            cand, lqb = propose_noise_birth(dead, trng,
                                              td.toggleable; data=data,
                                              exclusion_groups = td.noise_exclusion_groups)
                            isfinite(lqb) || continue
                            lpc = spike_slab_log_prior(cand, strategy,
                                                         inclusion_prior)
                            isfinite(lpc) || continue
                            lpc, llc = split_ref(lpc, cand, tid)
                            Threads.atomic_add!(n_evals_atomic, 1)
                            isfinite(llc) || continue
                            αc = β * (llc - logL_arr[t, w]) +
                                 (lpc - logπ_arr[t, w]) + lqb + lq_death
                            if αc > best_α
                                best_α = αc; best_nt = cand
                                best_lp = lpc; best_ll = llc
                            end
                        end
                        if best_nt !== nothing
                            Threads.atomic_add!(propose_ntd[t], 1)
                            nb = findfirst(i -> best_nt.td.noise_active[i] &&
                                                !td_states[t, w].noise_active[i],
                                            1:n_noise_slots)
                            if nb !== nothing && n_birth_refine > 0
                                best_lp, best_ll = climb_newborn!(best_nt, nb,
                                                                    best_lp, best_ll)
                                best_α = β * (best_ll - logL_arr[t, w]) +
                                         (best_lp - logπ_arr[t, w])
                            end
                            if log(rand(trng)) < best_α
                                @inbounds for (j, idx) in enumerate(unfrozen_idx)
                                    state[t, w, j] = best_nt.values[idx]
                                end
                                copyto!(td_states[t, w].noise_active,
                                         best_nt.td.noise_active)
                                logπ_arr[t, w] = best_lp
                                logL_arr[t, w] = best_ll
                                Threads.atomic_add!(accept_ntd[t], 1)
                            end
                        end
                        continue
                    end
                end

                # Planet↔activity SWAP (burn-in only, DB-free): trade a planet
                # for the activity model that absorbs its signal, in ONE move.
                # A planet redundant with an activity model can't be killed by
                # the standard death — at β≈1 dropping to Np−1 with nothing to
                # cover the freed signal tanks logL, so the planet entrenches
                # and a true null fakes a planet (measured: clean-indicator toy
                # P(Np=0)=0 despite AD winning on evidence). Here we kill a
                # planet and birth a noise model whose params are FIT to the
                # freed residual (propose_noise_birth's OLS-informed AD sees the
                # planet's signal once it's deactivated). Judged at
                # β·ΔlogL+Δlogπ: Δlogπ removes the planet's param priors, so
                # Occam favors the trade ONLY when the activity model is
                # competitive — a REAL planet's large ΔL keeps it, a fake one
                # (≈AD on logL) trades to AD and the chain reaches Np=0.
                # `td.planets` guard: this kills a planet (Np→Np−1), so it must
                # NOT fire when the planet count is fixed (planets=false, e.g. a
                # noise-only model-selection run) — otherwise it leaks Np=0 states
                # despite the freeze. (The sibling swap at ~line 714 is already
                # td.planets-gated; this one was missed.)
                if td.planets && in_burnin && β > 0.3 && n_birth_tries > 1 &&
                   theta.td.n_planets_active > 0 && rand(trng) < 0.5
                    dead, lq_pd = propose_planet_death(theta, trng)
                    if isfinite(lq_pd)
                        Threads.atomic_add!(pswap_prop, 1)
                        best_α = -Inf; best_nt = nothing
                        best_lp = -Inf; best_ll = -Inf
                        for _ in 1:n_birth_tries
                            cand, lq_nb = propose_noise_birth(dead, trng,
                                              td.toggleable; data=data,
                                              exclusion_groups = td.noise_exclusion_groups)
                            isfinite(lq_nb) || continue
                            lpc = spike_slab_log_prior(cand, strategy, inclusion_prior)
                            isfinite(lpc) || continue
                            lpc, llc = split_ref(lpc, cand, tid)
                            Threads.atomic_add!(n_evals_atomic, 1)
                            isfinite(llc) || continue
                            αc = β * (llc - logL_arr[t, w]) +
                                 (lpc - logπ_arr[t, w]) + lq_nb + lq_pd
                            if αc > best_α
                                best_α = αc; best_nt = cand
                                best_lp = lpc; best_ll = llc
                            end
                        end
                        if best_nt !== nothing
                            Threads.atomic_add!(propose_ntd[t], 1)
                            nb = findfirst(i -> best_nt.td.noise_active[i] &&
                                                !td_states[t, w].noise_active[i],
                                            1:n_noise_slots)
                            if nb !== nothing && n_birth_refine > 0
                                best_lp, best_ll = climb_newborn!(best_nt, nb,
                                                                    best_lp, best_ll)
                                best_α = β * (best_ll - logL_arr[t, w]) +
                                         (best_lp - logπ_arr[t, w])
                            end
                            if log(rand(trng)) < best_α
                                @inbounds for (j, idx) in enumerate(unfrozen_idx)
                                    state[t, w, j] = best_nt.values[idx]
                                end
                                # full td copy — BOTH a planet death and a noise birth
                                copy_into!(td_states[t, w], best_nt.td)
                                logπ_arr[t, w] = best_lp
                                logL_arr[t, w] = best_ll
                                Threads.atomic_add!(accept_ntd[t], 1)
                                Threads.atomic_add!(pswap_acc, 1)
                            end
                        end
                        continue
                    end
                end

                do_nbirth = rand(trng) < 0.5

                # Burn-in multi-try for NOISE births (mirrors the planet
                # block): a single prior draw of a 12-param activity model
                # is never competitive, so propose `n_birth_tries`
                # candidates and keep the best by tempered acceptance.
                # DB-free since burn-in is discarded; the single-draw path
                # below remains the DB-correct sampling kernel.
                if do_nbirth && n_birth_tries > 1 && in_burnin && β > 0.3
                    best_α = -Inf
                    best_nt = nothing
                    best_lp = -Inf; best_ll = -Inf; best_lq = -Inf
                    for _ in 1:n_birth_tries
                        cand, lqc = propose_noise_birth(theta, trng,
                                          td.toggleable; data=data,
                                          exclusion_groups = td.noise_exclusion_groups)
                        isfinite(lqc) || continue
                        lpc = spike_slab_log_prior(cand, strategy,
                                                     inclusion_prior)
                        isfinite(lpc) || continue
                        lpc, llc = split_ref(lpc, cand, tid)
                        Threads.atomic_add!(n_evals_atomic, 1)
                        isfinite(llc) || continue
                        αc = β * (llc - logL_arr[t, w]) +
                             (lpc - logπ_arr[t, w]) + lqc
                        if αc > best_α
                            best_α = αc; best_nt = cand
                            best_lp = lpc; best_ll = llc; best_lq = lqc
                        end
                    end
                    best_nt === nothing && continue
                    Threads.atomic_add!(propose_ntd[t], 1)

                    # Pre-accept CLIMB (burn-in only ⇒ no DB obligation).
                    # Planet multi-try sees competitive candidates because
                    # planet births are data-informed (BLS/LS-seeded);
                    # noise births are BLIND prior draws, and best-of-N in
                    # 12-D is still nowhere near the mode — judged at the
                    # drawn config the accept bar is never reached and
                    # post-accept refinement never runs. So climb the
                    # selected candidate's newborn params with coordinate
                    # M-H FIRST, then decide acceptance on the climbed
                    # config: survival reflects the model's fitted
                    # quality, not the luck of the draw.
                    nb = findfirst(i -> best_nt.td.noise_active[i] &&
                                        !td_states[t, w].noise_active[i],
                                    1:n_noise_slots)
                    if nb !== nothing && n_birth_refine > 0
                        best_lp, best_ll = climb_newborn!(best_nt, nb,
                                                            best_lp, best_ll)
                        best_α = β * (best_ll - logL_arr[t, w]) +
                                 (best_lp - logπ_arr[t, w]) + best_lq
                    end

                    if log(rand(trng)) < best_α
                        @inbounds for (j, idx) in enumerate(unfrozen_idx)
                            state[t, w, j] = best_nt.values[idx]
                        end
                        copyto!(td_states[t, w].noise_active,
                                 best_nt.td.noise_active)
                        logπ_arr[t, w] = best_lp
                        logL_arr[t, w] = best_ll
                        Threads.atomic_add!(accept_ntd[t], 1)
                    end
                    continue
                end

                # ANNEALED birth (post-burn-in too). `climb_newborn!` above
                # relaxes a newborn before judging it — the right idea — but is
                # confined to burn-in because done naively it is not reversible.
                # The bridged version accumulates the relaxation's work into the
                # acceptance ratio, so the same idea is admissible in the
                # sampling phase, where the occupancy is actually measured.
                # n_noise_bridge = 1 recovers the ordinary single-draw birth.
                if do_nbirth && n_noise_bridge > 1
                    _ll(x) = last(split_ref(spike_slab_log_prior(x, strategy,
                                                inclusion_prior), x, tid))
                    _lp(x) = spike_slab_log_prior(x, strategy, inclusion_prior)
                    cand_a, acc_a, ll_a = propose_noise_birth_annealed(
                        theta, trng, td.toggleable, _ll, _lp; data=data,
                        exclusion_groups = td.noise_exclusion_groups,
                        beta = β, n_bridge = n_noise_bridge,
                        n_relax = n_noise_relax,
                        ll_current = logL_arr[t, w],
                        lp_current = logπ_arr[t, w])
                    isfinite(acc_a) || continue
                    Threads.atomic_add!(propose_ntd[t], 1)
                    Threads.atomic_add!(n_evals_atomic, 1)
                    if log(rand(trng)) < acc_a
                        lp_a = _lp(cand_a)
                        @inbounds for (j, idx) in enumerate(unfrozen_idx)
                            state[t, w, j] = cand_a.values[idx]
                        end
                        copyto!(td_states[t, w].noise_active,
                                 cand_a.td.noise_active)
                        logπ_arr[t, w] = lp_a; logL_arr[t, w] = ll_a
                        Threads.atomic_add!(accept_ntd[t], 1)
                    end
                    continue
                end

                new_theta, log_q = do_nbirth ?
                    propose_noise_birth(theta, trng, td.toggleable; data=data,
                                          exclusion_groups = td.noise_exclusion_groups) :
                    propose_noise_death(theta, trng, td.toggleable; data=data)
                isfinite(log_q) || continue
                Threads.atomic_add!(propose_ntd[t], 1)
                # MUST be the same prior function the rest of the sampler
                # stores in logπ_arr (eval_bounded!, planet moves, refinement
                # all use spike_slab_log_prior). The previous log_prior(...)
                # call here dropped the planet-indicator Bernoulli terms, so
                # every noise toggle carried a spurious −max_k·log(p_incl)
                # offset AND an accepted toggle stored the convention-less
                # value, biasing the NEXT within-model move too.
                new_log_pi = spike_slab_log_prior(new_theta, strategy,
                                                    inclusion_prior)
                isfinite(new_log_pi) || continue
                new_log_pi, new_log_L = split_ref(new_log_pi, new_theta, tid)
                Threads.atomic_add!(n_evals_atomic, 1)
                isfinite(new_log_L) || continue

                log_α = β * (new_log_L - logL_arr[t, w]) +
                        (new_log_pi - logπ_arr[t, w]) + log_q
                if log(rand(trng)) < log_α
                    nb = do_nbirth ?
                        findfirst(i -> new_theta.td.noise_active[i] &&
                                       !td_states[t, w].noise_active[i],
                                   1:n_noise_slots) : nothing
                    @inbounds for (j, idx) in enumerate(unfrozen_idx)
                        state[t, w, j] = new_theta.values[idx]
                    end
                    copyto!(td_states[t, w].noise_active,
                             new_theta.td.noise_active)
                    logπ_arr[t, w] = new_log_pi
                    logL_arr[t, w] = new_log_L
                    Threads.atomic_add!(accept_ntd[t], 1)
                    # Refinement is standard within-model M-H — DB-safe
                    # post-burn-in too (same argument as planet births).
                    do_nbirth && n_birth_refine > 0 && β > 0.3 &&
                        nb !== nothing &&
                        refine_noise_birth!(t, w, nb, β, tid, trng)
                end
                continue
            end

            n_active = theta.td.n_planets_active
            p_birth, _ = _birth_death_probs(n_active, max_k)
            do_birth = rand(trng) < p_birth

            # With probability `informed_birth_fraction`, use a DATA-INFORMED
            # birth: JointInformedBirth (BLS photometry peaks + RV Lomb-Scargle,
            # proposing P from the peak, K from depth→radius→mass, Tc from the
            # BLS t0) when photometry is present, self-degrading to RV-only
            # InformedBirth when t_phot is empty (birth_strategies.jl:748).
            # Otherwise the native MoMS Gaussian RW around off_values. Forward
            # and reverse use the SAME strategy family so detailed balance holds
            # (mirrors sample_moms): MoMS births pair with the deterministic
            # MoMS death; informed births pair with the generic uniform death
            # (its reverse density is carried in the informed log_q). The spike-
            # and-slab prior below always uses the MoMS `strategy` — its Bernoulli
            # structure is strategy-independent.
            # Informed births only on the cold side of the ladder: at β ≲ 0.3
            # the tempered gain β·ΔlogL can't overcome the informed proposal's
            # Hastings density penalty (~16 nats), so they are near-always
            # rejected AND each hot junk-state would mint a fresh state-keyed
            # BLS cache (~0.5 s build) — all cost, no benefit. Hot exploration
            # belongs to the prior/MoMS births. The informed/MoMS mixture
            # weight is fixed per temp and shared by the paired birth/death
            # kernels, so detailed balance within each temp is unaffected.
            use_informed = informed_birth_fraction > 0.0 && βs[t] > 0.3 &&
                            rand(trng) < informed_birth_fraction
            chosen_strategy = use_informed ? JointInformedBirth() : strategy
            β = βs[t]

            # (1)+(2) Generous multi-try exploration, BURN-IN ONLY (discarded, so
            # no detailed-balance obligation — this just initializes the ensemble
            # at the high-evidence modes; the post-burn-in chain samples DB-
            # correctly from there). A single birth proposal lands the new planet
            # at a rough config that usually misses its sharp transit mode; here
            # we propose `n_birth_tries` candidates and keep the best by tempered
            # acceptance, so real planets (huge ΔlogL) reliably get birthed and,
            # across the ensemble, the Np+1 modes get populated. Refinement then
            # climbs the winner. Self-contained: the single-try path below is
            # left exactly as-is for deaths and for post-burn-in sampling.
            # Multi-try only where it pays: burn-in (no DB obligation) and the
            # cold side — at β ≲ 0.3 the tempered selection inverts (picks the
            # candidate LEAST likely under the proposal) and burns 12× evals.
            if do_birth && n_birth_tries > 1 && in_burnin && β > 0.3
                best_α = -Inf; best_nt = nothing; best_lp = -Inf
                best_ll = -Inf; best_k = 0
                for _ in 1:n_birth_tries
                    cand, lqc = propose_planet_birth(theta, trng, chosen_strategy; data=data)
                    isfinite(lqc) || continue
                    kc = 0
                    for k in 1:max_k
                        if theta.td.planet_active[k] != cand.td.planet_active[k]
                            kc = k; break
                        end
                    end
                    kc == 0 && continue
                    # the sort may have moved the newborn off the bit-flip slot
                    kc = _newborn_slot(theta, cand, kc)
                    lpc = spike_slab_log_prior(cand, strategy, inclusion_prior)
                    isfinite(lpc) || continue
                    lpc, llc = split_ref(lpc, cand, tid)
                    Threads.atomic_add!(n_evals_atomic, 1)
                    isfinite(llc) || continue
                    if td_debug
                        _td_debug!(true, step, t, planet_P(cand, kc), 4,
                                    llc - logL_arr[t, w])
                        # cold-temp candidate anatomy: rr (code 5) + time
                        # anchor (code 6), keyed by the same P for filtering
                        blkc = theta.params.layout.planet_blocks[kc]
                        if t <= 2 && has_geometry(blkc)
                            _td_debug!(true, step, t, planet_P(cand, kc), 5,
                                        Float64(cand.values[blkc.r]))
                            _td_debug!(true, step, t, planet_P(cand, kc), 6,
                                        Float64(cand.values[blkc.t]))
                        end
                    end
                    αc = β * (llc - logL_arr[t, w]) + (lpc - logπ_arr[t, w]) + lqc
                    if αc > best_α
                        best_α = αc; best_nt = cand; best_lp = lpc
                        best_ll = llc; best_k = kc
                    end
                end
                best_nt === nothing && continue
                Threads.atomic_add!(propose_td[best_k], 1)
                Threads.atomic_add!(propose_pb[t], 1)   # multi-try is birth-only
                td_debug && _td_debug!(true, step, t, planet_P(best_nt, best_k), 3,
                                        best_ll - logL_arr[t, w])
                if log(rand(trng)) < best_α
                    @inbounds for (j, idx) in enumerate(unfrozen_idx)
                        state[t, w, j] = best_nt.values[idx]
                    end
                    copyto!(td_states[t, w].planet_active, best_nt.td.planet_active)
                    td_states[t, w].n_planets_active = best_nt.td.n_planets_active
                    logπ_arr[t, w] = best_lp; logL_arr[t, w] = best_ll
                    Threads.atomic_add!(accept_td[best_k], 1)
                    Threads.atomic_add!(accept_pb[t], 1)
                    td_debug && _td_debug!(true, step, t, planet_P(best_nt, best_k), 1)
                    n_birth_refine > 0 && β > 0.3 &&
                        refine_birth!(t, w, best_k, β, tid, trng)
                end
                continue
            end

            new_theta, log_q = if do_birth
                propose_planet_birth(theta, trng, chosen_strategy; data=data)
            elseif chosen_strategy isa MoMSBirth
                propose_planet_death(theta, trng, strategy)
            else
                propose_planet_death(theta, trng)
            end
            isfinite(log_q) || continue

            # Identify touched planet slot. For births the insertion sort may
            # have moved the newborn's params off the bit-flip slot; deaths
            # keep the bit-diff slot (the killed params lived there).
            k_touched = 0
            for k in 1:max_k
                if theta.td.planet_active[k] != new_theta.td.planet_active[k]
                    k_touched = k
                    break
                end
            end
            k_touched == 0 && continue
            do_birth && (k_touched = _newborn_slot(theta, new_theta, k_touched))

            new_log_pi = spike_slab_log_prior(new_theta, strategy,
                                                inclusion_prior)
            if !isfinite(new_log_pi)
                Threads.atomic_add!(propose_td[k_touched], 1)
                Threads.atomic_add!((do_birth ? propose_pb : propose_pd)[t], 1)
                continue
            end

            new_log_pi, new_log_L = split_ref(new_log_pi, new_theta, tid)
            Threads.atomic_add!(n_evals_atomic, 1)
            Threads.atomic_add!(propose_td[k_touched], 1)
            Threads.atomic_add!((do_birth ? propose_pb : propose_pd)[t], 1)
            isfinite(new_log_L) || continue

            β = βs[t]
            log_α = β * (new_log_L - logL_arr[t, w]) +
                    (new_log_pi - logπ_arr[t, w]) + log_q
            if log(rand(trng)) < log_α
                # Accept — copy new_theta into walker storage
                @inbounds for (j, idx) in enumerate(unfrozen_idx)
                    state[t, w, j] = new_theta.values[idx]
                end
                copyto!(td_states[t, w].planet_active,
                         new_theta.td.planet_active)
                td_states[t, w].n_planets_active = new_theta.td.n_planets_active
                logπ_arr[t, w] = new_log_pi
                logL_arr[t, w] = new_log_L
                Threads.atomic_add!(accept_td[k_touched], 1)
                Threads.atomic_add!((do_birth ? accept_pb : accept_pd)[t], 1)
                td_debug && _td_debug!(true, step, t,
                    do_birth ? planet_P(new_theta, k_touched) :
                               planet_P(theta, k_touched),
                    do_birth ? 1 : 2)

                # (3) Post-birth refinement. A freshly-birthed planet sits at a
                # rough (BLS-seeded) config and CANNOT be refined by the stretch
                # move — its only valid partners would need the same planet
                # active, and at birth there are none. So climb it to its mode
                # here with a few within-model RWM sweeps on JUST the new
                # planet's params. Once at the mode it is self-protecting: a
                # later stretch/death that degrades it fails its own M-H. These
                # are standard within-model moves → detailed balance is safe.
                # Refinement gated to the cold side (β > 0.3): hot junk births
                # don't merit the per-slot eval cost, and hot chains explore
                # via the standard moves anyway.
                do_birth && n_birth_refine > 0 && β > 0.3 &&
                    refine_birth!(t, w, k_touched, β, tid, trng)
            end
        end
    end

    # NP_TRACE: locate where planets=false leaks an Np<max_k state.
    _np_trace = haskey(ENV, "NP_TRACE")
    _np0() = count(s -> s.n_planets_active < max_k, td_states)
    _np_prev = Ref(_np0())
    function _np_chk(label, step)
        _np_trace || return
        c = _np0()
        (c > _np_prev[] && step <= 200) &&
            println("[NP_TRACE] Np<max $(_np_prev[])→$c  step=$step  AFTER $label")
        _np_prev[] = c
    end
    _np_trace && println("[NP_TRACE] td.planets=$(td.planets) max_k=$max_k " *
                          "post-init Np<max=$(_np0())/$(length(td_states))")

    pb = ProgressBar("td-ptemcee"; total = n_steps, enabled = show_progress)

    for step in 1:n_steps
        do_half_step!(tasks_h1, :h1); _np_chk("h1", step)
        do_half_step!(tasks_h2, :h2); _np_chk("h2", step)
        do_transdim_step!(step); _np_chk("transdim", step)

        @inbounds for t in 1:n_temps
            propose_within[t] += Threads.atomic_xchg!(propose_temp[t], 0)
            accept_within[t]  += Threads.atomic_xchg!(accept_temp[t],  0)
        end
        @inbounds for t in 1:n_temps
            propose_noise_transdim[t] += Threads.atomic_xchg!(propose_ntd[t], 0)
            accept_noise_transdim[t]  += Threads.atomic_xchg!(accept_ntd[t],  0)
        end
        @inbounds for t in 1:n_temps
            planet_birth_proposed[t] += Threads.atomic_xchg!(propose_pb[t], 0)
            planet_birth_accepted[t] += Threads.atomic_xchg!(accept_pb[t],  0)
            planet_death_proposed[t] += Threads.atomic_xchg!(propose_pd[t], 0)
            planet_death_accepted[t] += Threads.atomic_xchg!(accept_pd[t],  0)
        end
        @inbounds for k in 1:max_k
            window_props = Threads.atomic_xchg!(propose_td[k], 0)
            window_accs  = Threads.atomic_xchg!(accept_td[k],  0)
            propose_transdim[k] += window_props
            accept_transdim[k]  += window_accs
            if step <= n_burnin
                adapt_n_attempts[k] += window_props
                adapt_n_accepts[k]  += window_accs
            end
        end
        # Apply Robbins-Monro scale adaptation every `adapt_window`
        # burn-in steps. Skip post-burnin to preserve detailed balance.
        if step <= n_burnin && step % adapt_window == 0
            _adapt_moms_scales!(strategy, adapt_n_attempts, adapt_n_accepts,
                                  target_birth_accept, step, n_burnin)
            fill!(adapt_n_attempts, 0)
            fill!(adapt_n_accepts,  0)
        end

        # ---- Temperature swap (serial; same-walker swap is fine since
        # the indicator state is part of the walker we swap) ------------
        window_accs  = zeros(Int, n_temps - 1)
        window_props = zeros(Int, n_temps - 1)
        @inbounds for t in 1:(n_temps - 1)
            Δβ = βs[t] - βs[t + 1]
            for w in 1:n_walkers_eff
                w2 = rand(rng_master, 1:n_walkers_eff)
                ll1 = logL_arr[t, w]
                ll2 = logL_arr[t + 1, w2]
                log_ratio = Δβ * (ll2 - ll1)
                propose_swap[t] += 1
                window_props[t] += 1
                if log(rand(rng_master)) < log_ratio
                    for d in 1:n_dim
                        tmp = state[t, w, d]
                        state[t, w, d]      = state[t + 1, w2, d]
                        state[t + 1, w2, d] = tmp
                    end
                    # Swap the FULL TD state: planet, noise AND astrometric
                    # coupling. Any mask left out here is not merely dropped —
                    # it is left behind at the wrong temperature while the
                    # parameters move, so the two rungs end up describing
                    # different models with each other's numbers.
                    tds_tmp = TransDimState(td_states[t, w].n_planets_active,
                                              copy(td_states[t, w].planet_active),
                                              copy(td_states[t, w].noise_active),
                                              copy(td_states[t, w].as_active))
                    copyto!(td_states[t, w].planet_active,
                             td_states[t + 1, w2].planet_active)
                    td_states[t, w].n_planets_active =
                        td_states[t + 1, w2].n_planets_active
                    copyto!(td_states[t, w].noise_active,
                             td_states[t + 1, w2].noise_active)
                    copyto!(td_states[t + 1, w2].planet_active,
                             tds_tmp.planet_active)
                    td_states[t + 1, w2].n_planets_active =
                        tds_tmp.n_planets_active
                    copyto!(td_states[t + 1, w2].noise_active,
                             tds_tmp.noise_active)
                    copyto!(td_states[t, w].as_active,
                             td_states[t + 1, w2].as_active)
                    copyto!(td_states[t + 1, w2].as_active, tds_tmp.as_active)

                    lp_t                  = logπ_arr[t, w]
                    logπ_arr[t, w]        = logπ_arr[t + 1, w2]
                    logπ_arr[t + 1, w2]   = lp_t
                    logL_arr[t, w]        = ll2
                    logL_arr[t + 1, w2]   = ll1
                    accept_swap[t] += 1
                    window_accs[t] += 1
                end
            end
        end
        _np_chk("ptswap", step)

        # Vousden+ 2016 adaptive ladder (matches sample_ptemcee block).
        if adapt_ladder && step <= n_burnin && step % ladder_adapt_window == 0 &&
           n_temps >= 3
            γt = ladder_adapt_K / ((step / ladder_adapt_window) + ladder_adapt_ν0)
            T = 1.0 ./ βs
            α = [window_props[i] > 0 ? window_accs[i] / window_props[i] : 0.0
                  for i in 1:(n_temps - 1)]
            for i in 2:(n_temps - 1)
                spacing = T[i] - T[i - 1]
                T[i] = T[i - 1] + spacing * exp(γt * (α[i - 1] - α[i]))
            end
            for i in 2:(n_temps - 1)
                T[i] = max(T[i], T[i - 1] + 1e-6)
            end
            new_betas = 1.0 ./ T
            new_betas[1] = βs[1]
            new_betas[end] = βs[end]
            βs .= new_betas
        end

        if step > n_burnin
            @inbounds for w in 1:n_walkers_eff
                logL_walker = view(logL_arr, :, w)
                update_evidence!(evidence_acc, logL_walker, βs)
            end
            if (step - n_burnin) % thin == 0
                @inbounds for w in 1:n_walkers_eff
                    keep_idx += 1
                    for d in 1:n_dim
                        samples[keep_idx, d] = state[1, w, d]
                    end
                    samples[keep_idx, n_dim + 1] = Float64(td_states[1, w].n_planets_active)
                    for k in 1:max_k
                        samples[keep_idx, n_dim + 1 + k] =
                            Float64(td_states[1, w].planet_active[k])
                    end
                    for (j, nm_idx) in enumerate(noise_toggle_idx)
                        samples[keep_idx, n_dim + 1 + max_k + j] =
                            Float64(td_states[1, w].noise_active[nm_idx])
                    end
                    lp_samples[keep_idx] = logπ_arr[1, w] + logL_arr[1, w]
                end
            end
        end

        if show_progress
            total_props = sum(propose_within)
            total_acc   = sum(accept_within)
            acc_rate    = total_props > 0 ? total_acc / total_props : 0.0
            td_props = sum(propose_transdim)
            td_acc   = sum(accept_transdim)
            td_rate  = td_props > 0 ? td_acc / td_props : 0.0
            # RAW accepted counts, not rates: a 3-digit-rounded rate displays
            # 0.000 for anything under 0.05% and a 0/0 noise-only run reads
            # as "frozen" when it is in fact toggling.
            update!(pb; n_done = step,
                    fields = (:acc => round(acc_rate, digits=3),
                              :td_acc => round(td_rate, digits=3),
                              :td_n => td_acc,
                              :ntd_n => sum(accept_noise_transdim),
                              :nevals => n_evals_atomic[]))
        end
    end
    show_progress && finish!(pb)
    @info "planet↔activity swap: proposed=$(pswap_prop[]) accepted=$(pswap_acc[])" *
          "  | PN(pl→AD) $(pn_acc[])/$(pn_prop[])  NP(AD→pl) $(np_acc[])/$(np_prop[])"

    samples    = samples[1:keep_idx, :]
    lp_samples = lp_samples[1:keep_idx]

    # --- Evidence
    ev_report = evidence_report(evidence_acc, βs)
    log_z = isfinite(ev_report.hybrid[1]) ? ev_report.hybrid[1] :
            isfinite(ev_report.ti_plus[1]) ? ev_report.ti_plus[1] :
            ev_report.ti[1]

    # --- Output chains
    noise_active_names = [Symbol("noise_active_$(nm_idx)")
                            for nm_idx in noise_toggle_idx]
    param_names = vcat(Symbol.(layout.unfrozen_names),
                        [:n_planets],
                        [Symbol("planet_active_$k") for k in 1:max_k],
                        noise_active_names,
                        [:lp])
    samples_with_lp = hcat(samples, lp_samples)
    chains = MCMCChains.Chains(samples_with_lp, param_names)

    acc_within = accept_within ./ max.(propose_within, 1)
    acc_swap   = accept_swap   ./ max.(propose_swap,   1)
    acc_td     = accept_transdim ./ max.(propose_transdim, 1)

    if show_progress
        println("trans-dim moves: planet accepted/proposed = ",
                sum(accept_transdim), "/", sum(propose_transdim),
                " | noise toggles accepted/proposed per temp (cold→hot) = ",
                join(string.(accept_noise_transdim, "/",
                              propose_noise_transdim), " "))
        println("temp-swap acceptance per rung (cold→hot): ",
                join(string.(round.(acc_swap, digits=3)), " "))
        # Standard convergence gate — split-R̂ + bulk-ESS over the cold ensemble.
        # `n_walkers_eff` (not the requested NW) is the recorded walker count.
        # A chain is science-grade only if this PASSes; FAIL ⇒ sample more.
        try
            convergence_report(chains, n_walkers_eff; model_params=params)
        catch e
            @warn "convergence_report failed (chain may be too short)" exception=e
        end
    end

    return TransDimPTemceeResult(
        chains, log_z, ev_report,
        acc_within, acc_swap, acc_td,
        βs, n_evals_atomic[],
        copy(propose_transdim), copy(accept_transdim),
        copy(propose_noise_transdim), copy(accept_noise_transdim),
        copy(planet_birth_proposed), copy(planet_birth_accepted),
        copy(planet_death_proposed), copy(planet_death_accepted),
    )
end
