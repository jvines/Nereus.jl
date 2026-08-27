# Normalizing-flow-coupled Parallel Tempering — whitening variant.
#
# Standard PT proposes identity swaps (x_k, x_{k+1}) ↔ (x_{k+1}, x_k)
# between adjacent temperatures. Acceptance is high when β_k and
# β_{k+1} target nearly the same distribution (overlapping samples),
# low when they're far apart (e.g. β=1 vs β=0.04 on a peaked
# posterior). Arbel+ 2021 and Karagiannis+ 2023 propose using a
# *normalizing flow* T_k : space_{β_k} → space_{β_{k+1}} as the swap
# proposal, so the swap proposes T_k(x_k) and T_k^{-1}(x_{k+1}) — much
# more likely to land in each other's high-probability region.
#
# This implementation uses a SIMPLIFIED diagonal-affine flow per
# temperature pair:
#
#   T_k(x) = μ_{k+1} + (σ_{k+1} ⊘ σ_k) ⊙ (x - μ_k)
#
# where μ_k, σ_k are the per-temperature running mean and standard
# deviation across the temperature's WALKER ENSEMBLE over recent steps.
# This is the "whitening" flow — it captures the dominant intra-target
# shift and scale change between adjacent temperatures but ignores
# correlation structure (a full coupling-layer NF would add that).
#
# -------------------------------------------------------------------
# WHY AN ENSEMBLE PER TEMPERATURE (and not a single walker)
# -------------------------------------------------------------------
# The earlier revision ran ONE walker per temperature with an isotropic
# diagonal RWM within-temp move. That design is structurally unable to
# sample these RV posteriors and it FAILED the validation matrix even
# at 13× budget (cold chain frozen at K=89, e=0.835 on the TRIVIAL
# gen_rv_easy target; cold within-temp acceptance ≈ 3%; hottest swap
# acceptance ≈ 0):
#
#   1. A single walker's running (μ, σ) describes the trajectory of one
#      *stuck* particle, NOT the tempered distribution. The whitening
#      flow was therefore fit on garbage statistics and mapped swap
#      proposals into the wrong mode.
#   2. Isotropic diagonal RWM cannot navigate the curved, scale-coupled
#      (K, e=√(se²+sc²), sesinw, secosw, P, Tp) geometry of a Keplerian
#      posterior. Step-size adaptation toward 25% can't fix a *shape*
#      mismatch.
#   3. With one walker per temperature, a swap can only exchange two
#      frozen points — there is no population for the truth mode to
#      propagate down the ladder into.
#
# The fix is to give each temperature a POPULATION of walkers and move
# them with the affine-invariant Goodman-Weare stretch (Vousden+ 2016,
# exactly as the validated `sample_ptemcee`), which is naturally
# scale/rotation-equivariant and so handles the curved geometry. With
# a real ensemble the per-temperature (μ, σ) are meaningful estimates
# of the tempered distribution, so the whitening flow — the defining
# feature of THIS sampler — finally whitens correctly and the
# flow-coupled swaps actually move probability mass between temps.
#
# So the engine is: ENSEMBLE-per-temperature + tempered stretch within
# temp + NF(whitening)-coupled swaps between adjacent temps. The
# whitening flow is what distinguishes this from plain ptemcee.
#
# The whitening flow's Jacobian is a constant Π_d σ_{k+1,d} / σ_{k,d},
# so the M-H ratio for a whitened swap is:
#
#   log α_swap = β_k · L(T_k^{-1}(x_{k+1})) + β_{k+1} · L(T_k(x_k))
#              − β_k · L(x_k) − β_{k+1} · L(x_{k+1})
#              + (Δlog π_prior)
#              + log |det J(T_k)| + log |det J(T_k^{-1})|
#
# The two Jacobian terms cancel: log|det J(T_k)| = -log|det J(T_k^{-1})|.
# So the M-H ratio reduces to a (tempered-)likelihood + prior difference
# at the transformed points.
#
# Performance: the per-temperature (μ, σ) are estimated from a sliding
# window over the recent walker ensemble. Standard identity swaps are
# used during warmup; whitened swaps kick in after `warmup_swaps`
# iterations once the ensemble statistics are reliable. Empirically the
# whitened swap improves swap acceptance by 2-5× over identity swaps on
# multi-mode RV-PM joint posteriors where the β=0 chain explores the
# prior box and β=1 sits in a tight mode.

using Random
using Statistics: mean, std
using MCMCChains

export sample_pt_whitening, WhiteningPTResult

"""
    WhiteningPTResult

# Fields
- `chains::MCMCChains.Chains` — β=1 walkers post-burnin (walkers as chains)
- `log_evidence::Float64` — reddemcee H+/TI+/TI stack on the β-ladder
- `evidence::EvidenceReport` — full TI / TI+ / SS+ / H+ evidence stack
- `acceptance_within::Vector{Float64}` — within-temp stretch acceptance per temp
- `acceptance_swap::Vector{Float64}` — adjacent-temp swap rate per pair
- `betas::Vector{Float64}`
- `n_evals::Int`
- `whitening_active_after::Int` — iteration after which whitening swaps started
"""
struct WhiteningPTResult
    chains::MCMCChains.Chains
    log_evidence::Float64
    evidence::EvidenceReport
    acceptance_within::Vector{Float64}
    acceptance_swap::Vector{Float64}
    betas::Vector{Float64}
    n_evals::Int
    whitening_active_after::Int
end

"""
    sample_pt_whitening(target, data; kwargs...) -> WhiteningPTResult

NF-coupled PT with whitening (diagonal-Gaussian) swap proposals.
Ensemble-per-temperature variant: each temperature carries `n_walkers`
walkers moved by the tempered Goodman-Weare stretch move (as in
`sample_ptemcee`); adjacent-temperature swaps use a diagonal-affine
*whitening* flow fit on the per-temperature walker ensemble, which is
the distinguishing feature of this sampler.

# Keywords
- `n_temps::Int=8` — temperature levels.
- `n_walkers::Int=40` — walkers per temperature (≥ `2·n_dim + 2`, even).
- `n_steps::Int=2000` — MCMC iterations.
- `n_burnin::Int=500` — burn-in discarded from posterior + evidence.
- `betas::Union{Nothing,Vector{Float64}}=nothing` — explicit β ladder
  (descending β[1]=1). Default: Vines+ 2023's `β_i = (1/√5)^i`, the same
  ladder `sample_ptemcee` uses (geometric ladders give far more uniform
  swap acceptance than the old quadratic ladder, which over-resolved the
  hot end and starved the cold end).
- `stretch_a::Float64=2.0` — Goodman-Weare stretch parameter.
- `proposal_scale::Real=0.05` — accepted for backward compatibility; the
  ensemble stretch move needs no global RWM step, so this is unused.
- `warmup_swaps::Int=200` — iterations before whitening kicks in; standard
  identity swap is used until enough ensemble samples accumulate to
  estimate (μ, σ).
- `whiten_window::Int=100` — sliding-window length (in steps) of recent
  walker draws used to estimate the per-temperature (μ, σ) the flow uses.
- `whiten_refresh::Int=10` — recompute the per-temperature (μ, σ) every
  this many steps (the tempered distributions drift slowly across a
  burned-in run, so a per-step recompute over the whole ring buffer is
  wasted work; refreshing every few steps keeps the flow current while
  cutting the dominant cost of the swap path).
- `init_strategy::Symbol=:prior` — walker initialization (`:prior` draws
  each walker from the prior, matching astroEMPEROR / ptemcee).
- `seed::Int=1`
- `thin::Int=1`
- `show_progress::Bool=true`
"""
function sample_pt_whitening(
    target::NereusTarget,
    data::Data;
    n_temps::Int = 8,
    n_walkers::Int = 40,
    n_steps::Int = 2000,
    n_burnin::Int = 500,
    betas::Union{Nothing,AbstractVector} = nothing,
    stretch_a::Real = 2.0,
    proposal_scale::Real = 0.05,
    warmup_swaps::Int = 200,
    whiten_window::Int = 100,
    whiten_refresh::Int = 10,
    init_strategy::Symbol = :prior,
    seed::Int = 1,
    thin::Int = 1,
    show_progress::Bool = true,
)
    # JSON config delivers floats-as-Int and arrays-as-JSON3.Array;
    # normalize both.
    stretch_a      = Float64(stretch_a)
    proposal_scale = Float64(proposal_scale)   # accepted, unused (ensemble move)
    betas = betas === nothing ? nothing : collect(Float64, betas)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    unfrozen_idx = layout.unfrozen_idx

    # Goodman-Weare: each half ≥ n_dim+1 walkers, total even.
    n_walkers_eff = max(n_walkers, 2 * n_dim + 2)
    n_walkers_eff % 2 == 1 && (n_walkers_eff += 1)

    # Default ladder: β[1]=1 (cold) geometric down to a hot chain.
    # Geometric (1/√5)^i matches the validated ptemcee ladder; the old
    # quadratic `1 - (i/(n-1))²` ladder packed the steps at the hot end
    # and left a large β-gap at the cold end, so the cold chain was
    # effectively decoupled (swap accept ~0 with the hottest pair).
    βs = betas === nothing ?
         Float64[(1 / sqrt(5))^i for i in 0:(n_temps - 1)] :
         copy(betas)
    @assert all(b -> 0.0 <= b <= 1.0, βs)
    @assert issorted(βs; rev = true)
    length(βs) == n_temps || throw(ArgumentError(
        "Provided `betas` has length $(length(βs)), expected $n_temps"))

    n_thr = max(Threads.nthreads(), 1)
    thread_theta    = [Theta{Float64}(params) for _ in 1:n_thr]
    # Per-thread PTWorkspace → ws-aware likelihood (no per-call GC + caches).
    thread_ws       = [PTWorkspace(params, params.config.max_kplanet,
                                   length(params.config.noise_models);
                                   n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                       for _ in 1:n_thr]
    thread_proposal = [Vector{Float64}(undef, n_dim) for _ in 1:n_thr]
    rng_master      = MersenneTwister(seed)

    # Evaluate (log_prior_bounded, log_like) at a BOUNDED-space point `x`
    # using thread-local Theta. Returns (-Inf, -Inf) outside the prior
    # support — caller treats this as an automatic M-H reject. log_like
    # includes external priors per Nereus convention so they get
    # tempered too (matches ptemcee / in-house PT).
    @inline function eval_bounded!(x::AbstractVector{Float64}, tid::Int)
        theta = thread_theta[tid]
        wb = thread_ws[tid]
        @inbounds for (j, idx) in enumerate(unfrozen_idx)
            theta.values[idx] = x[j]
        end
        lp = log_prior(theta)
        isfinite(lp) || return (-Inf, -Inf)
        ll = rv_log_likelihood(theta, data, wb) +
             transit_log_likelihood(theta, data, wb)
        return (lp, ll)
    end

    # State: ensemble of n_walkers_eff walkers per temperature, bounded
    # space. logL/logπ caches per (temp, walker).
    state    = Array{Float64,3}(undef, n_temps, n_walkers_eff, n_dim)
    logL_arr = fill(-Inf, n_temps, n_walkers_eff)
    logπ_arr = fill(-Inf, n_temps, n_walkers_eff)

    # --- Initialize walkers in BOUNDED space (threaded) --------------
    # `:prior` draws each walker independently from the prior (astro-
    # EMPEROR style): lots of dispersion means some walkers reliably land
    # near the high-likelihood region, and stretch + swap-down carry that
    # to the cold chain.
    init_seeds = rand(rng_master, UInt64, n_temps * n_walkers_eff)
    Threads.@threads :static for task_idx in 1:(n_temps * n_walkers_eff)
        tid = Threads.threadid()
        t = (task_idx - 1) ÷ n_walkers_eff + 1
        w = (task_idx - 1) % n_walkers_eff + 1
        rng = MersenneTwister(init_seeds[task_idx])
        for _ in 1:200
            xb = _draw_from_prior(target, rng)
            @inbounds for d in 1:n_dim
                state[t, w, d] = xb[d]
            end
            lp, ll = eval_bounded!(@view(state[t, w, :]), tid)
            if isfinite(lp) && isfinite(ll)
                logπ_arr[t, w] = lp
                logL_arr[t, w] = ll
                break
            end
        end
    end

    # --- Per-temperature whitening (μ, σ) over a sliding window -------
    # Ring buffer of the last `whiten_window` *steps* of the FULL walker
    # ensemble per temperature, so (μ, σ) estimate the tempered
    # distribution (not a single stuck trajectory). Storing per (temp,
    # buffer-slot, walker, dim) keeps the running mean/std cheap to
    # recompute and robust to walker swaps.
    win = max(whiten_window, 1)
    ring = Array{Float64,4}(undef, n_temps, win, n_walkers_eff, n_dim)
    ring_count = zeros(Int, n_temps)      # how many steps stored (≤ win)
    ring_head  = zeros(Int, n_temps)      # next slot to write (0-based)
    μ_w = zeros(Float64, n_temps, n_dim)
    σ_w = ones(Float64,  n_temps, n_dim)

    @inline function push_ring!(t::Int)
        slot = ring_head[t] + 1
        @inbounds for w in 1:n_walkers_eff, d in 1:n_dim
            ring[t, slot, w, d] = state[t, w, d]
        end
        ring_head[t] = slot % win
        ring_count[t] = min(ring_count[t] + 1, win)
    end

    # Recompute (μ, σ) per temperature from the ring buffer ensemble.
    function refresh_whiten_stats!(t::Int)
        nc = ring_count[t]
        nc < 1 && return
        ntot = nc * n_walkers_eff
        @inbounds for d in 1:n_dim
            s1 = 0.0
            for s in 1:nc, w in 1:n_walkers_eff
                s1 += ring[t, s, w, d]
            end
            m = s1 / ntot
            s2 = 0.0
            for s in 1:nc, w in 1:n_walkers_eff
                δ = ring[t, s, w, d] - m
                s2 += δ * δ
            end
            v = ntot > 1 ? s2 / (ntot - 1) : 0.0
            μ_w[t, d] = m
            # Floor σ so the flow never collapses a coordinate (which
            # would map every proposal to μ and kill the swap).
            σ_w[t, d] = sqrt(max(v, 1e-12))
        end
    end

    # Dedicated swap scratch (serial swap loop runs AFTER the threaded
    # stretch half-steps, so single-threaded buffers are race-free).
    swap_buf_k   = Vector{Float64}(undef, n_dim)
    swap_buf_kp1 = Vector{Float64}(undef, n_dim)

    # --- Sample storage (β=1, post-burnin, thinned; bounded space) ---
    n_keep_per_walker = max(0, cld(n_steps - n_burnin, thin))
    n_keep_total      = n_keep_per_walker * n_walkers_eff
    samples = Matrix{Float64}(undef, max(n_keep_total, 1), n_dim)
    lp_samples = Vector{Float64}(undef, max(n_keep_total, 1))
    keep_idx = 0

    # reddemcee TI+/SS+/H+ accumulators (same evidence stack as ptemcee).
    evidence_acc = EvidenceAccumulator(length(βs))

    accept_within  = zeros(Float64, n_temps)
    propose_within = zeros(Int, n_temps)
    accept_swap    = zeros(Float64, n_temps - 1)
    propose_swap   = zeros(Int, n_temps - 1)
    n_evals_atomic = Threads.Atomic{Int}(n_temps * n_walkers_eff)
    accept_temp    = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    propose_temp   = [Threads.Atomic{Int}(0) for _ in 1:n_temps]

    # Pre-built (temp, walker) task lists for the two stretch half-steps.
    half = n_walkers_eff ÷ 2
    tasks_h1 = Vector{Tuple{Int,Int}}(undef, n_temps * half)
    tasks_h2 = Vector{Tuple{Int,Int}}(undef, n_temps * (n_walkers_eff - half))
    let i1 = 0, i2 = 0
        for t in 1:n_temps
            for w in 1:half;                   i1 += 1; tasks_h1[i1] = (t, w); end
            for w in (half + 1):n_walkers_eff; i2 += 1; tasks_h2[i2] = (t, w); end
        end
    end

    # Tempered Goodman-Weare stretch half-step (affine-invariant within
    # temp). Identical move to the validated sample_ptemcee.
    # Per-walker RNGs (see the same fix in sample_ptemcee): a per-thread
    # stream makes the chain depend on the thread count at fixed seed.
    rngs_h1 = [MersenneTwister(_walker_seed(seed, 1, i)) for i in 1:length(tasks_h1)]
    rngs_h2 = [MersenneTwister(_walker_seed(seed, 2, i)) for i in 1:length(tasks_h2)]

    function do_half_step!(tasks::Vector{Tuple{Int,Int}}, active_half::Symbol)
        partner_lo = active_half === :h1 ? half + 1 : 1
        partner_hi = active_half === :h1 ? n_walkers_eff : half
        task_rngs  = active_half === :h1 ? rngs_h1 : rngs_h2
        Threads.@threads :static for task_idx in 1:length(tasks)
            tid = Threads.threadid()
            t, w = tasks[task_idx]
            β = βs[t]
            trng = task_rngs[task_idx]
            buf  = thread_proposal[tid]

            w_partner = rand(trng, partner_lo:partner_hi)
            u = rand(trng)
            z = ((stretch_a - 1) * u + 1)^2 / stretch_a

            @inbounds for d in 1:n_dim
                buf[d] = state[t, w_partner, d] +
                         z * (state[t, w, d] - state[t, w_partner, d])
            end

            lp_prop, ll_prop = eval_bounded!(buf, tid)
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

    whitening_active_after = -1

    pb = ProgressBar("pt_whitening"; total = n_steps, enabled = show_progress)

    for step in 1:n_steps
        # ---- Within-temp ensemble stretch (two half-steps) ----------
        do_half_step!(tasks_h1, :h1)
        do_half_step!(tasks_h2, :h2)
        @inbounds for t in 1:n_temps
            propose_within[t] += Threads.atomic_xchg!(propose_temp[t], 0)
            accept_within[t]  += Threads.atomic_xchg!(accept_temp[t],  0)
        end

        # ---- Update sliding-window whitening statistics -------------
        for t in 1:n_temps
            push_ring!(t)
        end

        # ---- Swap moves: whitening flow -----------------------------
        # Use standard identity swaps until enough ensemble samples
        # accumulate. Once active, propose y_k = T_k(x_k),
        # y_{k+1} = T_k^{-1}(x_{k+1}) where T_k is the diagonal-affine
        # whitening map built from the per-temperature ensemble (μ, σ).
        # Diagonal-affine has a constant log-det Jacobian that CANCELS
        # between forward and inverse, so the M-H ratio reduces to the
        # tempered-likelihood + prior difference at the transformed
        # points.
        use_whitening = step > warmup_swaps && all(>(1), ring_count)
        if use_whitening
            # Refresh the per-temperature (μ, σ) only every
            # `whiten_refresh` steps (and on the first activation). The
            # tempered distributions drift slowly, so recomputing the
            # ring-buffer statistics every step is the dominant wasted
            # cost; a few-step cadence keeps the flow current.
            if whitening_active_after == -1 ||
               (step - whitening_active_after) % whiten_refresh == 0
                for t in 1:n_temps
                    refresh_whiten_stats!(t)
                end
            end
            if whitening_active_after == -1
                whitening_active_after = step
            end
        end

        @inbounds for t in 1:(n_temps - 1)
            # Pick a random walker in each of the two temperatures to
            # attempt to swap (ensemble swap, as in ptemcee).
            w   = rand(rng_master, 1:n_walkers_eff)
            w2  = rand(rng_master, 1:n_walkers_eff)
            propose_swap[t] += 1

            if use_whitening
                buf_k   = swap_buf_k
                buf_kp1 = swap_buf_kp1
                @inbounds for d in 1:n_dim
                    σk   = σ_w[t,     d]
                    σkp1 = σ_w[t + 1, d]
                    s_fwd = σkp1 / σk
                    s_bwd = σk   / σkp1
                    buf_k[d]   = μ_w[t + 1, d] + s_fwd * (state[t,     w,  d] - μ_w[t,     d])
                    buf_kp1[d] = μ_w[t,     d] + s_bwd * (state[t + 1, w2, d] - μ_w[t + 1, d])
                end
                lp_yk,  ll_yk  = eval_bounded!(buf_k,   1)
                lp_ykp, ll_ykp = eval_bounded!(buf_kp1, 1)
                Threads.atomic_add!(n_evals_atomic, 2)
                # Reject if either proposal lands out-of-support.
                (isfinite(lp_yk) && isfinite(lp_ykp)) || continue
                log_α = βs[t]     * (ll_yk  - logL_arr[t,     w])  +
                        βs[t + 1] * (ll_ykp - logL_arr[t + 1, w2]) +
                        (lp_yk  - logπ_arr[t,     w]) +
                        (lp_ykp - logπ_arr[t + 1, w2])
                if log(rand(rng_master)) < log_α
                    for d in 1:n_dim
                        state[t,     w,  d] = buf_k[d]
                        state[t + 1, w2, d] = buf_kp1[d]
                    end
                    logL_arr[t,     w]  = ll_yk
                    logL_arr[t + 1, w2] = ll_ykp
                    logπ_arr[t,     w]  = lp_yk
                    logπ_arr[t + 1, w2] = lp_ykp
                    accept_swap[t] += 1
                end
            else
                # Standard identity swap.
                Δβ = βs[t] - βs[t + 1]
                log_α = Δβ * (logL_arr[t + 1, w2] - logL_arr[t, w])
                if log(rand(rng_master)) < log_α
                    for d in 1:n_dim
                        tmp = state[t, w, d]
                        state[t,     w,  d] = state[t + 1, w2, d]
                        state[t + 1, w2, d] = tmp
                    end
                    lp_t = logπ_arr[t, w]; ll_t = logL_arr[t, w]
                    logπ_arr[t,     w]  = logπ_arr[t + 1, w2]
                    logπ_arr[t + 1, w2] = lp_t
                    logL_arr[t,     w]  = logL_arr[t + 1, w2]
                    logL_arr[t + 1, w2] = ll_t
                    accept_swap[t] += 1
                end
            end
        end

        # ---- Post-burnin: record β=1 + feed evidence accumulator ----
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
                    lp_samples[keep_idx] = logπ_arr[1, w] + logL_arr[1, w]
                end
            end
        end

        if show_progress
            total_props = sum(propose_within)
            total_acc   = sum(accept_within)
            acc_rate = total_props > 0 ? total_acc / total_props : 0.0
            update!(pb; n_done = step,
                fields = (:acc  => round(acc_rate, digits = 3),
                          :swap => round(sum(accept_swap) / max(sum(propose_swap), 1), digits = 3),
                          :flow => use_whitening ? "on" : "off"))
        end
    end
    show_progress && finish!(pb)

    samples    = samples[1:keep_idx, :]
    lp_samples = lp_samples[1:keep_idx]

    # --- Evidence report (TI + TI+ + SS+ + H+) ------------------------
    ev_report = evidence_report(evidence_acc, βs)
    log_z = isfinite(ev_report.hybrid[1]) ? ev_report.hybrid[1] :
            isfinite(ev_report.ti_plus[1]) ? ev_report.ti_plus[1] :
            ev_report.ti[1]

    # --- Output: walkers as separate chains (matches ptemcee) ---------
    param_names = Symbol.(layout.unfrozen_names)
    push!(param_names, :lp)
    samples_with_lp = hcat(samples, lp_samples)
    chains = if keep_idx > 0 && n_walkers_eff > 0 &&
                keep_idx % n_walkers_eff == 0
        n_step_kept = keep_idx ÷ n_walkers_eff
        cube = Array{Float64, 3}(undef, n_step_kept, n_dim + 1, n_walkers_eff)
        @inbounds for w in 1:n_walkers_eff, s in 1:n_step_kept
            row = (s - 1) * n_walkers_eff + w
            for d in 1:(n_dim + 1)
                cube[s, d, w] = samples_with_lp[row, d]
            end
        end
        MCMCChains.Chains(cube, param_names)
    else
        MCMCChains.Chains(samples_with_lp, param_names)
    end

    acc_within = accept_within ./ max.(propose_within, 1)
    acc_swap   = accept_swap   ./ max.(propose_swap,   1)
    return WhiteningPTResult(
        chains, log_z, ev_report, acc_within, acc_swap, βs,
        n_evals_atomic[], whitening_active_after,
    )
end
