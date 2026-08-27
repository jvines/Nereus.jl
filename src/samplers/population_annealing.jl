# Population Annealing (Hukushima & Iba 2003, J. Phys. Soc. Jpn. 72, 1604).
#
# A sequential-Monte-Carlo flavor of tempered sampling. Instead of running
# parallel chains at different temperatures and swapping between them
# (PT), maintain a single population of N replicas and slowly anneal
# them through the temperature schedule β = 0 → 1. At each annealing
# step:
#   1. Compute importance weights wᵢ = exp((β_new − β_old) · logL_i).
#   2. Accumulate log evidence: log dZ_k = log(⟨w⟩).
#   3. Resample replicas with probability ∝ wᵢ — clones replicas with
#      high likelihood, kills replicas in low-L regions.
#   4. MCMC mutation: each surviving replica takes `n_mcmc` Metropolis
#      steps at β_new to diversify clones.
# At β=1 the surviving replicas are samples from the posterior, and
# log Z = Σ log dZ_k falls out essentially for free.
#
# Why alongside PT/SMC: PA replaces swap dynamics with replica
# resampling, which is embarrassingly parallel (no inter-replica
# coupling within an MCMC mutation step). For multimodal posteriors,
# the resampling step bridges modes via replica cloning where PT
# relies on long random walks. log Z is unbiased to O(1/N) without
# needing the reddemcee TI+/SS+/H+ stack.
#
# This implementation:
# - **Replicas live in BOUNDED space** (same as ptemcee).
# - **Adaptive β schedule** — pick the next β so the effective sample
#   size (ESS) drops to ess_target × N. Avoids the "schedule tuning"
#   problem of static schedules.
# - **MCMC mutation** = simple Gaussian RWM with prior-scale step;
#   independent per replica → fully threaded (Threads.@threads :static).
# - **Systematic resampling** — low-variance, deterministic given the
#   weight vector. Preserves more replica diversity than multinomial.
# - **Per-thread Theta + RNG buffers** with :static scheduling to
#   avoid the threadid-migration bug (see feedback_threading_threadid.md).

using Random
using MCMCChains
using LinearAlgebra: norm, cholesky, Hermitian, I
using Statistics: cov

export sample_pa, PAResult

"""
    PAResult

Container for a `sample_pa` run.

# Fields
- `chains::MCMCChains.Chains` — β=1 replicas (after burn-mutation), bounded space.
- `log_evidence::Float64` — accumulated log Z (unbiased estimator).
- `log_evidence_history::Vector{Float64}` — per-step log dZ contributions.
- `beta_history::Vector{Float64}` — adaptive β schedule that the run produced.
- `ess_history::Vector{Float64}` — ESS / N at each step (target ~ess_target).
- `acceptance::Float64` — mean MCMC mutation acceptance rate.
- `n_evals::Int` — total likelihood evaluations.
"""
struct PAResult
    chains::MCMCChains.Chains
    log_evidence::Float64
    log_evidence_history::Vector{Float64}
    beta_history::Vector{Float64}
    ess_history::Vector{Float64}
    acceptance::Float64
    n_evals::Int
end

"""
    sample_pa(target, data; kwargs...) -> PAResult

Population Annealing sequential MC sampler.

# Keywords
- `n_replicas::Int=500` — population size. Larger N → tighter log Z + posterior.
- `n_mcmc::Int=10` — MINIMUM MCMC mutation sweeps per anneal step. The mutation
  loop is adaptive (see `mcmc_target_moves`): it always runs at least `n_mcmc`
  sweeps and then keeps sweeping until the population has decorrelated, so a
  small `n_mcmc` is no longer the magic constant that decides whether the
  sampler mixes.
- `ess_target::Float64=0.9` — adaptive β picks each step so ESS / N = this. This
  is the SMC *temperature-crawl* target: a HIGH value (0.9) makes β advance in
  many small increments so the population re-equilibrates between resamplings.
  The previous default 0.5 advanced β in ~10–15 giant jumps; the mutation
  kernel could not re-diversify the resampled population fast enough between
  those jumps, so replicas collapsed onto whichever (often wrong, high-e /
  period-alias) mode dominated the early-β weights — the systematic
  eccentricity bias and the 2-planet e→1 railing both trace to this. With a
  conservative crawl the e-marginal stays unbiased and the multi-planet modes
  separate correctly. 0.5–0.95 is the standard adaptive-SMC range (Jasra+ 2011,
  South+ 2019); we default to the conservative end because the gradient-free
  mutation needs the slack.
- `max_steps::Int=200` — hard cap on annealing steps (with `ess_target=0.9` a
  typical RV fit takes 30–70 steps, well under the cap).
- `mcmc_target_moves::Float64=5.0` — adaptive-mutation target: keep running
  mutation sweeps at the current β until each replica has, in expectation,
  accumulated this many ACCEPTED moves (cumulative acceptance-rate × sweeps ≥
  target). Decouples mixing from the raw `n_mcmc` count: a low-acceptance β step
  automatically gets more sweeps, a high-acceptance step fewer. This is what
  stops resampling from out-pacing mutation (the effective-sample-collapse that
  biased e). Set to 0 to disable adaptivity and run exactly `n_mcmc` sweeps.
- `max_mcmc_mult::Int=12` — safety cap on adaptive sweeps: never run more than
  `max_mcmc_mult × n_mcmc` sweeps at one β step, bounding worst-case cost.
- `step_scale::Float64=0.05` — only used by `mutation_kernel=:rwm`. RWM step
  size as fraction of prior range (or 1·σ for normal priors). Adapted between
  anneal steps to target acceptance ~0.3.
- `mutation_kernel::Symbol=:stretch` — `:stretch` / `:rwm` / `:adaptive_cov`.
  - `:stretch` uses Goodman-Weare's affine-
  invariant stretch move with the *other half of the live population as the
  ensemble*. Handles strongly correlated posteriors without per-dim step
  tuning — the recommended kernel and the default. `:rwm` falls back to per-
  dim Gaussian RWM with adaptive step size; works on low-d targets but
  collapses on high-d correlated posteriors (HD 18599 22-d RVPM joint).
- `stretch_a::Float64=2.0` — Goodman-Weare stretch parameter; only used when
  `mutation_kernel=:stretch`.
  - `:adaptive_cov` proposes from a Gaussian shaped like the *empirical
    covariance* of the population. Best for high-d (≥20) correlated
    posteriors where the partner-based stretch can't re-diversify a
    clone cluster: the covariance shrinks proportionally with the
    cluster so proposals stay in-scale. Step size adapts toward ~25%
    acceptance via Robbins-Monro. Recommended for `sample_smc` callers
    and any PA fit with >15 unfrozen parameters. Adds O(d²) per step
    for the Cholesky factor (refit every β step, cheap vs likelihood).
- `seed::Int=1`
- `show_progress::Bool=true` — display the per-step progress bar.
"""
function sample_pa(
    target::NereusTarget,
    data::Data;
    n_replicas::Int = 500,
    n_mcmc::Int = 10,
    ess_target::Real = 0.9,
    max_steps::Int = 200,
    step_scale::Real = 0.05,
    mutation_kernel::Symbol = :stretch,
    stretch_a::Real = 2.0,
    mcmc_target_moves::Real = 5.0,
    max_mcmc_mult::Int = 12,
    seed::Int = 1,
    show_progress::Bool = true,
)
    # JSON config delivers floats written without a decimal point as Int;
    # normalize the float-valued kwargs so downstream Float64 usage holds.
    ess_target = Float64(ess_target)
    step_scale = Float64(step_scale)
    stretch_a  = Float64(stretch_a)
    mcmc_target_moves = Float64(mcmc_target_moves)
    mutation_kernel in (:stretch, :rwm, :adaptive_cov) || throw(ArgumentError(
        "mutation_kernel must be :stretch (default), :rwm, or :adaptive_cov; " *
        "got :$mutation_kernel"))
    n_replicas >= 4 || throw(ArgumentError("n_replicas must be ≥ 4"))
    if mutation_kernel === :stretch && isodd(n_replicas)
        n_replicas += 1  # need even for two-half split
    end
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    unfrozen_idx = layout.unfrozen_idx
    unfrozen_priors = layout.unfrozen_priors

    # Per-thread state: each task uses thread-pinned buffers (:static)
    n_thr = max(Threads.nthreads(), 1)
    thread_theta = [Theta{Float64}(params) for _ in 1:n_thr]
    # Per-thread PTWorkspace → ws-aware likelihood (no per-call GC + caches).
    thread_ws = [PTWorkspace(params, params.config.max_kplanet,
                             length(params.config.noise_models);
                             n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                 for _ in 1:n_thr]
    rng_master   = MersenneTwister(seed)

    # Per-parameter MCMC step size from prior scale (1% of range for
    # uniform, 0.1·σ for normal). Used by the RWM mutation step.
    step_sigma = Vector{Float64}(undef, n_dim)
    @inbounds for j in 1:n_dim
        ps = unfrozen_priors[j]
        if isfinite(ps.lo) && isfinite(ps.hi)
            step_sigma[j] = step_scale * (ps.hi - ps.lo)
        else
            step_sigma[j] = max(step_scale, 1e-3)
        end
    end
    # Adaptive-cov state: empirical-covariance Cholesky factor + scalar
    # global step. Refit every β step from the post-mutation replicas;
    # step adapts via Robbins-Monro toward 25% acceptance.
    chol_L = Matrix{Float64}(I, n_dim, n_dim)   # placeholder identity
    cov_step = 2.38 / sqrt(n_dim)               # Roberts-Rosenthal optimal
    cov_ridge = 1e-6                            # regularize near-singular Σ

    # Replica state in bounded space + cached (log_prior, log_like)
    replicas = Matrix{Float64}(undef, n_replicas, n_dim)
    logπ     = fill(-Inf, n_replicas)
    logL     = fill(-Inf, n_replicas)
    n_evals  = Threads.Atomic{Int}(0)

    # In-place eval at point x (bounded space) using thread-local Theta
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

    # Pre-allocated per-thread proposal scratch — re-used across all
    # three mutation kernels. Each Julia thread owns one (y, ξ) pair;
    # since the per-replica @threads :static loop pins task→thread,
    # there's no race. Saves ~60 k Vector allocations on a typical
    # 200-replica × 10-mcmc × 30-β run.
    thread_y = [Vector{Float64}(undef, n_dim) for _ in 1:n_thr]
    thread_ξ = [Vector{Float64}(undef, n_dim) for _ in 1:n_thr]

    # --- Initialize replicas from the prior (β=0 target) -------------
    init_seeds = rand(rng_master, UInt64, n_replicas)
    Threads.@threads :static for i in 1:n_replicas
        tid = Threads.threadid()
        rng = MersenneTwister(init_seeds[i])
        for _ in 1:200
            x = _draw_from_prior(target, rng)
            @inbounds for d in 1:n_dim
                replicas[i, d] = x[d]
            end
            lp, ll = eval_bounded!(@view(replicas[i, :]), tid)
            if isfinite(lp) && isfinite(ll)
                logπ[i] = lp
                logL[i] = ll
                Threads.atomic_add!(n_evals, 1)
                break
            end
            Threads.atomic_add!(n_evals, 1)
        end
    end

    # --- Annealing loop -----------------------------------------------
    β_cur = 0.0
    log_z_total = 0.0
    log_z_history = Float64[]
    β_history     = Float64[0.0]
    ess_history   = Float64[]
    accept_history = Float64[]

    pb = ProgressBar("PA"; total = max_steps, enabled = show_progress)
    step = 0

    while β_cur < 1.0 && step < max_steps
        step += 1

        # ---- 1. Adaptive β: bisect to ESS/N = ess_target -----------
        # Effective sample size at proposed β_new given current logL:
        #   ESS = (Σ wᵢ)² / Σ wᵢ²
        # where wᵢ = exp((β_new − β_cur) · logL_i). We bisect Δβ ∈
        # [0, 1−β_cur] to land ESS/N at ess_target. Closed-form
        # logsumexp keeps everything numerically stable.
        function ess_at(Δβ::Float64)
            log_w = Δβ .* logL
            m = maximum(log_w)
            sum_w  = sum(exp.(log_w .- m))
            sum_w2 = sum(exp.(2 .* (log_w .- m)))
            return sum_w^2 / sum_w2 / n_replicas
        end
        lo, hi = 0.0, 1.0 - β_cur
        if ess_at(hi) >= ess_target
            Δβ = hi  # full step to β=1 keeps ESS above target
        else
            # Bisect for Δβ where ess_at(Δβ) ≈ ess_target
            for _ in 1:40
                mid = (lo + hi) / 2
                ess_at(mid) >= ess_target ? (lo = mid) : (hi = mid)
            end
            Δβ = lo
        end
        Δβ = max(Δβ, 1e-6)  # ensure forward progress
        β_new = min(β_cur + Δβ, 1.0)

        # ---- 2. Compute weights + log incremental Z ----------------
        log_w = Δβ .* logL
        m_log_w = maximum(log_w)
        sum_w = sum(exp.(log_w .- m_log_w))
        log_dZ = m_log_w + log(sum_w / n_replicas)
        log_z_total += log_dZ
        push!(log_z_history, log_dZ)
        push!(β_history, β_new)

        # Normalized weights (in log space → exp + renormalize)
        log_W = log_w .- m_log_w .- log(sum_w)
        W = exp.(log_W)
        ess_norm = 1.0 / sum(W .^ 2) / n_replicas
        push!(ess_history, ess_norm)

        # ---- 3. Systematic resampling ------------------------------
        # Low-variance resampling: divide [0, 1) into N equal bins,
        # offset by u/N where u ~ U(0, 1), and select replicas whose
        # CDF interval the bin lands in. Preserves more diversity
        # than multinomial.
        cdf = cumsum(W)
        u0 = rand(rng_master) / n_replicas
        indices = Vector{Int}(undef, n_replicas)
        j = 1
        @inbounds for k in 1:n_replicas
            u = u0 + (k - 1) / n_replicas
            while j < n_replicas && cdf[j] < u
                j += 1
            end
            indices[k] = j
        end

        new_replicas = Matrix{Float64}(undef, n_replicas, n_dim)
        new_logL = Vector{Float64}(undef, n_replicas)
        new_logπ = Vector{Float64}(undef, n_replicas)
        @inbounds for k in 1:n_replicas
            src = indices[k]
            for d in 1:n_dim
                new_replicas[k, d] = replicas[src, d]
            end
            new_logL[k] = logL[src]
            new_logπ[k] = logπ[src]
        end
        replicas = new_replicas
        logL     = new_logL
        logπ     = new_logπ

        # ---- 4. MCMC mutation at β_new (ADAPTIVE sweep count) ----
        # The resampling step in (3) duplicates high-weight replicas and
        # kills the rest, so immediately after a resample the population is
        # full of identical clones. Mutation must run long enough to
        # decorrelate those clones BEFORE the next β step resamples again,
        # otherwise the population collapses onto whichever mode dominated
        # the early-β weights — for RV targets that is the high-e /
        # period-alias corner, which is exactly the systematic eccentricity
        # bias (and the 2-planet e→1 railing) this sampler exhibited.
        #
        # A fixed `n_mcmc` cannot guarantee that: a low-acceptance β step
        # needs many more sweeps than a high-acceptance one. So we run an
        # ADAPTIVE number of sweeps — at least `n_mcmc`, then continue until
        # the population has accumulated ~`mcmc_target_moves` ACCEPTED moves
        # per replica (cumulative accepts / N ≥ target), capped at
        # `max_mcmc_mult × n_mcmc`. This is the standard adaptive-SMC
        # mutation criterion (decorrelate before reweighting; South+ 2019).
        #
        # Kernels are factored into single-sweep closures so the adaptive
        # loop is shared. Each sweep increments the shared atomic counters.
        # Per-dim independent Gaussian RWM (`:rwm`) is the 2-D-Gaussian-
        # tested kernel; in high-d correlated posteriors its step-size
        # adaptation under-explores and the population collapses. `:stretch`
        # (default) uses Goodman-Weare's affine-invariant stretch move with
        # the OTHER half of the live population as the ensemble — gradient-
        # free, no covariance to tune, handles correlations automatically.
        accept_count  = Threads.Atomic{Int}(0)
        propose_count = Threads.Atomic{Int}(0)
        # Per-replica RNGs persisted ACROSS sweeps within this β step (so the
        # adaptive extra sweeps draw fresh randomness, not a reset stream).
        mut_seeds = rand(rng_master, UInt64, n_replicas)
        # Per-replica for EVERY kernel. Keying the stretch kernel on the
        # thread instead made the run depend on the thread count at fixed
        # seed; a replica-keyed stream is consumed only by that replica.
        replica_rngs = [MersenneTwister(mut_seeds[i]) for i in 1:n_replicas]

        # Two-half ensemble bookkeeping (stretch only).
        half = n_replicas ÷ 2

        # adaptive_cov: refit Cholesky factor from the current (resampled)
        # population ONCE per β step (ridge-regularized for near-singular Σ).
        if mutation_kernel === :adaptive_cov
            Σ_emp = cov(replicas)               # n_dim × n_dim
            @inbounds for d in 1:n_dim
                Σ_emp[d, d] += cov_ridge        # diagonal ridge
            end
            try
                chol_L = cholesky(Hermitian(Σ_emp)).L
            catch err
                # Σ still indefinite after ridge — fall back to diagonal
                # of std-devs.
                σ_diag = [sqrt(max(Σ_emp[d, d], cov_ridge)) for d in 1:n_dim]
                chol_L = Matrix(LinearAlgebra.Diagonal(σ_diag))
            end
        end

        # One full population sweep for the active kernel.
        function stretch_sweep!()
            # Partners always come from the OTHER half so a single half-step
            # is race-free (active walkers write their own row, read partner
            # rows that aren't being written this half).
            for (active_lo, active_hi, partner_lo, partner_hi) in (
                    (1,        half,        half + 1, n_replicas),
                    (half + 1, n_replicas,  1,        half))
                Threads.@threads :static for i in active_lo:active_hi
                    tid = Threads.threadid()
                    trng = replica_rngs[i]
                    partner = rand(trng, partner_lo:partner_hi)
                    u = rand(trng)
                    z = ((stretch_a - 1) * u + 1)^2 / stretch_a
                    y = thread_y[tid]
                    @inbounds for d in 1:n_dim
                        y[d] = replicas[partner, d] +
                               z * (replicas[i, d] - replicas[partner, d])
                    end
                    lp_y, ll_y = eval_bounded!(y, tid)
                    Threads.atomic_add!(propose_count, 1)
                    Threads.atomic_add!(n_evals, 1)
                    isfinite(lp_y) || continue
                    # Stretch-move M-H ratio includes z^(n-1) Jacobian
                    log_α = (n_dim - 1) * log(z) +
                            β_new * (ll_y - logL[i]) +
                            (lp_y - logπ[i])
                    if log(rand(trng)) < log_α
                        @inbounds for d in 1:n_dim
                            replicas[i, d] = y[d]
                        end
                        logL[i] = ll_y; logπ[i] = lp_y
                        Threads.atomic_add!(accept_count, 1)
                    end
                end
            end
        end

        function adaptive_cov_sweep!()
            Threads.@threads :static for i in 1:n_replicas
                tid = Threads.threadid()
                local_rng = replica_rngs[i]
                x = @view replicas[i, :]
                ξ = thread_ξ[tid]
                y = thread_y[tid]
                @inbounds for d in 1:n_dim
                    ξ[d] = randn(local_rng)
                end
                # y = x + cov_step · L · ξ  (multivariate Gaussian)
                @inbounds for d in 1:n_dim
                    s = 0.0
                    for k in 1:d   # L is lower-triangular
                        s += chol_L[d, k] * ξ[k]
                    end
                    y[d] = x[d] + cov_step * s
                end
                lp_y, ll_y = eval_bounded!(y, tid)
                Threads.atomic_add!(propose_count, 1)
                Threads.atomic_add!(n_evals, 1)
                isfinite(lp_y) || continue
                log_α = β_new * (ll_y - logL[i]) + (lp_y - logπ[i])
                if log(rand(local_rng)) < log_α
                    @inbounds for d in 1:n_dim
                        replicas[i, d] = y[d]
                    end
                    logL[i] = ll_y; logπ[i] = lp_y
                    Threads.atomic_add!(accept_count, 1)
                end
            end
        end

        function rwm_sweep!()
            Threads.@threads :static for i in 1:n_replicas
                tid = Threads.threadid()
                local_rng = replica_rngs[i]
                x = @view replicas[i, :]
                y = thread_y[tid]
                @inbounds for d in 1:n_dim
                    y[d] = x[d] + step_sigma[d] * randn(local_rng)
                end
                lp_y, ll_y = eval_bounded!(y, tid)
                Threads.atomic_add!(propose_count, 1)
                Threads.atomic_add!(n_evals, 1)
                isfinite(lp_y) || continue
                log_α = β_new * (ll_y - logL[i]) + (lp_y - logπ[i])
                if log(rand(local_rng)) < log_α
                    @inbounds for d in 1:n_dim
                        replicas[i, d] = y[d]
                    end
                    logL[i] = ll_y; logπ[i] = lp_y
                    Threads.atomic_add!(accept_count, 1)
                end
            end
        end

        do_sweep! = mutation_kernel === :stretch       ? stretch_sweep!     :
                    mutation_kernel === :adaptive_cov  ? adaptive_cov_sweep! :
                                                         rwm_sweep!

        # Adaptive sweep loop: ≥ n_mcmc sweeps, then continue until each
        # replica has ~mcmc_target_moves accepted moves, capped at
        # max_mcmc_mult × n_mcmc. With mcmc_target_moves ≤ 0 the loop runs
        # exactly n_mcmc sweeps (legacy fixed behaviour).
        sweep_cap = mcmc_target_moves > 0 ? max(n_mcmc, max_mcmc_mult * n_mcmc) : n_mcmc
        n_sweeps  = 0
        while n_sweeps < sweep_cap
            do_sweep!()
            n_sweeps += 1
            n_sweeps >= n_mcmc || continue
            mcmc_target_moves > 0 || break
            # accepted moves per replica so far this β step
            (accept_count[] / n_replicas) >= mcmc_target_moves && break
        end

        acc_rate = accept_count[] / max(propose_count[], 1)
        push!(accept_history, acc_rate)

        # Adapt RWM step size only (stretch doesn't need it — z is drawn
        # fresh each step from a fixed g(z) ∝ 1/√z on [1/a, a]).
        if mutation_kernel === :rwm
            if acc_rate > 0.45
                step_sigma .*= 1.15
            elseif acc_rate < 0.15
                step_sigma .*= 0.85
            end
        elseif mutation_kernel === :adaptive_cov
            # Robbins-Monro step adaptation toward 25% acceptance.
            # `cov_step` scales the Cholesky-factor proposal so we
            # adjust the scalar here; chol_L is refit from the
            # population covariance each β step anyway.
            cov_step *= clamp(exp(0.5 * (acc_rate - 0.25)), 0.7, 1.5)
        end

        β_cur = β_new

        show_progress && update!(pb; n_done = step,
                                  fields = (:β => round(β_cur, digits=3),
                                            :acc => round(acc_rate, digits=3),
                                            :ess => round(ess_norm, digits=3),
                                            :logZ => round(log_z_total, digits=2)))
    end
    show_progress && finish!(pb)

    # --- Build output ------------------------------------------------
    param_names = Symbol.(layout.unfrozen_names)
    chains = MCMCChains.Chains(replicas, param_names)
    mean_acc = isempty(accept_history) ? 0.0 : sum(accept_history) / length(accept_history)
    return PAResult(
        chains, log_z_total, log_z_history, β_history, ess_history,
        mean_acc, n_evals[]
    )
end
