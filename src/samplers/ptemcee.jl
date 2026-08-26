# Parallel-tempered affine-invariant ensemble MCMC (ptemcee).
#
# Vousden, Farr & Mandel 2016 (arXiv:1501.05823): tempered Goodman-Weare
# ensemble sampler. `n_temps` temperatures × `n_walkers` walkers per
# temperature; stretch moves within each temp + adjacent-temp swaps.
# Successor to emcee 2.x's `PTSampler` (removed in emcee 3.x because the
# affine-invariance argument fails under tempering); the algorithm
# Vines+ 2023 used for HD 18599.
#
# Why alongside `sample_pt`: single-walker Pigeons PT is mathematically
# clean but brittle on multimodal posteriors. The ensemble-per-temp
# structure lets ptemcee exit local modes single-walker cold chains get
# stuck in — targeted fix for cases like HD 18599's
# Keplerian/activity/eccentricity degeneracy.
#
# This implementation:
# - **Walkers live in BOUNDED parameter space** — exactly matching
#   emcee 2.x's PTSampler. Proposals outside the prior support
#   (`log_prior(theta) = -Inf`) get auto-rejected by the M-H ratio,
#   no Jacobian, no transform. An earlier version of this file used
#   unconstrained space, which introduced a Jacobian gradient that
#   acted as a soft wall near prior edges; the stretch move overshot
#   into the smooth-but-penalized tail, killing acceptance. emcee
#   does it the simple way.
# - **Flat parallelism** across (temperature, half-ensemble walker)
#   pairs in each half-step — n_temps × n_walkers/2 independent tasks.
# - **Per-thread Theta, RNG, and proposal buffers** — race-free, no
#   per-step allocation.
# - **reddemcee evidence stack**: feeds EvidenceAccumulator for
#   TI+ / SS+ / H+ in addition to vanilla trapezoidal TI.
# - **Progress bar**: Nereus's `ProgressBar` with per-step accept rate
#   + ETA.
#
# Caveat: affine-invariance of the stretch move is broken under
# tempering. Cite Vousden+ 2016 (ptemcee), not emcee.

using Random
using MCMCChains
using Statistics: cov, quantile
using LinearAlgebra: cholesky, logdet, Symmetric, norm, I

export sample_ptemcee, PTemceeResult, mode_laplace_evidence

"""
    PTemceeResult

Container for a `sample_ptemcee` run.

# Fields
- `chains::MCMCChains.Chains` — β=1 walkers, post-burnin, flattened, in bounded space.
- `log_evidence::Float64` — evidence to USE: the TI/SS/H+ best when it agrees with
  the mode-Laplace cross-check, otherwise `log_evidence_laplace` (the tempered
  estimators fail catastrophically on phase-transition / signal-locked posteriors;
  mode-Laplace does not). See [`mode_laplace_evidence`](@ref).
- `log_evidence_laplace::Float64` — mode-anchored Laplace evidence from the cold
  chain; phase-transition-immune. NaN if not computable. **Reported, not
  headlined**: measured on 51 Peg (unimodal, tightly determined — the friendliest
  case it will ever see) it sat 24 nats from bridge and reference-path, which
  agree with each other to 4 and are validated against exact truth on a curved
  15-D target. It also assumes local Gaussianity about the mode, which is exactly
  what fails on the posteriors that break the tempered path.
- `log_evidence_bridge::Float64` — bridge-sampling evidence from the same cold
  chain (Meng & Wong optimal bridge). This is the HEADLINE when it is usable:
  it never touches the prior end, makes no Gaussianity assumption, and costs only
  `bridge_n` likelihood evaluations. NaN if not computable.
- `evidence::EvidenceReport` — full TI / TI+ / SS+ / H+ tempered stack (raw).
- `acceptance_within::Vector{Float64}` — within-temp stretch acceptance per temp.
- `acceptance_swap::Vector{Float64}` — swap acceptance between (k, k+1). A tiny
  `minimum(acceptance_swap)` (≪0.1) that won't rise with more temps signals a
  phase transition → the tempered evidence is untrustworthy (use the Laplace one).
- `betas::Vector{Float64}` — β ladder used.
- `n_evals::Int` — total likelihood evaluations.
"""
struct PTemceeResult
    chains::MCMCChains.Chains
    log_evidence::Float64
    log_evidence_laplace::Float64
    log_evidence_bridge::Float64
    evidence::EvidenceReport
    acceptance_within::Vector{Float64}
    acceptance_swap::Vector{Float64}
    betas::Vector{Float64}
    n_evals::Int
end

"""
    mode_laplace_evidence(target, chains; mode_frac=0.3) -> Float64

Laplace log-evidence anchored on the posterior MODE, computed from a PT run's own
cold chain — a fast estimator that is **immune to the phase-transition bias** that
makes tempered estimators (TI/SS/H+) fail on high-SNR / signal-locked posteriors
(it never tempers), and **robust to railed MAPs** (uses the sample covariance in
unconstrained space, not the Hessian, which goes singular at a rail).

`logZ ≈ logq(mode) + d/2·log2π + ½·logdet(Σ_local)`, where the mode is the
max-`lp` cold sample (NOT the mean — a multimodal mean sits in a density valley),
and `Σ_local` is the covariance of the closest `mode_frac` of samples to the mode
in unconstrained space (excludes multimodal spread). Carries a ~O(100)-nat
non-Gaussian bias on pathological posteriors — negligible for strong detections,
cross-check nested for marginal ones. Returns `NaN` if not computable (bounded
target, missing `:lp`, or too few samples).
"""
function mode_laplace_evidence(target::NereusTarget, chains::MCMCChains.Chains;
                               mode_frac::Real = 0.3)
    try
        target.transform === nothing && return NaN
        names = target.params.layout.unfrozen_names
        d = length(names)
        d >= 1 || return NaN
        lp = vec(Array(chains[:, :lp, :]))
        cols = [vec(Array(chains[:, Symbol(nm), :])) for nm in names]
        N = length(lp)
        N >= 2d + 2 || return NaN
        pt = target.transform
        Y = Matrix{Float64}(undef, d, N)
        @inbounds for i in 1:N
            Y[:, i] = transform_forward(Float64[cols[j][i] for j in 1:d], pt)
        end
        good = isfinite.(lp) .& vec(all(isfinite, Y; dims = 1))
        count(good) >= 2d + 2 || return NaN
        Y = Y[:, good]; lp = lp[good]; Nk = length(lp)
        imax = argmax(lp); mode = Y[:, imax]
        Σ  = cov(Y; dims = 2) + 1e-8 * I(d)
        Lc = cholesky(Symmetric(Σ)).L
        md = [norm(Lc \ (Y[:, i] .- mode)) for i in 1:Nk]
        sel = md .< quantile(md, mode_frac)
        Yc = count(sel) >= 2d + 2 ? Y[:, sel] : Y
        Sc = cov(Yc; dims = 2) + 1e-8 * I(d)
        return lp[imax] + 0.5 * d * log(2π) + 0.5 * logdet(Symmetric(Sc))
    catch
        return NaN
    end
end

# ------------------------------------------------------------------
# Periodic convergence diagnostic on the β=1 ensemble (walkers = chains).
# Returns (mean_rhat, max_rhat, mean_bulk_ess, min_tail_ess) over the
# `science_cols` columns of the flat `samples` buffer, or `nothing` when
# there are too few post-burnin steps. Steps are thinned to ≤ `cap` so the
# cost stays flat as the chain grows. The gate (convergence_stop) is on the
# WORST science param (max R-hat, min tail-ESS); nuisances are excluded so a
# poorly-constrained jitter/GP hyperparameter can never stall the run — and a
# nuisance that's coupled to a science param surfaces in that param's R-hat.
# ------------------------------------------------------------------
function _pt_convergence(samples::AbstractMatrix, keep_idx::Int,
                          n_walkers::Int, science_cols::Vector{Int};
                          cap::Int = 2000)
    (n_walkers < 2 || keep_idx < 4 * n_walkers) && return nothing
    keep_idx % n_walkers == 0 || return nothing
    n_step = keep_idx ÷ n_walkers
    n_step < 4 && return nothing
    ns = length(science_cols)
    ns == 0 && return nothing
    stride = max(1, n_step ÷ cap)
    kept = 1:stride:n_step
    nk = length(kept)
    cube = Array{Float64,3}(undef, nk, ns, n_walkers)
    @inbounds for w in 1:n_walkers
        for (si, s) in enumerate(kept)
            base = (s - 1) * n_walkers + w
            for (ci, c) in enumerate(science_cols)
                cube[si, ci, w] = samples[base, c]
            end
        end
    end
    local rh, bess, tess
    try
        ch = MCMCChains.Chains(cube)
        er = MCMCChains.ess_rhat(ch)
        rh = er[:, :rhat]; bess = er[:, :ess]
        tess = try MCMCChains.ess(ch; kind = :tail)[:, :ess] catch; bess end
    catch
        return nothing
    end
    # MCMCChains returns a SCALAR (not a 1-element vector) when there is exactly
    # one science column — which happens whenever most parameters are held at
    # FixedPrior (e.g. an RM-only fit). Normalise to a vector before filtering.
    _asvec(x) = x isa AbstractArray ? vec(x) : [x]
    rhf = filter(isfinite, _asvec(rh))
    bf  = filter(isfinite, _asvec(bess))
    tf  = filter(isfinite, _asvec(tess))
    isempty(rhf) && return nothing
    (mean(rhf), maximum(rhf),
     isempty(bf) ? 0.0 : mean(bf),
     isempty(tf) ? 0.0 : minimum(tf))
end

"""
    sample_ptemcee(target, data; kwargs...) -> PTemceeResult

Parallel-tempered affine-invariant ensemble MCMC (Vousden+ 2016).
Walkers operate in unconstrained space; bounded-space samples are
recovered via the target's inverse transform.

# Keywords
- `n_temps::Int=5` — temperature levels.
- `n_walkers::Int=100` — walkers per temperature (≥ `2·n_dim + 2`, even).
- `n_steps::Int=2000` — steps per walker.
- `n_burnin::Int=1000` — burn-in discarded from posterior + evidence.
- `betas::Union{Nothing,Vector{Float64}}=nothing` — explicit β ladder
  (descending β[1]=1). Default: Vines+ 2023's `β_i = (1/√5)^i`.
- `stretch_a::Float64=2.0` — Goodman-Weare stretch parameter.
- `init::Union{Nothing,Vector{Float64}}=nothing` — bounded-space walker
  init point (Gaussian scatter around it).
- `init_strategy::Symbol=:prior` — how walkers are initialized when
  `init === nothing`:
    - `:prior` (default, astroEMPEROR-style) — every walker at every
      temperature draws independently from the prior. Lots of dispersion
      means some walkers reliably land near the high-likelihood region;
      stretch moves on the ensemble + swap-down from hot chains bring
      that information to the cold β=1 chain. The robust choice on
      multimodal / wide-prior targets.
    - `:pathfinder` — `pathfinder_init` + per-walker draws. Tight
      cloud near a single basin. Fast but collapses ensemble diversity
      when the Pathfinder MVN approximation poorly matches the
      posterior (Pareto k > 1), so acceptance can die.
    - `:map_scatter` — `sample_map` MAP point + small Gaussian
      scatter. Even tighter than Pathfinder; only safe when the
      posterior is approximately Gaussian around the MAP.
- `pf_n_runs::Int=2` — Pathfinder L-BFGS basins (only used if
  `:pathfinder`).
- `seed::Int=1`
- `thin::Int=1` — thinning factor on β=1 posterior output.
- `show_progress::Bool=true` — display the per-step progress bar.
"""
function sample_ptemcee(
    target::NereusTarget,
    data::Data;
    n_temps::Int = 5,
    n_walkers::Int = 100,
    n_steps::Int = 2000,
    n_burnin::Int = 1000,
    betas::Union{Nothing,AbstractVector} = nothing,
    beta_min::Real = 1e-4,
    stretch_a::Real = 2.0,
    init::Union{Nothing,Vector{Float64}} = nothing,
    init_strategy::Symbol = :prior,
    pf_n_runs::Int = 2,
    seed::Int = 1,
    thin::Int = 1,
    show_progress::Bool = true,
    diag_every::Int = 0,
    convergence_stop::Bool = false,
    science_params::Union{Nothing,Vector{String}} = nothing,
    rhat_threshold::Real = 1.01,
    tail_ess_threshold::Real = 1000,
    n_converged_checks::Int = 3,
    min_steps::Int = 0,
    adapt_ladder::Bool = false,
    ladder_adapt_window::Int = 50,
    ladder_adapt_ν0::Real = 10.0,
    ladder_adapt_K::Real = 1.0,
    laplace_switch_tol::Real = 50.0,
    bridge_headline::Bool = true,
    bridge_n::Int = 20_000,
    bridge_warn_tol::Real = 20.0,
    untemper_transit::Bool = false,
)
    # JSON delivers floats-as-Int and arrays-as-JSON3.Array; normalize.
    stretch_a       = Float64(stretch_a)
    ladder_adapt_ν0 = Float64(ladder_adapt_ν0)
    ladder_adapt_K  = Float64(ladder_adapt_K)
    # UNTEMPERED TRANSIT: hold the transit term in the reference measure so the
    # ladder only has to bridge the RV likelihood. The 16k-point transit dominates
    # σ(logL) — hence Δβ·σ — while carrying no information that differs between
    # rungs, so tempering it costs swap acceptance and buys nothing. β=1 is still
    # the exact joint posterior; nothing is frozen or pre-fit.
    if untemper_transit
        if isempty(data.t_phot)
            @warn "untemper_transit=true but no photometry present — no effect."
        else
            @info "untemper_transit: π_β ∝ prior·L_transit·L_RV^β (transit at full " *
                  "strength in every rung; β=1 unchanged). NOTE: thermodynamic-" *
                  "integration logZ is now RELATIVE to the transit-conditioned " *
                  "reference — add the transit-only evidence for absolute logZ."
        end
    end
    betas = betas === nothing ? nothing : collect(Float64, betas)
    params = target.params
    layout = params.layout
    n_dim = length(layout.unfrozen_idx)
    unfrozen_idx = layout.unfrozen_idx

    # Convergence-diagnostic setup. Science params (the gating set) default to
    # the planet block (`*_k<n>`) plus rho_s; everything else (jitter, gammas,
    # offsets, GP/AD noise params, LD) is monitored but non-gating. Live
    # readout cadence: post-burnin, every `diag_cadence` steps.
    _sci_names = science_params === nothing ?
        String[nm for nm in layout.unfrozen_names if occursin(r"_k\d+$", nm) || nm == "rho_s"] :
        science_params
    science_cols = Int[i for (i, nm) in enumerate(layout.unfrozen_names) if nm in _sci_names]
    isempty(science_cols) && (science_cols = collect(1:n_dim))   # fallback: gate on all
    diag_cadence = diag_every > 0 ? diag_every : max(200, n_steps ÷ 40)
    # Walkers live in BOUNDED space — no transform used in the move
    # loop. The transform is irrelevant for ptemcee: M-H acceptance
    # only cares about target density ratios in whatever space the
    # walkers live, and bounded space gives clean auto-reject behavior
    # via log_prior(theta) = -Inf outside support.

    # Goodman-Weare: each half ≥ n_dim+1 walkers, total even.
    n_walkers_eff = max(n_walkers, 2 * n_dim + 2)
    n_walkers_eff % 2 == 1 && (n_walkers_eff += 1)

    # Geometric ladder across [beta_min, 1], matching sample_transdim_ptemcee.
    #
    # This used to be hardcoded (1/sqrt(5))^i with no beta_min, which puts the
    # hottest rung at 5^(-(n_temps-1)/2) — 0.04 at the default 5 temps. TI and
    # TI+ integrate <logL> only over [beta_min, 1] and ss_plus telescopes
    # Z(beta_min)->Z(1); neither adds the [0, beta_min] head, so that head was
    # silently dropped. Measured on the HD 18599 joint fit it is worth ~715 nats
    # between 5 and 10 rungs — an evidence error far larger than any model
    # comparison it would be used for.
    #
    # sample_transdim_ptemcee already took beta_min and built the ladder this
    # way; the fixed-dim sampler did not even accept the argument, so the two
    # returned incomparable evidences for the same problem.
    βs = betas === nothing ?
         Float64[Float64(beta_min)^(i / max(1, n_temps - 1)) for i in 0:(n_temps - 1)] :
         copy(betas)
    length(βs) == n_temps || throw(ArgumentError(
        "Provided `betas` has length $(length(βs)), expected $n_temps"))

    # --- Per-thread mutable state -------------------------------------
    n_thr = max(Threads.nthreads(), 1)
    thread_theta    = [Theta{Float64}(params) for _ in 1:n_thr]
    # Per-thread PTWorkspace → ws-aware likelihood: no per-call Vector allocs
    # (the ~27.7 MB/eval GC thrash on data-rich fits) + per-planet flux cache +
    # total phot-ll cache. ptemcee is the workhorse recoverer, so this speeds up
    # the whole fixed-dim menu. One ws per thread, indexed by threadid().
    thread_ws       = [PTWorkspace(params, params.config.max_kplanet,
                                   length(params.config.noise_models);
                                   n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                       for _ in 1:n_thr]
    thread_xb       = [Vector{Float64}(undef, n_dim) for _ in 1:n_thr]
    thread_proposal = [Vector{Float64}(undef, n_dim) for _ in 1:n_thr]
    thread_rngs     = [MersenneTwister(seed + 1_000_003 * t) for t in 1:n_thr]
    rng_master      = MersenneTwister(seed)

    # Evaluate (log_prior_bounded, log_like) at a BOUNDED-space point
    # `x` using thread-local Theta. Returns (-Inf, -Inf) if x is
    # outside the prior support — caller treats this as an automatic
    # M-H reject. log_like includes external priors (eccentricity,
    # rho_s) per Nereus convention, so these get tempered too —
    # matches the in-house PT path's convention.
    @inline function eval_bounded!(x::AbstractVector{Float64}, tid::Int)
        theta = thread_theta[tid]
        wb = thread_ws[tid]
        @inbounds for (j, idx) in enumerate(unfrozen_idx)
            theta.values[idx] = x[j]
        end
        lp = log_prior(theta)
        isfinite(lp) || return (-Inf, -Inf)
        ll_rv = rv_log_likelihood(theta, data, wb)
        ll_tr = transit_log_likelihood(theta, data, wb)
        # UNTEMPERED TRANSIT (see `untemper_transit`): π_β ∝ prior·L_transit·L_RV^β.
        # Folding the transit into `lp` makes the within-chain acceptance
        # (β·Δll + Δlp), the swap ratio (Δβ·Δll) and the TI accumulator over `ll`
        # all consistent with the sampled targets by construction. Patching the
        # swap ratio alone would break detailed balance.
        return untemper_transit ? (lp + ll_tr, ll_rv) : (lp, ll_rv + ll_tr)
    end

    # --- Pathfinder warm-start (recommended for ensemble PT) ----------
    # Walkers initialized broadly across a high-d posterior cannot
    # bootstrap a coherent stretch ensemble — acceptance collapses.
    # Pathfinder gives us a multivariate-normal posterior approximation
    # at one or more L-BFGS basins; draws from this mixture are already
    # concentrated near plausible modes. Each walker gets a fresh draw.
    # Pre-built init draws for :pathfinder and :map_scatter strategies.
    # `:prior` builds draws per-walker inside the parallel loop below.
    pf_draws = if init === nothing && init_strategy === :pathfinder
        n_pf_draws = n_temps * n_walkers_eff
        try
            pf = pathfinder_init(target;
                                  n_runs  = pf_n_runs,
                                  n_draws = n_pf_draws,
                                  seed    = seed,
                                  quiet   = true)
            avail = size(pf.draws, 2)
            avail > 0 ? pf.draws : nothing
        catch e
            @warn "Pathfinder warm-start failed; falling back to :map_scatter" exception=(e, catch_backtrace())
            nothing
        end
    elseif init === nothing && init_strategy === :map_scatter
        # MAP point in BOUNDED space + per-parameter scatter sized to
        # 1% of the prior range (for uniform priors) or 1·σ (for normal
        # priors). This keeps walkers in a tight in-support cluster
        # around the MAP — stretch moves between near-neighbors stay
        # in-box, acceptance is high.
        n_pf_draws = n_temps * n_walkers_eff
        map_res = sample_map(target; method = :LBFGS, maxiter = 500,
                             g_tol = 1e-6, n_starts = max(2, pf_n_runs))
        x_map = map_res.x_map  # bounded space
        scatter_per_dim = Vector{Float64}(undef, n_dim)
        @inbounds for j in 1:n_dim
            ps = layout.unfrozen_priors[j]
            if isfinite(ps.lo) && isfinite(ps.hi)
                scatter_per_dim[j] = 0.01 * (ps.hi - ps.lo)
            else
                scatter_per_dim[j] = max(0.01 * abs(x_map[j]), 1e-4)
            end
        end
        jitter_seed = MersenneTwister(seed)
        pf_d = Matrix{Float64}(undef, n_dim, n_pf_draws)
        for j in 1:n_pf_draws
            @inbounds for d in 1:n_dim
                pf_d[d, j] = x_map[d] + scatter_per_dim[d] * randn(jitter_seed)
            end
        end
        pf_d
    else
        nothing
    end

    # --- Initialize walkers in BOUNDED space (threaded) --------------
    # Default `:prior` strategy draws each walker independently from
    # the prior — matches astroEMPEROR. Pathfinder/MAP strategies are
    # available but `:prior` is the most robust on multimodal / high-d
    # targets (see earlier collapse failure modes with concentrated
    # init).
    state    = Array{Float64,3}(undef, n_temps, n_walkers_eff, n_dim)
    logL_arr = fill(-Inf, n_temps, n_walkers_eff)
    logπ_arr = fill(-Inf, n_temps, n_walkers_eff)
    init_seeds = rand(rng_master, UInt64, n_temps * n_walkers_eff)
    Threads.@threads :static for task_idx in 1:(n_temps * n_walkers_eff)
        tid = Threads.threadid()
        t = (task_idx - 1) ÷ n_walkers_eff + 1
        w = (task_idx - 1) % n_walkers_eff + 1
        rng = MersenneTwister(init_seeds[task_idx])
        for _ in 1:200
            if pf_draws !== nothing
                col = mod1(task_idx, size(pf_draws, 2))
                if init_strategy === :pathfinder
                    # Pathfinder draws are in unconstrained space.
                    y_uc = Vector{Float64}(undef, n_dim)
                    @inbounds for d in 1:n_dim
                        y_uc[d] = pf_draws[d, col] + 1e-4 * randn(rng)
                    end
                    x_b = target.transform === nothing ? y_uc :
                          transform_inverse(y_uc, target.transform)
                    @inbounds for d in 1:n_dim
                        state[t, w, d] = x_b[d]
                    end
                else
                    # :map_scatter — already in bounded space.
                    @inbounds for d in 1:n_dim
                        state[t, w, d] = pf_draws[d, col]
                    end
                end
            elseif init !== nothing
                @inbounds for d in 1:n_dim
                    state[t, w, d] = init[d] + 1e-3 * randn(rng)
                end
            else
                x_b = _draw_from_prior(target, rng)
                @inbounds for d in 1:n_dim
                    state[t, w, d] = x_b[d]
                end
            end
            lp, ll = eval_bounded!(@view(state[t, w, :]), tid)
            if isfinite(lp) && isfinite(ll)
                logπ_arr[t, w] = lp
                logL_arr[t, w] = ll
                break
            end
        end
    end

    # --- Sample storage (β=1, post-burnin, thinned; bounded space) ---
    n_keep_per_walker = max(0, cld(n_steps - n_burnin, thin))
    n_keep_total      = n_keep_per_walker * n_walkers_eff
    samples = Matrix{Float64}(undef, max(n_keep_total, 1), n_dim)
    lp_samples = Vector{Float64}(undef, max(n_keep_total, 1))
    keep_idx = 0

    # reddemcee TI+/SS+/H+ accumulators (one per pair of adjacent
    # temperatures; cf evidence.jl). We feed the FULL ensemble of
    # walkers at each (β_k, β_{k+1}) pair every post-burnin step.
    evidence_acc = EvidenceAccumulator(length(βs))
    # Threshold round for evidence accumulation matches reddemcee:
    # start streaming once burn-in is past so the leading transient
    # doesn't bias <logL>_β.

    accept_within  = zeros(Float64, n_temps)
    propose_within = zeros(Int, n_temps)
    accept_swap    = zeros(Float64, n_temps - 1)
    propose_swap   = zeros(Int, n_temps - 1)
    n_evals_atomic = Threads.Atomic{Int}(n_temps * n_walkers_eff)
    accept_temp    = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    propose_temp   = [Threads.Atomic{Int}(0) for _ in 1:n_temps]

    half = n_walkers_eff ÷ 2
    n_active_per_temp = half
    tasks_h1 = Vector{Tuple{Int,Int}}(undef, n_temps * n_active_per_temp)
    tasks_h2 = Vector{Tuple{Int,Int}}(undef, n_temps * n_active_per_temp)
    let i1 = 0, i2 = 0
        for t in 1:n_temps
            for w in 1:half;                  i1 += 1; tasks_h1[i1] = (t, w); end
            for w in (half + 1):n_walkers_eff; i2 += 1; tasks_h2[i2] = (t, w); end
        end
    end

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

    # --- Progress bar (per-step accept rate + ETA) --------------------
    pb = ProgressBar("ptemcee"; total = n_steps, enabled = show_progress)

    # --- Live convergence readout + optional run-until-converged ------
    rhat_str = "—"; ess_str = "—"      # cached "mean/worst" displays
    conv_run = 0; converged_at = 0     # consecutive passing checks; stop step

    # --- Main loop ----------------------------------------------------
    for step in 1:n_steps
        do_half_step!(tasks_h1, :h1)
        do_half_step!(tasks_h2, :h2)

        @inbounds for t in 1:n_temps
            propose_within[t] += Threads.atomic_xchg!(propose_temp[t], 0)
            accept_within[t]  += Threads.atomic_xchg!(accept_temp[t],  0)
        end

        # ---- Swap moves (serial; cheap, no likelihood evals) ---------
        window_accs   = zeros(Int, n_temps - 1)
        window_props  = zeros(Int, n_temps - 1)
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

        # ---- Adaptive β-ladder (Vousden+ 2016 Algorithm 1, eq 12) ----
        # Update during burn-in only. Adjusts inverse-temp spacings to
        # equalize per-pair swap acceptance, keeping β[1]=1.0 fixed
        # (cold) and β[end] fixed at its initial hot value.
        # Diminishing-adaptation: γ_t = 1/((t/window) + ν0).
        if adapt_ladder && step <= n_burnin && step % ladder_adapt_window == 0 &&
           n_temps >= 3
            γt = ladder_adapt_K / ((step / ladder_adapt_window) + ladder_adapt_ν0)
            T = 1.0 ./ βs   # temperatures, ascending from T=1 to T=∞ (β=0)
            α = [window_props[i] > 0 ? window_accs[i] / window_props[i] : 0.0
                  for i in 1:(n_temps - 1)]
            # Update interior temperatures (i = 2..n_temps-1):
            #   d log(T_i - T_{i-1}) = γ × (α_{i-1} - α_i)
            # so T_i ← T_{i-1} + (T_i - T_{i-1}) × exp(γ × (α_{i-1} - α_i))
            for i in 2:(n_temps - 1)
                spacing = T[i] - T[i - 1]
                T[i] = T[i - 1] + spacing * exp(γt * (α[i - 1] - α[i]))
            end
            # Keep T strictly increasing
            for i in 2:(n_temps - 1)
                T[i] = max(T[i], T[i - 1] + 1e-6)
            end
            # Convert back to β; pin endpoints
            new_betas = 1.0 ./ T
            new_betas[1] = βs[1]           # keep cold β=1
            new_betas[end] = βs[end]       # keep hot β fixed
            βs .= new_betas
        end

        # ---- Post-burnin: record β=1 + feed evidence accumulator ----
        if step > n_burnin
            # For each (β_k, β_{k+1}) pair, feed each walker's logL
            # to the evidence accumulator. The H+ / SS+ estimators in
            # evidence.jl average within and across walkers correctly.
            @inbounds for w in 1:n_walkers_eff
                # Pass per-temperature logL vector for this walker
                logL_walker = view(logL_arr, :, w)
                update_evidence!(evidence_acc, logL_walker, βs)
            end
            if (step - n_burnin) % thin == 0
                @inbounds for w in 1:n_walkers_eff
                    keep_idx += 1
                    # Walkers already live in bounded space.
                    for d in 1:n_dim
                        samples[keep_idx, d] = state[1, w, d]
                    end
                    lp_samples[keep_idx] = logπ_arr[1, w] + logL_arr[1, w]
                end
            end
        end

        # ---- Periodic convergence diagnostic (β=1 science params) ----
        # Mean/worst R-hat and bulk/tail ESS over the gating set; drives both
        # the live readout and (optionally) the run-until-converged stop.
        if step > n_burnin && (step - n_burnin) % diag_cadence == 0
            diag = _pt_convergence(samples, keep_idx, n_walkers_eff, science_cols)
            if diag !== nothing
                meanr, maxr, meanb, mte = diag
                rhat_str = string(round(meanr, digits = 3), "/", round(maxr, digits = 3))
                ess_str  = string(round(Int, meanb), "/", round(Int, mte))
                if convergence_stop
                    pass = step >= min_steps && maxr < rhat_threshold &&
                           mte > tail_ess_threshold
                    conv_run = pass ? conv_run + 1 : 0
                    conv_run >= n_converged_checks && (converged_at = step)
                end
            end
        end

        # ---- Progress update ----------------------------------------
        if show_progress
            total_props = sum(propose_within)
            total_acc = sum(accept_within)
            acc_rate = total_props > 0 ? total_acc / total_props : 0.0
            update!(pb; n_done = step,
                    fields = (:acc => round(acc_rate, digits = 3),
                              :β1_acc => round(accept_within[1] / max(propose_within[1], 1), digits = 3),
                              :Rhat => rhat_str,    # mean/worst over science params
                              :ESS => ess_str,      # mean(bulk)/worst(tail)
                              :nevals => n_evals_atomic[]))
        end

        # ---- Run-until-converged: stop once the science params clear the
        # gate for `n_converged_checks` consecutive diagnostics ----------
        if convergence_stop && converged_at > 0
            show_progress && @info "ptemcee: science params converged at step $converged_at " *
                "(Rhat<$(rhat_threshold), tail-ESS>$(tail_ess_threshold)); stopping early"
            break
        end
    end
    show_progress && finish!(pb)
    if convergence_stop && converged_at == 0
        @warn "ptemcee: hit max_steps=$n_steps WITHOUT meeting the convergence gate " *
              "(Rhat<$(rhat_threshold), tail-ESS>$(tail_ess_threshold)) on science params — " *
              "results may be unconverged (last $rhat_str Rhat, $ess_str ESS)"
    end

    samples = samples[1:keep_idx, :]
    lp_samples = lp_samples[1:keep_idx]

    # --- Evidence report (TI + TI+ + SS+ + H+) ------------------------
    ev_report = evidence_report(evidence_acc, βs)
    # H+ is the reddemcee recommended estimator; fall back to TI+
    # if H+ is degenerate (β* selection failed), TI as last resort.
    log_z = isfinite(ev_report.hybrid[1]) ? ev_report.hybrid[1] :
            isfinite(ev_report.ti_plus[1]) ? ev_report.ti_plus[1] :
            ev_report.ti[1]

    # --- Output -------------------------------------------------------
    # Append :lp (log-posterior at β=1) so downstream plotting can do
    # EMPEROR-style top-fraction filtering by best-fit cluster.
    #
    # Sample layout above is (step1_w1, step1_w2, …, step1_wN, step2_w1,
    # …, stepK_wN) — N walkers stacked per step. Reshape to
    # `(K, n_dim+1, N)` so MCMCChains exposes each walker as a separate
    # chain. plot_trace then plots one thin line per walker (catching
    # stuck walkers / mode-hops) instead of a flattened black blob.
    param_names = Symbol.(layout.unfrozen_names)
    push!(param_names, :lp)
    samples_with_lp = hcat(samples, lp_samples)
    chains = if keep_idx > 0 && n_walkers_eff > 0 &&
                keep_idx % n_walkers_eff == 0
        # Reshape: row k corresponds to walker ((k-1) mod N)+1 at step
        # ((k-1) ÷ N)+1. permutedims to (step, dim, walker).
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
        # Fallback: single flat chain (very small kept count)
        MCMCChains.Chains(samples_with_lp, param_names)
    end
    acc_within = accept_within ./ max.(propose_within, 1)
    acc_swap   = accept_swap   ./ max.(propose_swap,   1)

    # --- mode-Laplace cross-check + evidence selection (Jose's rule) --------
    # The tempered estimators (TI/SS/H+) fail catastrophically on phase-transition
    # posteriors — a signature is a tiny minimum swap acceptance that won't rise
    # with more temps. The mode-Laplace estimator is immune (it never tempers).
    # Rule: if they agree, keep the tempered best (exact when it works); if they
    # disagree by ≫ the Laplace non-Gaussian bias, the tempered one is broken —
    # use Laplace, and warn.
    # Under untemper_transit (+photometry) the two estimators live on DIFFERENT
    # scales: mode-Laplace (built from :lp = full posterior) is ABSOLUTE logZ,
    # while the tempered TI/SS/H+ integrate only L_RV and return logZ RELATIVE
    # to the transit-conditioned reference. Their difference then contains
    # Z_ref (~1e4-1e5 nats for a 16k-pt transit) — the cross-check would fire
    # on EVERY run, silently replace the relative evidence with the absolute
    # one (breaking the Bayes-factor cancellation the flag exists for), and
    # emit a spurious phase-transition warning. Skip the switch; keep both
    # numbers in the result (log_z relative, log_z_laplace absolute).
    laplace_check = !(untemper_transit && !isempty(data.t_phot))
    log_z_laplace = mode_laplace_evidence(target, chains)

    # Bridge sampling from the cold chain we already have. Costs bridge_n
    # likelihood evaluations (seconds), touches no hot rung, and assumes only
    # that the fitted reference OVERLAPS the posterior -- not that the posterior
    # is Gaussian about its mode. Requires an unconstrained target, as the
    # proposal is Gaussian on R^n.
    log_z_bridge = NaN
    bridge_overlap = NaN
    if bridge_headline && target.transform !== nothing
        try
            b = bridge_evidence(target, chains; n_proposal = bridge_n, seed = seed)
            if b.converged && isfinite(b.log_z)
                log_z_bridge = b.log_z
                bridge_overlap = b.overlap
            end
        catch err
            @debug "sample_ptemcee: bridge evidence unavailable" exception = err
        end
    end

    # A NON-FINITE tempered evidence is the loudest possible failure and used to
    # be the quietest: the switch below is gated on isfinite(log_z), so -Inf or
    # NaN skipped the check entirely and propagated into res.log_evidence with no
    # warning at all. One -Inf log-likelihood on any rung is enough to produce it
    # (see the guard in update_evidence!). Handle it first and explicitly.
    # HEADLINE SELECTION.
    #
    # Ordered by what each estimator has actually been measured to do, not by
    # what is cheapest. On 51 Peg (1691 RVs, 15 params, unimodal, P determined
    # to 5 decimals) the spread was:
    #
    #   TI+ / H+  -6073.24     SS+  -6097.41     ~178 nats low
    #   Laplace   -5920.97                        ~24 nats low
    #   bridge    -5897.0                         validated pair
    #   refpath   -5893.1
    #
    # The beta-path stack fails at the phase transition and cannot detect it --
    # TI+ and H+ agreed to the second decimal while both were 178 nats out. So
    # bridge is the headline when it is usable, Laplace only as a fallback, and
    # the tempered value only when it is corroborated.
    tempered_ok = isfinite(log_z)
    bridge_ok   = isfinite(log_z_bridge)
    log_z_tempered = log_z

    if bridge_ok
        # Corroborate rather than assume. Bridge cannot self-detect a posterior
        # its single reference fails to cover -- `overlap` counts draws landing
        # in SUPPORT, not mode coverage, and read 0.99 on a posterior with 23
        # period modes. A large bridge/Laplace gap is the cheapest available
        # signal that something is off; say so rather than picking silently.
        if isfinite(log_z_laplace) && abs(log_z_bridge - log_z_laplace) > bridge_warn_tol
            @warn "sample_ptemcee: bridge and mode-Laplace disagree by " *
                  "$(round(abs(log_z_bridge - log_z_laplace), digits=1)) nats. " *
                  "Reporting bridge (it is the validated one), but CHECK THE " *
                  "POSTERIOR IS NOT MULTIMODAL before quoting it — a single " *
                  "Gaussian reference cannot cover multiple modes and overlap " *
                  "will not tell you so." bridge = log_z_bridge laplace = log_z_laplace overlap = bridge_overlap
        end
        log_z = log_z_bridge
    elseif isfinite(log_z_laplace)
        @warn "sample_ptemcee: bridge evidence unavailable; falling back to " *
              "mode-Laplace. Treat it as indicative: on a clean unimodal target " *
              "it measured 24 nats from bridge, and it has no validation against " *
              "a known log Z." laplace = log_z_laplace
        log_z = log_z_laplace
    elseif tempered_ok
        @warn "sample_ptemcee: neither bridge nor mode-Laplace is available. " *
              "log_evidence is the TEMPERED value, which is biased low by " *
              "10^2-10^4 nats on a signal-locked posterior and cannot detect " *
              "that it is. Do not quote it without an independent check." tempered = log_z
    else
        @warn "sample_ptemcee: no usable evidence estimate for this run. " *
              "log_evidence is not quotable."
    end

    # Independent of which was chosen: a large tempered/headline gap is the
    # phase-transition signature and worth surfacing, because the tempered
    # numbers are still in `evidence` and someone will read them.
    if tempered_ok && isfinite(log_z) && laplace_check &&
       abs(log_z_tempered - log_z) > laplace_switch_tol
        min_swap = isempty(acc_swap) ? NaN : minimum(acc_swap)
        @warn "sample_ptemcee: the TEMPERED stack (TI/TI+/SS+/H+ in `evidence`) " *
              "is $(round(abs(log_z_tempered - log_z), digits=1)) nats from the " *
              "reported log_evidence — the phase-transition signature (min swap " *
              "accept = $(round(min_swap, digits=3))). Those four share one " *
              "mean_logL array, so their agreeing with EACH OTHER is not a " *
              "check." tempered = log_z_tempered reported = log_z
    end

    return PTemceeResult(
        chains, log_z, log_z_laplace, log_z_bridge, ev_report, acc_within, acc_swap,
        βs, n_evals_atomic[]
    )
end
