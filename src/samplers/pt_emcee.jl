# Parallel-tempered affine-invariant ensemble MCMC (pt_emcee).
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
# structure lets pt_emcee exit local modes single-walker cold chains get
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
#   A LogUniform / ModJeffreys parameter moves in log(x + s), where its
#   prior is exactly flat: still a hard-walled box, and the Jacobian
#   cancels the prior's 1/(x + s), so there is no gradient to act as a
#   wall (see the note in `sample_pt_emcee`).
#   The exception is a full-circle angle (Mo, Ω, λ, ω under `:ew`; see
#   src/circular.jl), whose window is a chart and not a wall: its seam
#   is moved to the emptiest arc of the cold ensemble halfway through
#   burn-in, with every pruning pass and at the end of burn-in
#   (`_pt_ensemble_recut!`, `_pt_recut_due`), so the chart is fixed for
#   every recorded step.
# - **Node flip** for astrometry-only planets: a tempered M-H move to the
#   mirror (Ω + π, ω + π), the exactly degenerate solution the stretch move
#   cannot reach (src/samplers/node_flip.jl).
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
# tempering. Cite Vousden+ 2016 (the `ptemcee` package), not emcee.

using Random

# Deterministic per-walker stream seed. Mixed arithmetically (not via
# `hash`) so it is stable across Julia versions.
@inline function _walker_seed(seed::Integer, half::Integer, i::Integer)
    s = reinterpret(UInt64, Int64(seed))
    s * 0x9e3779b97f4a7c15 + UInt64(half) * 0x517cc1b727220a95 + UInt64(i)
end
using MCMCChains
using Statistics: cov, quantile, median, std
using LinearAlgebra: cholesky, logdet, Symmetric, norm, I, issuccess

export sample_pt_emcee, PTemceeResult, mode_laplace_evidence

"""
    PTemceeResult

Container for a `sample_pt_emcee` run.

# Fields
- `chains::MCMCChains.Chains` — β=1 walkers, post-burnin, flattened, in bounded space
  (circular angles in the window re-charted at the end of burn-in, so contiguous).
- `log_evidence::Float64` — evidence to USE: `log_evidence_bridge` when it is
  usable; otherwise the tempered value (H+) when its ladder can carry it (the
  ladder reaches β = 0 and every adjacent pair swapped at least `swap_min`,
  0.05, of the time, which the default ladder, geometric down to `beta_min`,
  never does) and it agrees with the mode-Laplace cross-check within
  `laplace_switch_tol`; otherwise `log_evidence_laplace`, with a warning (the
  tempered estimators fail catastrophically on phase-transition / signal-locked
  posteriors; mode-Laplace does not). `NaN`, with a warning, only when none of
  them is available. Under `untemper_transit` with photometry the tempered value
  is relative and mode-Laplace absolute, so there is no switch. See
  [`mode_laplace_evidence`](@ref).
- `log_evidence_laplace::Float64` — mode-anchored Laplace evidence from the cold
  chain, a Gaussian approximation at the mode; phase-transition-immune. NaN if
  not computable. On 51 Peg (unimodal, tightly determined), converged, it sits
  5.5 nats above bridge and reference-path, which agree with each other to
  0.07 and are validated against exact truth on a curved 15-D target; on an
  RV + Hipparcos-IAD companion within 0.25 of bridge. A warning says when it
  is further from bridge than bridge's standard error + 1 nat.
- `log_evidence_bridge::Float64` — bridge-sampling evidence from the same cold
  chain (Meng & Wong optimal bridge). This is the HEADLINE when it is usable:
  it never touches the prior end and makes no Gaussianity assumption. Usable
  means `_bridge_guard` passes: the chain has converged (worst R-hat ≤ 1.1)
  and the least ESS is at least `q`'s d + d(d+1)/2 parameters; otherwise it
  is biased low, and the value here is NaN with a warning saying which. It costs
  `bridge_n` likelihood evaluations: the posterior at the kept draws is read
  off their `:lp`, after a check against a full evaluation at a few of them
  (`bridge_evidence`). NaN if not computable.
- `evidence::EvidenceReport` — full TI / TI+ / SS+ / H+ tempered stack (raw,
  whether or not it is quotable; summary.json carries the values only when it
  is).
- `acceptance_within::Vector{Float64}` — within-temp stretch acceptance per temp.
- `acceptance_swap::Vector{Float64}` — swap acceptance between (k, k+1). A tiny
  `minimum(acceptance_swap)` (≪0.1) that won't rise with more temps signals a
  phase transition; below `swap_min` the tempered evidence is not quotable.
- `betas::Vector{Float64}` — β ladder used.
- `n_evals::Int` — total likelihood evaluations.
- `ladder::LadderHistory` — the ladder, the swap acceptance and the swap mean
  distance at every step, and `⟨log L⟩` per rung: what [`plot_ladder_rates`](@ref)
  and [`plot_beta_ladder`](@ref) draw. See [`LadderHistory`](@ref).
- `stop::NamedTuple` — how the run ended: `(; stopped, step, n_steps, n_draws,
  stop_file)`. `stopped` is `"steps"` (ran to `n_steps`), `"converged"` (the
  `convergence_stop` gate), `"interrupted"` (Ctrl-C after burn-in),
  `"requested"` (the stop file appeared), `"max_seconds"` or `"max_evals"`;
  `step` is the last step recorded and `n_draws` the rows kept (walkers × kept
  steps).
- `acceptance_de::Vector{Float64}` — within-temp acceptance of the
  differential-evolution move per temp (`de_move`); NaN where none was proposed.
- `drift` — the collective-drift check of the β=1 ensemble
  ([`ensemble_drift`](@ref)): a `Dict` with each parameter's z for a trend of
  the walkers' mean through the recorded steps and the parameters over the
  threshold -- a drift every walker shares, which R-hat and ESS cannot see --
  or `nothing` when the run was too short to test.
"""
struct PTemceeResult
    chains::MCMCChains.Chains
    log_evidence::Float64
    log_evidence_laplace::Float64
    log_evidence_bridge::Float64
    evidence::EvidenceReport
    acceptance_within::Vector{Float64}
    acceptance_de::Vector{Float64}
    acceptance_swap::Vector{Float64}
    betas::Vector{Float64}
    n_evals::Int
    ladder::LadderHistory
    stop::NamedTuple
    drift::Union{Nothing, Dict{String, Any}}
end

"""
    mode_laplace_evidence(target, chains; mode_frac=0.3) -> Float64

Laplace log-evidence anchored on the posterior MODE, computed from a PT run's own
cold chain — a fast estimator that is **immune to the phase-transition bias** that
makes tempered estimators (TI/SS/H+) fail on high-SNR / signal-locked posteriors
(it never tempers), and **robust to railed MAPs** (uses the sample covariance in
unconstrained space, not the Hessian, which goes singular at a rail).

`logZ ≈ logq(mode) + d/2·log2π + ½·logdet(Σ_local)`, all in the unconstrained
space: `logq` is the log posterior density THERE, the bounded one plus the
log-Jacobian of the map, the mode is the max-`logq` cold sample (NOT the mean —
a multimodal mean sits in a density valley), and `Σ_local` is the covariance of
the closest `mode_frac` of samples to the mode (excludes multimodal spread),
divided by the factor a Gaussian's covariance shrinks by when cut to that
inner fraction. Carries a ~O(100)-nat non-Gaussian bias on pathological
posteriors — negligible for strong detections, cross-check nested for marginal
ones. Returns `NaN` if not computable (bounded target, or too few samples).

The chain's `:lp` is read as the log posterior in the bounded space (pt_emcee,
pt, pt_whitening) or in the unconstrained one (pt_hmc, NUTS), whichever it
matches at a few draws against a full evaluation (`_laplace_logq`); without
`:lp`, or matching neither, the density is evaluated at every draw. Both
corrections were missing: pt_emcee's `:lp`, read as `logq`, lacked the
log-Jacobian, and the cut-down covariance was used as it came. On an RV +
Hipparcos-IAD companion that put pt_emcee's value about 1.1 nats above bridge
and pt_hmc's 2.6 to 3.1 below.
"""
function mode_laplace_evidence(target::NereusTarget, chains::MCMCChains.Chains;
                               mode_frac::Real = 0.3)
    try
        target.transform === nothing && return NaN
        nms = target.params.layout.unfrozen_names
        d = length(nms)
        d >= 1 || return NaN
        cols = [_draws(chains, Symbol(nm)) for nm in nms]
        N = length(cols[1])
        lp = :lp in names(chains) ? Float64.(_draws(chains, :lp)) : fill(NaN, N)
        N >= 2d + 2 || return NaN
        pt = target.transform
        # Cyclic inclinations folded to [0, π] (src/circular.jl), and the
        # stellar spin put in its reported representative (`spin_flip`): one
        # representative's mode, whose Laplace volume holds half of Z per fold
        # -- the mirror ones hold the rest (`n_folds` below).
        cinc = cyclic_inclination_indices(target.params)
        sc = spin_canon(target.params)
        Y = Matrix{Float64}(undef, d, N)
        @inbounds for i in 1:N
            xi = Float64[cols[j][i] for j in 1:d]
            fold_cyclic_inclinations!(xi, target.params; idx = cinc, sf = nothing)
            spin_canon!(xi, sc)
            Y[:, i] = transform_forward(xi, pt)
        end
        good = vec(all(isfinite, Y; dims = 1))
        any(isfinite, lp) && (good .&= isfinite.(lp))
        count(good) >= 2d + 2 || return NaN
        Y = Y[:, good]; lp = lp[good]; Nk = length(lp)
        logq = _laplace_logq(target, Y, lp)
        logq === nothing && return NaN
        imax = argmax(logq); mode = Y[:, imax]
        Σ  = cov(Y; dims = 2) + 1e-8 * I(d)
        Lc = cholesky(Symmetric(Σ)).L
        md = [norm(Lc \ (Y[:, i] .- mode)) for i in 1:Nk]
        sel = md .< quantile(md, mode_frac)
        # The draws within a Mahalanobis radius √q of a Gaussian's centre have
        # covariance c·Σ, c = F_{d+2}(q) / F_d(q) (F_k the χ²_k CDF): for d = 7
        # and the inner 30%, c = 0.46 and ½·logdet is 2.7 nats short. Undo it.
        local_ = count(sel) >= 2d + 2
        Yc = local_ ? Y[:, sel] : Y
        Sc = cov(Yc; dims = 2) + 1e-8 * I(d)
        shrink = 0.0
        if local_
            f = count(sel) / Nk
            c = cdf(Chisq(d + 2), quantile(Chisq(d), f)) / f
            shrink = d * log(c)
        end
        lz = logq[imax] + 0.5 * d * log(2π) + 0.5 * (logdet(Symmetric(Sc)) - shrink)
        nf = n_folds(sc, cinc)
        return nf == 0 ? lz : lz + nf * log(2.0)
    catch
        return NaN
    end
end

# The log posterior density in the unconstrained space at each column of `Y`
# (folded draws mapped forward), from the chain's `:lp` when it is one of the
# two conventions Nereus chains carry: the bounded log posterior (pt_emcee, pt,
# pt_whitening), to which the map's log-Jacobian is added, or the unconstrained
# one (pt_hmc, NUTS), taken as it is. Which one is checked at a few draws
# against a full evaluation; with neither (or no `:lp`) every draw is
# evaluated. `nothing` if that leaves no finite value.
function _laplace_logq(target::NereusTarget, Y::AbstractMatrix, lp::AbstractVector)
    pt = target.transform
    N = size(Y, 2)
    # bridge's per-task evaluators: thread-safe scratch, and the same
    # unconstrained log density (`_bridge_logdensity!`).
    evs = [_BridgeEvaluator(target) for _ in 1:max(1, _bridge_ntasks(N))]
    lj = [Float64(transform_logabsdetjac_inv(Y[:, i], pt)) for i in 1:N]
    k = unique(round.(Int, range(1, N; length = min(N, 8))))
    fk = [_bridge_logdensity!(evs[1], Y[:, i]) for i in k]
    tol(v) = 1e-6 + 1e-10 * abs(v)
    if all(isfinite, lp[k]) && all(isfinite, fk)
        all(abs(lp[i] + lj[i] - f) <= tol(f) for (i, f) in zip(k, fk)) && return lp .+ lj
        all(abs(lp[i] - f) <= tol(f) for (i, f) in zip(k, fk)) && return Vector{Float64}(lp)
    end
    q = Vector{Float64}(undef, N)
    _bridge_eval!(q) do c, i
        _bridge_logdensity!(evs[c], Y[:, i])
    end
    return any(isfinite, q) ? q : nothing
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
# `canon` (an `AngleCanon`, src/circular.jl) reads every angle in its reported
# representative: cyclic inclinations folded to [0, π] (a walker that passed
# through π sits in the mirror representative, the same inclination), the
# stellar spin flipped by the layout's axis, and each full-circle angle
# relabelled into the window with its seam in the emptiest arc -- walkers one
# representative apart must not count as separate modes. `fold` alone folds
# those columns and nothing else.
# ------------------------------------------------------------------
function _pt_convergence(samples::AbstractMatrix, keep_idx::Int,
                          n_walkers::Int, science_cols::Vector{Int};
                          cap::Int = 2000, fold::AbstractVector{Int} = Int[],
                          canon::Union{Nothing, AngleCanon} = nothing)
    ac = canon === nothing ? AngleCanon(collect(Int, fold), nothing, Float64[], Int[]) : canon
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
            flip = canon_row_flips(ac, samples, base)
            for (ci, c) in enumerate(science_cols)
                cube[si, ci, w] = canon_value(ac, samples, base, c, flip)
            end
        end
    end
    for (ci, c) in enumerate(science_cols)
        c in ac.circ && unwrap_circular!(view(cube, :, ci, :))
    end
    local rh, bess, tess
    try
        ch = MCMCChains.Chains(cube)
        er = _diag_by_param(MCMCChains.ess_rhat, ch)
        rh = er[:, :rhat]; bess = er[:, :ess]
        tess = try _diag_by_param(MCMCChains.ess, ch; kind = :tail)[:, :ess] catch; bess end
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

# ------------------------------------------------------------------
# Circular re-cut of a tempered ensemble (see src/circular.jl).
#
# Walkers live in bounded space, so a full-circle angle's prior window acts as
# a wall: a posterior centred on 0 ≡ 2π burns in as two lobes at opposite ends
# of the window, the stretch move between them proposes through the empty
# interior and is rejected, and no rung can help -- the seam is the same wall
# at every β. The window is only a chart, so move it: `recut_circular_param!`
# picks the seam from the COLD walkers and moves the layout + packed priors +
# `transforms`, then every walker at every rung is relabelled into the new
# window. Serial, between steps, no randomness.
#
# The cached logπ / logL stay valid: the prior is still uniform over one
# period (same density; only rounding in hi - lo) and the likelihood is
# periodic in these angles (`is_circular` excludes Mo when TTVs are modelled),
# so each walker's physical state and log-density are unchanged, up to
# rounding in sin/cos of the shifted representative.
#
# Returns the unfrozen positions whose window moved, so a caller holding more
# copies of the positions (pt_whitening's ring buffer) can relabel those too.
# Records each new window in `moved` (name => (lo, hi)) for the run's report.
# ------------------------------------------------------------------
function _pt_ensemble_recut!(state::Array{Float64,3}, params::Params,
                             circ::Vector{Int}, transforms,
                             moved::Dict{String,NTuple{2,Float64}})
    n_t, n_w = size(state, 1), size(state, 2)
    dims = Int[]
    for d in circ
        win = recut_circular_param!(params, d, view(state, 1, :, d);
                                    transforms = transforms)
        win === nothing && continue
        lo, hi = win
        @inbounds for w in 1:n_w, t in 1:n_t
            state[t, w, d] = circular_relabel(state[t, w, d], lo, hi)
        end
        moved[params.layout.unfrozen_names[d]] = (Float64(lo), Float64(hi))
        push!(dims, d)
    end
    return dims
end

# When to re-cut: halfway through burn-in, with every pruning pass (every 50
# steps through 3/4 of it) and at its end. The seam has to follow the mass,
# and the mass still jumps after the halfway cut: a mode found late (pruning
# moves a rung onto it, swaps carry it down) lands wherever the old seam
# happens to be. Gaia-4 b (seed 6 of 21) reached its cold rung at step 910 of
# 1500 from the far side of the seam, Mo rising through 6.1 towards 0.6 + 2π:
# the rung piled up against the wall at 2π (median 6.27, max 6.2830) for 550
# steps, and only the end-of-burn-in cut released it, so it relaxed into the
# recorded chain (R-hat 1.09, log Z 526.9 against 537.6). On this schedule the
# seam moves within 50 steps of such a pile-up, and the last cut that answers
# a jump comes n_burnin/4 before the recording starts: the same time the
# walkers moved by the last prune get to settle. A cut costs no evaluation
# and keeps the seam unless it sits in a dense arc, so it is a no-op for an
# ensemble already clear of it.
_pt_recut_due(step::Integer, n_burnin::Integer) =
    step == n_burnin ÷ 2 || step == n_burnin ||
    (step % 50 == 0 && step <= (3 * n_burnin) ÷ 4)

# ------------------------------------------------------------------
# Stranded walkers (burn-in only).
#
# A stretch move proposes along the line through two walkers and never less
# than halfway from the partner: y = x_p + z(x_w - x_p), z in [1/a, a]. A walker
# on the no-signal plateau cannot land in a narrow mode by moving, and a walker
# in the mode cannot copy itself over. Swaps only permute states between rungs.
# So a rung holds as many in-mode states as the ladder has happened to find,
# and pads the rest of its slots with plateau states that stay there.
# Measured on an easy single-planet RV target (P = 12.3 d, K = 12 m/s, 60 RVs,
# LogUniform(0.1, 3000) period prior, default 2000/1000 steps): the cold rung
# filled at ~5 walkers per 100 steps, 30-50% of the recorded draws sat on the
# plateau 70 nats down, and R-hat was 1.4-2.3.
#
# So during burn-in, move walkers that cannot matter onto walkers that do, as
# Hou et al. 2012 (ApJ 745, 198) do by clustering walkers on likelihood. What
# makes it safe to do by default is a mass bound. Per rung, A = the walkers
# within a Gaussian-tail allowance of the best tempered density, and
#     Z_A        ≈  best density · √det(2π Σ_A)             (Laplace over A)
#     Z_{L^β<c}  =  ∫_{L^β<c} π L^β  ≤  c · ∫ π  =  c       (π is normalised)
# so with log c = log Z_A - `tau`, the whole region of parameter space where
# the tempered likelihood is below c weighs under e^-tau of A, however large
# it is. Every walker in it is moved onto a random member of A. A plateau that
# COULD carry mass -- a weak signal, where K ≈ 0 over the whole prior really
# is most of the posterior -- has a likelihood within the Occam factor of the
# peak, above c, and is left alone. So are the tails of the mode itself.
# `state` is on the flat scale (see sample_pt_emcee), so Laplace is too: on
# raw P a log-uniform prior's tail alone would inflate Σ_A by ~8 nats. A
# full-circle angle enters Σ_A as its offset from the best walker on the
# circle: a mode across the seam would otherwise span the whole window and
# overstate Z_A.
#
# Σ_A needs d + 2 walkers in A. The mode a rung has just found has one to
# three: Gaia-4 b, joint RVs + DR4 (d = 14, 24 rungs), kept 97 of its 100 cold
# walkers in a local mode ~200 nats below the 1-3 that had found the real one
# at every prune of burn-in, moved none of them, and recorded 46% of its draws
# 60+ nats below the mode (R-hat 1.59). With too few walkers in A, or a
# singular Σ_A (copies left by an earlier prune), the volume comes from the
# density itself instead: the drop of ℓ along each axis about the best walker
# gives a width σ_j no larger than the conditional one, 1/√H_jj (see
# `_pt_axis_widths`). By Hadamard's inequality det H ≤ ∏ H_jj, so
# ∏ √(2π) σ_j is at most the Gaussian volume √det(2πΣ); and away from the mode
# the dropped gradient term exp(½ gᵀH⁻¹g) is ≥ 1. The estimate can only
# understate Z_A, which only lowers the cut: the guarantee above holds as it
# is. Walkers moved this way are drawn about the best walker
# with those widths and re-evaluated, not copied: copies of fewer than d + 1
# walkers would leave the rung in their affine hull, which no stretch move
# leaves.
#
# The best walker is measured at the top of its mode, not where it landed. A
# walker that has just found a mode sits on its flank, where the estimate is
# weakest twice over: its density is below the peak, and the drops along each
# axis are steep with the gradient, so the widths come out small. Gaia-4 b's
# lone cold walkers sat 235-280 nats below the peak with Σ log σ_j ≈ -57 (-30
# at the top), and the cut stayed below a plateau the mode outweighs by
# e^200+: seeds 14 and 16 (of 21) recorded it, R-hat 1.65 and 2.10, log Z 198
# and 191 against 537.6. So the best walker first climbs (`_pt_climb`) and the
# widths are measured where it stops. Any point of the mode gives an
# understatement of Z_A by the argument above, so the guarantee is unchanged;
# the cut only lands where the mode actually is.
#
# That path waits for the rung to stop climbing. Early in burn-in the walkers
# are still rising from the prior and the best of them is no mode, only the
# fastest climber; the bound holds there too, but moving the rest onto it
# herds the ensemble onto one walker. Without a check, HD 114762 with an IAD
# jitter moved 97 of its 100 cold walkers (791 across the ladder) at step 200
# of 4000 onto a leader at log p = -2709, and Gaia-4 b 39 at step 50. So it
# runs only once the rung's median log-likelihood, tempered at the current β,
# rose by less than d/2 (the depth of its typical set) since the previous
# prune: Gaia-4 b's stranded group moved by -1.5 to +1.9 nats per 50 steps, a
# climbing rung by ~50-100. The first prune has no record and never takes it.
#
# Bounding the region by one of its walkers instead (sup over the group) does
# not work: the group below any density cut always holds some tail walker of
# the mode, 20 nats down, and the bound never clears the 70-nat plateau.
#
# Burn-in only, so every recorded step is ordinary MCMC. Returns the number of
# walkers moved at each rung. Serial, draws from `rng`. `evalf(y)` returns
# (log π, log L) on the flat scale, as `logπ_arr`/`logL_arr` hold them; without
# it a rung with too few walkers in A is left alone. `circ` lists the
# full-circle angles. `med_logL` is each rung's median log L after the previous
# call, NaN before the first; it is updated in place. Without it, or while a
# rung climbs, that rung's few-walker path is skipped. `held` lists coordinates
# every walker shares while `burnin_hold` holds them (src/samplers/burnin_hold.jl):
# they carry no volume, so the bound runs over the other d - h, the climb
# leaves them where they are, and a walker drawn about the best one keeps the
# held value. The bound is then on the posterior given the held values, whose
# prior mass is 1 over the other coordinates, so the held coordinates' own
# log prior density on this scale (`held_logπ`, the same in every walker's
# log π) comes out of the best walker's density: left in, a held value where
# its prior is dense would raise the cut by as much.
# ------------------------------------------------------------------
function _pt_prune_stranded!(state::Array{Float64,3}, logπ_arr::Matrix{Float64},
                             logL_arr::Matrix{Float64}, βs::Vector{Float64}, rng;
                             tau::Float64 = 10.0, evalf = nothing,
                             circ::AbstractVector{Int} = Int[],
                             med_logL::Union{Nothing,AbstractVector{Float64}} = nothing,
                             held::AbstractVector{Int} = Int[], held_logπ::Float64 = 0.0)
    n_t, n_w, nd = size(state)
    moved = zeros(Int, n_t)
    free = Int[j for j in 1:nd if !(j in held)]
    d = length(free)
    d >= 1 || return moved
    allow = quantile(Chisq(d), 1 - 1e-6) / 2
    # How far each rung climbed since the last call, at today's β (the ladder
    # adapts in between); NaN without a record, which compares false.
    rise = med_logL === nothing ? fill(NaN, n_t) :
           [βs[t] * (_pt_finite_median(view(logL_arr, t, :)) - med_logL[t]) for t in 1:n_t]
    ℓ = Vector{Float64}(undef, n_w)
    y = Vector{Float64}(undef, nd)
    for t in 1:n_t
        β = βs[t]
        @inbounds for w in 1:n_w
            v = logπ_arr[t, w] + β * logL_arr[t, w]
            ℓ[w] = isfinite(v) ? v : -Inf
        end
        b = argmax(ℓ)
        best = ℓ[b]
        isfinite(best) || continue
        below = [w for w in 1:n_w if ℓ[w] < best - allow]
        isempty(below) && continue
        A = findall(>=(best - allow), ℓ)
        yb = state[t, b, :]
        lpb, llb = logπ_arr[t, b], logL_arr[t, b]
        lv = length(A) >= d + 2 ? _pt_half_logdet_cov(state, t, A, yb, circ, free) : nothing
        σ = nothing
        if lv === nothing
            (evalf === nothing || !(rise[t] < d / 2)) && continue
            # A cost gate, not a bound: the whole rung's spread as the volume. A
            # rung that could not prune even then is not worth 2d+ evaluations.
            gate = _pt_half_logdet_cov(state, t, findall(isfinite, ℓ), yb, circ, free)
            gate === nothing && continue
            cg = (best - held_logπ) + 0.5 * d * log(2π) + gate - tau
            any(w -> β * logL_arr[t, w] < cg, below) || continue
            ℓat = x -> (r = evalf(x); v = r[1] + β * r[2]; isfinite(v) ? v : -Inf)
            h0 = [0.1 * _pt_spread(view(state, t, :, j), yb[j], j in circ) for j in free]
            σ = _pt_axis_widths(ℓat, yb, best, h0; dims = free)
            σ === nothing && continue
            # From where the best walker landed to the top of its mode.
            yc, ℓc = _pt_climb(ℓat, yb, best, σ; dims = free)
            σc = ℓc > best ? _pt_axis_widths(ℓat, yc, ℓc, h0; dims = free) : nothing
            if σc !== nothing
                yb, best, σ = yc, ℓc, σc
                lpb, llb = evalf(yb)
                below = [w for w in 1:n_w if ℓ[w] < best - allow]
            end
            lv = sum(log, σ)
        end
        cut = (best - held_logπ) + 0.5 * d * log(2π) + lv - tau
        S = [w for w in below if !(β * logL_arr[t, w] >= cut)]
        isempty(S) && continue
        if σ === nothing
            # Donors from the upper half of A: its lower edge sits `allow` down
            # (~20 nats at d = 7), far outside a d-dim typical set (~d/2 down),
            # and a copy placed there spends the rest of burn-in relaxing back in.
            donors = A[ℓ[A] .>= median(ℓ[A])]
            @inbounds for w in S
                a = donors[rand(rng, 1:length(donors))]
                for j in 1:nd
                    state[t, w, j] = state[t, a, j]
                end
                logπ_arr[t, w] = logπ_arr[t, a]
                logL_arr[t, w] = logL_arr[t, a]
            end
        else
            for w in S
                lp, ll = _pt_scatter!(y, evalf, yb, σ, rng; dims = free)
                if !(isfinite(lp) && isfinite(ll))
                    y .= yb; lp, ll = lpb, llb
                end
                @inbounds for j in 1:nd
                    state[t, w, j] = y[j]
                end
                logπ_arr[t, w] = lp
                logL_arr[t, w] = ll
            end
        end
        moved[t] = length(S)
    end
    if med_logL !== nothing
        for t in 1:n_t
            med_logL[t] = _pt_finite_median(view(logL_arr, t, :))
        end
    end
    return moved
end

_pt_finite_median(x) = (v = filter(isfinite, x); isempty(v) ? NaN : median(v))

# ½ log det of the covariance of walkers `ws` at rung `t` over coordinates
# `dims`; a full-circle angle (`circ`) as its offset from `yb` on the circle.
# `nothing` if singular.
function _pt_half_logdet_cov(state::Array{Float64,3}, t::Int, ws, yb::AbstractVector,
                             circ::AbstractVector{Int}, dims = 1:size(state, 3))
    Y = Matrix{Float64}(undef, length(ws), length(dims))
    @inbounds for (i, w) in enumerate(ws), (k, j) in enumerate(dims)
        Y[i, k] = j in circ ? rem2pi(state[t, w, j] - yb[j], RoundNearest) :
                              state[t, w, j]
    end
    C = cholesky(Symmetric(cov(Y; dims = 1)); check = false)
    return issuccess(C) ? 0.5 * logdet(C) : nothing
end

# Standard deviation of `x` (offsets from `x0` on the circle when `circular`),
# floored so a collapsed column still gives the width search a start.
function _pt_spread(x::AbstractVector, x0::Real, circular::Bool)
    s = circular ? std(rem2pi.(x .- x0, RoundNearest)) : std(x)
    return max(isfinite(s) ? s : 0.0, 1e-6 * max(1.0, abs(x0)))
end

# Widths of the tempered density `ℓat` about `y0` (where it is `ℓ0`), one axis
# at a time, from the drop to the steeper side:
#     D(h) = ℓ0 - min(ℓ(y0 + h e_j), ℓ(y0 - h e_j)),    σ_j = h / √(2D)
# with h rescaled by 4 from `h0` until D lies in [1/8, 2] nats, the quadratic
# core clear of rounding; once an h below and one above the window are known,
# bisected geometrically between them (a wall or support edge makes D jump by
# more than the factor 16 a ×4 step allows), and if none lands in it, taken at
# the narrowest h that overshot (a smaller σ). At the mode of a Gaussian that is the conditional
# width 1/√H_jj. Away from it the gradient g_j steepens one side,
# D = |g_j| h + ½ H_jj h², so σ_j only comes out smaller: a walker still
# climbing (Gaia-4 b's had its jitters 15 nats per m/s uphill) gets a smaller
# volume, never a larger one. A side outside the support leaves the other, at
# half the width (half a Gaussian). `nothing` if an axis never settles. Only
# the axes `dims` (h0 and the widths returned are theirs, in that order).
function _pt_axis_widths(ℓat, y0::AbstractVector{Float64}, ℓ0::Float64,
                         h0::AbstractVector{Float64}; max_iter::Int = 24,
                         dims = eachindex(y0))
    σ = Vector{Float64}(undef, length(dims))
    y = collect(Float64, y0)
    for (i, j) in enumerate(dims)
        h = h0[i]
        ok = false
        lo = 0.0          # the widest h seen with the drop under 1/8 nat
        hi = Inf          # the narrowest h seen with the drop over 2 nats (or off the support)
        up = nothing      # σ read at `hi`, when its drop was finite
        for _ in 1:max_iter
            y[j] = y0[j] + h; ℓp = ℓat(y)
            y[j] = y0[j] - h; ℓm = ℓat(y)
            y[j] = y0[j]
            if !isfinite(ℓp) && !isfinite(ℓm)
                hi = min(hi, h)
                h = lo > 0 ? sqrt(lo * hi) : h / 4
                continue
            end
            half = isfinite(ℓp) && isfinite(ℓm) ? 1.0 : 0.5
            drop = ℓ0 - min(isfinite(ℓp) ? ℓp : Inf, isfinite(ℓm) ? ℓm : Inf)
            if drop > 2
                # Too steep. The drop need not grow as h²: a wall (a support
                # edge, a prior that diverges) makes it jump by more than the
                # factor 16 a ×4 step allows, so once a flat h is known, bisect
                # between the two instead of stepping back and forth.
                if h < hi
                    hi = h
                    up = half * h / sqrt(2 * drop)
                end
                h = lo > 0 ? sqrt(lo * hi) : h / 4
            elseif drop >= 0.125
                σ[i] = half * h / sqrt(2 * drop)
                ok = true
                break
            else
                lo = max(lo, h)
                h = isfinite(hi) ? sqrt(lo * hi) : h * 4   # flat, or uphill both ways
            end
            # The bracket has closed on a jump: no h gives a drop in the window.
            if lo > 0 && hi < Inf && hi <= lo * (1 + 1e-6)
                break
            end
        end
        if !ok && up !== nothing
            # No h put the drop in the window: take the width at the narrowest h
            # that overshot it. A larger drop gives a smaller σ, so this only
            # underestimates the volume.
            σ[i] = up
            ok = true
        end
        ok || return nothing
    end
    return σ
end

# From the best walker `y0` (tempered density `ℓ0`) to the top of its mode:
# BFGS in u, y = y0 + σ ⊙ u, so the axis widths `σ` set the scale, with
# finite-difference gradients. Outside the support counts as a finite cliff,
# so the line search backs off it instead of failing. A lone walker that has
# just found a mode sits on its flank, where the bound it feeds is weakest:
# Gaia-4 b's sat 235-280 nats below the peak (cold rung), still climbing.
# At most `max_evals` evaluations; stops when an iteration gains less than
# `f_tol`. Deterministic. Returns (y, ℓ), or (y0, ℓ0) if nothing was better.
# Only the axes `dims` move (σ is theirs, in that order): a coordinate
# `burnin_hold` holds stays at y0's value, which every walker shares.
function _pt_climb(ℓat, y0::AbstractVector{Float64}, ℓ0::Float64,
                   σ::Vector{Float64}; dims = eachindex(y0),
                   max_evals::Int = 500 * length(dims), f_tol::Float64 = 1e-2)
    n = Ref(0)
    cliff = 1e8 - ℓ0
    function at(u)
        y = collect(Float64, y0)
        @inbounds for (i, j) in enumerate(dims)
            y[j] = y0[j] + σ[i] * u[i]
        end
        return y
    end
    f = u -> (n[] += 1; v = ℓat(at(u)); isfinite(v) ? -v : cliff)
    res = Optim.optimize(f, zeros(length(dims)), Optim.BFGS(),
                         Optim.Options(iterations = 10_000, f_abstol = f_tol,
                                       callback = _ -> n[] >= max_evals))
    yc = at(Optim.minimizer(res))
    ℓc = ℓat(yc)
    return isfinite(ℓc) && ℓc > ℓ0 ? (yc, ℓc) : (collect(Float64, y0), ℓ0)
end

# One draw y = yb + σ ⊙ z, z ~ N(0, I), redrawn (up to 20 times) until it is in
# the support. Returns `evalf(y)`; (-Inf, -Inf) if no draw was. Only the axes
# `dims` move (σ is theirs, in that order); the rest stay at yb.
function _pt_scatter!(y::Vector{Float64}, evalf, yb::AbstractVector{Float64},
                      σ::Vector{Float64}, rng; dims = eachindex(y))
    copyto!(y, yb)
    for _ in 1:20
        @inbounds for (i, j) in enumerate(dims)
            y[j] = yb[j] + σ[i] * randn(rng)
        end
        lp, ll = evalf(y)
        isfinite(lp) && isfinite(ll) && return (lp, ll)
    end
    return (-Inf, -Inf)
end

# One line per run, after the progress bar: which angles were re-charted and
# where their windows ended up. A silent move would leave the reader of a trace
# plot wondering why Mo now spans [-3.05, 3.23) when they wrote U(0, 2π).
function _pt_recut_info(sampler::AbstractString,
                        moved::Dict{String,NTuple{2,Float64}})
    isempty(moved) && return nothing
    r(x) = round(x; digits = 2)
    nms = sort!(collect(keys(moved)))
    msg = if length(nms) == 1
        lo, hi = moved[only(nms)]
        "moved the seam of $(only(nms)) to the emptiest arc of its posterior; " *
        "window now [$(r(lo)), $(r(hi)))"
    else
        "moved the seams of $(join(nms, ", ", " and ")) to the emptiest arc of " *
        "each posterior; windows now " *
        join(("$nm [$(r(moved[nm][1])), $(r(moved[nm][2])))" for nm in nms), ", ")
    end
    @info "$sampler: $msg. Still one full period under the same uniform prior, " *
          "so this relabels the circle and changes neither the posterior nor " *
          "the evidence."
    return nothing
end

"""
    sample_pt_emcee(target, data; kwargs...) -> PTemceeResult

Parallel-tempered affine-invariant ensemble MCMC (Vousden+ 2016).
Walkers operate in bounded space; a proposal outside the prior support is
rejected. Full-circle angles are the exception: their window is re-charted
during burn-in so the seam sits in the emptiest arc of the posterior (see
src/circular.jl), which moves `target.params.layout` and `target.transform`
in place.

# Keywords
- `n_temps::Int=16` — temperature levels. Was 5, which is slower AND worse:
  on the HD 114762 joint target 5 temps gave min swap acceptance 0.003 in
  5.1 min, 16 gave 0.014 in 2.0 min, 24 gave 0.019 in 2.5 min. Better mixing
  reaches the convergence checks sooner, so a denser ladder pays for itself.
  Joint astrometry targets want 24 — at 16 with one seed the eccentricity
  collapsed to 0.002 against a published 0.335, silently.
- `n_walkers::Int=100` — walkers per temperature (≥ `2·n_dim + 2`, even).
- `n_steps::Int=3000` — steps per walker.
- `convergence_stop::Bool=false` — end production once the gating parameters
  (the planet block and `rho_s`, or `science_params`) have worst split-R̂ below
  `rhat_threshold` and worst tail-ESS above `tail_ess_threshold` for
  `n_converged_checks` consecutive diagnostics, no earlier than `min_steps`.
  The diagnostics run every `diag_every` steps (default `max(200, n_steps ÷ 40)`)
  and also drive the live `Rhat` mean/worst and `ESS` readout, whether or not
  the gate is on. The draws kept so far are the result; `stop.stopped` is
  `"converged"`.
- `rhat_threshold::Real=1.01`, `tail_ess_threshold::Real=1000`,
  `n_converged_checks::Int=3`, `min_steps::Int=0`, `diag_every::Int=0` — the
  gate's settings, see `convergence_stop`.
- `max_seconds::Real=Inf`, `max_evals::Real=Inf` — end production gracefully
  once this much wall-clock time has passed since the sampler started, or this
  many likelihood evaluations have been made. Reached during burn-in, the run
  fails with an error (there is nothing to keep yet).
- `stop_file=nothing` — path of a file whose appearance asks the run to stop.
  Looked for about every 5 s of wall clock (and at every diagnostic), never per
  step. Once burn-in is over the run ends at the last recorded step and
  finishes normally with `stop.stopped == "requested"`; before that it fails
  with an error. The file is never deleted. `run_job` watches
  `<output_dir>/STOP` unless the job's `stop_file` says otherwise.
- An `InterruptException` (Ctrl-C) after burn-in ends production the same way,
  with `stop.stopped == "interrupted"`; the row being written when it arrives
  is dropped. During burn-in, or a second interrupt while the run is being
  wound up, it aborts as before.
- `n_burnin::Int=2000` — burn-in discarded from posterior + evidence.
  Circular angles are re-charted at steps `n_burnin ÷ 2` and `n_burnin`, and
  every 50 steps through 3/4 of burn-in with the pruning (`_pt_recut_due`).
  Was 2000/1000, which is too short to find a well-measured period on a
  wide log-uniform prior and settle on it: on an easy 12.3 d RV target over
  LogUniform(0.1, 3000), 1 of 8 seed/data combinations failed R-hat at
  2000/1000 and several sat near 1.04; at 3000/2000 all 8 passed with
  worst R-hat ≤ 1.010 and ESS ≥ 17k.
- `betas::Union{Nothing,Vector{Float64}}=nothing` — explicit β ladder
  (descending β[1]=1). Default: Vines+ 2023's `β_i = (1/√5)^i`.
- `stretch_a::Float64=2.0` — Goodman-Weare stretch parameter.
- `de_move::Real=DE_MOVE_DEFAULT` (0.5) — per walker and step, the probability
  of a differential-evolution move, x' = x + γ(x_j − x_k) + ε with j ≠ k from
  the half the stretch partner comes from, in place of the stretch move (ter
  Braak 2006; src/samplers/de_move.jl). The proposal is symmetric, so it is
  accepted on the tempered posterior ratio alone. γ is 2.38/√(2m) for the m
  coordinates moved, jittered by 10 %, and 1 (a jump between modes) one time in
  ten. The stretch move proposes along the chord through two walkers, which
  leaves a thin curved ridge almost always, so a share of walkers that burn-in
  leaves on such a ridge stays there: on the default transit-only fit one seed
  in six kept 47 % of its draws at e > 0.5 on the constant-duration ridge where
  nested sampling puts 1.4 %, with R-hat 1.0002. `0` is the stretch-only
  kernel, and draws exactly what it did.
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
  Anything else is an `ArgumentError`. Ignored when `init` is given.
- `pf_n_runs::Int=2` — Pathfinder L-BFGS basins (only used if
  `:pathfinder`).
- `seed::Int=1`
- `thin::Int=1` — thinning factor on β=1 posterior output.
- `show_progress::Bool=true` — display the per-step progress bar.
- `swap_min::Real=0.05` — the least swap acceptance between adjacent rungs for
  the tempered evidence to be quotable (with a ladder that reaches β = 0); one
  that is not counts as broken, and mode-Laplace takes over. See
  `log_evidence` under [`PTemceeResult`](@ref).
- `laplace_switch_tol::Real=50.0` — the Laplace non-Gaussian bias, in nats: a
  quotable tempered value further than this from mode-Laplace is broken, and
  mode-Laplace takes over.
- `prune_stranded::Bool=true` — every 50 steps through the first three quarters
  of burn-in, move walkers stranded far below their rung's mode onto walkers in
  it. Only walkers whose likelihood is so low that the whole region below it
  weighs under e⁻¹⁰ of the mode (bounded by that likelihood times the
  normalised prior) are moved, so a plateau that could carry real mass — a
  weak signal — is left alone. The mode's mass is the Laplace volume of the
  walkers in it; while fewer than `n_dim + 2` have found it, and once the rung
  has stopped climbing, a smaller volume measured from the density along each
  axis at the top of the mode, which the best walker first climbs to (BFGS, at
  most 500 evaluations per dimension), and the walkers moved are then drawn
  about that point rather than copied, so the rung keeps its full rank. The
  stretch move cannot bring stranded walkers back itself, so without this a
  narrow mode (any well-measured period on a wide prior) fills its cold rung at
  the rate the ladder happens to rediscover it. Burn-in only: the recorded chain is ordinary
  MCMC. `false` restores the old behaviour.
- `node_flip::Real=0.1` — per walker and step, the probability of proposing
  the mirror (Ω, ω) → (Ω + π, ω + π) of each astrometry-only planet, accepted
  with the tempered Metropolis-Hastings ratio (see src/samplers/node_flip.jl).
  The stretch move cannot cross between those two exactly degenerate
  solutions, and swaps alone left Gaia-4 at 61/39 where the symmetry demands
  50/50, with R-hat 1.003. Costs one likelihood evaluation per proposal; never
  offered to a planet with RV, transit or any other data. `0` disables it.
- `burnin_hold=nothing` — parameters held at a fixed value, the same for every
  walker of every rung, for at least the first `burnin_hold_frac` of burn-in
  and until the β = 1 rung stops climbing (3/4 of burn-in at the latest), and
  then drawn from their priors walker by walker. `nothing` (or `true`) holds every
  IAD jitter `sigma_iad_<instrument>` at 0 (`Nereus.default_burnin_hold`):
  from the prior, inflating the jitter until the reflex stops mattering is the
  cheapest way up the likelihood, and on HD 114762 RV + Gaia DR4 every run
  ended there, 470-970 nats below an orbit no walker ever found. At 0 the
  likelihood is the unjittered one, so the held phase is the fit without the
  jitter, which finds the orbit. `false` holds nothing; a dictionary
  `name => value` holds those instead. Burn-in only: the recorded chain is
  ordinary MCMC. See src/samplers/burnin_hold.jl.
- `burnin_hold_frac::Real=0.5` — the fraction of `n_burnin` the hold lasts at
  least. From then on it ends at the first 50-step pass at which the β = 1
  rung's median log L rose by less than d/2 (the prune's test for a rung that
  has stopped climbing), and at 3/4 of burn-in whatever it does: released while
  the fit without the jitter still climbs, the jitter takes up the misfit and
  the orbit is lost again (`_burnin_hold_due!`).
- `checkpoint=nothing` — path of a state file. The sampler writes its whole
  state there (every rung's walkers, the ladder, every RNG stream, the counters,
  the ladder history, the evidence accumulators and the kept draws) at the end
  of the run and every `checkpoint_interval` seconds during it, replacing the
  file atomically. `run_job` sets it to `pt_emcee_state.jls` in `output_dir`
  unless the job gives one.
- `checkpoint_interval::Real=900` — seconds between checkpoints during the run.
- `resume::Bool=false` — continue from `checkpoint` instead of initialising:
  a run that was killed, or a finished one that needs more steps. `n_steps` is
  the new TOTAL, counted from the start of the original run; the draws already
  kept stay, and the new ones are appended. The continuation is bit-identical to
  an uninterrupted run of `n_steps`, at any thread count. Refused, with the
  differences listed, if the checkpoint came from different data, priors,
  parameters, walkers, rungs, burn-in, thinning, seed or move settings.
"""
function sample_pt_emcee(
    target::NereusTarget,
    data::Data;
    n_temps::Int = 16,
    n_walkers::Int = 100,
    n_steps::Int = 3000,
    n_burnin::Int = 2000,
    betas::Union{Nothing,AbstractVector} = nothing,
    beta_min::Real = 1e-4,
    stretch_a::Real = 2.0,
    de_move::Real = DE_MOVE_DEFAULT,
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
    max_seconds::Real = Inf,
    max_evals::Real = Inf,
    stop_file::Union{Nothing,AbstractString} = nothing,
    adapt_ladder::Bool = false,
    ladder_adapt_window::Int = 50,
    ladder_adapt_ν0::Real = 10.0,
    ladder_adapt_K::Real = 1.0,
    laplace_switch_tol::Real = 50.0,
    swap_min::Real = 0.05,
    bridge_headline::Bool = true,
    bridge_n::Int = 20_000,
    bridge_warn_tol::Real = 20.0,
    untemper_transit::Bool = false,
    prune_stranded::Bool = true,
    node_flip::Real = 0.1,
    burnin_hold = nothing,
    burnin_hold_frac::Real = 0.5,
    checkpoint::Union{Nothing,AbstractString,Symbol} = nothing,
    checkpoint_interval::Real = 900.0,
    resume::Bool = false,
)
    # JSON delivers floats-as-Int and arrays-as-JSON3.Array; normalize.
    stretch_a       = Float64(stretch_a)
    # The DE share (src/samplers/de_move.jl).
    p_de            = _de_check(de_move)
    node_flip       = Float64(node_flip)
    0 <= node_flip <= 1 || throw(ArgumentError(
        "node_flip is a probability; got $node_flip"))
    # Any other Symbol used to fall through to the prior draw without a word:
    # `init_strategy = "map"` from Python ran exactly the `:prior` fit.
    init_strategy in (:prior, :pathfinder, :map_scatter) || throw(ArgumentError(
        "init_strategy must be :prior (default), :pathfinder or :map_scatter; " *
        "got :$init_strategy"))
    ladder_adapt_ν0 = Float64(ladder_adapt_ν0)
    ladder_adapt_K  = Float64(ladder_adapt_K)
    ck_path = checkpoint === nothing ? nothing : String(checkpoint)
    resume && ck_path === nothing && throw(ArgumentError(
        "resume = true needs `checkpoint`, the state file to continue from"))
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
    # loop. The transform is irrelevant for pt_emcee: M-H acceptance
    # only cares about target density ratios in whatever space the
    # walkers live, and bounded space gives clean auto-reject behavior
    # via log_prior(theta) = -Inf outside support.
    #
    # Bounded, but on the scale each prior is flat on: a LogUniform or
    # ModJeffreys parameter moves as y = log(x + s) (`log_scale_shift`). The
    # Jacobian |dx/dy| = x + s cancels the prior's 1/(x + s) exactly, so inside
    # the box the prior is as flat as a Uniform's and the box is still a hard
    # wall -- none of the soft-wall gradient the logit transform had. What it
    # buys: a log-uniform period prior over 0.1-3000 d scatters walkers over
    # 4.5 decades, and a linear stretch between walkers at 12 d and 1500 d
    # throws the short one out to hundreds of days. On an easy 12.3 d RV target
    # the cold rung had not found the period by the end of the default burn-in.
    # `logπ_arr` holds the prior density on this scale (log p(x) + Σ y_j over
    # the log dimensions); draws and :lp are mapped back to x when recorded.
    #
    # Full-circle and polar angles are the exception: re-charted during
    # burn-in by `_pt_ensemble_recut!`, on `_pt_recut_due`'s schedule. `seam`
    # (both kinds) and `circ` (full-circle, which the pruning measures on the
    # circle) are computed once; `recut_moved` collects the final windows for
    # the single end-of-run report.
    circ        = circular_indices(params)
    seam        = seam_indices(params)
    recut_moved = Dict{String,NTuple{2,Float64}}()
    # Cyclic inclinations: also folded to [0, π] on the same schedule.
    cinc        = cyclic_inclination_indices(params)
    # The move scale per dimension (see above), and the pruning tally per rung.
    flat_shift = Union{Nothing,Float64}[log_scale_shift(ps)
                                        for ps in layout.unfrozen_priors]
    pruned = zeros(Int, n_temps)
    prune_med = fill(NaN, n_temps)      # each rung's median log L at the last prune
    # Astrometry-only planets get the (Ω, ω) → (Ω + π, ω + π) move
    # (src/samplers/node_flip.jl); empty, and never drawn for, anything else.
    flips = node_flip > 0 ? node_flips(params, data; flat_shift) : NodeFlip[]
    # Coordinates held early in burn-in (src/samplers/burnin_hold.jl). While
    # held, `held_x` pins them in `eval_bounded!` at their exact value: on the
    # flat scale exp(log(x + s)) - s need not round back to x, and a jitter
    # held at 0 must take the unjittered path. Emptied at the release.
    hold = _burnin_hold(params, burnin_hold, burnin_hold_frac, n_burnin;
                        forbid = reduce(vcat, (vcat(f.shift, f.negate) for f in flips);
                                        init = Int[]))
    held_x = Tuple{Int,Float64}[(unfrozen_idx[j], v) for (j, v) in zip(hold.dims, hold.values)]
    held_y = Float64[flat_shift[j] === nothing ? v : log(v + flat_shift[j])
                     for (j, v) in zip(hold.dims, hold.values)]
    held_lp = _burnin_hold_logπ(hold, params; flat_shift)
    hold_med = Ref(NaN)                 # the β = 1 median log L at the last pass

    # Goodman-Weare: each half ≥ n_dim+1 walkers, total even.
    # One assignment: the half-step closures capture it, and a variable
    # assigned twice is boxed there, so every walker's partner range was
    # typed Any.
    n_walkers_eff = let nw = max(n_walkers, 2 * n_dim + 2)
        nw + isodd(nw)
    end

    # Geometric ladder across [beta_min, 1], matching sample_transdim_pt_emcee.
    #
    # This used to be hardcoded (1/sqrt(5))^i with no beta_min, which puts the
    # hottest rung at 5^(-(n_temps-1)/2) — 0.04 at the default 5 temps. TI and
    # TI+ integrate <logL> only over [beta_min, 1] and ss_plus telescopes
    # Z(beta_min)->Z(1); neither adds the [0, beta_min] head, so that head was
    # silently dropped. Measured on the HD 18599 joint fit it is worth ~715 nats
    # between 5 and 10 rungs — an evidence error far larger than any model
    # comparison it would be used for.
    #
    # sample_transdim_pt_emcee already took beta_min and built the ladder this
    # way; the fixed-dim sampler did not even accept the argument, so the two
    # returned incomparable evidences for the same problem.
    βs = betas === nothing ?
         Float64[Float64(beta_min)^(i / max(1, n_temps - 1)) for i in 0:(n_temps - 1)] :
         copy(betas)
    length(βs) == n_temps || throw(ArgumentError(
        "Provided `betas` has length $(length(βs)), expected $n_temps"))

    # --- Checkpoint / resume (src/checkpoint.jl) ------------------------
    # The fingerprint takes the ladder as built, before any adaptation. On resume
    # the saved state replaces initialisation below and is put back just before
    # the main loop, which then carries on from step `step0 + 1`.
    # `de_move` enters only when non-zero: a checkpoint written before the move
    # existed ran the stretch-only kernel, which is `de_move = 0`.
    ck_fp = run_fingerprint(params, data; n_temps, n_walkers = n_walkers_eff,
        n_burnin, thin, seed, stretch_a, ladder = copy(βs), adapt_ladder,
        ladder_adapt_window, ladder_adapt_ν0, ladder_adapt_K, untemper_transit,
        prune_stranded, node_flip, _burnin_hold_fp(hold, params)...,
        _de_fp(p_de)...)
    ck = resume ? read_checkpoint(ck_path, "pt_emcee", ck_fp) : nothing
    step0 = ck === nothing ? 0 : ck.step::Int
    step0 <= n_steps || throw(ArgumentError(
        "the checkpoint is at step $step0; n_steps = $n_steps would end before it"))

    # --- Per-slot mutable state -----------------------------------------
    # Keyed by CHUNK, never by `Threads.threadid()`: the id spans every
    # threadpool while `Threads.nthreads()` counts only the default one, so
    # `thread_theta[threadid()]` overran this vector on stock Julia >= 1.12
    # (see src/threading.jl).
    n_slots = _nthread_chunks()
    thread_theta    = [Theta{Float64}(params) for _ in 1:n_slots]
    # Per-slot PTWorkspace → ws-aware likelihood: no per-call Vector allocs
    # (the ~27.7 MB/eval GC thrash on data-rich fits) + per-planet flux cache +
    # total phot-ll cache. pt_emcee is the workhorse recoverer, so this speeds up
    # the whole fixed-dim menu. One ws per slot, indexed by the chunk index.
    thread_ws       = [PTWorkspace(params, params.config.max_kplanet,
                                   length(params.config.noise_models);
                                   n_obs = length(data.t_rv), n_phot = length(data.t_phot))
                       for _ in 1:n_slots]
    thread_xb       = [Vector{Float64}(undef, n_dim) for _ in 1:n_slots]
    thread_proposal = [Vector{Float64}(undef, n_dim) for _ in 1:n_slots]
    rng_master      = MersenneTwister(seed)

    # Evaluate (log_prior_bounded, log_like) at a BOUNDED-space point
    # `x` using the slot's own Theta. Returns (-Inf, -Inf) if x is
    # outside the prior support — caller treats this as an automatic
    # M-H reject. log_like includes external priors (eccentricity,
    # rho_s) per Nereus convention, so these get tempered too —
    # matches the in-house PT path's convention.
    @inline function eval_bounded!(y::AbstractVector{Float64}, slot::Int)
        theta = thread_theta[slot]
        wb = thread_ws[slot]
        log_jac = 0.0
        @inbounds for (j, idx) in enumerate(unfrozen_idx)
            s = flat_shift[j]
            if s === nothing
                theta.values[idx] = y[j]
            else
                theta.values[idx] = exp(y[j]) - s
                log_jac += y[j]
            end
        end
        @inbounds for (idx, v) in held_x
            theta.values[idx] = v
        end
        lp = log_prior(theta) + log_jac
        isfinite(lp) || return (-Inf, -Inf)
        # The residual maps (Doppler tomography) are data: tempered with the RV.
        ll_rv = rv_log_likelihood(theta, data, wb) + tomogram_log_likelihood(theta, data, wb)
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
    pf_draws = if ck === nothing && init === nothing && init_strategy === :pathfinder
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
            @warn "Pathfinder warm-start failed; falling back to prior draws" exception=(e, catch_backtrace())
            nothing
        end
    elseif ck === nothing && init === nothing && init_strategy === :map_scatter
        # MAP point in BOUNDED space + per-parameter scatter sized to
        # 1% of the prior range (for uniform priors) or 1·σ (for normal
        # priors). This keeps walkers in a tight in-support cluster
        # around the MAP — stretch moves between near-neighbors stay
        # in-box, acceptance is high.
        n_pf_draws = n_temps * n_walkers_eff
        map_res = sample_map(target; method = :LBFGS, maxiter = 500,
                             g_tol = 1e-6, n_starts = max(2, pf_n_runs))
        # Bounded space, circular angles in the layout's current windows
        # (`sample_map` may have moved them; see src/circular.jl).
        x_map = circular_relabel_point!(copy(map_res.x_map), params)
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
            # A circular scatter wraps instead of leaving the window: each
            # walker retries its own column below, so an out-of-window one
            # would never start in support.
            circular_relabel_point!(view(pf_d, :, j), params)
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
    # A caller's `init` is written in the user's window; a target reused after
    # a fit carries moved circular windows, where Mo = 6.2 can sit outside
    # [-3.05, 3.23) and fail every retry below. Relabel a copy into them.
    x_init = init === nothing ? nothing : circular_relabel_point!(copy(init), params)
    init_seeds = rand(rng_master, UInt64, n_temps * n_walkers_eff)
    # Buffers are keyed by CHUNK, never by `Threads.threadid()` (see
    # src/threading.jl); `:static` keeps the 1:1 chunk-to-thread mapping, but
    # the buffers no longer depend on it.
    # On resume no walker is drawn: the loop gets no chunks, and the saved
    # walkers and rng_master (whose draw of `init_seeds` above is undone by the
    # restore) are put back before the main loop.
    init_chunks = _chunk_ranges(ck === nothing ? n_temps * n_walkers_eff : 0, n_slots)
    Threads.@threads :static for slot in 1:length(init_chunks)
        for task_idx in init_chunks[slot]
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
                elseif x_init !== nothing
                    @inbounds for d in 1:n_dim
                        state[t, w, d] = x_init[d] + 1e-3 * randn(rng)
                    end
                else
                    x_b = _draw_from_prior(target, rng)
                    @inbounds for d in 1:n_dim
                        state[t, w, d] = x_b[d]
                    end
                end
                # Every branch above fills x; walkers move on the flat scale.
                @inbounds for d in 1:n_dim
                    s = flat_shift[d]
                    s === nothing && continue
                    x = state[t, w, d]
                    state[t, w, d] = x + s > 0 ? log(x + s) : -Inf
                end
                # Every walker of every rung starts at the held value.
                @inbounds for (d, yv) in zip(hold.dims, held_y)
                    state[t, w, d] = yv
                end
                lp, ll = eval_bounded!(@view(state[t, w, :]), slot)
                if isfinite(lp) && isfinite(ll)
                    logπ_arr[t, w] = lp
                    logL_arr[t, w] = ll
                    break
                end
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
    # The DE move's own tally per rung (`de_move`); the two above count the
    # stretch move alone, as they always have.
    de_proposed    = zeros(Int, n_temps)
    de_accepted    = zeros(Int, n_temps)
    de_prop_temp   = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    de_acc_temp    = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    # ε's scale along each coordinate, per rung: the spread of the half the
    # partners come from, refreshed before each half-step (src/samplers/de_move.jl).
    de_scale       = zeros(n_temps, n_dim)
    is_held_dim    = falses(n_dim)
    is_held_dim[hold.dims] .= true
    no_held_dim    = falses(n_dim)
    # Ladder history (src/samplers/ladder_history.jl): the ladder, the swap
    # acceptance and the swap mean distance at every step, where the two
    # counters above only keep the run totals.
    β_hist    = Matrix{Float64}(undef, n_steps, n_temps)
    swap_hist = zeros(Float64, n_steps, n_temps - 1)
    smd_hist  = zeros(Float64, n_steps, n_temps - 1)
    smd_scale = _swap_distance_scale(layout.unfrozen_priors, flat_shift)
    n_done    = 0
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
    # Chunk ranges for the half-step loops below, computed once since
    # `tasks_h1`/`tasks_h2` are fixed for the whole run (not a per-step
    # allocation). Keyed by CHUNK, never by `Threads.threadid()` — same
    # reasoning as the init loop above (see src/threading.jl).
    chunks_h1 = _chunk_ranges(length(tasks_h1), n_slots)
    chunks_h2 = _chunk_ranges(length(tasks_h2), n_slots)
    # The half-step uses the chunk count only, for its number of slots; the
    # walkers themselves are handed out through this counter.
    next_task = Threads.Atomic{Int}(1)

    # One RNG per WALKER SLOT, not per thread. Task->thread assignment
    # changes with `-t`, so drawing from a per-thread stream makes the
    # chain depend on the thread count at fixed seed (measured: K differs
    # in the 4th decimal between -t 2 and -t 4). Keying the stream on
    # task_idx makes a run bit-identical across thread counts. A stretch
    # update consumes three draws (partner, stretch u, accept), a DE update
    # its own fixed sequence, and the choice between them one more (none at
    # `de_move` = 0); all from the walker's stream, so reusing the object
    # across sweeps stays deterministic.
    rngs_h1 = [MersenneTwister(_walker_seed(seed, 1, i)) for i in 1:length(tasks_h1)]
    rngs_h2 = [MersenneTwister(_walker_seed(seed, 2, i)) for i in 1:length(tasks_h2)]
    # Node flip: one stream per (rung, walker), same reasoning. Built only when
    # a planet has the move, so every other fit draws exactly what it did.
    rngs_flip = isempty(flips) ? MersenneTwister[] :
        [MersenneTwister(_walker_seed(seed, 5, i)) for i in 1:(n_temps * n_walkers_eff)]
    flip_prop_temp = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    flip_acc_temp  = [Threads.Atomic{Int}(0) for _ in 1:n_temps]
    flip_proposed  = zeros(Int, n_temps)
    flip_accepted  = zeros(Int, n_temps)
    flip_eval(buf, t, w, slot) = eval_bounded!(buf, slot)
    # The pruning's own evaluations (`_pt_prune_stranded!`), counted.
    prune_eval(y) = (Threads.atomic_add!(n_evals_atomic, 1); eval_bounded!(y, 1))

    function do_half_step!(tasks::Vector{Tuple{Int,Int}}, active_half::Symbol)
        partner_lo = active_half === :h1 ? half + 1 : 1
        partner_hi = active_half === :h1 ? n_walkers_eff : half
        task_rngs  = active_half === :h1 ? rngs_h1 : rngs_h2
        chunks     = active_half === :h1 ? chunks_h1 : chunks_h2
        # Every thread has a slot of walkers: the likelihood's own threaded
        # loops would only queue tasks behind the other slots (src/threading.jl).
        fills_threads = length(chunks) >= Threads.nthreads()
        n_tasks = length(tasks)
        next_task[] = 1
        # The dimensions the stretch scales: a held coordinate is shared by the
        # whole ensemble, so the move leaves it and the Jacobian is z^(m - 1).
        n_move = n_dim - length(held_x)
        held_dims = isempty(held_x) ? no_held_dim : is_held_dim
        p_de > 0 && _de_spread!(de_scale, state, partner_lo, partner_hi)
        Threads.@threads :static for slot in 1:length(chunks)
            fills_threads && _serial_inner_loops!()
            # Each slot takes the next walker as it frees up, not a fixed block:
            # `tasks` runs cold to hot and a walker's cost differs between rungs,
            # so fixed blocks left threads idle at the end of every half-step
            # (NGTS-33, 90 threads: threads 1-12 0.9 busy, the last 50 0.45). The
            # chain stays bit-identical: a walker's draws come from its own RNG,
            # and it reads only the other half, which no task writes this pass.
            while true
                task_idx = Threads.atomic_add!(next_task, 1)
                task_idx > n_tasks && break
                t, w = tasks[task_idx]
                β = βs[t]
                trng = task_rngs[task_idx]
                buf  = thread_proposal[slot]

                if _de_chosen(trng, p_de)
                    # Differential evolution (src/samplers/de_move.jl): on the
                    # same flat-scale, charted coordinates as the stretch, and
                    # leaving the held ones as it does. Symmetric given the
                    # other half: no Jacobian.
                    j, k = _de_pair(trng, partner_lo, partner_hi)
                    γ = _de_gamma(trng, _de_gamma0(n_move))
                    _de_fill!(buf, state, t, w, j, k, γ, de_scale, held_dims, trng)
                    lp_prop, ll_prop = eval_bounded!(buf, slot)
                    log_ratio = β * (ll_prop - logL_arr[t, w]) +
                                (lp_prop - logπ_arr[t, w])
                    Threads.atomic_add!(de_prop_temp[t], 1)
                    if log(rand(trng)) < log_ratio
                        @inbounds for d in 1:n_dim
                            state[t, w, d] = buf[d]
                        end
                        logπ_arr[t, w] = lp_prop
                        logL_arr[t, w] = ll_prop
                        Threads.atomic_add!(de_acc_temp[t], 1)
                    end
                    continue
                end

                w_partner = rand(trng, partner_lo:partner_hi)
                u = rand(trng)
                z = ((stretch_a - 1) * u + 1)^2 / stretch_a

                @inbounds for d in 1:n_dim
                    buf[d] = state[t, w_partner, d] +
                             z * (state[t, w, d] - state[t, w_partner, d])
                end

                lp_prop, ll_prop = eval_bounded!(buf, slot)
                log_ratio = (n_move - 1) * log(z) +
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
        end
        Threads.atomic_add!(n_evals_atomic, length(tasks))
    end

    # The end of `burnin_hold` (src/samplers/burnin_hold.jl): every walker of
    # every rung draws its held coordinates from their priors and is evaluated
    # there. A walker none of 100 draws gives a finite density for keeps the
    # held value. Threaded like the init; each walker's stream is seeded from
    # rng_master, so the run does not depend on the thread count. The prune's
    # climbing record restarts: the release moves every rung's log L at once.
    function release_hold!()
        empty!(held_x)
        n_tot = n_temps * n_walkers_eff
        rel_seeds = rand(rng_master, UInt64, n_tot)
        rel_chunks = _chunk_ranges(n_tot, n_slots)
        Threads.@threads :static for slot in 1:length(rel_chunks)
            buf = thread_proposal[slot]
            for task_idx in rel_chunks[slot]
                t = (task_idx - 1) ÷ n_walkers_eff + 1
                w = (task_idx - 1) % n_walkers_eff + 1
                rng = MersenneTwister(rel_seeds[task_idx])
                for _ in 1:100
                    @inbounds for d in 1:n_dim
                        buf[d] = state[t, w, d]
                    end
                    _burnin_hold_draw!(buf, hold, layout.unfrozen_priors, rng; flat_shift)
                    lp, ll = eval_bounded!(buf, slot)
                    Threads.atomic_add!(n_evals_atomic, 1)
                    if isfinite(lp) && isfinite(ll)
                        @inbounds for d in hold.dims
                            state[t, w, d] = buf[d]
                        end
                        logπ_arr[t, w] = lp
                        logL_arr[t, w] = ll
                        break
                    end
                end
            end
        end
        prune_med .= NaN
        return nothing
    end

    # --- Progress bar (per-step accept rate + ETA) --------------------
    pb = ProgressBar("pt_emcee"; total = n_steps, enabled = show_progress,
                     start = step0)

    # --- Live convergence readout + optional run-until-converged ------
    rhat_str = "—"; ess_str = "—"      # cached "mean/worst" displays
    conv_run = 0; converged_at = 0     # consecutive passing checks; stop step
    sc = StopControl(; max_seconds, max_evals, stop_file)
    stop_reason = "steps"; stop_step = 0   # how the loop ended; last step kept
    cur_step = 0; step_keep0 = 0           # the step in flight; keep_idx before it

    # --- Resume: put the saved run back exactly as it stopped ----------
    # In place throughout: the closures above captured these arrays, and
    # rebinding a captured variable would box it. A checkpoint that stopped on
    # the convergence gate starts with the gate open again (`converged_at` = 0):
    # resuming means more steps were asked for.
    if ck !== nothing
        state .= ck.state; logπ_arr .= ck.logπ; logL_arr .= ck.logL; βs .= ck.betas
        copy!(rng_master, ck.rng_master)
        foreach(copy!, rngs_h1, ck.rngs_h1); foreach(copy!, rngs_h2, ck.rngs_h2)
        foreach(copy!, rngs_flip, ck.rngs_flip)
        accept_within .= ck.accept_within; propose_within .= ck.propose_within
        accept_swap .= ck.accept_swap; propose_swap .= ck.propose_swap
        flip_proposed .= ck.flip_proposed; flip_accepted .= ck.flip_accepted
        hasproperty(ck, :de_proposed) && (de_proposed .= ck.de_proposed;
                                          de_accepted .= ck.de_accepted)
        pruned .= ck.pruned
        hasproperty(ck, :prune_med) && (prune_med .= ck.prune_med)
        n_evals_atomic[] = ck.n_evals::Int
        β_hist[1:step0, :] .= ck.β_hist; swap_hist[1:step0, :] .= ck.swap_hist
        smd_hist[1:step0, :] .= ck.smd_hist
        keep_idx = ck.keep_idx::Int
        samples[1:keep_idx, :] .= ck.samples; lp_samples[1:keep_idx] .= ck.lp_samples
        for f in fieldnames(EvidenceAccumulator)
            setfield!(evidence_acc, f, getfield(ck.evidence, f))
        end
        conv_run = ck.conv_run::Int; rhat_str = ck.rhat_str::String
        ess_str = ck.ess_str::String
        # Windows the saved walkers live in (moved by the burn-in re-cut).
        for (nm, (lo, _)) in ck.windows
            set_circular_window!(params, findfirst(==(nm), layout.unfrozen_names),
                                 lo; transforms = (target.transform,))
        end
        merge!(recut_moved, ck.recut_moved)
        # The spin-flip axis the saved walkers were canonicalised by (src/circular.jl).
        hasproperty(ck, :spin_axis) && (params.layout.spin_axis .= ck.spin_axis)
        n_done = step0
        # Past the release, the saved walkers carry their own values.
        if hasproperty(ck, :hold_on)
            ck.hold_on || empty!(held_x)
            hold_med[] = ck.hold_med
        else
            step0 >= hold.step && empty!(held_x)
        end
    end
    last_ck = time()
    show_progress && !isempty(held_x) &&
        _burnin_hold_info("pt_emcee", hold, params, n_burnin)

    # --- Main loop ----------------------------------------------------
    # An InterruptException after burn-in ends production at the last step that
    # was fully recorded; the step it landed in is dropped whole (`keep_idx`
    # goes back to its value at the top of that step). Before burn-in is over,
    # or with nothing recorded, it propagates and the run aborts.
    try
    for step in (step0 + 1):n_steps
        cur_step = step; step_keep0 = keep_idx
        do_half_step!(tasks_h1, :h1)
        do_half_step!(tasks_h2, :h2)

        @inbounds for t in 1:n_temps
            propose_within[t] += Threads.atomic_xchg!(propose_temp[t], 0)
            accept_within[t]  += Threads.atomic_xchg!(accept_temp[t],  0)
            de_proposed[t]    += Threads.atomic_xchg!(de_prop_temp[t], 0)
            de_accepted[t]    += Threads.atomic_xchg!(de_acc_temp[t],  0)
        end

        # ---- Node flip (threaded; astrometry-only planets) -----------
        if !isempty(flips)
            Threads.atomic_add!(n_evals_atomic,
                _node_flip_sweep!(flip_eval, state, logπ_arr, logL_arr, βs, flips,
                                  params, node_flip, rngs_flip, thread_proposal,
                                  flip_prop_temp, flip_acc_temp))
            @inbounds for t in 1:n_temps
                flip_proposed[t] += Threads.atomic_xchg!(flip_prop_temp[t], 0)
                flip_accepted[t] += Threads.atomic_xchg!(flip_acc_temp[t],  0)
            end
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
                    smd_hist[step, t] += _swap_distance(state, t, w, w2, smd_scale)
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
            # A rejected proposal moved nothing: it counts as zero distance.
            swap_hist[step, t]  = window_accs[t] / window_props[t]
            smd_hist[step, t]  /= window_props[t]
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
        β_hist[step, :] .= βs
        n_done = step

        # ---- Cyclic inclinations (serial; burn-in only) -------------
        # The walkers are folded back to [0, π] (src/circular.jl), every
        # walker at every rung, on the re-cut schedule, so the stretch moves
        # pair walkers in one representative; the re-cut below then moves the
        # chart's seam clear of the folded draws and their mirror images.
        # Before the pruning, which measures each rung's spread. Exact for the
        # cached logπ / logL, as the re-cut is.
        # Under the stellar spin flip (`spin_flip`) the walkers are also put in
        # one representative of the spin, the axis chosen from the cold rung;
        # before the re-cut, which then cuts λ's seam from the moved draws.
        !isempty(cinc) && _pt_recut_due(step, n_burnin) &&
            fold_cyclic_inclinations!(state, params; idx = cinc,
                                      decide_from = view(state, 1, :, :))

        # ---- Stranded walkers (serial; burn-in only) -----------------
        # See `_pt_prune_stranded!`. Stops at 3/4 of burn-in so the moved
        # walkers decorrelate from the ones they were copied onto, and runs
        # before the re-cut so the seam is chosen from the cleaned cold rung.
        # Its evaluations run here, between steps, on slot 1's buffers.
        if prune_stranded && step % 50 == 0 && step <= (3 * n_burnin) ÷ 4
            pruned .+= _pt_prune_stranded!(state, logπ_arr, logL_arr, βs,
                                           rng_master; evalf = prune_eval, circ,
                                           med_logL = prune_med,
                                           held = isempty(held_x) ? Int[] : hold.dims,
                                           held_logπ = isempty(held_x) ? 0.0 : held_lp)
        end

        # ---- Circular re-cut (serial; threads idle) ------------------
        # Halfway through burn-in, while the ensemble can still merge the
        # two lobes a seam-centred angle burns in as; with every pruning
        # pass, so a mode found later is not held behind the seam (see
        # `_pt_recut_due`); and at the end of burn-in, which fixes the chart
        # for every recorded step (usually a no-op by then: circular_cut
        # keeps the current seam unless another arc is clearly emptier).
        # Relabels `state` at every rung; logπ_arr and logL_arr stay valid,
        # see `_pt_ensemble_recut!`.
        if !isempty(seam) && _pt_recut_due(step, n_burnin)
            _pt_ensemble_recut!(state, params, seam, (target.transform,),
                                recut_moved)
        end

        # ---- End of `burnin_hold` (threaded; within burn-in) -----------
        !isempty(held_x) && _burnin_hold_due!(hold_med, hold, step, n_burnin,
                                              view(logL_arr, 1, :),
                                              n_dim - length(hold.dims)) &&
            release_hold!()

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
                    # Back from the flat scale to x, and :lp with it.
                    log_jac = 0.0
                    for d in 1:n_dim
                        s = flat_shift[d]
                        if s === nothing
                            samples[keep_idx, d] = state[1, w, d]
                        else
                            samples[keep_idx, d] = exp(state[1, w, d]) - s
                            log_jac += state[1, w, d]
                        end
                    end
                    lp_samples[keep_idx] = logπ_arr[1, w] - log_jac + logL_arr[1, w]
                end
            end
        end

        # ---- Periodic convergence diagnostic (β=1 science params) ----
        # Mean/worst R-hat and bulk/tail ESS over the gating set; drives both
        # the live readout and (optionally) the run-until-converged stop.
        if step > n_burnin && (step - n_burnin) % diag_cadence == 0
            diag = _pt_convergence(samples, keep_idx, n_walkers_eff, science_cols;
                                   canon = AngleCanon(params))
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
                              # the DE move's at β = 1, when it is on
                              (p_de > 0 ? (:β1_de => round(de_accepted[1] /
                                  max(de_proposed[1], 1), digits = 3),) : ())...,
                              # The LADDER's health, not the move's. This is the
                              # half that fails silently: once the rungs stop
                              # exchanging, the cold chain is stuck in one mode
                              # and R-hat happily certifies convergence to it.
                              # The docstring has warned about a tiny
                              # minimum(acceptance_swap) since the sampler was
                              # written; the counters were right here and the
                              # readout never showed them.
                              # One rung (n_temps = 1) has no swaps: NaN, not
                              # `minimum` of an empty vector, which threw.
                              :min_swap => isempty(accept_swap) ? NaN :
                                  round(minimum(accept_swap ./ max.(propose_swap, 1)),
                                        digits = 3),
                              :Rhat => rhat_str,    # mean/worst over science params
                              :ESS => ess_str,      # mean(bulk)/worst(tail)
                              :nevals => n_evals_atomic[]))
        end

        # ---- Checkpoint: every `checkpoint_interval` s, on the last step, and
        # before an early stop (src/checkpoint.jl) -------------------------
        # ---- Out-of-band stops: stop file, wall-clock and evaluation limits.
        # The file is looked for every few seconds (and at every diagnostic),
        # never per step. During burn-in there is nothing to keep: an error.
        _step_hook!(step, sc)
        stop_now = _stop_check!(sc, n_evals_atomic[];
                    force = step > n_burnin && (step - n_burnin) % diag_cadence == 0)
        if stop_now !== nothing
            step <= n_burnin && throw(_stop_in_burnin_error(
                "pt_emcee", stop_now, sc, step, n_burnin))
            keep_idx > 0 || (stop_now = nothing)   # `thin`: no row recorded yet
        end

        if ck_path !== nothing && (step == n_steps || (convergence_stop && converged_at > 0) ||
                                   stop_now !== nothing ||
                                   time() - last_ck >= checkpoint_interval)
            write_checkpoint(ck_path, "pt_emcee", ck_fp, (; step,
                state, logπ = logπ_arr, logL = logL_arr, betas = βs, rng_master,
                rngs_h1, rngs_h2, rngs_flip, accept_within, propose_within,
                accept_swap, propose_swap, flip_proposed, flip_accepted, pruned,
                prune_med, de_proposed, de_accepted,
                n_evals = n_evals_atomic[], β_hist = β_hist[1:step, :],
                swap_hist = swap_hist[1:step, :], smd_hist = smd_hist[1:step, :],
                keep_idx, samples = samples[1:keep_idx, :],
                lp_samples = lp_samples[1:keep_idx], evidence = evidence_acc,
                conv_run, rhat_str, ess_str, windows = circular_windows(params),
                spin_axis = copy(params.layout.spin_axis),
                recut_moved, hold_on = !isempty(held_x), hold_med = hold_med[]))
            last_ck = time()
        end

        # ---- Run-until-converged: stop once the science params clear the
        # gate for `n_converged_checks` consecutive diagnostics ----------
        if convergence_stop && converged_at > 0
            show_progress && @info "pt_emcee: science params converged at step $converged_at " *
                "(Rhat<$(rhat_threshold), tail-ESS>$(tail_ess_threshold)); stopping early"
            stop_reason = "converged"; stop_step = step
            break
        end
        if stop_now !== nothing
            stop_reason = stop_now; stop_step = step
            break
        end
    end
    catch err
        err isa InterruptException && cur_step > n_burnin && step_keep0 > 0 ||
            rethrow()
        keep_idx = step_keep0
        n_done = cur_step - 1
        stop_reason = "interrupted"; stop_step = cur_step - 1
        _drain_workers!(next_task)
    end
    stop_reason == "steps" && (stop_step = n_done)
    show_progress && finish!(pb)
    stop_reason == "steps" ||
        _announce_stop(stop_reason, stop_step, n_steps, keep_idx)
    show_progress && _pt_recut_info("pt_emcee", recut_moved)
    show_progress && _node_flip_info("pt_emcee", flips, params, flip_proposed,
                                     flip_accepted)
    show_progress && sum(pruned) > 0 && @info "pt_emcee: burn-in moved " *
        "$(sum(pruned)) stranded walker state(s) onto their rung's mode " *
        "($(pruned[1]) at β = 1), all from a region whose total posterior mass " *
        "is bounded below e⁻¹⁰ of the mode's. The recorded chain is ordinary MCMC."
    if convergence_stop && converged_at == 0 && stop_reason == "steps"
        @warn "pt_emcee: hit max_steps=$n_steps WITHOUT meeting the convergence gate " *
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
    # use Laplace, and warn. Bridge, when it is usable, comes first, and a
    # tempered value from a ladder that cannot carry it (`_tempered_quotable`)
    # is broken too (HEADLINE SELECTION below).
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

    # Bridge sampling from the cold chain we already have. Touches no hot rung,
    # and assumes only that the fitted reference OVERLAPS the posterior -- not
    # that the posterior is Gaussian about its mode. Requires an unconstrained
    # target, as the proposal is Gaussian on R^n.
    #
    # Costs `bridge_n` likelihood evaluations: the posterior at every kept
    # draw is its `:lp` (log prior + log L in the bounded space) plus the
    # transform's log Jacobian, checked against a full evaluation at a few
    # draws. Without that it would be `n_kept + bridge_n` evaluations, which
    # single-threaded was 42 s against 89 s for the entire MCMC on a Gaia DR4
    # IAD fit.
    #
    # NOTE: on the fit_* entry points this is computed AGAIN, post-fit, by
    # `_augment_evidence!` (src/runner.jl) -- which exists because run_job's
    # targets are bounded and never get here. Same estimator, same chains, so
    # the second call is redundant on this path; it is cheap now, but it is
    # still two passes over the posterior.
    log_z_bridge = NaN
    bridge_overlap = NaN
    bridge_se = NaN
    if bridge_headline && target.transform !== nothing
        try
            b = bridge_evidence(target, chains; n_proposal = bridge_n, seed = seed)
            if b.converged && isfinite(b.log_z)
                log_z_bridge = b.log_z
                bridge_overlap = b.overlap
                bridge_se = b.se
            end
        catch err
            @debug "sample_pt_emcee: bridge evidence unavailable" exception = err
        end
    end

    # Bridge is reported only from draws it can be trusted on (`_bridge_guard`):
    # a chain that has converged (worst R-hat at most 1.1), and a reference `q`
    # fitted from enough INDEPENDENT points (least ESS at least the
    # n_dim + n_dim(n_dim+1)/2 parameters of q). Either way it fails silently
    # LOW -- `overlap` reads ~1 throughout -- by fractions of a nat to tens:
    # on a curved 15-D target with exactly known log Z bridge fell to -3.2 nats
    # at n_eff ~ 22 while reference-path held to +0.17, and 32 walkers that had
    # not found an easy RV planet's mode read 70 to 90 nats low at n_eff above
    # the bound. Well-mixed chains are unaffected: 51 Peg runs at n_eff ~ 8000
    # against 135 parameters.
    bridge_refused = ""
    if isfinite(log_z_bridge)
        g = _bridge_guard(chains, target.params)
        if !g.ok
            # reference-path recovers from a poorly estimated q (within 0.11
            # nats on stationary chains at d <= 15 where bridge was 0.2 to 2
            # low), but not from draws that never reached the posterior.
            alt = occursin("R-hat", g.why) ? "Sample longer." :
                  "Sample longer, or use `reference_path_evidence`, which " *
                  "anneals away from q and holds where bridge does not."
            @warn "sample_pt_emcee: dropping bridge as the headline: $(g.why). " *
                  "Bridge is biased LOW here and none of its diagnostics detect " *
                  "it. $alt" bridge = log_z_bridge min_ess = g.n_eff rhat = g.rhat n_params_in_q = g.need
            bridge_refused = g.why
            log_z_bridge = NaN
        end
    end

    # HEADLINE SELECTION.
    #
    # Ordered by what each estimator has actually been measured to do, not by
    # what is cheapest. On 51 Peg (1691 RVs, 15 params, unimodal, P determined
    # to 5 decimals), converged (two seeds of 15 000 steps on the quickstart's
    # 8 rungs, R-hat 1.016 and 1.014), the spread was:
    #
    #   TI+ / H+  -6005.59 / -5986.97              93 to 112 nats low
    #   SS+       -5959.52 / -5948.62              55 to 66 nats low
    #   Laplace   -5888.33 / -5888.39              5.5 nats high
    #   bridge    -5893.86 / -5893.91              validated pair
    #   refpath   -5893.79 / -5893.88
    #
    # The beta-path stack fails at the phase transition and cannot detect it --
    # TI+ and H+ agreed exactly (H+ is TI+ when it finds no crossover) while
    # both were ~100 nats out (that ladder stops at beta_min, and the
    # [0, beta_min] head it leaves out would only lower them further) -- and on
    # an RV + Hipparcos-IAD companion (bridge 163.88, matched by pt_hmc's
    # bridge, Daedalus and population annealing) SS+ read 157.85 from 8 rungs.
    #
    # So, in order: bridge when it is usable. Else the tempered value when its
    # ladder can carry it (`_tempered_quotable`: the ladder reaches β = 0 and
    # every adjacent pair swapped at least `swap_min` of the time, which the
    # default ladder, geometric down to beta_min, never does) and it agrees with
    # mode-Laplace within `laplace_switch_tol`. Else mode-Laplace, with a
    # warning (Jose's rule above). No number only when none of them exists.
    # Under untemper_transit + photometry there is no switch (above).
    tempered_ok, tempered_why = _tempered_quotable(βs, acc_swap; swap_min)
    if tempered_ok && !isfinite(log_z)
        tempered_ok, tempered_why = false, "it is not finite"
    end
    log_z_tempered = log_z
    laplace_ok = isfinite(log_z_laplace)
    gap = abs(log_z_tempered - log_z_laplace)
    min_swap = isempty(acc_swap) ? NaN : minimum(acc_swap)
    why_bridge = !bridge_headline ? "bridge_headline = false" :
                 target.transform === nothing ? "the target is bounded " *
                     "(unconstrained = false); run_job computes bridge after the fit" :
                 !isempty(bridge_refused) ? "refused, $bridge_refused" :
                 "bridge sampling did not give a usable value"
    switched = false

    if isfinite(log_z_bridge)
        # Corroborate rather than assume. Bridge cannot self-detect a posterior
        # its single reference fails to cover -- `overlap` counts draws landing
        # in SUPPORT, not mode coverage, and read 0.99 on a posterior with 23
        # period modes. A large bridge/Laplace gap is the cheapest signal that
        # something is off; past `bridge_warn_tol` the warning says so.
        _warn_laplace_vs_bridge("sample_pt_emcee", log_z_laplace, log_z_bridge,
                                bridge_se; multimodal_tol = bridge_warn_tol)
        log_z = log_z_bridge
    elseif !laplace_check
        # Relative tempered value, absolute Laplace: no switch, both kept.
        tempered_ok || @warn "sample_pt_emcee: the tempered log-evidence is not " *
              "quotable: $tempered_why. Under untemper_transit with photometry " *
              "it is relative to the transit-conditioned reference and " *
              "mode-Laplace is absolute, so Laplace does not replace it; both " *
              "are kept." tempered = log_z_tempered laplace = log_z_laplace
        log_z = log_z_tempered
    elseif tempered_ok && laplace_ok && gap <= laplace_switch_tol
        log_z = log_z_tempered
    elseif laplace_ok
        why_t = tempered_ok ?
            "it is $(round(gap, digits = 1)) nats from mode-Laplace, beyond the " *
            "Laplace non-Gaussian bias (laplace_switch_tol = $laplace_switch_tol): " *
            "the phase-transition signature (min swap accept = " *
            "$(round(min_swap, digits = 3)))" :
            "it is not quotable: $tempered_why"
        @warn "sample_pt_emcee: log_evidence is mode-Laplace, a Gaussian " *
              "approximation at the posterior mode. Bridge: $why_bridge. The " *
              "tempered value is broken: $why_t. Treat it as indicative: on 51 " *
              "Peg, unimodal and converged, it sits 5.5 nats above bridge." laplace = log_z_laplace tempered = log_z_tempered
        log_z = log_z_laplace
        switched = true
    elseif tempered_ok
        @warn "sample_pt_emcee: neither bridge (" * why_bridge * ") nor " *
              "mode-Laplace is available. log_evidence is the tempered value; " *
              "its ladder can carry it, but nothing corroborates it." tempered = log_z_tempered
        log_z = log_z_tempered
    else
        @warn "sample_pt_emcee: no usable evidence estimate for this run. " *
              "Bridge: $why_bridge. mode-Laplace is not available. The tempered " *
              "stack is not quotable: $tempered_why. log_evidence is NaN."
        log_z = NaN
    end

    # Independent of which was chosen: a large tempered/headline gap is the
    # phase-transition signature and worth surfacing, because the tempered
    # numbers are still in `evidence` and someone will read them.
    if !switched && isfinite(log_z_tempered) && isfinite(log_z) && laplace_check &&
       abs(log_z_tempered - log_z) > laplace_switch_tol
        @warn "sample_pt_emcee: the TEMPERED stack (TI/TI+/SS+/H+ in `evidence`) " *
              "is $(round(abs(log_z_tempered - log_z), digits=1)) nats from the " *
              "reported log_evidence — the phase-transition signature (min swap " *
              "accept = $(round(min_swap, digits=3))). Those four share one " *
              "mean_logL array, so their agreeing with EACH OTHER is not a " *
              "check." tempered = log_z_tempered reported = log_z
    end

    # Collective drift of the cold ensemble, which R-hat cannot see (see
    # `ensemble_drift`); warns when it finds one.
    drift = _pt_drift(samples, keep_idx, n_walkers_eff, params; sampler = "pt_emcee",
                      de_move = p_de)

    return PTemceeResult(
        chains, log_z, log_z_laplace, log_z_bridge, ev_report, acc_within,
        _de_acceptance(de_accepted, de_proposed), acc_swap,
        βs, n_evals_atomic[],
        _ladder_history(β_hist, swap_hist, smd_hist, evidence_acc, n_done, n_burnin),
        _stop_record(stop_reason, stop_step, n_steps, keep_idx, sc), drift
    )
end
