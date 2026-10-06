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
#   and at the end of burn-in (`_pt_ensemble_recut!`), so the chart is
#   fixed for every recorded step.
# - **Node flip** for astrometry-only planets: a tempered M-H move to the
#   mirror (Ω + π, ω + π), the exactly degenerate solution the stretch move
#   cannot reach (src/samplers/node_flip.jl).
# - **Flat parallelism** across (temperature, half-ensemble walker)
#   pairs in each half-step — n_temps × n_walkers/2 independent tasks.
# - **Per-thread Theta, RNG, and proposal buffers** — race-free, no
#   per-step allocation.
# - **reddemcee evidence stack**: feeds EvidenceAccumulator for
#   TI+ / SS+ / H+ in addition to vanilla trapezoidal TI.
# - **Progress bar**: Nereus's `ProgressBar` with acceptance, the ladder's
#   swap rates, the cold log-posterior, R-hat/ESS naming the worst science
#   parameter (over a rolling window during burn-in) and ETA; and an optional
#   JSON-lines progress log of the same, one line per check.
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
using Statistics: cov, quantile, median
using LinearAlgebra: cholesky, logdet, Symmetric, norm, I, issuccess
import Dates, JSON3

export sample_pt_emcee, PTemceeResult, mode_laplace_evidence

"""
    PTemceeResult

Container for a `sample_pt_emcee` run.

# Fields
- `chains::MCMCChains.Chains` — β=1 walkers, post-burnin, flattened, in bounded space
  (circular angles in the window re-charted at the end of burn-in, so contiguous).
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
  it never touches the prior end and makes no Gaussianity assumption. It costs
  `n_kept + bridge_n` likelihood evaluations, NOT `bridge_n` — it re-evaluates
  the posterior at every kept draw, so the price scales with the chain
  (170k evaluations at the default 100 walkers × 1500 post-burnin steps).
  Threaded, so it is seconds rather than a minute. NaN if not computable.
- `evidence::EvidenceReport` — full TI / TI+ / SS+ / H+ tempered stack (raw).
- `acceptance_within::Vector{Float64}` — within-temp stretch acceptance per temp.
- `acceptance_swap::Vector{Float64}` — swap acceptance between (k, k+1). A tiny
  `minimum(acceptance_swap)` (≪0.1) that won't rise with more temps signals a
  phase transition → the tempered evidence is untrustworthy (use the Laplace one).
- `betas::Vector{Float64}` — β ladder used.
- `n_evals::Int` — total likelihood evaluations.
- `ladder::LadderHistory` — the ladder, the swap acceptance and the swap mean
  distance at every step, and `⟨log L⟩` per rung: what [`plot_ladder_rates`](@ref)
  and [`plot_beta_ladder`](@ref) draw. See [`LadderHistory`](@ref).
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
    ladder::LadderHistory
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
# Returns, over the `science_cols` columns of the flat `samples` buffer, the
# per-parameter split R-hat, bulk ESS and tail ESS (NaN where undefined) and
# their summaries: mean/max R-hat, mean bulk ESS and min tail ESS over the
# finite values, with the position in `science_cols` of the worst R-hat and of
# the worst tail ESS (0 when none is finite). `nothing` when there are too few
# post-burnin steps. Steps are thinned to ≤ `cap` so the cost stays flat as the
# chain grows. The gate (convergence_stop) is on the WORST science param (max
# R-hat, min tail-ESS); nuisances are excluded so a poorly-constrained
# jitter/GP hyperparameter can never stall the run — and a nuisance that's
# coupled to a science param surfaces in that param's R-hat.
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
    return _pt_cube_diag(cube)
end

# The diagnostics of a (step, parameter, walker) cube; see `_pt_convergence`.
function _pt_cube_diag(cube::Array{Float64,3})
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
    _asvec(x) = Float64[isfinite(v) ? v : NaN for v in (x isa AbstractArray ? vec(x) : [x])]
    rh, bess, tess = _asvec(rh), _asvec(bess), _asvec(tess)
    # Worst = max R-hat / min tail ESS over the finite values; NaN never wins.
    i_r = 0; i_t = 0
    for i in eachindex(rh)
        isfinite(rh[i]) && (i_r == 0 || rh[i] > rh[i_r]) && (i_r = i)
        isfinite(tess[i]) && (i_t == 0 || tess[i] < tess[i_t]) && (i_t = i)
    end
    i_r == 0 && return nothing
    bf = filter(isfinite, bess)
    return (; rhat = rh, bulk_ess = bess, tail_ess = tess,
              mean_rhat = mean(filter(isfinite, rh)), max_rhat = rh[i_r], worst_rhat = i_r,
              mean_bulk = isempty(bf) ? 0.0 : mean(bf),
              min_tail = i_t == 0 ? 0.0 : tess[i_t], worst_tail = i_t)
end

# The same over the burn-in window (see sample_pt_emcee), a ring of
# (row, parameter, walker) with `count` rows filled and, once it is full, the
# oldest at `next`; in time order, as split R-hat compares the two halves.
function _pt_ring_diag(ring::Array{Float64,3}, next::Int, count::Int)
    count < 4 && return nothing
    n = size(ring, 1)
    order = count < n ? (1:count) : vcat(next:n, 1:(next - 1))
    return _pt_cube_diag(ring[order, :, :])
end

# The readout of a diagnostic: "mean/worst(name)" for R-hat and
# "mean bulk/worst tail(name)" for ESS, e.g. 1.004/1.031(K_k2) and 8100/950(P_k1).
function _pt_diag_strings(diag, names::AbstractVector{<:AbstractString})
    rn = diag.worst_rhat > 0 ? "(" * names[diag.worst_rhat] * ")" : ""
    tn = diag.worst_tail > 0 ? "(" * names[diag.worst_tail] * ")" : ""
    return (@sprintf("%.3f/%.3f%s", diag.mean_rhat, diag.max_rhat, rn),
            string(round(Int, diag.mean_bulk), "/", round(Int, diag.min_tail), tn))
end

# The ladder's swap acceptance so far: per adjacent pair, mean and min over the
# pairs, and the colder rung of the worst pair (0 with one rung, which has no
# swaps).
function _pt_swap_summary(accept_swap::Vector{Float64}, propose_swap::Vector{Int})
    rates = accept_swap ./ max.(propose_swap, 1)
    isempty(rates) && return (; rates, mean = NaN, min = NaN, worst = 0)
    k = argmin(rates)
    return (; rates, mean = mean(rates), min = rates[k], worst = k)
end

# Median and max over the walkers of the cold (β = 1) log-posterior, on the
# scale the draws are recorded on (the :lp column).
function _pt_cold_lp(state::Array{Float64,3}, logπ_arr::Matrix{Float64},
                     logL_arr::Matrix{Float64},
                     flat_shift::Vector{Union{Nothing,Float64}})
    n_w = size(state, 2)
    lp = Vector{Float64}(undef, n_w)
    @inbounds for w in 1:n_w
        log_jac = 0.0
        for d in eachindex(flat_shift)
            flat_shift[d] === nothing || (log_jac += state[1, w, d])
        end
        lp[w] = logπ_arr[1, w] - log_jac + logL_arr[1, w]
    end
    return (median(lp), maximum(lp))
end

# Compact readouts for the progress line.
_pt_lp_str(x::Real) = !isfinite(x) ? string(x) :
                      abs(x) >= 1e4 ? @sprintf("%.0f", x) : @sprintf("%.1f", x)
_pt_count_str(n::Integer) = n < 10_000 ? string(n) :
                            n < 10^6 ? @sprintf("%.1fk", n / 1e3) :
                            n < 10^9 ? @sprintf("%.1fM", n / 1e6) : @sprintf("%.2fG", n / 1e9)
_pt_swap_str(s) = s.worst == 0 ? "—" :
                  @sprintf("%.3f/%.3f(%d-%d)", s.mean, s.min, s.worst, s.worst + 1)

# ------------------------------------------------------------------
# The progress log (`progress_log` in sample_pt_emcee): one JSON object per
# line, per diagnostic check and one at the end, so a long run leaves a
# history that can be read back while it runs. Each line goes out as one
# append of the whole line, so a reader polling the file never sees half of
# one. Logging must never cost the run: any failure warns once and the run
# carries on.
# ------------------------------------------------------------------
function _pt_log_warn(path::String, err, warned::Base.RefValue{Bool})
    warned[] || @warn "sample_pt_emcee: cannot write the progress log; the run " *
                      "continues without it" path exception = err
    warned[] = true
    return nothing
end

# A fresh run starts the file over. A resumed one keeps the lines up to the
# step it resumes from, dropping any that a killed run wrote after its last
# checkpoint (and a torn last line), so the steps in the file only increase.
function _pt_log_start!(path::String, resume::Bool, step0::Int,
                        warned::Base.RefValue{Bool})
    try
        mkpath(dirname(abspath(path)))
        kept = String[]
        if resume && isfile(path)
            for ln in eachline(path)
                rec = try JSON3.read(ln) catch; nothing end
                rec isa JSON3.Object && get(rec, :step, typemax(Int)) <= step0 &&
                    push!(kept, ln)
            end
        end
        tmp = path * ".tmp"
        open(io -> foreach(ln -> println(io, ln), kept), tmp, "w")
        mv(tmp, path; force = true)
    catch err
        _pt_log_warn(path, err, warned)
    end
    return nothing
end

# The [first, last] step the burn-in window covers at `step`.
function _pt_burn_window(step::Int, stride::Int, count::Int)
    count == 0 && return nothing
    hi = (step ÷ stride) * stride
    return [hi - (count - 1) * stride, hi]
end

# Builds the line from `kw` (see `_pt_progress_record`) inside the guard too.
function _pt_log_line(path::String, warned::Base.RefValue{Bool}; kw...)
    try
        line = JSON3.write(_pt_progress_record(; kw...)) * "\n"
        open(io -> write(io, line), path, "a")
    catch err
        _pt_log_warn(path, err, warned)
    end
    return nothing
end

# One line of the log. `diag` is a `_pt_cube_diag` result or `nothing`, and
# `window` the [first, last] step it covers: the burn-in window during burn-in,
# every kept draw after it. Non-finite numbers are written as null.
function _pt_progress_record(; step::Int, n_steps::Int, n_burnin::Int, final::Bool,
                             elapsed::Float64, rate::Float64, n_evals::Int,
                             accept_within, propose_within, accept_swap, propose_swap,
                             lp::NTuple{2,Float64}, diag, window,
                             names::AbstractVector{<:AbstractString})
    r(x, d) = isfinite(x) ? round(x; digits = d) : nothing
    per(f, d) = diag === nothing ? nothing :
        NamedTuple{Tuple(Symbol.(names))}(Tuple(r(x, d) for x in getfield(diag, f)))
    sumd(f, d) = diag === nothing ? nothing : r(getfield(diag, f), d)
    worst(f) = diag === nothing || getfield(diag, f) == 0 ? nothing :
               names[getfield(diag, f)]
    sw = _pt_swap_summary(accept_swap, propose_swap)
    n_prop = sum(propose_within)
    return (; time = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z",
              step, n_steps, n_burnin, phase = step <= n_burnin ? "burn" : "sample",
              final, elapsed_s = r(elapsed, 1),
              eta_s = final ? 0.0 : rate > 0 ? r((n_steps - step) / rate, 1) : nothing,
              steps_per_s = r(rate, 4), n_evals,
              acc = n_prop > 0 ? r(sum(accept_within) / n_prop, 4) : nothing,
              beta1_acc = r(accept_within[1] / max(propose_within[1], 1), 4),
              swap_acc = [r(x, 4) for x in sw.rates],
              swap_mean = r(sw.mean, 4), swap_min = r(sw.min, 4),
              swap_worst_pair = sw.worst == 0 ? nothing : [sw.worst, sw.worst + 1],
              lp_median = r(lp[1], 3), lp_max = r(lp[2], 3),
              diag_window = window,
              rhat_mean = sumd(:mean_rhat, 5), rhat_max = sumd(:max_rhat, 5),
              rhat_worst = worst(:worst_rhat),
              bulk_ess_mean = sumd(:mean_bulk, 1), tail_ess_min = sumd(:min_tail, 1),
              tail_ess_worst = worst(:worst_tail),
              rhat = per(:rhat, 5), bulk_ess = per(:bulk_ess, 1),
              tail_ess = per(:tail_ess, 1))
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
# raw P a log-uniform prior's tail alone would inflate Σ_A by ~8 nats.
#
# Bounding the region by one of its walkers instead (sup over the group) does
# not work: the group below any density cut always holds some tail walker of
# the mode, 20 nats down, and the bound never clears the 70-nat plateau.
#
# Burn-in only, so every recorded step is ordinary MCMC. Returns the number of
# walkers moved at each rung. Serial, draws from `rng`.
# ------------------------------------------------------------------
function _pt_prune_stranded!(state::Array{Float64,3}, logπ_arr::Matrix{Float64},
                             logL_arr::Matrix{Float64}, βs::Vector{Float64}, rng;
                             tau::Float64 = 10.0)
    n_t, n_w, d = size(state)
    moved = zeros(Int, n_t)
    d >= 1 || return moved
    allow = quantile(Chisq(d), 1 - 1e-6) / 2
    ℓ = Vector{Float64}(undef, n_w)
    for t in 1:n_t
        β = βs[t]
        @inbounds for w in 1:n_w
            v = logπ_arr[t, w] + β * logL_arr[t, w]
            ℓ[w] = isfinite(v) ? v : -Inf
        end
        best = maximum(ℓ)
        isfinite(best) || continue
        A = findall(>=(best - allow), ℓ)
        length(A) >= d + 2 || continue
        C = cholesky(Symmetric(cov(view(state, t, A, :); dims = 1)); check = false)
        issuccess(C) || continue
        cut = best + 0.5 * d * log(2π) + 0.5 * logdet(C) - tau
        S = [w for w in 1:n_w if !(β * logL_arr[t, w] >= cut) && ℓ[w] < best - allow]
        isempty(S) && continue
        # Donors from the upper half of A: its lower edge sits `allow` down
        # (~20 nats at d = 7), far outside a d-dim typical set (~d/2 down), and
        # a copy placed there spends the rest of burn-in relaxing back in.
        donors = A[ℓ[A] .>= median(ℓ[A])]
        @inbounds for w in S
            a = donors[rand(rng, 1:length(donors))]
            for j in 1:d
                state[t, w, j] = state[t, a, j]
            end
            logπ_arr[t, w] = logπ_arr[t, a]
            logL_arr[t, w] = logL_arr[t, a]
        end
        moved[t] = length(S)
    end
    return moved
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
- `n_burnin::Int=2000` — burn-in discarded from posterior + evidence.
  Circular angles are re-charted at steps `n_burnin ÷ 2` and `n_burnin`.
  Was 2000/1000, which is too short to find a well-measured period on a
  wide log-uniform prior and settle on it: on an easy 12.3 d RV target over
  LogUniform(0.1, 3000), 1 of 8 seed/data combinations failed R-hat at
  2000/1000 and several sat near 1.04; at 3000/2000 all 8 passed with
  worst R-hat ≤ 1.010 and ESS ≥ 17k.
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
- `show_progress::Bool=true` — display the per-step progress bar. After the
  bar: `acc` (stretch acceptance, all rungs) and `β1_acc` (cold rung);
  `swap=mean/min(k-(k+1))`, the adjacent-pair swap acceptance over the ladder
  and the pair with the lowest; `lp=median/max`, the cold rung's log-posterior
  over the walkers; `Rhat=mean/worst(name)` and `ESS=mean bulk/worst
  tail(name)` over the science parameters (`science_params`; default the
  planet parameters and rho_s) from the kept draws, every `diag_every` steps
  (default `max(200, n_steps ÷ 40)`). Until the first such check they come from
  a rolling window of the last `diag_every` burn-in steps of the cold rung and
  read `Rhat(burn)=` / `ESS(burn)=`: whether the walkers agree with each other
  now, not a statement about the posterior, and never used by
  `convergence_stop`.
- `prune_stranded::Bool=true` — every 50 steps through the first three quarters
  of burn-in, move walkers stranded far below their rung's mode onto walkers in
  it. Only walkers whose likelihood is so low that the whole region below it
  weighs under e⁻¹⁰ of the mode (bounded by that likelihood times the
  normalised prior) are moved, so a plateau that could carry real mass — a
  weak signal — is left alone. The stretch move cannot bring stranded walkers
  back itself, so without this a narrow mode (any well-measured period on a
  wide prior) fills its cold rung at the rate the ladder happens to rediscover
  it. Burn-in only: the recorded chain is ordinary MCMC. `false` restores the
  old behaviour.
- `node_flip::Real=0.1` — per walker and step, the probability of proposing
  the mirror (Ω, ω) → (Ω + π, ω + π) of each astrometry-only planet, accepted
  with the tempered Metropolis-Hastings ratio (see src/samplers/node_flip.jl).
  The stretch move cannot cross between those two exactly degenerate
  solutions, and swaps alone left Gaia-4 at 61/39 where the symmetry demands
  50/50, with R-hat 1.003. Costs one likelihood evaluation per proposal; never
  offered to a planet with RV, transit or any other data. `0` disables it.
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
- `progress_log=nothing` — path of a progress log, so a long run leaves a
  history that can be read while it runs (the progress line is overwritten in
  place). One JSON object per line, at every diagnostic check of the readout
  above (burn-in included) and one at the end (`"final": true`), with `step`,
  `n_steps`, `n_burnin`, `phase` (`"burn"`/`"sample"`), `time` (UTC),
  `elapsed_s`, `eta_s`, `steps_per_s`, `n_evals`, `acc`, `beta1_acc`,
  `swap_acc` (per adjacent pair, cold first), `swap_mean`, `swap_min`,
  `swap_worst_pair`, `lp_median`, `lp_max`, and per science parameter `rhat`,
  `bulk_ess` and `tail_ess` (name => value) with `rhat_mean`, `rhat_max`,
  `rhat_worst`, `bulk_ess_mean`, `tail_ess_min` and `tail_ess_worst`.
  `diag_window` is the [first, last] step those cover: the rolling window
  during burn-in, every kept draw after it. Non-finite values are null; the
  timings count from this call, so they restart on a resume. A fresh run
  starts the file over; a resumed one keeps the lines up to the step it
  resumes from and appends. Each line is one append, so a reader never sees
  half of one, and a failure to write warns once and never stops the run.
  Neither this nor `show_progress` changes the chain. `run_job` sets it to
  `progress.jsonl` in `output_dir` unless the job gives one (`null` turns it
  off).
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
    prune_stranded::Bool = true,
    node_flip::Real = 0.1,
    checkpoint::Union{Nothing,AbstractString,Symbol} = nothing,
    checkpoint_interval::Real = 900.0,
    resume::Bool = false,
    progress_log::Union{Nothing,AbstractString,Symbol} = nothing,
)
    # JSON delivers floats-as-Int and arrays-as-JSON3.Array; normalize.
    stretch_a       = Float64(stretch_a)
    node_flip       = Float64(node_flip)
    0 <= node_flip <= 1 || throw(ArgumentError(
        "node_flip is a probability; got $node_flip"))
    ladder_adapt_ν0 = Float64(ladder_adapt_ν0)
    ladder_adapt_K  = Float64(ladder_adapt_K)
    ck_path = checkpoint === nothing ? nothing : String(checkpoint)
    resume && ck_path === nothing && throw(ArgumentError(
        "resume = true needs `checkpoint`, the state file to continue from"))
    plog = progress_log === nothing ? nothing : String(progress_log)
    plog_warned = Ref(false)
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
    # readout cadence: every `diag_cadence` steps (during burn-in, over a
    # rolling window; see `burn_ring` below).
    _sci_names = science_params === nothing ?
        String[nm for nm in layout.unfrozen_names if occursin(r"_k\d+$", nm) || nm == "rho_s"] :
        science_params
    science_cols = Int[i for (i, nm) in enumerate(layout.unfrozen_names) if nm in _sci_names]
    isempty(science_cols) && (science_cols = collect(1:n_dim))   # fallback: gate on all
    sci_names = layout.unfrozen_names[science_cols]   # for the readout
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
    # Full-circle angles are the exception: re-charted during burn-in by
    # `_pt_ensemble_recut!`. `circ` is computed once; `recut_moved` collects
    # the final windows for the single end-of-run report.
    circ        = circular_indices(params)
    recut_moved = Dict{String,NTuple{2,Float64}}()
    # The move scale per dimension (see above), and the pruning tally per rung.
    flat_shift = Union{Nothing,Float64}[log_scale_shift(ps)
                                        for ps in layout.unfrozen_priors]
    pruned = zeros(Int, n_temps)
    # Astrometry-only planets get the (Ω, ω) → (Ω + π, ω + π) move
    # (src/samplers/node_flip.jl); empty, and never drawn for, anything else.
    flips = node_flip > 0 ? node_flips(params, data; flat_shift) : NodeFlip[]

    # Goodman-Weare: each half ≥ n_dim+1 walkers, total even.
    n_walkers_eff = max(n_walkers, 2 * n_dim + 2)
    n_walkers_eff % 2 == 1 && (n_walkers_eff += 1)

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
    ck_fp = run_fingerprint(params, data; n_temps, n_walkers = n_walkers_eff,
        n_burnin, thin, seed, stretch_a, ladder = copy(βs), adapt_ladder,
        ladder_adapt_window, ladder_adapt_ν0, ladder_adapt_K, untemper_transit,
        prune_stranded, node_flip)
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
            @warn "Pathfinder warm-start failed; falling back to :map_scatter" exception=(e, catch_backtrace())
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
    # task_idx makes a run bit-identical across thread counts. Each task
    # consumes exactly three draws per sweep (partner, stretch u, accept),
    # so reusing the object across sweeps stays deterministic.
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

                w_partner = rand(trng, partner_lo:partner_hi)
                u = rand(trng)
                z = ((stretch_a - 1) * u + 1)^2 / stretch_a

                @inbounds for d in 1:n_dim
                    buf[d] = state[t, w_partner, d] +
                             z * (state[t, w, d] - state[t, w_partner, d])
                end

                lp_prop, ll_prop = eval_bounded!(buf, slot)
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
        end
        Threads.atomic_add!(n_evals_atomic, length(tasks))
    end

    # --- Progress bar (per-step accept rate + ETA) --------------------
    # A short bar: the readout after it is long, and a line wider than the
    # terminal wraps and defeats the in-place refresh.
    pb = ProgressBar("pt_emcee"; total = n_steps, enabled = show_progress,
                     start = step0, width = 12)

    # --- Live convergence readout + optional run-until-converged ------
    rhat_str = "—"; ess_str = "—"      # cached "mean/worst" displays
    conv_run = 0; converged_at = 0     # consecutive passing checks; stop step

    # Burn-in diagnostics, for the readout only. The post-burn-in R-hat needs
    # kept draws, so for the whole burn-in -- an hour of a long fit -- nothing
    # said whether the cold rung was settling. During burn-in the same split
    # R-hat and ESS are taken every `diag_cadence` steps (and at its last step)
    # over a rolling window: the cold rung's science parameters at the last
    # `diag_cadence` steps, every `burn_stride`-th one so the window holds at
    # most 500 rows. Memory and cost are flat in the run length. A window says
    # whether the walkers agree NOW; it is not the gate, which reads only the
    # kept draws. Not checkpointed: after a resume the window refills.
    burn_on     = n_burnin > 0 && (show_progress || plog !== nothing)
    burn_stride = max(1, cld(diag_cadence, 500))
    burn_ring   = Array{Float64,3}(undef, burn_on ? max(4, diag_cadence ÷ burn_stride) : 0,
                                   length(science_cols), n_walkers_eff)
    burn_next   = 1; burn_count = 0   # slot written next; slots filled
    burn_rhat_str = "—"; burn_ess_str = "—"

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
        pruned .= ck.pruned
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
        burn_rhat_str = get(ck, :burn_rhat_str, "—")::String
        burn_ess_str  = get(ck, :burn_ess_str, "—")::String
        # Windows the saved walkers live in (moved by the burn-in re-cut).
        for (nm, (lo, _)) in ck.windows
            set_circular_window!(params, findfirst(==(nm), layout.unfrozen_names),
                                 lo; transforms = (target.transform,))
        end
        merge!(recut_moved, ck.recut_moved)
        n_done = step0
    end
    last_ck = time()
    plog === nothing || _pt_log_start!(plog, resume, step0, plog_warned)
    t_loop = time()

    # --- Main loop ----------------------------------------------------
    for step in (step0 + 1):n_steps
        do_half_step!(tasks_h1, :h1)
        do_half_step!(tasks_h2, :h2)

        @inbounds for t in 1:n_temps
            propose_within[t] += Threads.atomic_xchg!(propose_temp[t], 0)
            accept_within[t]  += Threads.atomic_xchg!(accept_temp[t],  0)
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

        # ---- Stranded walkers (serial; burn-in only) -----------------
        # See `_pt_prune_stranded!`. Stops at 3/4 of burn-in so the moved
        # walkers decorrelate from the ones they were copied onto, and runs
        # before the re-cut so the seam is chosen from the cleaned cold rung.
        if prune_stranded && step % 50 == 0 && step <= (3 * n_burnin) ÷ 4
            pruned .+= _pt_prune_stranded!(state, logπ_arr, logL_arr, βs,
                                           rng_master)
        end

        # ---- Circular re-cut (serial; threads idle) ------------------
        # Halfway through burn-in, while the ensemble can still merge the
        # two lobes a seam-centred angle burns in as; and at the end of
        # burn-in, which fixes the chart for every recorded step (usually a
        # no-op by then: circular_cut keeps the current seam unless another
        # arc is clearly emptier). Relabels `state` at every rung; logπ_arr
        # and logL_arr stay valid, see `_pt_ensemble_recut!`. The burn-in
        # window holds earlier cold-rung positions, so it is relabelled too.
        if !isempty(circ) && (step == n_burnin ÷ 2 || step == n_burnin)
            for d in _pt_ensemble_recut!(state, params, circ, (target.transform,),
                                         recut_moved)
                j = findfirst(==(d), science_cols)
                j === nothing && continue
                lo, hi = bounds(layout.unfrozen_priors[d])
                @inbounds for w in 1:n_walkers_eff, r in 1:burn_count
                    burn_ring[r, j, w] = circular_relabel(burn_ring[r, j, w], lo, hi)
                end
            end
        end

        # ---- Burn-in window: the cold rung's science parameters ------
        # Back from the flat scale to x, like the kept draws. Read-only.
        if burn_on && step <= n_burnin && step % burn_stride == 0
            @inbounds for w in 1:n_walkers_eff, (j, c) in enumerate(science_cols)
                s = flat_shift[c]
                burn_ring[burn_next, j, w] = s === nothing ? state[1, w, c] :
                                             exp(state[1, w, c]) - s
            end
            burn_next  = burn_next % size(burn_ring, 1) + 1
            burn_count = min(burn_count + 1, size(burn_ring, 1))
        end
        burn_check = burn_on && step <= n_burnin &&
                     (step % diag_cadence == 0 || step == n_burnin)
        bdiag = burn_check ? _pt_ring_diag(burn_ring, burn_next, burn_count) : nothing
        if bdiag !== nothing
            burn_rhat_str, burn_ess_str = _pt_diag_strings(bdiag, sci_names)
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
        post_check = step > n_burnin && (step - n_burnin) % diag_cadence == 0
        pdiag = post_check ?
            _pt_convergence(samples, keep_idx, n_walkers_eff, science_cols) : nothing
        if pdiag !== nothing
            rhat_str, ess_str = _pt_diag_strings(pdiag, sci_names)
            if convergence_stop
                pass = step >= min_steps && pdiag.max_rhat < rhat_threshold &&
                       pdiag.min_tail > tail_ess_threshold
                conv_run = pass ? conv_run + 1 : 0
                conv_run >= n_converged_checks && (converged_at = step)
            end
        end

        # ---- Progress log: one line per diagnostic check -------------
        if plog !== nothing && (burn_check || post_check)
            el = time() - t_loop
            _pt_log_line(plog, plog_warned; step, n_steps, n_burnin,
                final = false, elapsed = el, rate = (step - step0) / max(el, 1e-9),
                n_evals = n_evals_atomic[], accept_within, propose_within,
                accept_swap, propose_swap,
                lp = _pt_cold_lp(state, logπ_arr, logL_arr, flat_shift),
                diag = burn_check ? bdiag : pdiag,
                window = burn_check ? _pt_burn_window(step, burn_stride, burn_count) :
                                      [n_burnin + 1, step],
                names = sci_names)
        end

        # ---- Progress update ----------------------------------------
        if show_progress
            total_props = sum(propose_within)
            total_acc = sum(accept_within)
            acc_rate = total_props > 0 ? total_acc / total_props : 0.0
            lp_med, lp_max = _pt_cold_lp(state, logπ_arr, logL_arr, flat_shift)
            # The burn-in window's figures until the kept draws have their own,
            # marked as such: they describe the last window, not the posterior.
            burn_shown = rhat_str == "—" && burn_rhat_str != "—"
            update!(pb; n_done = step,
                    fields = (:acc => @sprintf("%.3f", acc_rate),
                              :β1_acc => @sprintf("%.3f", accept_within[1] /
                                                          max(propose_within[1], 1)),
                              # The LADDER's health, not the move's. This is the
                              # half that fails silently: once the rungs stop
                              # exchanging, the cold chain is stuck in one mode
                              # and R-hat happily certifies convergence to it.
                              # mean/min over the adjacent pairs, and which pair
                              # is the min. One rung (n_temps = 1) has no swaps.
                              :swap => _pt_swap_str(_pt_swap_summary(accept_swap,
                                                                     propose_swap)),
                              # Cold-rung log-posterior, median/max over walkers.
                              :lp => string(_pt_lp_str(lp_med), "/", _pt_lp_str(lp_max)),
                              # mean/worst over science params; ESS is
                              # mean(bulk)/worst(tail)
                              burn_shown ? (Symbol("Rhat(burn)") => burn_rhat_str) :
                                           (:Rhat => rhat_str),
                              burn_shown ? (Symbol("ESS(burn)") => burn_ess_str) :
                                           (:ESS => ess_str),
                              :nevals => _pt_count_str(n_evals_atomic[])))
        end

        # ---- Checkpoint: every `checkpoint_interval` s, on the last step, and
        # before an early stop (src/checkpoint.jl) -------------------------
        if ck_path !== nothing && (step == n_steps || (convergence_stop && converged_at > 0) ||
                                   time() - last_ck >= checkpoint_interval)
            write_checkpoint(ck_path, "pt_emcee", ck_fp, (; step,
                state, logπ = logπ_arr, logL = logL_arr, betas = βs, rng_master,
                rngs_h1, rngs_h2, rngs_flip, accept_within, propose_within,
                accept_swap, propose_swap, flip_proposed, flip_accepted, pruned,
                n_evals = n_evals_atomic[], β_hist = β_hist[1:step, :],
                swap_hist = swap_hist[1:step, :], smd_hist = smd_hist[1:step, :],
                keep_idx, samples = samples[1:keep_idx, :],
                lp_samples = lp_samples[1:keep_idx], evidence = evidence_acc,
                conv_run, rhat_str, ess_str, burn_rhat_str, burn_ess_str,
                windows = circular_windows(params), recut_moved))
            last_ck = time()
        end

        # ---- Run-until-converged: stop once the science params clear the
        # gate for `n_converged_checks` consecutive diagnostics ----------
        if convergence_stop && converged_at > 0
            show_progress && @info "pt_emcee: science params converged at step $converged_at " *
                "(Rhat<$(rhat_threshold), tail-ESS>$(tail_ess_threshold)); stopping early"
            break
        end
    end
    show_progress && finish!(pb)
    show_progress && _pt_recut_info("pt_emcee", recut_moved)
    show_progress && _node_flip_info("pt_emcee", flips, params, flip_proposed,
                                     flip_accepted)
    show_progress && sum(pruned) > 0 && @info "pt_emcee: burn-in moved " *
        "$(sum(pruned)) stranded walker state(s) onto their rung's mode " *
        "($(pruned[1]) at β = 1), all from a region whose total posterior mass " *
        "is bounded below e⁻¹⁰ of the mode's. The recorded chain is ordinary MCMC."
    if convergence_stop && converged_at == 0
        @warn "pt_emcee: hit max_steps=$n_steps WITHOUT meeting the convergence gate " *
              "(Rhat<$(rhat_threshold), tail-ESS>$(tail_ess_threshold)) on science params — " *
              "results may be unconverged (last $rhat_str Rhat, $ess_str ESS)"
    end

    # The last line of the progress log: where the run ended, with the
    # diagnostics of everything it kept (or of the burn-in window, if it
    # ended there).
    if plog !== nothing
        el = time() - t_loop
        ended_post = n_done > n_burnin
        _pt_log_line(plog, plog_warned; step = n_done, n_steps, n_burnin,
            final = true, elapsed = el, rate = (n_done - step0) / max(el, 1e-9),
            n_evals = n_evals_atomic[], accept_within, propose_within,
            accept_swap, propose_swap,
            lp = _pt_cold_lp(state, logπ_arr, logL_arr, flat_shift),
            diag = ended_post ?
                _pt_convergence(samples, keep_idx, n_walkers_eff, science_cols) :
                _pt_ring_diag(burn_ring, burn_next, burn_count),
            window = ended_post ? [n_burnin + 1, n_done] :
                                  _pt_burn_window(n_done, burn_stride, burn_count),
            names = sci_names)
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

    # Bridge sampling from the cold chain we already have. Touches no hot rung,
    # and assumes only that the fitted reference OVERLAPS the posterior -- not
    # that the posterior is Gaussian about its mode. Requires an unconstrained
    # target, as the proposal is Gaussian on R^n.
    #
    # Costs `n_kept + bridge_n` likelihood evaluations, not `bridge_n`: it
    # re-evaluates the posterior at EVERY kept draw. At the defaults that is
    # 170k evaluations, which single-threaded was 42 s against 89 s for the
    # entire MCMC on a Gaia DR4 IAD fit. `bridge_evidence` threads both
    # evaluation loops now, bit-identically; mind the scaling anyway, because
    # n_kept grows with n_walkers × (n_steps - n_burnin).
    #
    # NOTE: on the fit_* entry points this is computed AGAIN, post-fit, by
    # `_augment_evidence!` (src/runner.jl) -- which exists because run_job's
    # targets are bounded and never get here. Same estimator, same chains, so
    # the second call is redundant on this path; it is cheap now, but it is
    # still two passes over the posterior.
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
            @debug "sample_pt_emcee: bridge evidence unavailable" exception = err
        end
    end

    # Bridge is biased LOW when its reference `q` is estimated from too few
    # INDEPENDENT points. `q` carries n_dim + n_dim(n_dim+1)/2 free parameters,
    # and on a curved 15-D target with exactly known log Z bridge fell to -3.2
    # nats at n_eff ~ 22 while reference-path held to +0.17. Splitting the
    # sample does not rescue it (-2.8), so it is the ESTIMATE of q that is bad,
    # not re-use of the draws -- and no diagnostic bridge returns will say so
    # (`overlap` reads ~1 throughout). Refuse the headline rather than report a
    # silently-low number. Well-mixed chains are unaffected: 51 Peg runs at
    # n_eff ~ 8000 against 135 parameters.
    if isfinite(log_z_bridge)
        n_eff_min = try
            minimum(filter(isfinite, vec(MCMCChains.ess(chains)[:, :ess])))
        catch err
            @debug "sample_pt_emcee: ESS unavailable for the bridge guard" exception = err
            NaN
        end
        n_par_q = n_dim + n_dim * (n_dim + 1) ÷ 2
        if isfinite(n_eff_min) && n_eff_min < n_par_q
            @warn "sample_pt_emcee: dropping bridge as the headline — its " *
                  "reference is fitted from too few effective samples " *
                  "(min ESS $(round(Int, n_eff_min)) < $n_par_q free parameters " *
                  "in q). Bridge is biased LOW in this regime and none of its " *
                  "diagnostics detect it. Sample longer, or use " *
                  "`reference_path_evidence`, which anneals away from q and is " *
                  "unaffected." min_ess = n_eff_min n_params_in_q = n_par_q
            log_z_bridge = NaN
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
            @warn "sample_pt_emcee: bridge and mode-Laplace disagree by " *
                  "$(round(abs(log_z_bridge - log_z_laplace), digits=1)) nats. " *
                  "Reporting bridge (it is the validated one), but CHECK THE " *
                  "POSTERIOR IS NOT MULTIMODAL before quoting it — a single " *
                  "Gaussian reference cannot cover multiple modes and overlap " *
                  "will not tell you so." bridge = log_z_bridge laplace = log_z_laplace overlap = bridge_overlap
        end
        log_z = log_z_bridge
    elseif isfinite(log_z_laplace)
        @warn "sample_pt_emcee: bridge evidence unavailable; falling back to " *
              "mode-Laplace. Treat it as indicative: on a clean unimodal target " *
              "it measured 24 nats from bridge, and it has no validation against " *
              "a known log Z." laplace = log_z_laplace
        log_z = log_z_laplace
    elseif tempered_ok
        @warn "sample_pt_emcee: neither bridge nor mode-Laplace is available. " *
              "log_evidence is the TEMPERED value, which is biased low by " *
              "10^2-10^4 nats on a signal-locked posterior and cannot detect " *
              "that it is. Do not quote it without an independent check." tempered = log_z
    else
        @warn "sample_pt_emcee: no usable evidence estimate for this run. " *
              "log_evidence is not quotable."
    end

    # Independent of which was chosen: a large tempered/headline gap is the
    # phase-transition signature and worth surfacing, because the tempered
    # numbers are still in `evidence` and someone will read them.
    if tempered_ok && isfinite(log_z) && laplace_check &&
       abs(log_z_tempered - log_z) > laplace_switch_tol
        min_swap = isempty(acc_swap) ? NaN : minimum(acc_swap)
        @warn "sample_pt_emcee: the TEMPERED stack (TI/TI+/SS+/H+ in `evidence`) " *
              "is $(round(abs(log_z_tempered - log_z), digits=1)) nats from the " *
              "reported log_evidence — the phase-transition signature (min swap " *
              "accept = $(round(min_swap, digits=3))). Those four share one " *
              "mean_logL array, so their agreeing with EACH OTHER is not a " *
              "check." tempered = log_z_tempered reported = log_z
    end

    return PTemceeResult(
        chains, log_z, log_z_laplace, log_z_bridge, ev_report, acc_within, acc_swap,
        βs, n_evals_atomic[],
        _ladder_history(β_hist, swap_hist, smd_hist, evidence_acc, n_done, n_burnin)
    )
end
