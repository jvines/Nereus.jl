# Informed starts and period jumps for the fixed-k PT samplers.
#
# The trans-dimensional births already propose a planet from the residual
# periodogram (InformedBirth, src/transdim/birth_strategies.jl): a mixture over
# the periodogram's peaks in log P plus a uniform-in-log floor, with the RV
# amplitude and the time anchor centred on each peak's least-squares sinusoid.
# A fixed-k run had nothing of the kind: its walkers start from the prior and
# reach a period only by local stretch / DE steps, which cannot cross the
# likelihood valley between a period and its yearly or daily alias, or a
# harmonic. This file gives the fixed-k ensembles (pt_emcee, pt_whitening) and
# the in-house PT (pt) the same two tools, both opt-in:
#
#   * informed starts (`informed_init`): each walker places every eligible
#     planet from the informed mixture of that planet's residual periodogram;
#   * a period jump (`period_jump_prob`): an MCMC move that redraws one planet's
#     (P, K, time anchor) from the same mixture, or multiplies its period by a
#     harmonic ratio, accepted with the tempered Metropolis-Hastings ratio.
#
# The densities, the peak finder, the window function and its cache, and the
# alias ratios are the trans-dimensional ones (`_find_peaks`,
# `_informed_draw!`, `_informed_slot_log_q`, `ALIAS_RATIOS`); only the choice of
# residual and when it is refreshed are new here.
#
# ---------------------------------------------------------------------------
# The residual and its periodogram
# ---------------------------------------------------------------------------
# Planet k's setup S_k is built from the RV residual after every OTHER planet
# (`_compute_rv_residuals(...; exclude = (k,))`, per-instrument medians for the
# zero points as for the births), searched over k's own period prior. The top
# `PERIOD_JUMP_ALIAS_PARENTS` peaks also bring their yearly and sidereal-daily
# aliases, f ± 1/365.25 d and f ± 1/0.99727 d, as extra components (weight
# `PERIOD_JUMP_ALIAS_WEIGHT` of the parent's), with their own LS amplitude and
# phase, so walkers are spread over the peaks AND their aliases. Each peak's
# width in log P is the periodogram's resolution, σ = P / 2T over the RV
# baseline T (at most the births' 5 %): a period off by more than ~P/T is out of
# phase across the data.
#
# ---------------------------------------------------------------------------
# Detailed balance of the period jump
# ---------------------------------------------------------------------------
# For walker θ at rung β, planet k is chosen with fixed probability
# `period_jump_prob`, and then one of two kernels with a fixed probability
# (`PERIOD_JUMP_HARMONIC_FRAC` for the harmonic one; the informed one
# otherwise, or always the harmonic one while S_k is `nothing`). Each kernel is
# reversible with respect to π_β(θ) ∝ p(θ) L(θ)^β on its own, so the mixture is.
#
#  1. Informed (an independence proposal on the block u = (P, K, t) of planet k,
#     every other coordinate held). The draw is the mixture
#         q_S(u) = Σ_c π_c q(x_P | c) Π_slots q(u_s | c)
#     of `_informed_draw!`, scored by `_informed_slot_log_q` (with
#     `strict_wrap`, so a time anchor the wrapped draw cannot produce scores
#     zero, and a draw outside the prior is a rejection, never a clamp). S_k is
#     a fixed object during the move, so the proposal does not depend on the
#     current u and the Hastings ratio is that of an independence sampler:
#         log α = β ΔlnL + Δln p_x + ln q_S(u) − ln q_S(u').
#     q_S is a density on the raw parameters x (the period's includes the
#     d log x / dx Jacobian, as the prior does). The ensembles move on the flat
#     scale y = log(x + s) of LogUniform / ModJeffreys parameters, and their
#     cached log π is ln p_x + Σ y over those coordinates, so the move uses
#     Δln p_x = Δ(log π) − Δ(Σ y over the moved coordinates).
#  2. Harmonic (Green's update move of `propose_planet_alias_jump`): P → rP,
#     r uniform over `ALIAS_RATIOS`, a set closed under inversion, so the
#     choice of r is symmetric and the ratio carries the Jacobian alone:
#         log α = β ΔlnL + Δln p_x + ln r.
#
# What S_k may depend on. Within one move S_k is the same object for the
# forward and the reverse density, which is all the ratio above needs if S_k is
# fixed. The samplers refresh S_k from the cold rung's best walker every
# `PERIOD_JUMP_REFRESH` steps of burn-in and at its last step, and then never
# again: the recorded chain uses one fixed proposal (a homogeneous Markov chain
# with π_β invariant at every rung), and burn-in, which is discarded, is
# piecewise homogeneous. The setups and the move's RNG streams are in the
# checkpoint, so a resumed run proposes exactly what an uninterrupted one would.
#
# Eligible planets: a period slot that is sampled, an RV semi-amplitude, RVs in
# the data, not an SB2 block (two amplitudes) and not the `:a_driven` mass
# parametrization (the period slot holds a, as for the harmonic jump). Others
# are left to the ordinary moves.

const PERIOD_JUMP_HARMONIC_FRAC = 0.25
const PERIOD_JUMP_REFRESH       = 50
const PERIOD_JUMP_ALIAS_PARENTS = 3
const PERIOD_JUMP_ALIAS_WEIGHT  = 0.5
const _PJ_YEAR_DAYS     = 365.25
const _PJ_SIDEREAL_DAYS = 0.99726957

"""
    PeriodJumpPlan

The fixed-k planets the period jump and the informed init act on: `planets`
(layout planet indices), the layout slots each redraws (`slots`, period first,
then K under `:K_driven` and the time anchor, when sampled), their unfrozen
positions (`dims`), and the current informed setup of each (`setups`;
`nothing` when its residual periodogram found no peak). See the header of
src/samplers/period_jump.jl.
"""
struct PeriodJumpPlan
    planets::Vector{Int}
    slots::Vector{Vector{Int}}
    dims::Vector{Vector{Int}}
    setups::Vector{Union{Nothing,_InformedSetup}}
end

"""
    period_jump_plan(params, data) -> PeriodJumpPlan

The planets eligible for the period jump and the informed init (see the
header of src/samplers/period_jump.jl), with no setups computed yet.
"""
function period_jump_plan(params::Params, data::Data)
    L, cfg = params.layout, params.config
    planets = Int[]; slots = Vector{Int}[]; dims = Vector{Int}[]
    uf(slot) = findfirst(==(slot), L.unfrozen_idx)
    if !isempty(data.t_rv) && cfg.parametrization.mass !== :a_driven
        for (k, block) in enumerate(L.planet_blocks)
            k <= cfg.max_kplanet || break
            (has_K(block) && !(block isa SB2Block)) || continue
            uf(block.P) === nothing && continue
            sl = [block.P]
            cfg.parametrization.mass === :K_driven && uf(block.K) !== nothing &&
                push!(sl, block.K)
            uf(block.t) !== nothing && push!(sl, block.t)
            push!(planets, k); push!(slots, sl); push!(dims, [uf(s) for s in sl])
        end
    end
    return PeriodJumpPlan(planets, slots, dims,
                          Union{Nothing,_InformedSetup}[nothing for _ in planets])
end

_period_jump_dims(plan::PeriodJumpPlan) = reduce(vcat, plan.dims; init = Int[])

# Planet k's period range in its slot's own variable, and the periods searched.
function _pj_period_range(params::Params, theta::Theta, k::Int)
    L = params.layout
    block = L.planet_blocks[k]
    P_pos = findfirst(==(block.P), L.unfrozen_idx)
    x_lo, x_hi = bounds(L.unfrozen_priors[P_pos])
    m = _period_slot_map(theta, block, theta.values)
    log_x_min, log_x_max, (P_lo, P_hi) = _slot_period_range(m, x_lo, x_hi)
    return m, log_x_min, log_x_max, Float64(P_lo), Float64(P_hi)
end

# The LS-only setup from a residual: the periodogram's peaks (`_find_peaks`)
# plus the yearly / sidereal-daily aliases of the strongest ones.
function _pj_setup_from_resid(m::_PeriodSlotMap, log_x_min, log_x_max, P_lo, P_hi,
                              resid::Vector{Float64}, data::Data)
    (isfinite(log_x_min) && log_x_min < log_x_max && P_lo < P_hi) || return nothing
    periods, weights, K_est, Mo_est = _find_peaks(data.t_rv, resid, P_lo, P_hi;
                                                  t_ref = data.t_ref, σ = data.rv_err)
    isempty(periods) && return nothing
    lp = log.(periods); w = copy(weights); Ks = copy(K_est); Ms = copy(Mo_est)
    for i in 1:min(PERIOD_JUMP_ALIAS_PARENTS, length(periods))
        f = 1 / periods[i]
        for δ in (1 / _PJ_SIDEREAL_DAYS, 1 / _PJ_YEAR_DAYS), sgn in (-1, 1)
            fa = f + sgn * δ
            fa > 0 || continue
            Pa = 1 / fa
            P_lo <= Pa <= P_hi || continue
            any(l -> abs(log(Pa) - l) < INFORMED_SIGMA_P, lp) && continue
            Ka, Ma = _estimate_K_phase(data.t_rv, resid, fa, data.t_ref; σ = data.rv_err)
            push!(lp, log(Pa)); push!(w, PERIOD_JUMP_ALIAS_WEIGHT * weights[i])
            push!(Ks, Ka); push!(Ms, Ma)
        end
    end
    s = sum(w)
    w = s > 0 ? w ./ s : fill(1 / length(w), length(w))
    S = _ls_setup_from_peaks(m, log_x_min, log_x_max, lp, w, Ks, Ms)
    # Peak widths matched to the periodogram's resolution, δ log P = P / T over
    # a baseline T (half of it as σ), instead of the births' fixed 5 %: a period
    # off by more than ~P/T is out of phase across the data, so the wide kernel
    # mostly proposes dephased orbits. Never wider than the births' width.
    T = maximum(data.t_rv) - minimum(data.t_rv)
    if T > 0
        S.sigma_x .= m.slope .* clamp.(0.5 .* exp.(lp) ./ T, INFORMED_SIGMA_P_FLOOR,
                                       INFORMED_SIGMA_P)
    end
    return S
end

"""
    _period_jump_setup(theta, k, data) -> Union{Nothing,_InformedSetup}

Planet `k`'s informed setup at the state `theta`: the residual periodogram
after every other planet, over `k`'s period prior, with the aliases of its
strongest peaks.
"""
function _period_jump_setup(theta::Theta, k::Int, data::Data)
    m, log_x_min, log_x_max, P_lo, P_hi = _pj_period_range(theta.params, theta, k)
    resid = _compute_rv_residuals(theta, data; exclude = (k,))
    return _pj_setup_from_resid(m, log_x_min, log_x_max, P_lo, P_hi, resid, data)
end

"""
    _period_jump_refresh!(plan, theta, data)

Recompute every planet's setup at `theta` (the cold rung's best walker).
"""
function _period_jump_refresh!(plan::PeriodJumpPlan, theta::Theta, data::Data)
    for (i, k) in enumerate(plan.planets)
        plan.setups[i] = _period_jump_setup(theta, k, data)
    end
    return plan
end

"""
    _period_jump_propose!(theta, i, plan, data, rng) -> log_q_ratio

Propose, in place on `theta.values`, the period jump of the `i`-th planet of
`plan`, and return log[q(θ' → θ) / q(θ → θ')] + log|∂θ'/∂θ| on the raw
parameters (−Inf when the draw left the prior or the current state is not one
the informed draw can produce). See the header of src/samplers/period_jump.jl.
"""
function _period_jump_propose!(theta::Theta, i::Int, plan::PeriodJumpPlan,
                               data::Data, rng::AbstractRNG)
    k = plan.planets[i]
    block = theta.params.layout.planet_blocks[k]
    S = plan.setups[i]
    if S === nothing || rand(rng) < PERIOD_JUMP_HARMONIC_FRAC
        r = ALIAS_RATIOS[rand(rng, 1:length(ALIAS_RATIOS))]
        theta.values[block.P] *= r
        return log(r)
    end
    slots = plan.slots[i]
    lq_old = _informed_slot_log_q(theta, block, theta.values, S, data, false;
                                  slots, strict_wrap = true)
    isfinite(lq_old) || return -Inf
    _informed_draw!(theta.values, theta, block, S, data, rng, false; slots) === nothing &&
        return -Inf
    lq_new = _informed_slot_log_q(theta, block, theta.values, S, data, false;
                                  slots, strict_wrap = true)
    isfinite(lq_new) || return -Inf
    return lq_old - lq_new
end

# Raw parameters of walker (t, w) into `theta`, from the flat scale.
@inline function _pj_load!(theta::Theta, state::Array{Float64,3}, t::Int, w::Int,
                           unfrozen_idx, flat_shift)
    @inbounds for (j, idx) in enumerate(unfrozen_idx)
        y = state[t, w, j]
        s = flat_shift === nothing ? nothing : flat_shift[j]
        theta.values[idx] = s === nothing ? y : exp(y) - s
    end
    return theta
end

"""
    _period_jump_sweep!(eval!, state, logπ_arr, logL_arr, βs, plan, data, prob,
                        rngs, bufs, thetas, unfrozen_idx, flat_shift,
                        proposed, accepted) -> n_evals

One period-jump pass over every (rung, walker) of an ensemble: for each
planet of `plan`, with probability `prob`, propose its jump and accept it with
the tempered ratio β·Δlog L + Δlog π − ΔΣy + log q-ratio (see the header).
`eval!(buf, t, w, slot)` is the sampler's `(log π, log L)` on its walker
scale; `flat_shift` its per-dimension log-scale shift (`nothing` for walkers
on the raw parameters). `rngs` holds one stream per (rung, walker), `bufs` and
`thetas` one scratch each per chunk slot (never the Theta `eval!` writes).
"""
function _period_jump_sweep!(eval!::F, state::Array{Float64,3},
                             logπ_arr::Matrix{Float64}, logL_arr::Matrix{Float64},
                             βs::AbstractVector{Float64}, plan::PeriodJumpPlan,
                             data::Data, prob::Float64,
                             rngs::Vector{MersenneTwister},
                             bufs::Vector{Vector{Float64}},
                             thetas::Vector{Theta{Float64}},
                             unfrozen_idx::AbstractVector{Int}, flat_shift,
                             proposed::Vector{Threads.Atomic{Int}},
                             accepted::Vector{Threads.Atomic{Int}}) where {F}
    n_t, n_w, n_dim = size(state)
    n_evals = Threads.Atomic{Int}(0)
    chunks = _chunk_ranges(n_t * n_w, length(bufs))
    Threads.@threads :static for slot in 1:length(chunks)
        for task_idx in chunks[slot]
            t = (task_idx - 1) ÷ n_w + 1
            w = (task_idx - 1) % n_w + 1
            rng = rngs[task_idx]
            buf = bufs[slot]
            theta = thetas[slot]
            for i in eachindex(plan.planets)
                rand(rng) < prob || continue
                _pj_load!(theta, state, t, w, unfrozen_idx, flat_shift)
                Threads.atomic_add!(proposed[t], 1)
                lq = _period_jump_propose!(theta, i, plan, data, rng)
                isfinite(lq) || continue
                @inbounds for d in 1:n_dim
                    buf[d] = state[t, w, d]
                end
                ΔJ = 0.0; ok = true
                @inbounds for d in plan.dims[i]
                    x = theta.values[unfrozen_idx[d]]
                    s = flat_shift === nothing ? nothing : flat_shift[d]
                    if s === nothing
                        buf[d] = x
                    else
                        x + s > 0 || (ok = false; break)
                        buf[d] = log(x + s)
                        ΔJ += buf[d] - state[t, w, d]
                    end
                end
                ok || continue
                lp, ll = eval!(buf, t, w, slot)
                Threads.atomic_add!(n_evals, 1)
                log_ratio = βs[t] * (ll - logL_arr[t, w]) + (lp - logπ_arr[t, w]) -
                            ΔJ + lq
                if log(rand(rng)) < log_ratio
                    @inbounds for d in 1:n_dim
                        state[t, w, d] = buf[d]
                    end
                    logπ_arr[t, w] = lp
                    logL_arr[t, w] = ll
                    Threads.atomic_add!(accepted[t], 1)
                end
            end
        end
    end
    return n_evals[]
end

# The step at whose start the setups are refreshed: the first, every
# `PERIOD_JUMP_REFRESH` steps of burn-in, and the first recorded step (from the
# state burn-in ended in). Never after: the recorded chain keeps one proposal.
_period_jump_refresh_due(step::Int, b_end::Int) =
    step == 1 || (step - 1 <= b_end &&
                  ((step - 1) % PERIOD_JUMP_REFRESH == 0 || step - 1 == b_end))

# The cold rung's best walker (highest log π + log L), as raw parameters.
function _pj_best_cold!(theta::Theta, state::Array{Float64,3},
                        logπ_arr::Matrix{Float64}, logL_arr::Matrix{Float64},
                        unfrozen_idx, flat_shift)
    best = 1; bv = -Inf
    for w in 1:size(state, 2)
        v = logπ_arr[1, w] + logL_arr[1, w]
        (isfinite(v) && v > bv) && (bv = v; best = w)
    end
    return _pj_load!(theta, state, 1, best, unfrozen_idx, flat_shift)
end

# Argument checks shared by the samplers.
function _period_jump_check(prob::Real)
    p = Float64(prob)
    0 <= p <= 1 || throw(ArgumentError("period_jump_prob is a probability; got $prob"))
    return p
end

# Enters the checkpoint fingerprint only when on, so a run without it keeps the
# fingerprint it always had.
_period_jump_fp(p::Float64) = p > 0 ? (; period_jump_prob = p) : (;)

# =====================================================================
# Informed starts
# =====================================================================

# y = K cos(2πf(t − t_ref) + Mo): the sinusoid `_estimate_K_phase` fits.
_pj_sinusoid(t, P, K, Mo, t_ref) = K * cos(2π * (t - t_ref) / P + Mo)

"""
    informed_init_plan(params, data) -> PeriodJumpPlan

The setups the informed init draws every walker's planets from. The residual
after the per-instrument medians is searched greedily (strongest peak, its
least-squares sinusoid subtracted, again) for as many signals as there are
eligible planets; the signals are given to the planets in order of period,
each to the first planet whose period prior holds it; and planet k's setup is
then the periodogram of the residual after the signals given to the OTHER
planets, as the period jump's is after the other planets. Its weights are
flattened halfway to uniform (½ wᵢ + ½ / n), so walkers spread over the top
peaks and their aliases instead of piling onto the strongest.
"""
function informed_init_plan(params::Params, data::Data)
    plan = period_jump_plan(params, data)
    isempty(plan.planets) && return plan
    theta0 = Theta{Float64}(params)
    resid0 = _compute_rv_residuals(theta0, data; exclude = 1:length(params.layout.planet_blocks))
    rng_info = [_pj_period_range(params, theta0, k) for k in plan.planets]
    P_lo = minimum(r[4] for r in rng_info); P_hi = maximum(r[5] for r in rng_info)
    t = data.t_rv; t_ref = data.t_ref
    signals = NTuple{3,Float64}[]
    r = copy(resid0)
    for _ in plan.planets
        P_lo < P_hi || break
        periods, _, K_est, Mo_est = _find_peaks(t, r, P_lo, P_hi; t_ref, σ = data.rv_err)
        isempty(periods) && break
        push!(signals, (periods[1], K_est[1], Mo_est[1]))
        r .-= _pj_sinusoid.(t, periods[1], K_est[1], Mo_est[1], t_ref)
    end
    # Signals to planets: shortest period first, each to the first free planet
    # whose prior holds it (else the first free planet).
    owner = zeros(Int, length(signals))
    free = trues(length(plan.planets))
    for si in sortperm([s[1] for s in signals])
        P = signals[si][1]
        i = findfirst(j -> free[j] && rng_info[j][4] <= P <= rng_info[j][5],
                      eachindex(plan.planets))
        i === nothing && (i = findfirst(free))
        i === nothing && break
        owner[si] = i; free[i] = false
    end
    for (i, _) in enumerate(plan.planets)
        ri = copy(resid0)
        for (si, s) in enumerate(signals)
            (owner[si] == 0 || owner[si] == i) && continue
            ri .-= _pj_sinusoid.(t, s[1], s[2], s[3], t_ref)
        end
        m, lxmin, lxmax, plo, phi = rng_info[i]
        S = _pj_setup_from_resid(m, lxmin, lxmax, plo, phi, ri, data)
        if S !== nothing
            n = length(S.w)
            S.w .= 0.5 .* S.w .+ 0.5 / n
        end
        plan.setups[i] = S
    end
    return plan
end

"""
    _informed_init_draw!(theta, plan, data, rng) -> theta

Place every planet of `plan` that has a setup from it, in place on
`theta.values` (all of the block's slots, as an informed birth draws them:
period from the mixture, K and time anchor from the drawn peak's sinusoid,
the eccentricity vector near zero, the rest from the prior). A draw that falls
outside a prior leaves the slots it had not reached as they were; the
sampler's retry loop rejects a walker whose density is not finite.
"""
function _informed_init_draw!(theta::Theta, plan::PeriodJumpPlan, data::Data,
                              rng::AbstractRNG)
    L = theta.params.layout
    _pj_seed_gammas!(theta, data, rng)
    for (i, k) in enumerate(plan.planets)
        S = plan.setups[i]
        S === nothing && continue
        _informed_draw!(theta.values, theta, L.planet_blocks[k], S, data, rng, false)
    end
    return theta
end

# The RV zero points near the data: each γ slot at the median of its
# instruments' RVs plus a scatter of its standard error (never one value for
# every walker, which the stretch move could not move off), clamped to the
# prior. The residual periodogram is taken about those medians, so the planets
# drawn from it fit the data only at zero points like these; a γ drawn from a
# broad prior would bury them.
function _pj_seed_gammas!(theta::Theta, data::Data, rng::AbstractRNG)
    L = theta.params.layout
    isempty(data.t_rv) && return theta
    for slot in unique(L.systemic.rv_gamma)
        slot == 0 && continue
        uf_pos = findfirst(==(slot), L.unfrozen_idx)
        uf_pos === nothing && continue
        mask = [L.systemic.rv_gamma[i] == slot for i in data.rv_inst]
        n = count(mask)
        n > 0 || continue
        v = view(data.rv, mask)
        se = n > 1 ? std(v) / sqrt(n) : 0.0
        lo, hi = bounds(L.unfrozen_priors[uf_pos])
        theta.values[slot] = clamp(median(v) + se * randn(rng), lo, hi)
    end
    return theta
end

# One line per run, after the progress bar, like the node flip's.
function _period_jump_info(sampler::AbstractString, plan::PeriodJumpPlan,
                           params::Params, proposed::Vector{Int},
                           accepted::Vector{Int})
    (isempty(plan.planets) || sum(proposed) == 0) && return nothing
    nms = [params.layout.unfrozen_names[first(d)] for d in plan.dims]
    cold = proposed[1] > 0 ? round(accepted[1] / proposed[1]; digits = 3) : NaN
    all_ = round(sum(accepted) / sum(proposed); digits = 3)
    @info "$sampler: period jump on $(join(nms, ", ")): $(sum(proposed)) " *
          "proposals, acceptance $cold at β = 1 ($all_ over all rungs). Draws " *
          "from the residual periodogram's peaks and their aliases, or a harmonic " *
          "ratio, accepted with the tempered Metropolis-Hastings ratio."
    return nothing
end

# The period jump's share of a checkpoint (empty when off, so a checkpoint
# without it is what it always was).
_period_jump_ck(plan, rngs, proposed, accepted) = plan === nothing ? (;) :
    (; rngs_pj = rngs, pj_setups = copy(plan.setups),
       pj_proposed = proposed, pj_accepted = accepted)

# `sample_pt`'s share of a checkpoint (its chain RNGs are saved already).
_pt_period_jump_ck(plan, proposed, accepted) = plan === nothing ? (;) :
    (; pj_setups = copy(plan.setups), pj_proposed = proposed, pj_accepted = accepted)

"""
    _pt_period_jump!(rep, scratch, plan, prob, data, beta, rng, ctr, ws,
                     proposed, accepted, c)

`sample_pt`'s period jump on one replica (raw parameters, its own `log π`):
for each planet of `plan`, with probability `prob`, propose the jump on
`scratch` and accept it with β·Δlog L + Δlog π + log q-ratio (see the header).
"""
function _pt_period_jump!(rep, scratch::Theta{Float64}, plan::PeriodJumpPlan,
                          prob::Float64, data::Data, beta::Float64,
                          rng::AbstractRNG, ctr, ws, proposed::Vector{Int},
                          accepted::Vector{Int}, c::Int)
    for i in eachindex(plan.planets)
        rand(rng) < prob || continue
        scratch.values .= rep.theta.values
        proposed[c] += 1
        lq = _period_jump_propose!(scratch, i, plan, data, rng)
        isfinite(lq) || continue
        lp = log_prior(scratch)
        isfinite(lp) || continue
        ll = _eval_ll(scratch, data, ctr, ws)
        isfinite(ll) || continue
        if log(rand(rng)) < beta * (ll - rep.log_L) + (lp - rep.log_pi) + lq
            rep.theta.values .= scratch.values
            rep.log_pi = lp
            rep.log_L = ll
            accepted[c] += 1
        end
    end
    return nothing
end
