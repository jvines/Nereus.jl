# Hamiltonian parallel tempering (fixed-dim, differentiable models).
#
# NUTS as the within-temperature mutation of a parallel-tempered chain. PT
# tempers the likelihood only — at inverse-temperature β the target is
# `(prior + Jacobian) + β·loglike` — so β=0 samples the prior and β=1 the
# posterior, and `∫₀¹ ⟨logL⟩_β dβ` is the evidence (TI). Gradient-guided NUTS
# mixes a single mode in far fewer steps than the stretch move, so each
# temperature's ⟨logL⟩_β is sharper → tighter TI/SS/H evidence. It does NOT
# replace tempering: between-mode hops still come from the temperature swaps.
# Gated to fixed-dim, ForwardDiff-differentiable models (no trans-dim; no
# N-body / stability-sortperm where the gradient breaks).
#
# The core (`_pt_hmc_core`) is target-agnostic: it takes a `parts(y) ->
# (prior+jac, loglike)` closure + ForwardDiff, so it runs on a `NereusTarget`
# AND on an analytic Gaussian (the logZ regression gate). `sample_pt_hmc` is the
# NereusTarget wrapper. The ladder is β = 0 and a geometric run on
# [beta_min, 1]; an ADAPTIVE re-grid (pilot run → β placed by path-sampling
# thermodynamic length ∝ √Var(logL)) then minimises the TI discretisation
# error left on it.

using AdvancedHMC
using ForwardDiff
using MCMCChains
using Random
using Statistics: mean

# One NUTS chain pinned to a temperature. `Minv` (diagonal inverse mass matrix)
# and `ε` (step size) are what its frozen kernel is built from, kept so that a
# checkpoint can rebuild the kernel exactly (`_frozen_nuts`).
mutable struct _HMCTempState
    h::AdvancedHMC.Hamiltonian
    κ::Any
    θ::Vector{Float64}
    logL::Float64
    Minv::Vector{Float64}
    ε::Float64
end

# Called as `hook(phase, sweep)` after every checkpoint write; `nothing` in use.
# test_pt_hmc_resume sets it to stop a run right after a chosen checkpoint: a
# killed run is the only way to stop inside the pilot or right after a warm-up,
# whose lengths do not depend on `n_sweeps`.
const _PT_HMC_CHECKPOINT_HOOK = Ref{Any}(nothing)

# The kernel a temperature samples with once warm-up is over: diagonal metric
# `Minv`, step size `ε`, multinomial NUTS. Built here and only here, at the end
# of warm-up and when a checkpoint is put back, so the two are the same kernel.
function _frozen_nuts(ℓ_fn, ∂ℓ_fn, Minv::Vector{Float64}, ε::Float64)
    h = AdvancedHMC.Hamiltonian(AdvancedHMC.DiagEuclideanMetric(Minv),
                                AdvancedHMC.GaussianKinetic(), ℓ_fn, ∂ℓ_fn)
    κ = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            AdvancedHMC.Leapfrog(ε), AdvancedHMC.GeneralisedNoUTurn()))
    return h, κ
end

# (ℓ, ∂ℓ) closures for the tempered logdensity g(y)=(prior+jac)+β·loglike from a
# `parts(y)->(pj,ll)` closure, gradient by ForwardDiff (works for NereusTarget
# via _logdensity_parts and for any differentiable analytic target).
function _temper_closures(parts, β::Float64)
    g = y -> (pj_ll = parts(y); pj_ll[1] + β * pj_ll[2])
    ℓ_fn  = g
    ∂ℓ_fn = y -> (g(y), ForwardDiff.gradient(g, y))
    return ℓ_fn, ∂ℓ_fn
end

# Warm-up one temperature (Stan windowed adaptor → step size + diagonal mass
# matrix), run on for another `n_warmup` iterations with what it adapted, then
# FREEZE that kernel: the adapted diagonal metric and the dual-averaged step
# size. The frozen kernel used to take the variance of those last iterations
# as its metric and a step size re-found at their last point by
# `find_good_stepsize`, a one-point heuristic that only seeds the dual
# averaging. On an RV fit it gave the β = 1 rung 0.8 or 1.6, where the adapted
# 0.2 accepts 0.9: every trajectory diverged and the cold chain kept one point
# for all 800 sweeps.
#
# The loop is `AdvancedHMC.sample(rng, h0, κ0, θ0, 2n_warmup, adaptor, n_warmup;
# drop_warmup = true)` written out, step for step, so that the seams can be
# re-cut at the adaptation's phase boundaries with every other chain of the
# core (`_hmc_recut!`; `recut` the run's `_HMCRecut`, `nothing` without angles
# to re-cut). `vote`: whether this chain's visits place the seams (the β = 1
# rung's); `c` its index at the rendezvous. Returns `nothing` if the rendezvous
# was aborted, which only a failed warm-up of another chain does.
function _warmup_temp(rng, ℓ_fn, ∂ℓ_fn, θ0::Vector{Float64}, n_warmup::Int,
                       target_accept::Float64, dim::Int;
                       recut::Union{Nothing,_HMCRecut} = nothing, vote::Bool = false,
                       c::Int = 0)
    metric0 = AdvancedHMC.DiagEuclideanMetric(dim)
    h0 = AdvancedHMC.Hamiltonian(metric0, AdvancedHMC.GaussianKinetic(), ℓ_fn, ∂ℓ_fn)
    ε0 = AdvancedHMC.find_good_stepsize(rng, h0, θ0)
    integ0 = AdvancedHMC.Leapfrog(ε0)
    κ0 = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integ0, AdvancedHMC.GeneralisedNoUTurn()))
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric0),
        AdvancedHMC.StepSizeAdaptor(target_accept, integ0))
    cs = recut === nothing ? nothing : _HMCRecutChain(recut, adaptor, n_warmup; vote)
    hw, t = AdvancedHMC.sample_init(rng, h0, θ0)
    z, κw = t.z, κ0
    for i in 1:(2 * n_warmup)
        t = AdvancedHMC.transition(rng, hw, κw, z)
        tstat = AdvancedHMC.stat(t)
        hw, κw, _ = AdvancedHMC.Adaptation.adapt!(hw, κw, adaptor, i, n_warmup, t.z.θ,
                                                  tstat.acceptance_rate)
        z = t.z
        if cs !== nothing && i <= cs.last_mark
            _hmc_trace!(cs, z.θ)
            if i in cs.marks
                r = _hmc_recut!(cs, i, c, z, hw, adaptor)
                r === nothing && return nothing
                _, z, hw = r
            end
        end
    end
    θ_final = copy(z.θ)
    M⁻¹ = copy(hw.metric.M⁻¹)
    ε = Float64(AdvancedHMC.nom_step_size(κw.τ.integrator))
    h, κ = _frozen_nuts(ℓ_fn, ∂ℓ_fn, M⁻¹, ε)
    return h, κ, θ_final, M⁻¹, ε
end

# The default ladder: β = 0, then `n_temps - 1` rungs geometric on
# [beta_min, 1] (see `beta_min` in `sample_pt_hmc`).
function _pt_hmc_ladder(n_temps::Int, beta_min::Float64)
    n_temps >= 2 || throw(ArgumentError("n_temps must be ≥ 2"))
    return vcat(0.0, Float64[beta_min^((n_temps - 1 - i) / max(1, n_temps - 2))
                             for i in 1:(n_temps - 1)])
end

# Path-sampling-optimal ladder re-grid: place β to equalise thermodynamic length
# ∫√Var(logL) dβ. Keeps β[1]=0, β[end]=1. `varlogL` is per-temp Var(logL) on the
# current `βs`.
function _adapt_ladder(βs::Vector{Float64}, varlogL::Vector{Float64})
    n = length(βs)
    n < 3 && return βs
    d = sqrt.(max.(varlogL, 0.0)) .+ 1e-6                    # length density
    cum = zeros(n)
    @inbounds for i in 2:n
        cum[i] = cum[i - 1] + 0.5 * (d[i] + d[i - 1]) * (βs[i] - βs[i - 1])
    end
    total = cum[end]
    total > 0 || return βs
    targets = collect(range(0.0, total; length = n))
    newβ = similar(βs)
    newβ[1] = 0.0; newβ[end] = 1.0
    @inbounds for k in 2:(n - 1)
        tk = targets[k]
        j = findlast(c -> c <= tk, cum)
        j = clamp(j === nothing ? 1 : j, 1, n - 1)
        frac = (cum[j + 1] - cum[j]) > 0 ? (tk - cum[j]) / (cum[j + 1] - cum[j]) : 0.0
        newβ[k] = βs[j] + frac * (βs[j + 1] - βs[j])
    end
    # guard strict monotonicity
    @inbounds for k in 2:n
        newβ[k] = max(newβ[k], newβ[k - 1] + 1e-6)
    end
    newβ[end] = 1.0
    return newβ
end

# Everything `_pt_hmc_core`'s sweep loop carries from one sweep to the next, as
# a checkpoint stores it (src/checkpoint.jl): per temperature × walker the
# position, its logL and the frozen kernel's mass matrix and step size, plus the
# ladder, the sweep RNG stream, the ⟨logL⟩ sums, the evidence accumulator and
# the cold draws kept so far. A function, not a closure: `nL` is rebound every
# sweep and a closure capturing it would box it.
_pt_hmc_snapshot(sweep, βs, states, sweep_rng, sumL, sumL2, nL, evidence_acc,
                 cold_ys, swap_acc) =
    (; sweep, betas = βs, θ = map(st -> st.θ, states),
       logL = map(st -> st.logL, states), Minv = map(st -> st.Minv, states),
       ε = map(st -> st.ε, states), sweep_rng, sumL, sumL2, nL,
       evidence = evidence_acc, cold_ys, swap_acc)

"""
    _pt_hmc_core(parts, dim, init_ys, βs; kwargs...) -> NamedTuple

Target-agnostic Hamiltonian-PT engine. `parts(y) -> (prior+jac, loglike)` (must
be ForwardDiff-differentiable); `init_ys` = one start per (temp×walker) in
sampler space. Returns `(cold_ys, report, meanlogL, varlogL, swap_acc)`,
`swap_acc[i]` the fraction of swaps accepted between rungs `i` and `i + 1`.

Checkpointing: `save(state)`, when given, receives the `_pt_hmc_snapshot` after
warm-up, every `checkpoint_interval` seconds and after the last sweep. A
`saved` snapshot (read back from a checkpoint; `init_ys` is then `nothing`)
replaces the warm-up: the chains, kernels, RNG stream and accumulators are put
back and the sweeps carry on from `saved.sweep + 1` to `n_sweeps`, exactly as
an uninterrupted run would have.

Each sweep's seed is drawn from `sweep_rng`, one per sweep, so the first k
sweeps do not depend on `n_sweeps`. They used to come from one
`rand(rng, UInt64, n_sweeps)`, whose every value changes with its length.
"""
function _pt_hmc_core(parts, dim::Int,
                       init_ys::Union{Nothing,Vector{Vector{Float64}}},
                       βs::Vector{Float64};
                       n_sweeps::Int, n_warmup::Int, swap_interval::Int,
                       target_accept::Float64, W::Int, progress::Bool,
                       rng::AbstractRNG, label::String = "PT-HMC",
                       saved::Union{Nothing,NamedTuple} = nothing,
                       save = nothing, checkpoint_interval::Real = Inf,
                       recut::Union{Nothing,NamedTuple} = nothing)
    n_temps = length(βs)
    states = Matrix{_HMCTempState}(undef, n_temps, W)
    if saved === nothing
        init_ys === nothing && throw(ArgumentError(
            "_pt_hmc_core needs `init_ys` unless it continues a `saved` state"))
        seeds = rand(rng, UInt64, n_temps * W)
        tasks = Matrix{Task}(undef, n_temps, W)
        # Every chain of every rung re-cuts the seams with the others during
        # its warm-up (`_hmc_recut!`), the β = 1 rung voting; one that ends
        # aborts the rendezvous so that none waits for it.
        rc = recut === nothing ? nothing :
             _HMCRecut(recut.params, recut.transform, _Rendezvous(n_temps * W))
        for i in 1:n_temps, w in 1:W
            idx = (i - 1) * W + w
            θ0 = init_ys[idx]; s = seeds[idx]; β = βs[i]
            tasks[i, w] = Threads.@spawn try
                crng = MersenneTwister(s)
                ℓ_fn, ∂ℓ_fn = _temper_closures(parts, β)
                r = _warmup_temp(crng, ℓ_fn, ∂ℓ_fn, θ0, n_warmup, target_accept, dim;
                                 recut = rc, vote = β == 1.0, c = idx)
                r === nothing ? nothing :
                    _HMCTempState(r[1], r[2], r[3], parts(r[3])[2], r[4], r[5])
            finally
                rc === nothing || _rendezvous_abort!(rc.rv)
            end
        end
        # In order, so a failed warm-up rethrows here; the chains it aborted
        # return `nothing`.
        warm = [fetch(tasks[i, w]) for i in 1:n_temps, w in 1:W]
        any(isnothing, warm) && error("a pt_hmc warm-up was aborted with no chain failing")
        states .= warm
    else
        # Put the saved chains back: positions, logL and the frozen kernels.
        θ_s = saved.θ::Matrix{Vector{Float64}}
        L_s = saved.logL::Matrix{Float64}
        M_s = saved.Minv::Matrix{Vector{Float64}}
        ε_s = saved.ε::Matrix{Float64}
        size(L_s) == (n_temps, W) || throw(ArgumentError(
            "the checkpoint holds $(size(L_s)) chains, not ($n_temps, $W)"))
        for i in 1:n_temps, w in 1:W
            ℓ_fn, ∂ℓ_fn = _temper_closures(parts, βs[i])
            h, κ = _frozen_nuts(ℓ_fn, ∂ℓ_fn, M_s[i, w], ε_s[i, w])
            states[i, w] = _HMCTempState(h, κ, θ_s[i, w], L_s[i, w], M_s[i, w],
                                         ε_s[i, w])
        end
    end

    # One assignment each, fresh or restored: nothing below is rebound.
    sweep0 = saved === nothing ? 0 : saved.sweep::Int
    sweep_rng = saved === nothing ? MersenneTwister(rand(rng, UInt64)) :
                copy(saved.sweep_rng::MersenneTwister)
    cold_ys = saved === nothing ? Vector{Vector{Float64}}() :
              copy(saved.cold_ys::Vector{Vector{Float64}})
    evidence_acc = saved === nothing ? EvidenceAccumulator(n_temps) :
                   deepcopy(saved.evidence::EvidenceAccumulator)
    logL_buf = Vector{Float64}(undef, n_temps)
    # running mean/var of per-temp ⟨logL⟩ (walker-averaged) for ladder adaptation
    sumL = saved === nothing ? zeros(n_temps) : copy(saved.sumL::Vector{Float64})
    sumL2 = saved === nothing ? zeros(n_temps) : copy(saved.sumL2::Vector{Float64})
    nL = saved === nothing ? 0 : saved.nL::Int
    # Swaps accepted per adjacent pair, over the sweeps so far (W per sweep).
    swap_acc = saved === nothing || !haskey(saved, :swap_acc) ? zeros(Int, n_temps - 1) :
               copy(saved.swap_acc::Vector{Int})
    pb = ProgressBar(label; total = n_sweeps, enabled = progress, start = sweep0)

    # The warm-up is the expensive part of a short run: keep it.
    save !== nothing && saved === nothing &&
        save(_pt_hmc_snapshot(0, βs, states, sweep_rng, sumL, sumL2, nL,
                              evidence_acc, cold_ys, swap_acc))
    last_ck = time()

    for sweep in (sweep0 + 1):n_sweeps
        sweep_seed = rand(sweep_rng, UInt64)
        Threads.@threads :static for idx in 1:(n_temps * W)
            i = (idx - 1) ÷ W + 1; w = (idx - 1) % W + 1
            st = states[i, w]
            crng = MersenneTwister(sweep_seed + idx)
            samp, _ = AdvancedHMC.sample(crng, st.h, st.κ, st.θ, swap_interval + 1;
                                         progress = false, verbose = false)
            st.θ = copy(samp[end])
            ll = parts(st.θ)[2]
            st.logL = isfinite(ll) ? ll : st.logL
        end
        srng = MersenneTwister(sweep_seed)
        for w in 1:W, i in 1:(n_temps - 1)
            a, b = states[i, w], states[i + 1, w]
            logα = (βs[i + 1] - βs[i]) * (a.logL - b.logL)
            if log(rand(srng)) < logα
                a.θ, b.θ = b.θ, a.θ; a.logL, b.logL = b.logL, a.logL
                swap_acc[i] += 1
            end
        end
        @inbounds for i in 1:n_temps
            s = 0.0; for w in 1:W; s += states[i, w].logL; end
            sumL[i] += s / W; sumL2[i] += (s / W)^2
        end
        nL += 1
        # One update per walker. The walker-averaged log L went in instead,
        # which TI does not mind but SS+ does: its terms are averages of
        # L^(±Δβ/2), and exp of a mean is not a mean of exp (Jensen), so with
        # more than one walker per rung every SS+ ratio was biased.
        for w in 1:W
            @inbounds for i in 1:n_temps
                logL_buf[i] = states[i, w].logL
            end
            update_evidence!(evidence_acc, logL_buf, βs)
        end
        for w in 1:W; push!(cold_ys, copy(states[n_temps, w].θ)); end
        update!(pb; n_done = sweep,
                fields = (:logL => round(states[n_temps, 1].logL, digits = 2),))
        if save !== nothing &&
           (sweep == n_sweeps || time() - last_ck >= checkpoint_interval)
            save(_pt_hmc_snapshot(sweep, βs, states, sweep_rng, sumL, sumL2, nL,
                                  evidence_acc, cold_ys, swap_acc))
            last_ck = time()
        end
    end
    finish!(pb)

    meanL = sumL ./ max(nL, 1)
    varL = max.(sumL2 ./ max(nL, 1) .- meanL .^ 2, 0.0)
    report = evidence_report(evidence_acc, βs)
    return (cold_ys = cold_ys, report = report, meanlogL = meanL, varlogL = varL,
            swap_acc = swap_acc ./ max(1, nL * W))
end

"""
    sample_pt_hmc(target::NereusTarget; kwargs...)
        -> (chains, log_evidence, report, evidence)

Hamiltonian parallel tempering for fixed-dim, ForwardDiff-differentiable models.
Returns the cold-chain `MCMCChains.Chains` (with `:lp`), the log-evidence, the
tempered `EvidenceReport` (TI, TI+, SS+, H+) and a named tuple `evidence` of
`(; bridge, laplace, tempered, tempered_ok, tempered_why, reported, betas,
swap_acc)`.

The log-evidence is chosen as pt_emcee chooses it: bridge sampling on the cold
chain when its reference has the effective samples it needs, else the tempered
SS+ when it can be trusted (`tempered_ok`), else `NaN`, with a warning saying
why: a number is not printed for an evidence the run cannot support. SS+ is
trusted when the ladder starts at β = 0 and every adjacent pair of rungs
swapped at least `swap_min` of the time; `tempered_why` gives the reason it is
not. A quotable SS+ more than `evidence_tol` nats from bridge is warned about.

# Keywords
- `n_temps=12`, `n_sweeps=1500`, `n_warmup=400`
- `n_walkers_per_temp=1` : √M-tighter ⟨logL⟩_β at ~M× gradient cost
- `swap_interval=1`, `target_accept=0.8`
- `adapt_ladder=true` : pilot run → re-grid β by thermodynamic length (√Var logL)
  → final run. Minimises TI discretisation error (the loose-error fix).
- `n_pilot=nothing` : sweeps of that pilot run; `nothing` is
  `max(100, n_sweeps ÷ 4)`. The pilot sets the ladder, so it is part of the run
  `resume` checks: extending a run to an `n_sweeps` that moves the default is
  refused until `n_pilot` is given the original run's value.
- `beta_min=1e-4` : the default ladder is β = 0 and `n_temps - 1` rungs
  geometric on `[beta_min, 1]`. It used to be quadratic from β = 0,
  `((i - 1)/(n_temps - 1))²`: on an RV + IAD companion ⟨log L⟩ is −5.6e4 at
  β = 0 and −236 at the second of 12 rungs (β = 0.008), and the TI+ segment
  between them, where the integrand climbs through four decades, put log Z at
  −5 (4 rungs: −624 to −1329) where bridge sampling and nested INS give 164 to
  166. Geometric from 1e-4 without the β = 0 rung, as pt_emcee's ladder is,
  the integrals start at 1e-4 and leave out a head worth −1.3 nats at i = 5°
  and −4.5 at i = 89.5° (TI+ 166.9 and 168.4 against bridge 165.5 and 163.9);
  with it, SS+ gives 165.5 and 163.9.
- `betas=nothing` : custom β-ladder (overrides default + disables adaptation)
- `swap_min=0.05` : the least swap acceptance between adjacent rungs at which
  the tempered SS+ is quoted (above).
- `evidence_tol=2.0` : nats; a quotable SS+ this far from bridge is warned
  about.
- `bridge_n=20_000` : proposal draws of the bridge estimate; `0` skips it.
- `warm_start=true` : start from a short pt_emcee pre-search. With it off the
  starts are scouted prior draws (`n_scouts`), and the seams of the
  full-circle angles are placed first by a short pathfinder pilot
  (`_seam_pilot!`). Either way the chains re-cut them together at the phase
  boundaries of each warm-up's adaptation (`_hmc_recut!`, voted by the β = 1
  rung).
- `n_scouts=16` : without the warm start, every chain starts from one of the
  prior draws, each moved by slice updates until it stops climbing, that the
  bound of the stranded-walker pruning keeps (src/samplers/scouts.jl), as in
  `sample_nuts`. Each chain's kernel is frozen at the end of its warm-up, so
  where the warm-up ran decides how it samples. From one prior draw each, the
  β = 1 chain of an RV + IAD companion ended its warm-up on a plateau 1,500
  nats below the mode with a metric ten times too wide; brought to the mode by
  a swap, it accepted no trajectory, and the run reported one point for all
  1000 sweeps. `0` starts every chain from one draw.
- `seed=1`, `progress=true`
- `checkpoint=nothing` : path of a state file. The sampler writes its whole
  state there (the ladder; every temperature's positions, logL, mass matrix and
  step size; the master and per-sweep RNG streams; the evidence accumulator and
  ⟨logL⟩ sums; the circular windows; the cold draws kept so far) after each
  warm-up, every `checkpoint_interval` seconds, and after the last sweep of the
  pilot and of the run, replacing the file atomically. `run_job` sets it to
  `pt_hmc_state.jls` in `output_dir` unless the job gives one.
- `checkpoint_interval::Real=900` : seconds between checkpoints during the
  sweeps.
- `resume::Bool=false` : continue from `checkpoint` instead of initialising: a
  run that was killed, or a finished one that needs more sweeps. `n_sweeps` is
  the new TOTAL, counted from the start of the original run; the cold draws
  already kept stay, and the new ones are appended. The continuation is
  bit-identical to an uninterrupted run of `n_sweeps`. Checkpoints fall
  between sweeps, so a run killed during a warm-up redoes it from the
  checkpoint before, which gives the same kernels; one killed during the
  first warm-up has no checkpoint yet and starts again. Refused, with the
  differences listed, if the checkpoint came from different data, priors or
  parameters, or a different ladder, pilot length, warm-up, walker count, swap
  interval, target acceptance, warm start or seed, or if `n_sweeps` is below
  the sweeps it already holds.
"""
function sample_pt_hmc(
    target::NereusTarget;
    n_temps::Int = 12,
    n_sweeps::Int = 1500,
    n_warmup::Int = 400,
    n_walkers_per_temp::Int = 1,
    swap_interval::Int = 1,
    target_accept::Real = 0.8,
    adapt_ladder::Bool = true,
    n_pilot::Union{Nothing,Int} = nothing,
    betas::Union{Nothing, AbstractVector} = nothing,
    warm_start::Bool = true,
    seed::Int = 1,
    progress::Bool = true,
    checkpoint::Union{Nothing,AbstractString,Symbol} = nothing,
    checkpoint_interval::Real = 900.0,
    resume::Bool = false,
    n_scouts::Int = 16,
    beta_min::Real = 1e-4,
    swap_min::Real = 0.05,
    evidence_tol::Real = 2.0,
    bridge_n::Int = 20_000,
)
    n_scouts >= 0 || throw(ArgumentError("n_scouts must be ≥ 0"))
    0 < beta_min < 1 || throw(ArgumentError("beta_min must be in (0, 1); got $beta_min"))
    target_accept = Float64(target_accept)
    ck_path = checkpoint === nothing ? nothing : String(checkpoint)
    resume && ck_path === nothing && throw(ArgumentError(
        "resume = true needs `checkpoint`, the state file to continue from"))
    n_pilot === nothing || n_pilot >= 1 || throw(ArgumentError(
        "n_pilot is a number of sweeps; got $n_pilot"))
    if !(target.transform isa PackedTransforms)
        target = NereusTarget(target.params, target.data; unconstrained = true)
    end
    dim = LogDensityProblems.dimension(target)
    rng = MersenneTwister(seed)
    W = max(1, n_walkers_per_temp)
    # `target` is rebound above, so every closure capturing it holds a Box and
    # reads it untyped. This one is the likelihood every NUTS step calls: give
    # it a binding of its own.
    parts = let target = target
        y -> _logdensity_parts(target, y)
    end

    βs = betas === nothing ? _pt_hmc_ladder(n_temps, Float64(beta_min)) :
         collect(Float64, betas)
    βs[end] = 1.0
    n_temps = length(βs)
    do_pilot = adapt_ladder && betas === nothing
    pilot_sweeps = do_pilot ? something(n_pilot, max(100, n_sweeps ÷ 4)) : 0

    function fresh_inits(n)
        # Without the warm start's re-chart, nothing here moves a seam: a
        # pathfinder pilot places the windows first (`_seam_pilot!`).
        warm_start || isempty(seam_indices(target.params)) ||
            _seam_pilot!(target, rand(rng, 1:typemax(Int32)))
        b = warm_start ? _warmstart_points(target, n, rng) :
                         _cold_starts(target, n, rng, n_scouts)
        [transform_forward(x, target.transform) for x in b]
    end

    # --- Checkpoint / resume (src/checkpoint.jl) ------------------------
    # The run has two phases, each one `_pt_hmc_core`: the pilot (when the
    # ladder adapts) and the final run. A checkpoint names its phase and holds
    # that core's state, the master RNG (which the warm start and the next
    # phase's warm-up draw from) and the circular windows (the warm start's
    # pre-search moves them on `target.params`). The fingerprint takes the
    # ladder as built, before the pilot re-grids it.
    ck_fp = run_fingerprint(target.params, target.data; n_temps, ladder = copy(βs),
        adapt_ladder = do_pilot, n_pilot = pilot_sweeps, n_warmup,
        n_walkers_per_temp = W, swap_interval, target_accept, warm_start, seed,
        # The seams re-cut during each warm-up (`_hmc_recut!`): a checkpoint
        # from before that is another run, as is one from before polar
        # charts moved.
        seam_recut = :adaptation_phases, polar_chart_key(target.params)...,
        # Not the default: the final run's starts are drawn after the pilot.
        (n_scouts == 16 ? (;) : (; n_scouts))...)
    ck = nothing
    if resume
        ck = try
            read_checkpoint(ck_path, "pt_hmc", ck_fp)
        catch e
            (e isa ArgumentError && do_pilot && n_pilot === nothing &&
             occursin(r"these differ: .*\bn_pilot\b", e.msg)) || rethrow()
            throw(ArgumentError(e.msg * ". The pilot defaults to " *
                "max(100, n_sweeps ÷ 4) sweeps, so this n_sweeps changed it; " *
                "pass the original run's `n_pilot` to continue it"))
        end
    end
    phase = ck === nothing ? :none : ck.phase::Symbol
    if phase === :final
        sweep0 = ck.core.sweep::Int
        sweep0 <= n_sweeps || throw(ArgumentError(
            "the checkpoint is at sweep $sweep0; n_sweeps = $n_sweeps would end before it"))
    end
    # Resume: put back what lives outside the cores, in place (`fresh_inits`
    # captured `rng`, and rebinding it would box it).
    if ck !== nothing
        copy!(rng, ck.rng::MersenneTwister)
        names_ = target.params.layout.unfrozen_names
        for (nm, (lo, _)) in ck.windows::Dict{String,Tuple{Float64,Float64}}
            set_circular_window!(target.params, findfirst(==(nm), names_), lo;
                                 transforms = (target.transform,))
        end
    end
    # A core's `save`: the checkpoint of phase `ph` around the core's state.
    # The windows the warm-ups re-cut (`_hmc_recut!`): the target's own.
    recut = _has_recut_angles(target.params) ?
            (; params = target.params, transform = target.transform) : nothing
    saver(ph::Symbol) = ck_path === nothing ? nothing : core -> begin
        write_checkpoint(ck_path, "pt_hmc", ck_fp, (; phase = ph, rng,
            windows = circular_windows(target.params), core))
        hook = _PT_HMC_CHECKPOINT_HOOK[]
        hook === nothing || hook(ph, core.sweep)
        nothing
    end

    # adaptive ladder: short pilot → re-grid β by √Var(logL) → final run
    if do_pilot && phase !== :final
        pilot = _pt_hmc_core(parts, dim, phase === :pilot ? nothing :
                             fresh_inits(n_temps * W), βs;
                             n_sweeps = pilot_sweeps,
                             n_warmup = max(150, n_warmup ÷ 2),
                             swap_interval = swap_interval, target_accept = target_accept,
                             W = W, progress = progress, rng = rng, label = "PT-HMC pilot",
                             saved = phase === :pilot ? ck.core : nothing,
                             save = saver(:pilot),
                             checkpoint_interval = checkpoint_interval, recut)
        βs = _adapt_ladder(βs, pilot.varlogL)
    elseif phase === :final
        βs = copy(ck.core.betas::Vector{Float64})
    end

    res = _pt_hmc_core(parts, dim, phase === :final ? nothing :
                       fresh_inits(n_temps * W), βs;
                       n_sweeps = n_sweeps, n_warmup = n_warmup,
                       swap_interval = swap_interval, target_accept = target_accept,
                       W = W, progress = progress, rng = rng,
                       saved = phase === :final ? ck.core : nothing,
                       save = saver(:final), checkpoint_interval = checkpoint_interval, recut)

    n_post = length(res.cold_ys)
    pnames = vcat(Symbol.(target.params.layout.unfrozen_names), [:lp])
    mat = Matrix{Float64}(undef, n_post, dim + 1)
    @inbounds for i in 1:n_post
        y = res.cold_ys[i]
        mat[i, 1:dim] = transform_inverse(y, target.transform)
        pj, ll = _logdensity_parts(target, y)
        mat[i, dim + 1] = (isfinite(pj) && isfinite(ll)) ? pj + ll : -Inf
    end
    chains = MCMCChains.Chains(mat, pnames)
    report = res.report
    ev = _pt_hmc_evidence(target, chains, report, βs, res.swap_acc, Float64(swap_min),
                          Float64(evidence_tol), bridge_n, seed)
    if progress
        @info "PT-HMC: $(n_post) cold samples, $(n_temps) temps × $(W) walkers" *
              (adapt_ladder && betas === nothing ? " (adaptive ladder)" : "")
        @info "  TI+ = $(round(report.ti_plus[1], digits=2)) ± $(round(report.ti_plus[2], digits=3))  " *
              "SS+ = $(round(report.ss_plus[1], digits=2))  " *
              "H+ = $(round(report.hybrid[1], digits=2))" *
              (ev.tempered_ok ? "" : "  (not quotable: $(ev.tempered_why))")
        @info "  swap acceptance: " * join(string.(round.(res.swap_acc, digits = 2)), " ")
        @info "  bridge = $(round(ev.bridge, digits=2))  log Z reported: " *
              (isfinite(ev.log_z) ? "$(round(ev.log_z, digits=2)) ($(ev.reported))" : "none")
    end
    return chains, ev.log_z, report,
           (; bridge = ev.bridge, laplace = ev.laplace, tempered = report.ss_plus[1],
              ev.tempered_ok, ev.tempered_why, ev.reported, betas = copy(βs),
              swap_acc = res.swap_acc)
end

# Whether the tempered SS+ can be quoted: `(ok, why)`, `why` empty when it can:
# `_tempered_quotable` (src/samplers/evidence.jl), which pt_emcee and the
# summary writers share, and SS+ finite. The ladder must reach β = 0, or the
# [0, β₁] head is not in the integral
# (it was worth −1.3 to −4.5 nats on the RV + IAD companion from β₁ = 1e-4); and
# every adjacent pair of rungs must overlap, which a stepping-stone ratio needs
# and the swaps between them measure: at least `swap_min` of them accepted.
# SS+ rather than TI+: from β = 0 the integrand climbs through four decades in
# the first segment, which PCHIP does not resolve -- TI+ came out 1.1 to 3.6
# nats low at 8 and 12 rungs, its error 6.6 to 42 nats, where SS+ matched
# bridge sampling to 0.1.
function _pt_hmc_tempered_ok(report::EvidenceReport, βs::AbstractVector,
                             swap_acc::AbstractVector, swap_min::Float64)
    ok, why = _tempered_quotable(βs, swap_acc; swap_min)
    ok || return (ok, why)
    isfinite(report.ss_plus[1]) || return (false, "SS+ is not finite")
    return (true, "")
end

# The log-evidence pt_hmc reports, as pt_emcee picks its own: bridge sampling
# on the cold chain when its reference `q` has the effective samples it needs
# (d + d(d+1)/2; with fewer it is biased low and no diagnostic shows it), else
# SS+ when `_pt_hmc_tempered_ok`, else NaN. A warning says why whenever the
# tempered value is not quotable, and when bridge and a quotable SS+ disagree
# by more than `tol` nats.
function _pt_hmc_evidence(target::NereusTarget, chains::MCMCChains.Chains,
                          report::EvidenceReport, βs::AbstractVector,
                          swap_acc::AbstractVector, swap_min::Float64, tol::Float64,
                          bridge_n::Int, seed::Int)
    ok, why = _pt_hmc_tempered_ok(report, βs, swap_acc, swap_min)
    lz_bridge = NaN
    if bridge_n > 0 && target.transform !== nothing
        try
            b = bridge_evidence(target, chains; n_proposal = bridge_n, seed = seed)
            b.converged && isfinite(b.log_z) && (lz_bridge = b.log_z)
        catch err
            @debug "sample_pt_hmc: bridge evidence unavailable" exception = err
        end
    end
    if isfinite(lz_bridge)
        d = LogDensityProblems.dimension(target)
        need = d + d * (d + 1) ÷ 2
        n_eff = try
            _bridge_min_ess(chains, target.params)   # folded, as bridge reads them
        catch err
            @debug "sample_pt_hmc: ESS unavailable for the bridge guard" exception = err
            NaN
        end
        if !(isfinite(n_eff) && n_eff >= need)
            @warn "sample_pt_hmc: bridge is not reported — its reference is " *
                  "fitted from too few effective samples (min ESS " *
                  "$(isfinite(n_eff) ? round(Int, n_eff) : "?") < $need free " *
                  "parameters in q), where it is biased low." bridge = lz_bridge
            lz_bridge = NaN
        end
    end
    laplace = try
        mode_laplace_evidence(target, chains)
    catch err
        @debug "sample_pt_hmc: mode-Laplace evidence unavailable" exception = err
        NaN
    end
    ssp = report.ss_plus[1]
    if isfinite(lz_bridge)
        ok || @warn "sample_pt_hmc: the tempered evidence (TI/TI+/SS+/H+) is not " *
                    "quotable: $why. Reporting bridge." bridge = lz_bridge ss_plus = ssp ti_plus = report.ti_plus[1]
        ok && abs(ssp - lz_bridge) > tol &&
            @warn "sample_pt_hmc: bridge and the tempered SS+ differ by " *
                  "$(round(abs(ssp - lz_bridge), digits=1)) nats. Reporting bridge; " *
                  "check that the posterior is not multimodal." bridge = lz_bridge ss_plus = ssp
        return (; log_z = lz_bridge, bridge = lz_bridge, laplace, tempered_ok = ok,
                  tempered_why = why, reported = "bridge")
    elseif ok
        return (; log_z = ssp, bridge = NaN, laplace, tempered_ok = ok, tempered_why = why,
                  reported = "ss_plus")
    end
    @warn "sample_pt_hmc: no quotable log-evidence for this run. The tempered " *
          "stack is not ($why) and bridge sampling is unavailable. log_evidence " *
          "is NaN; add rungs (n_temps) or sweeps." ss_plus = ssp ti_plus = report.ti_plus[1] laplace = laplace
    return (; log_z = NaN, bridge = NaN, laplace, tempered_ok = ok, tempered_why = why,
              reported = "none")
end
