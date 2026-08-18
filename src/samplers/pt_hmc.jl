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
# NereusTarget wrapper. An ADAPTIVE ladder (pilot run → re-grid β by
# path-sampling thermodynamic length ∝ √Var(logL)) minimises the TI
# discretisation error that a fixed power-law ladder leaves near β=1.

using AdvancedHMC
using ForwardDiff
using MCMCChains
using Random
using Statistics: mean, var

# One NUTS chain pinned to a temperature.
mutable struct _HMCTempState
    h::AdvancedHMC.Hamiltonian
    κ::Any
    θ::Vector{Float64}
    logL::Float64
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
# matrix), then FREEZE a kernel whose metric is the post-warm-up sample variance
# and whose step size is re-found — avoids reaching into adaptor internals.
function _warmup_temp(rng, ℓ_fn, ∂ℓ_fn, θ0::Vector{Float64}, n_warmup::Int,
                       target_accept::Float64, dim::Int)
    metric0 = AdvancedHMC.DiagEuclideanMetric(dim)
    h0 = AdvancedHMC.Hamiltonian(metric0, AdvancedHMC.GaussianKinetic(), ℓ_fn, ∂ℓ_fn)
    ε0 = AdvancedHMC.find_good_stepsize(rng, h0, θ0)
    integ0 = AdvancedHMC.Leapfrog(ε0)
    κ0 = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            integ0, AdvancedHMC.GeneralisedNoUTurn()))
    adaptor = AdvancedHMC.StanHMCAdaptor(
        AdvancedHMC.MassMatrixAdaptor(metric0),
        AdvancedHMC.StepSizeAdaptor(target_accept, integ0))
    samples, _ = AdvancedHMC.sample(rng, h0, κ0, θ0, 2 * n_warmup, adaptor, n_warmup;
                                    drop_warmup = true, progress = false, verbose = false)
    θ_final = isempty(samples) ? θ0 : copy(samples[end])
    M⁻¹ = ones(Float64, dim)
    if length(samples) > 4
        Y = reduce(hcat, samples)
        @inbounds for d in 1:dim
            v = var(@view Y[d, :]); M⁻¹[d] = (isfinite(v) && v > 0) ? v : 1.0
        end
    end
    metric = AdvancedHMC.DiagEuclideanMetric(M⁻¹)
    h = AdvancedHMC.Hamiltonian(metric, AdvancedHMC.GaussianKinetic(), ℓ_fn, ∂ℓ_fn)
    ε = AdvancedHMC.find_good_stepsize(rng, h, θ_final)
    κ = AdvancedHMC.HMCKernel(AdvancedHMC.Trajectory{AdvancedHMC.MultinomialTS}(
            AdvancedHMC.Leapfrog(ε), AdvancedHMC.GeneralisedNoUTurn()))
    return h, κ, θ_final
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

"""
    _pt_hmc_core(parts, dim, init_ys, βs; kwargs...) -> NamedTuple

Target-agnostic Hamiltonian-PT engine. `parts(y) -> (prior+jac, loglike)` (must
be ForwardDiff-differentiable); `init_ys` = one start per (temp×walker) in
sampler space. Returns `(cold_ys, report, meanlogL, varlogL)`.
"""
function _pt_hmc_core(parts, dim::Int, init_ys::Vector{Vector{Float64}},
                       βs::Vector{Float64};
                       n_sweeps::Int, n_warmup::Int, swap_interval::Int,
                       target_accept::Float64, W::Int, progress::Bool,
                       rng::AbstractRNG, label::String = "PT-HMC")
    n_temps = length(βs)
    states = Matrix{_HMCTempState}(undef, n_temps, W)
    seeds = rand(rng, UInt64, n_temps * W)
    tasks = Matrix{Task}(undef, n_temps, W)
    for i in 1:n_temps, w in 1:W
        θ0 = init_ys[(i - 1) * W + w]; s = seeds[(i - 1) * W + w]; β = βs[i]
        tasks[i, w] = Threads.@spawn begin
            crng = MersenneTwister(s)
            ℓ_fn, ∂ℓ_fn = _temper_closures(parts, β)
            h, κ, θf = _warmup_temp(crng, ℓ_fn, ∂ℓ_fn, θ0, n_warmup, target_accept, dim)
            _HMCTempState(h, κ, θf, parts(θf)[2])
        end
    end
    for i in 1:n_temps, w in 1:W; states[i, w] = fetch(tasks[i, w]); end

    cold_ys = Vector{Vector{Float64}}()
    evidence_acc = EvidenceAccumulator(n_temps)
    logL_buf = Vector{Float64}(undef, n_temps)
    # running mean/var of per-temp ⟨logL⟩ (walker-averaged) for ladder adaptation
    sumL = zeros(n_temps); sumL2 = zeros(n_temps); nL = 0
    sweep_seeds = rand(rng, UInt64, n_sweeps)
    pb = ProgressBar(label; total = n_sweeps, enabled = progress)

    for sweep in 1:n_sweeps
        Threads.@threads :static for idx in 1:(n_temps * W)
            i = (idx - 1) ÷ W + 1; w = (idx - 1) % W + 1
            st = states[i, w]
            crng = MersenneTwister(sweep_seeds[sweep] + idx)
            samp, _ = AdvancedHMC.sample(crng, st.h, st.κ, st.θ, swap_interval + 1;
                                         progress = false, verbose = false)
            st.θ = copy(samp[end])
            ll = parts(st.θ)[2]
            st.logL = isfinite(ll) ? ll : st.logL
        end
        srng = MersenneTwister(sweep_seeds[sweep])
        for w in 1:W, i in 1:(n_temps - 1)
            a, b = states[i, w], states[i + 1, w]
            logα = (βs[i + 1] - βs[i]) * (a.logL - b.logL)
            if log(rand(srng)) < logα
                a.θ, b.θ = b.θ, a.θ; a.logL, b.logL = b.logL, a.logL
            end
        end
        @inbounds for i in 1:n_temps
            s = 0.0; for w in 1:W; s += states[i, w].logL; end
            logL_buf[i] = s / W
            sumL[i] += logL_buf[i]; sumL2[i] += logL_buf[i]^2
        end
        nL += 1
        update_evidence!(evidence_acc, logL_buf, βs)
        for w in 1:W; push!(cold_ys, copy(states[n_temps, w].θ)); end
        update!(pb; n_done = sweep,
                fields = (:logL => round(states[n_temps, 1].logL, digits = 2),))
    end
    finish!(pb)

    meanL = sumL ./ max(nL, 1)
    varL = max.(sumL2 ./ max(nL, 1) .- meanL .^ 2, 0.0)
    report = evidence_report(evidence_acc, βs)
    return (cold_ys = cold_ys, report = report, meanlogL = meanL, varlogL = varL)
end

"""
    sample_pt_hmc(target::NereusTarget; kwargs...) -> (chains, log_evidence, report)

Hamiltonian parallel tempering for fixed-dim, ForwardDiff-differentiable models.
Returns the cold-chain `MCMCChains.Chains` (with `:lp`), the TI⁺ log-evidence,
and the `EvidenceReport`.

# Keywords
- `n_temps=12`, `n_sweeps=1500`, `n_warmup=400`
- `n_walkers_per_temp=1` : √M-tighter ⟨logL⟩_β at ~M× gradient cost
- `swap_interval=1`, `target_accept=0.8`
- `adapt_ladder=true` : pilot run → re-grid β by thermodynamic length (√Var logL)
  → final run. Minimises TI discretisation error (the loose-error fix).
- `betas=nothing` : custom β-ladder (overrides default + disables adaptation)
- `warm_start=true`, `seed=1`, `progress=true`
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
    betas::Union{Nothing, AbstractVector} = nothing,
    warm_start::Bool = true,
    seed::Int = 1,
    progress::Bool = true,
)
    target_accept = Float64(target_accept)
    if !(target.transform isa PackedTransforms)
        target = NereusTarget(target.params, target.data; unconstrained = true)
    end
    dim = LogDensityProblems.dimension(target)
    rng = MersenneTwister(seed)
    W = max(1, n_walkers_per_temp)
    parts = y -> _logdensity_parts(target, y)

    βs = betas === nothing ?
         Float64[((i - 1) / (n_temps - 1))^2 for i in 1:n_temps] :
         collect(Float64, betas)
    βs[end] = 1.0
    n_temps = length(βs)

    function fresh_inits(n)
        b = warm_start ? _warmstart_points(target, n, rng) :
                         [_draw_from_prior(target, rng) for _ in 1:n]
        [transform_forward(x, target.transform) for x in b]
    end

    # adaptive ladder: short pilot → re-grid β by √Var(logL) → final run
    if adapt_ladder && betas === nothing
        pilot = _pt_hmc_core(parts, dim, fresh_inits(n_temps * W), βs;
                             n_sweeps = max(100, n_sweeps ÷ 4),
                             n_warmup = max(150, n_warmup ÷ 2),
                             swap_interval = swap_interval, target_accept = target_accept,
                             W = W, progress = progress, rng = rng, label = "PT-HMC pilot")
        βs = _adapt_ladder(βs, pilot.varlogL)
    end

    res = _pt_hmc_core(parts, dim, fresh_inits(n_temps * W), βs;
                       n_sweeps = n_sweeps, n_warmup = n_warmup,
                       swap_interval = swap_interval, target_accept = target_accept,
                       W = W, progress = progress, rng = rng)

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
    if progress
        @info "PT-HMC: $(n_post) cold samples, $(n_temps) temps × $(W) walkers" *
              (adapt_ladder && betas === nothing ? " (adaptive ladder)" : "")
        @info "  TI+ = $(round(report.ti_plus[1], digits=2)) ± $(round(report.ti_plus[2], digits=3))  " *
              "SS+ = $(round(report.ss_plus[1], digits=2))  " *
              "H+ = $(round(report.hybrid[1], digits=2))"
    end
    return chains, report.ti_plus[1], report
end
