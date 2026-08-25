# Reference-path evidence: thermodynamic integration that never touches the prior.
#
# WHY THIS EXISTS. Every beta-path estimator in evidence.jl (TI, TI+, SS+, H+)
# tempers from the PRIOR to the posterior, so all of them depend on
# <log L>_beta at the hot end. On a signal-locked posterior that path has a
# first-order phase transition: <log L>(beta) jumps almost discontinuously near
# beta = 1, one pair of adjacent temperatures sticks at ~0.6% swap acceptance no
# matter how many rungs are added, and TI/SS+/H+ come out biased LOW by
# 10^2-10^4 nats. Measured on HD 18599: TI+ sat 118 nats from the truth while
# self-reporting an error of 0.48, and adding rungs, cooling beta_min or
# enabling the adaptive ladder each moved it FURTHER away. That is a bias that
# does not shrink with computation and cannot be tuned out, because it is a
# property of the PATH, not of the estimator -- the estimators themselves
# reproduce an analytic log Z to <0.01 nats on a well-behaved target.
#
# The fix is to move the cold end. Let q be a NORMALISED reference fitted to the
# posterior (the same Gaussian / Student-t that bridge_evidence builds). Define
#
#     gamma_beta(y) = q(y) * exp(beta * f(y)),    f(y) = log p*(y) - log q(y)
#
# so gamma_0 = q with Z_0 = 1 (and we can sample it EXACTLY), and gamma_1 = p*
# with Z_1 = Z. Then
#
#     log Z = INT_0^1 E_{p_beta}[f] d beta                         (path sampling)
#
# and the same run gives the annealed-importance-sampling estimator
#
#     log w = SUM_k (beta_k - beta_{k-1}) f(y_{k-1}),   Z ~ mean(w)
#
# Both are computed here. Unlike TI+/SS+/H+ -- which all read one shared
# mean_logL array, so their mutual agreement proves nothing -- these two use the
# particle weights and the path integral respectively, so when they agree that
# IS evidence.
#
# The path is short because q already resembles the posterior, so there is no
# transition to resolve and no expectation taken under the prior.

using Statistics: mean, std, cov
using LinearAlgebra: cholesky, logdet, Symmetric, I
using Random: MersenneTwister, randn, rand
using SpecialFunctions: loggamma

export reference_path_evidence

"""
    reference_path_evidence(target, chains; n_particles, n_beta, n_steps,
                            proposal, seed, step_scale) -> NamedTuple

Log-evidence by thermodynamic integration along a path from a fitted reference
`q` to the posterior, never passing through the prior.

Returns
`(; log_z, log_z_ti, log_z_ais, se, ess, n_particles, n_beta, accept, proposal)`.

`log_z_ais` is the annealed-importance-sampling estimate and is what `log_z`
reports. It is unbiased in `Z` for ANY kernel that leaves `p_beta` invariant, so
it does not care how well the particles mix.

`log_z_ti` integrates `E_{p_beta}[f]` along the same path. Path sampling DOES
require the particles to be equilibrated at each rung, so `log_z_ti` degrades
when `n_steps` is too small. Measured on the 22-D analytic target with a
deliberately mismatched Student-t reference, TI's error runs
-3.21 / -0.79 / -0.10 / -0.02 nats at `n_steps` = 2 / 8 / 32 / 128 while AIS
stays within 0.03 throughout.

**That makes `log_z_ais - log_z_ti` a mixing diagnostic**, not a redundancy: the
two share the particles but not the estimator algebra, and only one of them is
sensitive to equilibration. Contrast TI+/SS+/H+ in `evidence.jl`, which all read
one shared `mean_logL` array — their mutual agreement cannot detect a common
bias, and on HD 18599 they agreed to 2.6 nats while all being 147 nats wrong.
If AIS and TI disagree here, raise `n_steps` (and watch `ess` rise with it).

`chains` must be the beta = 1 chain in the target's own parameter space, and the
target must be `unconstrained = true`, exactly as for [`bridge_evidence`](@ref):
chains are stored BOUNDED, the log-density wants UNCONSTRAINED, so the draws are
mapped through `transform_forward` before `q` is fitted.

`proposal` is `:gaussian` (default) or `:student` (nu = 4). Note this default is
the OPPOSITE of `bridge_evidence`, deliberately. Bridge only needs `q` to
overlap the posterior, so heavy tails are free insurance there. Here `q` is the
cold end of a path the particles must be annealed along, so a reference that is
much broader than the target lengthens the path and costs equilibration: on the
analytic target the Student-t reference needed ~16x more `n_steps` to bring TI
into agreement, at no benefit to AIS. Use `:student` if the posterior is heavy
tailed enough that a Gaussian would miss its bulk.
"""
function reference_path_evidence(target::NereusTarget, chains::MCMCChains.Chains;
                                 n_particles::Int = 512,
                                 n_beta::Int = 64,
                                 n_steps::Int = 8,
                                 proposal::Symbol = :gaussian,
                                 seed::Int = 1,
                                 step_scale::Real = 1.0,
                                 beta_power::Real = 3.0,
                                 ess_threshold::Real = 0.5,
                                 ess_target::Real = 0.95)
    names = target.params.layout.unfrozen_names
    d = length(names)
    d >= 1 || return _rp_bad(n_beta, proposal)

    # --- posterior draws in the target's own (unconstrained) space -----------
    cols = [vec(Array(chains[:, Symbol(nm), :])) for nm in names]
    N = length(cols[1])
    pt = target.transform
    Y = Matrix{Float64}(undef, d, N)
    @inbounds for i in 1:N
        xi = Float64[cols[j][i] for j in 1:d]
        Y[:, i] = pt === nothing ? xi : transform_forward(xi, pt)
    end
    Y = Y[:, vec(all(isfinite, Y; dims = 1))]
    size(Y, 2) >= 2d + 2 || return _rp_bad(n_beta, proposal)

    logp(y) = (a = _logdensity_parts(target, y); Float64(a[1]) + Float64(a[2]))
    return _reference_path_core(logp, Y; n_particles, n_beta, n_steps,
                                proposal, seed, step_scale, beta_power,
                                ess_threshold, ess_target)
end


"ESS of a log-weight vector, ignoring non-finite (zero-weight) entries."
function _rp_ess(logw::AbstractVector{Float64})
    lw = filter(isfinite, logw)
    isempty(lw) && return 0.0
    m = maximum(lw); w = exp.(lw .- m)
    return sum(w)^2 / sum(abs2, w)
end

_rp_bad(n_beta, proposal) =
    (; log_z = NaN, log_z_ti = NaN, log_z_ais = NaN, se = NaN, ess = NaN,
       support_frac = NaN, n_resample = 0, n_particles = 0, n_effective = 0,
       n_beta = n_beta, accept = NaN, proposal = proposal)

"""
    _reference_path_core(logp, Y; kwargs...) -> NamedTuple

The estimator proper: `logp` is the UNNORMALISED log target on R^d and `Y` a
`d x N` matrix of posterior draws in that same space, used only to fit the
reference. Kept separate from `reference_path_evidence` so it can be validated
against a target whose log Z is known analytically, with no Nereus plumbing in
the way.
"""
function _reference_path_core(logp::Function, Y::AbstractMatrix{Float64};
                              n_particles::Int = 512,
                              n_beta::Int = 64,
                              n_steps::Int = 8,
                              proposal::Symbol = :gaussian,
                              seed::Int = 1,
                              step_scale::Real = 1.0,
                              beta_power::Real = 3.0,
                              ess_threshold::Real = 0.5,
                              ess_target::Real = 0.95)
    d = size(Y, 1)
    bad = _rp_bad(n_beta, proposal)
    # A reference fitted from fewer draws than it has free covariance entries is
    # rank-deficient; the 1e-10*I ridge below would still factor, so it would
    # return a confident number built on nothing. Refuse instead.
    (d >= 1 && size(Y, 2) >= 2d + 2 && n_particles >= 2) || return bad

    # --- the reference q, normalised ----------------------------------------
    μ  = vec(mean(Y; dims = 2))
    Σ  = cov(Y; dims = 2) + 1e-10 * I(d)
    Lc = cholesky(Symmetric(Σ)).L
    ν  = 4.0
    ldL = logdet(Lc)
    function logq(y)
        z = Lc \ (y .- μ)
        s = sum(abs2, z)
        return proposal === :gaussian ?
            -0.5 * s - 0.5 * d * log(2π) - ldL :
            loggamma((ν + d) / 2) - loggamma(ν / 2) - 0.5 * d * log(ν * π) -
                ldL - 0.5 * (ν + d) * log1p(s / ν)
    end

    rng = MersenneTwister(seed)
    nu_i = Int(ν)
    drawq() = proposal === :gaussian ? μ .+ Lc * randn(rng, d) :
              μ .+ Lc * (randn(rng, d) / sqrt(sum(abs2, randn(rng, nu_i)) / ν))

    # f = log p* - log q. Non-finite p* (outside support) is -Inf and the
    # particle simply never moves there.
    f(y) = (v = logp(y); isfinite(v) ? v - logq(y) : -Inf)

    # --- initialise at beta = 0: EXACT draws from q --------------------------
    P  = Matrix{Float64}(undef, d, n_particles)
    fv = Vector{Float64}(undef, n_particles)
    # ONE draw per particle, no retry. Retrying until f is finite would be
    # rejection sampling from q RESTRICTED to the support, i.e. drawing from
    # q/Z_q with Z_q = P_q(supp p*) < 1 -- and the AIS/SMC identity assumes
    # draws from the NORMALISED q with Z_0 = 1. Feeding it restricted draws
    # makes it estimate Z/Z_q, overestimating log Z by -log Z_q. That is not
    # hypothetical: it read +4.5 nats high against a well-conditioned bridge
    # estimate on a 13-D RV posterior, and the sign is diagnostic, because an
    # SMC log-Z estimate is unbiased in Z and so should skew LOW by Jensen.
    #
    # A particle drawn outside the support is simply a zero-weight particle: it
    # carries w = 0 along the whole path and still counts in N.
    @inbounds for i in 1:n_particles
        y = drawq()
        P[:, i] = y
        fv[i] = f(y)
    end
    n_in = count(isfinite, fv)
    n_in > 0 || return bad
    # Fraction of reference draws inside the target's support, and now a real
    # efficiency figure: (1 - support_frac) of the particles are dead from the
    # start, so a low value means most of the budget is wasted and the estimate
    # rests on few particles. It also decides whether the path integral exists
    # at all -- TI integrates E_{p_beta}[f], which at beta = 0 is E_q[f] = -Inf
    # whenever q places mass where log p* = -Inf, so log_z_ti is reported only
    # when this is 1. The AIS weights remain valid either way.
    support_frac = n_in / n_particles

    # Non-uniform beta grid. With a heavy-tailed reference, f = log p* - log q
    # has its curvature at the COLD end (where q and the target differ most),
    # so a uniform grid under-resolves exactly the region that matters and the
    # trapezoid term decays only like 1/n_beta. beta_k = (k/K)^beta_power
    # concentrates rungs near 0. It does not affect the AIS weights (which are
    # exact for any schedule) -- it is what makes the TI cross-check sharp
    # enough to be worth reading.
    # ADAPTIVE beta schedule. A fixed grid takes the same step everywhere, but
    # the path is not uniformly hard: on a real posterior the early rungs (where
    # q and the target still differ) need fine steps and the late ones do not.
    # Fixed grids therefore either waste rungs or blow through the hard part --
    # measured on a 13-D RV target, refining a fixed grid from 64 to 1024 rungs
    # made ESS WORSE (1.5 -> 1.1), because the difficulty is not where the extra
    # rungs went.
    #
    # Instead pick each beta so the incremental weights lose a controlled
    # fraction of ESS (Del Moral/Jasra adaptive SMC): bisect for the largest
    # step whose relative ESS stays at `ess_target`. `n_beta` becomes a cap on
    # the number of rungs rather than the schedule itself.
    logw  = zeros(Float64, n_particles)   # log UNNORMALISED weights since the last resample
    logZ  = 0.0                            # accumulated log normalising constant
    n_resample = 0
    betas_used = Float64[0.0]
    Ef = Float64[mean(filter(isfinite, fv))]

    # RWM scale in the reference's whitened frame, adapted per rung toward the
    # standard 0.234 acceptance. A fixed 2.38/sqrt(d) sat at ~0.15 here, which
    # under-moves the particles and is exactly what biases the path integral.
    c = step_scale * 2.38 / sqrt(d)
    n_acc = 0; n_try = 0
    β = 0.0
    k = 0

    while β < 1.0 && k < n_beta
        k += 1
        ess0 = _rp_ess(logw)
        target = ess_target * max(ess0, 1.0)
        # bisect on the increment: ESS is monotone decreasing in dbeta
        lo, hi = 0.0, 1.0 - β
        trial = hi
        for _ in 1:40
            cand = logw .+ trial .* fv
            if _rp_ess(cand) >= target
                lo = trial
            else
                hi = trial
            end
            trial = 0.5 * (lo + hi)
            hi - lo < 1e-12 && break
        end
        dβ = max(lo, 1e-10)            # never stall
        β  = min(1.0, β + dβ)

        prev = _logsumexp(logw)
        for i in 1:n_particles
            logw[i] += dβ * fv[i]
        end
        cur = _logsumexp(filter(isfinite, logw))
        logZ += cur - prev

        # Adaptive resampling. Plain AIS has no defence against weight
        # degeneracy: on that same 13-D target the weights collapsed onto ONE
        # particle (ESS 1/512). Resampling below a fraction of N is the standard
        # cure and makes this SMC; the log-Z accumulation above already permits
        # it.
        if _rp_ess(logw) < ess_threshold * n_particles
            mx = maximum(filter(isfinite, logw))
            wfull = [isfinite(logw[i]) ? exp(logw[i] - mx) : 0.0 for i in 1:n_particles]
            tot = sum(wfull)
            if tot > 0
                cw = cumsum(wfull ./ tot)
                u0 = rand(rng) / n_particles
                newP = similar(P); newf = similar(fv)
                j = 1
                for i in 1:n_particles
                    u = u0 + (i - 1) / n_particles
                    while j < n_particles && cw[j] < u; j += 1; end
                    newP[:, i] = @view P[:, j]; newf[i] = fv[j]
                end
                P .= newP; fv .= newf
                fill!(logw, 0.0)
                n_resample += 1
            end
        end

        # move under gamma_beta:  log gamma = log q + beta*f
        acc_k = 0; try_k = 0
        for i in 1:n_particles
            y  = @view P[:, i]
            lq = logq(y)
            lg = isfinite(fv[i]) ? lq + β * fv[i] : -Inf
            for _ in 1:n_steps
                prop = y .+ c .* (Lc * randn(rng, d))
                fp   = f(prop)
                lgp  = isfinite(fp) ? logq(prop) + β * fp : -Inf
                try_k += 1
                if isfinite(lgp) && log(rand(rng)) < lgp - lg
                    P[:, i] = prop; fv[i] = fp; lg = lgp; acc_k += 1
                    y = @view P[:, i]
                end
            end
        end
        n_acc += acc_k; n_try += try_k
        # step-size adaptation toward 0.234 (Roberts/Gelman/Gilks optimal RWM)
        if try_k > 0
            a = acc_k / try_k
            c *= exp(0.6 * (a - 0.234))
            c = clamp(c, 1e-6, 10.0)
        end
        push!(betas_used, β)
        push!(Ef, mean(filter(isfinite, fv)))
    end

    # --- the two estimates ---------------------------------------------------
    fin = filter(isfinite, logw)
    isempty(fin) && return bad
    # logZ already telescopes the per-rung weight-sum ratios (and survives
    # resampling), so it IS the estimate. With no resampling it reduces exactly
    # to the plain AIS form logsumexp(logw) - log(n_particles).
    log_z_ais = logZ
    # trapezoid over the beta grid
    log_z_ti = 0.0
    @inbounds for j in 1:(length(betas_used) - 1)
        log_z_ti += 0.5 * (Ef[j] + Ef[j+1]) * (betas_used[j+1] - betas_used[j])
    end

    # ESS of the AIS weights — the honest precision diagnostic.
    lw = fin .- maximum(fin)
    w  = exp.(lw)
    ess = sum(w)^2 / sum(abs2, w)
    se  = sqrt(max(0.0, 1 / ess))

    # Only report the path integral when it is mathematically defined.
    ti_out = support_frac >= 1.0 ? log_z_ti : NaN

    return (; log_z = log_z_ais, log_z_ti = ti_out, log_z_ais = log_z_ais,
              se = se, ess = ess, n_resample = n_resample,
              support_frac = support_frac,
              n_particles = n_particles, n_effective = length(fin),
              n_beta = length(betas_used) - 1,
              accept = n_try == 0 ? NaN : n_acc / n_try,
              proposal = proposal)
end
