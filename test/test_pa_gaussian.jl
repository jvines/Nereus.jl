#!/usr/bin/env julia
# Population Annealing smoke test on a 2D Gaussian.
#
# Mirrors test_ptemcee_gaussian.jl: re-implement the PA loop on a plain
# (log_prior, log_like) target, no NereusTarget. If this converges
# (recovered μ and Σ within tolerance, log Z within ~0.1 of analytical
# truth), the PA algorithm is correct. If sample_pa on a NereusTarget
# then fails, the bug is in the Nereus interface, not the algorithm.

using Random
using Statistics: mean, std, cov
using LinearAlgebra: det, inv
using Printf

function pa_simple(log_prior::Function, log_like::Function,
                   init_replicas::Matrix{Float64};   # (n_replicas, n_dim)
                   ess_target::Float64 = 0.5, max_steps::Int = 200,
                   n_mcmc::Int = 10, step_sigma::Vector{Float64},
                   seed::Int = 1)
    n_replicas, n_dim = size(init_replicas)
    rng = MersenneTwister(seed)

    state = copy(init_replicas)
    logπ  = [log_prior(view(state, i, :)) for i in 1:n_replicas]
    logL  = [log_like(view(state, i, :))  for i in 1:n_replicas]

    β_cur = 0.0
    log_z = 0.0
    accepts = 0; proposals = 0
    n_evals = 0

    while β_cur < 1.0
        # Adaptive Δβ for ESS = ess_target
        function ess_at(Δβ)
            log_w = Δβ .* logL
            m = maximum(log_w)
            sw  = sum(exp.(log_w .- m))
            sw2 = sum(exp.(2 .* (log_w .- m)))
            return sw^2 / sw2 / n_replicas
        end
        lo, hi = 0.0, 1.0 - β_cur
        if ess_at(hi) >= ess_target
            Δβ = hi
        else
            for _ in 1:40
                mid = (lo + hi) / 2
                ess_at(mid) >= ess_target ? (lo = mid) : (hi = mid)
            end
            Δβ = lo
        end
        Δβ = max(Δβ, 1e-6)
        β_new = min(β_cur + Δβ, 1.0)

        # Weights + log dZ
        log_w = Δβ .* logL
        m = maximum(log_w)
        sw = sum(exp.(log_w .- m))
        log_z += m + log(sw / n_replicas)

        W = exp.(log_w .- m) ./ sw

        # Systematic resampling
        cdf = cumsum(W)
        u0 = rand(rng) / n_replicas
        indices = Vector{Int}(undef, n_replicas)
        j = 1
        for k in 1:n_replicas
            u = u0 + (k - 1) / n_replicas
            while j < n_replicas && cdf[j] < u
                j += 1
            end
            indices[k] = j
        end

        new_state = state[indices, :]
        new_logL = logL[indices]; new_logπ = logπ[indices]
        state = new_state; logL = new_logL; logπ = new_logπ

        # MCMC mutation at β_new
        for i in 1:n_replicas
            x = state[i, :]
            for _ in 1:n_mcmc
                y = x .+ step_sigma .* randn(rng, n_dim)
                lp_y = log_prior(y)
                proposals += 1
                n_evals += 1
                isfinite(lp_y) || continue
                ll_y = log_like(y)
                log_α = β_new * (ll_y - logL[i]) + (lp_y - logπ[i])
                if log(rand(rng)) < log_α
                    x = y
                    logL[i] = ll_y; logπ[i] = lp_y
                    accepts += 1
                end
            end
            state[i, :] = x
        end

        β_cur = β_new
    end

    return (samples = state, log_z = log_z,
            acc = accepts / max(proposals, 1), n_evals = n_evals)
end

# 2D Gaussian target
const μ_true = [3.0, -1.5]
const Σ_true = [1.0 0.5; 0.5 2.0]
const Σ_inv  = inv(Σ_true)
const log_det_Σ = log(det(Σ_true))

const lo, hi = [-10.0, -10.0], [10.0, 10.0]

log_prior_box(x) = all(lo .≤ x .≤ hi) ? -log((hi[1]-lo[1]) * (hi[2]-lo[2])) : -Inf
function log_like_gauss(x)
    d = x .- μ_true
    return -0.5 * (d' * Σ_inv * d) - 0.5 * length(x) * log(2π) - 0.5 * log_det_Σ
end

# log Z_truth = ∫ pi(x) L(x) dx = ∫ (1/Vol_box) L(x) dx
# For wide-enough box, L integrates to ~1, so log Z ≈ -log(Vol_box) = -log(400)
log_z_truth = -log(20.0 * 20.0)

n_replicas = 500
rng_init = MersenneTwister(42)
init_replicas = Matrix{Float64}(undef, n_replicas, 2)
for i in 1:n_replicas
    init_replicas[i, 1] = lo[1] + (hi[1] - lo[1]) * rand(rng_init)
    init_replicas[i, 2] = lo[2] + (hi[2] - lo[2]) * rand(rng_init)
end

@printf("=== Population Annealing 2D Gaussian sanity test ===\n")
@printf("Truth: μ = %s, Σ = %s\n", μ_true, Σ_true)
@printf("log Z (analytic) = %.4f\n", log_z_truth)
@printf("n_replicas = %d, ess_target = 0.5, n_mcmc = 10\n\n", n_replicas)

t0 = time()
res = pa_simple(log_prior_box, log_like_gauss, init_replicas;
                ess_target = 0.5, n_mcmc = 10,
                step_sigma = [1.0, 1.5], seed = 1)
@printf("Done in %.2fs, n_evals = %d, accept = %.3f\n", time() - t0, res.n_evals, res.acc)
@printf("log Z (PA)      = %.4f\n", res.log_z)
@printf("Δlog Z          = %+.4f\n", res.log_z - log_z_truth)

μ_est = vec(mean(res.samples, dims = 1))
Σ_est = cov(res.samples)
@printf("\nRecovered μ = [%.3f, %.3f]   (truth: [%.3f, %.3f])\n",
        μ_est[1], μ_est[2], μ_true[1], μ_true[2])
@printf("Recovered Σ:\n")
@printf("  [%.3f  %.3f]   (truth: [%.3f  %.3f])\n",
        Σ_est[1,1], Σ_est[1,2], Σ_true[1,1], Σ_true[1,2])
@printf("  [%.3f  %.3f]   (truth: [%.3f  %.3f])\n",
        Σ_est[2,1], Σ_est[2,2], Σ_true[2,1], Σ_true[2,2])

μ_err = maximum(abs.(μ_est .- μ_true))
Σ_err = maximum(abs.(Σ_est .- Σ_true))
Z_err = abs(res.log_z - log_z_truth)
@printf("\nmax |μ_err| = %.4f   (target < 0.2)\n", μ_err)
@printf("max |Σ_err| = %.4f   (target < 0.4)\n", Σ_err)
@printf("|Δlog Z|    = %.4f   (target < 0.2)\n", Z_err)
@printf("Verdict: %s\n",
        (μ_err < 0.2 && Σ_err < 0.4 && Z_err < 0.2) ?
        "PASS — PA algorithm is correct" : "FAIL — algorithm has a bug")
