#!/usr/bin/env julia
# Sanity test for ptemcee: a self-contained re-implementation of the
# Vousden+ 2016 algorithm on a plain 2D Gaussian target, no NereusTarget
# in the way. If this converges (acceptance ~30-50%, recovered mean and
# cov within tolerance), the algorithm is correct and any failure of
# `sample_ptemcee` on real Nereus targets must be in the Nereus
# interface (Theta buffers, log_prior / log_likelihood split, etc.).
# If this fails, the algorithm itself is wrong.

using Random
using Statistics: mean, std, cov
using LinearAlgebra: det, inv
using Printf

# ----- Standalone ptemcee on a generic target ----------------------
# log_prior(x): -Inf outside support. log_like(x): finite likelihood.
function ptemcee_simple(log_prior::Function, log_like::Function,
                        init_walkers::Array{Float64,3};  # (n_temps, n_walkers, n_dim)
                        betas::Vector{Float64},
                        n_steps::Int, n_burnin::Int, stretch_a::Float64 = 2.0,
                        seed::Int = 1)
    n_temps, n_walkers, n_dim = size(init_walkers)
    @assert n_walkers % 2 == 0 "n_walkers must be even"
    half = n_walkers ÷ 2

    state    = copy(init_walkers)
    logπ_arr = fill(-Inf, n_temps, n_walkers)
    logL_arr = fill(-Inf, n_temps, n_walkers)
    for t in 1:n_temps, w in 1:n_walkers
        x = view(state, t, w, :)
        lp = log_prior(x)
        if isfinite(lp)
            ll = log_like(x)
            if isfinite(ll)
                logπ_arr[t, w] = lp
                logL_arr[t, w] = ll
            end
        end
    end

    rng = MersenneTwister(seed)

    accept_within  = zeros(Float64, n_temps)
    propose_within = zeros(Int, n_temps)
    accept_swap    = zeros(Float64, n_temps - 1)
    propose_swap   = zeros(Int, n_temps - 1)

    n_keep = n_steps - n_burnin
    samples = Array{Float64,3}(undef, n_keep, n_walkers, n_dim)

    for step in 1:n_steps
        # Two-half ensemble stretch
        for (active, inactive) in ((1:half, (half+1):n_walkers),
                                    ((half+1):n_walkers, 1:half))
            for t in 1:n_temps
                β = betas[t]
                for w in active
                    w_partner = rand(rng, inactive)
                    u = rand(rng)
                    z = ((stretch_a - 1) * u + 1)^2 / stretch_a
                    proposal = Vector{Float64}(undef, n_dim)
                    @inbounds for d in 1:n_dim
                        proposal[d] = state[t, w_partner, d] +
                                      z * (state[t, w, d] - state[t, w_partner, d])
                    end
                    lp_prop = log_prior(proposal)
                    propose_within[t] += 1
                    if isfinite(lp_prop)
                        ll_prop = log_like(proposal)
                        log_ratio = (n_dim - 1) * log(z) +
                                    β * (ll_prop - logL_arr[t, w]) +
                                    (lp_prop - logπ_arr[t, w])
                        if log(rand(rng)) < log_ratio
                            @inbounds for d in 1:n_dim
                                state[t, w, d] = proposal[d]
                            end
                            logπ_arr[t, w] = lp_prop
                            logL_arr[t, w] = ll_prop
                            accept_within[t] += 1
                        end
                    end
                end
            end
        end

        # Adjacent-temperature swaps
        for t in 1:(n_temps - 1)
            Δβ = betas[t] - betas[t + 1]
            for w in 1:n_walkers
                w2 = rand(rng, 1:n_walkers)
                ll1 = logL_arr[t, w]
                ll2 = logL_arr[t + 1, w2]
                log_ratio = Δβ * (ll2 - ll1)
                propose_swap[t] += 1
                if log(rand(rng)) < log_ratio
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
                end
            end
        end

        # Record β=1 walkers post-burnin
        if step > n_burnin
            keep_step = step - n_burnin
            for w in 1:n_walkers, d in 1:n_dim
                samples[keep_step, w, d] = state[1, w, d]
            end
        end
    end

    return (samples = samples,
            acc_within = accept_within ./ max.(propose_within, 1),
            acc_swap   = accept_swap   ./ max.(propose_swap,   1))
end

# ----- 2D Gaussian target ------------------------------------------
# Truth: μ = [3.0, -1.5], Σ = [[1.0, 0.5]; [0.5, 2.0]]
const μ_true = [3.0, -1.5]
const Σ_true = [1.0 0.5; 0.5 2.0]
const Σ_inv  = inv(Σ_true)
const log_det_Σ = log(det(Σ_true))

# Uniform prior box (wide vs posterior std for realism)
const lo = [-10.0, -10.0]
const hi = [10.0, 10.0]

log_prior_box(x) = all(lo .≤ x .≤ hi) ? 0.0 : -Inf
function log_like_gauss(x)
    d = x .- μ_true
    return -0.5 * (d' * Σ_inv * d) - 0.5 * length(x) * log(2π) - 0.5 * log_det_Σ
end

# ----- Init walkers from prior box ---------------------------------
const n_temps_test   = 5
const n_walkers_test = 100
const n_dim          = 2
betas_test  = Float64[(1 / sqrt(5))^i for i in 0:(n_temps_test - 1)]

rng_init = MersenneTwister(42)
init_walkers = Array{Float64,3}(undef, n_temps_test, n_walkers_test, n_dim)
for t in 1:n_temps_test, w in 1:n_walkers_test
    init_walkers[t, w, 1] = lo[1] + (hi[1] - lo[1]) * rand(rng_init)
    init_walkers[t, w, 2] = lo[2] + (hi[2] - lo[2]) * rand(rng_init)
end

@printf("=== ptemcee 2D Gaussian sanity test ===\n")
@printf("True μ = %s\n", μ_true)
@printf("True Σ = %s\n", Σ_true)
@printf("Prior box: [%s, %s] × [%s, %s]\n", lo[1], hi[1], lo[2], hi[2])
@printf("n_temps=%d, n_walkers=%d, n_steps=2000, burnin=500\n",
        n_temps_test, n_walkers_test)

t0 = time()
res = ptemcee_simple(log_prior_box, log_like_gauss, init_walkers;
                     betas    = betas_test,
                     n_steps  = 2000,
                     n_burnin = 500,
                     stretch_a = 2.0,
                     seed     = 1)
@printf("Done in %.1fs\n", time() - t0)
@printf("Acceptance per temp: %s\n",
        join(map(a -> @sprintf("%.3f", a), res.acc_within), ", "))
@printf("Swap acceptance:     %s\n",
        join(map(a -> @sprintf("%.3f", a), res.acc_swap), ", "))

# Flatten β=1 samples
flat = reshape(res.samples, :, n_dim)
μ_est = vec(mean(flat, dims = 1))
Σ_est = cov(flat)

@printf("\nRecovered μ = [%.3f, %.3f]   (truth: [%.3f, %.3f])\n",
        μ_est[1], μ_est[2], μ_true[1], μ_true[2])
@printf("Recovered Σ:\n")
@printf("  [%.3f  %.3f]   (truth: [%.3f  %.3f])\n",
        Σ_est[1,1], Σ_est[1,2], Σ_true[1,1], Σ_true[1,2])
@printf("  [%.3f  %.3f]   (truth: [%.3f  %.3f])\n",
        Σ_est[2,1], Σ_est[2,2], Σ_true[2,1], Σ_true[2,2])

μ_err = maximum(abs.(μ_est .- μ_true))
Σ_err = maximum(abs.(Σ_est .- Σ_true))
@printf("\nmax |μ_err| = %.4f  (target < 0.1)\n", μ_err)
@printf("max |Σ_err| = %.4f  (target < 0.3)\n", Σ_err)
@printf("Verdict: %s\n",
        (μ_err < 0.1 && Σ_err < 0.3 && minimum(res.acc_within[1:1]) > 0.1) ?
        "PASS — ptemcee algorithm is correct" :
        "FAIL — algorithm has a bug")
