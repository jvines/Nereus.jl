#!/usr/bin/env julia
# Diagnose the MLFriends sampling slowness: build a representative CONSTRAINED
# live set (top-L of uniform draws on the n=60 ecc RV target, in unit-cube
# u-space), fit MLFriends, then instrument rand(): retries to produce one point
# and the per-gate rejection breakdown (overlap / cube / ellipsoid). Pinpoints
# whether the whitening/radius blows up (degenerate ridge) or a gate over-rejects.

using Nereus, Random, Printf, LinearAlgebra, Statistics
using Distributions: quantile
using NestedSamplers: Bounds

function kepler_rv(t, P, K, e, ω; t0 = 0.0)
    M = 2π .* (t .- t0) ./ P; E = copy(M)
    for _ in 1:60; E .-= (E .- e .* sin.(E) .- M) ./ (1 .- e .* cos.(E)); end
    ν = 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
    return K .* (cos.(ν .+ ω) .+ e * cos(ω))
end
rng = MersenneTwister(11); n = 60
t  = sort(rand(rng, n) .* 80.0)
rv = kepler_rv(t, 7.3, 9.0, 0.3, 0.8) .+ 1.2 .* randn(rng, n)
data = Data(t_rv = t, rv = rv, rv_err = fill(1.2, n), rv_inst = ones(Int, n))
pr = Dict{String,PriorSpec}(
    "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
    "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
    "Mo_k1" => UniformPrior(0.0, 2π),
    "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))
params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
    instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
    parametrization = ParametrizationConfig(time = :Mo), priors = pr)
layout = params.layout
dists = [layout.unfrozen_priors[j].dist for j in 1:length(layout.unfrozen_idx)]
d = length(dists)

# L(u): map unit cube → physical via prior quantiles, evaluate likelihood
function loglike_u(u)
    th = Nereus.Theta{Float64}(params)
    x = [quantile(dists[j], u[j]) for j in 1:d]
    Nereus.set_unfrozen!(th, x)
    Nereus.rv_log_likelihood(th, data)
end

# representative constrained live set: top-NLIVE of M uniform draws
M = 60_000; NLIVE = 250
rng2 = MersenneTwister(7)
U = rand(rng2, d, M)
L = [loglike_u(view(U, :, i)) for i in 1:M]
keep = partialsortperm(L, 1:NLIVE; rev = true)
us = U[:, keep]
@printf("live set: %d-D, %d points, logL range [%.1f, %.1f]\n", d, NLIVE, minimum(L[keep]), maximum(L[keep]))

b = Bounds.fit(Bounds.MLFriends, us)
@printf("maxradiussq=%.4g  enlarge=%.4g  est.volume=%.3g\n", b.maxradiussq, b.enlarge, Bounds.volume(b))

# whitened-space spread (detect degenerate blow-up) + cube extent of a ball
evspread = let
    dd = us .- vec(mean(us, dims=2)); C = (dd*dd')/(size(us,2)-1)*(d+2)
    ev = eigen(Symmetric(C)).values
    @printf("cov eigenvalues: min=%.3g max=%.3g cond=%.3g\n", minimum(ev), maximum(ev), maximum(ev)/max(minimum(ev),1e-300))
end

# instrument rand(): replicate the loop with counters (in a function to avoid soft-scope)
function diagnose(b, d, NT, seed)
    r = sqrt(b.maxradiussq); N = size(b.unormed, 2)
    retries = 0; rej_overlap = 0; rej_cube = 0; rej_ell = 0
    rng3 = MersenneTwister(seed)
    for _ in 1:NT
        while true
            retries += 1
            i = rand(rng3, 1:N)
            v = randn(rng3, d); v .*= (rand(rng3)^(1/d)*r)/sqrt(sum(abs2,v)); v .+= b.unormed[:,i]
            cnt = 0
            @inbounds for j in 1:N
                s=0.0; for k in 1:d; Δ=v[k]-b.unormed[k,j]; s+=Δ*Δ; end
                s ≤ b.maxradiussq && (cnt+=1)
            end
            if cnt==0 || rand(rng3)*cnt ≥ 1; rej_overlap+=1; continue; end
            u = b.Winv*v .+ b.ctr
            if !all(0 .< u .< 1); rej_cube+=1; continue; end
            de = u .- b.ell_ctr
            if dot(de, b.ell_invcov*de) > b.enlarge; rej_ell+=1; continue; end
            break
        end
    end
    return retries, rej_overlap, rej_cube, rej_ell
end
NT = 2000
retries, rej_overlap, rej_cube, rej_ell = diagnose(b, d, NT, 3)
@printf("\nrand() over %d points: %.1f avg attempts/point\n", NT, retries/NT)
@printf("  rejections: overlap=%d (%.0f%%)  cube=%d (%.0f%%)  ellipsoid=%d (%.0f%%)\n",
        rej_overlap, 100rej_overlap/retries, rej_cube, 100rej_cube/retries, rej_ell, 100rej_ell/retries)
@printf("  acceptance = %.4f  (1 / %.0f attempts)\n", NT/retries, retries/NT)
