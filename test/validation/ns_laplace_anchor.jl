#!/usr/bin/env julia
# Independent tie-breaker for the n=60 ndim=7 single-planet target:
#   TI (PT-HMC)  = -115.69 ± 0.26
#   NS (rslice)  ≈ -122 to -124 (plateaus; slice- AND enlarge-saturated)
# Direct LAPLACE at the mode (near-Gaussian: e≈0.56 away from the e=0 cusp,
# strong signal) is independent of both β-ladders and ellipsoidal bounds:
#   logZ ≈ g(y*) + (d/2)log(2π) - 0.5·logdet(-H),  g=logprior+jac+loglike, H=∇²g(y*)
# Whichever of TI/NS Laplace matches is the correct one.

using Nereus, Optim, ForwardDiff, LinearAlgebra, Printf, Random

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
target = NereusTarget(params, data; unconstrained = true)
d = length(params.layout.unfrozen_idx)
g(y) = sum(Nereus._logdensity_parts(target, y))     # log unnormalized posterior in y

# mode via Nereus's proven multi-start optimiser (x_map returned regardless of
# its own converged verdict), then local polish in unconstrained space.
mres = sample_map(target; n_starts = 64, seed = 7)
best_y = Nereus.transform_forward(mres.x_map, target.transform)
# a few LBFGS polish steps on -g (finite-diff grad) to sit exactly at the max
let
    orng = MersenneTwister(3)
    for _ in 1:8
        x0 = (best_y .+ 0.0)
        r = try
            Optim.optimize(y -> -g(y), x0, Optim.NelderMead(),
                           Optim.Options(g_tol = 1e-6, iterations = 4000))
        catch; nothing; end
        r === nothing && break
        ym = Optim.minimizer(r)
        g(ym) > g(best_y) && (global best_y = ym)
    end
end
best_g = g(best_y)

# Finite-difference Hessian of g at the mode (nested AD through the RV
# likelihood doesn't dispatch; central differences are robust here).
function fd_hessian(f, y; h = 1e-4)
    m = length(y); H = zeros(m, m)
    for i in 1:m, j in i:m
        yp = copy(y); yp[i] += h; yp[j] += h; a = f(yp)
        yp = copy(y); yp[i] += h; yp[j] -= h; b = f(yp)
        yp = copy(y); yp[i] -= h; yp[j] += h; c = f(yp)
        yp = copy(y); yp[i] -= h; yp[j] -= h; d_ = f(yp)
        H[i, j] = H[j, i] = (a - b - c + d_) / (4h^2)
    end
    return H
end
H = fd_hessian(g, best_y)
negH = -H
# symmetrize + check negative-definiteness of H (negH PD)
negH = (negH + negH') / 2
ev = eigvals(negH)
logdet_negH = sum(log, ev)
logZ_lap = best_g + (d / 2) * log(2π) - 0.5 * logdet_negH

xm = Nereus.transform_inverse(best_y, target.transform)
nm = params.layout.unfrozen_names
@printf("MAP: %s\n", join([@sprintf("%s=%.3f", nm[i], xm[i]) for i in eachindex(xm)], "  "))
@printf("g(y*) = %.3f   min eig(-H) = %.3e (PD=%s)\n", best_g, minimum(ev), all(>(0), ev))
@printf("\n[Laplace] logZ = %.3f      vs  TI -115.69 | NS ~-122.5\n", logZ_lap)
@printf("→ Laplace closer to %s\n", abs(logZ_lap - (-115.69)) < abs(logZ_lap - (-122.5)) ? "TI (NS bound-biased low)" : "NS (TI biased high)")
