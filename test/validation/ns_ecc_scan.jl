#!/usr/bin/env julia
# Gate #2 / paper spine: does the NS evidence bias SCALE with eccentricity?
# For e in {0.15,0.3,0.45,0.6}: ground truth = direct Laplace at the mode
# (matches TI to ~0.1 nat for e>0; degenerate only at e=0). NS = multi+rslice,
# 2 seeds. Report NS−Laplace vs e. Monotone growth ⇒ the "bias vs e" figure.

using Nereus, MCMCChains, Statistics, Printf, Random, Optim, LinearAlgebra

function kepler_rv(t, P, K, e, ω; t0 = 0.0)
    M = 2π .* (t .- t0) ./ P; E = copy(M)
    for _ in 1:60; E .-= (E .- e .* sin.(E) .- M) ./ (1 .- e .* cos.(E)); end
    ν = 2 .* atan.(sqrt(1 + e) .* sin.(E ./ 2), sqrt(1 - e) .* cos.(E ./ 2))
    return K .* (cos.(ν .+ ω) .+ e * cos(ω))
end
function build(e)
    rng = MersenneTwister(11); n = 60
    t  = sort(rand(rng, n) .* 80.0)
    rv = kepler_rv(t, 7.3, 9.0, e, 0.8) .+ 1.2 .* randn(rng, n)
    data = Data(t_rv = t, rv = rv, rv_err = fill(1.2, n), rv_inst = ones(Int, n))
    pr = Dict{String,PriorSpec}(
        "P_k1" => LogUniformPrior(2.0, 30.0), "K_k1" => UniformPrior(0.0, 30.0),
        "sesinw_k1" => UniformPrior(-0.9, 0.9), "secosw_k1" => UniformPrior(-0.9, 0.9),
        "Mo_k1" => UniformPrior(0.0, 2π),
        "gamma_I1" => UniformPrior(-20.0, 20.0), "sigma_I1" => LogUniformPrior(0.1, 10.0))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
        instruments = InstrumentConfig(rv = ["I1"]), data = data, M_s = 1.0,
        parametrization = ParametrizationConfig(time = :Mo), priors = pr)
    return params, data
end

function laplace_logz(params, data)
    target = NereusTarget(params, data; unconstrained = true)
    d = length(params.layout.unfrozen_idx)
    g(y) = sum(Nereus._logdensity_parts(target, y))
    mres = sample_map(target; n_starts = 48, seed = 7)
    ym = Nereus.transform_forward(mres.x_map, target.transform)
    let orng = MersenneTwister(3)
        for _ in 1:6
            r = try Optim.optimize(y -> -g(y), ym .+ 0.0, Optim.NelderMead(),
                                   Optim.Options(iterations = 3000)); catch; nothing; end
            r === nothing && break
            Optim.minimizer(r) |> z -> (g(z) > g(ym) && (ym = z))
        end
    end
    h = 1e-4; H = zeros(d, d)
    for i in 1:d, j in i:d
        a=copy(ym); a[i]+=h; a[j]+=h; b=copy(ym); b[i]+=h; b[j]-=h
        c=copy(ym); c[i]-=h; c[j]+=h; e_=copy(ym); e_[i]-=h; e_[j]-=h
        H[i,j]=H[j,i]=(g(a)-g(b)-g(c)+g(e_))/(4h^2)
    end
    ev = eigvals(Symmetric(-H))
    return g(ym) + (d/2)*log(2π) - 0.5*sum(log, ev), all(>(0), ev)
end

@printf("=== NS evidence bias vs eccentricity (n=60) ===\n")
@printf("%-6s %10s %10s %10s %8s\n", "e", "Laplace", "NS(s1)", "NS(s2)", "bias")
for e in (0.15, 0.3, 0.45, 0.6)
    params, data = build(e)
    lap, pd = laplace_logz(params, data)
    mk() = NereusTarget(params, data; unconstrained = true)
    lz = [sample_nested(mk(), data; n_live=1500, dlogz=0.05, bounds=:multi, proposal=:rslice, seed=s)[2] for s in 1:2]
    @printf("%-6.2f %10.2f %10.2f %10.2f %8.2f%s\n", e, lap, lz[1], lz[2], mean(lz)-lap, pd ? "" : " (Lap PD=false)")
end
@printf("\nbias growing with e ⇒ the eccentricity-ridge mechanism (paper spine figure)\n")
