# The differential-evolution move of the ensemble samplers (src/samplers/de_move.jl).
#
# 1. Its draws: pairs uniform over distinct walkers, nothing drawn at
#    de_move = 0, a probability checked.
# 2. Invariance on a known target, through the kernel the samplers call: a
#    correlated Gaussian pair and a von Mises angle centred 0.05 rad inside one
#    wall of its chart [lo, lo + 2π), so its mass sits at both ends of the chart
#    -- the samplers' representation of a full-circle angle, with density zero
#    past the walls. An ensemble moved by the DE move alone (γ jittered, γ = 1
#    hops, ε) must return the Gaussian's moments and the angle's circular mean,
#    concentration and distribution.
# 3. The same through sample_pt_emcee, every within-rung move a DE move, on an
#    eccentric RV planet whose ω and Mo sit on the seams of their windows: the
#    draws come back inside the moved windows, none at a wall, one lobe, the
#    truth inside.

using Nereus, MCMCChains, Random, Test
using Statistics: mean, median, quantile, std, cov
using SpecialFunctions: besseli
using Nereus: Theta, set_param!

@testset "DE move: its draws" begin
    rng = MersenneTwister(1)
    cnt = zeros(Int, 4, 4)
    for _ in 1:80_000
        j, k = Nereus._de_pair(rng, 5, 8)
        @test j != k
        cnt[j - 4, k - 4] += 1
    end
    off = [cnt[a, b] for a in 1:4, b in 1:4 if a != b]
    @test all(c -> abs(c - 80_000 / 12) < 5 * sqrt(80_000 / 12), off)
    # Nothing drawn when the move is off: the stretch-only stream is untouched.
    r1 = MersenneTwister(7); r2 = MersenneTwister(7)
    @test !Nereus._de_chosen(r1, 0.0)
    @test rand(r1) == rand(r2)
    @test_throws ArgumentError Nereus._de_check(1.5)
    @test_throws ArgumentError Nereus._de_check(-0.1)
    @test Nereus._de_check(1) === 1.0
end

# A von Mises angle near a wall of its chart and a correlated Gaussian pair.
const _DE_LO = -π
const _DE_MU = -π + 0.05
const _DE_KAPPA = 4.0
const _DE_C = [1.0 0.8; 0.8 1.0]
const _DE_CI = inv(_DE_C)
function _de_logp(x)
    θ = x[3]
    _DE_LO <= θ < _DE_LO + 2π || return -Inf
    g = x[1:2]
    return -0.5 * (g' * _DE_CI * g) + _DE_KAPPA * cos(θ - _DE_MU)
end

# Walkers S[1, w, :], the two halves moved in turn against each other by the
# DE move alone, as sample_pt_emcee does at one rung.
function _de_mini_run(; nw = 64, n_steps = 6000, n_burn = 1000, seed = 3)
    rng = MersenneTwister(seed)
    S = zeros(1, nw, 3)
    for w in 1:nw
        S[1, w, 1:2] = randn(rng, 2)
        S[1, w, 3] = _DE_LO + 2π * rand(rng)
    end
    lp = [_de_logp(S[1, w, :]) for w in 1:nw]
    buf = zeros(3); scale = zeros(1, 3); held = falses(3)
    h = nw ÷ 2
    out = Matrix{Float64}(undef, 3, 0)
    γ0 = Nereus._de_gamma0(3)
    acc = 0; prop = 0
    for step in 1:n_steps
        for (act, plo, phi) in ((1:h, h + 1, nw), ((h + 1):nw, 1, h))
            Nereus._de_spread!(scale, S, plo, phi)
            for w in act
                j, k = Nereus._de_pair(rng, plo, phi)
                γ = Nereus._de_gamma(rng, γ0)
                Nereus._de_fill!(buf, S, 1, w, j, k, γ, scale, held, rng)
                l = _de_logp(buf)
                prop += 1
                if log(rand(rng)) < l - lp[w]
                    S[1, w, :] .= buf; lp[w] = l; acc += 1
                end
            end
        end
        step > n_burn && step % 5 == 0 && (out = hcat(out, S[1, :, :]'))
    end
    return out, acc / prop
end

@testset "DE move: leaves a known target invariant, an angle at its seam" begin
    X, acc = _de_mini_run()
    @test 0.1 < acc < 0.6
    n = size(X, 2)                       # 64 walkers × 1000 recorded steps
    # Moments of the Gaussian pair. Recorded every 5th step, the draws keep some
    # autocorrelation: the bounds are ~5 standard errors at an ESS of n/10.
    se = 1 / sqrt(n / 10)
    @test abs(mean(X[1, :])) < 5se
    @test abs(mean(X[2, :])) < 5se
    C = cov(X[1:2, :]')
    @test all(abs.(C .- _DE_C) .< 10se)
    # The angle: its mass split between the two ends of the chart ...
    θ = X[3, :]
    @test all(_DE_LO .<= θ .< _DE_LO + 2π)
    @test 0.2 < mean(θ .> 0) < 0.8
    # ... and as a circular distribution, the von Mises it is.
    R = sqrt(mean(cos.(θ))^2 + mean(sin.(θ))^2)
    @test abs(R - besseli(1, _DE_KAPPA) / besseli(0, _DE_KAPPA)) < 5se
    @test abs(rem2pi(atan(mean(sin.(θ)), mean(cos.(θ))) - _DE_MU, RoundNearest)) < 5se
    # The distribution on the circle against the von Mises CDF (numerical), a
    # Kolmogorov distance within ~5 times its ESS-scaled scale.
    φ = sort(rem2pi.(θ .- _DE_MU, RoundNearest))       # centred, on (-π, π]
    gridc = range(-π, π; length = 4001)
    densc = exp.(_DE_KAPPA .* cos.(gridc))
    cdfc = cumsum(densc) ./ sum(densc)
    D = maximum(abs(i / n - cdfc[clamp(searchsortedfirst(gridc, φ[i]), 1, 4001)])
                for i in 1:n)
    @test D < 5 / sqrt(n / 10)
end

# An eccentric RV planet with ω = π - 0.03 on the seam of U(-π, π) and Mo = 0.02
# on the seam of U(0, 2π) (as test_seam_samplers.jl).
function _de_seam_target()
    un(a, b) = Dict{String, Any}("type" => "UniformPrior", "args" => [a, b])
    function build(t, rv)
        n = length(t)
        cfg = Dict{String, Any}(
            "star" => Dict("M_s" => 1.0, "R_s" => 1.0),
            "priors" => Dict{String, Any}(
                "P_k1" => un(12.0, 12.6), "Mo_k1" => un(0.0, 2π),
                "K_k1" => un(1.0, 60.0), "ecc_k1" => un(0.0, 0.9),
                "w_k1" => un(-π, π), "gamma_SIM" => un(-50.0, 50.0),
                "sigma_SIM" => un(0.01, 20.0)),
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"],
                                         "parametrization" => Dict("ew" => "ew")),
            "noise_models" => Any[],
            "data" => Dict{String, Any}("rv" => Dict("values" => Dict(
                "bjd" => t, "rv" => rv, "rv_err" => fill(2.0, n),
                "instrument" => fill("SIM", n)))))
        d, irv, ipm = Nereus._build_data(cfg["data"])
        p, _, _ = Nereus._build_model(cfg, d, Nereus._build_star(cfg["star"]), irv, ipm)
        return NereusTarget(p, d; unconstrained = false)
    end
    rng = MersenneTwister(42)
    t = sort(55000 .+ 200 .* rand(rng, 60))
    tg0 = build(t, zeros(60))
    th = Theta{Float64}(tg0.params)
    for (nm, v) in ("P_k1" => 12.3, "K_k1" => 25.0, "ecc_k1" => 0.3, "w_k1" => π - 0.03,
                    "Mo_k1" => 0.02, "gamma_SIM" => 0.0, "sigma_SIM" => 0.01)
        set_param!(th, nm, v)
    end
    model, _ = Nereus.rv_predictions(th, tg0.data)
    return build(t, model .+ sqrt(13.0) .* randn(rng, 60))
end

@testset "DE move: sample_pt_emcee on seams, every within-rung move DE" begin
    tg = _de_seam_target()
    r = sample_pt_emcee(tg, tg.data; n_temps = 4, n_walkers = 32, n_steps = 3000,
                        n_burnin = 1500, seed = 2, de_move = 1.0, show_progress = false)
    @test all(iszero, r.acceptance_within)         # no stretch proposed
    @test 0.05 < r.acceptance_de[1] < 0.9
    ch = r.chains; p = tg.params
    for (nm, tru) in (("Mo_k1", 0.02), ("w_k1", π - 0.03))
        v = vec(Array(ch[:, Symbol(nm), :]))
        lo, hi = Nereus.bounds(p.layout.unfrozen_priors[findfirst(==(nm), p.layout.unfrozen_names)])
        @test all(lo - 1e-9 .<= v .<= hi + 1e-9)
        @test count(x -> x - lo < deg2rad(0.5) || hi - x < deg2rad(0.5), v) == 0
        @test maximum(v) - minimum(v) < π
        tv = median(v) + rem2pi(tru - median(v), RoundNearest)
        @test quantile(v, 0.005) < tv < quantile(v, 0.995)
        @test deg2rad(3) < std(v) < deg2rad(8)        # stretch-only: 4.4° and 5.0°
    end
end

# The drift check (src/diagnostics/drift.jl): walkers that each mix (AR(1),
# ρ = 0.5) pass; the same walkers sharing a slow common trend -- every walker
# moving together, which R-hat cannot see -- are flagged, by name.
@testset "ensemble_drift: a drift every walker shares" begin
    rng = MersenneTwister(11)
    T, W = 1000, 32
    cube = zeros(T, 2, W)
    for w in 1:W, j in 1:2
        x = randn(rng)
        for s in 1:T
            x = 0.5 * x + sqrt(1 - 0.25) * randn(rng)
            cube[s, j, w] = x + (j == 2 ? 0.6 * (s - T / 2) / T : 0.0)
        end
    end
    d = ensemble_drift(cube, ["still", "moving"])
    @test d isa Dict
    @test d["flagged"]
    @test [r["parameter"] for r in d["drifted"]] == ["moving"]
    @test abs(d["z"]["still"]) < 4
    @test d["z"]["moving"] > Nereus.DRIFT_Z
    r = only(d["drifted"])
    @test r["late_mean"] > r["early_mean"] && 0.2 < r["shift_sd"] < 0.5
    @test d["max_parameter"] == "moving" && d["n_steps"] == T && d["n_walkers"] == W
    # Too short to say anything; and independent groups tested on their own.
    @test ensemble_drift(cube[1:50, :, :], ["still", "moving"]) === nothing
    g = ensemble_drift(cube, ["still", "moving"]; groups = [1:16, 17:32])
    @test g["flagged"] && only(g["drifted"])["parameter"] == "moving"
    # A run with the check's warning names what drifted and what to do.
    @test_logs (:warn, r"moving.*run longer") match_mode = :any Nereus._drift_warn(d, "test")
end
