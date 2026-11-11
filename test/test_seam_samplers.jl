# Every sampler moves a full-circle angle's seam off the posterior
# (src/circular.jl). An eccentric RV planet under :ew with ω = π - 0.03 on the
# seam of its U(-π, π) window and Mo = 0.02 on the seam of U(0, 2π): a sampler
# that kept the default charts would split both posteriors between two walls
# (or keep one sliver). Each run must come back with every active draw inside
# its final window, none piled at a wall, one contiguous lobe per angle, and
# the truth inside it. Several chains share the windows, so the trans-dim and
# ensemble samplers run two, re-cut together at a rendezvous.
#
# Then the ensemble's stranded walkers: an RM fit at λ = 0 where, without the
# burn-in pruning, one walker in 32 ended on λ's seam and stayed there (3 % of
# the draws at the wall, λ's posterior sd 30° against 2°).

using Nereus, MCMCChains, Random, Test
using Statistics: median, quantile, std
using Nereus: Theta, set_param!

function _ss_target(; unconstrained)
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
        return NereusTarget(p, d; unconstrained)
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

function _ss_check(ch, p)
    pn = MCMCChains.names(ch, :parameters)
    act = :planet_active_1 in pn ? vec(Array(ch[:, :planet_active_1, :])) .> 0.5 :
          trues(size(ch, 1) * size(ch, 3))
    @test count(act) > 0.9 * length(act)
    for (nm, tru) in (("Mo_k1", 0.02), ("w_k1", π - 0.03))
        v = vec(Array(ch[:, Symbol(nm), :]))[act]
        lo, hi = Nereus.bounds(p.layout.unfrozen_priors[findfirst(==(nm), p.layout.unfrozen_names)])
        @test all(lo - 1e-9 .<= v .<= hi + 1e-9)
        @test count(x -> x - lo < deg2rad(0.5) || hi - x < deg2rad(0.5), v) == 0
        @test maximum(v) - minimum(v) < π
        tv = median(v) + rem2pi(tru - median(v), RoundNearest)
        @test quantile(v, 0.005) < tv < quantile(v, 0.995)
        @test deg2rad(3) < std(v) < deg2rad(8)        # pt_emcee: 4.4° and 5.0°
    end
end

@testset "every sampler moves a seam off the posterior" begin
    td = Nereus.TransDimConfig(max_kplanet = 1, planets = true)
    runs = [
        "rjmcmc, 2 chains" => (false, tg -> sample_rjmcmc(tg, tg.data; td, n_samples = 6000,
            n_warmup = 3000, seed = 3, n_chains = 2, show_progress = false)[1]),
        "moms, 2 chains" => (false, tg -> Nereus.sample_moms(tg, tg.data; td, n_samples = 6000,
            n_warmup = 3000, seed = 3, n_chains = 2, show_progress = false)[1]),
        "ensemble, 2 chains" => (true, tg -> sample_ensemble(tg; n_walkers = 32, n_steps = 4000,
            n_burnin = 2000, n_chains = 2, seed = 3, show_progress = false)),
        "daedalus" => (false, tg -> Nereus.sample_daedalus(tg, tg.data; td, n_live = 200,
            seed = 3, show_progress = false)[1]),
        "population annealing" => (false, tg -> Nereus.sample_pa(tg, tg.data; n_replicas = 400,
            seed = 3, show_progress = false).chains),
        "nested (INS)" => (false, tg -> Nereus.sample_nested_ins(tg, tg.data; n_live = 400,
            seed = 3, show_progress = false).chains),
        "pathfinder" => (true, tg -> begin
            pf = Nereus.pathfinder_init(tg; n_runs = 8, n_draws = 400, seed = 3)
            X = reduce(hcat, [Nereus.transform_inverse(pf.draws[:, j], tg.transform)
                              for j in axes(pf.draws, 2)])
            MCMCChains.Chains(permutedims(X), Symbol.(tg.params.layout.unfrozen_names))
        end),
    ]
    for (label, (unc, run)) in runs
        @testset "$label" begin
            tg = _ss_target(; unconstrained = unc)
            _ss_check(run(tg), tg.params)
        end
    end
end

@testset "ensemble: stranded walkers pruned in burn-in" begin
    P, Tc = 3.5, 2459000.0
    rng = MersenneTwister(11)
    sims = [Nereus.simulate_rm_night(; P = P, Tc = Tc + j * P, a_Rs = 8.0, λ = 0.0,
                                     vsini = 12.0, rr = 0.11, T14 = 3.0, sigma_rv = 6.0,
                                     b = 0.3, tag = "N$j", rng = rng) for j in 0:1]
    d, nms = Nereus.obliquity_data([s.night for s in sims])
    p = Nereus.obliquity_params(d, nms; P = P, Tc = Tc, b = (0.3, 0.03), a_Rs = (8.0, 0.2),
                                rr = (0.11, 0.003), vsini = (12_000.0, 2000.0), arome = true,
                                sigma0 = Dict(s.night.tag => s.night.σ0 for s in sims),
                                rv_noise = :white)
    tg = NereusTarget(p, d; unconstrained = true)
    ch = sample_ensemble(tg; n_walkers = 32, n_steps = 3000, n_burnin = 1500, seed = 1,
                         show_progress = false)
    v = vec(Array(ch[:, :lambda_k1, :]))
    lo, hi = Nereus.bounds(p.layout.unfrozen_priors[findfirst(==("lambda_k1"),
                                                             p.layout.unfrozen_names)])
    @test count(x -> x - lo < deg2rad(0.5) || hi - x < deg2rad(0.5), v) == 0
    @test maximum(abs.(rem2pi.(v, RoundNearest))) < deg2rad(20)
    @test std(v) < deg2rad(3)                          # pt_emcee: 2.2°
end
