# Prior-only regression: with a likelihood that carries no information (errors of
# 1e6 m/s) every k-planet evidence is 1 under the slab prior normalised over its
# allowed region, so the occupancy ratios k+1 : k must follow the documented prior
# P(k) ∝ (c p/(1-p))^k / k!  (docs/src/transdim.md), c = π/4 the share of the
# (√e sin ω, √e cos ω) box with e < 1. Equivalently ln B between adjacent k, which
# divides that prior out, is zero within its error.
using Test, Nereus, Random, Statistics

@testset "trans-dim prior-only occupancy follows the documented prior on k" begin
    rng = MersenneTwister(5)
    n = 20
    t = sort(2459000.0 .+ 1000.0 .* rand(rng, n))
    cfg = Dict{String, Any}("seed" => 7, "output_dir" => mktempdir(),
        "data" => Dict("rv" => Dict("values" => Dict("bjd" => t, "rv" => randn(rng, n),
            "rv_err" => fill(1e6, n), "instrument" => fill("HARPS", n)))),
        "model" => Dict("max_kplanet" => 2, "planet_modes" => ["RV_ONLY", "RV_ONLY"],
                        "stability" => "none"),
        "transdim" => Dict("max_kplanet" => 2),
        "sampler" => Dict("name" => "transdim_pt_emcee", "kwargs" => Dict(
            "show_progress" => false, "n_temps" => 2, "n_walkers" => 30,
            "n_steps" => 400, "n_burnin" => 200)),
        "output" => Dict("plots" => String[], "ppc" => false,
                         "detection_limits" => false, "loo" => false))
    s = Nereus.run_job(cfg)
    td = s["run_info"]["transdim"]
    c = td["prior"]["slab_fraction"]
    @test abs(c - π / 4) < 0.02
    @test td["prior"]["p_k"] ≈ Nereus.transdim_k_prior(2; slab_fraction = c)
    occ = [td["occupancy"][string(k)] for k in 0:2]
    pr = td["prior"]["p_k"]
    for r in td["ln_bayes_factor_adjacent"]
        k = r["k_lo"]
        # the occupancy ratio is the prior ratio (c p/(1-p)/(k+1)), within MC error
        @test abs(log(occ[k + 2] / occ[k + 1]) - log(pr[k + 2] / pr[k + 1])) < 4 * r["se"] + 0.15
        @test abs(r["ln_bayes_factor"]) < 4 * r["se"] + 0.15
    end
    # without the 1/k! or the slab fraction the k = 2 : 1 ratio would be off by ln 2 / ln(1/c)
    @test abs(log(occ[3] / occ[2]) - log(1)) > 0.3
end
