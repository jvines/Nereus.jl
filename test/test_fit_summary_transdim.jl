# The fit_* route reports the planet-count posterior under the key run_job uses,
# `n_planets_posterior`, which is what the Python client reads.

using Nereus, Random, Test

function _sine_rv(seed)
    rng = Random.MersenneTwister(seed)
    t = sort!(200 .* rand(rng, 40))
    rv = 15 .* sin.(2π .* t ./ 12.3) .+ randn(rng, 40)
    return Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.0, 40)))
end

_tiny(engine) = Dict("engine" => engine,
                     "options" => Dict("n_temps" => 2, "n_walkers" => 16,
                                       "n_steps" => 60, "n_burnin" => 20))

@testset "fit_* summary: n_planets_posterior" begin
    td = fit_rv(_sine_rv(5); planets = 1, transdim = true, science = false,
                engine = _tiny("transdim_pt_emcee"), output_dir = mktempdir())
    pp = td.summary["n_planets_posterior"]
    @test pp isa AbstractDict
    @test sum(values(pp)) ≈ 1
    @test all(k -> k in ("0", "1"), keys(pp))

    fixed = fit_rv(_sine_rv(5); planets = 1, science = false,
                   engine = _tiny("pt_emcee"), output_dir = mktempdir())
    @test !haskey(fixed.summary, "n_planets_posterior")
end
