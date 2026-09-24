# transdim_pt_emcee at its default settings finds an easy planet.
#
# Split out of test_pt_emcee_stranded.jl: it is one of the two full default-
# settings fits that made that file the slowest unit of the suite, and on its
# own it can run in a different CI shard (test/shards.jl).

using Test, Nereus, Random, Statistics, MCMCChains
include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))

@testset "transdim_pt_emcee: defaults find an easy planet" begin
    cfg = _easy_rv_cfg()

    # Trans-dim, defaults, same target, up to 2 planets. Before: 5 temps and
    # blind births found no planet (P(Nₚ = 0) = 0.84, no draw at 12.3 d).
    @testset "transdim_pt_emcee defaults find an easy planet" begin
        tcfg = merge(cfg, Dict{String,Any}(
            "output_dir" => mktempdir(),
            "model" => Dict("max_kplanet" => 2, "planet_modes" => ["RV_ONLY", "RV_ONLY"]),
            "transdim" => Dict("max_kplanet" => 2),
            "sampler" => Dict("name" => "transdim_pt_emcee",
                              "kwargs" => Dict("show_progress" => false))))
        s = Nereus.run_job(tcfg)
        ch, _ = Nereus.load_chains(joinpath(tcfg["output_dir"], "chains.nc"))
        np = vec(Array(ch[:n_planets]))
        @test mean(np .== 1) > 0.95
        has = falses(length(np))
        for k in 1:2
            on = vec(Array(ch[Symbol("planet_active_$k")])) .> 0.5
            Pk = vec(Array(ch[Symbol("P_k$k")]))
            has .|= on .& (abs.(Pk .- 12.3) ./ 12.3 .< 0.01)
        end
        @test mean(has) > 0.95
        # Both health signals: fit_health (masked to active draws) and the
        # sampler's own report (components, not slots).
        @test s["fit_health"]["checks"]["convergence"]["status"] == "ok"
        # sigma_HARPS sits at 0 (the errors are exact): a jitter can be zero,
        # and this fit printed "FIT HEALTH: FAIL" over it on every seed.
        @test s["fit_health"]["overall"] == "ok"
        @test s["run_info"]["convergence"]["pass"] == true
    end
end
