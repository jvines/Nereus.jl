# fit_joint(...; use_rho_s = true): the stellar density as a sampled parameter in
# an RV + transit fit, as a run_job config's `parametrization.use_rho_s` has it.
# fit_joint had no such keyword, so the only way to fit a transit with a density
# prior jointly with RVs was run_job.
using Nereus, Distributions, ForwardDiff, LogDensityProblems, MCMCChains, Random, Test
using Statistics: median, mean

@testset "fit_joint: use_rho_s" begin
    P = 3.0
    tr = collect(range(0.0, 6.0; length = 240))
    ph = mod.(tr .+ 0.5P, P) .- 0.5P
    flux = [abs(x) < 0.07 ? 0.99 : 1.0 for x in ph] .+ 2e-4 .* randn(MersenneTwister(1), 240)
    rvch = Dict("source" => "RV", "data" => Dict("A" => (
        t = collect(0.0:0.7:6.0), rv = 20.0 .* sin.(2π .* collect(0.0:0.7:6.0) ./ P),
        rv_err = fill(2.0, 9))))
    phch = Dict("source" => "PM", "data" => Dict("T" => (
        t = tr, flux = flux, flux_err = fill(2e-4, 240))))
    planets = (k1 = (P = UniformPrior(2.9, 3.1), K = UniformPrior(0.0, 100.0),
                     rr = UniformPrior(0.02, 0.3), b = UniformPrior(0.0, 1.0),
                     sesinw = 0.0, secosw = 0.0),)
    layout_names(tgt) = tgt.params.layout.unfrozen_names

    @testset "the density is a sampled parameter only on request" begin
        on  = Nereus._target_from([rvch, phch], planets; use_rho_s = true,
                                  time_anchor = :phic)
        off = Nereus._target_from([rvch, phch], planets; time_anchor = :phic)
        @test on.params.config.parametrization.use_rho_s
        @test "rho_s" in layout_names(on)
        @test !off.params.config.parametrization.use_rho_s
        @test !("rho_s" in layout_names(off))
    end

    @testset "it needs a transit channel" begin
        rv2 = Dict("source" => "RV", "data" => Dict("B" => (
            t = [0.0, 1.0, 2.0], rv = [1.0, 2.0, 3.0], rv_err = [1.0, 1.0, 1.0])))
        err = try fit_joint(rvch, rv2; planets = planets, use_rho_s = true)
              catch e; e end
        @test err isa ArgumentError
        @test occursin("transit channel", sprint(showerror, err))
        @test_throws ArgumentError build_target(
            planets = planets, use_rho_s = true,
            rv = (A = (data = rvch["data"]["A"],),))
    end

    @testset "fit_joint samples rho_s under its prior" begin
        r = fit_joint(rvch, phch; planets = planets, use_rho_s = true,
                      time_anchor = :phic,
                      priors = Dict{String,PriorSpec}("rho_s" => NormalPrior(1.5, 0.1, 0.5, 5.0)),
                      engine = Dict("engine" => "pt_emcee", "options" => Dict(
                          :n_temps => 2, :n_walkers => 16, :n_steps => 120,
                          :n_burnin => 60, :seed => 3)),
                      science = false)
        ρ = r.summary["params"]["rho_s"]
        @test 0.5 <= ρ["lo3sig"] && ρ["hi3sig"] <= 5.0
        @test isfinite(ρ["median"])
        @test !haskey(r.summary["params"], "a_k1")
    end
end

@testset "fit_transit: use_rho_s" begin
    P = 3.0
    tr = collect(range(0.0, 6.0; length = 240))
    ph = mod.(tr .+ 0.5P, P) .- 0.5P
    flux = [abs(x) < 0.07 ? 0.99 : 1.0 for x in ph] .+ 2e-4 .* randn(MersenneTwister(1), 240)
    phot = Dict("T" => (t = tr, flux = flux, flux_err = fill(2e-4, 240)))
    pri = Dict{String,PriorSpec}("P_k1" => UniformPrior(2.9, 3.1),
                                 "rho_s" => NormalPrior(1.5, 0.1, 0.5, 5.0))
    eng = Dict("engine" => "pt_emcee", "options" => Dict(
        :n_temps => 2, :n_walkers => 16, :n_steps => 120, :n_burnin => 60, :seed => 3))
    r = fit_transit(phot; use_rho_s = true, time_anchor = :phic, priors = pri,
                    engine = eng, science = false)
    ρ = r.summary["params"]["rho_s"]
    @test 0.5 <= ρ["lo3sig"] && ρ["hi3sig"] <= 5.0
    r0 = fit_transit(phot; time_anchor = :phic,
                     priors = Dict{String,PriorSpec}("P_k1" => UniformPrior(2.9, 3.1)),
                     engine = eng, science = false)
    @test !haskey(r0.summary["params"], "rho_s")
    # the old `rho_star` keyword points at the new spelling
    err = try fit_transit(phot; rho_star = (0.53, 0.05)) catch e; e end
    @test err isa ArgumentError && occursin("use_rho_s", sprint(showerror, err))
end
