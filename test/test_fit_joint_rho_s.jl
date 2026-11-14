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

# --- The transit channel's options reach the model ---------------------------
# `exposure_time`, `dilution`, `jitter`, `limb_darkening`, `gravity_darkening` and
# `supersample` were carried by the channel and never read.
@testset "transit channel options" begin
    P = 3.0
    tr = collect(range(0.0, 9.0; length = 430))            # 30-min cadence
    ph = mod.(tr .+ 0.5P, P) .- 0.5P
    flux = [abs(x) < 0.08 ? 0.985 : 1.0 for x in ph]
    chan(; kw...) = Dict{String,Any}("source" => "PM", "data" => Dict("T" => (
        t = tr, flux = flux, flux_err = fill(3e-4, length(tr)))), (String(k) => v for (k, v) in kw)...)
    planets = (k1 = (P = UniformPrior(2.9, 3.1), rr = UniformPrior(0.02, 0.3),
                     b = UniformPrior(0.0, 1.0), sesinw = 0.0, secosw = 0.0),)
    mk(c; kw...) = Nereus._target_from([c], planets; time_anchor = :phic, M_s = 1.0, R_s = 1.0, kw...)
    function theta_y(tg)
        th = Theta{Float64}(tg.params); L = tg.params.layout; rng = MersenneTwister(5)
        for (d, ps) in enumerate(L.unfrozen_priors)
            lo, hi = bounds(ps); v = rand(rng, ps.dist); k = 0
            while !(lo <= v <= hi) && k < 1000; v = rand(rng, ps.dist); k += 1; end
            th.values[L.unfrozen_idx[d]] = v
        end
        Nereus.transform_forward([th.values[i] for i in L.unfrozen_idx], tg.transform)
    end
    lp(tg, y) = LogDensityProblems.logdensity(tg, y)

    base = mk(chan()); y = theta_y(base)
    @test isempty(base.data.exposure_times)
    lp0 = lp(base, y)

    @testset "exposure_time" begin
        t_exp = mk(chan(exposure_time = 1800.0))
        @test t_exp.data.exposure_times ≈ fill(1800.0 / 86400, length(tr))
        @test abs(lp(t_exp, y) - lp0) > 1.0                 # finite exposure smears a 30-min cadence
        @test lp(mk(chan(exposure_time = Dict("T" => 1800.0))), y) == lp(t_exp, y)
        @test lp(mk(chan(exposure_time = 120.0)), y) ≈ lp0   # 2-min: no supersampling, as before
        @test_throws ArgumentError mk(chan(exposure_time = -1.0))
    end

    @testset "dilution and jitter are priors on the instrument's slot" begin
        d = mk(chan(dilution = Dict("T" => 0.2)))
        @test d.params.config.priors["dilution_T"] == FixedPrior(0.2)
        @test abs(lp(d, y) - lp0) > 1.0
        j = mk(chan(jitter = Dict("type" => "uniform", "lo" => 0.0, "hi" => 0.01)))
        @test "jitter_T" in j.params.layout.unfrozen_names
        @test_throws ArgumentError mk(chan(dilution = 0.1);
            priors = Dict{String,PriorSpec}("dilution_T" => FixedPrior(0.0)))
    end

    @testset "what the model lacks throws" begin
        @test_throws ArgumentError mk(chan(limb_darkening = "linear"))
        @test_throws ArgumentError mk(chan(limb_darkening = "nonlinear"))
        @test_throws ArgumentError mk(chan(gravity_darkening = true))
        @test_throws ArgumentError mk(chan(supersample = 5))
        @test mk(chan(limb_darkening = "kipping")) isa NereusTarget      # same model as "quadratic"
    end
end
