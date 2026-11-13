# summary.json's `provenance` block: the engine, the Nereus version and every
# prior's family and bounds/parameters, from run_job and from the prior types
# directly.
using Test, Nereus, Random, JSON3
include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))

@testset "summary provenance" begin
    pp = Nereus.prior_provenance
    @testset "prior families" begin
        @test pp(UniformPrior(1.0, 2.0)) ==
              Dict("family" => "uniform", "fixed" => false, "lo" => 1.0, "hi" => 2.0)
        @test pp(LogUniformPrior(0.1, 10.0))["family"] == "loguniform"
        @test pp(SinePrior())["family"] == "sine"
        n = pp(NormalPrior(5.0, 2.0))
        @test n["family"] == "normal" && n["mu"] == 5.0 && n["sigma"] == 2.0 &&
              n["lo"] === nothing && n["hi"] === nothing          # unbounded: null
        t = pp(NormalPrior(5.0, 2.0, 0.0, 9.0))
        @test t["family"] == "truncated_normal" && t["lo"] == 0.0 && t["hi"] == 9.0 &&
              t["mu"] == 5.0 && t["sigma"] == 2.0
        b = pp(BetaPrior(0.867, 3.03))
        @test b["family"] == "beta" && b["a"] == 0.867 && b["b"] == 3.03
        @test pp(BetaPrior(2.0, 3.0; lo = 1.0, hi = 5.0))["b"] == 3.0
        m = pp(ModJeffreysPrior(1.0, 10.0))
        @test m["family"] == "modified_jeffreys" && m["knee"] == 1.0 && m["upper"] == 10.0
        @test pp(FixedPrior(3.5)) == Dict("family" => "fixed", "fixed" => true, "value" => 3.5)
        @test pp(WrappedUniformPrior(0.0, 2π))["family"] == "wrapped_uniform"
        # Every entry is JSON-writable.
        @test JSON3.read(JSON3.write(pp(NormalPrior(0.0, 1.0))))["lo"] === nothing
    end

    @testset "version" begin
        v = Nereus._nereus_version_info()
        @test v["nereus_version"] == string(pkgversion(Nereus))
        @test v["nereus_commit"] === nothing || occursin(r"^[0-9a-f]{40}$", v["nereus_commit"])
        @test Nereus._git_commit(mktempdir()) === nothing
    end

    function _layout_order(pr)
        cfg = _easy_rv_cfg()
        data, a, b, c, d, _ = Nereus._build_data(cfg["data"])
        params, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()), a, b, c, d)
        return filter(n -> n in String.(collect(keys(pr))), params.layout.names)
    end

    function run_with(engine, kwargs)
        cfg = _easy_rv_cfg()
        cfg["sampler"] = Dict("name" => engine, "kwargs" => kwargs)
        cfg["priors"] = Dict{String, Any}(
            "gamma_HARPS" => Dict("type" => "fixed", "args" => [5.0]),
            "K_k1" => Dict("type" => "truncnorm", "args" => [12.0, 5.0, 0.0, 60.0]),
            "P_k1" => Dict("type" => "loguniform", "args" => [12.0, 12.6]))
        Nereus.run_job(cfg)
        return JSON3.read(read(joinpath(cfg["output_dir"], "summary.json"), String);
                          allow_inf = true)
    end

    for (engine, kw) in (("pt_emcee", Dict("show_progress" => false, "n_temps" => 2,
                                           "n_walkers" => 32, "n_steps" => 200,
                                           "n_burnin" => 100)),
                         ("nested", Dict("show_progress" => false, "n_live" => 100)))
        @testset "run_job $engine" begin
            disk = run_with(engine, kw)
            p = disk["provenance"]
            @test p["engine"] == engine
            @test p["nereus_version"] == string(pkgversion(Nereus))
            @test haskey(p, "nereus_commit")
            pr = p["priors"]
            @test pr["gamma_HARPS"]["family"] == "fixed" && pr["gamma_HARPS"]["value"] == 5.0
            @test pr["K_k1"]["family"] == "truncated_normal" && pr["K_k1"]["sigma"] == 5.0 &&
                  pr["K_k1"]["hi"] == 60.0
            @test pr["P_k1"]["family"] == "loguniform" && pr["P_k1"]["lo"] == 12.0
            @test !any(k -> startswith(String(k), "rho_circ") || startswith(String(k), "eta_"),
                       keys(pr))
            # The model's parameter order, not a hash order.
            @test String.(collect(keys(pr))) == _layout_order(pr)
            @test String.(collect(keys(p))) ==
                  ["engine", "nereus_version", "nereus_commit", "priors"]
            engine == "nested" && println("PROVENANCE ", JSON3.write(p))
        end
    end

    @testset "fit_rv" begin
        rng = MersenneTwister(7)
        t = sort(2459000.0 .+ 100 .* rand(rng, 40))
        rv = 15 .* sin.(2π .* t ./ 12.3) .+ randn(rng, 40)
        data = Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.0, 40)))
        pri = Dict("P_k1" => UniformPrior(12.2, 12.4), "K_k1" => UniformPrior(10.0, 20.0),
                   "gamma_SIM" => FixedPrior(0.0))
        out = mktempdir()
        r = fit_rv(data; planets = 1, priors = pri, science = false, output_dir = out,
                   plots = String[],
                   engine = Dict("engine" => "pt_emcee", "options" => Dict{String,Any}(
                       "n_temps" => 2, "n_walkers" => 16, "n_steps" => 150,
                       "n_burnin" => 50, "show_progress" => false)))
        for p in (r.summary["provenance"],
                  JSON3.read(read(joinpath(out, "summary.json"), String);
                             allow_inf = true)["provenance"])
            @test p["engine"] == "pt_emcee"
            @test p["priors"]["gamma_SIM"]["family"] == "fixed"
            @test p["priors"]["P_k1"]["hi"] == 12.4
        end
    end
end
