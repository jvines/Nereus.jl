# Channel options reach the model or throw. The Python client sends fit_rv a
# whole RV channel; the daemon used to pass on its `data` alone, so every option
# on it -- trend_order, jitter, gamma_per_instrument, share, marginalize_gamma --
# was dropped, and `fit_rv(rv, trend_order=1)` fitted no trend. `_target_from`
# also ignored all of them but jitter and trend_order (fit_joint, replot).
using Test, Nereus, Random, Statistics

@testset "channel options reach the model" begin
    t1 = collect(range(0, 300, length = 30)); t2 = t1 .+ 3
    d = Dict{String,Any}("A" => (t = t1, rv = sin.(t1), rv_err = fill(1.0, 30)),
                         "B" => (t = t2, rv = sin.(t2), rv_err = fill(1.0, 30)))
    ch(; kw...) = Dict{String,Any}("source" => "RV", "data" => d,
                                   (String(k) => v for (k, v) in kw)...)
    pl = Nereus._planet_spec(1; block = Nereus.default_rv_planet(t1, sin.(t1)))
    tgt(c; kw...) = Nereus._target_from(c isa AbstractVector ? c : [c], pl; kw...)
    nm(c; kw...) = tgt(c; kw...).params.layout.unfrozen_names

    @testset "RV options" begin
        @test !("dvdt" in nm(ch()))
        @test "dvdt" in nm(ch(trend_order = 1))
        @test "d2vdt2" in nm(ch(trend_order = 2))
        # the keyword and the channel agree, or it is an error
        @test "dvdt" in nm(ch(trend_order = 1); trend_order = 1)
        @test "dvdt" in nm(ch(); trend_order = 1)
        @test_throws ArgumentError nm(ch(trend_order = 1); trend_order = 2)

        g = filter(startswith("gamma_"), nm(ch(gamma_per_instrument = false)))
        @test length(g) == 1 && occursin("+", only(g))
        s = filter(startswith("sigma_"), nm(ch(share = Dict("sigma" => [["A", "B"]]))))
        @test length(s) == 1 && occursin("+", only(s))
        @test_throws ArgumentError nm(ch(gamma_per_instrument = false);
                                      sharing = Dict("gamma" => [["A", "B"]]))
        @test_throws ArgumentError nm(ch(share = Dict("gamma" => [["A", "C"]])))
        @test_throws ArgumentError nm(ch(share = Dict("dvdt" => [["A", "B"]])))

        m = tgt(ch(marginalize_gamma = true)).params
        @test m.config.parametrization.marginalize_gamma
        @test !any(startswith("gamma_"), m.layout.unfrozen_names)

        p = tgt(ch(jitter = Dict("type" => "uniform", "lo" => 0.0, "hi" => 5.0))).params
        @test p.config.priors["sigma_A"] isa PriorSpec
        @test p.config.priors["sigma_A"].hi == 5.0
        @test !("sigma_A" in nm(ch(jitter = Dict("A" => 2.0))))
        @test "sigma_B" in nm(ch(jitter = Dict("A" => 2.0)))
        @test_throws ArgumentError nm(ch(jitter = Dict("C" => 2.0)))
        @test_throws ArgumentError nm(ch(jitter = 2.0);
                                      priors = Dict("sigma_A" => FixedPrior(1.0)))

        # cannot be honoured, so refused
        @test_throws ArgumentError nm(ch(indicators = Dict("fwhm" => ones(60))))
        @test_throws ArgumentError nm(ch(foo = 1))
        @test_throws ArgumentError nm([ch(), ch()])
        @test_throws ArgumentError nm(Dict{String,Any}("source" => "RM_R", "data" => d))
        @test_throws ArgumentError nm([ch(), Dict{String,Any}("source" => "TTV")])
    end

    @testset "transit options" begin
        ph = Dict("T" => (t = t1, flux = ones(30), flux_err = fill(1e-3, 30)))
        pch(; kw...) = Dict{String,Any}("source" => "PM", "data" => ph,
                                        (String(k) => v for (k, v) in kw)...)
        # (M_s, R_s) give a/R★ -- a transit planet with no source for it is refused
        tp = Nereus._target_from([pch(exposure_time = 1800.0,
                                      dilution = Dict("T" => 0.1))], 1;
                                 M_s = 1.0, R_s = 1.0)
        @test all(tp.data.exposure_times .≈ 1800.0 / 86400.0)
        @test tp.params.config.priors["dilution_T"].lo == 0.1
        @test_throws ArgumentError Nereus._target_from([pch(rho_star = 1.0)], 1; M_s = 1.0, R_s = 1.0)
        @test_throws ArgumentError Nereus._target_from([pch(supersample = 5)], 1; M_s = 1.0, R_s = 1.0)
    end

    @testset "fit_rv takes the channel" begin
        @test_throws ArgumentError fit_rv(Dict{String,Any}("source" => "PM", "data" => d))
        @test_throws ArgumentError fit_rv(Dict{String,Any}("source" => "RV",
                                                         "data" => Dict("A" => (t = t1,))))
        rng = MersenneTwister(11)
        t = sort!(1000.0 .* rand(rng, 80))
        rv = 0.02 .* (t .- 500) .+ 1.5 .* randn(rng, 80) .+ 3.0
        c = Dict{String,Any}("source" => "RV", "trend_order" => 1,
                             "data" => Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.5, 80))))
        eng = Dict("engine" => "pt_emcee",
                   "options" => Dict("n_temps" => 3, "n_walkers" => 24,
                                     "n_steps" => 1200, "n_burnin" => 400))
        r = fit_rv(c; planets = 0, engine = eng, science = false)
        @test abs(median(vec(Array(r.chains[:dvdt]))) - 0.02) < 0.003
    end
end
