# Per-instrument photometric baseline trends.
#
# `phot_trend_order` given as a Dict sets the polynomial order instrument by
# instrument (unlisted instruments get none), and every trend is evaluated in the
# time of its own instrument group: x = (t − mean) / (span / 2), so x runs over
# [−1, 1] within that group whatever the full photometric baseline. A single-night
# ground-based light curve next to a multi-year survey therefore gets a usable
# trend, which the old full-span normalisation (x ≈ 2×10⁻⁴ over one night of a
# 1000-d baseline) could not give.
using Test
using Nereus
using Statistics: mean

@testset "per-instrument photometric trend" begin
    tS = collect(range(0.0, 1000.0; length = 400))        # survey, 1000-d baseline
    tN = collect(range(500.0, 500.25; length = 60))        # one ground-based night
    slope = 0.02                                           # flux per day on the night
    fN = 1.0 .+ slope .* (tN .- mean(tN))
    data = Data(; t_phot = vcat(tS, tN), flux = vcat(ones(length(tS)), fN),
                flux_err = fill(1e-3, length(tS) + length(tN)),
                phot_inst = vcat(fill(1, length(tS)), fill(2, length(tN))))
    ic = InstrumentConfig(pm = ["SURVEY", "NIGHT"])

    @testset "orders per instrument" begin
        p = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[], instruments = ic,
                   data = data, stability = :none,
                   phot_trend_order = Dict("NIGHT" => 1))
        nm = p.layout.names
        @test "phot_c1_NIGHT" in nm
        @test !("phot_c1_SURVEY" in nm)
        @test !any(startswith("phot_c2_"), nm)
        @test p.config.phot_trend_order == 0
        @test p.config.phot_trend_orders == Dict("NIGHT" => 1)

        # mixed orders, the global Int still works and applies to every instrument
        p2 = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[], instruments = ic,
                    data = data, stability = :none,
                    phot_trend_order = Dict("NIGHT" => 2, "SURVEY" => 1))
        @test "phot_c2_NIGHT" in p2.layout.names
        @test "phot_c1_SURVEY" in p2.layout.names && !("phot_c2_SURVEY" in p2.layout.names)
        p3 = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[], instruments = ic,
                    data = data, stability = :none, phot_trend_order = 1)
        @test "phot_c1_NIGHT" in p3.layout.names && "phot_c1_SURVEY" in p3.layout.names

        @test_throws ArgumentError Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
            instruments = ic, data = data, stability = :none,
            phot_trend_order = Dict("NIGHT" => 3))
        @test_throws ArgumentError Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
            instruments = ic, data = data, stability = :none,
            phot_trend_order = Dict("NOPE" => 1))
        # instruments sharing an offset share the trend, so their orders must agree
        @test_throws ArgumentError Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
            instruments = ic, data = data, stability = :none,
            sharing = Dict(:pm_offset => [["SURVEY", "NIGHT"]]),
            phot_trend_order = Dict("NIGHT" => 1))
    end

    @testset "trend time is the instrument's own" begin
        p = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[], instruments = ic,
                   data = data, stability = :none,
                   phot_trend_order = Dict("NIGHT" => 1),
                   priors = Dict{String, PriorSpec}("phot_c1_NIGHT" => UniformPrior(-0.5, 0.5)))
        th = Theta(p)
        c = 0.0025                                        # = slope × span / 2
        th.values[p.layout.name_to_idx["phot_c1_NIGHT"]] = c
        for nm in ("offset_SURVEY", "offset_NIGHT")
            haskey(p.layout.name_to_idx, nm) && (th.values[p.layout.name_to_idx[nm]] = 0.0)
        end
        pred, _ = Nereus.phot_predictions(th, data)
        night = pred[length(tS)+1:end]
        # x = −1 at the first cadence of the night and +1 at the last
        @test night[1] ≈ 1.0 - c atol = 1e-12
        @test night[end] ≈ 1.0 + c atol = 1e-12
        @test night ≈ fN atol = 1e-12                    # recovers the injected slope
        @test all(pred[1:length(tS)] .== 1.0)            # the survey has no trend
        # the likelihood sees the same continuum: the exact slope beats no slope
        ll(cv) = (th.values[p.layout.name_to_idx["phot_c1_NIGHT"]] = cv;
                  Nereus.transit_log_likelihood(th, data))
        l0 = ll(0.0)
        @test ll(c) > l0 + 50                            # Δ ln L ≈ 64.6 here
    end

    @testset "job config accepts a per-instrument map" begin
        pm(t, f, name) = Dict{String, Any}("instrument" => name, "exposure_time" => 0.0,
            "values" => Dict("bjd" => t, "flux" => f, "flux_err" => fill(1e-3, length(t))))
        cfg(order) = Dict{String, Any}("version" => "1.0", "output_dir" => mktempdir(),
            "data" => Dict{String, Any}("transit_photometry" => Any[
                pm(tS, ones(length(tS)), "SURVEY"), pm(tN, fN, "NIGHT")]),
            "model" => Dict{String, Any}("max_kplanet" => 0, "planet_modes" => String[],
                                         "phot_trend_order" => order),
            "sampler" => Dict{String, Any}("name" => "pt_emcee"))
        @test Nereus._validate_config(cfg(Dict("NIGHT" => 1, "SURVEY" => 0))) === nothing
        @test_throws ArgumentError Nereus._validate_config(cfg(Dict("NIGHT" => 5)))
        @test_throws ArgumentError Nereus._validate_config(cfg(3))
        # the map reaches the Params the runner builds
        c = cfg(Dict("NIGHT" => 1))
        kw = Nereus._phot_trend_order_kw(c["model"]["phot_trend_order"])
        @test kw == Dict("NIGHT" => 1)
        d, _, ipm = Nereus._build_data(c["data"])
        p = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                   instruments = InstrumentConfig(pm = ipm), data = d, stability = :none,
                   phot_trend_order = kw)
        @test "phot_c1_NIGHT" in p.layout.names && !("phot_c1_SURVEY" in p.layout.names)
    end
end
