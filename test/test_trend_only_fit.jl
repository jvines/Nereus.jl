# A fit with no planet: offset, jitter and, with `trend_order`, a trend. It used
# to stop before sampling -- build_target refused an empty planet set, and an
# empty planets list (a JSON `[]`, a Vector{Any}) had no maximum to take.
using Test, Nereus, Random, Statistics
include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))

const _TO_DVDT = 0.02

function _to_data(; trend = true)
    rng = MersenneTwister(11)
    n = 80
    t = sort!(1000.0 .* rand(rng, n))
    rv = (trend ? _TO_DVDT .* (t .- 500) : zeros(n)) .+ 1.5 .* randn(rng, n) .+ 3.0
    return t, rv
end

const _TO_ENGINE = Dict("engine" => "pt_emcee",
                        "options" => Dict("n_temps" => 3, "n_walkers" => 24,
                                          "n_steps" => 1200, "n_burnin" => 400))

_to_med(r, name) = median(vec(Array(r.chains[Symbol(name)])))

@testset "trend-only fit" begin
    t, rv = _to_data()
    data = Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.5, length(t))))

    @testset "fit_rv, planets = $(repr(pl)), trend_order 1" for pl in (0, Int[], Any[])
        out = mktempdir()
        r = fit_rv(data; planets = pl, trend_order = 1, engine = _TO_ENGINE,
                   output_dir = out, plots = ["rv_timeseries", "rv_components"])
        @test r.summary["n_draws_kept"] > 0
        @test abs(_to_med(r, "dvdt") - _TO_DVDT) < 0.003
        @test isfile(joinpath(out, "plots", "models", "rv_timeseries.png"))
        @test isfile(joinpath(out, "tables", "fitted.csv"))
    end

    @testset "fit_rv, trend_order 2" begin
        r = fit_rv(data; planets = 0, trend_order = 2, engine = _TO_ENGINE,
                   output_dir = mktempdir())
        @test abs(_to_med(r, "dvdt") - _TO_DVDT) < 0.005
    end

    @testset "no planet, no trend: offset and jitter only" begin
        t0, rv0 = _to_data(trend = false)
        d0 = Dict("SIM" => (t = t0, rv = rv0, rv_err = fill(1.5, length(t0))))
        r = fit_rv(d0; planets = 0, engine = _TO_ENGINE, output_dir = mktempdir())
        @test abs(_to_med(r, "gamma_SIM") - 3.0) < 1.0
    end

    @testset "no planet needs RV alone" begin
        @test_throws ArgumentError build_target(planets = NamedTuple(), rv = NamedTuple())
    end

    @testset "run_job with an Any[] planet list, trend_order = $to" for to in (0, 1)
        cfg = _easy_rv_cfg()
        cfg["data"]["rv"]["values"] = Dict{String,Any}(
            "bjd" => t, "rv" => rv, "rv_err" => fill(1.5, length(t)),
            "instrument" => fill("HARPS", length(t)))
        cfg["model"] = Dict{String,Any}("max_kplanet" => 0, "planet_modes" => Any[],
                                        "trend_order" => to)
        cfg["sampler"]["kwargs"] = Dict("show_progress" => false, "n_temps" => 3,
                                        "n_walkers" => 24, "n_steps" => 1200,
                                        "n_burnin" => 400)
        cfg["output"]["plots"] = ["rv_timeseries"]
        res = Nereus.run_job(cfg)
        s = res isa AbstractDict ? res : res.summary
        @test s["status"] == "ok"
        tab = read(joinpath(cfg["output_dir"], "tables", "fitted.csv"), String)
        @test occursin("gamma_HARPS", tab)
        @test occursin("dvdt", tab) == (to == 1)
        @test isfile(joinpath(cfg["output_dir"], "plots", "models", "rv_timeseries.png"))
    end
end
