# The `post-fit` progress lines of run_job and the fit_* route (`_PostFit`,
# src/runner.jl), and the stage timings they leave in the summary.
#
# A production fit finished its last plot -- the plot bar at 100% -- and then
# spent an hour in the bridge bootstrap with nothing printed, so its caller
# could not tell a long post-fit stage from a hung one. Every post-fit stage
# now prints a line as it starts, and `stage=done` comes only once
# summary.json is on disk.
using Test, Nereus, Random, JSON3
include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))

function _pf_capture(f)
    out = mktemp() do path, io
        redirect_stdout(io) do
            f()
        end
        flush(io); read(path, String)
    end
    return filter(!isempty, split(out, r"[\r\n]+"))
end

_pf_stage(l) = (m = match(r"^post-fit .*\| stage=([a-z_]+)", l); m === nothing ? nothing : m[1])

@testset "post-fit progress" begin
    @testset "_PostFit" begin
        lines = _pf_capture() do
            pf = Nereus._PostFit(("a", "b"))
            Nereus._postfit_stage!(pf, "a")
            @test_throws ArgumentError Nereus._postfit_stage!(pf, "nope")
            Nereus._postfit_stage!(pf, "b")
            t = Nereus._postfit_timing(pf, Dict{String, Any}())
            @test sort(collect(keys(t["post_fit"]))) == ["a", "b"] && !haskey(t, "bridge")
            Nereus._postfit_done!(pf; stage = "write_failed")
        end
        @test _pf_stage.(lines) == ["a", "b", "write_failed"]
        @test occursin("0/2 (0.0%)", lines[1]) && occursin("2/2 (100.0%)", lines[3])
        @test !any(l -> occursin("ETA", l), lines)
        @test isempty(_pf_capture(() -> Nereus._postfit_done!(Nereus._PostFit(("a",);
                                                                             enabled = false))))
    end

    stages = ["ppc", "detection_limits", "loo", "fit_health", "diagnostics", "summary",
              "evidence", "science", "write"]

    @testset "run_job" begin
        cfg = _easy_rv_cfg()
        cfg["sampler"]["kwargs"] = Dict("show_progress" => false, "n_temps" => 2,
                                        "n_walkers" => 32, "n_steps" => 300,
                                        "n_burnin" => 100)
        res = nothing
        lines = _pf_capture(() -> (res = Nereus.run_job(cfg)))
        s = res isa AbstractDict ? res : res.summary
        @test filter(!isnothing, _pf_stage.(lines)) == vcat(stages, "done")
        # The done line is the last thing printed, and the summary on disk
        # carries the timings taken just before it was written.
        @test _pf_stage(lines[end]) == "done"
        disk = JSON3.read(read(joinpath(cfg["output_dir"], "summary.json"), String);
                          allow_inf = true)          # the summary carries NaN
        pft = disk["timing"]["post_fit"]
        @test sort(String.(collect(keys(pft)))) == sort(setdiff(stages, ["write"]))
        @test all(v -> v >= 0, values(pft))
        # The bridge reports between the evidence and science lines.
        ie = findfirst(l -> _pf_stage(l) == "evidence", lines)
        is = findfirst(l -> _pf_stage(l) == "science", lines)
        ib = findfirst(l -> startswith(l, "bridge posterior"), lines)
        @test ib !== nothing && ie < ib < is
        bt = s["timing"]["bridge"]
        @test bt["n_post"] > 0 && haskey(bt, "posterior_evals") && haskey(bt, "primary_solve")
        @test disk["timing"]["bridge"]["n_post"] == bt["n_post"]

        # show_progress = false: no post-fit or bridge lines, the same timings.
        cfg2 = _easy_rv_cfg()
        cfg2["sampler"]["kwargs"] = cfg["sampler"]["kwargs"]
        cfg2["output"]["show_progress"] = false
        res2 = nothing
        lines2 = _pf_capture(() -> (res2 = Nereus.run_job(cfg2)))
        @test !any(l -> startswith(l, "post-fit") || startswith(l, "bridge"), lines2)
        s2 = res2 isa AbstractDict ? res2 : res2.summary
        @test haskey(s2["timing"], "post_fit")
    end

    @testset "fit_rv" begin
        rng = MersenneTwister(5)
        t = sort!(200 .* rand(rng, 40))
        rv = 15 .* sin.(2π .* t ./ 12.3) .+ randn(rng, 40)
        eng = Dict("engine" => "pt_emcee",
                   "options" => Dict("n_temps" => 2, "n_walkers" => 16,
                                     "n_steps" => 200, "n_burnin" => 50,
                                     "show_progress" => false))
        r = nothing
        lines = _pf_capture() do
            r = fit_rv(Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.0, 40)));
                       planets = 1, engine = eng, output_dir = mktempdir())
        end
        @test filter(!isnothing, _pf_stage.(lines)) ==
              ["ppc", "detection_limits", "loo", "fit_health", "evidence", "science",
               "write", "done"]
        @test haskey(r.summary["timing"]["post_fit"], "evidence")
        # science = false: the write is the only stage.
        lines = _pf_capture() do
            fit_rv(Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.0, 40)));
                   planets = 1, engine = eng, output_dir = mktempdir(), science = false)
        end
        @test filter(!isnothing, _pf_stage.(lines)) == ["write", "done"]
    end
end
