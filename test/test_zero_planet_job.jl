# A zero-planet job read from a JSON file: `model.planet_modes = []` parses as
# an empty JSON3.Array, and an untyped comprehension over it is a
# Vector{Union{}}, which Params' `planet_modes::AbstractVector{PlanetDataSources}`
# refused with a TypeError before any sampling. Noise-only baselines (N = 0)
# are an ordinary model-comparison job.
using Test, Nereus, JSON3, Random

@testset "run_job: zero-planet job from a JSON file" begin
    rng = MersenneTwister(11)
    n = 40
    t = sort(2459000.0 .+ 300 .* rand(rng, n))
    rv = 3.0 .* randn(rng, n)
    out = mktempdir()
    cfg = Dict{String,Any}(
        "seed" => 3, "output_dir" => out,
        "star" => Dict("M_s" => 1.0, "R_s" => 1.0),
        "model" => Dict("max_kplanet" => 0, "planet_modes" => String[]),
        "priors" => Dict{String,Any}(),
        "data" => Dict("rv" => Dict("values" => Dict(
            "bjd" => t, "rv" => rv, "rv_err" => fill(3.0, n),
            "instrument" => fill("A", n)))),
        "sampler" => Dict("name" => "pt_emcee",
            "kwargs" => Dict("n_temps" => 2, "n_walkers" => 16, "n_steps" => 40,
                             "n_burnin" => 10, "show_progress" => false)),
        "output" => Dict("plots" => String[], "ppc" => false,
                         "detection_limits" => false, "loo" => false,
                         "show_progress" => false))
    path = joinpath(out, "config.json")
    open(io -> JSON3.write(io, cfg), path, "w")

    # The parsed config is what trips it: the empty array is a JSON3.Array.
    jc = JSON3.read(read(path, String))
    @test jc[:model][:planet_modes] isa JSON3.Array && isempty(jc[:model][:planet_modes])

    summary = Nereus.run_job(path)
    @test summary["status"] == "ok"
    @test !any(startswith(String(k), "P_k") for k in keys(summary["fitted"]["parameters"]))
    @test haskey(summary["fitted"]["parameters"], "sigma_A")
end
