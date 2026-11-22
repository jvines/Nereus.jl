# `sampler: "nested"` must run on the NestedSamplers fork that ships in
# vendor/NestedSamplers. A dependency's [sources] entry is honoured only by
# Pkg 1.13, so an install on Julia 1.11/1.12 (`Pkg.add(url=...)`, the Argus
# worker) used to resolve the registry NestedSamplers 0.9.0 instead, and every
# nested job died on `MethodError: no method matching
# fit(::MersenneTwister, ::Type{MultiEllipsoid}, ::Matrix; pointvol, enlarge,
# bootstrap)`. The fork is now compiled into Nereus, so no Pkg setting can swap it.
using Test, Random, Logging, Nereus
include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))

@testset "nested uses the in-tree NestedSamplers fork" begin
    @test parentmodule(Nereus.NestedSamplers) === Nereus
    # Nereus no longer depends on the registry package of that name.
    @test Base.identify_package(Nereus, "NestedSamplers") === nothing
    B = Nereus.NestedSamplers.Bounds
    @test hasmethod(B.fit, Tuple{Random.AbstractRNG, Type, Matrix{Float64}})
    @test isdefined(B, :MLFriends)
end

@testset "run_job sampler nested, RV only, small n_live" begin
    cfg = _easy_rv_cfg()
    cfg["sampler"] = Dict("name" => "nested",
        "kwargs" => Dict("show_progress" => false, "n_live" => 60, "dlogz" => 5.0))
    res = Logging.with_logger(Logging.NullLogger()) do
        Nereus.run_job(cfg)
    end
    @test res["status"] == "ok"
    @test isfile(joinpath(cfg["output_dir"], "chains.nc"))
end
