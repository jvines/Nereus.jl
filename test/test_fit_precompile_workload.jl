# Nereus precompiles the fit path (src/fit_precompile.jl) so the first fit in a
# fresh process does not spend about two minutes compiling. The workload runs
# only while Nereus precompiles, where an error is swallowed into a failed
# precompilation of the whole package -- so run its inputs here, through the
# same entry points, and check what it relies on: both fits finish and write
# their summary, inside a temporary directory only.
using Test, Nereus, Logging

@testset "fit precompile workload" begin
    @test Nereus._PRECOMPILE_FIT === true     # on unless the preference says off
    fx = Nereus._fit_precompile_inputs()
    dir = mktempdir()
    try
        Logging.with_logger(Logging.NullLogger()) do
            redirect_stdio(; stdout = devnull, stderr = devnull) do
                Nereus.fit_rv(fx.data; engine = fx.engine, output_dir = joinpath(dir, "fit_rv"))
                cfg = copy(fx.cfg); cfg["output_dir"] = joinpath(dir, "run_job")
                Nereus.run_job(cfg)
            end
        end
        for sub in ("fit_rv", "run_job")
            @test isfile(joinpath(dir, sub, "summary.json"))
            @test isfile(joinpath(dir, sub, "chains.nc"))
        end
        # The same seeds, the same chain.
        dir2 = mktempdir()
        cfg = copy(fx.cfg); cfg["output_dir"] = dir2
        Logging.with_logger(Logging.NullLogger()) do
            redirect_stdio(; stdout = devnull, stderr = devnull) do
                Nereus.run_job(cfg)
            end
        end
        c1, _ = Nereus.load_chains(joinpath(dir, "run_job", "chains.nc"))
        c2, _ = Nereus.load_chains(joinpath(dir2, "chains.nc"))
        @test Array(c1.value.data) == Array(c2.value.data)
        rm(dir2; recursive = true, force = true)
    finally
        rm(dir; recursive = true, force = true)
    end
end
