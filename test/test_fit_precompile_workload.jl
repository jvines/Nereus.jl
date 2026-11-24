# Nereus precompiles the fit path (src/fit_precompile.jl) so the first fit in a
# fresh process does not spend about two minutes compiling. The workload runs
# only while Nereus precompiles, where an error is swallowed into a failed
# precompilation of the whole package -- so run its inputs here, through the
# same entry points (`fit_rv`, `fit_transit`, `fit_astrometry`, `fit_joint`,
# `run_job`), and check what it relies on: every fit finishes and writes its
# summary, inside a temporary directory only, and the same seeds give the same
# chain.
using Test, Nereus, Logging

@testset "fit precompile workload" begin
    @test Nereus._PRECOMPILE_FIT === true     # on unless the preference says off
    fx = Nereus._fit_precompile_inputs()
    quiet(f) = Logging.with_logger(Logging.NullLogger()) do
        redirect_stdio(f; stdout = devnull, stderr = devnull)
    end
    dir = mktempdir(); dir2 = mktempdir()
    try
        quiet() do
            Nereus._fit_precompile_run(fx, dir)
        end
        for sub in ("fit_rv", "fit_transit", "fit_astrometry", "fit_joint", "run_job")
            @test isfile(joinpath(dir, sub, "summary.json"))
            @test isfile(joinpath(dir, sub, "chains.nc"))
        end
        # The same seeds, the same chains (the astrometry fit and the joint fit
        # stand for the rest).
        quiet() do
            Nereus._fit_precompile_run(fx, dir2)
        end
        for sub in ("fit_astrometry", "fit_joint", "run_job")
            c1, _ = Nereus.load_chains(joinpath(dir, sub, "chains.nc"))
            c2, _ = Nereus.load_chains(joinpath(dir2, sub, "chains.nc"))
            @test Array(c1.value.data) == Array(c2.value.data)
        end
    finally
        rm(dir; recursive = true, force = true)
        rm(dir2; recursive = true, force = true)
    end
end
