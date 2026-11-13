# The release bundle precompiles a plot set (src/plotting/precompile.jl), but
# only when the `precompile_plots` preference is set, so the suite never runs
# that workload. Run its fixture and plot set here, under the default
# parametrization, so a renamed parameter fails a test instead of the bundle.
using Test, Nereus, Logging

@testset "plot precompile workload" begin
    fx = Nereus._plot_precompile_fixture()
    @test Set(fx.params.layout.unfrozen_names) ==
          Set(String.(filter(!=(:lp), names(fx.chains))))
    dir = mktempdir()
    try
        Logging.with_logger(Logging.NullLogger()) do
            Nereus._make_plots(fx.cfg, fx.chains, fx.params, fx.data, dir)
        end
        pngs = filter(endswith(".png"), readdir(joinpath(dir, "plots")))
        @test "corner_K1.png" in pngs
        @test "corner_noise.png" in pngs
        allpngs = [f for (_, _, fs) in walkdir(dir) for f in fs if endswith(f, ".png")]
        @test length(allpngs) > length(pngs)   # the RV, posterior and trace figures too
    finally
        rm(dir; recursive = true, force = true)
    end
end
