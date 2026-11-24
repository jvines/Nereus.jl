# fit_transit took `limb_darkening`, `rho_star` and `gravity_darkening` and read
# none of them: `gravity_darkening = true` fitted a symmetric transit and said
# nothing, and a `rho_star` prior was dropped. `rho_star` and a limb-darkening
# law the model lacks throw, before any target is built; `gravity_darkening =
# true` is the PM_GD planet mode.
using Test, Nereus

@testset "fit_transit: keywords it does not implement throw" begin
    t = collect(0.0:0.01:1.0)
    phot = Dict("TESS" => (t = t, flux = ones(length(t)), flux_err = fill(1e-3, length(t))))
    @test_throws ArgumentError fit_transit(phot; gravity_darkening = "yes")
    @test_throws ArgumentError fit_transit(phot; rho_star = (0.53, 0.05))
    @test_throws ArgumentError fit_transit(phot; limb_darkening = :nonlinear)
    @test_throws ArgumentError fit_transit(phot; limb_darkening = "linear")
    # the message says what replaces it
    err = try fit_transit(phot; rho_star = (0.53, 0.05)) catch e; e end
    @test occursin("use_rho_s", sprint(showerror, err))
    # gravity_darkening = true is the PM_GD planet mode, not a flag nothing reads
    ch = Dict{String,Any}("source" => "PM", "gravity_darkening" => true,
                          "data" => phot)
    tg = Nereus._target_from([ch], 1)
    @test Nereus.has_gd(tg.params.config.planet_modes[1])
end
