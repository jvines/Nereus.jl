# fit_transit took `limb_darkening`, `rho_star` and `gravity_darkening` and read
# none of them: `gravity_darkening = true` fitted a symmetric transit and said
# nothing, and a `rho_star` prior was dropped. Each throws now, before any
# target is built, unless left at its default.
using Test, Nereus

@testset "fit_transit: keywords it does not implement throw" begin
    t = collect(0.0:0.01:1.0)
    phot = Dict("TESS" => (t = t, flux = ones(length(t)), flux_err = fill(1e-3, length(t))))
    @test_throws ArgumentError fit_transit(phot; gravity_darkening = true)
    @test_throws ArgumentError fit_transit(phot; rho_star = (0.53, 0.05))
    @test_throws ArgumentError fit_transit(phot; limb_darkening = :nonlinear)
    @test_throws ArgumentError fit_transit(phot; limb_darkening = "linear")
    # the message says where the option does exist
    err = try fit_transit(phot; gravity_darkening = true) catch e; e end
    @test occursin("PM_GD", sprint(showerror, err))
end
