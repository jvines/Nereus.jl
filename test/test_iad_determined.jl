# Intermediate astrometry with n <= n_q abscissae makes iad_log_likelihood a
# constant 0 (the design is exactly determined, every orbit fits perfectly): a
# fit that is the prior, silently. It is refused where the target is built.
using Test, Nereus, Random

@testset "IAD with n <= n_q is refused at build time" begin
    function iad(n; seed = 1)
        rng = MersenneTwister(seed)
        t = sort(48000.0 .+ 3000.0 .* rand(rng, n))
        Nereus.IADData(t = t, abscissa = 0.3 .* randn(rng, n), abscissa_err = fill(0.3, n),
                       psi = 2π .* rand(rng, n),
                       parallax_factor = 0.9 .* sin.(2π .* (t .- 48000.0) ./ 365.25),
                       pm_factor = (t .- 49500.0) ./ 365.25)
    end
    target(d; kw...) = Nereus._target_from(
        [Dict{String, Any}("source" => "AS", "iad" => d, "hip_scans" => false)],
        Nereus._planet_spec(1; block = Nereus.default_astrom_planet()); kw...)

    # one instrument: n_q = 4
    @test Nereus._iad_n_q(1) == 4
    for n in (1, 3, 4)
        err = try target(iad(n)) catch e; e end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("n = $n", msg) && occursin("n_q = 4", msg) && occursin("IAD", msg)
    end
    @test target(iad(5)) isa Nereus.NereusTarget
    # the fit entry point refuses it before sampling anything
    @test_throws ArgumentError fit_astrometry(; iad = iad(4), hip_scans = false)
    # two instruments: n_q = 6, and the message names both
    two = Nereus.merge_iad([iad(3; seed = 2), iad(3; seed = 3)])
    err = try target(two) catch e; e end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("n = 6", msg) && occursin("n_q = 6", msg) && occursin("2 instrument", msg)
    # no IAD at all is not an IAD that is too short
    @test Nereus.require_iad_determined(nothing, (; iad = nothing)) === nothing
end
