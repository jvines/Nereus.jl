# The quadratic limb-darkened flux the likelihood evaluates for Float64
# (`_quad_flux`, src/transit.jl) is `Transits.compute` restructured, and has to
# be the same function to the bit: same values, same exceptions, every branch
# of compute_uniform / compute_linear / compute_quadratic, and every n_max.
# Duals still go through `Transits.compute`.
using Test
using Nereus
using Random
using ForwardDiff
import Transits
using Nereus: _quad_flux, transit_flux

@testset "quadratic limb darkening: restructured flux === Transits.compute" begin
    rng = MersenneTwister(4)
    lds = [Transits.QuadLimbDark(u) for u in ([0.4, 0.26], [0.0, 0.5], [0.7, -0.1],
                                              [2.0, -1.0], [0.1, 0.9], [0.3], Float64[])]
    out(f) = try f() catch err; typeof(err) end
    n = 0; same = 0; n_err = 0
    for it in 1:200_000
        ld = lds[rand(rng, 1:length(lds))]
        r = it % 10 == 0 ? rand(rng, (0.5, 1e-6, 0.999999, 1.0, 1.5, 0.25, 0.75)) :
            rand(rng) < 0.5 ? 0.3 * rand(rng) : 1.2 * rand(rng)
        u = rand(rng)
        # every case of compute_linear: b = r (r below, at and above 1/2), b + r
        # on either side of 1 and on it, the contacts, full overlap, b = 0
        b = u < 0.05 ? 0.0 : u < 0.1 ? r : u < 0.15 ? 1 - r : u < 0.2 ? 1 + r :
            u < 0.25 ? r - 1 : u < 0.3 ? (1 - r) * (1 + 1e-15 * randn(rng)) :
            (1.3 + r) * rand(rng)
        b < 0 && continue
        a = out(() -> _quad_flux(ld, b, r))
        c = out(() -> Transits.compute(ld, b, r))
        n += 1
        same += a === c
        n_err += a isa Type
    end
    @test same == n
    @test n > 150_000 && n_err > 0          # the throwing inputs throw in both

    # transit_flux takes it for Float64, and keeps the uniform-disc fast path
    ld = Transits.QuadLimbDark([0.4, 0.26])
    for z in (0.0, 0.3, 0.8, 0.9, 1.05, 1.2), p in (0.01, 0.116, 0.6)
        @test transit_flux(ld, z, p) === Transits.compute(ld, z, p)
    end
    @test transit_flux(Transits.QuadLimbDark([0.0, 0.0]), 0.5, 0.1) ===
          Nereus.transit_flux_uniform(0.5, 0.1)

    # Duals take Transits.compute and its derivatives
    for (z, p) in ((0.3, 0.1), (0.95, 0.12), (1.05, 0.2))
        g = ForwardDiff.gradient(v -> transit_flux(ld, v[1], v[2]), [z, p])
        h = ForwardDiff.gradient(v -> Transits.compute(ld, v[1], v[2]), [z, p])
        @test g == h
    end
end
