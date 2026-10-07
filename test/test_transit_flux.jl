# The quadratic limb-darkened flux the likelihood evaluates (`_quad_flux`,
# src/transit.jl) is `Transits.compute` restructured, and has to be the same
# function to the bit: same values, every branch of compute_uniform /
# compute_linear / compute_quadratic, every n_max, and for ForwardDiff duals the
# same partials. The departures are on purpose, both one ulp inside the second
# and third contacts: there `compute` throws a DomainError, or at r = 0.5 returns
# a flux 0.74 below its neighbours, and `_quad_flux` returns the flux at the
# tangency instead -- a sampler must never abort on a geometry.
using Test
using Nereus
using Random
using ForwardDiff
import Transits
using Nereus: _quad_flux, transit_flux, transit_flux_uniform

_out(f) = try f() catch err; typeof(err) end
_ulps(x, k) = (for _ in 1:abs(k); x = k > 0 ? nextfloat(x) : prevfloat(x); end; x)

@testset "quadratic limb darkening: restructured flux === Transits.compute" begin
    rng = MersenneTwister(4)
    lds = [Transits.QuadLimbDark(u) for u in ([0.4, 0.26], [0.0, 0.5], [0.7, -0.1],
                                              [2.0, -1.0], [0.1, 0.9], [0.3], Float64[])]
    n = 0; same = 0; n_err = 0; worst = 0.0
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
        a = _quad_flux(ld, b, r)
        c = _out(() -> Transits.compute(ld, b, r))
        n += 1
        if c isa Type || (r == 0.5 && b == prevfloat(0.5))   # the tangency
            n_err += 1
            nb = (_quad_flux(ld, prevfloat(b), r), _quad_flux(ld, nextfloat(b), r))
            worst = max(worst, maximum(abs.(nb .- a)))
        else
            same += a === c
        end
    end
    @test same == n - n_err                # bit-identical everywhere else
    @test n > 150_000 && n_err > 0         # ... and the tangency occurs
    @test worst < 1e-14                    # where it does, continuous
    # r = 0.5, b = prevfloat(0.5): compute's own value is 0.74 off
    ld = lds[1]
    @test abs(Transits.compute(ld, prevfloat(0.5), 0.5) - Transits.compute(ld, 0.5, 0.5)) > 0.5
    @test abs(_quad_flux(ld, prevfloat(0.5), 0.5) - _quad_flux(ld, 0.5, 0.5)) < 1e-15

    # transit_flux takes it for Float64, and keeps the uniform-disc fast path
    ld = Transits.QuadLimbDark([0.4, 0.26])
    for z in (0.0, 0.3, 0.8, 0.9, 1.05, 1.2), p in (0.01, 0.116, 0.6)
        @test transit_flux(ld, z, p) === Transits.compute(ld, z, p)
        @test transit_flux(z, p, 0.4, 0.26) === Transits.compute(ld, z, p)
    end
    @test transit_flux(Transits.QuadLimbDark([0.0, 0.0]), 0.5, 0.1) ===
          Nereus.transit_flux_uniform(0.5, 0.1)

    # Duals take the same function, with Transits.compute's partials to the bit:
    # in (b, r), and in the limb-darkening coefficients
    for (z, p) in ((0.3, 0.1), (0.95, 0.12), (1.05, 0.2))
        g = ForwardDiff.gradient(v -> transit_flux(ld, v[1], v[2]), [z, p])
        h = ForwardDiff.gradient(v -> Transits.compute(ld, v[1], v[2]), [z, p])
        @test g == h
    end
    n = 0; same = 0
    for _ in 1:20_000
        ldr = lds[rand(rng, 1:length(lds))]
        r = rand(rng) < 0.5 ? 0.3 * rand(rng) : 1.2 * rand(rng)
        u = rand(rng)
        b = u < 0.1 ? r : u < 0.2 ? 1 - r : u < 0.3 ? 1 + r : (1.3 + r) * rand(rng)
        b < 0 && continue
        h = _out(() -> ForwardDiff.gradient(v -> Transits.compute(ldr, v[1], v[2]), [b, r]))
        h isa Type && continue
        n += 1
        same += isequal(ForwardDiff.gradient(v -> _quad_flux(ldr, v[1], v[2]), [b, r]), h)
    end
    @test n > 15_000 && same == n
    for (b, r) in ((0.3, 0.1), (0.95, 0.12), (0.1, 0.1), (0.5, 0.5))
        g = ForwardDiff.gradient(u -> _quad_flux(Transits.QuadLimbDark(u), b, r), [0.4, 0.26])
        h = ForwardDiff.gradient(u -> Transits.compute(Transits.QuadLimbDark(u), b, r), [0.4, 0.26])
        @test g == h
    end
end

# Every geometry a sampler can propose, at and a few ulp either side of each
# place the branches change: the second and third contacts b = 1 − r, the first
# and fourth b = 1 + r, b = r (compute_linear's cases 5-7), b = r − 1 for a
# planet larger than the star, and b = 0. Before the fix `_quad_flux` threw a
# DomainError at b = 1 − r − 1 ulp (sqrt of kc² ≈ −1e-16) and the uniform-disc
# formula at 1 − p + 1 ulp, 1 + p − 1 ulp and p − 1 + 1 ulp (acos of 1 + 1 ulp).
@testset "no geometry throws: contacts ± k ulp, r >= 1" begin
    rng = MersenneTwister(8)
    rs = vcat(0.3 .* rand(rng, 60), rand(rng, 60), 1 .+ rand(rng, 30),
              [1e-6, 1e-3, 0.116, 0.25, 0.5, 0.75, 0.999999, 1.0, 1.5, 2.0])
    lds = [Transits.QuadLimbDark(u) for u in ([0.4, 0.26], [0.7, -0.1], [0.3], Float64[])]
    ld = lds[1]
    n_threw_before = 0; n_uniform_out = 0; n = 0
    finite = true; jump = 0.0; jump_u = 0.0; dual_finite = true
    for r in rs, b0 in (1 - r, 1 + r, r, r - 1, 0.0), k in -6:6
        b = _ulps(b0, k)
        b < 0 && continue
        n += 1
        for l in lds
            f = _quad_flux(l, b, r)
            n_threw_before += _out(() -> Transits.compute(l, b, r)) isa Type
            finite &= isfinite(f)
            jump = max(jump, abs(f - _quad_flux(l, nextfloat(b), r)))
        end
        # the uniform disc, and the (z, p, u1, u2) entry point the RM model and
        # gravity darkening use
        fu = transit_flux_uniform(b, r)
        finite &= isfinite(fu) && isfinite(transit_flux(b, r, 0.4, 0.26)) &&
                  isfinite(transit_flux(b, r, 0.0, 0.0))
        jump_u = max(jump_u, abs(fu - transit_flux_uniform(nextfloat(b), r)))
        if b > 0
            a1 = (b^2 + 1 - r^2) / (2b); a2 = (b^2 + r^2 - 1) / (2b * r)
            n_uniform_out += (b < 1 + r && b > abs(1 - r)) && (abs(a1) > 1 || abs(a2) > 1)
        end
        # Duals: no throw, and finite partials in (b, r) and in the LD
        # coefficients. The uniform disc's lens formula has unbounded slopes
        # (acos, sqrt) within a few ulp of a contact whether or not it is
        # clamped, so there it only has to return. Neither has finite partials
        # at a subnormal b (b² underflows), as in `compute`; a sky separation is
        # 0 or above ~1e-8 (src/transit.jl, `_sky_separation_signed`).
        g = ForwardDiff.gradient(v -> transit_flux(ld, v[1], v[2]), [b, r])
        gl = ForwardDiff.gradient(u -> _quad_flux(Transits.QuadLimbDark(u), b, r), [0.4, 0.26])
        ForwardDiff.gradient(v -> transit_flux_uniform(v[1], v[2]), [b, r])
        (b == 0 || b > 1e-100) && (dual_finite &= all(isfinite, g) && all(isfinite, gl))
    end
    @test n > 4_000
    @test n_threw_before > 0               # Transits.compute does throw on these
    @test n_uniform_out > 0                # ... and so did the uniform formula
    @test finite
    @test jump < 1e-14                     # continuous through every contact
    @test jump_u < 1e-7                    # the lens formula's own rounding there
    @test dual_finite
end
