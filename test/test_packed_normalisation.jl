# The packed priors the samplers evaluate are the normalised densities the
# Distributions route evaluates. A truncated Normal used to be packed without
# its truncation mass and a Beta without log B(a, b): constants that cancel in a
# fixed-dimension acceptance ratio but not in trans-dimensional birth and death
# ratios (which evaluate the born parameters' priors) nor in the evidence. An
# affine Beta (`BetaPrior(a, b; lo, hi)` off [0, 1]) fell through to a Uniform.

# Five-point Gauss-Legendre on `n` panels between consecutive points of `pts`
# (the support's ends, any interior kinks of a folded density, and points
# crowding an integrable endpoint singularity such as a Beta's with a < 1).
const _PN_GL_X = (-0.9061798459386640, -0.5384693101056831, 0.0,
                  0.5384693101056831, 0.9061798459386640)
const _PN_GL_W = (0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                  0.4786286704993665, 0.2369268850561891)
function _pn_mass(f, lo, hi; breaks = Float64[], n = 2000)
    pts = sort(unique([lo; filter(b -> lo < b < hi, breaks); hi]))
    s = 0.0
    for k in 1:length(pts)-1, j in 1:n
        a = pts[k] + (j - 1) * (pts[k+1] - pts[k]) / n
        b = pts[k] + j * (pts[k+1] - pts[k]) / n
        m, h = (a + b) / 2, (b - a) / 2
        s += h * sum(_PN_GL_W[q] * f(m + h * _PN_GL_X[q]) for q in 1:5)
    end
    return s
end

@testset "packed priors are normalised and agree with Distributions" begin
    cases = [
        ("truncated Normal, both sides",   NormalPrior(0.3, 0.2, 0.0, 1.0)),
        ("truncated Normal, mostly cut",   NormalPrior(-1.0, 0.5, 0.0, 2.0)),
        ("truncated Normal, far tail",     NormalPrior(10.0, 1.0, 0.0, 5.0)),
        ("truncated Normal, lower only",   NormalPrior(1.0, 2.0, 0.0, Inf)),
        ("truncated Normal, upper only",   NormalPrior(1.0, 2.0, -Inf, 0.5)),
        ("Normal, untruncated",            NormalPrior(1.0, 2.0)),
        ("Beta on [0, 1]",                 BetaPrior(0.867, 3.03)),
        ("Beta on [2, 7]",                 BetaPrior(2.0, 5.0; lo = 2.0, hi = 7.0)),
        ("Uniform",                        UniformPrior(-3.0, 4.0)),
        ("LogUniform",                     LogUniformPrior(0.1, 100.0)),
        ("ModJeffreys",                    ModJeffreysPrior(1.0, 50.0)),
        ("Sine",                           SinePrior()),
        ("WrappedSine",                    WrappedSinePrior()),
        ("polar truncated Normal",         Nereus.polar_lift(NormalPrior(1.2, 0.4, 0.0, 1.0π))),
        ("circular truncated Normal",      Nereus.circular_lift(NormalPrior(0.5, 0.8, -1.0π, 1.0π))),
        ("circular Uniform, two periods",  Nereus.circular_lift(UniformPrior(-2π, 2π))),
    ]
    for (label, ps) in cases
        @testset "$label" begin
            pp = Nereus.pack_priors([ps])
            # A typed packing, not the Uniform fallback.
            @test !(pp.type_ids[1] == Nereus.PACKED_UNIFORM && !(ps.dist isa Distributions.Uniform))
            lo, hi = ps.lo, ps.hi
            a = isfinite(lo) ? lo : -40.0
            b = isfinite(hi) ? hi : 40.0
            for t in range(0.013, 0.987; length = 9)
                x = a + t * (b - a)
                @test Nereus.eval_packed_logpdf(x, pp, 1) ≈ logpdf(ps, x) atol = 1e-10
            end
            # The six-field form recomputes the normaliser itself.
            if pp.type_ids[1] in (Nereus.PACKED_NORMAL, Nereus.PACKED_BETA)
                x = a + 0.37 * (b - a)
                @test Nereus.eval_packed_logpdf(x, pp.type_ids[1], pp.params[1, 1],
                                                pp.params[1, 2], lo, hi) ≈
                      logpdf(ps, x) atol = 1e-10
            end
            mass = _pn_mass(x -> exp(Nereus.eval_packed_logpdf(x, pp, 1)), a, b;
                            breaks = [0.0, π / 2, 1.0π, 1e-8, 1e-6, 1e-4, 1e-2])
            @test mass ≈ 1.0 atol = 1e-6
        end
    end
end

@testset "PHYSICAL_BOUNDS are in the model's units" begin
    # The model works in radians; "inc" was (0, 180), degrees.
    @test physical_bounds("inc_k1") == (0.0, Float64(π))
    for (base, (lo, hi)) in Nereus.PHYSICAL_BOUNDS
        # No angle bound in degrees: nothing wider than a few periods.
        if base in ("w", "Mo", "phic", "inc", "Omega", "lambda", "i_star")
            @test abs(lo) <= 2π + 1e-12 && abs(hi) <= 2π + 1e-12
        end
    end
end
