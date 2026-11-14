# The exact mass function, solved analytically (Cardano), and the places where it
# meets the transit: M_p << M_star is nowhere in Nereus.
#
#   x = M_p / M_star,  F = f / (M_star sin^3 i):   x^3 - F x^2 - 2F x - F = 0
#
# One positive root, and (discriminant -F^2 (4F + 27) < 0) the other two are
# complex for every F > 0, so no trigonometric branch is needed.

using Nereus, ForwardDiff, Test
using MCMCChains: Chains
import Nereus: mass_ratio_from_F, msec_from_K, mass_function, coupled_geometry,
               planet_geometry, rho_s_to_a_Rs

# The root to 400 bits: Newton from above the root (the cubic is convex there).
function _x_ref(F::BigFloat)
    x = F + 3 + cbrt(F)
    for _ in 1:3000
        g = x^3 - F * (1 + x)^2
        x -= g / (3x^2 - 2F * (1 + x))
    end
    return x
end

@testset "exact mass function" begin
    setprecision(BigFloat, 400) do
        @testset "Cardano root against a BigFloat root, F = 1e-16 .. 1e4" begin
            worst, worst_d = 0.0, 0.0
            for lF in range(-16, 4; length = 2001)
                F = 10.0^lF
                xr = _x_ref(BigFloat(F))
                x = mass_ratio_from_F(F)
                worst = max(worst, Float64(abs(x - xr) / xr))
                # dx/dF from the implicit function: (1+x)^2 / (3x^2 - 2F(1+x))
                dr = (1 + xr)^2 / (3xr^2 - 2 * BigFloat(F) * (1 + xr))
                d = ForwardDiff.derivative(mass_ratio_from_F, F)
                worst_d = max(worst_d, Float64(abs(d - dr) / dr))
            end
            @test worst < 2e-15
            @test worst_d < 5e-15
        end
    end

    @testset "discriminant: a single real root for every F > 0" begin
        # disc(x^3 - F x^2 - 2F x - F) = -F^2 (4F + 27)
        for F in (1e-12, 0.3, 1.0, 40.0, 1e5)
            a, b, c, d = 1.0, -F, -2F, -F
            disc = 18a*b*c*d - 4b^3*d + b^2*c^2 - 4a*c^3 - 27a^2*d^2
            @test disc ≈ -F^2 * (4F + 27) rtol = 1e-9
        end
        @test mass_ratio_from_F(0.0) == 0.0
    end

    @testset "K -> M -> K round trip: Jupiter, brown dwarf, stellar binary" begin
        Kof(m, M, P, e, s) = cbrt(m^3 * s^3 / (M + m)^2 /
                                  (Nereus._F_M_FACTOR * P * (1 - e^2)^1.5))
        for (m, M, P, e, s) in ((9.546e-4, 1.0, 4332.0, 0.05, 1.0),     # Jupiter, Sun
                                (9.546e-4, 1.0, 4332.0, 0.05, 0.6),
                                (3.0e-6,   1.0, 365.25, 0.017, 1.0),    # Earth, Sun
                                (0.08,     0.5, 400.0,  0.2,  1.0),     # brown dwarf
                                (0.08,     0.5, 400.0,  0.2,  0.5),
                                (1.0,      1.0, 12.0,   0.3,  1.0),     # q = 1
                                (0.7,      0.3, 3.0,    0.0,  0.9))     # q > 1
            K = Kof(m, M, P, e, s)
            @test msec_from_K(K, P, e, s, M) ≈ m rtol = 1e-12
        end
    end

    @testset "msec_from_K is differentiable in K" begin
        P, e, M = 400.0, 0.2, 0.5
        K = 3000.0                       # a brown dwarf around an M dwarf
        d = ForwardDiff.derivative(k -> msec_from_K(k, P, e, 1.0, M), K)
        h = 1e-3
        fd = (msec_from_K(K + h, P, e, 1.0, M) - msec_from_K(K - h, P, e, 1.0, M)) / 2h
        @test d ≈ fd rtol = 1e-8
    end
end

@testset "coupled rho_star / mass / inclination solve" begin
    # A q-companion with known truth. a/R* from Kepler III with M + M_p; cos i
    # from b; K from the exact mass function at that sin i.
    function truth(q; rho = 0.9, P = 5.0, b = 0.4, e = 0.1, w = 0.7, M = 1.0)
        a = rho_s_to_a_Rs(rho, P, q)
        ci = b * (1 + e * sin(w)) / (a * (1 - e^2))
        m = q * M
        f = m^3 * (1 - ci^2)^1.5 / (M + m)^2
        K = cbrt(f / (Nereus._F_M_FACTOR * P * (1 - e^2)^1.5))
        return (; rho, P, b, e, w, M, a, ci, K, q)
    end

    @testset "q = 0.1 companion: a/R*, mass and cos i recovered" begin
        t = truth(0.1)
        a0 = rho_s_to_a_Rs(t.rho, t.P)                  # the q = 0 value
        @test a0 < t.a                                  # the planet is not negligible
        a, q, ci = coupled_geometry(a0, t.b, t.e, sin(t.w),
                                    mass_function(t.K, t.P, t.e), t.M)
        @test a ≈ t.a rtol = 1e-13
        @test q ≈ 0.1 rtol = 1e-12
        @test ci ≈ t.ci rtol = 1e-12
    end

    @testset "converges for q up to ~0.5 and beyond" begin
        for q in (1e-6, 1e-3, 0.05, 0.1, 0.3, 0.5, 0.8, 1.5)
            t = truth(q; P = 8.0, b = 0.3)
            a0 = rho_s_to_a_Rs(t.rho, t.P)
            a, qs, _ = coupled_geometry(a0, t.b, t.e, sin(t.w),
                                        mass_function(t.K, t.P, t.e), t.M)
            @test qs ≈ q rtol = 1e-11
            @test a ≈ t.a rtol = 1e-12
        end
    end

    @testset "sampled a/R* (scale_a = false) keeps a/R*" begin
        t = truth(0.2)
        a, q, _ = coupled_geometry(t.a, t.b, t.e, sin(t.w),
                                   mass_function(t.K, t.P, t.e), t.M; scale_a = false)
        @test a == t.a
        @test q ≈ 0.2 rtol = 1e-12
    end

    @testset "ForwardDiff through the fixed point" begin
        t = truth(0.3)
        a0 = rho_s_to_a_Rs(t.rho, t.P)
        f(K) = last(coupled_geometry(a0, t.b, t.e, sin(t.w),
                                     mass_function(K, t.P, t.e), t.M)[[1, 2]])
        K = t.K
        d = ForwardDiff.derivative(f, K)
        h = K * 1e-6
        @test d ≈ (f(K + h) - f(K - h)) / 2h rtol = 1e-6
    end

    @testset "no mass information: the transit alone is unchanged" begin
        # the q = 0 default is the old function, bit for bit
        old(rho, P) = cbrt(Nereus.G_CGS * (rho * Nereus.RHO_SUN_CGS) * (P * 86400.0)^2 / (3π))
        for (rho, P) in ((1.0, 1.0), (0.532, 3.3), (2.2, 400.0))
            @test rho_s_to_a_Rs(rho, P) == old(rho, P)
        end
        # no K, no M_sec: q = 0
        a, q, _ = coupled_geometry(10.0, 0.3, 0.0, 0.0, 0.0, 1.0)
        @test q == 0.0 && a == 10.0
    end

    @testset "model level: RVPM planet with rho_s" begin
        t = truth(0.1; rho = 1.0, P = 6.0)
        n = 30
        tt = collect(range(0.0, 30.0; length = n)) .+ 2_450_000.0
        data = Data(; t_rv = tt[1:10], rv = zeros(10), rv_err = ones(10),
                      rv_inst = ones(Int, 10), t_phot = tt, flux = ones(n),
                      flux_err = fill(1e-3, n), phot_inst = ones(Int, n))
        mk(mode) = Params(; max_kplanet = 1, planet_modes = [mode],
                      instruments = InstrumentConfig(rv = ["HARPS"], pm = ["TESS"]),
                      data, parametrization = ParametrizationConfig(time = :Tc, use_rho_s = true),
                      M_s = t.M, R_s = 1.0, stability = :none)
        params = mk(RVPM)
        th = Theta{Float64}(params)
        ix = params.layout.name_to_idx
        # Written as the physical coordinates they are: under density =
        # :rho_circ, ρ★ is derived from the sampled ρ_circ and (e, ω), and
        # set_param! of it sets ρ_circ (src/density.jl); a raw slot write would not.
        for (nm, v) in ("P" => t.P, "K" => t.K, "Tc" => 2_450_003.0, "b" => t.b, "rr" => 0.1,
                        "sesinw" => sqrt(t.e) * sin(t.w), "secosw" => sqrt(t.e) * cos(t.w))
            Nereus.set_param!(th, "$(nm)_k1", v)
        end
        Nereus.set_param!(th, "rho_s", t.rho)
        a, q, ci = planet_geometry(th, 1, t.P)
        @test a ≈ t.a rtol = 1e-12
        @test q ≈ 0.1 rtol = 1e-11
        @test ci ≈ t.ci rtol = 1e-11
        @test Nereus.planet_a_Rs(th, 1, t.P) == a
        # the RV mass is the true mass at the transit's sin i, not the minimum
        @test Nereus.planet_M_sec(th, 1) ≈ 0.1 * t.M rtol = 1e-11

        # transit-only planet: no mass to insert
        params_pm = mk(PM_ONLY)
        thp = Theta{Float64}(params_pm)
        Nereus.set_param!(thp, "b_k1", t.b)
        Nereus.set_param!(thp, "rho_s", t.rho)
        a0, q0, _ = planet_geometry(thp, 1, t.P)
        @test q0 == 0.0
        @test a0 == rho_s_to_a_Rs(t.rho, t.P)

        # the derived table follows: same a/R*, the true mass, and rho_star
        # recovered from the transit once the planet's mass is taken out
        nms = sort!(collect(keys(ix)))
        vals = [th.values[ix[nm]] for nm in nms]
        ch = Chains(repeat(reshape(vals, 1, :, 1), 4, 1, 1), Symbol.(nms))
        d = compute_derived(ch, params; M_s = t.M, R_s = 1.0)[1].values
        @test only(unique(d["a_Rs"])) ≈ t.a rtol = 1e-12
        @test only(unique(d["mass_jup"])) ≈ 0.1 * t.M * Nereus.M_SUN_G / Nereus.M_JUP_G rtol = 1e-11
        @test only(unique(d["rho_star_transit"])) ≈ t.rho * Nereus.RHO_SUN_CGS rtol = 1e-12
        @test only(unique(d["a_au"])) ≈ t.a * Nereus.R_SUN_CM / Nereus.AU_CM rtol = 1e-12

    end
end
@testset "stability filter uses the exact masses and Kepler III with M + M_p" begin
    # an eccentric brown dwarf pair: the closed forms were ~9% off in mass
    Ps, Ks, es = [30.0, 90.0], [4000.0, 2500.0], [0.1, 0.05]
    M = 0.5
    order = zeros(Int, 2); mas = zeros(2); sma = zeros(2)
    Nereus.check_stability(Ps, Ks, es, M, :gladman, order, mas, sma)
    for i in 1:2
        @test mas[i] == msini(M, Ks[i], Ps[i], es[i])
        @test sma[i] == semimajor_axis_au(Ps[i], M, mas[i] * Nereus.M_JUP_G / Nereus.M_SUN_G)
        # the planet's mass is in the semi-major axis
        @test sma[i] > semimajor_axis_au(Ps[i], M)
    end
    # the allocating method agrees with the workspace method
    for mode in (:amd, :gladman)
        @test Nereus.check_stability(Ps, Ks, es, M, mode) ==
              Nereus.check_stability(Ps, Ks, es, M, mode, order, mas, sma)
    end
end

@testset "one set of solar constants: the rho_star route and the (M_s, R_s) route agree" begin
    # derived from M_SUN_G and R_SUN_CM, so the Sun is rho = 1 in solar units
    @test Nereus.RHO_SUN_CGS ≈ 1.4098 rtol = 1e-3
    @test Nereus.RHO_SUN_CGS == 3 * Nereus.M_SUN_G / (4π * Nereus.R_SUN_CM^3)
    for (M, R, P) in ((1.0, 1.0, 10.0), (0.532^1.5, 0.532^0.5, 3.3), (1.3, 1.6, 400.0))
        rho = M / R^3                                    # solar units
        a_rho = rho_s_to_a_Rs(rho, P)
        a_kepler = cbrt(Nereus.GM_SUN_CGS * M * (P * 86400.0)^2 / (4π^2)) /
                   (R * Nereus.R_SUN_CM)
        @test a_rho ≈ a_kepler rtol = 1e-14
    end
    @test rho_s_to_a_Rs(1.0, 10.0) ≈
          cbrt(Nereus.GM_SUN_CGS * (10 * 86400.0)^2 / (4π^2)) / Nereus.R_SUN_CM rtol = 1e-14
    # and with a companion, M_star + M_p on both routes
    q = 0.07
    @test rho_s_to_a_Rs(1.0, 10.0, q) ≈
          cbrt(Nereus.GM_SUN_CGS * (1 + q) * (10 * 86400.0)^2 / (4π^2)) / Nereus.R_SUN_CM rtol = 1e-14
end
