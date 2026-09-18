# Tests for orbit projection: mass function, Kepler 3rd law, sky-plane
# offsets via PlanetOrbits.jl.

using PlanetOrbits

@testset "Astrometry orbit projection" begin

    @testset "mass_function — Jupiter" begin
        # Wright & Howard 2009 Eq. 7 reference: K=12.5 m/s, P=4332 d, e=0.05
        # → M_sec sin i ≈ 1.0 M_J.
        fM = mass_function(12.5, 4332.0, 0.05)
        # f(M) ≈ K^3 P (1-e²)^(3/2) / (2πG) ≈ 8.7e-10 M_sun
        @test fM > 8e-10 && fM < 1e-9
    end

    @testset "msec_from_K — Jupiter recovery" begin
        # Inputs: K=12.7 m/s, P=4332 d, e=0.05, sin i=1, M_pri=1
        m = msec_from_K(12.7, 4332.0, 0.05, 1.0, 1.0)
        # Expect ≈ M_J = 9.546e-4 M_sun (Jupiter)
        m_jup_in_msun = 1 / 1047.57
        @test m ≈ m_jup_in_msun rtol=0.05
    end

    @testset "msec_from_K — Earth recovery" begin
        # Earth induces K ≈ 0.0894 m/s on the Sun, P=365.25 d, e=0
        m = msec_from_K(0.0894, 365.25, 0.0, 1.0, 1.0)
        m_earth_in_msun = 3.003e-6
        @test m ≈ m_earth_in_msun rtol=0.05
    end

    @testset "msec_from_K — sin i scaling" begin
        # With sin i = 0.5, the inferred M_sec doubles from edge-on
        m_edge = msec_from_K(12.7, 4332.0, 0.05, 1.0, 1.0)
        m_60   = msec_from_K(12.7, 4332.0, 0.05, sind(60), 1.0)
        # Approximately m × (1 / sin i)
        @test m_60 / m_edge ≈ 1 / sind(60) rtol=0.02
    end

    @testset "a_from_P — Earth orbit" begin
        # Sun-Earth: P=365.25 d, M_total=1 M_sun → a = 1 AU
        @test a_from_P(365.25, 1.0) ≈ 1.0 atol=1e-3
    end

    @testset "a_from_P — Jupiter" begin
        # P=4332 d, M_total ≈ 1.001 M_sun → a ≈ 5.20 AU
        @test a_from_P(4332.0, 1.001) ≈ 5.205 rtol=0.005
    end

    @testset "build_orbit — Jupiter face-on" begin
        orb = Nereus.build_orbit(4332.0, 0.05, 0.0, 0.0,
                                   deg2rad(1.0),  # nearly face-on
                                   1.0, 0.001, 0.0, 100.0)
        @test isa(orb, PlanetOrbits.Visual)
        @test orb.parent.a ≈ 5.205 rtol=0.005
        @test orb.parent.e ≈ 0.05
        @test orb.plx == 100.0
    end

    @testset "relastrom_offset — circular face-on" begin
        # Face-on circular: ρ ≈ a × plx; θ rotates uniformly with M.
        # i = 1° (almost face-on), e=0, M_total=1, a=1 AU, plx=100 mas → ρ ≈ 100 mas.
        orb = Nereus.build_orbit(365.25, 0.0, 0.0, 0.0,
                                   deg2rad(1.0), 1.0, 0.001, 0.0, 100.0)
        Δra1, Δdec1 = relastrom_offset(orb, 0.0)        # at periastron
        Δra2, Δdec2 = relastrom_offset(orb, 365.25/4)   # quarter orbit
        ρ1 = hypot(Δra1, Δdec1)
        ρ2 = hypot(Δra2, Δdec2)
        @test ρ1 ≈ 100.0 rtol=0.01
        @test ρ2 ≈ 100.0 rtol=0.01
    end

    @testset "star_reflex_offset — sign and scale" begin
        # Star reflex points opposite the planet, scaled by M_sec / M_total.
        orb = Nereus.build_orbit(365.25, 0.0, 0.0, 0.0,
                                   deg2rad(1.0), 1.0, 0.001, 0.0, 100.0)
        Δra_p,  Δdec_p  = relastrom_offset(orb, 100.0)
        Δra_s,  Δdec_s  = star_reflex_offset(orb, 100.0, 0.001)
        # Ratio: M_sec / M_total = 0.001 / 1.001 ≈ 9.99e-4 (PlanetOrbits
        # uses M_planet / M_central, see test in earlier session — ratio
        # was -1e-3 exactly).
        @test Δra_s / Δra_p ≈ -0.001 atol=1e-6
    end

    @testset "star_reflex_pm — orientation" begin
        # PM is the time derivative of the offset; for a circular orbit
        # the magnitude should be roughly 2π × ρ_max / P_yr (mas/yr).
        orb = Nereus.build_orbit(365.25, 0.0, 0.0, 0.0,
                                   deg2rad(1.0), 1.0, 0.001, 0.0, 100.0)
        μra, μdec = star_reflex_pm(orb, 0.0, 0.001)
        ρ_max_yr = 0.1  # star reflex ρ ≈ 100 × 0.001 = 0.1 mas
        expected_mag = 2π * ρ_max_yr  # mas/yr
        # Order of magnitude check:
        @test 0 < hypot(μra, μdec) < 10 * expected_mag
    end
end
