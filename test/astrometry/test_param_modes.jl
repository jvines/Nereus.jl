# Tests for the three mass parametrizations:
#   :K_driven      — sample (P, K), derive M_sec via mass function
#   :M_sec_driven  — sample (P, M_sec), derive K
#   :a_driven      — sample (a, M_sec), derive (P, K)
#
# Round-trip: feed each mode self-consistent inputs and verify
# planet_P, planet_K, planet_M_sec, planet_a return matching values
# regardless of the parametrization in use.

@testset "Parametrization modes (K, M_sec, a)" begin

    function _make_setup(mass_mode::Symbol)
        instruments = InstrumentConfig(rv=["X"])
        t_rv = [55000.0, 55100.0]
        hgca = HGCAData(
            epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.,0.,0.), pmdec=(0.,0.,0.),
            sigma_pmra=(0.5,0.02,0.03), sigma_pmdec=(0.4,0.02,0.03),
            plx=10.0, plx_err=0.1, hip_id=1)
        data = Data(t_rv=t_rv, rv=[0.0,0.0], rv_err=[1.0,1.0], hgca=hgca)
        params = Params(
            max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=1.0,
            parametrization = ParametrizationConfig(mass=mass_mode),
        )
        return data, params
    end

    @testset ":K_driven (default)" begin
        _, params = _make_setup(:K_driven)
        @test "P_k1" in params.layout.names
        @test "K_k1" in params.layout.names
        @test !("a_k1" in params.layout.names)
        @test !("M_sec_k1" in params.layout.names)
    end

    @testset ":M_sec_driven" begin
        _, params = _make_setup(:M_sec_driven)
        @test "P_k1" in params.layout.names
        @test "M_sec_k1" in params.layout.names
        @test !("K_k1" in params.layout.names)
        @test !("a_k1" in params.layout.names)
    end

    @testset ":a_driven" begin
        _, params = _make_setup(:a_driven)
        @test "a_k1" in params.layout.names
        @test "M_sec_k1" in params.layout.names
        @test !("P_k1" in params.layout.names)
        @test !("K_k1" in params.layout.names)
    end

    @testset "Roundtrip: K_driven derives the same M_sec/a as direct" begin
        # Set up a Jupiter-like orbit:
        #   P = 4332 d, K = 12.7 m/s, e = 0.05, sin i = 1, M_pri = 1
        # Expect M_sec ≈ 9.5e-4 M_sun, a ≈ 5.2 AU
        _, params = _make_setup(:K_driven)
        theta = Theta(params)
        set_param!(theta, "n_p", 1.0)
        set_param!(theta, "P_k1", 4332.0)
        set_param!(theta, "K_k1", 12.7)
        set_param!(theta, "sesinw_k1", sqrt(0.05))
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", π/2)   # edge-on
        set_param!(theta, "Omega_k1", 0.0)
        set_param!(theta, "plx", 10.0)
        set_param!(theta, "M_pri", 1.0)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)

        @test planet_P(theta, 1) ≈ 4332.0
        @test planet_K(theta, 1) ≈ 12.7
        m_sec = planet_M_sec(theta, 1)
        @test m_sec > 9e-4 && m_sec < 1.05e-3   # roughly Jupiter
        a_au = planet_a(theta, 1)
        @test a_au > 5.0 && a_au < 5.5           # ~5.2 AU
    end

    @testset "Roundtrip: M_sec_driven and a_driven give same orbit when fed equivalent inputs" begin
        # M_sec_driven: feed M_sec, P, derive K
        _, params_m = _make_setup(:M_sec_driven)
        _, params_a = _make_setup(:a_driven)
        # Common physical orbit
        M_pri  = 1.0
        M_sec  = 9.55e-4   # Jupiter
        P_d    = 4332.0
        e      = 0.05
        ω      = 0.0
        Mo     = 0.0
        inc    = π/2
        Ω      = 0.0
        plx    = 10.0
        # a from Kepler 3rd
        a_au   = (M_pri + M_sec)^(1/3) * (P_d/365.25)^(2/3)

        function _set_common(θ, name)
            set_param!(θ, "n_p", 1.0)
            set_param!(θ, "sesinw_k1", sqrt(e) * sin(ω))
            set_param!(θ, "secosw_k1", sqrt(e) * cos(ω))
            set_param!(θ, "Mo_k1", Mo)
            set_param!(θ, "inc_k1", inc)
            set_param!(θ, "Omega_k1", Ω)
            set_param!(θ, "plx", plx)
            set_param!(θ, "M_pri", M_pri)
            set_param!(θ, "gamma_X", 0.0)
            set_param!(θ, "sigma_X", 1.0)
        end

        θ_m = Theta(params_m)
        _set_common(θ_m, :M_sec_driven)
        set_param!(θ_m, "P_k1", P_d)
        set_param!(θ_m, "M_sec_k1", M_sec)

        θ_a = Theta(params_a)
        _set_common(θ_a, :a_driven)
        set_param!(θ_a, "a_k1", a_au)
        set_param!(θ_a, "M_sec_k1", M_sec)

        # Both should report the same physical orbit
        @test planet_P(θ_m, 1) ≈ planet_P(θ_a, 1) rtol=1e-6
        @test planet_K(θ_m, 1) ≈ planet_K(θ_a, 1) rtol=1e-6
        @test planet_M_sec(θ_m, 1) ≈ planet_M_sec(θ_a, 1) rtol=1e-12
        @test planet_a(θ_m, 1) ≈ planet_a(θ_a, 1) rtol=1e-6
    end

    @testset "Invalid mass mode rejected" begin
        @test_throws ArgumentError ParametrizationConfig(mass=:bogus)
    end
end

# =====================================================================
# RVPMASBlock: inclination derived from transit impact parameter b.
# =====================================================================
#
# Geometric relation:
#   cos i = b · (1 + e sin ω) / ((a/R*) · (1 − e²))
#
# Sampled slots are (P, K, e1, e2, t, b, r, Omega) — 8 per planet.
# Compared to the pre-fix code (P, K, e1, e2, t, b, r, inc, Omega) =
# 9, this is exactly one fewer.
@testset "RVPMASBlock: inclination derived from b" begin

    function _make_rvpmas_setup(; use_rho_s::Bool=false, M_s::Float64=1.0,
                                  R_s::Float64=1.0)
        instruments = InstrumentConfig(rv=["X"], pm=["Y"])
        t_rv = [55000.0, 55100.0]
        t_phot = [55050.0, 55051.0]
        hgca = HGCAData(
            epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.,0.,0.), pmdec=(0.,0.,0.),
            sigma_pmra=(0.5,0.02,0.03), sigma_pmdec=(0.4,0.02,0.03),
            plx=10.0, plx_err=0.1, hip_id=1)
        data = Data(
            t_rv=t_rv, rv=[0.0,0.0], rv_err=[1.0,1.0],
            t_phot=t_phot, flux=[1.0,1.0], flux_err=[0.001,0.001],
            rv_inst=[1,1], phot_inst=[1,1],
            hgca=hgca,
        )
        params = Params(
            max_kplanet=1, planet_modes=[RVPMAS],
            instruments=instruments, data=data, stability=:none,
            M_s=M_s, R_s=R_s,
            parametrization = ParametrizationConfig(use_rho_s=use_rho_s),
        )
        return data, params
    end

    @testset "Layout: inc_kN dropped, Omega_kN kept" begin
        _, params = _make_rvpmas_setup()
        @test "b_k1" in params.layout.names
        @test "rr_k1" in params.layout.names
        @test "Omega_k1" in params.layout.names
        @test !("inc_k1" in params.layout.names)
        @test !("inc_k1" in params.layout.unfrozen_names)
    end

    @testset "Block fields: no `inc`" begin
        _, params = _make_rvpmas_setup()
        block = params.layout.planet_blocks[1]
        @test block isa RVPMASBlock
        # 8 slots (P, K, e1, e2, t, b, r, Omega), no `inc`.
        @test fieldnames(typeof(block)) ==
            (:P, :K, :e1, :e2, :t, :b, :r, :Omega)
    end

    @testset "Slot count drops by exactly 1 vs. pre-fix layout" begin
        # RVPMAS planet block now has 8 fields (P, K, e1, e2, t, b, r,
        # Omega), one fewer than the pre-fix 9 (which included `inc`).
        # The corresponding planet adds the same number of names to the
        # global layout. Verify by counting names with the "_k1" suffix.
        _, params = _make_rvpmas_setup()
        per_planet = count(n -> endswith(n, "_k1"), params.layout.names)
        # P, K, sesinw, secosw, Mo, b, rr, Omega → 8 (no inc).
        @test per_planet == 8

        # Pre-fix would have been 9 → confirm it's exactly 8 now.
        @test !("inc_k1" in params.layout.names)

        # The block carries 8 slots, matching the per-planet name count.
        block = params.layout.planet_blocks[1]
        @test length(fieldnames(typeof(block))) == 8
    end

    @testset "Derived inclination matches geometric relation" begin
        # Setup: P = 10 d, M_s = R_s = 1 → a/R* via Kepler 3
        _, params = _make_rvpmas_setup(use_rho_s=false, M_s=1.0, R_s=1.0)
        theta = Theta(params)
        set_param!(theta, "n_p", 1.0)
        set_param!(theta, "P_k1", 10.0)
        set_param!(theta, "K_k1", 1.0)
        # Circular orbit (e=0, ω=0): cos i = b / (a/R*).
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "b_k1", 0.5)
        set_param!(theta, "rr_k1", 0.1)
        set_param!(theta, "Omega_k1", 0.0)
        set_param!(theta, "plx", 10.0)
        set_param!(theta, "M_pri", 1.0)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)
        set_param!(theta, "offset_Y", 1.0)
        set_param!(theta, "jitter_Y", 0.001)
        set_param!(theta, "q1_Y", 0.5)
        set_param!(theta, "q2_Y", 0.5)

        # Compute expected a/R* from M_s=1, R_s=1, P=10 d.
        P_s = 10.0 * 86400.0
        GM = 1.3271244e26
        a_cm = cbrt(GM * P_s^2 / (4 * π^2))
        a_Rs = a_cm / 6.9570e10
        expected_i = acos(0.5 / a_Rs)

        @test planet_inc(theta, 1) ≈ expected_i rtol=1e-10
    end

    @testset "Derived inclination respects e and ω" begin
        # cos i = b · (1 + e sin ω) / ((a/R*) · (1 − e²))
        _, params = _make_rvpmas_setup(use_rho_s=false, M_s=1.0, R_s=1.0)
        theta = Theta(params)

        e = 0.3
        ω = π/4
        b = 0.4

        set_param!(theta, "n_p", 1.0)
        set_param!(theta, "P_k1", 10.0)
        set_param!(theta, "K_k1", 1.0)
        set_param!(theta, "sesinw_k1", sqrt(e) * sin(ω))
        set_param!(theta, "secosw_k1", sqrt(e) * cos(ω))
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "b_k1", b)
        set_param!(theta, "rr_k1", 0.1)
        set_param!(theta, "Omega_k1", 0.0)
        set_param!(theta, "plx", 10.0)
        set_param!(theta, "M_pri", 1.0)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)
        set_param!(theta, "offset_Y", 1.0)
        set_param!(theta, "jitter_Y", 0.001)
        set_param!(theta, "q1_Y", 0.5)
        set_param!(theta, "q2_Y", 0.5)

        P_s = 10.0 * 86400.0
        GM = 1.3271244e26
        a_cm = cbrt(GM * P_s^2 / (4 * π^2))
        a_Rs = a_cm / 6.9570e10
        expected_cos_i = b * (1 + e * sin(ω)) / (a_Rs * (1 - e^2))
        expected_i = acos(expected_cos_i)

        @test planet_inc(theta, 1) ≈ expected_i rtol=1e-10
    end

    @testset "Edge clamp: cos i argument outside [-1, 1] doesn't error" begin
        # Configure a/R* small enough that b · (1 + e sin ω) / (a_Rs · (1-e²))
        # would exceed 1 if not clamped. With M_s=1, R_s=1, P=10 d we have
        # a/R* ≈ 19.5, so we need a different config — push M_s tiny and
        # R_s huge to shrink a/R* (a/R* ∝ M_s^{1/3}/R_s).
        _, params = _make_rvpmas_setup(use_rho_s=false, M_s=0.001, R_s=10.0)
        theta = Theta(params)
        set_param!(theta, "n_p", 1.0)
        set_param!(theta, "P_k1", 1.0)        # 1-day orbit
        set_param!(theta, "K_k1", 1.0)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "b_k1", 0.99)       # b near unity
        set_param!(theta, "rr_k1", 0.1)
        set_param!(theta, "Omega_k1", 0.0)
        set_param!(theta, "plx", 10.0)
        set_param!(theta, "M_pri", 1.0)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)
        set_param!(theta, "offset_Y", 1.0)
        set_param!(theta, "jitter_Y", 0.001)
        set_param!(theta, "q1_Y", 0.5)
        set_param!(theta, "q2_Y", 0.5)

        i = planet_inc(theta, 1)   # must not throw
        @test isfinite(i)
        @test 0 <= i <= π
    end

    @testset "Default priors: no inc_kN prior emitted for RVPMAS" begin
        # The default prior generator should NOT include `inc_k1` for an
        # RVPMAS planet — its slot doesn't exist anymore.
        _, params = _make_rvpmas_setup()
        @test !("inc_k1" in params.layout.unfrozen_names)
        @test "Omega_k1" in params.layout.unfrozen_names
    end

    @testset "RVAS still samples inc directly" begin
        # Make sure the fix doesn't accidentally affect the RVAS-only
        # (no PM) configuration.
        instruments = InstrumentConfig(rv=["X"])
        t_rv = [55000.0, 55100.0]
        hgca = HGCAData(
            epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.,0.,0.), pmdec=(0.,0.,0.),
            sigma_pmra=(0.5,0.02,0.03), sigma_pmdec=(0.4,0.02,0.03),
            plx=10.0, plx_err=0.1, hip_id=1)
        data = Data(t_rv=t_rv, rv=[0.0,0.0], rv_err=[1.0,1.0], hgca=hgca)
        params = Params(
            max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=1.0,
        )
        @test "inc_k1" in params.layout.names
        @test "inc_k1" in params.layout.unfrozen_names
        @test params.layout.planet_blocks[1] isa RVASBlock
    end
end
