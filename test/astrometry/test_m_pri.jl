# Tests for sampled M_pri (host stellar mass) in astrometry models.

@testset "M_pri sampling" begin

    function _make_setup(; sample_M_pri::Bool=false, M_s_val=1.0)
        instruments = InstrumentConfig(rv=["X"])
        hgca = HGCAData(
            epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.,0.,0.), pmdec=(0.,0.,0.),
            sigma_pmra=(0.5,0.02,0.03), sigma_pmdec=(0.4,0.02,0.03),
            plx=10.0, plx_err=0.1, hip_id=1)
        data = Data(t_rv=[55000.0, 55100.0], rv=[0.0,0.0], rv_err=[1.0,1.0], hgca=hgca)
        priors = sample_M_pri ?
            Dict("M_pri" => NormalPrior(M_s_val, 0.05, 0.5, 1.5)) :
            Dict{String, PriorSpec}()
        params = Params(
            max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=M_s_val,
            priors=priors,
        )
        return data, params
    end

    @testset "M_pri slot exists when AS planets present" begin
        _, params = _make_setup()
        @test "M_pri" in params.layout.names
        @test params.layout.systemic.m_pri > 0
    end

    @testset "M_pri default is FixedPrior(M_s)" begin
        _, params = _make_setup(M_s_val=0.85)
        # M_pri should be in frozen (default = Fixed)
        m_pri_idx = params.layout.name_to_idx["M_pri"]
        @test m_pri_idx in params.layout.frozen_idx
        # The frozen value should equal M_s
        i = findfirst(==(m_pri_idx), params.layout.frozen_idx)
        @test params.layout.frozen_values[i] ≈ 0.85
    end

    @testset "M_pri sampled when user provides Gaussian prior" begin
        _, params = _make_setup(sample_M_pri=true, M_s_val=0.85)
        @test "M_pri" in params.layout.unfrozen_names
    end

    @testset "astrom_M_pri reads sampled value" begin
        _, params = _make_setup(sample_M_pri=true, M_s_val=0.85)
        theta = Theta(params)
        set_param!(theta, "M_pri", 0.93)
        @test astrom_M_pri(theta) ≈ 0.93
    end

    @testset "astrom_M_pri returns config.M_s when slot is fixed" begin
        _, params = _make_setup(M_s_val=0.85)
        theta = Theta(params)
        # Even with M_pri frozen, astrom_M_pri should return its frozen value (0.85)
        @test astrom_M_pri(theta) ≈ 0.85
    end

    @testset "M_pri propagates to mass derivation under :K_driven" begin
        # In K_driven, M_sec is derived from K, P, e, sin i, M_pri.
        # Doubling M_pri at fixed K, P, e, i → M_sec changes (mass function nonlinear)
        instruments = InstrumentConfig(rv=["X"])
        hgca = HGCAData(epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.,0.,0.), pmdec=(0.,0.,0.),
            sigma_pmra=(0.5,0.02,0.03), sigma_pmdec=(0.4,0.02,0.03),
            plx=10.0, plx_err=0.1, hip_id=1)
        data = Data(t_rv=[55000.0,55100.0], rv=[0.0,0.0], rv_err=[1.0,1.0], hgca=hgca)
        params = Params(max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=1.0,
            priors=Dict("M_pri" => NormalPrior(1.0, 0.05, 0.5, 1.5)))
        theta = Theta(params)
        set_param!(theta, "n_p", 1)
        set_param!(theta, "P_k1", 4332.0)
        set_param!(theta, "K_k1", 12.7)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", π/2)
        set_param!(theta, "Omega_k1", 0.0)
        set_param!(theta, "plx", 10.0)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)

        set_param!(theta, "M_pri", 1.0)
        m1 = planet_M_sec(theta, 1)
        set_param!(theta, "M_pri", 2.0)
        m2 = planet_M_sec(theta, 1)
        # M_sec must change with M_pri (nonzero derivative)
        @test m2 > m1   # heavier host → heavier companion at same K (since K ∝ M_sec/M_total^(2/3))
    end
end
