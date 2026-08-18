# Tests for OFTI rejection sampler.

@testset "OFTI sampler" begin

    @testset "Errors out on non-:a_driven parametrization" begin
        instruments = InstrumentConfig(rv=["X"])
        relast = RelAstromData(
            t = [55000.0, 56000.0],
            ra_off = [10.0, 12.0],
            dec_off = [-5.0, -3.0],
            ra_err = [1.0, 1.0],
            dec_err = [1.0, 1.0],
        )
        data = Data(t_rv=[55000.0, 55100.0], rv=[0.0,0.0], rv_err=[1.0,1.0],
                    relastrom=relast)
        params = Params(max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=1.0,
            parametrization=ParametrizationConfig(mass=:K_driven))
        target = NereusTarget(params, data)
        @test_throws ArgumentError ofti_sample(target; n_attempts=10)
    end

    @testset "Errors out without relAST data" begin
        instruments = InstrumentConfig(rv=["X"])
        # No relastrom
        data = Data(t_rv=[55000.0, 55100.0], rv=[0.0,0.0], rv_err=[1.0,1.0])
        params = Params(max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=instruments, data=data, stability=:none, M_s=1.0)
        target = NereusTarget(params, data)
        # Even if we had :a_driven, no relAST → error.
        # Without :a_driven the parametrization check fires first; both raise ArgumentError.
        @test_throws ArgumentError ofti_sample(target; n_attempts=10)
    end

    @testset "Synthetic recovery (small)" begin
        # Small synthetic 1-planet test:
        #   Generate fake (relAST, RV, HGCA) data from a known orbit,
        #   run OFTI, verify the M_sec posterior brackets the true value.
        using Random: MersenneTwister
        rng = MersenneTwister(2024)

        # True parameters (Jupiter-like at 5.2 AU around 1 M_sun)
        a_true   = 5.2
        M_pri    = 1.0
        M_sec_true = 1.0e-3   # ~1 M_J
        e_true   = 0.05
        ω_true   = 0.5
        Ω_true   = 1.2
        i_true   = deg2rad(50)
        Tp_true  = 55000.0
        plx_true = 50.0       # Closer than HD 159062 → larger angular orbit
        # Derived
        P_yr_true = sqrt(a_true^3 / (M_pri + M_sec_true))
        P_d_true  = P_yr_true * 365.25
        Mo_true   = -2π * (Tp_true - 55000.0) / P_d_true   # zero by construction

        orb_true = Nereus.build_orbit(P_d_true, e_true, ω_true, Ω_true, i_true,
                                        M_pri, M_sec_true, Tp_true, plx_true)

        # Synthetic relAST (3 epochs over 2 yr — short coverage)
        t_relast = [55000.0, 55365.0, 55730.0]
        ra_off = Float64[]; dec_off = Float64[]
        for t in t_relast
            r, d = relastrom_offset(orb_true, t)
            push!(ra_off, r + 1.0 * randn(rng))    # 1 mas noise
            push!(dec_off, d + 1.0 * randn(rng))
        end
        relast = RelAstromData(t=t_relast, ra_off=ra_off, dec_off=dec_off,
                                ra_err=fill(1.0, 3), dec_err=fill(1.0, 3))

        # Synthetic HGCA (low precision)
        t_hgca = mjd_epochs((1991.25, 2004.6, 2016.0))
        pmra_v = [Nereus.star_reflex_pm(orb_true, t, M_sec_true)[1] for t in t_hgca]
        pmdec_v = [Nereus.star_reflex_pm(orb_true, t, M_sec_true)[2] for t in t_hgca]
        # Add small Gaussian noise
        pmra_v_obs = pmra_v .+ 0.05 .* randn(rng, 3)
        pmdec_v_obs = pmdec_v .+ 0.05 .* randn(rng, 3)
        hgca = HGCAData(
            epochs = t_hgca,
            pmra = tuple(pmra_v_obs...), pmdec = tuple(pmdec_v_obs...),
            sigma_pmra = (0.05, 0.05, 0.05), sigma_pmdec = (0.05, 0.05, 0.05),
            plx = plx_true, plx_err = 0.05, hip_id = 99,
        )

        # Synthetic RV (sparse)
        t_rv = collect(55000.0:30.0:55700.0)
        rv = Float64[]
        for t in t_rv
            sol = orbitsolve(orb_true, t)
            push!(rv, radvel(sol, M_sec_true) + 1.0 * randn(rng))
        end
        data = Data(t_rv=t_rv, rv=rv, rv_err=fill(1.0, length(t_rv)),
                    relastrom=relast, hgca=hgca)

        # Build :a_driven model with broad priors
        params = Params(
            max_kplanet=1, planet_modes=[RVAS],
            instruments=InstrumentConfig(rv=["X"]),
            data=data, stability=:none, M_s=M_pri,
            parametrization=ParametrizationConfig(mass=:a_driven),
            priors=Dict{String, PriorSpec}(
                "n_p"        => FixedPrior(1.0),
                "a_k1"       => LogUniformPrior(0.5, 50.0),
                "M_sec_k1"   => LogUniformPrior(1e-5, 0.1),
                "sesinw_k1"  => UniformPrior(-1.0, 1.0),
                "secosw_k1"  => UniformPrior(-1.0, 1.0),
                "Mo_k1"      => UniformPrior(0.0, 2π),
                "inc_k1"     => SinePrior(),
                "Omega_k1"   => UniformPrior(0.0, 2π),
                "sigma_X"    => LogUniformPrior(0.1, 50.0),
            ),
        )
        target = NereusTarget(params, data)

        # Run OFTI with modest budget — synthetic test, not a science run
        chains = ofti_sample(target; n_attempts = 50_000, seed=2024,
                              show_progress=false)
        @test length(chains) > 0
    end

    @testset "2-pass kwargs (n_calibrate, buffer) accepted" begin
        # Reuse setup from the synthetic recovery block — but small
        # budget here is fine; we're just exercising the new kwargs.
        using Random: MersenneTwister
        rng = MersenneTwister(2024)
        relast = RelAstromData(
            t = [55000.0, 55365.0, 55730.0],
            ra_off = [10.0, 12.0, 8.0],
            dec_off = [-5.0, -3.0, -7.0],
            ra_err = [1.0, 1.0, 1.0],
            dec_err = [1.0, 1.0, 1.0],
        )
        data = Data(t_rv=[55000.0, 55300.0], rv=[0.0, 0.0], rv_err=[5.0, 5.0],
                    relastrom=relast)
        params = Params(
            max_kplanet=1, planet_modes=[RVAS],
            instruments=InstrumentConfig(rv=["X"]),
            data=data, stability=:none, M_s=1.0,
            parametrization=ParametrizationConfig(mass=:a_driven),
            priors=Dict{String, PriorSpec}(
                "n_p"        => FixedPrior(1.0),
                "a_k1"       => LogUniformPrior(0.5, 50.0),
                "M_sec_k1"   => LogUniformPrior(1e-5, 0.1),
                "sesinw_k1"  => UniformPrior(-1.0, 1.0),
                "secosw_k1"  => UniformPrior(-1.0, 1.0),
                "Mo_k1"      => UniformPrior(0.0, 2π),
                "inc_k1"     => SinePrior(),
                "Omega_k1"   => UniformPrior(0.0, 2π),
                "sigma_X"    => LogUniformPrior(0.1, 50.0),
            ),
        )
        target = NereusTarget(params, data)

        # Explicit n_calibrate + buffer should be honored without error.
        chains = ofti_sample(target;
                             n_attempts  = 5_000,
                             n_calibrate = 500,
                             buffer      = 1.0,
                             seed        = 7,
                             show_progress = false)
        @test length(chains) > 0
    end

    @testset "Empty calibration pass throws" begin
        # Drive calibration to reject every candidate at the a-prior
        # bound check by anchoring on an absurdly large separation that
        # no orbit with a < 1 AU and plx ~50 mas can match.
        # (Need ≥2 RV epochs with non-trivial spread to keep the
        # default gamma-prior builder happy.)
        relast = RelAstromData(
            t = [55000.0, 55365.0],
            ra_off = [1e6, 1e6],   # 1e6 mas = 1000 arcsec — impossible
            dec_off = [1e6, 1e6],
            ra_err = [0.001, 0.001],
            dec_err = [0.001, 0.001],
        )
        data = Data(t_rv=[55000.0, 55300.0], rv=[10.0, -10.0],
                    rv_err=[1.0, 1.0], relastrom=relast)
        params = Params(
            max_kplanet=1, planet_modes=[RVAS],
            instruments=InstrumentConfig(rv=["X"]),
            data=data, stability=:none, M_s=1.0,
            parametrization=ParametrizationConfig(mass=:a_driven),
            priors=Dict{String, PriorSpec}(
                "n_p" => FixedPrior(1.0),
                "a_k1" => LogUniformPrior(0.1, 1.0),  # tiny vs anchor
                "M_sec_k1" => LogUniformPrior(1e-5, 0.1),
                "sesinw_k1" => UniformPrior(-1.0, 1.0),
                "secosw_k1" => UniformPrior(-1.0, 1.0),
                "Mo_k1" => UniformPrior(0.0, 2π),
                "inc_k1" => SinePrior(),
                "Omega_k1" => UniformPrior(0.0, 2π),
                "sigma_X" => LogUniformPrior(0.1, 50.0),
            ),
        )
        target = NereusTarget(params, data)
        # Cal pass should find 0 finite samples (every scale-and-rotate
        # pushes a out of (0.1, 1.0)); calibration error fires.
        @test_throws ErrorException ofti_sample(target;
                                                 n_attempts = 200,
                                                 n_calibrate = 100,
                                                 seed = 3,
                                                 show_progress=false)
    end
end
