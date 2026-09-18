# Tests for the Hipparcos IAD + Gaia GOST scaffolding.
#
# These exercise the data containers, loader skeletons, and stub
# log-likelihoods. They do NOT validate numerical correctness against a
# reference implementation — that gates on the htof port (TODO).

@testset "IAD/GOST scaffolding" begin

    @testset "IADData — basic construction" begin
        d = IADData(;
            t            = [48000.0, 48100.0, 48200.0],
            abscissa     = [0.5, -0.3, 0.1],
            abscissa_err = [1.0, 1.2, 0.9],
            psi          = [0.0, π/4, π/2],
        )
        @test n_iad(d) == 3
        @test d.psi[2] ≈ π/4
        # Defaults for parallax_factor and pm_factor
        @test all(d.parallax_factor .== 0.0)
        @test all(d.pm_factor .== 0.0)
    end

    @testset "IADData — supplies parallax_factor and pm_factor" begin
        d = IADData(;
            t = [0.0, 1.0],
            abscissa = [0.0, 0.0],
            abscissa_err = [1.0, 1.0],
            psi = [0.0, 0.0],
            parallax_factor = [0.3, 0.4],
            pm_factor = [0.0, 0.5],
        )
        @test d.parallax_factor == [0.3, 0.4]
        @test d.pm_factor == [0.0, 0.5]
    end

    @testset "IADData — positive errors required" begin
        @test_throws ArgumentError IADData(
            t = [0.0], abscissa = [0.0],
            abscissa_err = [-1.0], psi = [0.0])
    end

    @testset "IADData — length mismatches caught" begin
        @test_throws ArgumentError IADData(
            t = [0.0, 1.0], abscissa = [0.0],
            abscissa_err = [1.0, 1.0], psi = [0.0, 0.0])
        @test_throws ArgumentError IADData(
            t = [0.0], abscissa = [0.0],
            abscissa_err = [1.0], psi = [0.0, 0.0])
        @test_throws ArgumentError IADData(
            t = [0.0], abscissa = [0.0],
            abscissa_err = [1.0], psi = [0.0],
            parallax_factor = [0.1, 0.2])
    end

    @testset "GOSTData — basic construction" begin
        d = GOSTData(;
            t = [57000.0, 57100.0],
            psi = [0.1, 0.2],
        )
        @test n_gost(d) == 2
        @test all(d.parallax_factor .== 0.0)
        @test all(d.along_scan_pos .== 0.0)
    end

    @testset "GOSTData — full keyword constructor" begin
        d = GOSTData(;
            t = [57000.0, 57100.0],
            psi = [0.1, 0.2],
            parallax_factor = [0.3, -0.4],
            along_scan_pos  = [0.0, 0.5],
        )
        @test d.parallax_factor == [0.3, -0.4]
        @test d.along_scan_pos == [0.0, 0.5]
    end

    @testset "GOSTData — length mismatch caught" begin
        @test_throws ArgumentError GOSTData(
            t = [0.0, 1.0], psi = [0.0])
    end

    @testset "along_scan_projection sign convention" begin
        # ψ = 0 → projection is purely Δdec
        @test Nereus.along_scan_projection(1.0, 2.0, 0.0) ≈ 2.0
        # ψ = π/2 → projection is purely Δra
        @test Nereus.along_scan_projection(1.0, 2.0, π/2) ≈ 1.0
        # ψ = π/4 → mixes both
        @test Nereus.along_scan_projection(1.0, 1.0, π/4) ≈
              sqrt(2.0) atol = 1e-12
    end

    @testset "load_hip_iad — round-trip on synthetic file" begin
        path = tempname() * ".iad"
        # Two transits, jyear epoch column, all 6 columns in default order.
        # 1991.25 → MJD 48348.5625
        open(path, "w") do io
            write(io, "# synthetic IAD\n")
            write(io, "# epoch[jyr] plx_factor pm_factor abscissa[mas] sigma[mas] psi[rad]\n")
            write(io, "1991.25 0.30 0.00 0.50 1.00 0.000\n")
            write(io, "1991.50 0.40 0.25 -0.20 1.20 0.785398163\n")
        end
        d = load_hip_iad(path)
        @test n_iad(d) == 2
        @test d.t[1] ≈ jyear_to_mjd(1991.25)
        @test d.t[2] ≈ jyear_to_mjd(1991.50)
        @test d.parallax_factor == [0.30, 0.40]
        @test d.pm_factor == [0.00, 0.25]
        @test d.abscissa == [0.50, -0.20]
        @test d.abscissa_err == [1.00, 1.20]
        @test d.psi[2] ≈ π/4 atol = 1e-7
        rm(path)
    end

    @testset "load_hip_iad — :mjd epoch_kind passes through" begin
        path = tempname() * ".iad"
        open(path, "w") do io
            write(io, "# epoch_mjd plx_fac pm_fac absc sigma psi\n")
            write(io, "48000.0 0.0 0.0 0.0 1.0 0.0\n")
        end
        d = load_hip_iad(path; epoch_kind = :mjd)
        @test d.t[1] == 48000.0
        rm(path)
    end

    @testset "load_gost — round-trip on synthetic file" begin
        path = tempname() * ".csv"
        # Synthetic GOST CSV with default column names.
        open(path, "w") do io
            write(io, "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB],",
                      "scanAngle[rad],parallaxFactorAlongScan,",
                      "Position[Heliocentric_Astrometric_Position]_AlongScan\n")
            write(io, "2457000.5,0.5,0.30,0.00\n")
            write(io, "2457100.5,1.0,-0.20,0.10\n")
        end
        d = load_gost(path)
        @test n_gost(d) == 2
        # BJD 2_457_000.5 → MJD 57000.0
        @test d.t[1] ≈ 57000.0
        @test d.t[2] ≈ 57100.0
        @test d.psi == [0.5, 1.0]
        @test d.parallax_factor == [0.30, -0.20]
        @test d.along_scan_pos == [0.00, 0.10]
        rm(path)
    end

    @testset "load_gost — missing along_scan_pos column tolerated" begin
        path = tempname() * ".csv"
        open(path, "w") do io
            write(io, "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB],",
                      "scanAngle[rad],parallaxFactorAlongScan\n")
            write(io, "2457000.5,0.5,0.30\n")
        end
        d = load_gost(path)
        @test n_gost(d) == 1
        @test d.along_scan_pos == [0.0]
        rm(path)
    end

    @testset "load_gost — :mjd time scale" begin
        path = tempname() * ".csv"
        open(path, "w") do io
            write(io, "t,scanAngle[rad],parallaxFactorAlongScan\n")
            write(io, "57000.0,0.5,0.30\n")
        end
        d = load_gost(path;
                      t_col = "t", t_kind = :mjd)
        @test d.t == [57000.0]
        rm(path)
    end

    # ---- Likelihood-stub plumbing ----
    # Build a minimal RVAS setup with both IAD and GOST attached and
    # verify astrom_log_likelihood is finite.

    function _rvas_setup_with_iad_gost()
        instruments = InstrumentConfig(rv = ["HARPS"], pm = String[])

        t_rv = collect(55000.0 : 30.0 : 55360.0)
        rv = zeros(length(t_rv))
        rv_err = fill(1.0, length(t_rv))

        relast = RelAstromData(
            t       = [55000.0, 55090.0, 55180.0, 55270.0],
            ra_off  = [10.0, 8.0, -5.0, -10.0],
            dec_off = [0.0, 6.0, 9.0, 0.0],
            ra_err  = [1.0, 1.0, 1.0, 1.0],
            dec_err = [1.0, 1.0, 1.0, 1.0],
        )
        hgca = HGCAData(
            epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra   = (5.0, 4.95, 4.9),
            pmdec  = (-3.0, -3.05, -3.1),
            sigma_pmra  = (0.2, 0.2, 0.2),
            sigma_pmdec = (0.2, 0.2, 0.2),
            plx = 25.0, plx_err = 0.05, hip_id = 1,
        )
        # 5 fake Hipparcos transits clustered around the mission epoch
        iad = IADData(
            t            = [48000.0, 48100.0, 48200.0, 48300.0, 48400.0],
            abscissa     = zeros(5),
            abscissa_err = fill(1.0, 5),
            psi          = [0.0, π/6, π/3, π/2, 2π/3],
        )
        # 4 fake Gaia GOST transits across the DR3 window
        gost = GOSTData(
            t   = [57000.0, 57100.0, 57200.0, 57300.0],
            psi = [0.1, 0.5, 1.0, 1.5],
        )

        data = Data(
            t_rv = t_rv, rv = rv, rv_err = rv_err,
            relastrom = relast, hgca = hgca,
            iad = iad, gost = gost,
        )

        params = Params(
            max_kplanet = 1,
            planet_modes = [RVAS],
            instruments = instruments,
            data = data,
            stability = :none,
            M_s = 1.0,
        )
        return data, params
    end

    @testset "Data plumbing — iad/gost fields propagated" begin
        data, _ = _rvas_setup_with_iad_gost()
        @test data.iad isa IADData
        @test data.gost isa GOSTData
        @test n_iad(data.iad) == 5
        @test n_gost(data.gost) == 4
        @test has_astrometry(data)
    end

    @testset "iad_log_likelihood is finite at sensible params" begin
        data, params = _rvas_setup_with_iad_gost()
        theta = Theta(params)
        set_param!(theta, "n_p", 1)
        set_param!(theta, "P_k1", 365.25)
        set_param!(theta, "K_k1", 50.0)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", deg2rad(60))
        set_param!(theta, "Omega_k1", deg2rad(30))
        set_param!(theta, "plx", 25.0)
        set_param!(theta, "sigma_HARPS", 5.0)

        ll_iad  = iad_log_likelihood(theta, data)
        ll_gost = gost_log_likelihood(theta, data)
        @test isfinite(ll_iad)
        @test isfinite(ll_gost)
        # GOST stub MUST return exactly zero so it doesn't perturb
        # the rest of the likelihood while structural correctness is
        # being validated.
        @test ll_gost == 0.0
    end

    @testset "astrom_log_likelihood includes IAD/GOST in the sum" begin
        data, params = _rvas_setup_with_iad_gost()
        theta = Theta(params)
        set_param!(theta, "n_p", 1)
        set_param!(theta, "P_k1", 365.25)
        set_param!(theta, "K_k1", 50.0)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", deg2rad(60))
        set_param!(theta, "Omega_k1", deg2rad(30))
        set_param!(theta, "plx", 25.0)
        set_param!(theta, "sigma_HARPS", 5.0)

        ll_total = astrom_log_likelihood(theta, data)
        ll_parts = relastrom_log_likelihood(theta, data) +
                   hgca_log_likelihood(theta, data) +
                   iad_log_likelihood(theta, data) +
                   gost_log_likelihood(theta, data)
        @test isfinite(ll_total)
        @test ll_total ≈ ll_parts
    end

    @testset "HGCA Mode A vs Mode B (with GOST)" begin
        # Mode A and Mode B should give different HGCA log-likelihoods
        # for a short-period orbit where the orbital signal varies
        # significantly across Gaia's mission window. For long-period
        # orbits the two should agree to small differences.
        instruments = InstrumentConfig(rv = ["X"], pm = String[])
        # Short-period orbit: P ~ 1.5 yr, well within Gaia's 3-yr window
        relast = RelAstromData(
            t = [55000.0], ra_off = [10.0], dec_off = [0.0],
            ra_err = [1.0], dec_err = [1.0],
        )
        hgca = HGCAData(
            epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra = (5.0, 4.95, 4.9), pmdec = (-3.0, -3.05, -3.1),
            sigma_pmra  = (0.2, 0.2, 0.2), sigma_pmdec = (0.2, 0.2, 0.2),
            plx = 25.0, plx_err = 0.05, hip_id = 1,
        )
        # GOST: 30 transits across Gaia DR3 window
        gost = GOSTData(
            t   = collect(range(56863.5, 58028.5, length = 30)),
            psi = mod.(2π .* (1:30) ./ 7, 2π),
            parallax_factor = sin.(2π .* (1:30) ./ 365.25),
        )
        # Build two parallel datasets, one without GOST (Mode A), one
        # with GOST (Mode B), everything else identical.
        data_A = Data(t_rv = [55000.0, 55100.0], rv = [0.0, 0.0],
                      rv_err = [1.0, 1.0],
                      relastrom = relast, hgca = hgca)
        data_B = Data(t_rv = [55000.0, 55100.0], rv = [0.0, 0.0],
                      rv_err = [1.0, 1.0],
                      relastrom = relast, hgca = hgca, gost = gost)
        params = Params(
            max_kplanet = 1, planet_modes = [RVAS],
            instruments = instruments,
            data = data_A, stability = :none, M_s = 1.0,
        )
        theta = Theta(params)
        # Short-period orbit
        set_param!(theta, "n_p", 1)
        set_param!(theta, "P_k1", 1.5 * 365.25)
        set_param!(theta, "K_k1", 200.0)
        set_param!(theta, "sesinw_k1", 0.0); set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)
        set_param!(theta, "inc_k1", deg2rad(60))
        set_param!(theta, "Omega_k1", deg2rad(30))
        set_param!(theta, "plx", 25.0)
        set_param!(theta, "sigma_X", 5.0)

        ll_A = hgca_log_likelihood(theta, data_A)
        ll_B = hgca_log_likelihood(theta, data_B)
        @test isfinite(ll_A) && isfinite(ll_B)
        # For a short-period orbit, Mode B differs from Mode A.
        # We don't pin a specific direction (depends on phase), only
        # that they're different beyond float noise.
        @test abs(ll_A - ll_B) > 1e-6
    end

    @testset "Empty IAD/GOST returns 0 cleanly" begin
        instruments = InstrumentConfig(rv = ["HARPS"], pm = String[])
        t_rv = [55000.0, 55100.0]
        data = Data(t_rv = t_rv, rv = [1.0, -1.0], rv_err = [1.0, 1.0])
        params = Params(
            max_kplanet = 1,
            planet_modes = [RV_ONLY],
            instruments = instruments,
            data = data,
            stability = :none,
            M_s = 1.0,
        )
        theta = Theta(params)
        @test iad_log_likelihood(theta, data) == 0.0
        @test gost_log_likelihood(theta, data) == 0.0
        @test g23h_log_likelihood(theta, data) == 0.0
    end

    # -------------------------------------------------------------------
    # G23H (Thompson+ 2026) — 5-PM joint absolute astrometry
    # -------------------------------------------------------------------
    @testset "G23HData construction + likelihood" begin
        # Helper: build a sane block-diagonal 10×10 covariance (5 × 2×2
        # within-epoch, σ_pmra = σ_pmdec = 0.1 mas/yr, ρ_within = 0.0).
        function _block_diag_cov()
            cov = zeros(10, 10)
            σ = 0.1
            for ie in 1:5
                cov[2*ie - 1, 2*ie - 1] = σ^2
                cov[2*ie    , 2*ie    ] = σ^2
            end
            return cov
        end

        @testset "Basic construction" begin
            cov = _block_diag_cov()
            d = G23HData(
                epochs = mjd_epochs((1991.25, 2004.6, 2015.0, 2015.5, 2016.0)),
                pmra   = (5.0, 4.95, 4.92, 4.94, 4.9),
                pmdec  = (-3.0, -3.05, -3.07, -3.06, -3.1),
                cov    = cov,
                plx    = 25.0, plx_err = 0.05, hip_id = 1,
            )
            @test n_g23h(d) == 5
            @test d.epochs[1] ≈ jyear_to_mjd(1991.25)
            @test d.epochs[5] ≈ jyear_to_mjd(2016.0)
            @test size(d.cov) == (10, 10)
        end

        @testset "Construction rejects non-PD cov" begin
            cov = -ones(10, 10)
            @test_throws ArgumentError G23HData(
                epochs = mjd_epochs((1991.25, 2004.6, 2015.0, 2015.5, 2016.0)),
                pmra   = (0.0, 0.0, 0.0, 0.0, 0.0),
                pmdec  = (0.0, 0.0, 0.0, 0.0, 0.0),
                cov    = cov,
                plx    = 25.0, plx_err = 0.05, hip_id = 1,
            )
        end

        @testset "Construction rejects wrong-size cov" begin
            cov = zeros(6, 6)
            for k in 1:6; cov[k, k] = 0.01; end
            @test_throws ArgumentError G23HData(
                epochs = mjd_epochs((1991.25, 2004.6, 2015.0, 2015.5, 2016.0)),
                pmra   = (0.0, 0.0, 0.0, 0.0, 0.0),
                pmdec  = (0.0, 0.0, 0.0, 0.0, 0.0),
                cov    = cov,
                plx    = 25.0, plx_err = 0.05, hip_id = 1,
            )
        end

        @testset "Likelihood finite at sensible params" begin
            cov = _block_diag_cov()
            g23h = G23HData(
                epochs = mjd_epochs((1991.25, 2004.6, 2015.0, 2015.5, 2016.0)),
                pmra   = (5.0, 4.95, 4.92, 4.94, 4.9),
                pmdec  = (-3.0, -3.05, -3.07, -3.06, -3.1),
                cov    = cov,
                plx    = 25.0, plx_err = 0.05, hip_id = 1,
            )
            data = Data(t_rv = [55000.0, 55100.0], rv = [0.0, 0.0],
                        rv_err = [1.0, 1.0], g23h = g23h)
            params = Params(
                max_kplanet = 1, planet_modes = [RVAS],
                instruments = InstrumentConfig(rv = ["X"]),
                data = data, stability = :none, M_s = 1.0,
            )
            theta = Theta(params)
            set_param!(theta, "n_p", 1)
            set_param!(theta, "P_k1", 365.25)
            set_param!(theta, "K_k1", 50.0)
            set_param!(theta, "sesinw_k1", 0.0); set_param!(theta, "secosw_k1", 0.0)
            set_param!(theta, "Mo_k1", 0.0)
            set_param!(theta, "inc_k1", deg2rad(60))
            set_param!(theta, "Omega_k1", deg2rad(30))
            set_param!(theta, "plx", 25.0)
            set_param!(theta, "sigma_X", 5.0)

            ll = g23h_log_likelihood(theta, data)
            @test isfinite(ll)
        end

        @testset "G23H disfavors wrong M_sec compared to truth" begin
            # Generate synthetic G23H from a known orbit
            using Random: MersenneTwister
            rng = MersenneTwister(42)
            M_pri = 1.0; M_sec = 0.05
            P_d = 5 * 365.25
            e = 0.1; ω = 0.4; Ω = 1.0; i = deg2rad(60)
            tp = 55000.0; plx = 30.0
            orb = Nereus.build_orbit(P_d, e, ω, Ω, i, M_pri, M_sec, tp, plx)
            epochs5 = mjd_epochs((1991.25, 2004.6, 2015.0, 2015.5, 2016.0))
            pmra_true = Tuple(zeros(5)); pmdec_true = Tuple(zeros(5))
            σ = 0.05
            pmra_obs = ntuple(5) do ie
                μra, _ = Nereus.star_reflex_pm(orb, epochs5[ie], M_sec)
                μra + σ * randn(rng)
            end
            pmdec_obs = ntuple(5) do ie
                _, μdec = Nereus.star_reflex_pm(orb, epochs5[ie], M_sec)
                μdec + σ * randn(rng)
            end
            cov = zeros(10, 10)
            for ie in 1:5
                cov[2*ie - 1, 2*ie - 1] = σ^2
                cov[2*ie    , 2*ie    ] = σ^2
            end
            g23h = G23HData(
                epochs = epochs5, pmra = pmra_obs, pmdec = pmdec_obs,
                cov = cov, plx = plx, plx_err = 0.05, hip_id = 1,
            )
            data = Data(t_rv = [55000.0, 55300.0], rv = [10.0, -10.0],
                        rv_err = [1.0, 1.0], g23h = g23h)
            params = Params(
                max_kplanet = 1, planet_modes = [RVAS],
                instruments = InstrumentConfig(rv = ["X"]),
                data = data, stability = :none, M_s = M_pri,
                parametrization = ParametrizationConfig(mass = :a_driven),
                priors = Dict{String, PriorSpec}(
                    "n_p"=>FixedPrior(1.0),
                    "a_k1"=>LogUniformPrior(0.5, 100.0),
                    "M_sec_k1"=>LogUniformPrior(0.001, 0.5),
                    "sesinw_k1"=>UniformPrior(-1.0, 1.0),
                    "secosw_k1"=>UniformPrior(-1.0, 1.0),
                    "Mo_k1"=>UniformPrior(0.0, 2π),
                    "inc_k1"=>SinePrior(),
                    "Omega_k1"=>UniformPrior(0.0, 2π),
                    "sigma_X"=>LogUniformPrior(0.1, 10.0),
                    "M_pri"=>FixedPrior(M_pri),
                ),
            )
            theta = Theta(params)
            a_true = ((M_pri + M_sec) * (P_d / 365.25)^2)^(1/3)
            set_param!(theta, "n_p", 1)
            set_param!(theta, "a_k1", a_true)
            set_param!(theta, "M_sec_k1", M_sec)
            set_param!(theta, "sesinw_k1", sqrt(e) * sin(ω))
            set_param!(theta, "secosw_k1", sqrt(e) * cos(ω))
            t_ref_data = (55000.0 + 55300.0) / 2
            set_param!(theta, "Mo_k1", 2π * (t_ref_data - tp) / P_d)
            set_param!(theta, "inc_k1", i)
            set_param!(theta, "Omega_k1", Ω)
            set_param!(theta, "plx", plx)
            set_param!(theta, "gamma_X", 0.0)
            set_param!(theta, "sigma_X", 1.0)

            ll_truth = g23h_log_likelihood(theta, data)
            @test isfinite(ll_truth)

            set_param!(theta, "M_sec_k1", 5 * M_sec)
            ll_5m = g23h_log_likelihood(theta, data)
            @test ll_truth > ll_5m + 5    # 5× M_sec → strongly disfavored
        end
    end

    # -------------------------------------------------------------------
    # Synthetic-recovery test for the marginalized IAD likelihood:
    # generate IAD from a known orbit (with realistic n_transits ~ 80),
    # confirm the likelihood discriminates against wrong M_sec and i.
    # -------------------------------------------------------------------
    @testset "iad_log_likelihood discriminates against wrong orbit" begin
        using Random: MersenneTwister
        rng = MersenneTwister(42)

        # Truth — P short enough vs Hipparcos's ~3.4-yr baseline that the
        # orbital signature is NOT degenerate with the linear PM
        # subspace. Long-period orbits (P >> baseline) genuinely look
        # like linear PM in IAD alone, which is why HGCA/Gaia exist.
        M_pri = 1.0; M_sec = 0.05
        P_d   = 5 * 365.25
        e     = 0.3; ω = 0.5; Ω_true = 1.2
        i_true = deg2rad(60); plx = 30.0
        tp = 49000.0
        orb = Nereus.build_orbit(P_d, e, ω, Ω_true, i_true,
                                   M_pri, M_sec, tp, plx)

        # 80 transits over ~3.4 yr Hipparcos baseline
        n   = 80
        t_ref_jd = jyear_to_mjd(1991.25)
        t_min = t_ref_jd - 365.25 * 1.4
        t_max = t_ref_jd + 365.25 * 2.0
        t_transits = collect(range(t_min, t_max, length = n))
        psi = 2π .* rand(rng, n)

        abscissa = Vector{Float64}(undef, n)
        abscissa_err = fill(1.5, n)
        for j in 1:n
            Δra, Δdec = Nereus.star_reflex_offset(orb, t_transits[j], M_sec)
            Δη = Nereus.along_scan_projection(Δra, Δdec, psi[j])
            abscissa[j] = Δη + abscissa_err[j] * randn(rng)
        end

        # Realistic-ish parallax factor: 1-yr-period oscillation projected
        # onto each scan, NOT collinear with sin ψ / cos ψ (otherwise the
        # 5-param design matrix is singular).
        year_phase = 2π .* (t_transits .- t_ref_jd) ./ 365.25
        parallax_factor = sin.(year_phase .- psi)
        pm_factor       = (t_transits .- t_ref_jd) ./ 365.25

        iad = IADData(t = t_transits,
                      abscissa = abscissa,
                      abscissa_err = abscissa_err,
                      psi = psi,
                      parallax_factor = parallax_factor,
                      pm_factor = pm_factor)

        data = Data(t_rv = [49000.0, 49300.0], rv = [10.0, -10.0],
                    rv_err = [1.0, 1.0], iad = iad)
        params = Params(
            max_kplanet = 1, planet_modes = [RVAS],
            instruments = InstrumentConfig(rv = ["X"]),
            data = data, stability = :none, M_s = M_pri,
            parametrization = ParametrizationConfig(mass = :a_driven),
            priors = Dict{String, PriorSpec}(
                "n_p"        => FixedPrior(1.0),
                "a_k1"       => LogUniformPrior(0.5, 100.0),
                "M_sec_k1"   => LogUniformPrior(0.001, 0.5),
                "sesinw_k1"  => UniformPrior(-1.0, 1.0),
                "secosw_k1"  => UniformPrior(-1.0, 1.0),
                "Mo_k1"      => UniformPrior(0.0, 2π),
                "inc_k1"     => SinePrior(),
                "Omega_k1"   => UniformPrior(0.0, 2π),
                "sigma_X"    => LogUniformPrior(0.1, 10.0),
                "M_pri"      => FixedPrior(M_pri),
            ),
        )
        theta = Theta(params)

        a_true = ((M_pri + M_sec) * (P_d / 365.25)^2)^(1/3)
        set_param!(theta, "n_p", 1)
        set_param!(theta, "a_k1", a_true)
        set_param!(theta, "M_sec_k1", M_sec)
        set_param!(theta, "sesinw_k1", sqrt(e) * sin(ω))
        set_param!(theta, "secosw_k1", sqrt(e) * cos(ω))
        t_ref_data = (49000.0 + 49300.0) / 2
        set_param!(theta, "Mo_k1", 2π * (t_ref_data - tp) / P_d)
        set_param!(theta, "inc_k1", i_true)
        set_param!(theta, "Omega_k1", Ω_true)
        set_param!(theta, "plx", plx)
        set_param!(theta, "gamma_X", 0.0)
        set_param!(theta, "sigma_X", 1.0)

        ll_truth = iad_log_likelihood(theta, data)
        @test isfinite(ll_truth)

        # Perturbations that should be penalized
        set_param!(theta, "M_sec_k1", 3 * M_sec)
        ll_3m = iad_log_likelihood(theta, data)
        set_param!(theta, "M_sec_k1", M_sec)
        set_param!(theta, "inc_k1", deg2rad(30))
        ll_i30 = iad_log_likelihood(theta, data)
        set_param!(theta, "inc_k1", i_true)
        set_param!(theta, "Omega_k1", Ω_true + 0.5)
        ll_Ω = iad_log_likelihood(theta, data)
        set_param!(theta, "Omega_k1", Ω_true)

        # Truth should beat all perturbations by a clearly-resolvable margin
        # (synthetic noise ~1.5 mas, signal ~14 mas peak → tens of
        # log-likelihood units of separation typical).
        @test ll_truth > ll_3m + 5      # 3× M_sec → strongly disfavored
        @test ll_truth > ll_i30 + 1     # i=30° vs 60° — moderate
        @test ll_truth > ll_Ω  + 1      # Ω+0.5 rad — moderate

        # GOST window-averaged PM smoke test (HGCA Mode B path).
        # For an orbit with M_sec=0 the window-averaged PM should be
        # exactly zero. With non-zero M_sec it should be non-zero and
        # finite.
        gost = GOSTData(
            t   = collect(range(57000.0, 58000.0, length = 30)),
            psi = mod.(2π .* (1:30) ./ 7, 2π),
            parallax_factor = sin.(2π .* (1:30) ./ 365.25),
        )
        orb_test = Nereus.build_orbit(P_d, e, ω, Ω_true, i_true,
                                        M_pri, M_sec, tp, plx)
        μra_b, μdec_b = gost_window_avg_pm(orb_test, gost, M_sec)
        @test isfinite(μra_b) && isfinite(μdec_b)
        orb_zero = Nereus.build_orbit(P_d, e, ω, Ω_true, i_true,
                                        M_pri, 0.0, tp, plx)
        μra_z, μdec_z = gost_window_avg_pm(orb_zero, gost, 0.0)
        @test abs(μra_z)  < 1e-10
        @test abs(μdec_z) < 1e-10

        # Below the 5-transit minimum the likelihood returns 0 cleanly.
        iad_short = IADData(t = t_transits[1:4],
                            abscissa = abscissa[1:4],
                            abscissa_err = abscissa_err[1:4],
                            psi = psi[1:4])
        data_short = Data(t_rv = [49000.0, 49300.0], rv = [10.0, -10.0],
                          rv_err = [1.0, 1.0], iad = iad_short)
        params_short = Params(
            max_kplanet = 1, planet_modes = [RVAS],
            instruments = InstrumentConfig(rv = ["X"]),
            data = data_short, stability = :none, M_s = M_pri,
            parametrization = ParametrizationConfig(mass = :a_driven),
            priors = Dict{String, PriorSpec}(
                "n_p"=>FixedPrior(1.0),
                "a_k1"=>LogUniformPrior(0.5, 100.0),
                "M_sec_k1"=>LogUniformPrior(0.001, 0.5),
                "sesinw_k1"=>UniformPrior(-1.0, 1.0),
                "secosw_k1"=>UniformPrior(-1.0, 1.0),
                "Mo_k1"=>UniformPrior(0.0, 2π),
                "inc_k1"=>SinePrior(),
                "Omega_k1"=>UniformPrior(0.0, 2π),
                "sigma_X"=>LogUniformPrior(0.1, 10.0),
                "M_pri"=>FixedPrior(M_pri),
            ),
        )
        theta_short = Theta(params_short)
        for nm in params_short.layout.unfrozen_names
            haskey(Dict(params.layout.name_to_idx), nm) &&
                set_param!(theta_short, nm, theta.values[params.layout.name_to_idx[nm]])
        end
        @test iad_log_likelihood(theta_short, data_short) == 0.0
    end
end
