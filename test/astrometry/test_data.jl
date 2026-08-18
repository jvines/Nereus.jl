# Tests for RelAstromData and HGCAData containers.

@testset "Astrometry data structs" begin

    @testset "RelAstromData — basic construction" begin
        d = RelAstromData(;
            t       = [55000.0, 55365.0, 55730.0],
            ra_off  = [10.0, 12.0, 8.0],
            dec_off = [-5.0, -3.0, -7.0],
            ra_err  = [1.0, 1.0, 1.0],
            dec_err = [1.0, 1.0, 1.0],
        )
        @test n_relast(d) == 3
        @test all(d.corr .== 0.0)
        @test all(d.planet_idx .== 1)
    end

    @testset "RelAstromData — corr in [-1, 1]" begin
        @test_throws ArgumentError RelAstromData(
            t=[0.0], ra_off=[0.0], dec_off=[0.0],
            ra_err=[1.0], dec_err=[1.0], corr=[1.5])
    end

    @testset "RelAstromData — positive errors" begin
        @test_throws ArgumentError RelAstromData(
            t=[0.0], ra_off=[0.0], dec_off=[0.0],
            ra_err=[-1.0], dec_err=[1.0])
    end

    @testset "RelAstromData — multi-companion via planet_idx" begin
        d = RelAstromData(;
            t       = [55000.0, 55000.0, 55365.0],
            ra_off  = [10.0, -50.0, 12.0],
            dec_off = [-5.0, 30.0, -3.0],
            ra_err  = [1.0, 2.0, 1.0],
            dec_err = [1.0, 2.0, 1.0],
            planet_idx = [1, 2, 1],
        )
        @test d.planet_idx == [1, 2, 1]
    end

    @testset "HGCAData — basic construction with within-epoch corr" begin
        h = HGCAData(;
            epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra   = (5.0, 4.8, 4.9),
            pmdec  = (-3.0, -2.9, -2.95),
            sigma_pmra      = (0.5, 0.02, 0.03),
            sigma_pmdec     = (0.4, 0.02, 0.03),
            corr_pmra_pmdec = (0.2, 0.1, 0.3),
            plx    = 25.0, plx_err = 0.05, hip_id = 12345,
        )
        @test h.epochs[1] ≈ jyear_to_mjd(1991.25)
        @test h.plx == 25.0
        @test length(h.cov_ep) == 3
        @test size(h.cov_ep[1]) == (2, 2)
        # Check off-diagonal of Hip epoch: ρ × σ_RA × σ_Dec
        @test h.cov_ep[1][1, 2] ≈ 0.2 * 0.5 * 0.4
    end

    @testset "HGCAData — invalid σ rejected" begin
        @test_throws ArgumentError HGCAData(
            epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.0,0.0,0.0), pmdec=(0.0,0.0,0.0),
            sigma_pmra=(-1.0, 0.02, 0.03), sigma_pmdec=(0.4, 0.02, 0.03),
            plx=10.0, plx_err=0.1, hip_id=1)
    end

    @testset "HGCAData — invalid corr rejected" begin
        @test_throws ArgumentError HGCAData(
            epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra=(0.0,0.0,0.0), pmdec=(0.0,0.0,0.0),
            sigma_pmra=(0.5, 0.02, 0.03), sigma_pmdec=(0.4, 0.02, 0.03),
            corr_pmra_pmdec=(1.1, 0.0, 0.0),
            plx=10.0, plx_err=0.1, hip_id=1)
    end

    @testset "jyear_to_mjd round-trip" begin
        # Hipparcos reference epoch
        @test jyear_to_mjd(1991.25) ≈ 48348.5625 atol=1e-8
        # J2000.0 → MJD 51544.5
        @test jyear_to_mjd(2000.0) ≈ 51544.5
    end
end
