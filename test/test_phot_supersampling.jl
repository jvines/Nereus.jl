# Finite-exposure supersampling (Kipping 2010) is per cadence.
#
# The number of sub-samples used to integrate the transit over an exposure has to
# come from THAT cadence's exposure. It used to come from the longest exposure in
# the dataset and was applied to every point, so a fit mixing a 10-min band with
# 2-min or 5-s photometry integrated the short cadences ten times over: slower,
# and a different model from the one the short cadences call for (instantaneous).
#
# Also tested: a photometry block can carry one exposure per cadence. A light curve
# stitched from TESS sectors flown at 1800, 600, 200 and 120 s needs that; a single
# value per block integrates some sectors over the wrong exposure.

using Test
using Nereus
using Nereus: _phot_n_super_point, _phot_transit_product, _parse_photometry_blocks,
               QuadLimbDark, sky_separation, transit_flux

@testset "Per-cadence supersampling" begin

    @testset "sub-samples from the cadence's own exposure" begin
        d(s) = s / 86_400.0
        # Up to 3 min a cadence is evaluated at its mid-time; above, ~1 sub-sample
        # per minute, at most 30.
        @test _phot_n_super_point(d(5.0))    == 1      # Swope-like
        @test _phot_n_super_point(d(120.0))  == 1      # TESS 2-min
        @test _phot_n_super_point(d(121.5))  == 1      # NGTS: was 3 under a 2-min limit
        @test _phot_n_super_point(d(180.0))  == 1      # at the 3-min limit
        @test _phot_n_super_point(d(181.0))  == 4
        @test _phot_n_super_point(d(200.0))  == 4      # TESS 200-s
        @test _phot_n_super_point(d(600.0))  == 10     # TESS 10-min
        @test _phot_n_super_point(d(1800.0)) == 30     # TESS 30-min
        @test _phot_n_super_point(d(7200.0)) == 30     # capped
        @test _phot_n_super_point(0.0) == 1
        @test _phot_n_super_point(NaN) == 1
    end

    @testset "a short cadence is not integrated because a long one is present" begin
        # One planet on a circular orbit, transit centred at t = 0.
        P, b, aRs, rr = 2.83, 0.75, 6.8, 0.116
        ld = QuadLimbDark([0.3, 0.25])
        Ps, es, ws, Tps = [P], [0.0], [π / 2], [0.0]   # ω = π/2, e = 0: transit at T_p
        bs, a_Rs, rrs = [b], [aRs], [rr]
        Tc_centers, T_dur_safe = [0.0], [0.2]
        transits, r_for_j = [true], [0]
        prod(t, texp, n_super) = _phot_transit_product(t, texp, ld, 1, transits,
            Ps, es, ws, Tps, bs, a_Rs, rrs, Tc_centers, T_dur_safe, r_for_j,
            nothing, n_super)
        # z as the product computes it, from the call's orbit constants
        orb = Nereus._sky_orbit(P, 0.0, π / 2, 0.0, b, aRs)
        instant(t) = transit_flux(ld, Nereus._sky_separation_signed(orb, t)[1], rr)

        t_in = 0.03                                  # in transit, near ingress
        n_dataset = 10                               # a 600-s band is in the fit
        # 120-s and 121.5-s cadences: instantaneous, whatever the dataset maximum is.
        @test prod(t_in, 120.0 / 86_400, n_dataset) == instant(t_in)
        @test prod(t_in, 121.5 / 86_400, n_dataset) == instant(t_in)
        # 600-s cadence: the mean over its own 10 sub-samples.
        texp = 600.0 / 86_400
        manual = sum(instant(t_in + ((2s - 10 - 1) / 20) * texp) for s in 1:10) / 10
        @test prod(t_in, texp, n_dataset) ≈ manual rtol = 1e-14
        @test prod(t_in, texp, n_dataset) != instant(t_in)
        # A dataset with no long exposures switches integration off entirely.
        @test prod(t_in, texp, 1) == instant(t_in)
    end

    @testset "exposure_time per cadence in a photometry block" begin
        t = collect(0.0:0.01:0.05)
        blk(ex) = Dict("values" => Dict("bjd" => t, "flux" => ones(length(t)),
                                        "flux_err" => fill(1e-3, length(t))),
                       "instrument" => "TESS", "exposure_time" => ex)
        ex = [1800.0, 1800.0, 600.0, 600.0, 120.0, 120.0]
        _, _, _, _, _, exp_d = _parse_photometry_blocks([blk(ex)])
        @test exp_d ≈ ex ./ 86_400.0
        # A scalar still applies to the whole block.
        _, _, _, _, _, exp_d = _parse_photometry_blocks([blk(600.0)])
        @test all(==(600.0 / 86_400.0), exp_d)
        # A vector of the wrong length is refused, not silently recycled.
        @test_throws ArgumentError _parse_photometry_blocks([blk([600.0, 600.0])])
    end
end
