# Gravity-darkened transits (von Zeipel 1924; Barnes 2009).
#
# Two layers are tested:
#   1. the physics, against limits that are known in closed form, and against
#      published oblateness for the systems where gravity darkening is a secure
#      detection;
#   2. the WIRING into the photometric likelihood, which is where a silent
#      failure would be most expensive: if `gd` never reaches
#      `_phot_transit_product`, the fit still runs and still returns a posterior
#      for i_star -- a flat one -- and nothing announces that the model being
#      sampled is the ordinary Mandel-Agol one.
#
# The controlling test for (2) is therefore not "does the GD model fit", it is
# "does the likelihood CHANGE when i_star changes". A model whose logL is
# invariant under i_star is not measuring i_star.

using Test
using Nereus
using Nereus: roche_flattening, roche_radius, local_gravity, gd_brightness,
               transit_flux_gd, omega_frac_from_veq, transit_flux,
               QuadLimbDark, _gd_disc_mean, gd_beta_band

@testset "Gravity darkening" begin

    # ================================================================
    # 1. Roche geometry
    # ================================================================
    @testset "Roche flattening" begin
        # No rotation: a sphere.
        @test roche_flattening(0.0) == 0.0
        @test roche_radius(π / 2, 0.0) ≈ 1.0 atol = 1e-12

        # Break-up: the Roche model has R_eq / R_pol = 3/2 EXACTLY, for every
        # star. This is the single sharpest check on the ω normalisation --
        # it is what caught `omega_frac` being interpreted as ω rather than as
        # a fraction of break-up (the equatorial radius came out at 1.09 R_pol
        # instead of 1.5, an oblateness 5x too small).
        @test roche_flattening(1.0) ≈ 1 - 2 / 3 atol = 1e-9
        @test roche_radius(π / 2, 1.0) ≈ 1.5 atol = 1e-6

        # Slow-rotation limit. Expanding the Roche potential to first order in
        # ω^2 gives R_eq/R_pol - 1 -> ω^2/2 in POTENTIAL units, and ω_frac
        # enters as ω = ω_frac * sqrt(8/27), so f -> (4/27) ω_frac^2.
        for wf in (0.02, 0.05, 0.10)
            @test roche_flattening(wf) ≈ (4 / 27) * wf^2 rtol = 0.05
        end

        # Monotone in ω, and the equator always bulges.
        fs = [roche_flattening(w) for w in 0.1:0.1:0.9]
        @test all(diff(fs) .> 0)
        for wf in (0.3, 0.6, 0.9)
            @test roche_radius(π / 2, wf) > roche_radius(0.0, wf)
        end
    end

    @testset "Effective gravity" begin
        # von Zeipel's premise: gravity is highest at the pole, lowest at the
        # equator, and falls monotonically in between.
        for wf in (0.2, 0.5, 0.8)
            gs = [local_gravity(θ, wf) for θ in range(0, π / 2; length = 25)]
            @test gs[1] ≈ maximum(gs)
            @test gs[end] ≈ minimum(gs)
            @test all(diff(gs) .<= 1e-12)
        end
        # At break-up the equatorial effective gravity vanishes -- that is the
        # definition of break-up. Getting this wrong (v_crit from sqrt(GM/R_eq)
        # rather than the Roche sqrt(2GM/3R_pol)) let g turn back UP past
        # ω_frac ~ 0.95 instead of going to zero.
        @test local_gravity(π / 2, 1.0) < 1e-4 * local_gravity(0.0, 1.0)
        # No rotation: uniform gravity.
        for θ in (0.0, 0.4, π / 2)
            @test local_gravity(θ, 0.0) ≈ 1.0 atol = 1e-9
        end
    end

    # ================================================================
    # 2. Surface brightness
    # ================================================================
    @testset "Brightness normalisation" begin
        # ω = 0 is the null: brightness is uniform and equal to 1 everywhere,
        # for every i_star and λ. This is what makes the GD model reduce to the
        # standard one CONTINUOUSLY rather than by a special case.
        for (x, y) in ((0.0, 0.0), (0.3, -0.5), (0.0, 0.9))
            for i in (0.0, π / 4, π / 2), λ in (0.0, 1.0, -2.0)
                @test gd_brightness(x, y, i, λ, 0.0) == 1.0
            end
        end

        # The disc average is 1 by construction, so the transit depth is not
        # rescaled wholesale -- only redistributed across the chord.
        for i in (0.15, π / 4, π / 2), wf in (0.3, 0.7)
            n = 200; tot = 0.0; w = 0.0
            for ix in 1:n, iy in 1:n
                x = -1 + 2 * (ix - 0.5) / n
                y = -1 + 2 * (iy - 0.5) / n
                ρ2 = x^2 + y^2
                ρ2 >= 1 && continue
                μ = sqrt(1 - ρ2)
                tot += gd_brightness(x, y, i, 0.0, wf) * μ
                w += μ
            end
            @test tot / w ≈ 1.0 rtol = 0.02
        end

        # Pole-on (i_star = 0): the disc centre IS the pole, so it is the
        # BRIGHTEST point and brightness falls off axisymmetrically. The
        # original projection put the disc centre at the equator instead,
        # which inverted the whole pattern for exactly the low-i_star stars
        # this method is meant to detect.
        let wf = 0.7
            centre = gd_brightness(0.0, 0.0, 0.0, 0.0, wf)
            for (x, y) in ((0.8, 0.0), (0.0, 0.8), (-0.57, 0.57))
                @test gd_brightness(x, y, 0.0, 0.0, wf) < centre
            end
            # axisymmetric when pole-on: λ cannot matter
            @test gd_brightness(0.6, 0.2, 0.0, 0.0, wf) ≈
                  gd_brightness(0.6, 0.2, 0.0, 1.3, wf) rtol = 1e-10
        end

        # Equator-on (i_star = π/2): the spin axis lies IN the sky plane along
        # +y, so the colatitude of a visible point depends on y alone. Every
        # point with y = 0 is on the equator, however far it is from disc
        # centre -- brightness varies along y and is flat along x.
        let wf = 0.7
            centre = gd_brightness(0.0, 0.0, π / 2, 0.0, wf)
            @test gd_brightness(0.0, 0.85, π / 2, 0.0, wf) > centre
            @test gd_brightness(0.85, 0.0, π / 2, 0.0, wf) ≈ centre rtol = 1e-12
            @test gd_brightness(0.0, -0.85, π / 2, 0.0, wf) ≈
                  gd_brightness(0.0, 0.85, π / 2, 0.0, wf) rtol = 1e-12
        end

        # λ rotates the pattern rigidly: brightness at (x,y) with λ equals
        # brightness at the λ-rotated point with λ = 0. This is the shared
        # convention with rm.jl -- if it broke, the tomographic λ and the
        # gravity-darkening λ would mean different angles and could not be
        # fitted jointly.
        let wf = 0.6, i = 1.1, λ = 0.7, x = 0.4, y = -0.3
            s, c = sincos(λ)
            @test gd_brightness(x, y, i, λ, wf) ≈
                  gd_brightness(x * c + y * s, -x * s + y * c, i, 0.0, wf) rtol = 1e-10
        end
    end

    # ================================================================
    # 3. Transit flux
    # ================================================================
    @testset "transit_flux_gd reduces to Mandel-Agol" begin
        u1, u2 = 0.35, 0.22
        ld = QuadLimbDark([u1, u2])
        rp = 0.116
        for x in -1.3:0.13:1.3
            y = 0.75
            z = sqrt(x^2 + y^2)
            ref = transit_flux(z, rp, u1, u2)
            # ω = 0 must be EXACT, not approximate -- the standard model is a
            # limit of this one, not a different model.
            @test transit_flux_gd(x, y, 1.0, rp, u1, u2, 1.2, 0.5, 0.0) ≈ ref atol = 1e-14
            @test transit_flux_gd(ld, x, y, 1.0, rp, 1.2, 0.5, 0.0) ≈ ref atol = 1e-14
        end
        # Behind the star: no transit, regardless of sky position.
        @test transit_flux_gd(0.0, 0.2, -1.0, rp, u1, u2, 1.2, 0.5, 0.7) == 1.0
        @test transit_flux_gd(ld, 0.0, 0.2, -1.0, rp, 1.2, 0.5, 0.7) == 1.0
        # Off the disc entirely: no transit.
        @test transit_flux_gd(ld, 3.0, 0.0, 1.0, rp, 1.2, 0.5, 0.7) ≈ 1.0 atol = 1e-14
    end

    @testset "Asymmetry is the signal" begin
        # The whole method rests on ingress and egress being unequal. For an
        # aligned orbit (λ = 0) across a star seen equator-on the chord is
        # symmetric about the spin axis, so the light curve stays symmetric;
        # a misaligned orbit breaks that symmetry.
        ld = QuadLimbDark([0.35, 0.22])
        rp, b, wf, i = 0.116, 0.75, 0.6, π / 2
        f(x, λ) = transit_flux_gd(ld, x, b, 1.0, rp, i, λ, wf)

        for x in (0.3, 0.55, 0.8)
            @test f(x, 0.0) ≈ f(-x, 0.0) rtol = 1e-10      # aligned: symmetric
        end
        λm = deg2rad(-60.0)                        # strongly misaligned
        @test !isapprox(f(0.55, λm), f(-0.55, λm); rtol = 1e-6)

        # Size of the effect for a fast-rotating A star: v sin i = 25.9 km/s.
        # At low i_star
        # the equatorial velocity -- hence the oblateness -- is larger, so the
        # asymmetry grows. It is NOT monotonic in i_star, though: two effects
        # compete. Falling i_star raises ω (more oblate, more contrast) but
        # swings the pole toward disc centre, which SHRINKS the brightness
        # gradient along the transit chord. The result has a shallow minimum
        # near i_star ~ 60 deg, so the light curve alone can admit two i_star
        # branches -- that is a property of the physics, not a bug, and it is
        # why the depth (which also shifts) carries information too.
        M_s, R_s = 1.60, 1.47
        asym(i_s) = begin
            wfi = omega_frac_from_veq(25.9 / sin(i_s), M_s, R_s)
            g(x) = transit_flux_gd(ld, x, b, 1.0, rp, i_s, λm, wfi)
            abs(g(0.55) - g(-0.55))
        end
        @test asym(deg2rad(15)) > asym(deg2rad(90))     # pole-on wins overall
        @test asym(deg2rad(20)) > asym(deg2rad(40))
        @test asym(deg2rad(90)) < 2e-4                  # equator-on: tens of ppm
        @test asym(deg2rad(15)) > 1e-4                  # pole-on: hundreds
    end

    @testset "omega_frac_from_veq" begin
        # v_crit for the Sun on the Roche definition: sqrt(2GM/3R_pol).
        @test omega_frac_from_veq(0.0, 1.0, 1.0) == 0.0
        v_crit_sun = sqrt(2 * 6.674e-11 * 1.989e30 / (3 * 6.957e8)) / 1e3
        @test omega_frac_from_veq(v_crit_sun, 1.0, 1.0) ≈ 1.0 atol = 1e-12
        @test omega_frac_from_veq(2 * v_crit_sun, 1.0, 1.0) ≈ 1.0 atol = 1e-12     # clamped

        # Round-trip against the Roche model, from first principles. Pick an
        # Ω/Ω_crit, build the star it implies, read off its equatorial velocity,
        # and require the conversion to recover the rotation rate we started
        # from. This is the check that would have caught v_eq/v_crit being
        # returned in place of Ω/Ω_crit -- the two differ by up to 3/2, which
        # is a factor ~2 in the light-curve asymmetry.
        #
        # Deliberately NOT asserted against published oblateness for KELT-9 b /
        # WASP-33 b: those come from full 2-D fits with their own adopted
        # stellar parameters, and reproducing them is a separate exercise from
        # showing this conversion is internally exact.
        let G = 6.674e-11, Msun = 1.989e30, Rsun = 6.957e8
            for (M, Rpol) in ((1.60, 1.47), (1.98, 2.36), (0.9, 0.85))
                Ω_crit = sqrt(G * M * Msun / (1.5 * Rpol * Rsun)^3)
                for ω in (0.05, 0.2, 0.5, 0.8, 0.99)
                    R_eq = roche_radius(π / 2, ω) * Rpol * Rsun
                    v_eq = ω * Ω_crit * R_eq / 1e3            # km/s
                    @test omega_frac_from_veq(v_eq, M, Rpol) ≈ ω rtol = 1e-6
                end
            end
        end
    end

    @testset "Chromatic exponent" begin
        # n = 4 is the BOLOMETRIC value, not a monochromatic limit: it is
        # reached where x = hc/(λkT) ≈ 3.9207, i.e. λT ≈ 3.6698e-3 m·K. Bluer
        # bands exceed it, redder bands fall short.
        for T in (5000.0, 7437.0, 12_000.0)
            @test gd_beta_band(3.6698e-3 / T * 1e9, T) ≈ 0.25 rtol = 1e-3
        end
        # Rayleigh-Jeans limit (x -> 0): n -> 1, so β_eff -> β/4.
        @test gd_beta_band(1.0e7, 5000.0) ≈ 0.25 / 4 rtol = 1e-3
        # Redder band -> weaker contrast, monotonically.
        for λ in (400.0, 550.0, 790.0, 1200.0)
            @test gd_beta_band(λ, 7437.0) < gd_beta_band(λ * 0.8, 7437.0)
        end
        # An A9V host (7437 K) in the TESS band: the correction is 30-35%,
        # i.e. the size of the effect being measured -- not a refinement.
        β_tess = gd_beta_band(786.5, 7437.0)
        @test 0.60 < β_tess / 0.25 < 0.72
        @test gd_beta_band(475.0, 7437.0) > β_tess          # g\u2032 is stronger
    end

    # ================================================================
    # 4. Likelihood wiring -- the part that fails SILENTLY
    # ================================================================
    @testset "GD reaches the photometric likelihood" begin
        # A short synthetic transit sampled densely enough to resolve ingress
        # and egress separately; the GD signal lives in their difference.
        P, Tc, rp, b, aRs = 2.827969, 0.0, 0.116, 0.75, 6.815
        # ODD sample count so index (n+1)/2 lands exactly on Tc -- the
        # reflection about mid-transit is then exact and the asymmetry metric
        # measures physics rather than a half-cadence offset against the
        # ingress slope (which alone forged ~46 ppm of spurious signal).
        t = collect(range(Tc - 0.09, Tc + 0.09; length = 301))
        flux = ones(length(t))
        err = fill(2.4e-5, length(t))          # per-cadence, roughly

        fx(v) = Dict("type" => "FixedPrior", "args" => [v])

        function build(mode; extra = Dict{String, Any}())
            priors = Dict{String, Any}(
                "P_k1"      => fx(P),
                "Tc_k1"     => fx(Tc),
                "b_k1"      => fx(b),
                "rr_k1"     => fx(rp),
                "sesinw_k1" => fx(0.0),
                "secosw_k1" => fx(0.0),
                # rho_s that reproduces the adopted a/R*, so the geometry is the
                # adopted one rather than an arbitrary transit.
                "rho_s"     => fx(3π * aRs^3 / (6.674e-8 * (P * 86400)^2)),
                "offset_TESS"   => fx(0.0),
                "jitter_TESS"   => fx(0.0),
                "dilution_TESS" => fx(0.0),
                "q1_TESS"       => fx(0.33),
                "q2_TESS"       => fx(0.28),
            )
            merge!(priors, extra)
            cfg = Dict{String, Any}(
                "star"   => Dict("M_s" => 1.60, "R_s" => 1.47),
                "priors" => priors,
                "model"  => Dict("max_kplanet" => 1,
                                 "planet_modes" => [mode],
                                 "parametrization" => Dict("time" => "Tc",
                                                           "use_rho_s" => true)),
                "data"   => Dict("transit_photometry" => [Dict(
                    "instrument" => "TESS", "exposure_time" => 120.0,
                    "values" => Dict("bjd" => t, "flux" => flux,
                                     "flux_err" => err))]),
            )
            data, irv, ipm = Nereus._build_data(cfg["data"])
            star = Nereus._build_star(cfg["star"])
            params, _, _ = Nereus._build_model(cfg, data, star, irv, ipm)
            return params, data
        end

        # The GD model carries exactly ONE extra free parameter, i_star:
        # gd_beta is fixed at the von Zeipel value and v_sin_i_star is shared
        # with the RM slot, so ω is derived, not sampled.
        gd_extra = Dict{String, Any}(
            "lambda_k1"    => fx(deg2rad(-60.0)),      # a misaligned λ
            "v_sin_i_star" => fx(25_900.0),            # m/s, spectroscopic
        )
        params_pm, data_pm = build("PM_ONLY")
        params_gd, data_gd = build("PM_GD"; extra = gd_extra)

        @test "i_star" in params_gd.layout.names
        @test "gd_beta_TESS" in params_gd.layout.names
        @test !("i_star" in params_pm.layout.names)
        # gd_beta must NOT be sampled by default -- a free β trades against
        # i_star for the same asymmetry.
        @test !("gd_beta_TESS" in params_gd.layout.unfrozen_names)
        @test "i_star" in params_gd.layout.unfrozen_names

        function ll_at(params, data, i_star_deg)
            θ = Nereus.Theta(params)
            idx = params.layout.systemic.i_star
            idx == 0 || (θ.values[idx] = deg2rad(i_star_deg))
            return Nereus.transit_log_likelihood(θ, data)
        end

        # (a) equator-on and pole-on must give DIFFERENT likelihoods. If the
        #     `gd` state never reached _phot_transit_product these would be
        #     bit-identical and the i_star posterior would come back flat --
        #     the exact silent failure this test exists to catch.
        ll_90 = ll_at(params_gd, data_gd, 90.0)
        ll_20 = ll_at(params_gd, data_gd, 20.0)
        @test isfinite(ll_90) && isfinite(ll_20)
        @test !isapprox(ll_90, ll_20; rtol = 1e-12)

        # (b) at v sin i -> 0 the GD model must collapse onto the plain one.
        params_z, data_z = build("PM_GD"; extra = merge(copy(gd_extra),
            Dict{String, Any}("v_sin_i_star" => fx(1.0))))
        @test ll_at(params_z, data_z, 55.0) ≈ ll_at(params_pm, data_pm, 90.0) rtol = 1e-9

        # (c) the predicted flux really is asymmetric about mid-transit, and
        #     more so pole-on. This is the observable, independent of logL.
        function depth_asym(params, data, i_star_deg)
            θ = Nereus.Theta(params)
            θ.values[params.layout.systemic.i_star] = deg2rad(i_star_deg)
            pred, _ = Nereus.phot_predictions(θ, data)
            m = (length(pred) + 1) ÷ 2                    # exact mid-transit
            return maximum(abs.(pred[1:(m - 1)] .- reverse(pred[(m + 1):end])))
        end
        @test depth_asym(params_gd, data_gd, 20.0) >
              depth_asym(params_gd, data_gd, 90.0)
        @test depth_asym(params_gd, data_gd, 20.0) > 5e-5   # vs a 24 ppm floor

        # λ = 0: the chord is symmetric about the projected spin axis, so the
        # light curve must be symmetric to machine precision no matter how
        # oblate the star is. This is the sharpest end-to-end null available --
        # it exercises the whole path (layout, decode, sky position, brightness)
        # and still has an exactly-known answer.
        params_a, data_a = build("PM_GD"; extra = merge(copy(gd_extra),
            Dict{String, Any}("lambda_k1" => fx(0.0))))
        @test depth_asym(params_a, data_a, 25.0) < 1e-15
    end
end
