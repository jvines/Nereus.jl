# Joint obliquity fit: the Doppler shadow and the RM velocities in one posterior.
#
# Three things are worth pinning here. The ARoME anomaly must have the right
# sign and vanish outside transit, or the joint fit measures lambda with the
# wrong handedness. Lambda must WRAP rather than be bounded: a hard edge at
# +/- pi is a wall on a circle, and it parked walkers at +/-180 deg and inflated
# the 3 sigma interval to the whole circle before it was caught. And the fit
# must recover an injected lambda from data where the truth is known, in all
# three configurations, because the value of the joint fit is that the three can
# be compared.
using Test
using Nereus
using Nereus: TomoNight, RMNight, rm_anomaly, joint_obliquity_logpost,
               joint_obliquity_fit, shadow_map, tomogram_residuals,
               circular_summary
using Random, Statistics

@testset "Joint obliquity fit" begin

    P, Tc = 2.827969, 0.0
    aRs, inc, rr = 6.81, deg2rad(83.6), 0.116
    vsini_ms, σ0 = 25_900.0, 15_700.0

    # ---- ARoME anomaly ---------------------------------------------------
    @testset "rm_anomaly" begin
        t = collect(range(-0.10, 0.10; length = 201))
        a = rm_anomaly(t, Tc, P, aRs, inc, 0.0, vsini_ms, σ0; rr = rr)

        # zero outside transit, non-zero inside
        out = abs.(t) .> 0.06
        @test all(a[out] .== 0)
        @test any(a .!= 0)

        # lambda = 0: the planet crosses approaching then receding limb, so the
        # anomaly is antisymmetric about mid-transit and changes sign
        inb = findall(a .!= 0)
        @test a[inb[1]] * a[inb[end]] < 0

        # the anomaly scales with the blocked area, so a bigger planet gives a
        # bigger signal at the same geometry
        big = rm_anomaly(t, Tc, P, aRs, inc, 0.0, vsini_ms, σ0; rr = 2rr)
        @test maximum(abs.(big)) > maximum(abs.(a))

        # lambda -> -lambda mirrors the sub-planet velocity, and at b != 0 the
        # curve is genuinely different rather than a pure sign flip
        m = rm_anomaly(t, Tc, P, aRs, inc, deg2rad(60.0), vsini_ms, σ0; rr = rr)
        p = rm_anomaly(t, Tc, P, aRs, inc, deg2rad(-60.0), vsini_ms, σ0; rr = rr)
        @test !isapprox(m, p; atol = 1e-8)
    end

    # ---- synthetic data --------------------------------------------------
    rng = MersenneTwister(4321)
    λtrue = deg2rad(-55.0)
    hrs = collect(range(-2.2, 2.2; length = 26))
    t   = Tc .+ hrs ./ 24
    grid = collect(range(-42, 42; length = 41))

    tnight = let
        M = shadow_map(t, Tc, P, aRs, inc, λtrue, 25.9, grid, 8.0; rr = rr)
        R = 6.0 .* M .+ 0.004 .* randn(rng, length(t), length(grid))
        TomoNight("SYN", t, R, grid, Tc)
    end
    rnight = let
        a = rm_anomaly(t, Tc, P, aRs, inc, λtrue, vsini_ms, σ0; rr = rr)
        RMNight("SYN", t, a .+ 40 .* randn(rng, length(t)),
                fill(40.0, length(t)), σ0, Tc)
    end

    # ---- lambda is periodic, not bounded ---------------------------------
    # THE regression test. lambda + 2*pi must give exactly the same posterior;
    # a hard bound returns -Inf there instead.
    @testset "lambda wraps" begin
        hours = [(tnight.t .- tnight.Tc) .* 24]
        θ = vcat([λtrue, 25.9, aRs*cos(inc), aRs],
                 [6.0, 8.0, log10(0.004), log10(8.0), log10(1.5), log10(0.004)],
                 [370.0], [0.0, log10(1600.0), 0.0, log10(6.0), log10(40.0)])
        base = joint_obliquity_logpost(θ, [tnight], hours, [rnight], P;
                    vsini_mu = 25.9, vsini_sd = 1.5, b_mu = aRs*cos(inc),
                    b_sd = 0.02, a_mu = aRs, a_sd = 0.1,
                    rr = rr, u1 = 0.32, u2 = 0.30)
        @test isfinite(base)
        for k in (-2, -1, 1, 2)
            θk = copy(θ); θk[1] = λtrue + 2π*k
            wrapped = joint_obliquity_logpost(θk, [tnight], hours, [rnight], P;
                    vsini_mu = 25.9, vsini_sd = 1.5, b_mu = aRs*cos(inc),
                    b_sd = 0.02, a_mu = aRs, a_sd = 0.1,
                    rr = rr, u1 = 0.32, u2 = 0.30)
            @test isfinite(wrapped)
            @test isapprox(wrapped, base; atol = 1e-8)
        end
    end

    # ---- recovery in all three configurations ----------------------------
    # Short chains: this asks whether the machinery finds the right answer at
    # all, not what its uncertainty is.
    common = (P = P, vsini = (25.9, 1.5), b = (aRs*cos(inc), 0.02),
              a_Rs = (aRs, 0.1), K = (370.0, 30.0), rr = rr, u1 = 0.32,
              u2 = 0.30, n_walkers = 60, n_steps = 1500, n_burn = 750,
              thin = 5, seed = 99)

    @testset "recovers injected lambda" begin
        for (lab, tn, rn, ut) in (("tomogram", [tnight], RMNight[], true),
                                  ("velocities", [tnight], [rnight], false),
                                  ("both", [tnight], [rnight], true))
            r = joint_obliquity_fit(tn, rn; use_tomogram = ut, common...)
            c = circular_summary(r.λ)
            @test abs(mod(c.mode - rad2deg(λtrue) + 180, 360) - 180) < 25
            # the chain must carry a name per sampled parameter
            @test length(r.names) == size(r.chain, 1)
        end
    end

    # ---- the three configurations share one parameter space ---------------
    # Otherwise a difference between them confounds likelihood with
    # parameterisation, which is the whole point of being able to compare them.
    @testset "shared parameter space" begin
        a = joint_obliquity_fit([tnight], RMNight[];  use_tomogram = true,  common...)
        b = joint_obliquity_fit([tnight], [rnight];   use_tomogram = false, common...)
        c = joint_obliquity_fit([tnight], [rnight];   use_tomogram = true,  common...)
        @test size(b.chain, 1) == size(c.chain, 1)
        @test b.names == c.names
        # tomogram-only drops the RV block, so it is shorter by K + 5 per night
        @test size(a.chain, 1) == size(c.chain, 1) - 6
    end
end
