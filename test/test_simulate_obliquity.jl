# Forward simulation of tomographic and RM nights.
#
# The bar for a simulator is not that it produces plausible-looking arrays. It
# is that data it produced, pushed through the SAME analysis path as real data,
# returns the geometry that went in. Every test here injects a known lambda and
# asks the real estimator to find it; nothing inspects the simulator's internals.
using Test
using Nereus
using Nereus: simulate_tomogram, simulate_rm_night, tomogram_residuals,
               tomogram_pooled,
               tomogram_matched_filter, shadow_track, RMNight
using Random, Statistics

# A synthetic mid-F rotator on a grazing transit: fast enough that the shadow
# is resolved, and impact parameter high enough that the track is short. The
# values are representative, not any particular star.
const SP = (P = 2.83, Tc = 2459000.5, a_Rs = 6.84, b = 0.75,
            vsini = 25.9, rr = 0.116, T14 = 2.62, λ = deg2rad(-47.0))

recover(s; λs = range(-π, π; length = 721)) = begin
    g, R = tomogram_residuals(s.profiles, s.vgrid, s.in_transit; bervs = s.bervs)
    λv, sc = tomogram_matched_filter(R, g, s.times, s.Tc, SP.P, SP.a_Rs,
                                     s.truth.inc, SP.vsini; λs = λs)
    (λ = λv[argmax(sc)], score = maximum(sc), grid = g, R = R)
end

@testset "obliquity forward simulation" begin

    # ---- sampling and geometry ------------------------------------------
    @testset "sampling" begin
        s = simulate_tomogram(; SP..., sigma_pixel = 0.0, cadence = 632.0)
        # span defaults to 2*T14, so half the frames are in transit and half out.
        @test count(s.in_transit) > 0
        @test count(.!s.in_transit) > 0
        @test isapprox(count(s.in_transit) / s.truth.n_frames, 0.5; atol = 0.1)
        @test length(s.times) == size(s.profiles, 1)
        @test length(s.vgrid) == size(s.profiles, 2)
        @test s.truth.b ≈ SP.b atol = 1e-6
        # b and inc are two spellings of one thing.
        s2 = simulate_tomogram(; SP..., b = nothing, inc = s.truth.inc,
                               sigma_pixel = 0.0, cadence = 632.0)
        @test s2.profiles ≈ s.profiles
        @test_throws ArgumentError simulate_tomogram(; SP..., inc = 1.4,
                                                     sigma_pixel = 0.0)
    end

    # ---- the shadow is where the track says it is -----------------------
    # Noiseless: the deepest pixel of each in-transit residual row must sit at
    # the sub-planet velocity. This ties the simulator to `shadow_track`
    # directly, without going through the filter.
    @testset "shadow position" begin
        λ = deg2rad(-47.0)
        s = simulate_tomogram(; SP..., λ = λ, sigma_pixel = 0.0, cadence = 632.0)
        g, R = tomogram_residuals(s.profiles, s.vgrid, s.in_transit; bervs = s.bervs)
        tr = shadow_track(s.times, s.Tc, SP.P, SP.a_Rs, s.truth.inc, λ, SP.vsini)
        for i in findall(s.in_transit)
            isfinite(tr[i]) || continue
            @test abs(g[argmin(view(R, i, :))] - tr[i]) < 1.0   # within 2 pixels
        end
    end

    # ---- noiseless recovery across lambda --------------------------------
    @testset "noiseless recovery" begin
        for λ_deg in (-47.0, 0.0, 33.0, 120.0)
            s = simulate_tomogram(; SP..., λ = deg2rad(λ_deg),
                                  sigma_pixel = 0.0, cadence = 632.0)
            r = recover(s)
            @test abs(rad2deg(r.λ) - λ_deg) < 5.0
        end
    end

    # ---- the BERV trap ----------------------------------------------------
    # A night at nonzero BERV must recover the same lambda, and must FAIL to if
    # the shift is not passed. If both branches agree the test proves nothing,
    # so the second assertion is what gives the first its meaning.
    @testset "berv handling" begin
        λ = deg2rad(-47.0)
        s = simulate_tomogram(; SP..., λ = λ, sigma_pixel = 0.0,
                              cadence = 632.0, berv = 13.4)
        @test abs(rad2deg(recover(s).λ) - (-47.0)) < 5.0
        g, R = tomogram_residuals(s.profiles, s.vgrid, s.in_transit)  # berv dropped
        λv, sc = tomogram_matched_filter(R, g, s.times, s.Tc, SP.P, SP.a_Rs,
                                         s.truth.inc, SP.vsini;
                                         λs = range(-π, π; length = 721))
        @test abs(rad2deg(λv[argmax(sc)]) - (-47.0)) > 10.0
    end

    # ---- noise is what the caller asked for ------------------------------
    # With rr = 0 there is no shadow, so the residual map is pure noise and its
    # per-pixel scatter must be `sigma_pixel` in unit-line-depth units. This is
    # the property that makes the argument comparable to a measured number from
    # a real night.
    @testset "noise calibration" begin
        rng = MersenneTwister(4242)
        s = simulate_tomogram(; SP..., rr = 0.0, sigma_pixel = 0.066,
                              cadence = 632.0, span = 20.0, rng = rng)
        g, R = tomogram_residuals(s.profiles, s.vgrid, s.in_transit; bervs = s.bervs)
        core = abs.(g) .< 40                       # away from the wing-fit region
        @test isapprox(std(vec(R[:, core])), 0.066; rtol = 0.15)
        # line_depth must not leak into the normalised units
        s3 = simulate_tomogram(; SP..., rr = 0.0, sigma_pixel = 0.066,
                               cadence = 632.0, span = 20.0, line_depth = 0.058,
                               rng = MersenneTwister(4242))
        _, R3 = tomogram_residuals(s3.profiles, s3.vgrid, s3.in_transit;
                                   bervs = s3.bervs)
        @test isapprox(std(vec(R3[:, core])), 0.066; rtol = 0.15)
    end

    @testset "snr scaling" begin
        mk(σ) = simulate_tomogram(; SP..., sigma_pixel = σ, cadence = 632.0).truth.snr_white
        @test mk(0.066) / mk(0.132) ≈ 2.0 rtol = 1e-8
        # rr enters squared: half the radius ratio is a quarter of the signal.
        half = simulate_tomogram(; SP..., rr = SP.rr / 2, sigma_pixel = 0.066,
                                 cadence = 632.0).truth.snr_white
        @test mk(0.066) / half ≈ 4.0 rtol = 0.02
        # a single night at this cadence and noise is a marginal 3-4 sigma in
        # the white limit — which is why pooling exists.
        @test 2.5 < mk(0.066) < 5.0
    end

    # ---- recovery survives realistic noise -------------------------------
    @testset "noisy recovery" begin
        λ_deg = -47.0
        hits = 0
        for seed in 1:12
            s = simulate_tomogram(; SP..., λ = deg2rad(λ_deg), sigma_pixel = 0.066,
                                  cadence = 632.0, rng = MersenneTwister(seed))
            abs(rad2deg(recover(s).λ) - λ_deg) < 25.0 && (hits += 1)
        end
        # a single ~3.6 sigma night is marginal by construction; most should land
        @test hits >= 8
    end

    @testset "correlated noise and injected frames" begin
        rng = MersenneTwister(7)
        s = simulate_tomogram(; SP..., rr = 0.0, sigma_pixel = 0.05,
                              cadence = 632.0, span = 20.0, sigma_ip = 2.6, rng = rng)
        g, R = tomogram_residuals(s.profiles, s.vgrid, s.in_transit; bervs = s.bervs)
        core = abs.(g) .< 40
        # smoothing changes the correlation structure, NOT the per-pixel sigma
        @test isapprox(std(vec(R[:, core])), 0.05; rtol = 0.2)
        # neighbouring pixels are now correlated, which white noise is not
        X = R[:, core]
        c = cor(vec(X[:, 1:end-1]), vec(X[:, 2:end]))
        @test c > 0.5

        base = simulate_tomogram(; SP..., sigma_pixel = 0.0, cadence = 632.0)
        nf = zeros(size(base.profiles))
        @test simulate_tomogram(; SP..., sigma_pixel = 0.0, cadence = 632.0,
                                noise_frames = nf).profiles ≈ base.profiles
        @test_throws DimensionMismatch simulate_tomogram(; SP..., sigma_pixel = 0.0,
                                                         cadence = 632.0,
                                                         noise_frames = zeros(3, 3))
    end

    # ---- the pooled p-value must respond to signal strength ---------------
    #
    # THIS IS THE TEST THAT WAS MISSING. `tomogram_pooled` used to divide each
    # lambda-scan by its own std() before comparing it against the permutation
    # null, and because a broad shadow raises that dispersion along with the
    # peak, the ratio was almost unchanged by shuffling: p sat near 0.5 whatever
    # the signal. Nothing caught it, because no test ever asked the p-value to
    # MOVE. Recovering lambda correctly does not exercise the normalisation at
    # all -- argmax is invariant to it -- so the existing recovery tests passed
    # throughout.
    #
    # The assertion is therefore about ordering, not about a threshold: a strong
    # injected shadow must be significant, pure noise must not, and any future
    # statistic that cannot tell those apart fails here.
    @testset "pooled p-value responds to S/N" begin
        # three nights months apart at different barycentric velocities — the
        # offsets are what make BERV handling load-bearing when pooling.
        cfg = [("N1", 2459000.5, -8.2, 786.0),
               ("N2", 2459360.5,  5.2, 627.0),
               ("N3", 2459411.5, 13.4, 632.0)]
        build(σ, rr) = [simulate_tomogram(; SP..., rr = rr, Tc = tc, berv = bv,
                                          cadence = cad, sigma_pixel = σ, tag = tg,
                                          rng = MersenneTwister(hash(tg) % 10000))
                        for (tg, tc, bv, cad) in cfg]
        pooled(ns) = tomogram_pooled(ns; P = SP.P, a_Rs = SP.a_Rs,
                                     inc = acos(SP.b / SP.a_Rs), vsini = SP.vsini,
                                     T14 = SP.T14 / 24, filter_pulsations = false,
                                     n_null = 150, rng = MersenneTwister(1))

        λ_hi, _, p_hi, _, _ = pooled(build(0.02, SP.rr))       # strong shadow
        @test abs(rad2deg(λ_hi) - rad2deg(SP.λ)) < 15.0
        @test p_hi < 0.05

        # rr = 0 is the honest control: identical noise, no planet at all.
        _, _, p_null, _, _ = pooled(build(0.02, 0.0))
        @test p_null > 0.2

        # and the statistic must actually separate the two cases
        @test p_null > 10 * max(p_hi, 1e-3)
    end

    # ---- RM night ---------------------------------------------------------
    @testset "rm night" begin
        r = simulate_rm_night(; SP..., λ = 0.0, sigma_rv = 0.0, cadence = 600.0)
        @test r.night isa RMNight
        n = r.night
        out = abs.((n.t .- n.Tc) .* 24) .> SP.T14 / 2
        @test all(abs.(n.rv[out]) .< 1e-6)          # anomaly is zero out of transit
        @test r.truth.amp_max > 10.0                # a real signal, tens of m/s

        # lambda = 0 gives an antisymmetric anomaly about mid-transit.
        inn = .!out
        early = n.rv[inn .& (n.t .< n.Tc)]
        late  = n.rv[inn .& (n.t .> n.Tc)]
        @test sign(mean(early)) == -sign(mean(late))

        # lambda = 90 deg parks the shadow off-centre: the anomaly stops
        # changing sign and keeps one.
        #
        # Test the NONZERO values only. The nominal T14 window and the window in
        # which the planet centre is actually on the disc are not the same
        # interval -- `rm_anomaly` gates on hypot(x, y) < 1 and returns exactly
        # zero outside it -- so the first and last frames of a T14 mask are
        # legitimately 0.0 and would break a strict all(>(0), ...).
        r90 = simulate_rm_night(; SP..., λ = deg2rad(90), sigma_rv = 0.0,
                                cadence = 600.0)
        n90 = r90.night
        inn90 = abs.((n90.t .- n90.Tc) .* 24) .<= SP.T14 / 2
        on_disc = n90.rv[inn90 .& .!iszero.(n90.rv)]
        @test length(on_disc) >= 8
        @test all(>(0), on_disc) || all(<(0), on_disc)

        # gamma and K enter as stated
        rg = simulate_rm_night(; SP..., λ = 0.0, sigma_rv = 0.0, cadence = 600.0,
                               gamma = 18930.0, K = 200.0)
        @test isapprox(mean(rg.night.rv), 18930.0; atol = 60.0)
        @test_throws ArgumentError simulate_rm_night(; SP..., inc = 1.4,
                                                     sigma_rv = 1.0)
    end
end
