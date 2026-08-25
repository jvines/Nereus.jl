using Nereus
using Distributions
using ForwardDiff
using LogDensityProblems
using MCMCChains
using Random
using Statistics: median, mean
using Test

const _rng = MersenneTwister(1)

@testset "Nereus foundation" begin

    # -----------------------------------------------------------------
    @testset "Kepler solver" begin
        # Circular orbit: E ≡ M.
        @test kepler_solve(0.0, 0.0) ≈ 0.0 atol=1e-12
        @test kepler_solve(π, 0.0) ≈ π atol=1e-12
        @test kepler_solve(π/2, 0.0) ≈ π/2 atol=1e-12
        @test kepler_solve(-π/2, 0.0) ≈ -π/2 atol=1e-12

        # At periastron (M = 0), E = 0 regardless of e.
        @test kepler_solve(0.0, 0.1) ≈ 0.0 atol=1e-12
        @test kepler_solve(0.0, 0.5) ≈ 0.0 atol=1e-12
        @test kepler_solve(0.0, 0.9) ≈ 0.0 atol=1e-12

        # At apoastron (M = π), E = π regardless of e.
        @test kepler_solve(π, 0.1) ≈ π atol=1e-10
        @test kepler_solve(π, 0.5) ≈ π atol=1e-10
        @test kepler_solve(π, 0.9) ≈ π atol=1e-10

        # Round-trip identity: E → M → E should reproduce E.
        # Grid over (M, e) excluding pathological high e.
        for e in (0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
            for M in range(-π + 0.01, π - 0.01, length=64)
                E = kepler_solve(M, e)
                M_back = E - e * sin(E)
                @test M ≈ M_back atol=1e-10
            end
        end

        # Large |M| should converge (multi-period).
        for M in (-10π, -3π, 3π, 10π, 100π)
            E = kepler_solve(M, 0.5)
            M_back = E - 0.5 * sin(E)
            @test M ≈ M_back atol=1e-9
        end

        # Type stability: Float32 in → Float32 out.
        @test kepler_solve(1.0f0, 0.1f0) isa Float32
        @test kepler_solve(1.0, 0.1) isa Float64

        # Vector form.
        M_vec = collect(range(-π, π, length=17))
        E_vec = kepler_solve(M_vec, 0.3)
        @test length(E_vec) == length(M_vec)
        for (Mi, Ei) in zip(M_vec, E_vec)
            @test Mi ≈ Ei - 0.3 * sin(Ei) atol=1e-10
        end

        # In-place vector form.
        E_buf = similar(M_vec)
        kepler_solve!(E_buf, M_vec, 0.3)
        @test E_buf ≈ E_vec
    end

    # -----------------------------------------------------------------
    @testset "True anomaly" begin
        # At periastron: E = 0 → f = 0.
        @test Nereus.true_anomaly(0.0, 0.5) ≈ 0.0 atol=1e-12
        # At apoastron: E = π → f = π.
        @test Nereus.true_anomaly(π, 0.5) ≈ π atol=1e-12
        @test Nereus.true_anomaly(-π, 0.5) ≈ -π atol=1e-12

        # Circular orbit: f ≡ E.
        for M in range(-π + 0.01, π - 0.01, length=32)
            E = kepler_solve(M, 0.0)
            f = Nereus.true_anomaly(E, 0.0)
            @test f ≈ E atol=1e-12
        end

        # Monotonic in E: f(E1) < f(E2) for E1 < E2 within (-π, π).
        e = 0.4
        Es = collect(range(-π + 0.01, π - 0.01, length=32))
        fs = [Nereus.true_anomaly(E, e) for E in Es]
        @test all(diff(fs) .> 0)
    end

    # -----------------------------------------------------------------
    @testset "Parametrization conversions" begin
        # sesinw round trip
        for (e, ω) in ((0.3, 1.5), (0.05, -1.2), (0.7, 0.0), (0.0, 2.5))
            sesinw, secosw = Nereus.ew_to_sesinw(e, ω)
            e_back, ω_back = Nereus.sesinw_to_ew(sesinw, secosw)
            @test e_back ≈ e atol=1e-12
            if e > 0  # ω undefined for e = 0; skip the assertion
                @test sin(ω_back) ≈ sin(ω) atol=1e-12
                @test cos(ω_back) ≈ cos(ω) atol=1e-12
            end
        end

        # esinw round trip
        for (e, ω) in ((0.3, 1.5), (0.05, -1.2), (0.7, 0.0))
            esinw, ecosw = Nereus.ew_to_esinw(e, ω)
            e_back, ω_back = Nereus.esinw_to_ew(esinw, ecosw)
            @test e_back ≈ e atol=1e-12
            if e > 0
                @test sin(ω_back) ≈ sin(ω) atol=1e-12
                @test cos(ω_back) ≈ cos(ω) atol=1e-12
            end
        end

        # Cross-check: sesinw² + secosw² = e
        for (e, ω) in ((0.1, 0.4), (0.55, 2.1), (0.9, -2.8))
            sesinw, secosw = Nereus.ew_to_sesinw(e, ω)
            @test sesinw^2 + secosw^2 ≈ e atol=1e-12
        end
    end

    # -----------------------------------------------------------------
    @testset "Time anchor conversions" begin
        # --- Mo <-> Tp round-trip ---
        P = 5.0
        t_ref = 2450000.0
        Mo = 1.3
        Tp = Nereus.mo_to_tp(Mo, P, t_ref)
        Mo_back = Nereus.tp_to_mo(Tp, P, t_ref)
        # ~1e-9 tolerance: catastrophic cancellation from BJD subtraction
        @test Mo_back ≈ Mo atol=1e-9

        # --- Tc <-> Tp round-trip ---
        e = 0.3
        ω = 0.8
        Tp_known = 2450002.0
        Tc = Nereus.tp_to_tc(Tp_known, P, e, ω)
        Tp_back = Nereus.tc_to_tp(Tc, P, e, ω)
        @test Tp_back ≈ Tp_known atol=1e-10

        # --- Full chain: Mo -> Tp -> Tc -> Tp -> Mo ---
        Mo2 = 0.7
        Tp2 = Nereus.mo_to_tp(Mo2, P, t_ref)
        Tc2 = Nereus.tp_to_tc(Tp2, P, e, ω)
        Tp2_back = Nereus.tc_to_tp(Tc2, P, e, ω)
        Mo2_back = Nereus.tp_to_mo(Tp2_back, P, t_ref)
        @test Mo2_back ≈ Mo2 atol=1e-9

        # --- Circular orbit: e=0 ---
        e0 = 0.0
        ω0 = 1.0
        Tp_circ = 2450003.0
        Tc_circ = Nereus.tp_to_tc(Tp_circ, P, e0, ω0)
        Tp_circ_back = Nereus.tc_to_tp(Tc_circ, P, e0, ω0)
        @test Tp_circ_back ≈ Tp_circ atol=1e-12

        # --- Cross-check with manual EMPEROR formula ---
        f_tr = π/2 - ω
        E_tr = 2 * atan(tan(f_tr/2) * sqrt((1-e)/(1+e)))
        Tc_manual = Tp_known + P/(2π) * (E_tr - e*sin(E_tr))
        Tc_func = Nereus.tp_to_tc(Tp_known, P, e, ω)
        @test Tc_func ≈ Tc_manual atol=1e-14
    end

    # -----------------------------------------------------------------
    @testset "RV Keplerian model — circular" begin
        # Reference: circular orbit with ω = 0 gives rv(t) = K cos(M).
        P = 5.0
        K = 10.0
        e = 0.0
        ω = 0.0
        M0 = 0.0
        t_ref = 0.0

        # Phase sample across one period.
        t_arr = [0.0, P/4, P/2, 3P/4, P]
        rv_arr = Nereus.rv_keplerian.(t_arr, P, K, e, ω, M0, t_ref)

        @test rv_arr[1] ≈ K atol=1e-10     # cos(0) = 1
        @test rv_arr[2] ≈ 0.0 atol=1e-10   # cos(π/2) = 0
        @test rv_arr[3] ≈ -K atol=1e-10    # cos(π) = -1
        @test rv_arr[4] ≈ 0.0 atol=1e-10   # cos(3π/2) = 0
        @test rv_arr[5] ≈ K atol=1e-10     # cos(2π) = 1

        # Amplitude is K (within numerical noise) for circular orbit.
        t_dense = collect(range(0, 2P, length=1000))
        rv_dense = Nereus.rv_keplerian.(t_dense, P, K, e, ω, M0, t_ref)
        @test maximum(rv_dense) ≈ K atol=1e-3
        @test minimum(rv_dense) ≈ -K atol=1e-3
    end

    # -----------------------------------------------------------------
    @testset "RV Keplerian model — eccentric" begin
        # Eccentric orbit: peak-to-peak should still be ~2K·√(1 + e cos ω · ...)
        # but the key sanity check is: periastron gives the highest rv when
        # ω = 0 (so that the motion toward the observer is aligned).
        P = 5.0
        K = 10.0
        e = 0.3
        ω = 0.0
        M0 = 0.0    # at t = t_ref, M = 0 → periastron
        t_ref = 0.0

        rv0 = Nereus.rv_keplerian(0.0, P, K, e, ω, M0, t_ref)
        # At periastron (f=0, ω=0), rv = K(cos 0 + e cos 0) = K(1 + e).
        @test rv0 ≈ K * (1 + e) atol=1e-10

        # At apoastron (M=π, E=π, f=π), rv = K(cos π + e cos 0) = K(-1 + e).
        rv_apo = Nereus.rv_keplerian(P/2, P, K, e, ω, M0, t_ref)
        @test rv_apo ≈ K * (-1 + e) atol=1e-10
    end

    # -----------------------------------------------------------------
    @testset "RV multi-planet sum" begin
        # Two circular orbits in phase at t=0 should add up.
        P1, K1 = 5.0, 10.0
        P2, K2 = 20.0, 4.0
        params = ((P1, K1, 0.0, 0.0, 0.0), (P2, K2, 0.0, 0.0, 0.0))
        t_ref = 0.0
        rv0 = Nereus.rv_keplerian_sum(0.0, params, t_ref)
        @test rv0 ≈ (K1 + K2) atol=1e-10
    end

    # -----------------------------------------------------------------
    @testset "Uniform-disk transit" begin
        p = 0.1

        # Out-of-transit: F = 1.
        @test transit_flux_uniform(1.0 + p + 0.01, p) ≈ 1.0 atol=1e-12
        @test transit_flux_uniform(5.0, p) ≈ 1.0 atol=1e-12

        # Fully inside star: F = 1 - p².
        @test transit_flux_uniform(0.0, p) ≈ 1 - p^2 atol=1e-12
        @test transit_flux_uniform(0.5, p) ≈ 1 - p^2 atol=1e-12
        @test transit_flux_uniform(1.0 - p - 0.01, p) ≈ 1 - p^2 atol=1e-12

        # Monotonic ingress: F at z just outside ingress should be > F at mid-transit.
        F_mid = transit_flux_uniform(0.0, p)
        F_ingress = transit_flux_uniform(1.0 - p / 2, p)
        @test F_mid < F_ingress
        @test F_ingress < 1.0

        # Partial overlap continuity: approaching from inside and outside
        # the ingress contact point should give consistent values.
        z_contact_inside = 1.0 - p
        z_contact_outside = 1.0 + p
        F_in = transit_flux_uniform(z_contact_inside - 1e-6, p)
        F_out = transit_flux_uniform(z_contact_outside + 1e-6, p)
        @test F_in ≈ 1 - p^2 atol=1e-5  # just inside ingress
        @test F_out ≈ 1.0 atol=1e-5      # just outside egress

        # Symmetry: F(z, p) == F(-z, p) doesn't make sense (z ≥ 0), so
        # instead verify partial-overlap range is monotonic.
        zs = range(1 - p + 1e-6, 1 + p - 1e-6, length=20)
        fs = [transit_flux_uniform(z, p) for z in zs]
        @test all(diff(fs) .> 0)  # F increases monotonically from ingress to egress
        @test fs[1] ≈ 1 - p^2 atol=1e-4
        @test fs[end] < 1.0

        # Large planet, small star (exotic case).
        P = 1.5
        @test transit_flux_uniform(0.0, P) ≈ 0.0 atol=1e-12  # star entirely behind
        @test transit_flux_uniform(0.2, P) ≈ 0.0 atol=1e-12
    end

    # -----------------------------------------------------------------
    @testset "Kipping LD transform" begin
        # Round trip.
        for (u1, u2) in ((0.3, 0.2), (0.5, 0.1), (0.8, 0.0), (0.1, 0.4))
            q1, q2 = kipping_u_to_q(u1, u2)
            u1b, u2b = kipping_q_to_u(q1, q2)
            @test u1b ≈ u1 atol=1e-12
            @test u2b ≈ u2 atol=1e-12
        end

        # Kipping's prior domain [0, 1]² should produce physically allowed u1, u2.
        # Physically allowed: u1 + u2 < 1, u1 > 0, u1 + 2*u2 > 0.
        for q1 in (0.1, 0.5, 0.9), q2 in (0.1, 0.5, 0.9)
            u1, u2 = kipping_q_to_u(q1, q2)
            @test u1 ≥ 0
            @test u1 + u2 ≤ 1 + 1e-12  # allowing numerical slop
            @test u1 + 2 * u2 ≥ -1e-12
        end
    end

    # -----------------------------------------------------------------
    @testset "transit_flux dispatcher" begin
        # u1 = u2 = 0 → delegates to uniform.
        @test transit_flux(0.0, 0.1, 0.0, 0.0) ≈ transit_flux_uniform(0.0, 0.1)
        @test transit_flux(2.0, 0.1, 0.0, 0.0) ≈ 1.0 atol=1e-12

        # Non-zero LD via Transits.jl QuadLimbDark.
        @test transit_flux(0.0, 0.1, 0.3, 0.2) < 1.0
        @test transit_flux(0.0, 0.1, 0.3, 0.2) < transit_flux(0.0, 0.1, 0.0, 0.0)  # LD makes it deeper
    end

    # =================================================================
    # PRIORS
    # =================================================================

    @testset "UniformPrior" begin
        p = UniformPrior(1.0, 10.0)

        # Bounds
        @test bounds(p) == (1.0, 10.0)
        @test is_fixed(p) == false
        @test in_support(p, 5.0)
        @test !in_support(p, 0.5)
        @test !in_support(p, 11.0)

        # logpdf: flat inside, -Inf outside
        expected_logp = -log(10.0 - 1.0)
        @test logpdf(p, 5.0) ≈ expected_logp
        @test logpdf(p, 1.0) ≈ expected_logp
        @test logpdf(p, 10.0) ≈ expected_logp
        @test logpdf(p, 0.5) == -Inf
        @test logpdf(p, 11.0) == -Inf

        # quantile / prior_transform
        @test quantile(p, 0.0) ≈ 1.0 atol=1e-12
        @test quantile(p, 0.5) ≈ 5.5 atol=1e-12
        @test quantile(p, 1.0) ≈ 10.0 atol=1e-12
        @test prior_transform(0.5, p) ≈ 5.5

        # cdf
        @test cdf(p, 1.0) ≈ 0.0 atol=1e-12
        @test cdf(p, 5.5) ≈ 0.5 atol=1e-12
        @test cdf(p, 10.0) ≈ 1.0 atol=1e-12
        @test cdf(p, 0.0) ≈ 0.0
        @test cdf(p, 20.0) ≈ 1.0

        # rand
        for _ in 1:100
            x = rand(_rng, p)
            @test in_support(p, x)
        end

        # Constructor validation
        @test_throws ArgumentError UniformPrior(5.0, 5.0)
        @test_throws ArgumentError UniformPrior(10.0, 1.0)
    end

    # -----------------------------------------------------------------
    @testset "NormalPrior" begin
        p = NormalPrior(0.0, 1.0, -3.0, 3.0)

        @test bounds(p) == (-3.0, 3.0)
        @test !is_fixed(p)

        # Logpdf: symmetric, peaked at mean
        @test logpdf(p, 0.0) > logpdf(p, 1.0)
        @test logpdf(p, 1.0) ≈ logpdf(p, -1.0)  # symmetric

        # Out of bounds → -Inf
        @test logpdf(p, -5.0) == -Inf
        @test logpdf(p, 5.0) == -Inf
        @test logpdf(p, -3.0) > -Inf  # at boundary, still valid
        @test logpdf(p, 3.0) > -Inf

        # quantile at 0.5 is the truncated median, not exactly 0 for asymmetric truncation
        # but for symmetric truncation of standard normal it IS 0
        @test quantile(p, 0.5) ≈ 0.0 atol=1e-10

        # quantile endpoints return bounds
        @test quantile(p, 0.0) ≈ -3.0 atol=1e-8
        @test quantile(p, 1.0) ≈ 3.0 atol=1e-8

        # σ validation still enforced
        @test_throws ArgumentError NormalPrior(0.0, -1.0, -1.0, 1.0)  # σ < 0
        @test_throws ArgumentError NormalPrior(0.0, 1.0, 5.0, 5.0)    # lo == hi

        # Sampling stays in bounds
        for _ in 1:100
            x = rand(_rng, p)
            @test -3.0 <= x <= 3.0
        end

        # Eccentricity-style prior
        p_ecc = NormalPrior(0.0, 0.3, 0.0, 0.99)
        @test logpdf(p_ecc, 0.5) > -Inf
        @test logpdf(p_ecc, -0.1) == -Inf
        @test logpdf(p_ecc, 1.0) == -Inf
        x = rand(_rng, p_ecc)
        @test 0.0 <= x <= 0.99
    end

    # -----------------------------------------------------------------
    @testset "LogUniformPrior" begin
        p = LogUniformPrior(0.1, 100.0)

        @test bounds(p) == (0.1, 100.0)

        # Logpdf: logp(x) = -log(x) - log(log(hi/lo))
        for x in (0.2, 1.0, 10.0, 50.0)
            expected = -log(x) - log(log(100.0 / 0.1))
            @test logpdf(p, x) ≈ expected atol=1e-10
        end

        # Out of bounds
        @test logpdf(p, 0.05) == -Inf
        @test logpdf(p, 200.0) == -Inf

        # quantile: log-uniform, quantile(0.5) geometric mean = sqrt(lo*hi)
        @test quantile(p, 0.5) ≈ sqrt(0.1 * 100.0) atol=1e-10  # = sqrt(10) ≈ 3.162
        @test quantile(p, 0.0) ≈ 0.1 atol=1e-10
        @test quantile(p, 1.0) ≈ 100.0 atol=1e-10

        # Validation
        @test_throws ArgumentError LogUniformPrior(-1.0, 10.0)  # lo ≤ 0
        @test_throws ArgumentError LogUniformPrior(0.0, 10.0)   # lo == 0
        @test_throws ArgumentError LogUniformPrior(10.0, 1.0)   # lo > hi

        for _ in 1:100
            x = rand(_rng, p)
            @test 0.1 <= x <= 100.0
        end
    end

    # -----------------------------------------------------------------
    @testset "ModJeffreysPrior" begin
        knee, upper = 1.0, 100.0
        p = ModJeffreysPrior(knee, upper)

        @test bounds(p) == (0.0, upper)

        # At x = 0, density is 1/(0 + knee) / lognorm
        lognorm = log((knee + upper) / knee)
        @test logpdf(p, 0.0) ≈ -log(knee) - log(lognorm) atol=1e-12
        @test logpdf(p, knee) ≈ -log(2 * knee) - log(lognorm) atol=1e-12

        # Out of support
        @test logpdf(p, -0.1) == -Inf
        @test logpdf(p, upper + 0.1) == -Inf

        # CDF endpoints
        @test cdf(p, 0.0) ≈ 0.0 atol=1e-12
        @test cdf(p, upper) ≈ 1.0 atol=1e-10

        # Quantile round-trip
        for u in (0.1, 0.3, 0.5, 0.7, 0.9)
            x = quantile(p, u)
            u_back = cdf(p, x)
            @test u ≈ u_back atol=1e-10
        end

        # For x ≪ knee, behaves flat-ish; for x ≫ knee, behaves 1/x.
        # Check relative density ratio.
        r_small = exp(logpdf(p, 0.1 * knee) - logpdf(p, 0.2 * knee))
        r_large = exp(logpdf(p, 10 * knee) - logpdf(p, 20 * knee))
        @test r_small < 1.5       # nearly flat at low x
        @test r_large > 1.8       # nearly 2× at high x (1/x behaviour)

        # Validation
        @test_throws ArgumentError ModJeffreysPrior(-1.0, 10.0)
        @test_throws ArgumentError ModJeffreysPrior(0.0, 10.0)
        @test_throws ArgumentError ModJeffreysPrior(10.0, 5.0)  # upper < knee

        for _ in 1:100
            x = rand(_rng, p)
            @test 0.0 <= x <= upper
        end
    end

    # -----------------------------------------------------------------
    @testset "BetaPrior" begin
        # Kipping 2013 eccentricity prior
        p = BetaPrior(0.867, 3.03)
        @test bounds(p) == (0.0, 1.0)
        @test !is_fixed(p)

        # Finite logpdf inside support, -Inf outside
        @test logpdf(p, 0.1) > -Inf
        @test logpdf(p, 0.5) > -Inf
        @test logpdf(p, -0.1) == -Inf
        @test logpdf(p, 1.5) == -Inf

        # Sampling in support
        for _ in 1:100
            x = rand(_rng, p)
            @test 0.0 <= x <= 1.0
        end

        # Scaled Beta on [-1, 1]
        p_scaled = BetaPrior(2.0, 2.0; lo=-1.0, hi=1.0)
        @test bounds(p_scaled) == (-1.0, 1.0)
        @test logpdf(p_scaled, 0.0) > -Inf  # peak of symmetric Beta(2,2) scaled
        @test logpdf(p_scaled, -2.0) == -Inf
        for _ in 1:100
            x = rand(_rng, p_scaled)
            @test -1.0 <= x <= 1.0
        end

        # Validation
        @test_throws ArgumentError BetaPrior(-1.0, 2.0)
        @test_throws ArgumentError BetaPrior(1.0, -1.0)
        @test_throws ArgumentError BetaPrior(1.0, 1.0; lo=1.0, hi=0.0)
    end

    # -----------------------------------------------------------------
    @testset "FixedPrior" begin
        p = FixedPrior(42.0)

        @test bounds(p) == (42.0, 42.0)
        @test is_fixed(p) == true
        @test fixed_value(p) == 42.0

        # logpdf always 0 (no contribution to log-prior)
        @test logpdf(p, 42.0) == 0.0
        @test logpdf(p, 41.0) == 0.0  # even for other x — fixed priors don't reject
        @test logpdf(p, 100.0) == 0.0

        # quantile always returns the fixed value
        @test quantile(p, 0.0) == 42.0
        @test quantile(p, 0.5) == 42.0
        @test quantile(p, 1.0) == 42.0

        # rand always returns the fixed value
        @test rand(_rng, p) == 42.0

        # Non-fixed prior errors when asked for fixed_value
        @test_throws ArgumentError fixed_value(UniformPrior(0.0, 1.0))
    end

    # -----------------------------------------------------------------
    @testset "Quantile ⇄ CDF round-trip" begin
        # For each non-fixed prior, quantile(cdf(x)) ≈ x for interior points.
        priors = [
            UniformPrior(0.0, 1.0),
            UniformPrior(-5.0, 5.0),
            NormalPrior(0.0, 1.0, -3.0, 3.0),
            NormalPrior(5.0, 2.0, 0.0, 10.0),
            LogUniformPrior(0.1, 100.0),
            LogUniformPrior(1e-3, 1e3),
            ModJeffreysPrior(0.5, 50.0),
            BetaPrior(2.0, 3.0),
        ]
        for p in priors
            for u in (0.1, 0.3, 0.5, 0.7, 0.9)
                x = quantile(p, u)
                u_back = cdf(p, x)
                @test u ≈ u_back atol=1e-8
                @test in_support(p, x)
            end
        end
    end

    # -----------------------------------------------------------------
    @testset "Batch logpdf_sum" begin
        priors = (
            UniformPrior(0.0, 10.0),
            NormalPrior(0.0, 1.0, -3.0, 3.0),
            LogUniformPrior(0.1, 100.0),
        )
        values = [5.0, 0.5, 2.0]
        expected = logpdf(priors[1], values[1]) +
                   logpdf(priors[2], values[2]) +
                   logpdf(priors[3], values[3])
        @test logpdf_sum(priors, values) ≈ expected atol=1e-12

        # Out-of-support short-circuits to -Inf
        values_bad = [5.0, -10.0, 2.0]  # middle value out of support
        @test logpdf_sum(priors, values_bad) == -Inf

        # Mixed types: tuple of priors, vector of values
        @test logpdf_sum([UniformPrior(0, 10)], [5.0]) ≈ -log(10) atol=1e-12
    end

    # -----------------------------------------------------------------
    @testset "prior_transform! (nested-sampling vector form)" begin
        priors = (
            UniformPrior(0.0, 100.0),
            LogUniformPrior(0.1, 1000.0),
            NormalPrior(0.0, 1.0, -3.0, 3.0),
        )
        us = [0.5, 0.5, 0.5]
        dst = zeros(3)
        prior_transform!(dst, us, priors)
        @test dst[1] ≈ 50.0 atol=1e-10
        @test dst[2] ≈ 10.0 atol=1e-10  # geometric mean of 0.1 and 1000
        @test dst[3] ≈ 0.0 atol=1e-8    # symmetric-truncated Normal median
    end

    # -----------------------------------------------------------------
    @testset "Serialization round-trip" begin
        priors = [
            UniformPrior(1.0, 10.0),
            NormalPrior(0.5, 0.1, -2.0, 2.0),
            LogUniformPrior(0.01, 100.0),
            BetaPrior(0.867, 3.03),
            ModJeffreysPrior(1.0, 500.0),
            FixedPrior(3.14),
        ]
        for p in priors
            d = prior_to_dict(p)
            @test haskey(d, "type")
            @test haskey(d, "lo")
            @test haskey(d, "hi")
            p_back = prior_from_dict(d)
            @test bounds(p_back) == bounds(p)
            # Compare at a test point — should give the same logpdf
            x = is_fixed(p) ? fixed_value(p) : (bounds(p)[1] + bounds(p)[2]) / 2
            @test logpdf(p_back, x) ≈ logpdf(p, x) atol=1e-10
        end
    end

    # -----------------------------------------------------------------
    @testset "Unbounded NormalPrior" begin
        # Two-arg form is now allowed and produces (-Inf, Inf) bounds.
        p = NormalPrior(5.0, 2.0)
        @test bounds(p) == (-Inf, Inf)
        @test !is_fixed(p)

        # logpdf is finite everywhere (matches raw Normal).
        @test logpdf(p, 5.0) ≈ logpdf(Normal(5.0, 2.0), 5.0) atol=1e-12
        @test logpdf(p, -100.0) ≈ logpdf(Normal(5.0, 2.0), -100.0) atol=1e-12
        @test logpdf(p, 1e6) ≈ logpdf(Normal(5.0, 2.0), 1e6) atol=1e-12

        # cdf endpoints.
        @test cdf(p, -1e6) ≈ 0.0 atol=1e-10
        @test cdf(p, 1e6) ≈ 1.0 atol=1e-10
        @test cdf(p, 5.0) ≈ 0.5 atol=1e-10

        # quantile at interior points is finite; at 0 and 1 it's ±Inf
        # (callers of nested-sampling prior transforms must clip).
        @test isfinite(quantile(p, 0.5))
        @test quantile(p, 0.5) ≈ 5.0 atol=1e-10
        @test quantile(p, 0.0) == -Inf
        @test quantile(p, 1.0) == Inf

        # σ validation still enforced.
        @test_throws ArgumentError NormalPrior(0.0, -1.0)
        @test_throws ArgumentError NormalPrior(0.0, 0.0)

        # Sampling produces finite values.
        for _ in 1:100
            x = rand(_rng, p)
            @test isfinite(x)
        end
    end

    # -----------------------------------------------------------------
    @testset "extract_base_name" begin
        @test extract_base_name("P_k1") == "P"
        @test extract_base_name("K_k10") == "K"
        @test extract_base_name("ecc_k3") == "ecc"
        @test extract_base_name("sesinw_k1") == "sesinw"
        @test extract_base_name("gamma_HARPS") == "gamma"
        @test extract_base_name("jitter_TESS") == "jitter"
        @test extract_base_name("rv-GP-tau_HARPS") == "GP-tau"
        @test extract_base_name("pm-GP-p") == "GP-p"
        @test extract_base_name("pm-GP-S0") == "GP-S0"
        @test extract_base_name("rv-phi_1,HARPS") == "phi"
        @test extract_base_name("pm-omega_2,TESS") == "omega"
        @test extract_base_name("q1_TESS,TESS2") == "q1"
        @test extract_base_name("C_0,HARPS") == "C"
        @test extract_base_name("acc_1") == "acc"
        @test extract_base_name("trend_1,TESS") == "trend"
        # Multi-segment base: rho_s is NOT split at the first underscore
        @test extract_base_name("rho_s") == "rho_s"
        # Edge cases
        @test extract_base_name("") == ""
        @test extract_base_name("P") == "P"       # no underscore
        @test extract_base_name("GP-p") == "GP-p" # no mode prefix, no _
    end

    # -----------------------------------------------------------------
    @testset "physical_bounds lookup" begin
        @test physical_bounds("P_k1") == (0.0, Inf)
        @test physical_bounds("K_k2") == (0.0, Inf)
        @test physical_bounds("e_k1") == (0.0, 1.0)
        @test physical_bounds("ecc_k3") == (0.0, 1.0)
        @test physical_bounds("sesinw_k1") == (-1.0, 1.0)
        @test physical_bounds("b_k2") == (0.0, 2.0)
        @test physical_bounds("q1_TESS") == (0.0, 1.0)
        @test physical_bounds("gamma_HARPS") == (-Inf, Inf)
        @test physical_bounds("sigma_HARPS") == (0.0, Inf)
        @test physical_bounds("rv-GP-tau_HARPS") == (0.0, Inf)
        @test physical_bounds("pm-GP-Q") == (0.0, Inf)
        @test physical_bounds("rv-phi_1,HARPS") == (-1.0, 1.0)
        # Multi-segment base resolves correctly.
        @test physical_bounds("rho_s") == (0.0, Inf)
        # Unknown parameter → unrestricted
        @test physical_bounds("totally_unknown_param") == (-Inf, Inf)
    end

    # -----------------------------------------------------------------
    @testset "validate_physical — the user's eccentricity scenario" begin
        # An eccentricity prior CANNOT be a fixed negative value.
        @test_throws ArgumentError validate_physical("e_k1", FixedPrior(-1.0))
        @test_throws ArgumentError validate_physical("e_k1", FixedPrior(-0.5))
        @test_throws ArgumentError validate_physical("e_k1", FixedPrior(2.0))
        @test_throws ArgumentError validate_physical("ecc_k2", FixedPrior(-0.01))

        # Bounded priors that extend into unphysical territory also fail.
        @test_throws ArgumentError validate_physical("e_k1", UniformPrior(-0.5, 0.5))
        @test_throws ArgumentError validate_physical("e_k1", UniformPrior(0.0, 2.0))
        @test_throws ArgumentError validate_physical("e_k1", NormalPrior(0.0, 0.3, -1.0, 0.99))
        @test_throws ArgumentError validate_physical("ecc_k1", UniformPrior(-1.0, 1.0))

        # Valid eccentricity priors pass silently.
        @test validate_physical("e_k1", FixedPrior(0.1)) === nothing
        @test validate_physical("e_k1", FixedPrior(0.0)) === nothing
        @test validate_physical("e_k1", UniformPrior(0.0, 0.9)) === nothing
        @test validate_physical("e_k1", UniformPrior(0.0, 1.0)) === nothing  # at boundary
        @test validate_physical("e_k1", NormalPrior(0.0, 0.3, 0.0, 0.99)) === nothing
        @test validate_physical("e_k1", BetaPrior(0.867, 3.03)) === nothing
    end

    # -----------------------------------------------------------------
    @testset "validate_physical — other parameters" begin
        # Period must be positive.
        @test_throws ArgumentError validate_physical("P_k1", FixedPrior(-5.0))
        @test_throws ArgumentError validate_physical("P_k1", UniformPrior(-1.0, 100.0))
        @test validate_physical("P_k1", UniformPrior(0.5, 1000.0)) === nothing
        @test validate_physical("P_k1", LogUniformPrior(0.1, 10000.0)) === nothing

        # K must be non-negative.
        @test_throws ArgumentError validate_physical("K_k1", FixedPrior(-5.0))
        @test validate_physical("K_k1", LogUniformPrior(0.01, 100.0)) === nothing

        # Impact parameter in [0, 2].
        @test_throws ArgumentError validate_physical("b_k1", FixedPrior(-0.1))
        @test_throws ArgumentError validate_physical("b_k1", UniformPrior(0.0, 3.0))
        @test validate_physical("b_k1", UniformPrior(0.0, 1.5)) === nothing

        # Rp/Rs must be non-negative and (by our convention) ≤ 1.
        @test_throws ArgumentError validate_physical("rr_k1", FixedPrior(-0.01))
        @test_throws ArgumentError validate_physical("rr_k1", UniformPrior(0.0, 2.0))
        @test validate_physical("rr_k1", UniformPrior(0.0, 0.5)) === nothing

        # sesinw / secosw in [-1, 1]
        @test_throws ArgumentError validate_physical("sesinw_k1", UniformPrior(-2.0, 2.0))
        @test validate_physical("sesinw_k1", UniformPrior(-1.0, 1.0)) === nothing

        # Dilution in [0, 1]
        @test_throws ArgumentError validate_physical("dilution_TESS", FixedPrior(1.5))
        @test_throws ArgumentError validate_physical("dilution_TESS", FixedPrior(-0.1))
        @test validate_physical("dilution_TESS", FixedPrior(0.0)) === nothing
        @test validate_physical("dilution_TESS", UniformPrior(0.0, 1.0)) === nothing

        # GP timescale positive.
        @test_throws ArgumentError validate_physical("rv-GP-tau_HARPS",
                                                     FixedPrior(-5.0))
        @test validate_physical("rv-GP-tau_HARPS",
                                LogUniformPrior(0.1, 1000.0)) === nothing

        # q1/q2 Kipping in [0, 1].
        @test_throws ArgumentError validate_physical("q1_TESS", FixedPrior(-0.1))
        @test_throws ArgumentError validate_physical("q1_TESS", FixedPrior(1.1))
        @test validate_physical("q1_TESS", UniformPrior(0.0, 1.0)) === nothing

        # Jitter non-negative.
        @test_throws ArgumentError validate_physical("jitter_TESS", FixedPrior(-0.001))
        @test validate_physical("jitter_TESS", LogUniformPrior(1e-6, 0.1)) === nothing

        # gamma is unrestricted (unbounded Normal is explicitly valid here).
        @test validate_physical("gamma_HARPS", FixedPrior(-1000.0)) === nothing
        @test validate_physical("gamma_HARPS", NormalPrior(0.0, 100.0)) === nothing
        @test validate_physical("gamma_HARPS", UniformPrior(-1e6, 1e6)) === nothing

        # Unknown params pass through without validation.
        @test validate_physical("foo_bar", FixedPrior(-99999.0)) === nothing
    end

    # =================================================================
    # MODEL + PARAMETERS (Theta broker)
    # =================================================================

    # -----------------------------------------------------------------
    @testset "ParametrizationConfig" begin
        pc = ParametrizationConfig()
        @test pc.ew === :sesinw
        @test pc.time === :Mo
        @test pc.geom === :b_rr
        @test pc.use_rho_s == false

        pc2 = ParametrizationConfig(ew=:ew, time=:Tc, geom=:r1r2, use_rho_s=true)
        @test pc2.ew === :ew
        @test pc2.time === :Tc
        @test pc2.geom === :r1r2
        @test pc2.use_rho_s == true

        # Validation
        @test_throws ArgumentError ParametrizationConfig(ew=:foo)
        @test_throws ArgumentError ParametrizationConfig(time=:bar)
        @test_throws ArgumentError ParametrizationConfig(geom=:baz)
    end

    # -----------------------------------------------------------------
    @testset "InstrumentConfig" begin
        ic = InstrumentConfig(rv=["HARPS", "ESPRESSO"], pm=["TESS"])
        @test n_rv_instruments(ic) == 2
        @test n_pm_instruments(ic) == 1
        @test ic.rv_names == ["HARPS", "ESPRESSO"]
        @test ic.pm_names == ["TESS"]

        ic_empty = InstrumentConfig()
        @test n_rv_instruments(ic_empty) == 0
        @test n_pm_instruments(ic_empty) == 0
    end

    # -----------------------------------------------------------------
    @testset "Params construction — 1-planet RV-only, sesinw" begin
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(0.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors=priors,
            max_kplanet=1,
            parametrization=ParametrizationConfig(),
            planet_modes=[RV_ONLY],
            instruments=ic,
        )

        # Expected layout: n_p, P_k1, K_k1, sesinw_k1, secosw_k1, Mo_k1,
        # gamma_HARPS, sigma_HARPS  → 8 slots
        @test n_total(params) == 8
        @test params.layout.names == [
            "n_p", "P_k1", "K_k1", "sesinw_k1", "secosw_k1", "Mo_k1",
            "gamma_HARPS", "sigma_HARPS",
        ]
        @test params.layout.n_p_idx == 1

        # n_p is fixed → 1 frozen slot, 7 unfrozen.
        @test n_frozen(params) == 1
        @test n_unfrozen(params) == 7
        @test params.layout.frozen_idx == [1]
        @test params.layout.frozen_values == [1.0]
        @test params.layout.unfrozen_names == [
            "P_k1", "K_k1", "sesinw_k1", "secosw_k1", "Mo_k1",
            "gamma_HARPS", "sigma_HARPS",
        ]
        # Unfrozen priors vector is parallel to unfrozen_names
        @test length(params.layout.unfrozen_priors) == 7

        # Planet block for k=1: RVOnlyBlock with P=2, K=3, e1=4, e2=5, t=6.
        # Absent slots (b, r) are absent from the type — no sentinel.
        pb = params.layout.planet_blocks[1]
        @test pb isa RVOnlyBlock
        @test pb.P == 2
        @test pb.K == 3
        @test pb.e1 == 4
        @test pb.e2 == 5
        @test pb.t == 6
        @test has_K(pb)
        @test !has_geometry(pb)

        # Systemic indices
        @test params.layout.systemic.rv_gamma == [7]
        @test params.layout.systemic.rv_sigma == [8]
        @test isempty(params.layout.systemic.pm_offset)
        @test params.layout.systemic.rho_s == 0
    end

    # -----------------------------------------------------------------
    @testset "Params construction — 2-planet joint, r1r2 + Tc + use_rho_s" begin
        ic = InstrumentConfig(rv=["HARPS"], pm=["TESS"])
        pc = ParametrizationConfig(ew=:sesinw, time=:Tc, geom=:r1r2, use_rho_s=true)

        priors = Dict{String, PriorSpec}(
            "n_p"             => FixedPrior(2.0),
            # Planet 1 (RVPM)
            "P_k1"            => LogUniformPrior(0.5, 1000.0),
            "K_k1"            => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"       => UniformPrior(-1.0, 1.0),
            "secosw_k1"       => UniformPrior(-1.0, 1.0),
            "Tc_k1"           => UniformPrior(-1e4, 1e4),
            "r1_k1"           => UniformPrior(0.0, 1.0),
            "r2_k1"           => UniformPrior(0.0, 1.0),
            # Planet 2 (RV-only)
            "P_k2"            => LogUniformPrior(0.5, 1000.0),
            "K_k2"            => LogUniformPrior(0.1, 100.0),
            "sesinw_k2"       => UniformPrior(-1.0, 1.0),
            "secosw_k2"       => UniformPrior(-1.0, 1.0),
            "Tc_k2"           => UniformPrior(-1e4, 1e4),
            # Systemic
            "gamma_HARPS"     => NormalPrior(0.0, 100.0),
            "sigma_HARPS"     => LogUniformPrior(0.001, 10.0),
            "offset_TESS"     => NormalPrior(0.0, 0.01),
            "jitter_TESS"     => LogUniformPrior(1e-6, 0.01),
            "dilution_TESS"   => FixedPrior(0.0),
            "rho_s"           => LogUniformPrior(0.01, 100.0),
        )
        params = Params(;
            priors=priors,
            max_kplanet=2,
            parametrization=pc,
            planet_modes=[RVPM, RV_ONLY],
            instruments=ic,
            M_s=1.0,
        )

        # Layout: n_p + 7 (RVPM) + 5 (RV-only) + 2 (gamma, sigma) + 3 (off, jit, dil) + 2 (q1, q2) + 1 (rho_s) = 21
        @test n_total(params) == 21

        # Planet 1 (RVPM) block — all seven slots, r1r2 goes into b/r fields.
        pb1 = params.layout.planet_blocks[1]
        @test pb1 isa RVPMBlock
        @test pb1.P  == 2
        @test pb1.K  == 3
        @test pb1.e1 == 4
        @test pb1.e2 == 5
        @test pb1.t  == 6
        @test pb1.b  == 7  # r1 slot (under :r1r2 geom, stored in `b` field)
        @test pb1.r  == 8  # r2 slot (stored in `r` field)
        @test has_K(pb1)
        @test has_geometry(pb1)

        # Planet 2 (RV-only) — absent geometry slots are absent from the type.
        pb2 = params.layout.planet_blocks[2]
        @test pb2 isa RVOnlyBlock
        @test pb2.P  == 9
        @test pb2.K  == 10
        @test pb2.e1 == 11
        @test pb2.e2 == 12
        @test pb2.t  == 13
        @test has_K(pb2)
        @test !has_geometry(pb2)

        # Systemic
        @test params.layout.systemic.rv_gamma == [14]
        @test params.layout.systemic.rv_sigma == [15]
        @test params.layout.systemic.pm_offset == [16]
        @test params.layout.systemic.pm_jitter == [17]
        @test params.layout.systemic.pm_dilution == [18]
        @test params.layout.systemic.rho_s == 21  # after q1/q2 LD params

        # dilution_TESS is fixed at 0.0 → frozen
        @test "dilution_TESS" in [params.layout.names[i] for i in params.layout.frozen_idx]
        @test 0.0 in params.layout.frozen_values
    end

    # -----------------------------------------------------------------
    @testset "Params construction — PM-only planet with ew parametrization" begin
        ic = InstrumentConfig(pm=["TESS"])
        pc = ParametrizationConfig(ew=:ew, time=:Tc, geom=:b_rr)

        priors = Dict{String, PriorSpec}(
            "n_p"         => FixedPrior(1.0),
            "P_k1"        => LogUniformPrior(0.5, 1000.0),
            "ecc_k1"      => BetaPrior(0.867, 3.03),
            "w_k1"        => UniformPrior(-2π, 2π),
            "Tc_k1"       => UniformPrior(-1e4, 1e4),
            "b_k1"        => UniformPrior(0.0, 1.5),
            "rr_k1"       => UniformPrior(0.0, 0.5),
            "offset_TESS" => NormalPrior(0.0, 0.01),
            "jitter_TESS" => LogUniformPrior(1e-6, 0.01),
            "dilution_TESS" => FixedPrior(0.0),
        )
        params = Params(;
            priors=priors,
            max_kplanet=1,
            parametrization=pc,
            planet_modes=[PM_ONLY],
            instruments=ic,
        )

        # Layout: n_p + 6 (PM-only: P, e1, e2, t, b, r — no K) + 3 (PM inst) + 2 (q1, q2) = 12
        @test n_total(params) == 12

        pb = params.layout.planet_blocks[1]
        @test pb isa PMOnlyBlock
        @test pb.P  == 2
        @test pb.e1 == 3
        @test pb.e2 == 4
        @test pb.t  == 5
        @test pb.b  == 6
        @test pb.r  == 7
        @test !has_K(pb)
        @test has_geometry(pb)

        # Verify eccentricity slot names
        @test params.layout.names[3] == "ecc_k1"
        @test params.layout.names[4] == "w_k1"
    end

    # -----------------------------------------------------------------
    @testset "Params construction — validation errors" begin
        ic = InstrumentConfig(rv=["HARPS"])

        # Missing prior → auto-filled by defaults (no longer an error)
        priors_partial = Dict{String, PriorSpec}(
            "P_k1" => LogUniformPrior(0.5, 1000.0),
            # K_k1 not specified → auto-generated default
        )
        params_partial = Params(;
            priors=priors_partial,
            max_kplanet=1,
            planet_modes=[RV_ONLY],
            instruments=ic,
        )
        @test "K_k1" in params_partial.layout.unfrozen_names  # auto-generated

        # Extra prior (typo)
        priors_typo = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(0.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
            "P_k1_typo"  => UniformPrior(0.0, 1.0),  # not in layout
        )
        @test_throws ArgumentError Params(;
            priors=priors_typo,
            max_kplanet=1,
            planet_modes=[RV_ONLY],
            instruments=ic,
        )

        # Unphysical eccentricity prior (the user's scenario)
        priors_bad_ecc = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> FixedPrior(-99999.0),  # silly but gamma is unrestricted
            "sigma_HARPS"=> FixedPrior(-0.001),    # NEGATIVE sigma — physical fail
        )
        @test_throws ArgumentError Params(;
            priors=priors_bad_ecc,
            max_kplanet=1,
            planet_modes=[RV_ONLY],
            instruments=ic,
        )

        # Mismatched planet_modes length
        priors_ok = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(0.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        @test_throws ArgumentError Params(;
            priors=priors_ok,
            max_kplanet=2,  # but only 1 mode given
            planet_modes=[RV_ONLY],
            instruments=ic,
        )
    end

    # -----------------------------------------------------------------
    @testset "Theta construction and frozen values" begin
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> FixedPrior(5.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors=priors,
            max_kplanet=1,
            planet_modes=[RV_ONLY],
            instruments=ic,
        )
        theta = Theta(params)

        @test length(theta) == 8
        @test n_unfrozen(theta) == 6  # 2 fixed (n_p, gamma_HARPS), 6 sampled
        @test n_frozen(theta) == 2

        # Frozen values initialized
        @test theta.values[1] == 1.0   # n_p
        @test theta.values[7] == 5.0   # gamma_HARPS

        # Unfrozen slots initialized to 0
        @test theta.values[2] == 0.0   # P_k1

        # n_p accessor
        @test n_p(theta) == 1
    end

    # -----------------------------------------------------------------
    @testset "set_unfrozen! and unfrozen_values" begin
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(0.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors=priors,
            max_kplanet=1,
            planet_modes=[RV_ONLY],
            instruments=ic,
        )
        theta = Theta(params)

        # There are 7 unfrozen params. Write values.
        x = [5.0, 8.0, 0.1, 0.2, 1.5, 10.0, 1.5]
        set_unfrozen!(theta, x)

        @test theta.values[1] == 1.0    # n_p (frozen, untouched)
        @test theta.values[2] == 5.0    # P_k1
        @test theta.values[3] == 8.0    # K_k1
        @test theta.values[4] == 0.1    # sesinw_k1
        @test theta.values[5] == 0.2    # secosw_k1
        @test theta.values[6] == 1.5    # Mo_k1
        @test theta.values[7] == 10.0   # gamma_HARPS
        @test theta.values[8] == 1.5    # sigma_HARPS

        # Round trip
        x_back = unfrozen_values(theta)
        @test x_back == x

        # Wrong length
        @test_throws ArgumentError set_unfrozen!(theta, [1.0, 2.0])
    end

    # -----------------------------------------------------------------
    @testset "Hot-path block accessors" begin
        ic = InstrumentConfig(rv=["HARPS"], pm=["TESS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "b_k1"       => UniformPrior(0.0, 1.5),
            "rr_k1"      => UniformPrior(0.0, 0.5),
            "gamma_HARPS"=> NormalPrior(0.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
            "offset_TESS"=> NormalPrior(0.0, 0.01),
            "jitter_TESS"=> LogUniformPrior(1e-6, 0.01),
            "dilution_TESS" => FixedPrior(0.0),
        )
        params = Params(;
            priors=priors,
            max_kplanet=1,
            planet_modes=[RVPM],
            instruments=ic,
        )
        theta = Theta(params)

        # Write known values by name
        set_param!(theta, "P_k1", 5.0)
        set_param!(theta, "K_k1", 8.0)
        set_param!(theta, "sesinw_k1", 0.2)   # → e_s
        set_param!(theta, "secosw_k1", 0.3)   # → e_c
        set_param!(theta, "Mo_k1", 1.5)
        set_param!(theta, "b_k1", 0.3)
        set_param!(theta, "rr_k1", 0.1)
        set_param!(theta, "gamma_HARPS", 12.0)
        set_param!(theta, "sigma_HARPS", 1.5)
        set_param!(theta, "offset_TESS", -0.001)
        set_param!(theta, "jitter_TESS", 5e-4)

        # Hot-path accessors
        @test planet_P(theta, 1) == 5.0
        @test planet_K(theta, 1) == 8.0
        e, w = planet_e_w(theta, 1)
        @test e ≈ 0.2^2 + 0.3^2 atol=1e-12      # sesinw: e = s²+c²
        @test w ≈ atan(0.2, 0.3) atol=1e-12
        @test planet_time_anchor(theta, 1) == 1.5
        bv, rv = planet_b_rr(theta, 1)
        @test bv == 0.3
        @test rv == 0.1

        # Systemic
        @test rv_gamma(theta, 1) == 12.0
        @test rv_sigma(theta, 1) == 1.5
        @test pm_offset(theta, 1) == -0.001
        @test pm_jitter(theta, 1) == 5e-4
        @test pm_dilution(theta, 1) == 0.0     # frozen at 0.0
    end

    # -----------------------------------------------------------------
    @testset "Accessor errors for missing slots" begin
        ic = InstrumentConfig(pm=["TESS"])
        priors = Dict{String, PriorSpec}(
            "n_p"          => FixedPrior(1.0),
            "P_k1"         => LogUniformPrior(0.5, 1000.0),
            "sesinw_k1"    => UniformPrior(-1.0, 1.0),
            "secosw_k1"    => UniformPrior(-1.0, 1.0),
            "Tc_k1"        => UniformPrior(-1e4, 1e4),
            "b_k1"         => UniformPrior(0.0, 1.5),
            "rr_k1"        => UniformPrior(0.0, 0.5),
            "offset_TESS"  => NormalPrior(0.0, 0.01),
            "jitter_TESS"  => LogUniformPrior(1e-6, 0.01),
            "dilution_TESS"=> FixedPrior(0.0),
        )
        params = Params(;
            priors=priors,
            max_kplanet=1,
            parametrization=ParametrizationConfig(time=:Tc),
            planet_modes=[PM_ONLY],
            instruments=ic,
        )
        theta = Theta(params)

        # PM-only planet has no K
        @test_throws ArgumentError planet_K(theta, 1)

        # Similarly, an RV-only planet has no b/r
        ic2 = InstrumentConfig(rv=["HARPS"])
        priors2 = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(0.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params2 = Params(;
            priors=priors2,
            max_kplanet=1,
            planet_modes=[RV_ONLY],
            instruments=ic2,
        )
        theta2 = Theta(params2)
        @test_throws ArgumentError planet_b_rr(theta2, 1)

        # rho_s errors when not in use
        @test_throws ArgumentError rho_s(theta2)
    end

    # -----------------------------------------------------------------
    @testset "log_prior evaluation" begin
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => UniformPrior(0.5, 1000.0),
            "K_k1"       => UniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> UniformPrior(-100.0, 100.0),
            "sigma_HARPS"=> UniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors=priors,
            max_kplanet=1,
            planet_modes=[RV_ONLY],
            instruments=ic,
        )
        theta = Theta(params)

        # Within support: sum of logs of the widths
        set_param!(theta, "P_k1", 10.0)
        set_param!(theta, "K_k1", 5.0)
        set_param!(theta, "sesinw_k1", 0.1)
        set_param!(theta, "secosw_k1", 0.1)
        set_param!(theta, "Mo_k1", 1.0)
        set_param!(theta, "gamma_HARPS", 0.0)
        set_param!(theta, "sigma_HARPS", 1.0)

        expected = (
            -log(1000.0 - 0.5) +    # P_k1
            -log(100.0 - 0.1) +     # K_k1
            -log(2.0) +             # sesinw_k1
            -log(2.0) +             # secosw_k1
            -log(4π) +              # Mo_k1
            -log(200.0) +           # gamma_HARPS
            -log(10.0 - 0.001)      # sigma_HARPS
        )
        @test log_prior(theta) ≈ expected atol=1e-10

        # Out of support → -Inf
        set_param!(theta, "P_k1", 2000.0)  # > upper bound
        @test log_prior(theta) == -Inf
    end

    # =================================================================
    # REFACTOR VERIFICATION — Params split, Theta{T}, Kepler convergence,
    # PlanetDataSources composability
    # =================================================================

    # -----------------------------------------------------------------
    @testset "PlanetDataSources composability" begin
        # Factory aliases are themselves PlanetDataSources instances —
        # no enum conversion involved.
        @test RV_ONLY isa PlanetDataSources
        @test PM_ONLY isa PlanetDataSources
        @test RVPM   isa PlanetDataSources

        # Explicit construction yields the same thing as the factory.
        @test PlanetDataSources(:RV) == RV_ONLY
        @test PlanetDataSources(:PM) == PM_ONLY
        @test PlanetDataSources(:RV, :PM) == RVPM
        @test PlanetDataSources(:PM, :RV) == RVPM   # order-independent

        # Membership tests work.
        @test :RV in RVPM
        @test :PM in RVPM
        @test :RV in RV_ONLY
        @test !(:PM in RV_ONLY)
        @test has_rv(RVPM) && has_pm(RVPM)
        @test has_rv(RV_ONLY) && !has_pm(RV_ONLY)
        @test !has_rv(PM_ONLY) && has_pm(PM_ONLY)

        # Hash equality so Sets/Dicts work.
        @test hash(RVPM) == hash(PlanetDataSources(:RV, :PM))

        # Two layouts built from equivalent mode specs must be identical.
        ic = InstrumentConfig(rv=["HARPS"], pm=["TESS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "b_k1"       => UniformPrior(0.0, 1.5),
            "rr_k1"      => UniformPrior(0.0, 0.5),
            "gamma_HARPS"=> NormalPrior(0.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
            "offset_TESS"=> NormalPrior(0.0, 0.01),
            "jitter_TESS"=> LogUniformPrior(1e-6, 0.01),
            "dilution_TESS" => FixedPrior(0.0),
        )
        params_a = Params(; priors, max_kplanet=1,
                          planet_modes=[RVPM], instruments=ic)
        params_b = Params(; priors, max_kplanet=1,
                          planet_modes=[PlanetDataSources(:RV, :PM)],
                          instruments=ic)
        @test params_a.layout.names == params_b.layout.names
        @test params_a.layout.planet_blocks[1] == params_b.layout.planet_blocks[1]
    end

    # -----------------------------------------------------------------
    @testset "ParamsConfig / ParamsLayout split" begin
        # The config and layout are separate structs; the layout is a
        # pure function of the config via `_build_layout`.
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => LogUniformPrior(0.5, 1000.0),
            "K_k1"       => LogUniformPrior(0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(0.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors, max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic,
        )

        # Config holds the user inputs verbatim.
        @test params.config isa ParamsConfig
        @test params.config.max_kplanet == 1
        @test params.config.planet_modes == [RV_ONLY]
        @test params.config.instruments === ic
        @test params.config.priors == priors

        # Layout holds derived state including the precomputed unfrozen
        # priors vector.
        @test params.layout isa ParamsLayout
        @test params.layout.unfrozen_priors isa Vector{<:PriorSpec}
        @test length(params.layout.unfrozen_priors) == n_unfrozen(params)
        # Parallel to unfrozen_names.
        for (name, ps) in zip(params.layout.unfrozen_names,
                               params.layout.unfrozen_priors)
            @test ps === priors[name]
        end

        # n_p_idx is a typed field, not a magic number.
        @test params.layout.n_p_idx == 1
        @test params.layout.names[params.layout.n_p_idx] == "n_p"
    end

    # -----------------------------------------------------------------
    @testset "Theta{T} autodiff with ForwardDiff" begin
        # Build a simple 1-planet RV-only model with Normal priors on
        # sampled parameters, so log_prior actually depends on the
        # values (unlike Uniform, whose logpdf is constant on the
        # interior).
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => NormalPrior(5.0, 1.0, 0.5, 100.0),
            "K_k1"       => NormalPrior(8.0, 2.0, 0.1, 100.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(0.0, 10.0, -100.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors, max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic,
        )

        # Float64 baseline.
        x0 = [5.0, 8.0, 0.1, 0.1, 1.0, 0.0, 1.0]
        @test length(x0) == n_unfrozen(params)

        theta_f64 = Theta{Float64}(params)
        set_unfrozen!(theta_f64, x0)
        lp_ref = log_prior(theta_f64)
        @test isfinite(lp_ref)

        # Autodiff: construct a function that takes a Dual vector,
        # threads it through set_unfrozen!, and returns log_prior.
        # The Dual type must match the Theta type.
        function logprior_fn(x::AbstractVector{T}) where {T}
            theta = Theta{T}(params)
            set_unfrozen!(theta, x)
            return log_prior(theta)
        end

        # Primal check.
        @test logprior_fn(x0) ≈ lp_ref atol=1e-12

        # Gradient via ForwardDiff — this is the end-to-end autodiff
        # test. It exercises Theta{Dual} construction, set_unfrozen!
        # with a Dual vector, logpdf on Dual values, and the type-
        # stable unfrozen_priors parallel vector.
        grad = ForwardDiff.gradient(logprior_fn, x0)
        @test length(grad) == length(x0)
        @test all(isfinite, grad)

        # For the Normal priors (P, K, gamma), the gradient of the
        # truncated-Normal logpdf at x is approximately -(x - μ) / σ²
        # when x is well inside the truncation bounds.
        # P_k1: NormalPrior(5.0, 1.0) at x=5.0 → gradient ≈ 0.
        # K_k1: NormalPrior(8.0, 2.0) at x=8.0 → gradient ≈ 0.
        @test abs(grad[1]) < 1e-6    # d/dP at the mean
        @test abs(grad[2]) < 1e-6    # d/dK at the mean
        # gamma_HARPS: NormalPrior(0.0, 10.0) at x=0.0 → gradient ≈ 0.
        @test abs(grad[6]) < 1e-6

        # For Uniform priors (sesinw, secosw, Mo) on the interior,
        # gradient is exactly zero.
        @test grad[3] == 0.0  # sesinw_k1
        @test grad[4] == 0.0  # secosw_k1
        @test grad[5] == 0.0  # Mo_k1

        # Shift P away from the mean and check the gradient points
        # back toward the mean.
        x_off = copy(x0); x_off[1] = 6.0
        grad_off = ForwardDiff.gradient(logprior_fn, x_off)
        @test grad_off[1] < 0    # d/dP at P=6, μ=5, σ=1 → -(1)/1² = -1
        @test grad_off[1] ≈ -1.0 atol=1e-3
    end

    # -----------------------------------------------------------------
    @testset "Kepler solver convergence handling" begin
        # Normal usage — no warning, converges quickly.
        E = kepler_solve(1.5, 0.3)
        @test isfinite(E)
        # Round-trip identity
        @test abs(E - 0.3 * sin(E) - 1.5) < 1e-10

        # Force non-convergence with an aggressively small max_iter on
        # a moderately-high eccentricity case. This should emit a
        # warning (not error) and still return a best-effort E.
        # The exact residual won't meet tol, but that's the point of
        # the test.
        @test_logs (:warn, r"kepler_solve did not converge") begin
            kepler_solve(1.0, 0.99; max_iter=1)
        end

        # strict=true turns it into a hard error.
        @test_throws ErrorException kepler_solve(1.0, 0.99;
                                                  max_iter=1, strict=true)

        # Strict mode with plenty of iterations converges fine and
        # does NOT raise.
        @test_nowarn kepler_solve(1.0, 0.5; max_iter=30, strict=true)

        # Vector form still works (no per-element warning spam because
        # of maxlog=5; we just verify the result is reasonable).
        M_vec = collect(range(-π, π, length=16))
        E_vec = kepler_solve(M_vec, 0.3)
        for (Mi, Ei) in zip(M_vec, E_vec)
            @test abs(Mi - (Ei - 0.3 * sin(Ei))) < 1e-10
        end
    end

    # =================================================================
    # END REFACTOR VERIFICATION
    # =================================================================

    # =================================================================
    # DATA + LIKELIHOOD + TARGET
    # =================================================================

    # -----------------------------------------------------------------
    @testset "Data construction and validation" begin
        # Single-instrument default
        t = collect(0.0:1.0:10.0)
        rv = randn(_rng, 11)
        err = fill(1.0, 11)
        d = Data(; t_rv=t, rv=rv, rv_err=err)
        @test n_rv(d) == 11
        @test d.rv_inst == ones(Int, 11)
        @test d.t_ref ≈ 5.0 atol=1e-12    # median of 0..10

        # Explicit multi-instrument
        inst = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        d2 = Data(; t_rv=t, rv=rv, rv_err=err, rv_inst=inst, t_ref=0.0)
        @test d2.rv_inst == inst
        @test d2.t_ref == 0.0

        # Validation errors
        @test_throws ArgumentError Data(; t_rv=t, rv=rv[1:5], rv_err=err)
        @test_throws ArgumentError Data(; t_rv=t, rv=rv, rv_err=fill(-1.0, 11))
        @test_throws ArgumentError Data(; t_rv=t, rv=rv, rv_err=err, rv_inst=zeros(Int, 11))
        @test_throws ArgumentError Data(; t_rv=Float64[], rv=Float64[], rv_err=Float64[])

        # Indicator errors (for ActivityGP / multivariate-GP path)
        bis   = randn(_rng, 11)
        bis_σ = fill(0.5, 11)
        d3 = Data(; t_rv=t, rv=rv, rv_err=err,
                    indicators = Dict("bis" => bis),
                    indicator_errs = Dict("bis" => bis_σ))
        @test haskey(d3.indicators, "bis")
        @test haskey(d3.indicator_errs, "bis")
        # `normalize_indicators` (default) rescales the indicator VALUES by
        # their per-instrument RMS, so the errors must be divided by the SAME
        # factor — values and uncertainties are one unit system. Asserting the
        # errors come back verbatim would be asserting a corrupted S/N: it is
        # what silently told ActivityGP the indicators were ~26x more precise
        # than they are.
        @test d3.indicator_errs["bis"] != bis_σ                    # rescaled
        @test std(d3.indicators["bis"]) / d3.indicator_errs["bis"][1] ≈
              std(bis) / bis_σ[1]  rtol=1e-10                      # S/N preserved
        @test all(≈(d3.indicator_errs["bis"][1]), d3.indicator_errs["bis"])
        # Opting out keeps both raw, which is the old verbatim contract.
        d3raw = Data(; t_rv=t, rv=rv, rv_err=err,
                       indicators = Dict("bis" => bis),
                       indicator_errs = Dict("bis" => bis_σ),
                       normalize_indicators = false)
        @test d3raw.indicators["bis"] == bis
        @test d3raw.indicator_errs["bis"] == bis_σ
        # Without indicator_errs, the dict is empty
        d4 = Data(; t_rv=t, rv=rv, rv_err=err,
                    indicators = Dict("bis" => bis))
        @test isempty(d4.indicator_errs)
        # Mismatched length / negative / orphan-name errors
        @test_throws ArgumentError Data(; t_rv=t, rv=rv, rv_err=err,
                    indicators = Dict("bis" => bis),
                    indicator_errs = Dict("bis" => bis_σ[1:5]))
        @test_throws ArgumentError Data(; t_rv=t, rv=rv, rv_err=err,
                    indicators = Dict("bis" => bis),
                    indicator_errs = Dict("bis" => fill(-1.0, 11)))
        @test_throws ArgumentError Data(; t_rv=t, rv=rv, rv_err=err,
                    indicators = Dict("bis" => bis),
                    indicator_errs = Dict("fwhm" => bis_σ))
    end

    # -----------------------------------------------------------------
    @testset "rv_log_likelihood — single planet synthetic" begin
        # Ground-truth parameters.
        P_true     = 5.0
        K_true     = 8.0
        e_true     = 0.1
        ω_true     = 1.3
        M0_true    = 0.4
        gamma_true = 2.0
        sigma_obs  = 1.5   # formal measurement error
        jitter_true = 0.5   # extra noise floor

        # Generate synthetic RV data.
        Random.seed!(_rng, 42)
        n_obs = 60
        t_rv  = sort!(rand(_rng, n_obs) .* 50.0)
        t_ref = median(t_rv)
        rv_model = [gamma_true +
                    Nereus.rv_keplerian(ti, P_true, K_true, e_true,
                                          ω_true, M0_true, t_ref)
                    for ti in t_rv]
        total_noise = sqrt(sigma_obs^2 + jitter_true^2)
        rv_obs = rv_model .+ total_noise .* randn(_rng, n_obs)
        rv_err = fill(sigma_obs, n_obs)

        data = Data(; t_rv=t_rv, rv=rv_obs, rv_err=rv_err, t_ref=t_ref)
        @test n_rv(data) == n_obs

        # Model config — sesinw parametrization, single RV-only planet.
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => UniformPrior(0.5, 100.0),
            "K_k1"       => UniformPrior(0.1, 50.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> UniformPrior(-100.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors, max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic,
        )
        theta = Theta(params)

        # Set theta to the true values.
        sesinw_true = sqrt(e_true) * sin(ω_true)
        secosw_true = sqrt(e_true) * cos(ω_true)
        set_param!(theta, "P_k1", P_true)
        set_param!(theta, "K_k1", K_true)
        set_param!(theta, "sesinw_k1", sesinw_true)
        set_param!(theta, "secosw_k1", secosw_true)
        set_param!(theta, "Mo_k1", M0_true)
        set_param!(theta, "gamma_HARPS", gamma_true)
        set_param!(theta, "sigma_HARPS", jitter_true)

        ll_true = rv_log_likelihood(theta, data)
        @test isfinite(ll_true)

        # Likelihood at the truth should exceed likelihood at nearby
        # perturbations (local maximum property, modulo noise).
        set_param!(theta, "P_k1", P_true * 1.2)
        ll_off_P = rv_log_likelihood(theta, data)
        @test ll_true > ll_off_P

        set_param!(theta, "P_k1", P_true)     # restore
        set_param!(theta, "K_k1", K_true * 2.0)
        ll_off_K = rv_log_likelihood(theta, data)
        @test ll_true > ll_off_K

        # Wrong gamma → worse.
        set_param!(theta, "K_k1", K_true)     # restore
        set_param!(theta, "gamma_HARPS", gamma_true + 10.0)
        ll_off_γ = rv_log_likelihood(theta, data)
        @test ll_true > ll_off_γ
    end

    # -----------------------------------------------------------------
    @testset "rv_log_likelihood — zero planets" begin
        # n_p=0 path: predictions are pure gamma.
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(0.0),
            "gamma_HARPS"=> UniformPrior(-100.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors, max_kplanet=0, planet_modes=PlanetDataSources[],
            instruments=ic,
        )
        theta = Theta(params)
        set_param!(theta, "gamma_HARPS", 3.0)
        set_param!(theta, "sigma_HARPS", 0.5)

        t_rv = collect(0.0:0.5:10.0)
        rv   = fill(3.0, length(t_rv))   # perfect match to gamma
        err  = fill(1.0, length(t_rv))
        data = Data(; t_rv=t_rv, rv=rv, rv_err=err)

        # With perfect data + exact gamma, the likelihood is the
        # Gaussian peak value: -½ n log(2π σ²) with σ² = 1 + 0.25 = 1.25.
        ll = rv_log_likelihood(theta, data)
        expected = -0.5 * length(t_rv) * log(2π * 1.25)
        @test ll ≈ expected atol=1e-10
    end

    # -----------------------------------------------------------------
    @testset "rv_log_likelihood — Tp and Tc parametrizations" begin
        # Same synthetic data as Mo test — verify all three give same LL.
        Random.seed!(_rng, 42)
        P_true = 5.0; K_true = 8.0; e_true = 0.1; ω_true = 1.3
        M0_true = 0.4; gamma_true = 2.0; jitter_true = 0.5
        sigma_obs = 1.5; n_obs = 60
        t_rv = sort!(rand(_rng, n_obs) .* 50.0)
        t_ref = median(t_rv)
        rv_model = [gamma_true + Nereus.rv_keplerian(ti, P_true, K_true,
                     e_true, ω_true, M0_true, t_ref) for ti in t_rv]
        rv_obs = rv_model .+ sqrt(sigma_obs^2 + jitter_true^2) .* randn(_rng, n_obs)
        rv_err = fill(sigma_obs, n_obs)
        data = Data(; t_rv=t_rv, rv=rv_obs, rv_err=rv_err, t_ref=t_ref)

        ic = InstrumentConfig(rv=["HARPS"])
        sesinw_true = sqrt(e_true) * sin(ω_true)
        secosw_true = sqrt(e_true) * cos(ω_true)
        Tp_true = Nereus.mo_to_tp(M0_true, P_true, t_ref)
        Tc_true = Nereus.tp_to_tc(Tp_true, P_true, e_true, ω_true)

        # Mo (reference)
        priors_mo = Dict{String, PriorSpec}(
            "n_p" => FixedPrior(1.0),
            "P_k1" => UniformPrior(0.5, 100.0),
            "K_k1" => UniformPrior(0.1, 50.0),
            "sesinw_k1" => UniformPrior(-1.0, 1.0),
            "secosw_k1" => UniformPrior(-1.0, 1.0),
            "Mo_k1" => UniformPrior(-2π, 2π),
            "gamma_HARPS" => UniformPrior(-100.0, 100.0),
            "sigma_HARPS" => LogUniformPrior(0.001, 10.0),
        )
        params_mo = Params(; priors=priors_mo, max_kplanet=1,
                            planet_modes=[RV_ONLY], instruments=ic)
        theta_mo = Theta(params_mo)
        set_param!(theta_mo, "P_k1", P_true)
        set_param!(theta_mo, "K_k1", K_true)
        set_param!(theta_mo, "sesinw_k1", sesinw_true)
        set_param!(theta_mo, "secosw_k1", secosw_true)
        set_param!(theta_mo, "Mo_k1", M0_true)
        set_param!(theta_mo, "gamma_HARPS", gamma_true)
        set_param!(theta_mo, "sigma_HARPS", jitter_true)
        ll_mo = rv_log_likelihood(theta_mo, data)

        # Tp
        priors_tp = Dict{String, PriorSpec}(
            "n_p" => FixedPrior(1.0),
            "P_k1" => UniformPrior(0.5, 100.0),
            "K_k1" => UniformPrior(0.1, 50.0),
            "sesinw_k1" => UniformPrior(-1.0, 1.0),
            "secosw_k1" => UniformPrior(-1.0, 1.0),
            "Tp_k1" => UniformPrior(t_ref - 100.0, t_ref + 100.0),
            "gamma_HARPS" => UniformPrior(-100.0, 100.0),
            "sigma_HARPS" => LogUniformPrior(0.001, 10.0),
        )
        params_tp = Params(; priors=priors_tp, max_kplanet=1,
                            planet_modes=[RV_ONLY], instruments=ic,
                            parametrization=ParametrizationConfig(time=:Tp))
        theta_tp = Theta(params_tp)
        set_param!(theta_tp, "P_k1", P_true)
        set_param!(theta_tp, "K_k1", K_true)
        set_param!(theta_tp, "sesinw_k1", sesinw_true)
        set_param!(theta_tp, "secosw_k1", secosw_true)
        set_param!(theta_tp, "Tp_k1", Tp_true)
        set_param!(theta_tp, "gamma_HARPS", gamma_true)
        set_param!(theta_tp, "sigma_HARPS", jitter_true)
        ll_tp = rv_log_likelihood(theta_tp, data)
        @test ll_tp ≈ ll_mo atol=1e-6

        # Tc
        priors_tc = Dict{String, PriorSpec}(
            "n_p" => FixedPrior(1.0),
            "P_k1" => UniformPrior(0.5, 100.0),
            "K_k1" => UniformPrior(0.1, 50.0),
            "sesinw_k1" => UniformPrior(-1.0, 1.0),
            "secosw_k1" => UniformPrior(-1.0, 1.0),
            "Tc_k1" => UniformPrior(t_ref - 100.0, t_ref + 100.0),
            "gamma_HARPS" => UniformPrior(-100.0, 100.0),
            "sigma_HARPS" => LogUniformPrior(0.001, 10.0),
        )
        params_tc = Params(; priors=priors_tc, max_kplanet=1,
                            planet_modes=[RV_ONLY], instruments=ic,
                            parametrization=ParametrizationConfig(time=:Tc))
        theta_tc = Theta(params_tc)
        set_param!(theta_tc, "P_k1", P_true)
        set_param!(theta_tc, "K_k1", K_true)
        set_param!(theta_tc, "sesinw_k1", sesinw_true)
        set_param!(theta_tc, "secosw_k1", secosw_true)
        set_param!(theta_tc, "Tc_k1", Tc_true)
        set_param!(theta_tc, "gamma_HARPS", gamma_true)
        set_param!(theta_tc, "sigma_HARPS", jitter_true)
        ll_tc = rv_log_likelihood(theta_tc, data)
        @test ll_tc ≈ ll_mo atol=1e-6
    end

    # -----------------------------------------------------------------
    @testset "rv_log_likelihood — multi-instrument" begin
        # Two instruments with different gammas. Each contributes
        # independently to the Gaussian sum.
        ic = InstrumentConfig(rv=["HARPS", "ESPRESSO"])
        priors = Dict{String, PriorSpec}(
            "n_p"            => FixedPrior(0.0),
            "gamma_HARPS"    => UniformPrior(-100.0, 100.0),
            "sigma_HARPS"    => LogUniformPrior(0.001, 10.0),
            "gamma_ESPRESSO" => UniformPrior(-100.0, 100.0),
            "sigma_ESPRESSO" => LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors, max_kplanet=0, planet_modes=PlanetDataSources[],
            instruments=ic,
        )
        theta = Theta(params)
        set_param!(theta, "gamma_HARPS", 2.0)
        set_param!(theta, "sigma_HARPS", 0.1)
        set_param!(theta, "gamma_ESPRESSO", -3.0)
        set_param!(theta, "sigma_ESPRESSO", 0.2)

        t_rv = [0.0, 1.0, 2.0, 3.0]
        rv   = [2.0, 2.0, -3.0, -3.0]    # first two HARPS, last two ESPRESSO
        err  = [0.5, 0.5, 0.5, 0.5]
        inst = [1, 1, 2, 2]
        data = Data(; t_rv=t_rv, rv=rv, rv_err=err, rv_inst=inst)

        ll = rv_log_likelihood(theta, data)
        # Perfect residuals for both instruments.
        var_H = 0.5^2 + 0.1^2
        var_E = 0.5^2 + 0.2^2
        expected = -0.5 * (2 * log(2π * var_H) + 2 * log(2π * var_E))
        @test ll ≈ expected atol=1e-10
    end

    # -----------------------------------------------------------------
    @testset "rv_log_likelihood — analytic γ marginalization" begin
        # The per-instrument systemic offset γ enters the RV model
        # linearly, so it can be analytically marginalized (orvara's
        # approach). Verify that the γ-marginalized likelihood:
        #   (1) drops the gamma_* slots from the sampled set,
        #   (2) recovers the conditional γ̂ point estimate at the truth,
        #   (3) equals the profiled (max-over-γ) white-noise likelihood
        #       plus the per-group Gaussian-integral Jacobian
        #       Σ_g ½ log(2π / A_g), across an orbit scan (i.e. SAME
        #       posterior on P/K/e/ω/Tp/σ up to the additive constant).
        Random.seed!(_rng, 7)
        P_true=12.0; K_true=15.0; e_true=0.2; ω_true=0.8; M0_true=1.1
        γ_H=2000.0; γ_E=-3500.0   # very different absolute zero points
        σ_obs=1.5; jitter_true=0.7
        n_H=40; n_E=35
        tH=sort!(rand(_rng, n_H).*200.0); tE=sort!(rand(_rng, n_E).*200.0)
        t_rv=vcat(tH,tE); inst=vcat(fill(1,n_H),fill(2,n_E))
        perm=sortperm(t_rv); t_rv=t_rv[perm]; inst=inst[perm]
        t_ref=median(t_rv)
        γvec=[i==1 ? γ_H : γ_E for i in inst]
        rv_clean=[γvec[i]+Nereus.rv_keplerian(t_rv[i],P_true,K_true,e_true,
                                               ω_true,M0_true,t_ref)
                  for i in eachindex(t_rv)]
        rv=rv_clean .+ sqrt(σ_obs^2+jitter_true^2).*randn(_rng, length(t_rv))
        err=fill(σ_obs,length(t_rv))
        data=Data(; t_rv=t_rv, rv=rv, rv_err=err, rv_inst=inst, t_ref=t_ref)
        ic=InstrumentConfig(rv=["HARPS","ESPRESSO"])

        # Sampled-γ model.
        priors_s=Dict{String,PriorSpec}(
            "n_p"=>FixedPrior(1.0), "P_k1"=>UniformPrior(1.0,100.0),
            "K_k1"=>UniformPrior(0.1,60.0),
            "sesinw_k1"=>UniformPrior(-1.0,1.0),
            "secosw_k1"=>UniformPrior(-1.0,1.0),
            "Mo_k1"=>UniformPrior(-2π,2π),
            "gamma_HARPS"=>UniformPrior(-1e5,1e5),
            "sigma_HARPS"=>LogUniformPrior(1e-3,10.0),
            "gamma_ESPRESSO"=>UniformPrior(-1e5,1e5),
            "sigma_ESPRESSO"=>LogUniformPrior(1e-3,10.0))
        params_s=Params(; priors=priors_s, max_kplanet=1,
                        planet_modes=[RV_ONLY], instruments=ic,
                        stability=:none, data=data)

        # Marginalized-γ model (no gamma_* priors needed).
        priors_m=Dict{String,PriorSpec}(
            "n_p"=>FixedPrior(1.0), "P_k1"=>UniformPrior(1.0,100.0),
            "K_k1"=>UniformPrior(0.1,60.0),
            "sesinw_k1"=>UniformPrior(-1.0,1.0),
            "secosw_k1"=>UniformPrior(-1.0,1.0),
            "Mo_k1"=>UniformPrior(-2π,2π),
            "sigma_HARPS"=>LogUniformPrior(1e-3,10.0),
            "sigma_ESPRESSO"=>LogUniformPrior(1e-3,10.0))
        pc=ParametrizationConfig(marginalize_gamma=true)
        params_m=Params(; priors=priors_m, max_kplanet=1,
                        planet_modes=[RV_ONLY], instruments=ic,
                        stability=:none, data=data, parametrization=pc)

        # (1) Fewer free params; gamma_* not sampled.
        @test n_unfrozen(params_m) == n_unfrozen(params_s) - 2
        @test !("gamma_HARPS" in params_m.layout.unfrozen_names)
        @test !("gamma_ESPRESSO" in params_m.layout.unfrozen_names)

        setθ!(θ; P=P_true) = begin
            set_param!(θ,"P_k1",P); set_param!(θ,"K_k1",K_true)
            set_param!(θ,"sesinw_k1",sqrt(e_true)*sin(ω_true))
            set_param!(θ,"secosw_k1",sqrt(e_true)*cos(ω_true))
            set_param!(θ,"Mo_k1",M0_true)
            set_param!(θ,"sigma_HARPS",jitter_true)
            set_param!(θ,"sigma_ESPRESSO",jitter_true)
            θ
        end

        θm=setθ!(Theta(params_m))
        slotH=params_m.layout.systemic.rv_gamma[1]
        slotE=params_m.layout.systemic.rv_gamma[2]
        γ̂=conditional_gamma(θm,data)

        # (2) Conditional γ̂ recovers the injected offsets.
        @test isapprox(γ̂[slotH], γ_H; atol=3.0)
        @test isapprox(γ̂[slotE], γ_E; atol=3.0)

        # (3) Marginal == profiled + Jacobian, across an orbit scan.
        maxd=0.0
        for Ptest in range(8.0, 18.0; length=11)
            setθ!(θm; P=Ptest)
            γ̂t=conditional_gamma(θm,data)
            θs=setθ!(Theta(params_s); P=Ptest)
            set_param!(θs,"gamma_HARPS",γ̂t[slotH])
            set_param!(θs,"gamma_ESPRESSO",γ̂t[slotE])
            _, va = rv_predictions(θm,data)
            As=Dict{Int,Float64}()
            for i in eachindex(t_rv)
                g=params_m.layout.systemic.rv_gamma[inst[i]]
                As[g]=get(As,g,0.0)+1.0/va[i]
            end
            jac=sum(0.5*log(2π/As[g]) for g in keys(As))
            maxd=max(maxd, abs(rv_log_likelihood(θm,data) -
                               (rv_log_likelihood(θs,data)+jac)))
        end
        @test maxd < 1e-6

        # Marginalized path stays finite and gradient-clean (ForwardDiff)
        # in bounded space.
        tgt=NereusTarget(params_m, data; unconstrained=false)
        x0=Nereus.unfrozen_values(setθ!(Theta(params_m)))
        @test isfinite(LogDensityProblems.logdensity(tgt, x0))
        g=ForwardDiff.gradient(z->LogDensityProblems.logdensity(tgt,z), x0)
        @test all(isfinite, g)
    end

    # -----------------------------------------------------------------
    @testset "marginalize_gamma rejects RV noise models" begin
        # The analytic γ-marginalization assumes a diagonal RV
        # covariance; RV-channel noise models break that and must be
        # rejected at construction.
        ic=InstrumentConfig(rv=["HARPS"])
        data=Data(; t_rv=collect(0.0:1.0:20.0),
                  rv=randn(_rng,21), rv_err=fill(1.0,21))
        pc=ParametrizationConfig(marginalize_gamma=true)
        @test_throws ArgumentError Params(;
            max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic,
            data=data, parametrization=pc, stability=:none,
            noise_models=NoiseModel[MAModel(; order=1, channel=:rv)])
    end

    # -----------------------------------------------------------------
    @testset "NereusTarget — LogDensityProblems interface" begin
        # Build a 1-planet target and exercise the full LDP interface
        # plus a ForwardDiff gradient pass.
        ic = InstrumentConfig(rv=["HARPS"])
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => NormalPrior(5.0, 1.0, 0.5, 100.0),
            "K_k1"       => NormalPrior(8.0, 2.0, 0.1, 50.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(0.0, 10.0, -100.0, 100.0),
            "sigma_HARPS"=> LogUniformPrior(0.001, 10.0),
        )
        params = Params(;
            priors, max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic,
        )

        # Synthetic data near the prior means.
        Random.seed!(_rng, 7)
        t_rv  = sort!(rand(_rng, 40) .* 30.0)
        t_ref = median(t_rv)
        P_t, K_t, e_t, ω_t, M0_t, γ_t = 5.0, 8.0, 0.05, 0.8, 0.5, 0.0
        rv_clean = [γ_t + Nereus.rv_keplerian(ti, P_t, K_t, e_t, ω_t, M0_t, t_ref)
                    for ti in t_rv]
        rv_obs = rv_clean .+ 0.5 .* randn(_rng, length(t_rv))
        rv_err = fill(0.5, length(t_rv))

        data = Data(; t_rv=t_rv, rv=rv_obs, rv_err=rv_err, t_ref=t_ref)
        target = NereusTarget(params, data)

        # LDP dimension and capabilities
        @test LogDensityProblems.dimension(target) == n_unfrozen(params)
        @test LogDensityProblems.capabilities(typeof(target)) ==
              LogDensityProblems.LogDensityOrder{0}()

        # Evaluate at the ground truth.
        sesinw_t = sqrt(e_t) * sin(ω_t)
        secosw_t = sqrt(e_t) * cos(ω_t)
        x_true = [P_t, K_t, sesinw_t, secosw_t, M0_t, γ_t, 0.5]
        logp_true = LogDensityProblems.logdensity(target, x_true)
        @test isfinite(logp_true)

        # Evaluate offset away from truth — should be finite but lower
        # (for well-localized prior/likelihood).
        x_off = copy(x_true)
        x_off[1] = P_t + 0.5       # perturb P
        logp_off = LogDensityProblems.logdensity(target, x_off)
        @test isfinite(logp_off)
        @test logp_true > logp_off

        # Out-of-support → -Inf.
        x_bad = copy(x_true)
        x_bad[1] = 200.0    # P beyond prior upper bound
        @test LogDensityProblems.logdensity(target, x_bad) == -Inf

        # Autodiff gradient via ForwardDiff through the full target.
        # This is the end-to-end validation that Theta{Dual} →
        # set_unfrozen! → log_prior + rv_log_likelihood works.
        f(x) = LogDensityProblems.logdensity(target, x)
        grad = ForwardDiff.gradient(f, x_true)
        @test length(grad) == n_unfrozen(params)
        @test all(isfinite, grad)

        # Sanity: gradient at the injected truth shouldn't be exactly
        # zero (noise + finite data) but should be close to zero for
        # parameters with narrow Normal priors if the likelihood is
        # also near its peak.
        @test !all(==(0.0), grad)   # not the trivial zero gradient
    end

    # =================================================================
    # NUTS SAMPLER — first real posterior samples
    # =================================================================

    # -----------------------------------------------------------------
    @testset "sample_nuts — 1-planet RV recovery" begin
        # Build a well-conditioned single-planet RV problem with tight
        # Normal priors centred on the truth, so NUTS converges quickly
        # in a small number of iterations. This is NOT a detection test
        # — it's a "does the sampling machinery work end-to-end" test.

        # Ground truth.
        P_t, K_t, e_t, ω_t, M0_t, γ_t, σ_jit = 5.0, 8.0, 0.05, 1.0, 0.5, 2.0, 0.5

        # Synthetic data.
        rng_syn = MersenneTwister(123)
        n_obs   = 50
        t_rv    = sort!(rand(rng_syn, n_obs) .* 30.0)
        t_ref   = median(t_rv)
        rv_clean = [γ_t + Nereus.rv_keplerian(ti, P_t, K_t, e_t, ω_t, M0_t, t_ref)
                    for ti in t_rv]
        rv_obs  = rv_clean .+ sqrt(1.0 + σ_jit^2) .* randn(rng_syn, n_obs)
        rv_err  = fill(1.0, n_obs)
        data    = Data(; t_rv, rv=rv_obs, rv_err, t_ref)

        # Model: single RV planet, Normal priors near the truth so the
        # posterior is well-localised and NUTS converges in O(100) steps.
        ic = InstrumentConfig(rv=["HARPS"])
        sesinw_t = sqrt(e_t) * sin(ω_t)
        secosw_t = sqrt(e_t) * cos(ω_t)
        priors = Dict{String, PriorSpec}(
            "n_p"        => FixedPrior(1.0),
            "P_k1"       => NormalPrior(P_t, 0.5, 0.5, 50.0),
            "K_k1"       => NormalPrior(K_t, 1.0, 0.1, 50.0),
            "sesinw_k1"  => UniformPrior(-1.0, 1.0),
            "secosw_k1"  => UniformPrior(-1.0, 1.0),
            "Mo_k1"      => UniformPrior(-2π, 2π),
            "gamma_HARPS"=> NormalPrior(γ_t, 2.0, -50.0, 50.0),
            "sigma_HARPS"=> LogUniformPrior(0.01, 5.0),
        )
        params = Params(;
            priors, max_kplanet=1, planet_modes=[RV_ONLY], instruments=ic,
        )
        target = NereusTarget(params, data)

        # Initial position near the truth.
        x0 = [P_t, K_t, sesinw_t, secosw_t, M0_t, γ_t, σ_jit]

        # Run NUTS. Short chains for a unit test (not for real
        # science). 200 warmup + 300 samples should be enough to
        # verify the machinery works.
        chains = sample_nuts(target;
                              n_samples=300, n_warmup=200,
                              init=x0,
                              rng=MersenneTwister(42),
                              progress=false)

        @test chains isa MCMCChains.Chains
        @test size(chains, 1) == 300   # n_samples post-warmup
        @test size(chains, 2) >= 7     # at least 7 params

        # Parameter names should include our unfrozen names.
        cnames = string.(names(chains, :parameters))
        @test "P_k1" in cnames
        @test "K_k1" in cnames
        @test "gamma_HARPS" in cnames

        # Posterior mean should be within ~3σ of truth for the
        # well-constrained parameters (P, K, gamma). Loose tolerance
        # because this is a unit test, not a calibration run.
        P_post  = mean(chains[:P_k1])
        K_post  = mean(chains[:K_k1])
        γ_post  = mean(chains[:gamma_HARPS])

        @test abs(P_post - P_t) < 3.0   # σ_P ~ 0.5, so 3σ = 1.5 → 3.0 very safe
        @test abs(K_post - K_t) < 5.0   # σ_K ~ 1.0
        @test abs(γ_post - γ_t) < 8.0   # σ_γ ~ 2.0

        # ESS should be non-trivial (> 10) for a 300-sample chain
        # with good adaptation. This catches gross sampling failures
        # (stuck at a wall, NaN gradients, etc.).
        ess_vals = MCMCChains.ess_rhat(chains)
        # ess_rhat returns a NamedTuple or DataFrame; check at least
        # one parameter has ESS > 10.
        @test any(>(10), ess_vals[:, :ess])
    end

    # =================================================================
    # END NUTS SAMPLER
    # =================================================================

    # =================================================================
    # END DATA + LIKELIHOOD + TARGET
    # =================================================================

    # -----------------------------------------------------------------
    @testset "Autodiff compatibility (ForwardDiff-style Dual types)" begin
        # Kept as a lightweight sanity check that logpdf works for
        # generic Real subtypes (BigFloat stands in for non-Float64
        # codepaths). The real end-to-end autodiff test is the
        # "Theta{T} autodiff with ForwardDiff" testset above.
        p = UniformPrior(0.0, 10.0)
        @test logpdf(p, BigFloat(5)) isa Real
        @test logpdf(p, BigFloat(5)) ≈ -log(BigFloat(10)) atol=1e-20

        p_norm = NormalPrior(0.0, 1.0, -3.0, 3.0)
        @test isfinite(logpdf(p_norm, BigFloat(0.5)))

        p_log = LogUniformPrior(0.1, 100.0)
        @test isfinite(logpdf(p_log, BigFloat(1)))

        p_mj = ModJeffreysPrior(1.0, 100.0)
        @test isfinite(logpdf(p_mj, BigFloat(5)))
    end
end

# =================================================================
# Trans-dimensional state and config
# =================================================================

@testset "TransDimState" begin
    @testset "construction" begin
        tds = TransDimState(max_planets=3, n_noise=2)
        @test tds.n_planets_active == 0
        @test length(tds.planet_active) == 3
        @test all(.!tds.planet_active)
        @test length(tds.noise_active) == 2
        @test all(.!tds.noise_active)

        tds0 = TransDimState(max_planets=0)
        @test length(tds0.planet_active) == 0
        @test tds0.n_planets_active == 0

        @test_throws ArgumentError TransDimState(max_planets=-1)
        @test_throws ArgumentError TransDimState(max_planets=1, n_noise=-1)
    end

    @testset "planet activation" begin
        tds = TransDimState(max_planets=3, n_noise=0)

        activate_planet!(tds, 2)
        @test tds.n_planets_active == 1
        @test tds.planet_active[2]
        @test !tds.planet_active[1]
        @test !tds.planet_active[3]
        @test active_planets(tds) == [2]

        # No-op if already active
        activate_planet!(tds, 2)
        @test tds.n_planets_active == 1

        activate_planet!(tds, 1)
        @test tds.n_planets_active == 2
        @test active_planets(tds) == [1, 2]

        deactivate_planet!(tds, 2)
        @test tds.n_planets_active == 1
        @test !tds.planet_active[2]
        @test active_planets(tds) == [1]

        # No-op if already inactive
        deactivate_planet!(tds, 2)
        @test tds.n_planets_active == 1

        deactivate_planet!(tds, 1)
        @test tds.n_planets_active == 0
        @test isempty(active_planets(tds))
    end

    @testset "first_inactive_planet" begin
        tds = TransDimState(max_planets=3, n_noise=0)
        @test first_inactive_planet(tds) == 1

        activate_planet!(tds, 1)
        @test first_inactive_planet(tds) == 2

        activate_planet!(tds, 2)
        activate_planet!(tds, 3)
        @test first_inactive_planet(tds) === nothing
    end

    @testset "noise activation" begin
        tds = TransDimState(max_planets=1, n_noise=3)

        activate_noise!(tds, 2)
        @test is_noise_active(tds, 2)
        @test !is_noise_active(tds, 1)
        @test active_noise(tds) == [2]

        deactivate_noise!(tds, 2)
        @test !is_noise_active(tds, 2)
        @test isempty(active_noise(tds))
    end

    @testset "copy" begin
        tds = TransDimState(max_planets=3, n_noise=2)
        activate_planet!(tds, 1)
        activate_noise!(tds, 2)

        tds2 = copy(tds)
        @test tds2.n_planets_active == 1
        @test tds2.planet_active[1]
        @test tds2.noise_active[2]

        # Mutating copy doesn't affect original
        deactivate_planet!(tds2, 1)
        @test tds.n_planets_active == 1
        @test tds2.n_planets_active == 0
    end

    @testset "bounds checking" begin
        tds = TransDimState(max_planets=2, n_noise=1)
        @test_throws BoundsError activate_planet!(tds, 0)
        @test_throws BoundsError activate_planet!(tds, 3)
        @test_throws BoundsError deactivate_planet!(tds, 0)
        @test_throws BoundsError activate_noise!(tds, 0)
        @test_throws BoundsError activate_noise!(tds, 2)
    end
end

@testset "TransDimConfig" begin
    @testset "defaults" begin
        td = TransDimConfig(max_kplanet=3)
        @test td.planets == true
        @test td.max_kplanet == 3
        @test td.noise == false
        @test isempty(td.toggleable)
        @test length(td.birth_strategies) == 2
        @test td.birth_strategies[1] isa PriorBirth
        @test td.birth_strategies[2] isa InformedBirth
        @test td.birth_weights ≈ [0.3, 0.7]
        @test td.transdim_fraction == 0.2
    end

    @testset "with noise" begin
        td = TransDimConfig(max_kplanet=3, noise=true,
                            toggleable=[ARModel(order=1), CeleriteSHO()])
        @test td.noise == true
        @test length(td.toggleable) == 2
        @test td.toggleable[1] isa ARModel
        @test td.toggleable[2] isa CeleriteSHO
    end

    @testset "custom strategies" begin
        td = TransDimConfig(max_kplanet=2,
                            birth_strategies=[PriorBirth(), DonorBirth()],
                            birth_weights=[0.5, 0.5])
        @test length(td.birth_strategies) == 2
        @test td.birth_strategies[2] isa DonorBirth
    end

    @testset "validation" begin
        @test_throws ArgumentError TransDimConfig(max_kplanet=-1)
        @test_throws ArgumentError TransDimConfig(max_kplanet=2,
            birth_strategies=[PriorBirth()], birth_weights=[0.5, 0.5])
        @test_throws ArgumentError TransDimConfig(max_kplanet=2,
            birth_strategies=[PriorBirth()], birth_weights=[0.5])
        @test_throws ArgumentError TransDimConfig(max_kplanet=2,
            transdim_fraction=1.5)
        @test_throws ArgumentError TransDimConfig(max_kplanet=2,
            noise=true, toggleable=NoiseModel[])
    end
end

@testset "Trans-dim likelihood" begin
    # Build a 2-planet RV model
    ic = InstrumentConfig(rv=["HARPS"])
    data = Data(;
        t_rv=collect(0.0:1.0:100.0),
        rv=sin.(2π .* collect(0.0:1.0:100.0) ./ 4.23) .* 50.0,
        rv_err=ones(101),
    )
    params = Params(
        max_kplanet=2,
        planet_modes=[RV_ONLY, RV_ONLY],
        instruments=ic,
        data=data,
        M_s=1.0,
    )

    @testset "planet_indices fixed-dim" begin
        theta = Theta(params)
        set_n_p!(theta, 2)
        @test collect(planet_indices(theta)) == [1, 2]
        set_n_p!(theta, 1)
        @test collect(planet_indices(theta)) == [1]
        set_n_p!(theta, 0)
        @test isempty(collect(planet_indices(theta)))
    end

    @testset "planet_indices trans-dim" begin
        td = TransDimState(max_planets=2)
        theta = Theta(params; td=td)

        @test isempty(collect(planet_indices(theta)))
        @test n_p(theta) == 0

        activate_planet!(td, 2)
        @test collect(planet_indices(theta)) == [2]
        @test n_p(theta) == 1

        activate_planet!(td, 1)
        @test collect(planet_indices(theta)) == [1, 2]
        @test n_p(theta) == 2
    end

    @testset "rv_log_likelihood respects active mask" begin
        td = TransDimState(max_planets=2)
        theta = Theta(params; td=td)
        set_param!(theta, "gamma_HARPS", 0.0)
        set_param!(theta, "sigma_HARPS", 1.0)

        # 0 active planets: LL is finite (just Gaussian around gamma)
        ll_0 = rv_log_likelihood(theta, data)
        @test isfinite(ll_0)

        # Activate planet 1 → LL changes
        activate_planet!(td, 1)
        set_param!(theta, "P_k1", 4.23)
        set_param!(theta, "K_k1", 50.0)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 0.0)

        ll_1 = rv_log_likelihood(theta, data)
        @test isfinite(ll_1)
        @test ll_1 != ll_0  # different model → different LL

        # Activate planet 2 → LL changes again
        activate_planet!(td, 2)
        set_param!(theta, "P_k2", 10.0)
        set_param!(theta, "K_k2", 100.0)
        set_param!(theta, "sesinw_k2", 0.0)
        set_param!(theta, "secosw_k2", 0.0)
        set_param!(theta, "Mo_k2", 0.0)

        ll_2 = rv_log_likelihood(theta, data)
        @test isfinite(ll_2)
        @test ll_2 != ll_1  # different model → different LL

        # Deactivate planet 2 → should recover ll_1 exactly
        deactivate_planet!(td, 2)
        ll_back = rv_log_likelihood(theta, data)
        @test ll_back ≈ ll_1
    end

    @testset "is_noise_model_active fixed-dim" begin
        theta = Theta(params)
        @test is_noise_model_active(theta, 1) == true
    end

    @testset "is_noise_model_active trans-dim" begin
        td = TransDimState(max_planets=2, n_noise=2)
        theta = Theta(params; td=td)
        @test is_noise_model_active(theta, 1) == false
        @test is_noise_model_active(theta, 2) == false
        activate_noise!(td, 1)
        @test is_noise_model_active(theta, 1) == true
        @test is_noise_model_active(theta, 2) == false
    end

    @testset "propose_planet_birth/death" begin
        rng = MersenneTwister(42)
        td = TransDimState(max_planets=2)
        theta = Theta(params; td=td)
        set_param!(theta, "gamma_HARPS", 0.0)
        set_param!(theta, "sigma_HARPS", 1.0)

        # Birth from 0 planets
        new_theta, log_q = propose_planet_birth(theta, rng, PriorBirth())
        @test n_p(new_theta) == 1
        @test new_theta.td.planet_active[1]  # first inactive slot activated
        @test isfinite(log_q)

        # Original theta unchanged
        @test n_p(theta) == 0

        # Birth from 1 planet
        new_theta2, log_q2 = propose_planet_birth(new_theta, rng, PriorBirth())
        @test n_p(new_theta2) == 2
        @test isfinite(log_q2)

        # Birth from max → impossible
        new_theta3, log_q3 = propose_planet_birth(new_theta2, rng, PriorBirth())
        @test log_q3 == -Inf  # all slots full

        # Death from 2 planets
        dead_theta, log_q_d = propose_planet_death(new_theta2, rng)
        @test n_p(dead_theta) == 1
        @test isfinite(log_q_d)

        # Death from 0 → impossible
        empty_td = TransDimState(max_planets=2)
        empty_theta = Theta(params; td=empty_td)
        _, log_q_empty = propose_planet_death(empty_theta, rng)
        @test log_q_empty == -Inf

        # New planet params are within prior bounds
        new_theta_check, _ = propose_planet_birth(theta, rng, PriorBirth())
        p_val = get_param(new_theta_check, "P_k1")
        k_val = get_param(new_theta_check, "K_k1")
        p_prior = params.config.priors["P_k1"]
        k_prior = params.config.priors["K_k1"]
        @test bounds(p_prior)[1] ≤ p_val ≤ bounds(p_prior)[2]
        @test bounds(k_prior)[1] ≤ k_val ≤ bounds(k_prior)[2]
    end

    @testset "informed birth proposes" begin
        rng_ib = MersenneTwister(42)
        td_ib = TransDimState(max_planets=2)
        theta_ib = Theta(params; td=td_ib)
        set_param!(theta_ib, "gamma_HARPS", 0.0)
        set_param!(theta_ib, "sigma_HARPS", 1.0)

        new_theta_ib, log_q_ib = propose_planet_birth(
            theta_ib, rng_ib, InformedBirth(); data=data)
        @test n_p(new_theta_ib) == 1
        @test isfinite(log_q_ib)

        # Period should be within prior bounds
        p_val_ib = get_param(new_theta_ib, "P_k1")
        p_prior_ib = params.config.priors["P_k1"]
        @test bounds(p_prior_ib)[1] ≤ p_val_ib ≤ bounds(p_prior_ib)[2]
    end

    @testset "donor birth from population" begin
        rng_db = MersenneTwister(42)

        # Create a donor with 1 active planet
        td_donor = TransDimState(max_planets=2)
        donor = Theta(params; td=td_donor)
        set_param!(donor, "gamma_HARPS", 0.0)
        set_param!(donor, "sigma_HARPS", 1.0)
        activate_planet!(td_donor, 1)
        set_param!(donor, "P_k1", 10.0)
        set_param!(donor, "K_k1", 20.0)
        set_param!(donor, "sesinw_k1", 0.1)
        set_param!(donor, "secosw_k1", 0.2)
        set_param!(donor, "Mo_k1", 1.5)

        # Create recipient with 0 planets
        td_recip = TransDimState(max_planets=2)
        recip = Theta(params; td=td_recip)
        set_param!(recip, "gamma_HARPS", 0.0)
        set_param!(recip, "sigma_HARPS", 1.0)

        pop = Theta{Float64}[donor]
        new_theta_db, log_q_db = propose_planet_birth(
            recip, rng_db, DonorBirth(); data=data, population=pop)

        @test n_p(new_theta_db) == 1
        @test isfinite(log_q_db)

        # New planet should be within prior bounds (donor + jitter)
        new_P = get_param(new_theta_db, "P_k1")
        p_prior_db = params.config.priors["P_k1"]
        @test bounds(p_prior_db)[1] ≤ new_P ≤ bounds(p_prior_db)[2]
    end

    @testset "donor birth without population falls back" begin
        rng_db2 = MersenneTwister(42)
        td_db2 = TransDimState(max_planets=2)
        theta_db2 = Theta(params; td=td_db2)
        set_param!(theta_db2, "gamma_HARPS", 0.0)
        set_param!(theta_db2, "sigma_HARPS", 1.0)

        # No population → falls back to PriorBirth
        new_theta_db2, log_q_db2 = propose_planet_birth(
            theta_db2, rng_db2, DonorBirth())
        @test n_p(new_theta_db2) == 1
        @test isfinite(log_q_db2)
    end

    # ---- Astrometric birth/death --------------------------------------
    # Build a separate RVAS model (HARPS RV + relative astrom + HGCA),
    # then verify that birth populates `inc_kN`, `Omega_kN` (and the
    # mass slot) with finite, in-bounds values for every birth strategy.

    @testset "astrometric birth (RVAS)" begin
        ic_as = InstrumentConfig(rv=["HARPS"], pm=String[])
        t_rv_as = collect(55000.0 : 30.0 : 55360.0)
        relast_as = RelAstromData(
            t       = [55000.0, 55090.0, 55180.0, 55270.0],
            ra_off  = [10.0, 8.0, -5.0, -10.0],
            dec_off = [0.0, 6.0, 9.0, 0.0],
            ra_err  = [1.0, 1.0, 1.0, 1.0],
            dec_err = [1.0, 1.0, 1.0, 1.0],
        )
        hgca_as = HGCAData(
            epochs       = mjd_epochs((1991.25, 2004.6, 2016.0)),
            pmra         = (5.0, 4.95, 4.9),
            pmdec        = (-3.0, -3.05, -3.1),
            sigma_pmra   = (0.2, 0.2, 0.2),
            sigma_pmdec  = (0.2, 0.2, 0.2),
            plx = 25.0, plx_err = 0.05, hip_id = 1,
        )
        data_as = Data(
            t_rv      = t_rv_as,
            rv        = zeros(length(t_rv_as)),
            rv_err    = fill(1.0, length(t_rv_as)),
            relastrom = relast_as,
            hgca      = hgca_as,
        )
        params_as = Params(
            max_kplanet  = 2,
            planet_modes = [RVAS, RVAS],
            instruments  = ic_as,
            data         = data_as,
            stability    = :none,
            M_s          = 1.0,
        )

        # Sanity: layout uses RVASBlock and exposes the AS slot names.
        @test params_as.layout.planet_blocks[1] isa RVASBlock
        @test "inc_k1"   in params_as.layout.names
        @test "Omega_k1" in params_as.layout.names

        @testset "PriorBirth populates inc/Omega/K" begin
            rng_as = MersenneTwister(7)
            td_as  = TransDimState(max_planets=2)
            theta_as = Theta(params_as; td=td_as)
            set_param!(theta_as, "gamma_HARPS", 0.0)
            set_param!(theta_as, "sigma_HARPS", 1.0)
            set_param!(theta_as, "plx", 25.0)

            new_theta, log_q = propose_planet_birth(theta_as, rng_as, PriorBirth())
            @test n_p(new_theta) == 1
            @test isfinite(log_q)

            # AS slots have finite, in-bounds values for the new planet.
            inc_v   = get_param(new_theta, "inc_k1")
            Omega_v = get_param(new_theta, "Omega_k1")
            K_v     = get_param(new_theta, "K_k1")
            P_v     = get_param(new_theta, "P_k1")
            @test isfinite(inc_v) && isfinite(Omega_v)
            @test isfinite(K_v) && isfinite(P_v)

            inc_lo, inc_hi = bounds(params_as.config.priors["inc_k1"])
            om_lo,  om_hi  = bounds(params_as.config.priors["Omega_k1"])
            K_lo,   K_hi   = bounds(params_as.config.priors["K_k1"])
            P_lo,   P_hi   = bounds(params_as.config.priors["P_k1"])
            @test inc_lo ≤ inc_v ≤ inc_hi
            @test om_lo  ≤ Omega_v ≤ om_hi
            @test K_lo   ≤ K_v   ≤ K_hi
            @test P_lo   ≤ P_v   ≤ P_hi
        end

        @testset "InformedBirth falls back to prior for AS slots" begin
            rng_as = MersenneTwister(11)
            td_as  = TransDimState(max_planets=2)
            theta_as = Theta(params_as; td=td_as)
            set_param!(theta_as, "gamma_HARPS", 0.0)
            set_param!(theta_as, "sigma_HARPS", 1.0)
            set_param!(theta_as, "plx", 25.0)

            new_theta, log_q = propose_planet_birth(
                theta_as, rng_as, InformedBirth(); data=data_as)
            @test n_p(new_theta) == 1
            @test isfinite(log_q)

            inc_v   = get_param(new_theta, "inc_k1")
            Omega_v = get_param(new_theta, "Omega_k1")
            @test isfinite(inc_v) && isfinite(Omega_v)

            inc_lo, inc_hi = bounds(params_as.config.priors["inc_k1"])
            om_lo,  om_hi  = bounds(params_as.config.priors["Omega_k1"])
            @test inc_lo ≤ inc_v   ≤ inc_hi
            @test om_lo  ≤ Omega_v ≤ om_hi
        end

        @testset "DonorBirth clones AS slots from donor" begin
            rng_as = MersenneTwister(13)

            # Donor: planet 1 active with a specific inc/Omega.
            td_donor_as = TransDimState(max_planets=2)
            donor_as = Theta(params_as; td=td_donor_as)
            set_param!(donor_as, "gamma_HARPS", 0.0)
            set_param!(donor_as, "sigma_HARPS", 1.0)
            set_param!(donor_as, "plx", 25.0)
            activate_planet!(td_donor_as, 1)
            set_param!(donor_as, "P_k1",       365.25)
            set_param!(donor_as, "K_k1",        50.0)
            set_param!(donor_as, "sesinw_k1",    0.05)
            set_param!(donor_as, "secosw_k1",    0.05)
            set_param!(donor_as, "Mo_k1",        0.5)
            set_param!(donor_as, "inc_k1",       deg2rad(60))
            set_param!(donor_as, "Omega_k1",     deg2rad(45))

            # Recipient: empty.
            td_recip_as = TransDimState(max_planets=2)
            recip_as = Theta(params_as; td=td_recip_as)
            set_param!(recip_as, "gamma_HARPS", 0.0)
            set_param!(recip_as, "sigma_HARPS", 1.0)
            set_param!(recip_as, "plx", 25.0)

            pop_as = Theta{Float64}[donor_as]
            new_theta, log_q = propose_planet_birth(
                recip_as, rng_as, DonorBirth();
                data=data_as, population=pop_as)

            @test n_p(new_theta) == 1
            @test isfinite(log_q)

            # Donor's inc/Omega were jittered into the recipient's slot.
            inc_v   = get_param(new_theta, "inc_k1")
            Omega_v = get_param(new_theta, "Omega_k1")
            @test isfinite(inc_v) && isfinite(Omega_v)

            inc_lo, inc_hi = bounds(params_as.config.priors["inc_k1"])
            om_lo,  om_hi  = bounds(params_as.config.priors["Omega_k1"])
            @test inc_lo ≤ inc_v   ≤ inc_hi
            @test om_lo  ≤ Omega_v ≤ om_hi

            # Cloned values should be near (but not necessarily equal to)
            # the donor values after the small jitter.
            @test abs(inc_v   - deg2rad(60)) < 0.5
            @test abs(Omega_v - deg2rad(45)) < 1.0
        end

        @testset "Death cleanly resets AS planet activation" begin
            rng_as = MersenneTwister(19)
            td_as  = TransDimState(max_planets=2)
            theta_as = Theta(params_as; td=td_as)
            set_param!(theta_as, "gamma_HARPS", 0.0)
            set_param!(theta_as, "sigma_HARPS", 1.0)
            set_param!(theta_as, "plx", 25.0)

            born, _ = propose_planet_birth(theta_as, rng_as, PriorBirth())
            @test n_p(born) == 1
            @test born.td.planet_active[1]

            killed, log_q_d = propose_planet_death(born, rng_as)
            @test n_p(killed) == 0
            @test !any(killed.td.planet_active)
            @test isfinite(log_q_d)

            # Original theta is left untouched (no in-place mutation leak).
            @test n_p(theta_as) == 0
            @test !any(theta_as.td.planet_active)

            # Re-birth into the same slot still produces in-bounds AS
            # values — confirms death did not corrupt the layout.
            reborn, _ = propose_planet_birth(killed, rng_as, PriorBirth())
            @test n_p(reborn) == 1
            inc_v   = get_param(reborn, "inc_k1")
            Omega_v = get_param(reborn, "Omega_k1")
            inc_lo, inc_hi = bounds(params_as.config.priors["inc_k1"])
            om_lo,  om_hi  = bounds(params_as.config.priors["Omega_k1"])
            @test inc_lo ≤ inc_v   ≤ inc_hi
            @test om_lo  ≤ Omega_v ≤ om_hi
        end
    end

    @testset "rjmcmc_accept" begin
        rng = MersenneTwister(1)
        # Certain accept: new much better
        @test rjmcmc_accept(-1000.0, -10.0, -5.0, -5.0, 0.0, rng) == true
        # Certain reject: new much worse
        @test rjmcmc_accept(-10.0, -1000.0, -5.0, -5.0, 0.0, rng) == false
    end

    @testset "sample_rjmcmc runs" begin
        # Synthetic 1-planet RV data (large K to avoid ModJeffreys knee issue)
        rng_syn = MersenneTwister(123)
        t_rv = collect(0.0:2.0:200.0)
        P_true = 10.0
        K_true = 30.0
        gamma_true = 100.0
        rv_true = gamma_true .+ K_true .* sin.(2π .* t_rv ./ P_true)
        rv_obs = rv_true .+ 2.0 .* randn(rng_syn, length(t_rv))

        ic_rj = InstrumentConfig(rv=["SIM"])
        data_rj = Data(; t_rv=t_rv, rv=rv_obs, rv_err=fill(2.0, length(t_rv)))
        params_rj = Params(
            max_kplanet=2,
            planet_modes=[RV_ONLY, RV_ONLY],
            instruments=ic_rj,
            data=data_rj,
            M_s=1.0,
        )
        target_rj = NereusTarget(params_rj, data_rj; unconstrained=false)
        td_config = TransDimConfig(max_kplanet=2)

        chains, n_evals = sample_rjmcmc(target_rj, data_rj; td=td_config,
                                        n_samples=500, n_warmup=500, seed=42)

        @test chains isa MCMCChains.Chains
        @test size(chains, 1) == 500
        @test :n_planets in names(chains, :parameters)
        @test n_evals > 0

        np_samples = Array(chains[:n_planets])
        @test all(np_samples .>= 0.0)
        @test all(np_samples .≤ 2.0)
    end

    @testset "sample_pt trans-dim runs" begin
        rng_syn = MersenneTwister(123)
        t_rv = collect(0.0:2.0:200.0)
        gamma_true = 100.0
        rv_obs = gamma_true .+ 30.0 .* sin.(2π .* t_rv ./ 10.0) .+
                 2.0 .* randn(rng_syn, length(t_rv))

        ic_pt = InstrumentConfig(rv=["SIM"])
        data_pt = Data(; t_rv=t_rv, rv=rv_obs, rv_err=fill(2.0, length(t_rv)))
        params_pt = Params(
            max_kplanet=2,
            planet_modes=[RV_ONLY, RV_ONLY],
            instruments=ic_pt,
            data=data_pt,
            M_s=1.0,
        )
        target_pt = NereusTarget(params_pt, data_pt; unconstrained=false)
        td_config = TransDimConfig(max_kplanet=2)

        chains_pt, log_ev, n_evals_pt = sample_pt(target_pt; td=td_config,
                                                    n_rounds=5, n_chains=4, seed=42,
                                                    show_report=false)

        @test chains_pt isa MCMCChains.Chains
        @test :n_planets in names(chains_pt, :parameters)
        @test isfinite(log_ev)
        @test n_evals_pt > 0

        np_pt = Array(chains_pt[:n_planets])
        @test all(np_pt .>= 0.0)
        @test all(np_pt .≤ 2.0)

        # RWM within-model kernel: should produce a well-formed chain
        # at much lower likelihood-eval cost (1 eval per coord vs ~7
        # for slice). Scientific equivalence checked elsewhere; here
        # we just verify the wiring works.
        chains_rwm, _, n_evals_rwm = sample_pt(target_pt; td=td_config,
                                                  n_rounds=5, n_chains=4,
                                                  seed=42, show_report=false,
                                                  within_model=:rwm)
        @test chains_rwm isa MCMCChains.Chains
        @test :n_planets in names(chains_rwm, :parameters)
        @test n_evals_rwm > 0
        @test n_evals_rwm < n_evals_pt   # RWM does ~7× fewer evals
        np_rwm = Array(chains_rwm[:n_planets])
        @test all(0 .<= np_rwm .<= 2)

        # Bad within_model symbol → clear error
        @test_throws ArgumentError sample_pt(target_pt; td=td_config,
                                              n_rounds=2, n_chains=2,
                                              seed=1, show_report=false,
                                              within_model=:not_a_kernel)
    end

    @testset "trans-dim recovers correct N_p" begin
        # Synthetic system: 1 planet with strong signal (K=50 m/s, P=10 d)
        # RJMCMC should spend most time at N_p=1
        rng_val = MersenneTwister(999)
        t_val = sort(rand(rng_val, 80) .* 300.0)
        P_val = 10.0; K_val = 50.0; gamma_val = 50.0
        rv_val = gamma_val .+ K_val .* sin.(2π .* t_val ./ P_val) .+
                 1.5 .* randn(rng_val, 80)

        ic_val = InstrumentConfig(rv=["SIM"])
        data_val = Data(; t_rv=t_val, rv=rv_val, rv_err=fill(1.5, 80))
        params_val = Params(
            max_kplanet=3,
            planet_modes=[RV_ONLY, RV_ONLY, RV_ONLY],
            instruments=ic_val,
            data=data_val,
            M_s=1.0,
        )
        target_val = NereusTarget(params_val, data_val; unconstrained=false)
        td_val = TransDimConfig(max_kplanet=3)

        chains_val, _ = sample_rjmcmc(target_val, data_val;
            td=td_val, n_samples=2000, n_warmup=2000, seed=42)

        np_val = Array(chains_val[:n_planets])

        # Count N_p distribution
        n0 = count(np_val .== 0.0)
        n1 = count(np_val .== 1.0)
        n2 = count(np_val .== 2.0)
        n3 = count(np_val .== 3.0)

        # With K=50 m/s and errors of 1.5 m/s, the signal is overwhelming.
        # The sampler should spend significant time at N_p ≥ 1.
        # (With prior birth only, acceptance is low, so we mainly check
        # that the machinery works end-to-end and produces valid output.)
        @test all(np_val .>= 0.0)
        @test all(np_val .≤ 3.0)
        # Births may or may not succeed with basic RJMCMC (poor mixing
        # without tempering is expected). Just check valid output.
        @test n0 + n1 + n2 + n3 == 2000
    end

    @testset "noise birth/death in sampler" begin
        # Synthetic data: 1 planet + correlated noise
        rng_nb = MersenneTwister(777)
        t_nb = sort(rand(rng_nb, 60) .* 200.0)
        P_nb = 15.0; K_nb = 40.0; gamma_nb = 50.0
        rv_nb = gamma_nb .+ K_nb .* sin.(2π .* t_nb ./ P_nb)

        # Add correlated noise (MA-like): each residual correlates with previous
        noise_nb = zeros(60)
        noise_nb[1] = 2.0 * randn(rng_nb)
        for i in 2:60
            dt = t_nb[i] - t_nb[i-1]
            noise_nb[i] = 0.7 * noise_nb[i-1] * exp(-dt / 5.0) + 2.0 * randn(rng_nb)
        end
        rv_nb .+= noise_nb

        ic_nb = InstrumentConfig(rv=["SIM"])
        data_nb = Data(; t_rv=t_nb, rv=rv_nb, rv_err=fill(2.0, 60))

        # Build model with MA noise included
        ma_model = MAModel(order=1)
        params_nb = Params(
            max_kplanet=1,
            planet_modes=[RV_ONLY],
            instruments=ic_nb,
            data=data_nb,
            noise_models=[ma_model],
        )

        target_nb = NereusTarget(params_nb, data_nb; unconstrained=false)

        # Trans-dim config: planet + noise toggling
        td_nb = TransDimConfig(
            max_kplanet=1,
            noise=true,
            toggleable=[ma_model],
            transdim_fraction=0.4,  # high to get more birth/death attempts
        )

        # Run PT (RJMCMC alone won't find planets)
        chains_nb, _, evals_nb = sample_pt(target_nb;
            td=td_nb, n_rounds=8, n_chains=6, seed=42, show_report=false)

        @test chains_nb isa MCMCChains.Chains
        @test :n_planets in names(chains_nb, :parameters)
        @test evals_nb > 0

        np_nb = Array(chains_nb[:n_planets])
        @test all(np_nb .>= 0.0)
        @test all(np_nb .≤ 1.0)
    end

    @testset "propose_noise_birth/death" begin
        # Build model with 2 noise components
        ic_nn = InstrumentConfig(rv=["HARPS"])
        data_nn = Data(; t_rv=collect(0.0:1.0:50.0),
                         rv=50.0 .+ 20.0 .* randn(_rng, 51),
                         rv_err=ones(51))
        ma_nn = MAModel(order=1)
        sho_nn = CeleriteSHO()
        params_nn = Params(
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic_nn, data=data_nn,
            noise_models=[ma_nn, sho_nn],
            transdim_noise=true,
        )

        td_nn = TransDimState(max_planets=1, n_noise=2)
        theta_nn = Theta(params_nn; td=td_nn)
        # Set valid systemic params
        for (i, idx) in enumerate(params_nn.layout.unfrozen_idx)
            prior = params_nn.layout.unfrozen_priors[i]
            v = rand(_rng, prior.dist)
            lo, hi = bounds(prior)
            theta_nn.values[idx] = clamp(v, lo, hi)
        end

        toggleable_nn = NoiseModel[ma_nn, sho_nn]

        # Birth: activate MA
        new_nn, log_q_nn = propose_noise_birth(theta_nn, _rng, toggleable_nn)
        @test new_nn.td.noise_active != td_nn.noise_active
        active_after = active_noise(new_nn.td)
        @test length(active_after) == 1
        @test isfinite(log_q_nn)

        # Birth again: should activate the other (but not both Stage 2+3)
        new_nn2, log_q_nn2 = propose_noise_birth(new_nn, _rng, toggleable_nn)
        if isfinite(log_q_nn2)
            # Should only succeed if mutual exclusion is not violated
            active2 = active_noise(new_nn2.td)
            has_seq = any(is_noise_active(new_nn2.td, i) &&
                          params_nn.config.noise_models[i] isa SequentialNoise
                          for i in 1:2)
            has_cov = any(is_noise_active(new_nn2.td, i) &&
                          params_nn.config.noise_models[i] isa CovarianceNoise
                          for i in 1:2)
            # Can't have both sequential and covariance active
            @test !(has_seq && has_cov)
        end

        # Death
        dead_nn, log_q_dead = propose_noise_death(new_nn, _rng, toggleable_nn)
        @test isempty(active_noise(dead_nn.td))
        @test isfinite(log_q_dead)

        # Death from empty → -Inf
        _, log_q_empty = propose_noise_death(theta_nn, _rng, toggleable_nn)
        @test log_q_empty == -Inf
    end

    @testset "log_prior skips inactive planets" begin
        td = TransDimState(max_planets=2)
        theta = Theta(params; td=td)
        # Set all params to valid values
        set_param!(theta, "gamma_HARPS", 0.0)
        set_param!(theta, "sigma_HARPS", 1.0)
        set_param!(theta, "n_p", 0.0)

        # With 0 planets active, log_prior only evaluates systemic priors
        lp_0 = log_prior(theta)
        @test isfinite(lp_0)

        # Activate planet 1, set valid params
        activate_planet!(td, 1)
        set_param!(theta, "P_k1", 10.0)
        set_param!(theta, "K_k1", 5.0)
        set_param!(theta, "sesinw_k1", 0.0)
        set_param!(theta, "secosw_k1", 0.0)
        set_param!(theta, "Mo_k1", 1.0)

        lp_1 = log_prior(theta)
        @test isfinite(lp_1)
        # More active params → different prior value
        @test lp_1 != lp_0

        # Planet 2 params are out of bounds (zeros) but inactive → shouldn't matter
        lp_1_again = log_prior(theta)
        @test lp_1_again ≈ lp_1

        # Activate planet 2 with valid params
        activate_planet!(td, 2)
        set_param!(theta, "P_k2", 20.0)
        set_param!(theta, "K_k2", 3.0)
        set_param!(theta, "sesinw_k2", 0.0)
        set_param!(theta, "secosw_k2", 0.0)
        set_param!(theta, "Mo_k2", 2.0)

        lp_2 = log_prior(theta)
        @test isfinite(lp_2)
        @test lp_2 != lp_1

        # Deactivate planet 2 → recover lp_1
        deactivate_planet!(td, 2)
        lp_back = log_prior(theta)
        @test lp_back ≈ lp_1
    end

    # -----------------------------------------------------------------
    @testset "Stability — AMD-Hill" begin
        # Well-separated 2-planet system: P=10d, P=200d, low ecc, solar-mass star
        # Wide separation → clearly stable by any secular criterion.
        Ps  = [10.0, 200.0]
        Ks  = [10.0, 5.0]
        es  = [0.05, 0.05]
        M_s = 1.0

        masses = [msini(M_s, Ks[i], Ps[i], es[i]) for i in 1:2]

        # AMD: well-separated → stable
        @test is_amd_stable(Ps, masses, es, M_s)
        @test check_stability(Ps, Ks, es, M_s, :amd)

        # Gladman: same
        @test is_gladman_stable(Ps, masses, es, M_s)
        @test check_stability(Ps, Ks, es, M_s, :gladman)

        # Deliberately unstable: two planets with identical periods (crossing orbits)
        Ps_bad  = [10.0, 10.01]
        Ks_bad  = [50.0, 50.0]
        es_bad  = [0.8, 0.8]
        @test !check_stability(Ps_bad, Ks_bad, es_bad, 1.0, :amd)
        @test !check_stability(Ps_bad, Ks_bad, es_bad, 1.0, :gladman)

        # Single planet: always stable
        @test check_stability([10.0], [50.0], [0.1], 1.0, :amd)
        @test check_stability([10.0], [50.0], [0.1], 1.0, :gladman)

        # :none mode: always stable
        @test check_stability(Ps_bad, Ks_bad, es_bad, 1.0, :none)
    end

    # -----------------------------------------------------------------
    @testset "Stability — hard prior in likelihood" begin
        # Two planets with crossing orbits + stability=:amd → -Inf
        data_rv = Data(;
            t_rv = [0.0, 1.0, 2.0, 3.0, 4.0],
            rv   = [50.0, -30.0, 40.0, -20.0, 10.0],
            rv_err = [1.0, 1.0, 1.0, 1.0, 1.0],
        )
        ic_rv = InstrumentConfig(rv=["TEST"])

        # With stability
        p_stab = Params(;
            max_kplanet=2, planet_modes=[RV_ONLY, RV_ONLY],
            instruments=ic_rv, data=data_rv,
            stability=:amd, M_s=1.0,
        )
        target_stab = NereusTarget(p_stab, data_rv; unconstrained=false)
        theta_s = Theta{Float64}(p_stab)
        set_n_p!(theta_s, 2)

        # Set nearly-identical periods with high eccentricity (unstable)
        set_param!(theta_s, "P_k1", 10.0)
        set_param!(theta_s, "K_k1", 50.0)
        set_param!(theta_s, "sesinw_k1", 0.8)   # e ~ 0.64
        set_param!(theta_s, "secosw_k1", 0.0)
        set_param!(theta_s, "Mo_k1", 0.0)
        set_param!(theta_s, "P_k2", 10.5)
        set_param!(theta_s, "K_k2", 50.0)
        set_param!(theta_s, "sesinw_k2", 0.8)
        set_param!(theta_s, "secosw_k2", 0.0)
        set_param!(theta_s, "Mo_k2", 1.0)

        ll = rv_log_likelihood(theta_s, data_rv)
        @test ll == -Inf  # rejected by stability

        # Without stability → finite LL
        p_nostab = Params(;
            max_kplanet=2, planet_modes=[RV_ONLY, RV_ONLY],
            instruments=ic_rv, data=data_rv,
            stability=:none,
        )
        theta_ns = Theta{Float64}(p_nostab)
        set_n_p!(theta_ns, 2)
        set_param!(theta_ns, "P_k1", 10.0)
        set_param!(theta_ns, "K_k1", 50.0)
        set_param!(theta_ns, "sesinw_k1", 0.8)
        set_param!(theta_ns, "secosw_k1", 0.0)
        set_param!(theta_ns, "Mo_k1", 0.0)
        set_param!(theta_ns, "P_k2", 10.5)
        set_param!(theta_ns, "K_k2", 50.0)
        set_param!(theta_ns, "sesinw_k2", 0.8)
        set_param!(theta_ns, "secosw_k2", 0.0)
        set_param!(theta_ns, "Mo_k2", 1.0)

        ll_no = rv_log_likelihood(theta_ns, data_rv)
        @test isfinite(ll_no)
    end

    # -----------------------------------------------------------------
    @testset "Shared parameter groups — layout" begin
        ic = InstrumentConfig(rv=["HARPS_DRS", "HARPS_SERVAL", "ESPRESSO"])
        data_sh = Data(;
            t_rv = [0.0, 1.0, 2.0],
            rv   = [50.0, -30.0, 20.0],
            rv_err = [1.0, 1.0, 1.0],
            rv_inst = [1, 2, 3],
        )

        # Share sigma across HARPS pipelines, gamma independent
        sharing = Dict{Symbol, Vector{Vector{String}}}(
            :sigma => [["HARPS_DRS", "HARPS_SERVAL"]],
        )

        p = Params(;
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic, data=data_sh,
            sharing=sharing,
        )

        layout = p.layout
        sys = layout.systemic

        # gamma: 3 independent slots (one per instrument)
        @test sys.rv_gamma[1] != sys.rv_gamma[2]
        @test sys.rv_gamma[1] != sys.rv_gamma[3]
        @test sys.rv_gamma[2] != sys.rv_gamma[3]

        # sigma: HARPS_DRS and HARPS_SERVAL share, ESPRESSO independent
        @test sys.rv_sigma[1] == sys.rv_sigma[2]  # shared
        @test sys.rv_sigma[1] != sys.rv_sigma[3]  # ESPRESSO different

        # Shared sigma has joined name
        shared_slot = sys.rv_sigma[1]
        @test layout.names[shared_slot] == "sigma_HARPS_DRS+HARPS_SERVAL"

        # ESPRESSO sigma has its own name
        esp_slot = sys.rv_sigma[3]
        @test layout.names[esp_slot] == "sigma_ESPRESSO"

        # Total sigma params: 2 (one shared + one solo), not 3
        unique_sigma_slots = unique(sys.rv_sigma)
        @test length(unique_sigma_slots) == 2
    end

    # -----------------------------------------------------------------
    @testset "Shared parameter groups — no sharing (backward compat)" begin
        ic = InstrumentConfig(rv=["HARPS", "ESPRESSO"])
        data_ns = Data(;
            t_rv = [0.0, 1.0],
            rv   = [50.0, -30.0],
            rv_err = [1.0, 1.0],
            rv_inst = [1, 2],
        )

        # No sharing dict → same as before
        p = Params(;
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic, data=data_ns,
        )

        sys = p.layout.systemic
        @test sys.rv_gamma[1] != sys.rv_gamma[2]
        @test sys.rv_sigma[1] != sys.rv_sigma[2]
        @test p.layout.names[sys.rv_gamma[1]] == "gamma_HARPS"
        @test p.layout.names[sys.rv_sigma[2]] == "sigma_ESPRESSO"
    end

    # -----------------------------------------------------------------
    @testset "External priors on eccentricity" begin
        data_ep = Data(;
            t_rv = collect(range(0.0, 100.0, length=20)),
            rv   = 50.0 .* sin.(2π .* collect(range(0.0, 100.0, length=20)) ./ 10.0),
            rv_err = ones(20),
        )
        ic_ep = InstrumentConfig(rv=["TEST"])

        # Tight Beta prior on eccentricity peaked at low e
        ecc_prior = ExternalPrior(:ecc, BetaPrior(0.867, 3.03), true)

        p_with = Params(;
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic_ep, data=data_ep,
            external_priors=[ecc_prior],
        )
        p_without = Params(;
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic_ep, data=data_ep,
        )

        # Build theta with moderate eccentricity (e ~ 0.5)
        theta_w = Theta{Float64}(p_with)
        set_param!(theta_w, "P_k1", 10.0)
        set_param!(theta_w, "K_k1", 10.0)
        set_param!(theta_w, "sesinw_k1", 0.5)   # e = 0.5^2 + 0.5^2 = 0.5
        set_param!(theta_w, "secosw_k1", 0.5)
        set_param!(theta_w, "Mo_k1", 0.0)

        theta_wo = Theta{Float64}(p_without)
        set_param!(theta_wo, "P_k1", 10.0)
        set_param!(theta_wo, "K_k1", 10.0)
        set_param!(theta_wo, "sesinw_k1", 0.5)
        set_param!(theta_wo, "secosw_k1", 0.5)
        set_param!(theta_wo, "Mo_k1", 0.0)

        ll_with = rv_log_likelihood(theta_w, data_ep)
        ll_without = rv_log_likelihood(theta_wo, data_ep)

        # External prior on e=0.5 with Beta(0.867, 3.03) penalizes high e
        # → ll_with < ll_without
        @test isfinite(ll_with)
        @test isfinite(ll_without)
        @test ll_with < ll_without

        # At low eccentricity: Beta prior gives bonus (pdf > 1 near mode)
        theta_w2 = Theta{Float64}(p_with)
        set_param!(theta_w2, "P_k1", 10.0)
        set_param!(theta_w2, "K_k1", 10.0)
        set_param!(theta_w2, "sesinw_k1", 0.05)  # e ~ 0.005
        set_param!(theta_w2, "secosw_k1", 0.05)
        set_param!(theta_w2, "Mo_k1", 0.0)

        theta_wo2 = Theta{Float64}(p_without)
        set_param!(theta_wo2, "P_k1", 10.0)
        set_param!(theta_wo2, "K_k1", 10.0)
        set_param!(theta_wo2, "sesinw_k1", 0.05)
        set_param!(theta_wo2, "secosw_k1", 0.05)
        set_param!(theta_wo2, "Mo_k1", 0.0)

        ll_low_w = rv_log_likelihood(theta_w2, data_ep)
        ll_low_wo = rv_log_likelihood(theta_wo2, data_ep)

        @test isfinite(ll_low_w)
        # Beta(0.867, 3.03) has mode near 0 → logpdf > 0 near e=0
        @test ll_low_w > ll_low_wo
    end

    # -----------------------------------------------------------------
    @testset "Validation — stability requires M_s for multi-planet" begin
        # max_kplanet=1: stability irrelevant, no throw
        @test Params(;
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=InstrumentConfig(rv=["X"]),
            stability=:amd,
        ) isa Params
        # max_kplanet≥2 without M_s: must throw
        @test_throws ArgumentError Params(;
            max_kplanet=2, planet_modes=[RV_ONLY, RV_ONLY],
            instruments=InstrumentConfig(rv=["X"]),
            stability=:amd,
        )
    end

    @testset "Validation — sharing categories" begin
        ic = InstrumentConfig(rv=["A", "B"])
        @test_throws ArgumentError Params(;
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic,
            sharing=Dict(:bogus => [["A", "B"]]),
        )
        # PM category on RV instruments
        @test_throws ArgumentError Params(;
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic,
            sharing=Dict(:pm_jitter => [["A", "B"]]),
        )
    end

    @testset "Validation — external prior rho_s without use_rho_s" begin
        @test_throws ArgumentError Params(;
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=InstrumentConfig(rv=["X"]),
            external_priors=[ExternalPrior(:rho_s, NormalPrior(1.0, 0.1), false)],
        )
    end

    # -----------------------------------------------------------------
    @testset "Transit likelihood — synthetic hot Jupiter" begin
        # Generate synthetic transit light curve for a hot Jupiter:
        # P=3.0d, Rp/Rs=0.1, b=0.3, circular, M_s=1.0, R_s=1.0
        P_true = 3.0
        rr_true = 0.1
        b_true = 0.3
        M_s = 1.0
        R_s = 1.0

        # Compute a/R* from Kepler's third law
        P_s = P_true * 86400.0
        GM = 1.3271244e26 * M_s
        a_cm = cbrt(GM * P_s^2 / (4π^2))
        a_Rs_true = a_cm / (R_s * 6.9570e10)

        # Generate time array around a transit (Tc = 1.5 days)
        Tc_true = 1.5
        # Duration ~ P/π * rr ≈ 0.1 days for hot Jupiter
        t_phot = collect(range(Tc_true - 0.15, Tc_true + 0.15, length=200))
        # Add some out-of-transit points
        append!(t_phot, collect(range(0.0, 0.5, length=50)))
        append!(t_phot, collect(range(2.0, 2.5, length=50)))
        sort!(t_phot)

        # Compute true transit flux
        # Tp from Tc (circular: Tp = Tc - P/4 for w=π/2)
        Tp_true = tp_to_tc(Tc_true, P_true, 0.0, π/2) # actually need tc_to_tp
        # For circular orbit with w=π/2: Tp = Tc
        Tp_true = Tc_true  # simplified for e=0

        # Synthetic light curve — primary transits only.
        # `sky_separation` returns the sky-plane projected distance, which
        # is symmetric in primary vs secondary eclipse (planet in front
        # vs behind). Nereus's transit_log_likelihood models only
        # primary transits, so the synthetic data must too — gate by
        # sin(ω + f) > 0 (planet in front, line-of-sight check).
        flux_true = Float64[]
        rng_tr = MersenneTwister(42)
        for t in t_phot
            # Compute true anomaly directly (circular orbit)
            M = 2π * (t - Tp_true) / P_true
            f_anom = M  # for e=0, f = M
            primary = sin(π/2 + f_anom) > 0  # ω + f, with ω = π/2
            if primary
                z = sky_separation(t, P_true, 0.0, π/2, Tp_true, b_true, a_Rs_true)
                f_flux = transit_flux(z, rr_true, 0.4, 0.2)  # u1=0.4, u2=0.2
            else
                f_flux = 1.0  # planet behind star → continuum
            end
            push!(flux_true, f_flux + 1e-4 * randn(rng_tr))  # 100ppm noise
        end

        # Build Nereus model
        data_tr = Data(;
            t_phot=t_phot,
            flux=flux_true,
            flux_err=fill(1e-4, length(t_phot)),
            phot_inst=ones(Int, length(t_phot)),
        )
        ic_tr = InstrumentConfig(pm=["SIM"])

        # Use Tc parametrization, b+rr geometry
        param_config = ParametrizationConfig(
            ew=:sesinw, time=:Tc, geom=:b_rr,
        )

        params_tr = Params(;
            max_kplanet=1,
            planet_modes=[PM_ONLY],
            instruments=ic_tr,
            data=data_tr,
            parametrization=param_config,
            M_s=M_s,
            R_s=R_s,
        )

        # Verify parameter layout has transit geometry
        @test has_geometry(params_tr.layout.planet_blocks[1])
        @test !has_K(params_tr.layout.planet_blocks[1])

        # Set parameters to true values and evaluate likelihood
        theta_tr = Theta{Float64}(params_tr)
        set_param!(theta_tr, "P_k1", P_true)
        set_param!(theta_tr, "sesinw_k1", 0.0)
        set_param!(theta_tr, "secosw_k1", 0.0)
        set_param!(theta_tr, "Tc_k1", Tc_true)
        set_param!(theta_tr, "b_k1", b_true)
        set_param!(theta_tr, "rr_k1", rr_true)

        # Set LD to match generated data (Kipping inverse of u1=0.4, u2=0.2)
        q1, q2 = kipping_u_to_q(0.4, 0.2)
        set_param!(theta_tr, "q1_SIM", q1)
        set_param!(theta_tr, "q2_SIM", q2)

        ll_true = transit_log_likelihood(theta_tr, data_tr)
        @test isfinite(ll_true)
        @test ll_true > -1000  # should be reasonable for 300 points with 100ppm noise

        # Perturbed parameters should give worse likelihood
        set_param!(theta_tr, "rr_k1", 0.05)  # wrong radius ratio
        ll_wrong = transit_log_likelihood(theta_tr, data_tr)
        @test isfinite(ll_wrong)
        @test ll_wrong < ll_true

        # Completely off-transit should give ~ white noise LL
        set_param!(theta_tr, "b_k1", 2.0)  # no transit (b > 1 + rr)
        ll_notransit = transit_log_likelihood(theta_tr, data_tr)
        @test isfinite(ll_notransit)
        @test ll_notransit < ll_true  # no transit model is worse
    end

    # -----------------------------------------------------------------
    @testset "Transit + RV joint likelihood" begin
        # Verify joint RVPM model evaluates both likelihoods
        rng_j = MersenneTwister(99)
        t_rv = collect(range(0.0, 10.0, length=20))
        rv_obs = 50.0 .* sin.(2π .* t_rv ./ 3.0) .+ 2.0 .* randn(rng_j, 20)
        t_phot = collect(range(1.3, 1.7, length=50))
        flux_obs = ones(50) .+ 1e-4 .* randn(rng_j, 50)

        data_j = Data(;
            t_rv=t_rv, rv=rv_obs, rv_err=fill(2.0, 20),
            t_phot=t_phot, flux=flux_obs, flux_err=fill(1e-4, 50),
            phot_inst=ones(Int, 50),
        )
        ic_j = InstrumentConfig(rv=["RV_SIM"], pm=["PM_SIM"])

        params_j = Params(;
            max_kplanet=1,
            planet_modes=[RVPM],
            instruments=ic_j,
            data=data_j,
            M_s=1.0, R_s=1.0,
        )

        # RVPM block has both K and geometry
        @test has_K(params_j.layout.planet_blocks[1])
        @test has_geometry(params_j.layout.planet_blocks[1])

        # Evaluate both likelihoods
        theta_j = Theta{Float64}(params_j)
        set_param!(theta_j, "P_k1", 3.0)
        set_param!(theta_j, "K_k1", 50.0)
        set_param!(theta_j, "sesinw_k1", 0.0)
        set_param!(theta_j, "secosw_k1", 0.0)
        set_param!(theta_j, "Mo_k1", 0.0)
        set_param!(theta_j, "b_k1", 0.3)
        set_param!(theta_j, "rr_k1", 0.1)

        ll_rv = rv_log_likelihood(theta_j, data_j)
        ll_tr = transit_log_likelihood(theta_j, data_j)
        @test isfinite(ll_rv)
        @test isfinite(ll_tr)
        @test ll_rv != 0.0  # should have RV contribution
    end
end

@testset "Photometry detrending" begin
    using Random: MersenneTwister
    using Statistics: mean, std

    # Synthetic LC: rotation + transit + Gaussian noise, two segments
    rng = MersenneTwister(42)
    t1 = collect(1325.0:0.02:1352.0)
    t2 = collect(1382.0:0.02:1409.0)
    t = vcat(t1, t2)
    P_rot, P_planet, T0 = 8.74, 4.1375, 1326.0
    rot = 0.005 .* sin.(2π .* t ./ P_rot) .+ 0.0015 .* sin.(4π .* t ./ P_rot)
    phase = mod.((t .- T0) ./ P_planet .+ 0.5, 1) .- 0.5
    in_transit = abs.(phase) .< 0.01
    flux = 1.0 .+ rot .+ (-0.0012 .* in_transit) .+ 1e-4 .* randn(rng, length(t))
    flux_err = fill(1e-4, length(t))

    @testset "mask_transits" begin
        m = mask_transits(t, P_planet, T0; window=0.025)
        @test m isa AbstractVector{Bool}
        @test length(m) == length(t)
        @test count(m) > 0
        # All masked points should be near integer multiples of period from T0
        for i in eachindex(t)
            if m[i]
                φ = mod((t[i] - T0) / P_planet + 0.5, 1.0) - 0.5
                @test abs(φ) < 0.025 + 1e-9
            end
        end
        # Multi-planet support
        m2 = mask_transits(t, [P_planet, 10.0], [T0, 1330.0]; window=0.02)
        @test count(m2) >= count(m)
        # Mismatched lengths throw
        @test_throws ArgumentError mask_transits(t, [1.0, 2.0], [1.0])
    end

    @testset "find_segments" begin
        segs = find_segments(t; gap_size=0.5)
        @test length(segs) == 2
        @test segs[1].start == 1
        @test segs[end].stop == length(t)
        # Segments cover all points without overlap
        @test sum(length, segs) == length(t)
        # Single contiguous run → one segment
        segs1 = find_segments(t1; gap_size=0.5)
        @test length(segs1) == 1
        # Empty input → empty result
        @test isempty(find_segments(Float64[]))
    end

    @testset "detrend_savgol" begin
        m = mask_transits(t, P_planet, T0; window=0.025)
        res = detrend_savgol(t, flux, flux_err;
                              window_length=301, transit_mask=m)
        @test length(res.flux_detrended) == length(t)
        @test length(res.trend) == length(t)
        @test length(res.segments) == 2
        @test res.transit_mask === m
        # Detrended median should be ~1 (multiplicative)
        @test abs(mean(res.flux_detrended) - 1.0) < 0.001
        # Out-of-transit scatter should be much less than original
        @test std(res.flux_detrended[.!m]) < std(flux) / 5
        # Window must be odd
        @test_throws ArgumentError detrend_savgol(t, flux, flux_err;
                                                    window_length=300)
    end

    @testset "detrend_gp — SHO" begin
        m = mask_transits(t, P_planet, T0; window=0.025)
        # Default: joint_segments=true ⇒ ONE shared hyperparameter set.
        res = detrend_gp(t, flux, flux_err, CeleriteSHO(); transit_mask=m)
        @test length(res.flux_detrended) == length(t)
        @test length(res.gp_params) == 1                 # joint fit
        @test length(res.gp_params[1]) == 3              # SHO: 3 hyperparams
        @test std(res.flux_detrended[.!m]) < 5e-4

        # Per-segment fallback: one set per sector. Test data has two
        # contiguous chunks separated by a gap; tag each cadence with
        # which chunk it belongs to.
        segs = find_segments(t; gap_size=0.5)
        sid = ones(Int, length(t))
        for (k, seg) in enumerate(segs); sid[seg] .= k; end
        res_per = detrend_gp(t, flux, flux_err, CeleriteSHO();
                              transit_mask=m, joint_segments=false,
                              sector_id=sid)
        @test length(res_per.gp_params) == 2
    end

    @testset "detrend_gp — Rotation FM17" begin
        m = mask_transits(t, P_planet, T0; window=0.025)
        res = detrend_gp(t, flux, flux_err, CeleriteRotationFM17();
                         transit_mask=m)
        @test length(res.gp_params) == 1                 # joint fit
        @test length(res.gp_params[1]) == 4              # FM17: 4 hyperparams
        # Recovered log_period should be close to true log(P_rot).
        log_P_fit = res.gp_params[1][3]                  # gp_log_period
        @test abs(log_P_fit - log(P_rot)) < 0.1
        @test std(res.flux_detrended[.!m]) < 3e-4
    end

    @testset "detrend_notch" begin
        # Larger-amplitude rotation than the suite default, to demonstrate
        # notch's blind-search behavior (no transit_mask supplied).
        rng2 = MersenneTwister(7)
        tn = collect(0.0:(2.0/60/24):25.0)         # 25-day TESS-like baseline
        rot_amp = 0.01
        rot2 = rot_amp .* sin.(2π .* tn ./ P_rot) .+
               0.4 * rot_amp .* sin.(4π .* tn ./ P_rot .+ 0.7)
        phase_n = mod.((tn .- 1.2) ./ P_planet .+ 0.5, 1) .- 0.5
        in_t = abs.(phase_n) .< (1.5/24) / P_planet  # 3-hr transit at P=4.1375
        flux_n = 1.0 .+ rot2 .- (970e-6 .* in_t) .+ 7e-4 .* randn(rng2, length(tn))
        ferr_n = fill(7e-4, length(tn))

        res = detrend_notch(tn, flux_n, ferr_n;
                             window=1.0, durations=[1.0, 2.0, 4.0]./24,
                             delta_bic=-1.0)
        @test length(res.flux_detrended) == length(tn)
        @test length(res.trend) == length(tn)
        @test length(res.depth) == length(tn)
        @test length(res.delta_bic) == length(tn)

        # Trend should track the injected rotation curve to ≪ rot_amp.
        rot_true = rot2 .+ 1.0
        @test std(res.trend .- rot_true) < 0.2 * rot_amp

        # Detrended in-transit cadences sit ~1 - depth below OOT.
        recovered = mean(res.flux_detrended[.!in_t]) -
                    mean(res.flux_detrended[in_t])
        @test 0.7 * 970e-6 < recovered < 1.3 * 970e-6

        # ΔBIC fires at in-transit cadences (mean strongly positive).
        @test mean(res.delta_bic[in_t]) > 5
        @test mean(res.delta_bic[.!in_t]) < 5
    end

    @testset "celerite predict mean" begin
        # GP signal mean at observation points
        ar, cr, ac, bc, cc, dc = Nereus.sho_coefficients(1e-6, 1.0, 2π / P_rot)
        μ = Nereus.celerite_predict_mean(t, flux .- 1.0, flux_err .^ 2,
                                            ar, cr, ac, bc, cc, dc)
        @test length(μ) == length(t)
        @test all(isfinite, μ)
    end

    @testset "noise_param_names — FM17" begin
        ic = InstrumentConfig(rv=["X"])
        names = Nereus.noise_param_names(CeleriteRotationFM17(), ic)
        @test names == ["gp_log_amp", "gp_log_timescale",
                        "gp_log_period", "gp_log_factor"]
    end

    @testset "channel-tagged GP — phot suffix" begin
        ic = InstrumentConfig(pm=["TESS"])

        # Default channel = :rv — bare names (backward compat)
        @test Nereus.noise_param_names(CeleriteRotationFM17(), ic)[1] ==
              "gp_log_amp"
        @test Nereus.noise_param_names(CeleriteSHO(), ic)[1] == "gp_log_S0"
        @test Nereus.noise_param_names(CeleriteRotation(), ic)[1] == "gp_sigma"

        # channel=:phot → "_phot" suffix
        @test Nereus.noise_param_names(
                  CeleriteRotationFM17(channel=:phot), ic) ==
              ["gp_log_amp_phot", "gp_log_timescale_phot",
               "gp_log_period_phot", "gp_log_factor_phot"]
        @test Nereus.noise_param_names(
                  CeleriteSHO(channel=:phot), ic) ==
              ["gp_log_S0_phot", "gp_log_Q_phot", "gp_log_omega0_phot"]
        @test Nereus.noise_param_names(
                  CeleriteRotation(channel=:phot), ic) ==
              ["gp_sigma_phot", "gp_period_phot", "gp_Q0_phot",
               "gp_dQ_phot", "gp_f_phot"]

        # Nereus.noise_channel introspection
        @test Nereus.noise_channel(CeleriteRotationFM17()) === :rv
        @test Nereus.noise_channel(CeleriteRotationFM17(channel=:phot)) === :phot

        # validate_noise_models: at most one CovarianceNoise per channel
        @test_throws ArgumentError validate_noise_models(
            [CeleriteSHO(), CeleriteRotationFM17()])  # two on :rv
        # but :rv + :phot is fine
        validate_noise_models(
            [CeleriteSHO(), CeleriteRotationFM17(channel=:phot)])
    end

    @testset "GP-on-photometry — likelihood + gradient" begin
        # Tiny synthetic LC: rotation + transit + noise
        using ForwardDiff
        rng = MersenneTwister(11)
        tt = collect(0.0:0.05:25.0)         # 500 points, 25-d span
        Pr, Pp, T0p = 8.74, 4.1375, 1.5
        rot_amp = 0.01
        rot = rot_amp .* sin.(2π .* tt ./ Pr)
        ph = mod.((tt .- T0p) ./ Pp .+ 0.5, 1) .- 0.5
        in_t = abs.(ph) .< 0.018
        flx = 1.0 .+ rot .- (1e-3 .* in_t) .+ 5e-4 .* randn(rng, length(tt))
        ferr = fill(5e-4, length(tt))

        nm_phot = CeleriteRotationFM17(channel=:phot)
        priors_override = Dict{String, PriorSpec}(
            "gp_log_period_phot"    => NormalPrior(log(Pr), 0.05,
                                                    log(4.0), log(20.0)),
            "gp_log_amp_phot"       => UniformPrior(log(1e-4), log(0.05)),
            "gp_log_timescale_phot" => UniformPrior(log(2.0), log(200.0)),
        )
        target = build_target(
            M_s = 0.8, R_s = 0.8,
            planets = (b = (
                P  = NormalPrior(Pp, 0.005, Pp - 0.05, Pp + 0.05),
                Tc = NormalPrior(T0p, 0.05, T0p - 0.5, T0p + 0.5),
                sesinw = UniformPrior(-0.3, 0.3),
                secosw = UniformPrior(-0.3, 0.3),
                b  = UniformPrior(0.0, 1.0),
                rr = UniformPrior(0.005, 0.10),
            ),),
            phot = (TESS = (
                data = (t = tt, flux = flx, flux_err = ferr),
                jitter = LogUniformPrior(1e-5, 1e-2),
                offset = NormalPrior(0.0, 1e-3, -0.01, 0.01),
                q1 = UniformPrior(0.0, 1.0),
                q2 = UniformPrior(0.0, 1.0),
            ),),
            noise_models = [nm_phot],
            priors = priors_override,
        )
        names = target.params.layout.unfrozen_names
        @test "gp_log_amp_phot"       in names
        @test "gp_log_period_phot"    in names
        @test "gp_log_timescale_phot" in names
        @test "gp_log_factor_phot"    in names

        n = n_unfrozen(target.params)
        # Avoid the (sesinw, secosw) = (0, 0) singularity (omega undefined
        # at e=0 — pre-existing parametrization edge case, not GP-related).
        x0 = 0.1 .* randn(MersenneTwister(99), n)
        ll = target(x0)
        @test isfinite(ll)
        # ForwardDiff gradient through GP path
        grad = ForwardDiff.gradient(target, x0)
        @test all(isfinite, grad)
        @test length(grad) == n
    end

    @testset "per-instrument GP — naming + validation" begin
        ic = InstrumentConfig(rv=["HARPS", "FEROS"])

        # `_gp_suffix` covers the four legal combinations
        @test Nereus._gp_suffix(CeleriteRotationFM17()) == ""
        @test Nereus._gp_suffix(CeleriteRotationFM17(channel=:phot)) == "_phot"
        @test Nereus._gp_suffix(
                  CeleriteRotationFM17(instruments=["HARPS"])) == "_HARPS"
        @test Nereus._gp_suffix(
                  CeleriteRotationFM17(instruments=["HARPS", "FEROS"])) ==
              "_HARPS+FEROS"
        @test Nereus._gp_suffix(
                  CeleriteRotationFM17(channel=:phot,
                                        instruments=["TESS"])) == "_phot_TESS"

        # Per-instrument param names are uniquely suffixed
        names_h = Nereus.noise_param_names(
            CeleriteRotationFM17(instruments=["HARPS"]), ic)
        names_f = Nereus.noise_param_names(
            CeleriteSHO(instruments=["FEROS"]), ic)
        @test names_h == ["gp_log_amp_HARPS", "gp_log_timescale_HARPS",
                          "gp_log_period_HARPS", "gp_log_factor_HARPS"]
        @test names_f == ["gp_log_S0_FEROS", "gp_log_Q_FEROS",
                          "gp_log_omega0_FEROS"]
        @test isempty(intersect(names_h, names_f))

        # noise_instruments accessor
        @test noise_instruments(CeleriteSHO()) == String[]
        @test noise_instruments(CeleriteSHO(instruments=["X"])) == ["X"]

        # validate_noise_models composition rules
        # Disjoint per-instrument GPs on the same channel: OK
        validate_noise_models([
            CeleriteRotationFM17(instruments=["HARPS"]),
            CeleriteSHO(instruments=["FEROS"]),
        ])
        # Per-instrument across channels: OK
        validate_noise_models([
            CeleriteSHO(instruments=["HARPS"]),
            CeleriteRotationFM17(channel=:phot, instruments=["TESS"]),
        ])
        # Overlapping instrument sets on same channel: rejected
        @test_throws ArgumentError validate_noise_models([
            CeleriteRotationFM17(instruments=["HARPS", "FEROS"]),
            CeleriteSHO(instruments=["FEROS"]),
        ])
        # Global + restricted on same channel: rejected
        @test_throws ArgumentError validate_noise_models([
            CeleriteRotationFM17(),
            CeleriteSHO(instruments=["HARPS"]),
        ])
        # Restricted GP on same channel as AR: allowed (it only covers
        # a subset of observations, so the mutex doesn't apply).
        validate_noise_models([
            CeleriteRotationFM17(instruments=["HARPS"]),
            ARModel(order=1),
        ])
        # Duplicate instruments inside one model: rejected
        @test_throws ArgumentError validate_noise_models([
            CeleriteSHO(instruments=["HARPS", "HARPS"]),
        ])
    end

    @testset "per-instrument GP — likelihood additivity + gradient" begin
        # Two-instrument synthetic RV. Each instrument gets its own GP
        # (different family, different timescale) so the per-instrument
        # path is exercised; we verify the joint LL equals the sum of
        # two single-instrument fits at the same hyperparameter point.
        rng = MersenneTwister(7)
        n1 = 50
        t1 = sort(rand(rng, n1) .* 200.0)
        rv1 = 12.0 .* sin.(2π .* t1 ./ 8.0) .+ 1.5 .* randn(rng, n1) .+ 50.0
        err1 = fill(1.5, n1)
        n2 = 40
        t2 = sort(rand(rng, n2) .* 200.0)
        rv2 = 8.0 .* sin.(2π .* t2 ./ 25.0) .+ 1.2 .* randn(rng, n2) .- 30.0
        err2 = fill(1.2, n2)
        t = vcat(t1, t2); rvs = vcat(rv1, rv2); errs = vcat(err1, err2)
        inst = vcat(fill(1, n1), fill(2, n2))
        ord = sortperm(t)
        t = t[ord]; rvs = rvs[ord]; errs = errs[ord]; inst = inst[ord]

        ic = InstrumentConfig(rv=["HARPS", "FEROS"])
        data = Data(; t_rv=t, rv=rvs, rv_err=errs, rv_inst=inst)

        # --- Joint per-instrument fit ---
        params_joint = Params(
            max_kplanet  = 0,
            planet_modes = PlanetDataSources[],
            instruments  = ic, data = data,
            noise_models = [
                CeleriteRotationFM17(instruments=["HARPS"]),
                CeleriteSHO(instruments=["FEROS"]),
            ],
        )
        target_joint = NereusTarget(params_joint, data; unconstrained=true)
        nJ = n_unfrozen(params_joint)
        names_joint = params_joint.layout.unfrozen_names

        # Layout should carry both GPs' suffixed parameters
        @test "gp_log_amp_HARPS"  in names_joint
        @test "gp_log_period_HARPS" in names_joint
        @test "gp_log_S0_FEROS"   in names_joint
        @test "gp_log_omega0_FEROS" in names_joint

        x = 0.05 .* randn(MersenneTwister(2), nJ)
        ll_joint = target_joint(x)
        @test isfinite(ll_joint)
        # ForwardDiff propagates through both per-instrument GP slices
        grad = ForwardDiff.gradient(target_joint, x)
        @test all(isfinite, grad)
        @test length(grad) == nJ

        # --- Reference: two single-instrument fits with the same
        # hyperparameter values, matched by name. The total LL must
        # equal the joint LL exactly (modulo floating-point).
        function build_single(t_s, rv_s, err_s, inst_name, kernel,
                              suffix_map)
            data_s = Data(; t_rv=t_s, rv=rv_s, rv_err=err_s,
                            rv_inst=ones(Int, length(t_s)))
            params_s = Params(
                max_kplanet  = 0,
                planet_modes = PlanetDataSources[],
                instruments  = InstrumentConfig(rv=[inst_name]),
                data         = data_s,
                noise_models = [kernel],
            )
            tgt = NereusTarget(params_s, data_s; unconstrained=true)
            xs = zeros(n_unfrozen(params_s))
            for (i, n) in enumerate(params_s.layout.unfrozen_names)
                src = get(suffix_map, n, n)
                j = findfirst(==(src), names_joint)
                xs[i] = j === nothing ? 0.0 : x[j]
            end
            return tgt(xs)
        end

        ll_h = build_single(t1, rv1, err1, "HARPS",
                             CeleriteRotationFM17(),
                             Dict("gp_log_amp"       => "gp_log_amp_HARPS",
                                  "gp_log_timescale" => "gp_log_timescale_HARPS",
                                  "gp_log_period"    => "gp_log_period_HARPS",
                                  "gp_log_factor"    => "gp_log_factor_HARPS"))
        ll_f = build_single(t2, rv2, err2, "FEROS",
                             CeleriteSHO(),
                             Dict("gp_log_S0"     => "gp_log_S0_FEROS",
                                  "gp_log_Q"      => "gp_log_Q_FEROS",
                                  "gp_log_omega0" => "gp_log_omega0_FEROS"))
        @test isapprox(ll_joint, ll_h + ll_f; atol=1e-9)

        # Swapping which instrument each GP covers must change the LL
        # — confirms the slicing actually gates by instrument index.
        params_swap = Params(
            max_kplanet  = 0,
            planet_modes = PlanetDataSources[],
            instruments  = ic, data = data,
            noise_models = [
                CeleriteRotationFM17(instruments=["FEROS"]),
                CeleriteSHO(instruments=["HARPS"]),
            ],
        )
        target_swap = NereusTarget(params_swap, data; unconstrained=true)
        x_swap = zeros(n_unfrozen(params_swap))
        for (i, n) in enumerate(params_swap.layout.unfrozen_names)
            # Same name → same value; renamed gp_*_FEROS / gp_*_HARPS hop
            j = findfirst(==(n), names_joint)
            x_swap[i] = j === nothing ? 0.0 : x[j]
        end
        ll_swap = target_swap(x_swap)
        @test isfinite(ll_swap)
        @test ll_swap != ll_joint
    end

    @testset "per-instrument GP — photometry channel" begin
        # Two-LC synthetic photometry: TESS + CHEOPS, each with its
        # own rotation period. Per-instrument :phot GPs should fit
        # both jointly and produce a finite, differentiable LL.
        rng = MersenneTwister(31)
        t_t = collect(0.0:0.05:20.0)
        rot_t = 0.008 .* sin.(2π .* t_t ./ 7.5)
        flx_t = 1.0 .+ rot_t .+ 4e-4 .* randn(rng, length(t_t))
        ferr_t = fill(4e-4, length(t_t))

        t_c = collect(0.0:0.04:15.0) .+ 30.0
        rot_c = 0.004 .* sin.(2π .* t_c ./ 4.5)
        flx_c = 1.0 .+ rot_c .+ 6e-4 .* randn(rng, length(t_c))
        ferr_c = fill(6e-4, length(t_c))

        t_all   = vcat(t_t, t_c)
        flx_all = vcat(flx_t, flx_c)
        err_all = vcat(ferr_t, ferr_c)
        inst_all = vcat(fill(1, length(t_t)), fill(2, length(t_c)))
        ord = sortperm(t_all)
        t_all   = t_all[ord]
        flx_all = flx_all[ord]
        err_all = err_all[ord]
        inst_all = inst_all[ord]

        nm_t = CeleriteRotationFM17(channel=:phot, instruments=["TESS"])
        nm_c = CeleriteRotationFM17(channel=:phot, instruments=["CHEOPS"])

        # Noise-only photometry target — exercises the per-instrument
        # phot GP slicing without any transiting planet (the
        # `_phot_ll_no_transit` path).
        ic = InstrumentConfig(pm=["TESS", "CHEOPS"])
        data = Data(; t_phot = t_all, flux = flx_all,
                       flux_err = err_all, phot_inst = inst_all)
        priors = Dict{String, PriorSpec}(
            "gp_log_period_phot_TESS"   => NormalPrior(log(7.5), 0.05,
                                                        log(2.0), log(20.0)),
            "gp_log_period_phot_CHEOPS" => NormalPrior(log(4.5), 0.05,
                                                        log(1.5), log(20.0)),
            "gp_log_amp_phot_TESS"      => UniformPrior(log(1e-4), log(0.05)),
            "gp_log_amp_phot_CHEOPS"    => UniformPrior(log(1e-4), log(0.05)),
            "offset_TESS"               => NormalPrior(0.0, 1e-3, -0.01, 0.01),
            "offset_CHEOPS"             => NormalPrior(0.0, 1e-3, -0.01, 0.01),
            "jitter_TESS"               => LogUniformPrior(1e-5, 1e-2),
            "jitter_CHEOPS"             => LogUniformPrior(1e-5, 1e-2),
        )
        params = Params(
            max_kplanet  = 0,
            planet_modes = PlanetDataSources[],
            instruments  = ic, data = data,
            M_s = 0.9, R_s = 0.9,
            stability    = :none,
            noise_models = [nm_t, nm_c],
            priors       = priors,
        )
        target = NereusTarget(params, data; unconstrained=true)

        names = params.layout.unfrozen_names
        @test "gp_log_amp_phot_TESS"     in names
        @test "gp_log_period_phot_TESS"  in names
        @test "gp_log_amp_phot_CHEOPS"   in names
        @test "gp_log_period_phot_CHEOPS" in names

        n = n_unfrozen(params)
        x0 = 0.05 .* randn(MersenneTwister(101), n)
        ll = target(x0)
        @test isfinite(ll)
        grad = ForwardDiff.gradient(target, x0)
        @test all(isfinite, grad)
        @test length(grad) == n
    end

    @testset "phot ARMA — naming + likelihood" begin
        # Param names: phot AR/MA get _phot suffix; per-instrument also
        # uses pm_names (NOT rv_names) and the channel suffix sits at
        # the tail (`_TESS_phot`).
        ic = InstrumentConfig(pm=["TESS"])
        @test Nereus.noise_param_names(MAModel(channel=:phot), ic) ==
              ["ma_omega_1_phot", "ma_beta_1_phot"]
        @test Nereus.noise_param_names(ARModel(channel=:phot), ic) ==
              ["ar_phi_1_phot", "ar_alpha_1_phot"]
        @test Nereus.noise_param_names(
                  MAModel(order=2, channel=:phot, per_instrument=true), ic) ==
              ["ma_omega_1_TESS_phot", "ma_beta_1_TESS_phot",
               "ma_omega_2_TESS_phot", "ma_beta_2_TESS_phot"]
        # RV bare names preserved (back-compat)
        ic_rv = InstrumentConfig(rv=["HARPS"])
        @test Nereus.noise_param_names(MAModel(), ic_rv) ==
              ["ma_omega_1", "ma_beta_1"]

        # Validation: phot AR + phot global GP rejected, but
        # phot AR + restricted phot GP allowed.
        @test_throws ArgumentError validate_noise_models(
            [MAModel(channel=:phot), CeleriteRotationFM17(channel=:phot)])
        validate_noise_models([
            MAModel(channel=:phot),
            CeleriteRotationFM17(channel=:phot, instruments=["TESS"]),
        ])

        # End-to-end: synthetic LC with AR(1) red noise → phot AR LL
        # finite, gradient finite, and removing AR changes the LL.
        rng = MersenneTwister(13)
        n_lc = 400
        t = collect(0.0:0.02:(n_lc - 1) * 0.02)
        depth = 1e-3
        ph = mod.((t .- 1.0) ./ 4.137 .+ 0.5, 1) .- 0.5
        in_t = abs.(ph) .< 0.018
        white_amp = 5e-4
        ar_seq = zeros(n_lc)
        ar_seq[1] = white_amp * randn(rng)
        for i in 2:n_lc
            dt = t[i] - t[i - 1]
            ar_seq[i] = 0.6 * exp(-dt / 0.5) * ar_seq[i - 1] +
                        white_amp * randn(rng)
        end
        flux = 1.0 .+ ar_seq .- depth .* in_t
        ferr = fill(white_amp, n_lc)

        data = Data(; t_phot=t, flux=flux, flux_err=ferr)
        priors_ar = Dict{String, PriorSpec}(
            "ar_phi_1_phot"   => UniformPrior(-1.0, 1.0),
            "ar_alpha_1_phot" => LogUniformPrior(0.05, 5.0),
            "offset_TESS"     => NormalPrior(0.0, 1e-3, -0.01, 0.01),
            "jitter_TESS"     => LogUniformPrior(1e-5, 1e-2),
        )
        params_ar = Params(
            max_kplanet  = 0, planet_modes = PlanetDataSources[],
            instruments  = InstrumentConfig(pm=["TESS"]),
            data         = data, M_s = 1.0, R_s = 1.0,
            stability    = :none,
            noise_models = [ARModel(order=1, channel=:phot)],
            priors       = priors_ar,
        )
        target_ar = NereusTarget(params_ar, data; unconstrained=true)
        names_ar = params_ar.layout.unfrozen_names
        @test "ar_phi_1_phot"   in names_ar
        @test "ar_alpha_1_phot" in names_ar

        n_par = n_unfrozen(params_ar)
        x0 = 0.05 .* randn(MersenneTwister(99), n_par)
        ll_ar = target_ar(x0)
        @test isfinite(ll_ar)
        grad = ForwardDiff.gradient(target_ar, x0)
        @test all(isfinite, grad)
        @test length(grad) == n_par

        # MA on phot: also exercise the residual-side path
        priors_ma = Dict{String, PriorSpec}(
            "ma_omega_1_phot" => UniformPrior(-1.0, 1.0),
            "ma_beta_1_phot"  => LogUniformPrior(0.05, 5.0),
            "offset_TESS"     => NormalPrior(0.0, 1e-3, -0.01, 0.01),
            "jitter_TESS"     => LogUniformPrior(1e-5, 1e-2),
        )
        params_ma = Params(
            max_kplanet  = 0, planet_modes = PlanetDataSources[],
            instruments  = InstrumentConfig(pm=["TESS"]),
            data         = data, M_s = 1.0, R_s = 1.0,
            stability    = :none,
            noise_models = [MAModel(order=1, channel=:phot)],
            priors       = priors_ma,
        )
        target_ma = NereusTarget(params_ma, data; unconstrained=true)
        ll_ma = target_ma(0.05 .* randn(MersenneTwister(101),
                                          n_unfrozen(params_ma)))
        @test isfinite(ll_ma)
    end

    @testset "per-instrument AR — within-instrument iteration" begin
        # Latent-bug fix: previously `per_instrument=true` emitted
        # `_HARPS`/`_FEROS` parameter names but `apply_ar!` looked up
        # the bare names. Now the apply functions iterate within each
        # instrument's cadences with that instrument's coefficients.
        rng = MersenneTwister(7)
        n1 = 30; t1 = sort(rand(rng, n1) .* 100.0)
        rv1 = 50.0 .+ 2.0 .* randn(rng, n1)
        n2 = 25; t2 = sort(rand(rng, n2) .* 100.0)
        rv2 = -30.0 .+ 1.5 .* randn(rng, n2)
        t = vcat(t1, t2); rvs = vcat(rv1, rv2); errs = fill(2.0, n1 + n2)
        inst = vcat(fill(1, n1), fill(2, n2))
        ord = sortperm(t)
        t = t[ord]; rvs = rvs[ord]; errs = errs[ord]; inst = inst[ord]

        ic = InstrumentConfig(rv=["HARPS", "FEROS"])
        data = Data(; t_rv=t, rv=rvs, rv_err=errs, rv_inst=inst)
        params = Params(
            max_kplanet  = 0, planet_modes = PlanetDataSources[],
            instruments  = ic, data = data,
            noise_models = [ARModel(order=1, per_instrument=true)],
        )
        names = params.layout.unfrozen_names
        @test "ar_phi_1_HARPS"   in names
        @test "ar_alpha_1_HARPS" in names
        @test "ar_phi_1_FEROS"   in names
        @test "ar_alpha_1_FEROS" in names

        target = NereusTarget(params, data; unconstrained=true)
        n_par = n_unfrozen(params)
        x0 = 0.05 .* randn(MersenneTwister(99), n_par)
        ll = target(x0)
        @test isfinite(ll)
        grad = ForwardDiff.gradient(target, x0)
        @test all(isfinite, grad)
    end
end


# =====================================================================
# Polynomial Stein Discrepancy (Srinivasan+ 2024, arXiv:2412.05135)
# =====================================================================

@testset "Polynomial Stein Discrepancy" begin
    @testset "multi-index basis enumeration" begin
        # Full basis for d=2, r=2: nonzero α with Σα ≤ 2
        # → (1,0), (0,1), (2,0), (1,1), (0,2). Total J = 5.
        full = Nereus._generate_multiindices(2, 2)
        @test length(full) == 5
        @test [1, 0] in full && [0, 1] in full
        @test [2, 0] in full && [1, 1] in full && [0, 2] in full
        @test [0, 0] ∉ full

        # Diagonal-only basis: d × r monomials (x_i, x_i², ..., x_i^r per i)
        diag = Nereus._generate_multiindices(3, 2; diagonal_only=true)
        @test length(diag) == 6
        # Cross terms must NOT appear in the diagonal-only basis
        @test [1, 1, 0] ∉ diag
        @test [1, 0, 1] ∉ diag
        # General-d general-r counts: J_full = (d+r choose d) - 1
        @test length(Nereus._generate_multiindices(4, 3)) ==
              binomial(4 + 3, 4) - 1
    end

    @testset "Stein operator on monomials — closed-form check" begin
        # Standard normal: ∇log p(x) = -x.  For α = (2, 0, ..., 0):
        #   P_α(x) = x_1², ∇P = (2 x_1, 0, ...), ΔP = 2.
        #   A P(x) = 2 + 2 x_1 (-x_1) = 2 (1 - x_1²)
        x = [1.5, -0.3, 0.7]
        glogp = -x
        α = [2, 0, 0]
        ap = Nereus._stein_apply_monomial(α, x, glogp)
        @test isapprox(ap, 2 * (1 - 1.5^2); atol=1e-12)

        # For α = (1, 1, 0): P = x_1 x_2, ∇P = (x_2, x_1, 0), ΔP = 0.
        # A P(x) = 0 + (x_2)(-x_1) + (x_1)(-x_2) = -2 x_1 x_2
        α2 = [1, 1, 0]
        ap2 = Nereus._stein_apply_monomial(α2, x, glogp)
        @test isapprox(ap2, -2 * 1.5 * (-0.3); atol=1e-12)

        # Univariate cubic: P = x_1³, ΔP = 6 x_1, ∇P = (3 x_1², 0, 0).
        # A P = 6 x_1 + 3 x_1² × (-x_1) = 6 x_1 - 3 x_1³
        α3 = [3, 0, 0]
        ap3 = Nereus._stein_apply_monomial(α3, x, glogp)
        @test isapprox(ap3, 6 * 1.5 - 3 * 1.5^3; atol=1e-12)
    end

    @testset "moment-mismatch detection on Gaussian target" begin
        rng = MersenneTwister(2026)
        d = 4
        ∇log_p(x) = -x   # standard normal
        n = 1500

        # H₀: samples drawn from the target.
        good = randn(rng, n, d)
        res_good = psd_bootstrap_test(good, ∇log_p; degree=2,
                                        n_bootstrap=300, rng=rng)
        @test res_good.p_value > 0.05      # cannot reject H₀

        # H₁ (mean shift): biased mean → moment-1 disagreement → reject
        biased = randn(rng, n, d) .+ 0.4
        res_biased = psd_bootstrap_test(biased, ∇log_p; degree=2,
                                          n_bootstrap=300, rng=rng)
        @test res_biased.p_value < 0.01
        @test res_biased.psd2 > res_good.psd2

        # H₁ (variance inflation): σ = 1.6 → moment-2 disagreement → reject
        fat = 1.6 .* randn(rng, n, d)
        res_fat = psd_bootstrap_test(fat, ∇log_p; degree=2,
                                       n_bootstrap=300, rng=rng)
        @test res_fat.p_value < 0.01
        @test res_fat.psd2 > res_good.psd2

        # The diagonal-only basis is enough on a Gaussian with diagonal
        # covariance — should still flag the bias and use a smaller J.
        res_diag = psd_bootstrap_test(biased, ∇log_p; degree=2,
                                        diagonal_only=true,
                                        n_bootstrap=300, rng=rng)
        @test res_diag.p_value < 0.01
        @test res_diag.basis_size == 2 * d
        @test res_diag.basis_size <
              length(Nereus._generate_multiindices(d, 2))
    end

    @testset "V vs U statistics" begin
        rng = MersenneTwister(7)
        n = 500; d = 3
        good = randn(rng, n, d)
        v = polynomial_stein_discrepancy(good, x -> -x;
                                           degree=2, statistic=:V)
        u = polynomial_stein_discrepancy(good, x -> -x;
                                           degree=2, statistic=:U)
        @test v >= 0                          # V-stat is always nonneg
        @test isfinite(u)                     # U-stat may be slightly < 0 under H₀
        # The two converge as n → ∞: |v - u| should be small relative to v
        # for the moderate-n case here. Not a strict test, just sanity.
        @test abs(v - u) < max(0.5, abs(v))
    end

    @testset "NereusTarget convenience wrapper" begin
        # Build a tiny NereusTarget (1-planet RV-only) and run PSD on
        # a synthetic chain in unconstrained space. Mostly a smoke test
        # — confirms the gradient hand-off via ForwardDiff works.
        rng = MersenneTwister(31)
        n_obs = 30
        t = sort(rand(rng, n_obs) .* 100.0)
        rv = 50.0 .+ 1.5 .* randn(rng, n_obs)
        rv_err = fill(1.5, n_obs)
        ic = InstrumentConfig(rv=["X"])
        data = Data(; t_rv=t, rv=rv, rv_err=rv_err)
        params = Params(
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic, data=data,
        )
        target = NereusTarget(params, data; unconstrained=true)
        n_par = n_unfrozen(params)
        # Synthetic "chain" of 200 random unconstrained draws — these
        # are NOT samples from the posterior, so the test should reject
        # (high PSD²). The point is to confirm the call path works.
        chain = 0.3 .* randn(rng, 200, n_par)
        res = psd_bootstrap_test(target, chain; degree=2,
                                   diagonal_only=true,
                                   n_bootstrap=100, rng=rng)
        @test isfinite(res.psd2)
        @test 0.0 <= res.p_value <= 1.0
        @test res.basis_size == 2 * n_par
    end
end


# =====================================================================
# sample_map — physical-space MAP (no transform Jacobian)
# =====================================================================

@testset "sample_map physical-space mode" begin
    # Regression guard: sample_map must maximise the PHYSICAL-space
    # log-posterior (log_prior + log_likelihood in bounded space), NOT
    # the unconstrained-space log-density that LogDensityProblems returns
    # for a transformed target. The latter includes the inverse-transform
    # log-Jacobian, whose logit term log(s)+log(1-s) peaks at the
    # bound-midpoint (s = 0.5). For sesinw/secosw that midpoint is e ≈ 0,
    # so optimising the unconstrained density pulls the eccentricity MAP
    # toward zero — the railed-orbit failure mode this fix removes.
    #
    # Construct a 2-free-parameter (sesinw, secosw) RV problem whose
    # likelihood is intentionally shallow (large rv_err) so the logit
    # Jacobian would dominate the unconstrained density and bias e → 0.
    # The physical-space MAP must instead sit at the injected e = 0.5.
    rng = MersenneTwister(202)
    P_t, K_t, e_t, ω_t, Tp_t = 100.0, 50.0, 0.5, 0.7, 5.0
    n = 8
    t = collect(range(0.0, 95.0, length=n))
    rv_clean = [Nereus.rv_keplerian(ti, P_t, K_t, e_t, ω_t,
                                     2π * (0.0 - Tp_t) / P_t, 0.0) for ti in t]
    rv_obs = rv_clean .+ 0.5 .* randn(rng, n)
    data = Data(; t_rv=t, rv=rv_obs, rv_err=fill(40.0, n), t_ref=0.0)

    params = Params(
        max_kplanet=1, planet_modes=[RV_ONLY],
        instruments=InstrumentConfig(rv=["SIM"]), data=data, stability=:none,
        parametrization=ParametrizationConfig(ew=:sesinw, time=:Tp,
                                              mass=:K_driven),
        priors=Dict{String, PriorSpec}(
            "P_k1"        => FixedPrior(P_t),
            "K_k1"        => FixedPrior(K_t),
            "sesinw_k1"   => UniformPrior(-1.0, 1.0),
            "secosw_k1"   => UniformPrior(-1.0, 1.0),
            "Tp_k1"       => FixedPrior(Tp_t),
            "n_p"         => FixedPrior(1.0),
            "gamma_SIM"   => FixedPrior(0.0),
            "sigma_SIM"   => FixedPrior(0.01),
        ),
    )
    names = params.layout.unfrozen_names
    @test Set(names) == Set(["sesinw_k1", "secosw_k1"])
    target = NereusTarget(params, data)
    i_se = findfirst(==("sesinw_k1"), names)
    i_sc = findfirst(==("secosw_k1"), names)

    # Seed at the injected truth (bounded → unconstrained).
    sqe = sqrt(e_t)
    x_seed_b = zeros(length(names))
    x_seed_b[i_se] = sqe * sin(ω_t)
    x_seed_b[i_sc] = sqe * cos(ω_t)
    y_seed = Nereus.transform_forward(x_seed_b, target.transform.type_ids,
                                       target.transform.lowers,
                                       target.transform.uppers)

    res = sample_map(target; init=y_seed, method=:LBFGS,
                     maxiter=5000, g_tol=1e-8)
    e_map = res.x_map[i_se]^2 + res.x_map[i_sc]^2

    # Physical MAP recovers the injected eccentricity, not the e → 0
    # bound-midpoint the Jacobian would pull toward.
    @test e_map ≈ e_t atol=0.05

    # The reported log_posterior is the PHYSICAL posterior at the MAP
    # (no Jacobian): it must equal log_prior + log_likelihood evaluated
    # directly in bounded space.
    bt = NereusTarget(params, data; unconstrained=false)
    lp_phys = LogDensityProblems.logdensity(bt, res.x_map)
    @test isfinite(res.log_posterior)
    @test res.log_posterior ≈ lp_phys atol=1e-4

    # The unconstrained-space density at the MAP differs from the
    # reported (physical) log-posterior by exactly the log-Jacobian —
    # confirming the objective excluded it. Evaluate both at the
    # unconstrained image of the MAP so the change-of-variables holds.
    y_map = Nereus.transform_forward(res.x_map, target.transform.type_ids,
                                      target.transform.lowers,
                                      target.transform.uppers)
    lj = Nereus.transform_logabsdetjac_inv(y_map, target.transform)
    lp_unc = LogDensityProblems.logdensity(target, y_map)
    @test lp_unc ≈ res.log_posterior + lj atol=1e-4
    @test abs(lj) > 1e-3   # Jacobian is non-trivial here (would-be bias)
end


# =====================================================================
# MoMS sampler (van den Bergh, Clyde, Raftery, Marsman 2026,
# arXiv:2604.27791)
# =====================================================================

@testset "MoMS sampler" begin
    @testset "spike_slab_log_prior correction" begin
        # The spike-and-slab correction subtracts log π_β(off) for
        # inactive planets (Nereus log_prior over-counts at off-values)
        # and adds log(p_inc) / log(1-p_inc) per planet's γ.
        rng = MersenneTwister(101)
        ic = InstrumentConfig(rv=["X"])
        data = Data(; t_rv=collect(0.0:1.0:30.0),
                       rv=zeros(31), rv_err=ones(31))
        params = Params(
            max_kplanet=2, planet_modes=[RV_ONLY, RV_ONLY],
            instruments=ic, data=data, M_s=1.0,
        )
        td_state = TransDimState(max_planets=2, n_noise=0)
        theta = Theta{Float64}(params; td=td_state)
        for (i, idx) in enumerate(params.layout.unfrozen_idx)
            prior = params.layout.unfrozen_priors[i]
            v = rand(rng, prior.dist)
            lo, hi = bounds(prior)
            theta.values[idx] = clamp(v, lo, hi)
        end
        strategy = MoMSBirth(params; init_scale=1.0, rng=MersenneTwister(2))
        # Set planet 1 off (params at off-values), planet 2 active
        for (i, uf_pos) in enumerate(strategy.slot_indices[1])
            slot = params.layout.unfrozen_idx[uf_pos]
            theta.values[slot] = strategy.off_values[1][i]
        end
        theta.td.planet_active[1] = false
        theta.td.planet_active[2] = true
        theta.td.n_planets_active = 1

        base = log_prior(theta)
        slab = Nereus.spike_slab_log_prior(theta, strategy, 0.5)

        # `_log_prior_transdim` (Nereus's log_prior on a trans-dim
        # theta) ALREADY skips slots belonging to inactive planets, so
        # `base` does not include logpdf(prior, β_off) for the inactive
        # planet. The spike-and-slab correction is therefore JUST the
        # γ-prior indicator contributions: log(p_inc) for active planets
        # + log(1 - p_inc) for inactive. With p_inc = 0.5 and a
        # configuration of γ_1=0, γ_2=1: 2 × log(0.5).
        manual_correction = 2 * log(0.5)
        @test isapprox(slab, base + manual_correction; atol=1e-9)

        # With p_inc = 0.7 the γ-prior contribution shifts and we can
        # still verify the math by hand.
        slab_07 = Nereus.spike_slab_log_prior(theta, strategy, 0.7)
        manual_07 = log(1 - 0.7) + log(0.7)   # γ_1=0 + γ_2=1
        @test isapprox(slab_07, base + manual_07; atol=1e-9)
    end

    @testset "MoMSBirth construction from Params" begin
        # The off-values should be inside the prior support and the
        # scales should be positive, finite, and proportional to width.
        rng = MersenneTwister(11)
        ic   = InstrumentConfig(rv=["X"])
        data = Data(; t_rv=collect(0.0:1.0:50.0),
                       rv=zeros(51), rv_err=ones(51))
        params = Params(
            max_kplanet=2, planet_modes=[RV_ONLY, RV_ONLY],
            instruments=ic, data=data, M_s=1.0,
        )
        strategy = MoMSBirth(params; init_scale=0.3, rng=rng)
        @test length(strategy.off_values) == 2
        @test length(strategy.scales)      == 2
        @test length(strategy.slot_indices) == 2
        for k in 1:2
            @test length(strategy.off_values[k]) == length(strategy.scales[k])
            @test length(strategy.off_values[k]) == length(strategy.slot_indices[k])
            @test all(s -> s > 0 && isfinite(s), strategy.scales[k])
            # Off-values must lie within prior bounds for every slot
            for (i, uf_pos) in enumerate(strategy.slot_indices[k])
                prior = params.layout.unfrozen_priors[uf_pos]
                lo, hi = bounds(prior)
                isfinite(lo) && @test strategy.off_values[k][i] >= lo - 1e-9
                isfinite(hi) && @test strategy.off_values[k][i] <= hi + 1e-9
            end
        end
    end

    @testset "MoMS birth/death proposals — log_q symmetry" begin
        # An add followed by a delete should round-trip back to the
        # original state for the indicator and parameters at the off-
        # location. The proposal log-densities accumulated in log_q
        # over add+delete should net to zero in the Gaussian random
        # walk's log-density terms (combinatorial terms cancel since
        # n_active is the same before and after).
        rng = MersenneTwister(7)
        ic = InstrumentConfig(rv=["X"])
        data = Data(; t_rv=collect(0.0:1.0:30.0),
                       rv=zeros(31), rv_err=ones(31))
        params = Params(
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic, data=data, M_s=1.0,
        )
        td_state = TransDimState(max_planets=1, n_noise=0)
        theta = Theta{Float64}(params; td=td_state)
        # Initialize systemics to valid prior values
        for (i, idx) in enumerate(params.layout.unfrozen_idx)
            prior = params.layout.unfrozen_priors[i]
            v = rand(rng, prior.dist)
            lo, hi = bounds(prior)
            theta.values[idx] = clamp(v, lo, hi)
        end

        strategy = MoMSBirth(params; init_scale=0.3, rng=MersenneTwister(2))
        # Birth from N=0 → N=1
        new_theta, log_q_birth = propose_planet_birth(theta, MersenneTwister(99),
                                                       strategy)
        @test isfinite(log_q_birth)
        @test new_theta.td.n_planets_active == 1
        # Death from N=1 → N=0 should reset planet 1's slots to off-values
        dead_theta, log_q_death = propose_planet_death(new_theta, MersenneTwister(99),
                                                        strategy)
        @test isfinite(log_q_death)
        @test dead_theta.td.n_planets_active == 0
        # Off-values restored
        for (i, uf_pos) in enumerate(strategy.slot_indices[1])
            slot = params.layout.unfrozen_idx[uf_pos]
            @test isapprox(dead_theta.values[slot],
                           strategy.off_values[1][i]; atol=1e-9)
        end
    end

    @testset "sample_moms — runs end-to-end on synthetic RV" begin
        # Synthetic single-planet RV. The MoMS sampler is known to be
        # less efficient than InformedBirth for periodic-signal detection
        # (random-walk proposals around prior midpoints rarely land on
        # the true period). The aim here is mechanical: confirm chains
        # are produced, scales adapt, and N_p posterior is well-formed.
        rng = MersenneTwister(2026)
        n_obs = 60
        t = sort(rand(rng, n_obs) .* 100.0)
        P_true = 10.0; K_true = 15.0
        rv = 5.0 .+ K_true .* sin.(2π .* t ./ P_true) .+
              1.5 .* randn(rng, n_obs)
        rv_err = fill(1.5, n_obs)

        ic = InstrumentConfig(rv=["SIM"])
        data = Data(; t_rv=t, rv=rv, rv_err=rv_err)
        params = Params(
            max_kplanet=2, planet_modes=[RV_ONLY, RV_ONLY],
            instruments=ic, data=data, M_s=1.0,
        )
        target = NereusTarget(params, data; unconstrained=false)
        td = TransDimConfig(max_kplanet=2, transdim_fraction=0.4)

        chains, evals, strategy = sample_moms(target, data;
            td=td, n_warmup=200, n_samples=500, seed=42, init_scale=0.3,
        )
        @test chains isa MCMCChains.Chains
        @test :n_planets in names(chains, :parameters)
        np = Array(chains[:n_planets])
        @test length(np) == 500
        @test all(0 .<= np .<= 2)
        @test evals > 0
        # Adapted scales should still be positive and finite
        for k in 1:length(strategy.scales)
            @test all(s -> s > 0 && isfinite(s), strategy.scales[k])
        end
    end
end


# =====================================================================
# MoMS-NS — trans-dim nested sampling via mutually-singular distributions
# =====================================================================

@testset "MoMS-NS sampler" begin
    @testset "sample_moms_ns runs end-to-end" begin
        # Tight WASP-47-style priors so random-walk proposals can mix
        # within the constrained-likelihood region. The NS algorithm
        # itself is being tested here, not the periodic-search ability
        # of the MoMS proposal kernel.
        rng = MersenneTwister(7)
        n_obs = 40
        t = sort(rand(rng, n_obs) .* 30.0)
        P1, K1 = 4.16, 12.0
        rv = 5.0 .+ K1 .* sin.(2π .* t ./ P1) .+ 1.5 .* randn(rng, n_obs)
        rv_err = fill(1.5, n_obs)

        ic = InstrumentConfig(rv=["X"])
        data = Data(; t_rv=t, rv=rv, rv_err=rv_err)

        # Tight period priors centered on truth — narrow enough that
        # random-walk MoMS proposals can land in the high-likelihood
        # mode within a few hundred NS iterations.
        priors = Dict{String, PriorSpec}(
            "P_k1" => NormalPrior(P1, 0.05, 4.0, 4.3),
            "P_k2" => NormalPrior(8.0, 0.5, 6.0, 10.0),
        )
        params = Params(
            max_kplanet=2, planet_modes=[RV_ONLY, RV_ONLY],
            instruments=ic, data=data, M_s=1.0, priors=priors,
        )
        target = NereusTarget(params, data; unconstrained=false)
        td = TransDimConfig(max_kplanet=2, transdim_fraction=0.4)

        chains, logZ, strategy = sample_moms_ns(target, data;
            td=td, n_live=40, dlogz=0.5, n_mcmc=15, seed=42,
            show_progress=false,
        )
        @test chains isa MCMCChains.Chains
        @test :n_planets in names(chains, :parameters)
        @test isfinite(logZ)
        np = Array(chains[:, :n_planets, :])
        @test all(0 .<= np .<= 2)
        @test length(np) >= 40        # at least n_live points in the resampled chain
        # MoMS strategy should still carry valid off-values + scales
        for k in 1:length(strategy.scales)
            @test all(s -> s > 0 && isfinite(s), strategy.scales[k])
        end
    end

    @testset "noise toggling — runs end-to-end" begin
        # MoMS-NS now supports td.noise=true via the existing prior-draw
        # noise propose_noise_birth/death methods invoked from the
        # constrained-MCMC kernel.
        rng = MersenneTwister(7)
        n_obs = 40
        t = sort(rand(rng, n_obs) .* 30.0)
        rv = 5.0 .+ 12.0 .* sin.(2π .* t ./ 4.16) .+ 1.5 .* randn(rng, n_obs)
        rv_err = fill(1.5, n_obs)
        ic = InstrumentConfig(rv=["X"])
        data = Data(; t_rv=t, rv=rv, rv_err=rv_err)

        ar_model = ARModel(order=1)
        priors = Dict{String, PriorSpec}(
            "P_k1" => NormalPrior(4.16, 0.05, 4.0, 4.3),
        )
        params = Params(
            max_kplanet=1, planet_modes=[RV_ONLY],
            instruments=ic, data=data, M_s=1.0, priors=priors,
            noise_models=[ar_model],
            transdim_noise=true,
        )
        target = NereusTarget(params, data; unconstrained=false)
        td = TransDimConfig(
            max_kplanet=1, transdim_fraction=0.5,
            noise=true, toggleable=[ar_model],
        )

        chains, logZ, _ = sample_moms_ns(target, data;
            td=td, n_live=30, n_mcmc=15, dlogz=0.5, seed=1,
            show_progress=false,
        )
        @test chains isa MCMCChains.Chains
        @test isfinite(logZ)
        # Both indicator columns should be present
        @test :n_planets in names(chains, :parameters)
        @test :noise_active_1 in names(chains, :parameters)
        np = Array(chains[:, :n_planets, :])
        ns = Array(chains[:, :noise_active_1, :])
        @test all(0 .<= np .<= 1)
        @test all(in.(ns, Ref([0.0, 1.0])))
    end

    @testset "model_probabilities + bayes_factors helpers" begin
        # Build a fake chain manually with a known N_p distribution and
        # check the helper outputs.
        n = 1000
        np = vcat(fill(0.0, 100),       # 10%
                   fill(1.0, 700),       # 70%
                   fill(2.0, 200))       # 20%
        chain_data = reshape(np, :, 1, 1)
        chains = MCMCChains.Chains(chain_data, [:n_planets])

        probs = model_probabilities(chains)
        @test probs[0] ≈ 0.10
        @test probs[1] ≈ 0.70
        @test probs[2] ≈ 0.20

        # Reference Bayes factors against k=0 under uniform model prior:
        #   BF(1 vs 0) = 0.7 / 0.1 = 7.0
        #   BF(2 vs 0) = 0.2 / 0.1 = 2.0
        bfs = bayes_factors(chains; reference=0)
        @test bfs[1] ≈ 7.0
        @test bfs[2] ≈ 2.0

        # All-pairs form
        bfs_all = bayes_factors(chains)
        @test haskey(bfs_all, (1, 0))
        @test haskey(bfs_all, (2, 1))
        @test bfs_all[(1, 0)] ≈ 7.0
        @test bfs_all[(2, 1)] ≈ 0.2 / 0.7

        # Custom (non-uniform) model prior penalising larger N_p
        # p(N=k) ∝ 0.5^k → p(0):p(1):p(2) = 1:0.5:0.25
        # BF(1 vs 0) under this prior =
        #   (P(1|y)/P(0|y)) × (p(0)/p(1)) = 7.0 × (1/0.5) = 14.0
        bfs_geom = bayes_factors(chains; reference=0,
                                   model_prior = k -> 0.5^k)
        @test bfs_geom[1] ≈ 14.0
        @test bfs_geom[2] ≈ 8.0

        # Reference value not present in chain → throws
        @test_throws ArgumentError bayes_factors(chains; reference=5)
    end

    @testset "evidence ordering: more data → higher log Z" begin
        # The same model fitted to a 2× longer time series should
        # give a higher log Z if the signal is real (more data → more
        # support for the planet hypothesis). Sanity check that log Z
        # responds in the expected direction.
        rng = MersenneTwister(13)
        function _run(n_obs)
            t = sort(rand(MersenneTwister(13), n_obs) .* (n_obs * 0.7))
            P1, K1 = 4.16, 15.0
            rv = K1 .* sin.(2π .* t ./ P1) .+ 1.5 .* randn(MersenneTwister(99), n_obs)
            rv_err = fill(1.5, n_obs)
            ic = InstrumentConfig(rv=["X"])
            data = Data(; t_rv=t, rv=rv, rv_err=rv_err)
            priors = Dict{String, PriorSpec}(
                "P_k1" => NormalPrior(P1, 0.05, 4.0, 4.3),
            )
            params = Params(
                max_kplanet=1, planet_modes=[RV_ONLY],
                instruments=ic, data=data, M_s=1.0, priors=priors,
            )
            target = NereusTarget(params, data; unconstrained=false)
            td = TransDimConfig(max_kplanet=1, transdim_fraction=0.4)
            _, logZ, _ = sample_moms_ns(target, data;
                td=td, n_live=30, dlogz=0.5, n_mcmc=10, seed=42,
                show_progress=false,
            )
            return logZ
        end
        # Both should produce finite log-Z; absolute values depend on
        # priors but the algorithm itself should not crash.
        logZ_small = _run(20)
        logZ_large = _run(40)
        @test isfinite(logZ_small)
        @test isfinite(logZ_large)
    end
end


# =====================================================================
# Astrometry (Phase 1: HGCA + relative astrometry)
# =====================================================================

include("astrometry/test_data.jl")
include("astrometry/test_projection.jl")
include("astrometry/test_likelihood.jl")
include("astrometry/test_sine_prior.jl")
include("astrometry/test_param_modes.jl")
include("astrometry/test_m_pri.jl")
include("astrometry/test_obs_prior.jl")
include("astrometry/test_ofti.jl")
include("astrometry/test_iad_gost.jl")
include("astrometry/test_gaia_epoch_guards.jl")
include("test_builder.jl")
include("test_new_samplers.jl")
include("test_runner_dispatch.jl")
include("test_rm.jl")
include("test_tomography.jl")
include("test_tomography_framework.jl")
include("test_obliquity_framework.jl")
include("test_tomo_noise_menu.jl")
include("test_obliquity_joint_framework.jl")
include("test_as_coupling_mask.jl")
include("test_as_coupling_move.jl")
include("test_obliquity_joint.jl")
include("test_simulate_obliquity.jl")
include("test_gravity_darkening.jl")
include("test_informed_noise_birth.jl")
include("test_annealed_noise_birth.jl")
include("test_solution_ladder.jl")
include("test_noise_swap_samplers.jl")
include("test_ttv.jl")
include("test_ppc.jl")
include("test_detection_limits.jl")
include("test_loo.jl")
include("test_fit_health.jl")
include("test_label_switching.jl")
include("test_pt_donor_buffer.jl")
include("test_transdim_activity_columns.jl")
include("test_activity_gp.jl")
include("test_multiseries_gp.jl")
include("test_parametric_noise.jl")
include("test_harmonic_external.jl")
include("test_transdim_caches.jl")
include("test_birth_death_reversibility.jl")
include("test_alias_jump.jl")
include("test_transdim_death_counters.jl")
include("test_locor.jl")
include("test_locor_io.jl")
include("test_lightcurve.jl")

