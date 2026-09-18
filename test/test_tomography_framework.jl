# Tomography as a framework-native likelihood term.
#
# `tomogram_bayes` runs its own sampler over a bespoke theta vector. That works
# standalone but the tomographic term then cannot see anything else Nereus
# knows: no noise menu, no trans-dim, no shared lambda with the RM velocities,
# no joint fit with RV or transit.
#
# These tests pin the thing that makes the framework version worth having: the
# geometry is SHARED. If lambda and v sin i were private to the tomogram, a
# "joint" fit would be two independent fits reported together, and no test of
# the likelihood value alone would notice.
using Test
using Nereus
using Nereus: TomoNight, tomogram_log_likelihood, Theta, set_param!,
              shadow_map, planet_lambda, system_vsini, n_unfrozen
using Random, Statistics

@testset "tomography in the framework" begin
    P, Tc, aRs, inc_true, vsini_kms = 2.827969, 2459000.0, 6.81, 1.459, 25.9
    λ_true = deg2rad(-40.0)
    grid = collect(range(-70.0, 70.0; length = 60))
    t = collect(range(Tc - 0.05, Tc + 0.05; length = 24))

    function make_data(; with_shadow = true, seed = 5)
        rng = MersenneTwister(seed)
        M = shadow_map(t, Tc, P, aRs, inc_true, λ_true, vsini_kms, grid, 6.0;
                       rr = 0.116, u1 = 0.23, u2 = 0.15)
        R = (with_shadow ? 1.0 : 0.0) .* M .+ 0.01 .* randn(rng, size(M))
        Data(t_rv = [Tc - 0.4, Tc, Tc + 0.4], rv = [0.0, 5.0, -3.0],
             rv_err = fill(3.0, 3), rv_inst = [1, 1, 1],
             t_phot = t, flux = fill(1.0, length(t)),
             flux_err = fill(1e-3, length(t)), phot_inst = ones(Int, length(t)),
             tomo = [TomoNight("HARPS", t, R, grid, Tc)])
    end

    function build(data)
        Params(max_kplanet = 1, planet_modes = [RVPM_RM],
               instruments = InstrumentConfig(rv = ["I1"], pm = ["TESS"]),
               data = data, stability = :none, M_s = 1.0, R_s = 1.0)
    end

    @testset "slots exist and are per-night" begin
        d = make_data(); p = build(d)
        @test "tomo_alpha_HARPS" in p.layout.names
        @test "tomo_sigma_line_HARPS" in p.layout.names
        @test length(p.layout.systemic.tomo_alpha) == 1
        # lambda and v sin i are NOT tomography-private — they are the RM slots
        @test "lambda_k1" in p.layout.names
        @test "v_sin_i_star" in p.layout.names
    end

    function seat(p, d; λ = λ_true, α = 1.0)
        th = Theta{Float64}(p)
        for (i, slot) in enumerate(p.layout.unfrozen_idx)
            lo, hi = bounds(p.layout.unfrozen_priors[i])
            th.values[slot] = clamp(rand(MersenneTwister(slot),
                                    p.layout.unfrozen_priors[i].dist), lo, hi)
        end
        set_param!(th, "lambda_k1", λ)
        set_param!(th, "v_sin_i_star", vsini_kms * 1000)
        set_param!(th, "P_k1", P)
        set_param!(th, "tomo_alpha_HARPS", α)
        set_param!(th, "tomo_sigma_line_HARPS", 6.0)
        haskey(p.layout.name_to_idx, "rho_s") && set_param!(th, "rho_s",
            a_Rs_to_rho_s(aRs, P))
        return th
    end

    @testset "empty tomo costs exactly zero" begin
        d = Data(t_rv = [1.0, 2.0], rv = [0.0, 1.0], rv_err = [1.0, 1.0],
                 rv_inst = [1, 1])
        p = Params(max_kplanet = 0, planet_modes = PlanetDataSources[],
                   instruments = InstrumentConfig(rv = ["I1"]), data = d,
                   stability = :none)
        th = Theta{Float64}(p)
        @test tomogram_log_likelihood(th, d) == 0.0
    end

    # ---- the point: lambda is SHARED ------------------------------------
    @testset "likelihood responds to the SHARED lambda slot" begin
        d = make_data(); p = build(d)
        th = seat(p, d)
        ll_true = tomogram_log_likelihood(th, d)
        @test isfinite(ll_true)
        # move only lambda_k1 — the slot the RM velocities also read
        set_param!(th, "lambda_k1", λ_true + deg2rad(70))
        ll_wrong = tomogram_log_likelihood(th, d)
        @test isfinite(ll_wrong)
        @test ll_true > ll_wrong
    end

    @testset "likelihood responds to the SHARED v sin i slot" begin
        d = make_data(); p = build(d)
        th = seat(p, d)
        ll_true = tomogram_log_likelihood(th, d)
        set_param!(th, "v_sin_i_star", vsini_kms * 1000 * 2.5)
        @test tomogram_log_likelihood(th, d) < ll_true
    end

    @testset "amplitude is per-night and bounded below at zero" begin
        d = make_data(); p = build(d)
        i = p.layout.name_to_idx["tomo_alpha_HARPS"]
        u = findfirst(==(i), p.layout.unfrozen_idx)
        lo, _ = bounds(p.layout.unfrozen_priors[u])
        # a negative shadow is an emission feature; letting alpha go negative
        # lets the mirrored track win and be reported as a detection
        @test lo >= 0.0
    end

    @testset "no shadow ⇒ alpha = 0 is preferred" begin
        d = make_data(with_shadow = false); p = build(d)
        th0 = seat(p, d; α = 0.0)
        th1 = seat(p, d; α = 1.0)
        @test tomogram_log_likelihood(th0, d) > tomogram_log_likelihood(th1, d)
    end

    @testset "enters the target sum" begin
        d = make_data(); p = build(d)
        # bounded target: `unconstrained=true` is the default and would expect
        # a point in R^n, so constrained values there give -Inf and prove
        # nothing. (`isfinite(v) || v == -Inf` was a tautology.)
        tgt = NereusTarget(p, d; unconstrained = false)
        th = seat(p, d)
        x = [th.values[s] for s in p.layout.unfrozen_idx]
        @test isfinite(tgt(x))
        @test length(x) == n_unfrozen(p)
    end
end
