# Joint RM + tomography through the framework: one Params, shared geometry,
# INDEPENDENT noise selection per channel.
#
# What makes this a joint fit and not two fits reported together is that lambda,
# v sin i and the transit geometry are single slots read by both likelihood
# terms. What makes it useful is that the noise is NOT shared: the velocities
# get an :rv-channel model and the maps a :tomo-channel one, so a night whose
# velocities are pulsation-dominated can pick a different description from the
# map of the same night.
using Test
using Nereus
using Nereus: RMNight, TomoNight, obliquity_data, obliquity_params, Theta,
              set_param!, rv_log_likelihood, tomogram_log_likelihood,
              shadow_map, tc_to_tp, tp_to_mo, MaternGP, CeleriteSHO, NoiseModel,
              log_prior
using Random, Statistics

@testset "joint obliquity in the framework" begin
    P, Tc, VS = 2.827969, 2459000.0, 25.9
    AR, BIMP  = 6.81, 0.4
    INC       = acos(BIMP / AR)          # MUST match the b the model reads
    λ_true    = deg2rad(-40.0)
    grid = collect(range(-70.0, 70.0; length = 32))
    tt   = collect(range(Tc - 0.05, Tc + 0.05; length = 18))

    function setup(; map_noise = 0.003, tomo_model = nothing, seed = 7)
        rng = MersenneTwister(seed)
        M = shadow_map(tt, Tc, P, AR, INC, λ_true, VS, grid, 6.0;
                       rr = 0.116, u1 = 0.23, u2 = 0.15)
        tomo = [TomoNight("HARPSmap", tt, M .+ map_noise .* randn(rng, size(M)),
                          grid, Tc)]
        rm = [RMNight("CORALIE", collect(range(Tc-0.06, Tc+0.06; length = 20)),
                      25 .* randn(rng, 20), fill(25.0, 20), 6000.0, Tc)]
        d, names = obliquity_data(rm; tomo_nights = tomo)
        nms = Vector{NoiseModel}(default_noise_menu(d).noise_models)
        tomo_model === nothing || push!(nms, tomo_model)
        p = obliquity_params(d, names; P = P, Tc = Tc, b = (BIMP, 0.05),
                a_Rs = (AR, 0.2), rr = (0.116, 0.005),
                vsini = (VS*1000, 1000.0), M_s = 1.6, R_s = 1.7,
                noise_models = nms)
        return d, p
    end
    function seat(p, d)
        th = Theta{Float64}(p)
        for (i, s) in enumerate(p.layout.unfrozen_idx)
            lo, hi = bounds(p.layout.unfrozen_priors[i])
            th.values[s] = clamp((lo + hi) / 2, lo, hi)
        end
        set_param!(th, "n_p", 1.0)
        set_param!(th, "Mo_k1", mod(tp_to_mo(tc_to_tp(Tc,P,0.0,0.0), P, d.t_ref), 2π))
        for (k, v) in ("v_sin_i_star"=>VS*1000, "P_k1"=>P, "b_k1"=>BIMP,
                       "rr_k1"=>0.116, "K_k1"=>100.0,
                       "rho_s"=>Nereus._a_Rs_to_rho_s(AR, P),
                       "sesinw_k1"=>0.0, "secosw_k1"=>0.0,
                       "tomo_alpha_HARPSmap"=>1.0,
                       "tomo_sigma_line_HARPSmap"=>6.0,
                       "tomo_ell_v_HARPSmap"=>8.0,
                       "matern_sigma_tomo"=>0.003, "matern_rho_tomo"=>0.02)
            haskey(p.layout.name_to_idx, k) && set_param!(th, k, v)
        end
        return th
    end

    @testset "one Params carries both datasets" begin
        d, p = setup()
        @test length(d.rv) == 20 && length(d.tomo) == 1
        # exactly ONE lambda and ONE v sin i, read by both terms
        @test count(n -> occursin("lambda", n), p.layout.names) == 1
        @test count(==( "v_sin_i_star"), p.layout.names) == 1
    end

    @testset "noise is selected per channel, not shared" begin
        _, p = setup(tomo_model = MaternGP(channel = :tomo))
        rvp   = filter(n -> startswith(n, "gp_") && !occursin("_tomo", n), p.layout.names)
        tomop = filter(n -> occursin("_tomo", n) && occursin("matern", n), p.layout.names)
        @test !isempty(rvp)      # rv channel has its own kernel
        @test !isempty(tomop)    # tomo channel has a different one
        @test isempty(intersect(rvp, tomop))
    end

    # ---- recovery: the tomographic term must peak at the injected lambda ----
    @testset "tomographic term recovers the injected lambda" begin
        d, p = setup()
        th = seat(p, d)
        λs = collect(range(-90.0, 40.0; step = 10.0))
        lls = map(λs) do λ
            set_param!(th, "lambda_k1", deg2rad(λ))
            tomogram_log_likelihood(th, d)
        end
        @test all(isfinite, lls)
        @test abs(λs[argmax(lls)] - rad2deg(λ_true)) <= 10.0
    end

    @testset "the two terms are independent functions of lambda" begin
        # If the tomographic term were secretly reading a private lambda, the
        # RM term would move and it would not.
        d, p = setup()
        th = seat(p, d)
        set_param!(th, "lambda_k1", deg2rad(-40.0))
        a1, b1 = rv_log_likelihood(th, d), tomogram_log_likelihood(th, d)
        set_param!(th, "lambda_k1", deg2rad(20.0))
        a2, b2 = rv_log_likelihood(th, d), tomogram_log_likelihood(th, d)
        @test a1 != a2
        @test b1 != b2
    end

    @testset "geometry must be self-consistent to recover anything" begin
        # b and inc are not independent: b = a_Rs*cos(inc). A map injected at an
        # inc inconsistent with the b the model reads produces a DIFFERENT
        # shadow track, and lambda recovery silently fails — the fit converges
        # on a flat, mis-centred likelihood. Guard the relation the test relies on.
        @test AR * cos(INC) ≈ BIMP atol = 1e-12
    end

    @testset "the target sums both terms" begin
        d, p = setup()
        th = seat(p, d)
        set_param!(th, "lambda_k1", λ_true)
        # `unconstrained=true` is the DEFAULT, so a vector of constrained values
        # is not a valid argument to it — that silently gives -Inf and looks
        # like a broken model. Use the bounded target for a value check.
        tgt = NereusTarget(p, d; unconstrained = false)
        x = [th.values[s] for s in p.layout.unfrozen_idx]
        v = tgt(x)
        @test isfinite(v)
        @test v ≈ log_prior(th) + rv_log_likelihood(th, d) +
                  tomogram_log_likelihood(th, d) rtol = 1e-8
    end
end
