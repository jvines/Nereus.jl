# The tomographic temporal kernel comes from the NOISE MENU.
#
# The bespoke path hardwired a Matern-3/2 Kronecker GP. Whether the pulsations
# in a residual map are a damped oscillator, a rotation kernel or short-memory
# Matern is not something the tomography module should decide — it is exactly
# the question the menu answers for RV noise, with calibrated occupancy.
#
# The velocity axis stays a fitted smoothness kernel because it is
# INSTRUMENTAL: the CCF is oversampled and smoothed by the instrumental
# profile, so neighbouring bins are correlated whatever the star does. That is
# not a competing hypothesis. The TIME axis is.
using Test
using Nereus
using Nereus: TomoNight, Theta, set_param!, tomogram_log_likelihood, shadow_map,
              tc_to_tp, tp_to_mo, MaternGP, CeleriteSHO, NoiseModel,
              kron_gp_loglike, celerite_kernel_dense, _kern, _tomo_temporal_kernel
using Random, Statistics, LinearAlgebra

@testset "tomographic noise menu" begin
    P, Tc, VSINI = 2.827969, 2459000.0, 25.9
    grid = collect(range(-70.0, 70.0; length = 32))
    tt   = collect(range(Tc - 0.05, Tc + 0.05; length = 18))

    "Residual map with a shadow plus either white or time-correlated noise."
    function make(; correlated::Bool, seed = 3)
        rng = MersenneTwister(seed)
        M = shadow_map(tt, Tc, P, 6.81, 1.459, deg2rad(-40.0), VSINI, grid, 6.0;
                       rr = 0.116, u1 = 0.23, u2 = 0.15)
        N = 0.02 .* randn(rng, size(M))
        if correlated
            # smooth along TIME only: a slow drift shared across velocity bins,
            # which is what a pulsation looks like in a tomogram
            drift = cumsum(0.05 .* randn(rng, length(tt)))
            N .+= drift * ones(length(grid))'
        end
        Data(t_rv = [Tc-0.4, Tc, Tc+0.4], rv = [0.0, 5.0, -3.0],
             rv_err = fill(3.0, 3), rv_inst = [1,1,1],
             tomo = [TomoNight("HARPS", tt, M .+ N, grid, Tc)])
    end
    build(d, nms) = Params(max_kplanet = 1, planet_modes = [RVPM_RM],
        instruments = InstrumentConfig(rv = ["I1"], pm = String[]),
        data = d, stability = :none, M_s = 1.0, R_s = 1.0,
        noise_models = Vector{NoiseModel}(nms))
    function seat(p, d; extra = Dict{String,Float64}())
        th = Theta{Float64}(p)
        for (i, s) in enumerate(p.layout.unfrozen_idx)
            lo, hi = bounds(p.layout.unfrozen_priors[i])
            th.values[s] = clamp((lo + hi) / 2, lo, hi)
        end
        set_param!(th, "n_p", 1.0)
        set_param!(th, "Mo_k1", mod(tp_to_mo(tc_to_tp(Tc,P,0.0,0.0), P, d.t_ref), 2π))
        for (k,v) in ("lambda_k1"=>deg2rad(-40.0), "v_sin_i_star"=>VSINI*1000,
                      "P_k1"=>P, "b_k1"=>0.4, "rr_k1"=>0.116,
                      "tomo_alpha_HARPS"=>1.0, "tomo_sigma_line_HARPS"=>6.0,
                      "tomo_ell_v_HARPS"=>8.0, "sesinw_k1"=>0.0, "secosw_k1"=>0.0)
            haskey(p.layout.name_to_idx, k) && set_param!(th, k, v)
        end
        for (k,v) in extra
            haskey(p.layout.name_to_idx, k) && set_param!(th, k, v)
        end
        return th
    end

    @testset "kernels appear as _tomo parameters" begin
        d = make(correlated = false)
        @test isempty(filter(n -> occursin("_tomo", n),
                             build(d, NoiseModel[]).layout.names))
        pm = build(d, [MaternGP(channel = :tomo)])
        @test "matern_sigma_tomo" in pm.layout.names
        @test "matern_rho_tomo"   in pm.layout.names
        ps = build(d, [CeleriteSHO(channel = :tomo)])
        @test "gp_log_S0_tomo" in ps.layout.names
        # a :tomo model must NOT collide with an :rv one of the same family
        both = build(d, [MaternGP(channel = :tomo), MaternGP(channel = :rv)])
        @test "matern_sigma_tomo" in both.layout.names
        @test "matern_sigma"      in both.layout.names
    end

    @testset "white is the null when no :tomo model is active" begin
        d = make(correlated = false)
        p = build(d, NoiseModel[]); th = seat(p, d)
        @test _tomo_temporal_kernel(th, d.tomo[1]) === nothing
        @test isfinite(tomogram_log_likelihood(th, d))
    end

    @testset "an active :tomo model supplies the kernel" begin
        d = make(correlated = false)
        p = build(d, [MaternGP(channel = :tomo)])
        th = seat(p, d; extra = Dict("matern_sigma_tomo" => 0.05,
                                     "matern_rho_tomo"   => 0.02))
        K = _tomo_temporal_kernel(th, d.tomo[1])
        @test K !== nothing
        @test size(K) == (length(tt), length(tt))
        @test issymmetric(Symmetric(K))
        @test minimum(eigvals(Symmetric(K))) > -1e-8      # PSD
    end

    # ---- the discriminating pair ---------------------------------------
    @testset "white wins on white, GP wins on correlated" begin
        gp_par = Dict("matern_sigma_tomo" => 0.05, "matern_rho_tomo" => 0.02)

        dw = make(correlated = false)
        ll_w_white = tomogram_log_likelihood(seat(build(dw, NoiseModel[]), dw), dw)
        pmw = build(dw, [MaternGP(channel = :tomo)])
        ll_w_gp = tomogram_log_likelihood(seat(pmw, dw; extra = gp_par), dw)
        @test ll_w_white > ll_w_gp        # no structure ⇒ the GP only costs

        dc = make(correlated = true)
        ll_c_white = tomogram_log_likelihood(seat(build(dc, NoiseModel[]), dc), dc)
        pmc = build(dc, [MaternGP(channel = :tomo)])
        ll_c_gp = tomogram_log_likelihood(seat(pmc, dc; extra = gp_par), dc)
        @test ll_c_gp > ll_c_white        # real time-correlation ⇒ the GP earns it
    end

    @testset "generalised Kronecker reproduces the fixed-kernel form" begin
        r = randn(MersenneTwister(9), 10, 14)
        h = collect(range(-2.0, 2.0; length = 10))
        g = collect(range(-40.0, 40.0; length = 14))
        @test kron_gp_loglike(r, h, g, 0.5, 15.0, 0.3, 0.1) ≈
              kron_gp_loglike(r, _kern(h, 0.5), (0.3^2) .* _kern(g, 15.0), 0.1) atol = 1e-10
    end

    @testset "celerite dense kernel is a valid covariance" begin
        t = collect(range(0.0, 0.1; length = 12))
        ar, cr, ac, bc, cc, dc = Nereus.sho_coefficients(1e-3, 2.0, 200.0)
        K = celerite_kernel_dense(t, ar, cr, ac, bc, cc, dc)
        @test issymmetric(Symmetric(K))
        @test minimum(eigvals(Symmetric(K))) > -1e-8
        @test K[1,1] ≈ maximum(K) rtol = 1e-8     # peak at zero lag
    end
end
