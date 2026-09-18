# Framework-native obliquity fitting: RM velocities through the standard
# Params/Theta path, so the noise menu and trans-dim apply.
#
# `_decode_rm_state` has always been called from `rv_log_likelihood`, so the RM
# anomaly already flowed through the standard noise machinery — what was missing
# was an entry point that assembled a Params instead of a bespoke theta vector.
# These tests pin that the assembly is CORRECT, because two of its steps fail
# silently rather than loudly when they are wrong.
using Test
using Nereus
using Nereus: RMNight, obliquity_data, obliquity_params, Theta, set_param!,
              rv_log_likelihood, tc_to_tp, tp_to_mo, rho_s_to_a_Rs,
              _a_Rs_to_rho_s, TomoNight
using Random, Statistics

@testset "obliquity in the framework" begin
    P, Tc, VSINI = 2.827969, 2459000.0, 25_900.0
    mk(tag, tc, n, σ, rng) = RMNight(tag,
        collect(range(tc - 0.06, tc + 0.06; length = n)),
        σ .* randn(rng, n), fill(σ, n), 6000.0, tc)
    function setup(; menu = true, tomo = TomoNight[])
        rng = MersenneTwister(4)
        nights = [mk("CORALIE", Tc, 22, 25.0, rng),
                  mk("HARPS", Tc + 3P, 18, 12.0, rng)]
        d, names = obliquity_data(nights; tomo_nights = tomo)
        nm = menu ? default_noise_menu(d).noise_models : NoiseModel[]
        p = obliquity_params(d, names; P = P, Tc = Tc, b = (0.4, 0.05),
                a_Rs = (6.81, 0.2), rr = (0.116, 0.005),
                vsini = (VSINI, 1000.0), M_s = 1.6, R_s = 1.7,
                noise_models = Vector{NoiseModel}(nm), transdim_noise = menu)
        return d, names, p
    end
    function seat(p, d)
        th = Theta{Float64}(p)
        for (i, s) in enumerate(p.layout.unfrozen_idx)
            lo, hi = bounds(p.layout.unfrozen_priors[i])
            th.values[s] = clamp((lo + hi) / 2, lo, hi)
        end
        set_param!(th, "n_p", 1.0)
        set_param!(th, "Mo_k1",
            mod(tp_to_mo(tc_to_tp(Tc, P, 0.0, 0.0), P, d.t_ref), 2π))
        for (k, v) in ("v_sin_i_star" => VSINI, "P_k1" => P, "b_k1" => 0.4,
                       "rr_k1" => 0.116, "K_k1" => 100.0, "rho_s" => 0.52945,
                       "sesinw_k1" => 0.0, "secosw_k1" => 0.0)
            haskey(p.layout.name_to_idx, k) && set_param!(th, k, v)
        end
        return th
    end

    @testset "one night = one instrument" begin
        d, names, p = setup()
        @test names == ["CORALIE", "HARPS"]
        # private offset and jitter per night, which is why they are instruments
        @test "gamma_CORALIE" in p.layout.names && "gamma_HARPS" in p.layout.names
        @test "sigma_CORALIE" in p.layout.names && "sigma_HARPS" in p.layout.names
        rng = MersenneTwister(1)
        dup = [mk("A", Tc, 5, 1.0, rng), mk("A", Tc + P, 5, 1.0, rng)]
        @test_throws ArgumentError obliquity_data(dup)
        @test_throws ArgumentError obliquity_data(RMNight[])
    end

    @testset "the noise menu is attached" begin
        _, _, p = setup(menu = true)
        @test any(n -> startswith(n, "gp_"), p.layout.names)
        _, _, p0 = setup(menu = false)
        @test !any(n -> startswith(n, "gp_"), p0.layout.names)
    end

    # ---- the two silent failures --------------------------------------
    @testset "Tc reaches the right orbital phase" begin
        # Transit is at true anomaly pi/2 - omega, NOT mean anomaly 0. Getting
        # this wrong puts the planet a quarter-orbit off: no RM point lands in
        # transit and the anomaly is zero for EVERY lambda, so the fit runs,
        # converges, and reports a flat lambda posterior that reads as "no
        # detection". Nothing errors. This is the regression guard.
        d, _, p = setup(menu = false)
        th = seat(p, d)
        lls = Float64[]
        for λ in (-90.0, -55.0, 0.0, 55.0, 90.0)
            set_param!(th, "lambda_k1", deg2rad(λ))
            push!(lls, rv_log_likelihood(th, d))
        end
        @test all(isfinite, lls)
        @test length(unique(round.(lls, digits = 6))) > 1   # NOT a flat null
        # physics: symmetric in lambda about 0 for b fixed, and the anomaly
        # weakens toward polar where the track flattens
        @test lls[1] ≈ lls[5] rtol = 0.05
        @test lls[2] ≈ lls[4] rtol = 0.05
        @test lls[3] < lls[1]
    end

    @testset "responds to the shared v sin i slot" begin
        d, _, p = setup(menu = false)
        th = seat(p, d)
        set_param!(th, "lambda_k1", deg2rad(-55.0))
        a = rv_log_likelihood(th, d)
        set_param!(th, "v_sin_i_star", 2.5 * VSINI)
        @test rv_log_likelihood(th, d) != a
    end

    @testset "a_Rs is a constraint, not decoration" begin
        # Without use_rho_s the geometry is fixed by (M_s, R_s, P), so a passed
        # a_Rs would be silently ignored — the published value and its error
        # would vanish from the fit while appearing in the call.
        _, _, p = setup()
        @test "rho_s" in p.layout.names
        for (a, PP) in ((6.81, P), (12.0, 10.0), (3.5, 1.2))
            @test rho_s_to_a_Rs(_a_Rs_to_rho_s(a, PP), PP) ≈ a rtol = 1e-10
        end
    end

    @testset "lambda prior is uniform on the full circle" begin
        _, _, p = setup()
        u = findfirst(==(p.layout.name_to_idx["lambda_k1"]), p.layout.unfrozen_idx)
        lo, hi = bounds(p.layout.unfrozen_priors[u])
        # an obliquity fit that starts centred on zero is not measuring it
        @test lo ≈ -π && hi ≈ π
    end

    @testset "tomography joins the same Params" begin
        grid = collect(range(-70.0, 70.0; length = 40))
        tt = collect(range(Tc - 0.05, Tc + 0.05; length = 16))
        tn = [TomoNight("HARPS_map", tt, randn(MersenneTwister(2), 16, 40),
                        grid, Tc)]
        d, names, p = setup(tomo = tn)
        @test "tomo_alpha_HARPS_map" in p.layout.names
        # and lambda is still the SHARED slot, not a tomography-private one
        @test count(n -> occursin("lambda", n), p.layout.names) == 1
    end
end
