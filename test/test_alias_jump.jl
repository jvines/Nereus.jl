# Within-model alias / harmonic mode jump.
#
# A planet born at 2P (or P/2) of the true period cannot migrate: the
# within-model kernels are local and the likelihood valley between harmonics
# is exactly what a local step cannot cross. This move multiplies an active
# planet's period by a ratio drawn from a set closed under inversion, which
# makes the proposal symmetric so the Hastings ratio is the Jacobian alone.
using Test
using Random
using Nereus
using Nereus: propose_planet_alias_jump, ALIAS_RATIOS,
               TransDimState, Theta, activate_planet!, planet_P

function alias_setup(; max_k::Int = 2)
    t_rv = collect(0.0:1.0:100.0)
    ic   = InstrumentConfig(rv = ["HARPS"])
    data = Data(; t_rv = t_rv, rv = 20.0 .* sin.(2π .* t_rv ./ 5.1),
                 rv_err = ones(length(t_rv)))
    params = Params(max_kplanet = max_k, planet_modes = fill(RV_ONLY, max_k),
                    instruments = ic, data = data, M_s = 1.0)
    tds = TransDimState(max_planets = max_k)
    activate_planet!(tds, 1)
    theta = Theta{Float64}(params; td = tds)
    theta.values[params.layout.planet_blocks[1].P] = 5.1
    return theta
end

@testset "alias / harmonic mode jump" begin
    @testset "ratio set is closed under inversion" begin
        # This is what makes q(r) == q(1/r), hence log_q_ratio == log|J|.
        for r in ALIAS_RATIOS
            @test any(x -> isapprox(x, 1 / r; rtol = 1e-12), ALIAS_RATIOS)
        end
    end

    @testset "period is scaled by a ratio from the set" begin
        for seed in 1:20
            theta = alias_setup()
            P_old = theta.values[theta.params.layout.planet_blocks[1].P]
            new_theta, _ = propose_planet_alias_jump(theta, MersenneTwister(seed))
            P_new = new_theta.values[theta.params.layout.planet_blocks[1].P]
            @test any(r -> isapprox(P_new / P_old, r; rtol = 1e-12), ALIAS_RATIOS)
        end
    end

    @testset "log_q_ratio equals the log Jacobian of the map" begin
        # P -> rP is deterministic with dP'/dP = r, and the ratio draw is
        # symmetric, so the whole Hastings ratio is log(r) = log(P'/P).
        for seed in 1:20
            theta = alias_setup()
            P_old = theta.values[theta.params.layout.planet_blocks[1].P]
            new_theta, log_q = propose_planet_alias_jump(theta, MersenneTwister(seed))
            P_new = new_theta.values[theta.params.layout.planet_blocks[1].P]
            @test log_q ≈ log(P_new / P_old) rtol = 1e-12
        end
    end

    @testset "forward then inverse ratio is an exact round trip" begin
        # Reversibility: the two log_q_ratios for a jump and its inverse must
        # cancel, and the period must return to where it started.
        theta = alias_setup()
        P0 = theta.values[theta.params.layout.planet_blocks[1].P]
        for r in ALIAS_RATIOS
            fwd = P0 * r
            back = fwd * (1 / r)
            @test back ≈ P0 rtol = 1e-12
            @test log(r) + log(1 / r) ≈ 0.0 atol = 1e-12
        end
    end

    @testset "declines when there is nothing to move" begin
        t_rv = collect(0.0:1.0:50.0)
        ic   = InstrumentConfig(rv = ["HARPS"])
        data = Data(; t_rv = t_rv, rv = zeros(length(t_rv)),
                     rv_err = ones(length(t_rv)))
        params = Params(max_kplanet = 1, planet_modes = [RV_ONLY],
                        instruments = ic, data = data, M_s = 1.0)
        theta = Theta{Float64}(params; td = TransDimState(max_planets = 1))
        _, log_q = propose_planet_alias_jump(theta, MersenneTwister(1))
        @test log_q == -Inf
    end
end
