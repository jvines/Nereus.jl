# Detailed balance for the trans-dim birth/death pair.
#
# Both proposal routines return a Hastings ratio in the same convention:
# log(q_reverse / q_forward) for the move they performed. For ONE pair of
# states (k planets) <-> (k+1 planets) that means the birth's ratio and the
# death's ratio are the same quantity with opposite signs:
#
#     log(q_death/q_birth) + log(q_birth/q_death) == 0
#
# so their sum must vanish EXACTLY, whatever the priors, dimensions or
# combinatorial factors are. This is a closed-form identity — no chain, no
# sampling noise, no tolerance argument.
#
# The likelihood and prior terms cannot rescue an imbalance here: log_prior
# skips inactive slots and the likelihood ignores them, so for this pair
# ΔlogL and Δlogπ are exactly antisymmetric and cancel in the sum. Whatever
# is left in log_q_birth + log_q_death IS the detailed-balance error, in nats.
using Test
using Random
using Nereus
using Nereus: propose_planet_birth, propose_planet_death,
               PriorBirth, MoMSBirth, InformedBirth, TransDimState, Theta

function rv_params(max_k::Int)
    t_rv = collect(0.0:1.0:100.0)
    ic   = InstrumentConfig(rv = ["HARPS"])
    data = Data(; t_rv = t_rv, rv = 20.0 .* sin.(2π .* t_rv ./ 5.1),
                 rv_err = ones(length(t_rv)))
    params = Params(max_kplanet = max_k,
                    planet_modes = fill(RV_ONLY, max_k),
                    instruments = ic, data = data, M_s = 1.0)
    return params, data
end

# One birth immediately followed by the death that undoes it. `make_strategy`
# builds the birth strategy from `params`; `death` performs the reverse move.
function roundtrip_log_q(make_strategy, death; max_k::Int = 1, seed::Int = 42)
    params, data = rv_params(max_k)
    strategy = make_strategy(params)
    theta    = Theta{Float64}(params; td = TransDimState(max_planets = max_k))
    rng      = MersenneTwister(seed)

    born, log_q_birth = propose_planet_birth(theta, rng, strategy; data = data)
    isfinite(log_q_birth) || return NaN
    born.td.n_planets_active == 1 ||
        error("birth did not activate exactly one slot")

    _, log_q_death = death(born, rng, strategy)
    return log_q_birth + log_q_death
end

strategy_aware(th, rng, s) = propose_planet_death(th, rng, s)
strategy_blind(th, rng, _) = propose_planet_death(th, rng)

@testset "birth/death Hastings ratios are reciprocal" begin
    @testset "MoMS birth with MoMS death" begin
        for seed in (1, 2, 3)
            @test roundtrip_log_q(p -> MoMSBirth(p), strategy_aware;
                                   seed = seed) ≈ 0.0 atol = 1e-9
        end
    end

    @testset "PriorBirth with its strategy-aware death" begin
        for seed in (1, 2, 3)
            @test roundtrip_log_q(p -> PriorBirth(), strategy_aware;
                                   seed = seed) ≈ 0.0 atol = 1e-9
        end
    end

    @testset "PriorBirth with the strategy-blind death is NOT reversible" begin
        # Regression guard on the bug this file was written for: the
        # combinatorial-only death is short by Σ logpdf(slab_prior, β_killed).
        # Keep it pinned so nobody "simplifies" the dispatch back.
        for seed in (1, 2, 3)
            @test roundtrip_log_q(p -> PriorBirth(), strategy_blind;
                                   seed = seed) > 1.0
        end
    end

    @testset "InformedBirth with the strategy-blind death" begin
        # KNOWN OPEN: informed births are documented as pairing with the
        # generic uniform death, but that death carries no reverse density
        # either, so the same imbalance applies. Fixing it needs the informed
        # mixture density evaluated at β_killed.
        @test_broken roundtrip_log_q(p -> InformedBirth(), strategy_blind;
                                      seed = 1) ≈ 0.0 atol = 1e-9
    end
end
