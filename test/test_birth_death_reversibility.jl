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
# is left in log_q_birth + log_q_death IS the detailed-balance error, in ln p.
using Test
using Random
using Nereus
using Nereus: propose_planet_birth, propose_planet_death,
               PriorBirth, MoMSBirth, InformedBirth, TransDimState, Theta,
               birth_group_keys, group_first_inactive, activate_planet!

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

    _, log_q_death = death(born, rng, strategy, data)
    return log_q_birth + log_q_death
end

strategy_aware(th, rng, s, data) = propose_planet_death(th, rng, s; data = data)
strategy_blind(th, rng, _, _) = propose_planet_death(th, rng)

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

    @testset "InformedBirth with its own death" begin
        # The informed death carries the informed-birth density (the full
        # peak mixture) at the killed values, evaluated from the post-death
        # state. Before, the pair used the strategy-blind death and every
        # round trip was short by log q_informed(β_killed).
        for seed in (1, 2, 3, 4, 5)
            @test roundtrip_log_q(p -> InformedBirth(), strategy_aware;
                                   seed = seed) ≈ 0.0 atol = 1e-9
        end
        @test abs(roundtrip_log_q(p -> InformedBirth(), strategy_blind; seed = 1)) > 1.0
    end
end

# Named-planet menus give every slot its own period window. Such slots are not
# interchangeable, so each is its own birth group: from {1} every free slot is
# a birth target, and the death {1,3} -> {1} has its reverse birth. Grouped by
# planet mode alone, a birth from {1} could only target slot 2, while that death
# was still proposed — probability leaked out of every such state.
@testset "slots with their own priors are separate birth groups" begin
    t_rv = collect(0.0:1.0:100.0)
    data = Data(; t_rv = t_rv, rv = 5.0 .* sin.(2π .* t_rv ./ 3.1) .+
                                     3.0 .* sin.(2π .* t_rv ./ 17.0),
                 rv_err = ones(length(t_rv)))
    win(lo, hi) = UniformPrior(lo, hi)
    params = Params(max_kplanet = 3, planet_modes = fill(RV_ONLY, 3),
                    instruments = InstrumentConfig(rv = ["HARPS"]), data = data, M_s = 1.0,
                    priors = Dict{String,PriorSpec}("P_k1" => win(3.0, 3.2),
                                                    "P_k2" => win(5.0, 6.0),
                                                    "P_k3" => win(15.0, 20.0)))
    keys = birth_group_keys(params)
    @test length(unique(keys)) == 3
    # identical windows collapse to one group, as before
    same = Params(max_kplanet = 2, planet_modes = fill(RV_ONLY, 2),
                  instruments = InstrumentConfig(rv = ["HARPS"]), data = data, M_s = 1.0,
                  priors = Dict{String,PriorSpec}("P_k1" => win(2.0, 30.0),
                                                  "P_k2" => win(2.0, 30.0)))
    @test length(unique(birth_group_keys(same))) == 1

    for strategy in (PriorBirth(), MoMSBirth(params), InformedBirth())
        theta = Theta{Float64}(params; td = TransDimState(max_planets = 3))
        activate_planet!(theta.td, 1)
        # (the active planet's own values must sit inside its window)
        theta.values[params.layout.planet_blocks[1].P] = 3.1
        @test group_first_inactive(theta.td, keys) == [2, 3]
        n_checked = 0
        for seed in 1:60
            rng = MersenneTwister(seed)
            born, lqb = propose_planet_birth(theta, rng, strategy; data = data)
            isfinite(lqb) || continue
            born.td.planet_active[3] || continue          # the slot-3 birth
            # the death that kills exactly slot 3 must undo it
            for dseed in 1:200
                drng = MersenneTwister(dseed)
                dead, lqd = propose_planet_death(born, drng, strategy; data = data)
                isfinite(lqd) || continue
                dead.td.planet_active[3] && continue
                @test dead.td.planet_active == theta.td.planet_active
                @test lqb + lqd ≈ 0.0 atol = 1e-9
                n_checked += 1
                break
            end
            n_checked >= 3 && break
        end
        @test n_checked >= 1
    end
end


@testset "DonorBirth carries the donor mixture density both ways" begin
    # DonorBirth clones a donor's planet with jitter. It used to call that
    # proposal symmetric (combinatorial ratio only) and to share the
    # strategy-blind death, so neither side carried the mixture density.
    params, data = rv_params(2)
    rng = MersenneTwister(5)
    function donor(P, K)
        d = Theta{Float64}(params; td = TransDimState(max_planets = 2))
        for (i, slot) in enumerate(params.layout.unfrozen_idx)
            lo, hi = bounds(params.layout.unfrozen_priors[i])
            d.values[slot] = clamp(rand(rng, params.layout.unfrozen_priors[i].dist), lo, hi)
        end
        Nereus.set_param!(d, "P_k1", P)
        haskey(params.layout.name_to_idx, "K_k1") && Nereus.set_param!(d, "K_k1", K)
        activate_planet!(d.td, 1)
        return d
    end
    pop = [donor(5.1, 20.0), donor(5.12, 19.0), donor(12.0, 3.0)]
    for seed in 1:5
        th = Theta{Float64}(params; td = TransDimState(max_planets = 2))
        r = MersenneTwister(seed)
        born, lb = propose_planet_birth(th, r, Nereus.DonorBirth(); data = data,
                                        population = pop)
        isfinite(lb) || continue
        _, ld = propose_planet_death(born, r, Nereus.DonorBirth(); data = data,
                                     population = pop)
        @test lb + ld ≈ 0.0 atol = 1e-9
        # the blind death misses the density: its round trip is left with
        # exactly −log q_donor, which is not zero
        _, lblind = propose_planet_death(born, r)
        @test abs(lb + lblind) > 1e-3
    end
    # no donors: both directions are the PriorBirth pair
    th = Theta{Float64}(params; td = TransDimState(max_planets = 2))
    r = MersenneTwister(9)
    born, lb = propose_planet_birth(th, r, Nereus.DonorBirth(); data = data,
                                    population = Theta{Float64}[])
    _, ld = propose_planet_death(born, r, Nereus.DonorBirth(); data = data,
                                 population = Theta{Float64}[])
    @test lb + ld ≈ 0.0 atol = 1e-9
end

@testset "an informed death only reads the periodogram cache" begin
    # Births record the active-set keys whose cache entries a checkpoint must
    # carry. A death that stored an entry of its own (or triggered the
    # oversize reset) made a resumed rjmcmc run diverge from a straight one.
    params, data = rv_params(2)
    Nereus.reset_transdim_caches!()
    rng = MersenneTwister(4)
    th = Theta{Float64}(params; td = TransDimState(max_planets = 2))
    born, lb = propose_planet_birth(th, rng, InformedBirth(); data = data)
    @test isfinite(lb)
    before = Dict(k => (copy(v.periods), v.calls_since_update)
                  for (k, v) in Nereus._PEAK_CACHES)
    for _ in 1:20
        propose_planet_death(born, rng, InformedBirth(); data = data)
    end
    after = Dict(k => (copy(v.periods), v.calls_since_update)
                 for (k, v) in Nereus._PEAK_CACHES)
    @test after == before
end
