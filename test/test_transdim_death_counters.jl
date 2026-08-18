# Per-temperature planet birth/death accounting for trans-dim ptemcee.
#
# The PT immunity argument for trans-dim model selection is that hot chains
# see a softened likelihood, accept deaths that the cold chain cannot, and
# swaps propagate that back down to β=1. That argument only holds if the
# ladder actually reaches high enough temperature — and the only way to know
# is to measure death acceptance PER TEMPERATURE. If even the hottest rung's
# death acceptance is ~0, PT is not rescuing anything and the chain is frozen
# at whatever model order it entrenched in.
#
# Before this, that number was underivable from a run:
#   * td_proposed/td_accepted are per-planet-SLOT and pool births with deaths,
#   * noise_td_proposed/accepted are per-temperature but pool noise births,
#     noise deaths and two burn-in swap move classes.
# So no per-temperature rate existed for ANY move class.
using Test
using Random
using Nereus

@testset "per-temperature planet birth/death counters" begin
    t_rv = collect(0.0:1.0:120.0)
    ic   = InstrumentConfig(rv = ["HARPS"])
    data = Data(;
        t_rv   = t_rv,
        rv     = 30.0 .* sin.(2π .* t_rv ./ 7.3),
        rv_err = ones(length(t_rv)),
    )
    params = Params(max_kplanet = 2, planet_modes = [RV_ONLY, RV_ONLY],
                    instruments = ic, data = data, M_s = 1.0)
    target = NereusTarget(params, data; unconstrained = false)
    td     = TransDimConfig(max_kplanet = 2)

    n_temps = 3
    res = sample_transdim_ptemcee(target, data; td = td,
                                   n_temps = n_temps, n_walkers = 8,
                                   n_steps = 60, n_burnin = 30,
                                   seed = 7, show_progress = false)

    @testset "counters are per-temperature and split by direction" begin
        for f in (:planet_birth_proposed, :planet_birth_accepted,
                  :planet_death_proposed, :planet_death_accepted)
            @test hasproperty(res, f)
            @test length(getproperty(res, f)) == n_temps
        end
    end

    @testset "accepted never exceeds proposed at any temperature" begin
        @test all(res.planet_birth_accepted .<= res.planet_birth_proposed)
        @test all(res.planet_death_accepted .<= res.planet_death_proposed)
    end

    @testset "both directions are actually wired, not just allocated" begin
        @test sum(res.planet_birth_proposed) > 0
        @test sum(res.planet_death_proposed) > 0
    end

    @testset "direction split reconciles with the pooled per-slot counts" begin
        # Every planet move counted per-slot must appear in exactly one
        # direction bucket — this is what makes a death RATE meaningful.
        @test sum(res.planet_birth_proposed) + sum(res.planet_death_proposed) ==
              sum(res.td_proposed)
        @test sum(res.planet_birth_accepted) + sum(res.planet_death_accepted) ==
              sum(res.td_accepted)
    end
end
