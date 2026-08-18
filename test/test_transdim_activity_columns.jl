# planet_active_* serialization — trans-dim chains must carry per-slot
# activity, not force consumers onto the n_planets fallback.
#
# Before this, trans-dim RJMCMC and PT chains recorded only `n_planets`,
# so every consumer (science_tables, posterior_plots, label_switching)
# reconstructed activity as "slot k active iff k ≤ n_planets" — an ORDER
# assumption that is wrong for any draw taken between a mid-slot death and
# the next birth, and silently so (the label_switching @warn was even being
# filtered out of shell output by grep -v pipelines).
#
# Locked invariants, for BOTH trans-dim samplers:
#   1. planet_active_1..max_kplanet columns exist and are 0/1;
#   2. their per-draw sum equals n_planets (activate!/deactivate! keep the
#      counter and the mask in lockstep — this catches either drifting);
#   3. the columns survive a save_chains/load_chains round-trip;
#   4. label_switching consumes them without the fallback warning.

using Nereus, Test, MCMCChains
using Random: MersenneTwister
using Statistics: std

function _toy_target()
    rng = MersenneTwister(99)
    t = sort(60.0 .* rand(rng, 40))
    P, K = 4.23, 50.0
    rv = K .* sin.(2π .* t ./ P) .+ 2.0 .* randn(rng, 40)
    data = Data(; t_rv = t, rv = rv, rv_err = fill(2.0, 40),
                  rv_inst = ones(Int, 40))
    params = Params(; max_kplanet = 2, planet_modes = fill(RV_ONLY, 2),
                      instruments = InstrumentConfig(rv = ["I1"]),
                      data = data, M_s = 1.0)
    return NereusTarget(params, data; unconstrained = false), data, params
end

function _check_activity(chains, K)
    nms = names(chains, :parameters)
    for k in 1:K
        @test Symbol("planet_active_$k") in nms
    end
    act = hcat((vec(Array(chains[Symbol("planet_active_$k")])) for k in 1:K)...)
    @test all(v -> v == 0.0 || v == 1.0, act)
    np = vec(Array(chains[:n_planets]))
    @test vec(sum(act; dims = 2)) == np          # mask ↔ counter lockstep
    return act
end

@testset "planet_active_* columns in trans-dim chains" begin
    target, data, params = _toy_target()
    td = TransDimConfig(max_kplanet = 2)

    @testset "RJMCMC writer" begin
        chains, _ = sample_rjmcmc(target, data; td = td,
                                   n_samples = 1500, n_warmup = 800,
                                   seed = 3, show_progress = false)
        act = _check_activity(chains, 2)
        @test 0.0 < sum(act) / length(act)       # something was ever active

        # The diagnostic must take the real columns silently — the fallback
        # path warns, so "no warning" proves the columns were used.
        rep = @test_logs min_level = Base.CoreLogging.Warn label_switching(chains)
        @test rep isa LabelSwitchingReport
    end

    @testset "MoMS writer" begin
        chains, _ = sample_moms(target, data; td = td,
                                 n_samples = 1200, n_warmup = 800,
                                 seed = 3, show_progress = false)
        _check_activity(chains, 2)
    end

    @testset "MoMS-NS writer" begin
        out = sample_moms_ns(target, data; td = td, n_live = 60,
                              n_mcmc = 10, dlogz = 2.0, max_iter = 20_000,
                              seed = 3, show_progress = false)
        ch = out isa Tuple ? out[1] : (out isa MCMCChains.Chains ? out : out.chains)
        _check_activity(ch, 2)
    end

    @testset "PT writer + save/load round-trip" begin
        chains, _, _ = sample_pt(target; td = td, n_rounds = 7, n_chains = 4,
                                  seed = 3, show_report = false)
        act = _check_activity(chains, 2)

        path = joinpath(mktempdir(), "transdim_activity.nc")
        save_chains(path, chains, params; data = data)
        loaded = load_chains(path)
        loaded = loaded isa Tuple ? loaded[1] : loaded
        act2 = _check_activity(loaded, 2)
        @test act2 == act                         # bit-exact round-trip

        rep = @test_logs min_level = Base.CoreLogging.Warn label_switching(loaded)
        @test rep isa LabelSwitchingReport
    end
end
