# sample_pt / sample_pt_emcee warn once, after sampling, when the minimum
# adjacent swap acceptance is below 1%, naming the pair; sampling is untouched.
using Test, Random, Nereus

function _make_rv_target(seed::Int = 2027; n_obs::Int = 30,
                          K_true::Float64 = 12.0, P_true::Float64 = 8.0)
    rng    = MersenneTwister(seed)
    t      = sort(rand(rng, n_obs) .* 40.0)
    rv     = K_true .* sin.(2π .* t ./ P_true) .+ randn(rng, n_obs)
    rv_err = fill(1.0, n_obs)
    ic     = InstrumentConfig(rv = ["SIM"])
    data   = Data(; t_rv = t, rv = rv, rv_err = rv_err)
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY], instruments = ic,
                    data = data, M_s = 1.0)
    target = NereusTarget(params, data; unconstrained = false)
    return target, data, params
end

@testset "_warn_min_swap" begin
    βs = [1.0, 0.5, 0.1, 0.0]
    @test_logs (:warn, r"rungs 2 and 3.*more temperatures"s) begin
        @test Nereus._warn_min_swap("sample_pt", βs, [0.3, 0.004, 0.2])
    end
    @test_logs begin
        @test !Nereus._warn_min_swap("sample_pt", βs, [0.3, 0.05, 0.2])
        @test !Nereus._warn_min_swap("sample_pt", βs, [0.3, 0.01, 0.2])
        @test !Nereus._warn_min_swap("sample_pt", βs, [0.3, NaN, 0.2])
        @test !Nereus._warn_min_swap("sample_pt", βs, Float64[])
    end
end

@testset "sample_pt_emcee warns on a broken ladder; output unchanged" begin
    target, data, _ = _make_rv_target(2027; n_obs = 120, K_true = 30.0)
    # A hot rung 1000x below the cold one with a sharp likelihood: replica exchange across that gap collapses.
    kw = (n_temps = 2, betas = [1.0, 1e-3], n_walkers = 16, n_steps = 300,
          n_burnin = 150, seed = 3, show_progress = false)
    res = sample_pt_emcee(target, data; kw...)
    @test minimum(res.acceptance_swap) < 0.01
    @test_logs (:warn, r"minimum adjacent swap acceptance") match_mode = :any min_level = Base.CoreLogging.Warn sample_pt_emcee(target, data; kw...)
    res2 = sample_pt_emcee(target, data; kw...)
    @test res.chains.value.data == res2.chains.value.data   # warning leaves sampling alone
end

@testset "sample_pt still runs with the check in place" begin
    target, _, _ = _make_rv_target(2027)
    chains, log_ev = sample_pt(target; n_rounds = 4, n_chains = 3, seed = 3,
                               show_report = false)
    @test size(chains, 1) > 0
end
