# sample_pt_whitening warns once (per session) that it is deprecated.
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

@testset "pt_whitening warns that it is deprecated" begin
    target, data, _ = _make_rv_target(2027)
    run() = sample_pt_whitening(target, data; n_temps = 4, n_steps = 60,
                                n_burnin = 20, warmup_swaps = 20, seed = 1,
                                show_progress = false)
    dep(l) = l.level == Base.CoreLogging.Warn && occursin("sample_pt_whitening is deprecated", l.message)
    logs, _ = Test.collect_test_logs(run)
    @test count(dep, logs) == 1
    @test occursin("maxlog = 1", read(joinpath(@__DIR__, "..", "src", "samplers", "pt_whitening.jl"), String))
    src = read(joinpath(@__DIR__, "..", "src", "samplers", "pt_whitening.jl"), String)
    @test !occursin("matches the validated pt_emcee ladder", src)
    @test !occursin("improves swap acceptance by 2-5", src)
end
