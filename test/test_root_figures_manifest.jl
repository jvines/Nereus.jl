# ppc.png and detection_limits.png are written to the output directory root,
# outside the plots/ walk that builds summary["figures"], so a client never
# listed them. `_add_root_figures!` adds the ones THIS job wrote.
using Test, Nereus, MCMCChains, Random
using Random: MersenneTwister

@testset "_add_root_figures!" begin
    mktempdir() do dir
        write(joinpath(dir, "ppc.png"), "x")
        write(joinpath(dir, "detection_limits.png"), "x")
        write(joinpath(dir, "detection_limits.pdf"), "x")
        # Only what this job wrote (it is in summary["plots"]) is listed.
        s = Dict{String,Any}("plots" => [joinpath(dir, "ppc.png")],
                             "figures" => Dict{String,Any}("models/RV_phasefold_K1" => "a.png"))
        Nereus._add_root_figures!(s, dir)
        @test s["figures"]["ppc"] == joinpath(dir, "ppc.png")
        @test !haskey(s["figures"], "detection_limits")     # a stale leftover
        @test s["figures"]["models/RV_phasefold_K1"] == "a.png"
        s2 = Dict{String,Any}("plots" => [joinpath(dir, "detection_limits.png")])
        Nereus._add_root_figures!(s2, dir)                  # no manifest yet: created
        @test s2["figures"]["detection_limits"] == joinpath(dir, "detection_limits.png")
        @test s2["figures_pdf"]["detection_limits"] == joinpath(dir, "detection_limits.pdf")
    end
end

@testset "a PPC step's figure reaches the manifest" begin
    rng = MersenneTwister(42)
    n_obs = 40
    t = sort!(120.0 .* rand(rng, n_obs))
    data0 = Data(; t_rv = t, rv = zeros(n_obs), rv_err = fill(1.0, n_obs),
                   rv_inst = ones(Int, n_obs))
    params = Params(; parametrization = ParametrizationConfig(time = :Mo),
        max_kplanet = 1, planet_modes = [RV_ONLY],
        instruments = InstrumentConfig(rv = ["SIM"]), data = data0, M_s = 1.0)
    th = Theta{Float64}(params)
    set_param!(th, "P_k1", 20.0); set_param!(th, "K_k1", 5.0)
    set_param!(th, "sesinw_k1", 0.2); set_param!(th, "secosw_k1", 0.2)
    set_param!(th, "Mo_k1", 0.7); set_param!(th, "gamma_SIM", 0.0)
    set_param!(th, "sigma_SIM", 1.0)
    rv0, _ = rv_predictions(th, data0)
    data = Data(; t_rv = t, rv = rv0 .+ randn(rng, n_obs), rv_err = fill(1.0, n_obs),
                  rv_inst = ones(Int, n_obs))
    truth = Dict(:P_k1 => 20.0, :K_k1 => 5.0, :sesinw_k1 => 0.2, :secosw_k1 => 0.2,
                 :Mo_k1 => 0.7, :gamma_SIM => 0.0, :sigma_SIM => 1.0)
    names = Symbol.(params.layout.unfrozen_names)
    arr = zeros(200, length(names), 1)
    for (j, nm) in enumerate(names)
        arr[:, j, 1] .= get(truth, nm, 0.0) .+ 1e-6 .* randn(rng, 200)
    end
    chains = Chains(arr, names)
    mktempdir() do dir
        summary = Dict{String,Any}("figures" => Dict{String,Any}())
        Nereus._run_ppc!(Dict("output" => Dict("ppc" => true, "ppc_n_draws" => 50)),
                         chains, params, data, dir, summary)
        @test isfile(joinpath(dir, "ppc.png"))
        Nereus._add_root_figures!(summary, dir)
        @test summary["figures"]["ppc"] == joinpath(dir, "ppc.png")
    end
end
