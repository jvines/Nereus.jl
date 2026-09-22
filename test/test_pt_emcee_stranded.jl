# pt_emcee on an easy target with default settings.
#
# Reported: a single-planet RV fit with the default settings failed Nereus's own
# health check (R-hat 1.4-1.9), with walkers stranded across the 0.1-3000 d
# period prior. Three causes, each tested here:
#   - run_job defaulted to `Tp` over the whole baseline: baseline/P copies of
#     the orbit that an ensemble cannot mix across;
#   - walkers moved in linear P over a log-uniform prior spanning 4.5 decades;
#   - a stretch move cannot bring a walker from the plateau into a narrow mode,
#     so a found mode filled its cold rung at ~5 walkers per 100 steps.

using Test, Nereus, Random, Statistics, MCMCChains
using Nereus: _pt_prune_stranded!, _warn_repeating_time_window

# One rung of `n_mode` walkers in a Gaussian mode (σ = 0.01, centred on 0.5)
# and `n_flat` spread over the unit box `gap` nats of log-likelihood below it.
# Uniform prior on [0, 1]^d, so log π = 0 and ∫ π = 1.
function _rung(; d = 3, n_mode = 70, n_flat = 30, gap = 60.0, seed = 1)
    rng = MersenneTwister(seed)
    n = n_mode + n_flat
    state = Array{Float64,3}(undef, 1, n, d)
    logL = Matrix{Float64}(undef, 1, n)
    for w in 1:n_mode
        x = 0.5 .+ 0.01 .* randn(rng, d)
        state[1, w, :] = x
        logL[1, w] = -0.5 * sum(abs2, (x .- 0.5) ./ 0.01)
    end
    for w in (n_mode + 1):n
        state[1, w, :] = rand(rng, d)
        logL[1, w] = -gap
    end
    return state, zeros(1, n), logL
end

@testset "pt_emcee: stranded walkers and default convergence" begin

    @testset "prune moves a plateau that provably weighs nothing" begin
        # Z_mode ≈ (2π)^{3/2} 0.01³ → log ≈ -11.1. The plateau's whole region
        # weighs at most e^-60, so it is 49 nats below: moved.
        state, lπ, lL = _rung(gap = 60.0)
        before = copy(state)
        moved = _pt_prune_stranded!(state, lπ, lL, [1.0], MersenneTwister(2))
        @test moved == [30]
        @test state[1, 1:70, :] == before[1, 1:70, :]          # mode untouched
        mode_rows = [before[1, w, :] for w in 1:70]
        @test all(w -> state[1, w, :] in mode_rows, 71:100)  # copies of mode walkers
        @test all(lL[1, 71:100] .> -60)
    end

    @testset "prune leaves a plateau that could carry mass" begin
        # 18 nats down is past the Gaussian allowance (~15 for d = 3), but the
        # plateau fills the whole prior and the mode's Occam factor is 11 nats:
        # the plateau may weigh e^-7 of the mode. Not provably nothing: kept.
        state, lπ, lL = _rung(gap = 18.0)
        before = copy(state)
        @test _pt_prune_stranded!(state, lπ, lL, [1.0], MersenneTwister(2)) == [0]
        @test state == before
    end

    @testset "tempering shrinks the gap, so hot rungs keep their plateau" begin
        state, lπ, lL = _rung(gap = 60.0)
        @test _pt_prune_stranded!(state, lπ, lL, [0.1], MersenneTwister(2)) == [0]
    end

    @testset "no estimate of the mode without enough walkers in it" begin
        state, lπ, lL = _rung(n_mode = 4, n_flat = 96, gap = 60.0)
        @test _pt_prune_stranded!(state, lπ, lL, [1.0], MersenneTwister(2)) == [0]
    end

    @testset "trans-dim prune moves the model too" begin
        # 70 walkers with planet 1 active in the mode, 30 without it on the
        # no-planet plateau 60 nats down. All three dims belong to planet 1.
        state, lπ, lL = _rung(gap = 60.0)
        tds = [Nereus.TransDimState(w <= 70 ? 1 : 0, BitVector([w <= 70]),
                                    BitVector(), BitVector([true]))
               for _ in 1:1, w in 1:100]
        shift = Union{Nothing,Float64}[nothing, nothing, nothing]
        moved = Nereus._td_prune_stranded!(state, tds, lπ, lL, [1.0], shift,
                                           [1, 1, 1], 0, MersenneTwister(2))
        @test moved == [30]
        @test all(w -> tds[1, w].planet_active[1] && tds[1, w].n_planets_active == 1, 71:100)
        @test tds[1, 71].planet_active !== tds[1, 1].planet_active   # copied, not aliased

        # 18 nats down could still carry mass: left alone, model and all.
        state, lπ, lL = _rung(gap = 18.0)
        tds = [Nereus.TransDimState(w <= 70 ? 1 : 0, BitVector([w <= 70]),
                                    BitVector(), BitVector([true]))
               for _ in 1:1, w in 1:100]
        @test Nereus._td_prune_stranded!(state, tds, lπ, lL, [1.0], shift,
                                         [1, 1, 1], 0, MersenneTwister(2)) == [0]
        @test count(w -> tds[1, w].planet_active[1], 1:100) == 70
    end

    @testset "convergence_report: label switching is not non-convergence" begin
        # One planet, two exchangeable RV slots. Walkers 1-10 hold it in slot 2
        # for the whole run, the rest in slot 1 (births sort, deaths do not
        # compact). Flat chain, walker = fast index, as transdim_pt_emcee saves.
        rng = MersenneTwister(5)
        nw, ns = 100, 400
        n = nw * ns
        w_of(i) = mod1(i, nw)
        in2 = [w_of(i) <= 10 for i in 1:n]
        P = 12.3 .+ 0.001 .* randn(rng, n); K = 12.0 .+ 0.3 .* randn(rng, n)
        cols = Dict(
            :P_k1 => ifelse.(in2, 900.0, P), :K_k1 => ifelse.(in2, 3.0, K),
            :P_k2 => ifelse.(in2, P, 450.0), :K_k2 => ifelse.(in2, K, 7.0),
            :planet_active_1 => Float64.(.!in2), :planet_active_2 => Float64.(in2),
            :n_planets => ones(n))
        nm = [:P_k1, :K_k1, :P_k2, :K_k2, :n_planets, :planet_active_1, :planet_active_2]
        ch = Chains(hcat((cols[s] for s in nm)...), nm)
        params = Params(max_kplanet = 2, planet_modes = [RV_ONLY, RV_ONLY], stability = :none,
                        instruments = InstrumentConfig(rv = ["HARPS"]),
                        data = Data(; t_rv = collect(0.0:10.0:990.0),
                                    rv = zeros(100), rv_err = ones(100),
                                    rv_inst = ones(Int, 100)))
        r = Nereus.convergence_report(ch, nw; model_params = params, io = devnull)
        @test r.pass
        # The planet itself still has to mix: ten walkers stuck off-mode make
        # the canonical slot-1 period fail.
        cols[:P_k1][[i for i in 1:n if 50 <= w_of(i) <= 59]] .= 13.0
        ch2 = Chains(hcat((cols[s] for s in nm)...), nm)
        @test !Nereus.convergence_report(ch2, nw; model_params = params, io = devnull).pass
    end

    @testset "Tp/Tc windows wider than a period warn" begin
        pri = Dict("P_k1" => LogUniformPrior(0.1, 3000.0),
                   "Tp_k1" => UniformPrior(0.0, 929.0))
        @test_logs (:warn, r"identical copies") _warn_repeating_time_window(pri, 1, :Tp)
        pri["Tp_k1"] = UniformPrior(0.0, 0.05)
        @test_logs _warn_repeating_time_window(pri, 1, :Tp)
        @test_logs _warn_repeating_time_window(pri, 1, :Mo)
    end

    # The reported symptom, end to end: run_job with no parametrization and no
    # sampler settings, on an easy 12.3 d planet over a 1000 d baseline.
    rng = MersenneTwister(20260922)
    n = 60
    t = sort(2459000.0 .+ 1000.0 .* rand(rng, n))
    P, K, e, ω, tp = 12.3, 12.0, 0.1, 1.0, 2459003.0
    function _rv(ti)
        M = mod2pi(2π * (ti - tp) / P)
        E = M
        for _ in 1:50
            E -= (E - e * sin(E) - M) / (1 - e * cos(E))
        end
        ν = 2atan(sqrt((1 + e) / (1 - e)) * tan(E / 2))
        return K * (cos(ν + ω) + e * cos(ω))
    end
    rv = _rv.(t) .+ 2.0 .* randn(rng, n) .+ 5.0
    cfg = Dict{String,Any}(
        "seed" => 42,
        "output_dir" => mktempdir(),
        "data" => Dict("rv" => Dict("values" => Dict(
            "bjd" => t, "rv" => rv, "rv_err" => fill(2.0, n),
            "instrument" => fill("HARPS", n)))),
        "model" => Dict("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"]),
        "sampler" => Dict("name" => "pt_emcee",
                          "kwargs" => Dict("show_progress" => false)),
        "output" => Dict("plots" => String[], "ppc" => false,
                         "detection_limits" => false, "loo" => false))

    @testset "run_job defaults converge on an easy target" begin
        res = Nereus.run_job(cfg)
        s = res isa AbstractDict ? res : res.summary
        checks = s["fit_health"]["checks"]
        @test checks["convergence"]["status"] == "ok"
        @test checks["multimodality"]["status"] == "ok"
        # The reported false positive: P = 12.3 d on LogUniform(0.1, 3000).
        @test !occursin("P_k1", checks["prior_rail"]["message"])
        ch, _ = Nereus.load_chains(joinpath(cfg["output_dir"], "chains.nc"))
        @test :Mo_k1 in names(ch)                       # the default is Mo now
        Pd = vec(Array(ch[:P_k1]))
        @test mean(abs.(Pd .- 12.3) ./ 12.3 .< 0.01) == 1.0   # no stranded draws
    end

    # Trans-dim, defaults, same target, up to 2 planets. Before: 5 temps and
    # blind births found no planet (P(Nₚ = 0) = 0.84, no draw at 12.3 d).
    @testset "transdim_pt_emcee defaults find an easy planet" begin
        tcfg = merge(cfg, Dict{String,Any}(
            "output_dir" => mktempdir(),
            "model" => Dict("max_kplanet" => 2, "planet_modes" => ["RV_ONLY", "RV_ONLY"]),
            "transdim" => Dict("max_kplanet" => 2),
            "sampler" => Dict("name" => "transdim_pt_emcee",
                              "kwargs" => Dict("show_progress" => false))))
        s = Nereus.run_job(tcfg)
        ch, _ = Nereus.load_chains(joinpath(tcfg["output_dir"], "chains.nc"))
        np = vec(Array(ch[:n_planets]))
        @test mean(np .== 1) > 0.95
        has = falses(length(np))
        for k in 1:2
            on = vec(Array(ch[Symbol("planet_active_$k")])) .> 0.5
            Pk = vec(Array(ch[Symbol("P_k$k")]))
            has .|= on .& (abs.(Pk .- 12.3) ./ 12.3 .< 0.01)
        end
        @test mean(has) > 0.95
        # Both health signals: fit_health (masked to active draws) and the
        # sampler's own report (components, not slots).
        @test s["fit_health"]["checks"]["convergence"]["status"] == "ok"
        @test s["run_info"]["convergence"]["pass"] == true
    end

    @testset "draws and :lp are recorded in x, not on the move scale" begin
        data, irv, ipm, ias = Nereus._build_data(cfg["data"])
        _, target, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()),
                                           irv, ipm, ias)
        r = sample_pt_emcee(target, data; n_temps = 6, n_walkers = 30,
                            n_steps = 60, n_burnin = 30, seed = 3,
                            show_progress = false)
        nms = Symbol.(target.params.layout.unfrozen_names)
        X = hcat([vec(Array(r.chains[p])) for p in nms]...)
        lp = vec(Array(r.chains[:lp]))
        @test all(X[:, findfirst(==(:P_k1), nms)] .>= 0.1)
        for i in round.(Int, range(1, size(X, 1); length = 20))
            @test lp[i] ≈ Nereus.logdensity_bounded(target, X[i, :]) rtol = 1e-9
        end
    end
end
