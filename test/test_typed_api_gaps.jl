# The typed fit_* route reaches what a run_job config does: planet modes,
# noise models and menu, activity indicators, config_overrides, and the
# post-fit operations on a finished fit (loo, ppc, fit_health, evidence,
# detection_limits, label_switching, select_*, pre_white, lc_periodogram).
using Test, Nereus, Random, Statistics

const _rvdat(n; seed = 3) = let rng = MersenneTwister(seed)
    t = sort(rand(rng, n) .* 120) .+ 59000.0
    v = 12.0 .* sin.(2π .* (t .- 59000.0) ./ 17.3) .+ 1.0 .* randn(rng, n)
    (t = t, rv = v, rv_err = fill(1.0, n))
end

@testset "typed API gaps: target options" begin
    d1 = _rvdat(40); d2 = _rvdat(30; seed = 5)
    base(; kw...) = Dict{String,Any}("source" => "RV",
        "data" => Dict{String,Any}("A" => d1, "B" => d2), (String(k) => v for (k, v) in kw)...)
    pl = Nereus._planet_spec(1; block = Nereus.default_rv_planet(d1.t, d1.rv))
    tgt(c; kw...) = Nereus._target_from([c], pl; kw...)

    @testset "noise_models as job-config dicts" begin
        nmd = [Dict("kind" => "CeleriteRotation", "channel" => "rv",
                    "instruments" => ["A"])]
        t = tgt(base(); noise_models = nmd)
        @test length(t.params.config.noise_models) == 1
        @test t.params.config.noise_models[1] isa CeleriteRotation
        @test_throws ArgumentError tgt(base(); noise_models = [1])
    end

    @testset "indicator_cols and noise_menu" begin
        withind(d, s) = merge(d, (bis = randn(MersenneTwister(s), length(d.t)),
                                  bis_err = fill(0.5, length(d.t))))
        ch = Dict{String,Any}("source" => "RV", "indicator_cols" => ["bis"],
            "data" => Dict{String,Any}("A" => withind(d1, 1), "B" => withind(d2, 2)))
        t = tgt(ch)
        @test haskey(t.data.indicators, "bis")
        @test length(t.data.indicators["bis"]) == 70
        @test haskey(t.data.indicator_errs, "bis")
        # without indicator_cols the extra column is not read
        ch0 = Dict{String,Any}("source" => "RV", "data" => ch["data"])
        @test isempty(tgt(ch0).data.indicators)
        # a column an instrument lacks
        chb = Dict{String,Any}("source" => "RV", "indicator_cols" => ["bis"],
            "data" => Dict{String,Any}("A" => withind(d1, 1), "B" => d2))
        @test_throws ArgumentError tgt(chb)
        # `indicators` supplied by instrument
        chi = Dict{String,Any}("source" => "RV",
            "indicators" => Dict("A" => Dict("fwhm" => randn(40) .+ 5),
                                 "B" => Dict("fwhm" => randn(30) .+ 5)),
            "data" => Dict{String,Any}("A" => d1, "B" => d2))
        @test haskey(tgt(chi).data.indicators, "fwhm")
        # the menu
        menu = Any[]
        t2 = tgt(ch; noise_menu = Dict("indicators" => ["bis"]), menu_out = menu)
        @test !isempty(t2.params.config.noise_models)
        @test length(menu) == 1 && !isempty(menu[1].toggleable)
        @test_throws ArgumentError tgt(ch; noise_menu = Dict("indicators" => ["nope"]))
        @test_throws ArgumentError tgt(ch; noise_menu = Dict(),
                    noise_models = [Dict("kind" => "CeleriteRotation")])
    end

    @testset "planet_modes" begin
        ph = Dict{String,Any}("source" => "PM", "data" => Dict{String,Any}(
            "T" => (t = d1.t, flux = ones(40), flux_err = fill(1e-3, 40))))
        t = Nereus._target_from([ph], 1; planet_modes = ["PM_GD"])
        @test t.params.config.planet_modes[1] == Nereus.PM_GD
        @test Nereus.has_gd(t.params.config.planet_modes[1])
        t0 = Nereus._target_from([ph], 1)
        @test !Nereus.has_gd(t0.params.config.planet_modes[1])
        @test length(t.params.layout.unfrozen_names) > length(t0.params.layout.unfrozen_names)
        @test_throws ArgumentError Nereus._target_from([ph], 1; planet_modes = ["NOPE"])
        @test_throws ArgumentError Nereus._target_from([ph], 1; planet_modes = ["RVPM_GD"])
        @test_throws ArgumentError Nereus._target_from([ph], 1; planet_modes = ["PM_GD", "PM_GD"])
    end

    @testset "config_overrides on the target route" begin
        kw = Nereus._apply_target_overrides(
            Dict("star" => Dict("M_s" => 0.9, "T_eff" => 5500.0),
                 "model" => Dict("stability" => "amd", "trend_order" => 1,
                                 "parametrization" => Dict("time" => "Tp")),
                 "priors" => Dict("sigma_A" => Dict("type" => "uniform", "lo" => 0.0, "hi" => 3.0))),
            (priors = nothing,), "fit_rv")
        @test kw.M_s == 0.9 && kw.T_eff == 5500.0
        @test kw.stability === :amd && kw.trend_order == 1 && kw.time_anchor === :Tp
        @test kw.priors["sigma_A"] isa PriorSpec
        @test_throws ArgumentError Nereus._apply_target_overrides(
            Dict("sampler" => Dict("name" => "nuts")), (;), "fit_rv")
        @test_throws ArgumentError Nereus._apply_target_overrides(
            Dict("model" => Dict("bogus" => 1)), (;), "fit_rv")
        @test_throws ArgumentError Nereus._apply_target_overrides([1], (;), "fit_rv")
        @test Nereus._apply_target_overrides(nothing, (; a = 1), "x") == (; a = 1)
    end

    @testset "config_overrides on the config route" begin
        cfg = Dict{String,Any}("model" => Dict{String,Any}("max_kplanet" => 1,
                                                          "stability" => "none"),
                               "sampler" => Dict{String,Any}("name" => "pt_emcee"))
        Nereus._apply_config_overrides!(cfg, Dict("model" => Dict("stability" => "amd"),
                                                  "sampler" => Dict("kwargs" => Dict("n_steps" => 7)),
                                                  "seed" => 4), "fit_rm")
        @test cfg["model"]["stability"] == "amd" && cfg["model"]["max_kplanet"] == 1
        @test cfg["sampler"]["name"] == "pt_emcee" && cfg["sampler"]["kwargs"]["n_steps"] == 7
        @test cfg["seed"] == 4
        @test_throws ArgumentError Nereus._apply_config_overrides!(cfg, 3, "fit_rm")
        c2 = Dict{String,Any}("model" => Dict{String,Any}("planet_modes" => ["RVPM_RM"]))
        Nereus._cfg_route_blocks!(c2, "fit_rm"; planet_modes = "RVPM_RM_GD")
        @test c2["model"]["planet_modes"] == ["RVPM_RM_GD"]
        @test_throws ArgumentError Nereus._cfg_route_blocks!(c2, "fit_rm"; noise_models = [1])
    end
end

@testset "typed API gaps: operations on a finished fit" begin
    d1 = _rvdat(50)
    ch = Dict{String,Any}("source" => "RV", "data" => Dict{String,Any}("A" => d1))
    out = mktempdir()
    eng = Dict("engine" => "pt_emcee",
               "options" => Dict("n_temps" => 3, "n_walkers" => 16, "n_steps" => 150,
                                 "n_burnin" => 100, "seed" => 2, "show_progress" => false))
    r = Nereus.fit_rv(ch; planets = 1, engine = eng, output_dir = out,
                      science = false, priors = nothing)
    @test isfile(joinpath(out, "chains.nc"))
    # what a client sends: JSON-shaped, string keys
    req(; kw...) = Dict{String,Any}(
        "channels" => [Dict{String,Any}("source" => "RV",
            "data" => Dict("A" => Dict("t" => d1.t, "rv" => d1.rv, "rv_err" => d1.rv_err)))],
        "planets" => 1, "output_dir" => out,
        "engine" => eng, (String(k) => v for (k, v) in kw)...)
    A = Nereus.FEATURE_ACTIONS

    @test isempty(Nereus.FEATURE_NOT_IMPLEMENTED)
    for k in ("diagnostics.loo", "diagnostics.ppc", "diagnostics.fit_health",
              "diagnostics.label_switching", "evidence", "detection_limits",
              "select_planets", "select_noise", "pre_white", "detect.lc_periodogram")
        @test haskey(A, k)
    end

    loo = A["diagnostics.loo"](req(loo_n_draws = 50))
    @test haskey(loo, "elpd_loo") && loo["n_obs"] == 50 && isfinite(loo["elpd_waic"])
    ppc = A["diagnostics.ppc"](req(ppc_n_draws = 30))
    @test ppc isa AbstractDict && !isempty(ppc)
    fh = A["diagnostics.fit_health"](req())
    @test haskey(fh, "overall") && haskey(fh, "checks")
    ev = A["evidence"](req())
    @test haskey(ev, "evidence") && haskey(ev, "log_z")
    @test haskey(ev["evidence"], "mode_laplace") || haskey(ev["evidence"], "bridge")
    dl = A["detection_limits"](req(detection_limits_n_bins = 6))
    @test dl isa AbstractDict && (haskey(dl, "n_bins") || haskey(dl, "status"))
    # the model is needed to re-evaluate; chains alone are refused with the way out
    @test_throws ErrorException A["diagnostics.loo"](Dict{String,Any}("output_dir" => out))
    @test_throws ErrorException A["evidence"](req(output_dir = joinpath(out, "nowhere")))

    # chains-only operations
    ls = A["diagnostics.label_switching"](Dict{String,Any}("output_dir" => out))
    @test haskey(ls, "components") && haskey(ls, "status")
    @test_throws ErrorException A["select_planets"](Dict{String,Any}("output_dir" => out))
    @test_throws ErrorException A["select_noise"](Dict{String,Any}("output_dir" => out))
end

@testset "typed API gaps: select_* on trans-dim chains" begin
    using MCMCChains
    n = 400
    rng = MersenneTwister(1)
    np = rand(rng, [1.0, 1.0, 1.0, 2.0], n)
    na = Float64.(rand(rng, n) .< 0.8)
    ch = MCMCChains.Chains(reshape(hcat(np, na, randn(rng, n)), n, 3, 1),
                           [:n_planets, :noise_active_ErrorScale, :x])
    path = joinpath(mktempdir(), "chains.nc")
    # save_chains wants a Params; the readers only need the cube, so write it via a fit's layout
    d1 = _rvdat(20)
    tgt = Nereus._target_from([Dict{String,Any}("source" => "RV",
              "data" => Dict{String,Any}("A" => d1))],
              Nereus._planet_spec(1; block = Nereus.default_rv_planet(d1.t, d1.rv)))
    Nereus.save_chains(path, ch, tgt.params; data = tgt.data)
    A = Nereus.FEATURE_ACTIONS
    sp = A["select_planets"](Dict{String,Any}("chains_path" => path))
    @test sp["best"] == 1
    @test sum(r["probability"] for r in sp["table"]) ≈ 1.0
    sn = A["select_noise"](Dict{String,Any}("chains_path" => path))
    @test sn["best"] == "ErrorScale"
    @test 0.7 < sn["table"][1]["occupancy"] < 0.9
end

@testset "typed API gaps: light curves and prewhitening" begin
    rng = MersenneTwister(4)
    t = collect(range(0, 60, length = 800))
    f = 1 .+ 0.004 .* sin.(2π .* t ./ 7.3) .+ 0.0005 .* randn(rng, 800)
    A = Nereus.FEATURE_ACTIONS
    pg = A["detect.lc_periodogram"](Dict{String,Any}("t" => t, "flux" => f,
            "flux_err" => fill(5e-4, 800), "period_min" => 1.0, "period_max" => 30.0))
    @test haskey(pg, "peaks") && !isempty(pg["peaks"])
    @test abs(pg["peaks"][1]["period"] - 7.3) < 0.2

    tt = sort(rand(rng, 120) .* 200)
    y = 8 .* sin.(2π .* tt ./ 11.0) .+ 4 .* sin.(2π .* tt ./ 37.0 .+ 1) .+ 0.5 .* randn(rng, 120)
    pw = A["pre_white"](Dict{String,Any}("t" => tt, "y" => y, "y_err" => fill(0.5, 120),
                                         "max_signals" => 4, "fap" => 0.01))
    @test pw["n_signals"] >= 2
    ps = sort([s["period"] for s in pw["signals"][1:2]])
    @test abs(ps[1] - 11.0) < 0.3 && abs(ps[2] - 37.0) < 1.5
    @test length(pw["residuals"]) == 120
    @test pw["rms"] < 1.0
end

@testset "typed API gaps: the priors the Python client can express" begin
    mj = Nereus._as_prior(Dict("type" => "modjeffreys", "knee" => 1.0, "upper" => 100.0))
    @test mj isa PriorSpec && bounds(mj) == (0.0, 100.0)
    be = Nereus._as_prior(Dict("type" => "beta", "a" => 2.0, "b" => 5.0, "lo" => 0.0, "hi" => 5.0))
    @test be isa PriorSpec && bounds(be) == (0.0, 5.0)
    bn = Nereus._as_prior(Dict("type" => "normal", "mu" => 0.0, "sigma" => 5.0,
                               "lo" => -20.0, "hi" => 20.0))
    @test bn isa PriorSpec && bounds(bn) == (-20.0, 20.0)
    @test bounds(Nereus._as_prior(Dict("type" => "normal", "mu" => 0.0, "sigma" => 1.0,
                                       "hi" => 2.0)))[2] == 2.0
end

@testset "typed API gaps: JSON-keyed ttv slots and trend_order override" begin
    d1 = _rvdat(30)
    ph = Dict{String,Any}("source" => "PM", "data" => Dict{String,Any}(
        "T" => (t = d1.t, flux = ones(30), flux_err = fill(1e-3, 30))))
    rv = Dict{String,Any}("source" => "RV", "data" => Dict{String,Any}("A" => d1))
    pl = Nereus._planet_spec(1; block = Nereus.default_rv_planet(d1.t, d1.rv))
    # a JSON client names planets by string
    t = Nereus._target_from([rv, ph], pl; planet_modes = ["RVPM_TTV"],
                            ttv_epochs = Dict("1" => [0, 1, 2]))
    @test count(startswith("ttv_"), t.params.layout.unfrozen_names) == 3
    t2 = Nereus._target_from([rv, ph], pl; planet_modes = ["RVPM_TTV"],
                             ttv_n_transits = Dict("1" => 2))
    @test count(startswith("ttv_"), t2.params.layout.unfrozen_names) == 2
    # a channel at its default trend_order yields to the keyword
    rv0 = merge(rv, Dict("trend_order" => 0))
    @test "dvdt" in Nereus._target_from([rv0], pl; trend_order = 1).params.layout.unfrozen_names
    rv2 = merge(rv, Dict("trend_order" => 2))
    @test_throws ArgumentError Nereus._target_from([rv2], pl; trend_order = 1)
end

@testset "a trans-dim engine or a planet range is trans-dim mode" begin
    @test Nereus._planets_transdim(1, nothing, "f") === nothing
    @test Nereus._planets_transdim([], nothing, "f") === nothing
    @test Nereus._planets_transdim([1], nothing, "f") === nothing
    @test Nereus._planets_transdim([1, 2], nothing, "f") === true
    @test Nereus._planets_transdim(0:2, nothing, "f") === true
    @test Nereus._planets_transdim([1, 2], 3, "f") == 3
    @test_throws ArgumentError Nereus._planets_transdim([1, 2], false, "f")
    @test Nereus._planets_transdim(2, false, "f") === false

    d1 = _rvdat(40)
    ch = Dict{String,Any}("source" => "RV", "data" => Dict{String,Any}("A" => d1))
    tiny(name) = Dict("engine" => name, "options" => Dict("n_temps" => 3, "n_walkers" => 24,
        "n_steps" => 50, "n_burnin" => 30, "seed" => 2, "show_progress" => false))
    # the engine alone
    r = Nereus.fit_rv(ch; planets = 2, engine = tiny("transdim_pt_emcee"), science = false)
    @test r.summary["status"] == "ok"
    # a planet range alone
    r = Nereus.fit_rv(ch; planets = [1, 2], science = false,
                      engine = Dict("options" => Dict("n_temps" => 3, "n_walkers" => 24,
                                    "n_steps" => 50, "n_burnin" => 30, "show_progress" => false)))
    @test r.summary["status"] == "ok"
    # contradicted
    @test_throws ArgumentError Nereus.fit_rv(ch; planets = [1, 2], transdim = false, science = false)
    @test_throws ArgumentError Nereus.fit_rv(ch; planets = 2, transdim = false, science = false,
                                             engine = tiny("transdim_pt_emcee"))

    # a job config
    cfg(; kw...) = Dict{String,Any}("version" => "1.0", "output_dir" => mktempdir(),
        "data" => Nereus._data_block(; rv = Dict("A" => d1)),
        "model" => Dict{String,Any}("max_kplanet" => 2, "planet_modes" => ["RV_ONLY", "RV_ONLY"]),
        "priors" => Dict{String,Any}(),
        "sampler" => Dict{String,Any}("name" => "transdim_pt_emcee", "kwargs" => Dict(
            "n_temps" => 3, "n_walkers" => 24, "n_steps" => 50, "n_burnin" => 30,
            "show_progress" => false)), (String(k) => v for (k, v) in kw)...)
    s = Nereus.run_job(cfg())
    @test s["status"] == "ok"
    @test_throws Exception Nereus.run_job(cfg(transdim = false))
end

@testset "operations on a fit that ran as a job config" begin
    d1 = _rvdat(40)
    out = mktempdir()
    r = Nereus.fit_binary(; rv = Dict("A" => d1), output_dir = out, science = false,
        engine = Dict("engine" => "pt_emcee", "options" => Dict("n_temps" => 3,
            "n_walkers" => 24, "n_steps" => 60, "n_burnin" => 40, "seed" => 2,
            "show_progress" => false)))
    @test isfile(joinpath(out, "job_config.json"))
    A = Nereus.FEATURE_ACTIONS
    payload = Dict{String,Any}("output_dir" => out)
    @test haskey(A["diagnostics.loo"](payload), "elpd_waic")
    @test haskey(A["diagnostics.fit_health"](payload), "overall")
    @test haskey(A["evidence"](payload), "evidence")
    @test A["diagnostics.ppc"](payload) isa AbstractDict
end
