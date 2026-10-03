# An obliquity fit is a run_job config like any other: RM nights and
# line-profile stacks are data blocks, the model is `model.obliquity`, the
# priors are the usual `priors` block. This pins the schema checks, the
# end-to-end run with checkpoint / resume, and that EVERY Nereus sampler runs
# the target (briefly -- this is about the plumbing, not convergence).
using Test
using Nereus
using JSON3, NCDatasets
using Nereus: Theta
using Random, Statistics

@isdefined(OS_P) || include(joinpath(@__DIR__, "fixtures", "obliquity_synthetic.jl"))

const RJ_DIR = os_write_data(mktempdir())

function rj_config(out; fit = "joint", sampler = "pt_emcee", kwargs = Dict(),
                   tags = OS_TAGS, extra_model = Dict(), transdim = nothing)
    rm = [Dict("tag" => t, "file" => joinpath(RJ_DIR, "rm_$t.dat"),
               "sigma0" => OS_SIG0[t], "instrument" => "SIM", "pipeline" => "test")
          for t in tags]
    tomo = [Dict("tag" => t, "profiles" => joinpath(RJ_DIR, "tomo_$(t)_prof.dat"),
                 "vgrid" => joinpath(RJ_DIR, "tomo_$(t)_vgrid.dat"),
                 "times" => joinpath(RJ_DIR, "tomo_$(t)_t.dat"),
                 "berv" => joinpath(RJ_DIR, "tomo_$(t)_berv.dat"),
                 "vsys" => OS_VSYS, "grid" => [-42, 42, 25]) for t in OS_TAGS]
    obl = Dict{String,Any}("fit" => fit, "P" => OS_P, "Tc" => OS_TC0,
        "b" => collect(OS_B), "a_Rs" => collect(OS_ARS), "rr" => OS_RR,
        "vsini" => collect(OS_VSINI_MS), "K" => collect(OS_K),
        "limb_darkening" => [OS_U1, OS_U2], "occultation" => "point",
        "beta_p_floor" => 2000.0, "t14_hours" => 2 * OS_T14H)
    merge!(obl, extra_model)
    kw = Dict{String,Any}(kwargs)
    # quiet, where the sampler has the switch
    fn = Nereus.ENGINES[sampler]
    any(m -> :show_progress in Base.kwarg_decl(m), methods(fn)) && (kw["show_progress"] = false)
    cfg = Dict{String,Any}("output_dir" => out, "seed" => 5,
        "data" => Dict{String,Any}("rm_nights" => rm, "tomography" => tomo),
        "model" => Dict{String,Any}("obliquity" => obl),
        "sampler" => Dict{String,Any}("name" => sampler, "kwargs" => kw),
        "output" => Dict{String,Any}("plots" => String[]))
    transdim === nothing || (cfg["transdim"] = transdim)
    return cfg
end

validation_errors(cfg) = try
    Nereus._validate_config(cfg); ""
catch e
    sprint(showerror, e)
end

@testset "obliquity fits through run_job" begin

    @testset "schema: clear errors before anything runs" begin
        out = mktempdir()
        @test validation_errors(rj_config(out)) == ""
        c = rj_config(out); c["model"]["obliquity"]["fit"] = "both"
        @test occursin("`model.obliquity.fit` = `both` unknown", validation_errors(c))
        c = rj_config(out); delete!(c["data"]["rm_nights"][2], "sigma0")
        @test occursin("missing `sigma0`", validation_errors(c))
        c = rj_config(out); delete!(c["model"]["obliquity"], "t14_hours")
        @test occursin("t14_hours", validation_errors(c))
        c = rj_config(out); delete!(c["data"]["tomography"][1], "berv")
        @test occursin("barycentric", validation_errors(c))
        c = rj_config(out); c["model"]["obliquity"]["b"] = [0.75, -0.01]
        @test occursin("model.obliquity.b", validation_errors(c))
        c = rj_config(out); c["model"]["max_kplanet"] = 1
        @test occursin("cannot be combined with `model.obliquity`", validation_errors(c))
        c = rj_config(out); delete!(c["model"], "obliquity"); c["model"]["max_kplanet"] = 1
        @test occursin("add a `model.obliquity` block", validation_errors(c))
        c = rj_config(out; fit = "shadow"); delete!(c["data"], "tomography")
        @test occursin("needs line profiles", validation_errors(c))
        c = rj_config(out; extra_model = Dict("noise_menu" => true))
        @test occursin("add a `transdim` block", validation_errors(c))
        c = rj_config(out); c["data"]["rm_nights"][1]["file"] = "/nonexistent.dat"
        @test occursin("does not exist", validation_errors(c))
        c = rj_config(out); c["model"]["obliquity"]["occultation"] = "ring"
        @test occursin("occultation", validation_errors(c))
        # No light curve and no limb darkening would mean a uniform disc,
        # silently: refused, with the fix in the message.
        c = rj_config(out); delete!(c["model"]["obliquity"], "limb_darkening")
        @test occursin("limb_darkening", validation_errors(c)) &&
              occursin("uniform disc", validation_errors(c))
        c["model"]["obliquity"]["limb_darkening"] = [0, 0]   # uniform, said explicitly
        @test validation_errors(c) == ""
    end

    @testset "the fit kind selects the data blocks" begin
        cfg = rj_config(mktempdir(); fit = "velocities")
        data, rvn, pmn, _ = Nereus._build_data(cfg["data"]; obliquity = cfg["model"]["obliquity"])
        @test rvn == OS_TAGS && isempty(data.tomo)
        p, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()), rvn, pmn)
        @test p.config.planet_modes == [RVPM_RM_A]
        cfg = rj_config(mktempdir(); fit = "shadow")
        data, rvn, pmn, _ = Nereus._build_data(cfg["data"]; obliquity = cfg["model"]["obliquity"])
        @test isempty(rvn) && length(data.tomo) == 3 && isempty(data.rv)
        p, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()), rvn, pmn)
        @test p.config.planet_modes == [PM_DT]
        # the maps run_job builds are the bespoke drivers' maps
        _, tomo_ref = os_load(RJ_DIR)
        for (a, b) in zip(data.tomo, tomo_ref)
            @test a.R == b.R && a.Tc == b.Tc && a.grid == b.grid
        end
        # priors override anything by name
        cfg = rj_config(mktempdir())
        cfg["priors"] = Dict("tomo_sigma_line_A" => Dict("type" => "UniformPrior", "args" => [3, 9]),
                             "lambda_k1" => Dict("type" => "WrappedUniformPrior", "args" => [0, 2π]))
        data, rvn, pmn, _ = Nereus._build_data(cfg["data"]; obliquity = cfg["model"]["obliquity"])
        p, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()), rvn, pmn)
        @test Nereus.bounds(p.config.priors["tomo_sigma_line_A"]) == (3.0, 9.0)
        @test is_wrapped(p.config.priors["lambda_k1"]) && p.config.priors["lambda_k1"].lo == 0.0
    end

    @testset "end to end: chains.nc, summary.json, checkpoint and resume" begin
        out = mktempdir()
        cfg = rj_config(out; kwargs = Dict("n_temps" => 1, "n_walkers" => 80,
                                           "n_steps" => 40, "n_burnin" => 20, "thin" => 2))
        s = run_job(cfg)
        @test s["status"] == "ok"
        @test isfile(joinpath(out, "chains.nc")) && isfile(joinpath(out, "summary.json"))
        @test isfile(joinpath(out, "pt_emcee_state.jls"))
        js = JSON3.read(read(joinpath(out, "summary.json"), String); allow_inf = true)
        @test js["obliquity"]["fit"] == "joint"
        @test length(js["obliquity"]["rm_nights"]) == 3
        @test js["obliquity"]["options"]["occultation"] == "point"
        @test !haskey(js, "ppc")                      # RV-planet diagnostics off
        n1 = NCDataset(joinpath(out, "chains.nc")) do ds
            @test haskey(ds, "lambda_k1") && haskey(ds, "tomo_jit_B")
            size(ds["lambda_k1"], 1)
        end
        cfg["sampler"]["kwargs"]["n_steps"] = 60
        cfg["sampler"]["kwargs"]["resume"] = true
        @test run_job(cfg)["status"] == "ok"
        n2 = NCDataset(ds -> size(ds["lambda_k1"], 1), joinpath(out, "chains.nc"))
        @test n2 > n1
    end

    # Every sampler, briefly, through run_job. `transdim` samplers select the
    # noise per night from the obliquity menu.
    td = Dict("max_kplanet" => 1, "planets" => false, "noise" => true)
    runs = [
        ("pt_emcee",       Dict("n_temps" => 3, "n_walkers" => 80, "n_steps" => 30, "n_burnin" => 10), nothing),
        ("pt",             Dict("n_rounds" => 3, "n_chains" => 4), nothing),
        ("pt_whitening",   Dict("n_temps" => 2, "n_walkers" => 80, "n_steps" => 30, "n_burnin" => 10,
                                "warmup_swaps" => 5), nothing),
        ("ensemble",       Dict("n_walkers" => 80, "n_steps" => 30, "n_burnin" => 10), nothing),
        ("ess",            Dict("n_samples" => 30, "n_burnin" => 10), nothing),
        ("nuts",           Dict("n_samples" => 10, "n_warmup" => 10, "n_chains" => 1), nothing),
        ("pt_hmc",         Dict("n_temps" => 2, "n_sweeps" => 6, "n_warmup" => 6,
                                "adapt_ladder" => false, "warm_start" => false), nothing),
        ("nested",         Dict("n_live" => 60, "dlogz" => 50.0), nothing),
        ("nested_ins",     Dict("n_live" => 60, "dlogz" => 50.0, "max_iter" => 400), nothing),
        ("nested_dynamic", Dict("n_live_init" => 60, "n_live_batch" => 30, "dlogz_init" => 50.0,
                                "max_iter" => 400), nothing),
        ("pa",             Dict("n_replicas" => 60, "n_mcmc" => 2, "max_steps" => 20), nothing),
        ("smc",            Dict("n_replicas" => 60, "n_mcmc" => 2, "max_steps" => 20), nothing),
        ("transdim_pt_emcee", Dict("n_temps" => 2, "n_walkers" => 80, "n_steps" => 30,
                                   "n_burnin" => 10), td),
        ("rjmcmc",         Dict("n_samples" => 60, "n_warmup" => 20), td),
        ("moms",           Dict("n_samples" => 60, "n_warmup" => 20), td),
        ("daedalus",       Dict("n_live" => 40, "dlogz" => 50.0, "max_iter" => 300), td),
    ]
    # ESS centres its Gaussian on `init`; give it the truth.
    function truth_init()
        cfg = rj_config(mktempdir())
        data, rvn, pmn, _ = Nereus._build_data(cfg["data"]; obliquity = cfg["model"]["obliquity"])
        p, _, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()), rvn, pmn)
        val = Dict("lambda_k1" => OS_LAM, "v_sin_i_star" => 25_900.0, "b_k1" => OS_B[1],
                   "a_Rs_k1" => OS_ARS[1], "K_k1" => OS_K[1])
        return [get(val, nm, (q = p.layout.unfrozen_priors[i];
                              (startswith(nm, "gamma_") ? (q.lo + q.hi) / 2 :
                               isa(q.dist, Nereus.LogUniform) ? sqrt(q.lo * q.hi) : (q.lo + q.hi) / 2)))
                for (i, nm) in enumerate(p.layout.unfrozen_names)]
    end
    @testset "sampler $name" for (name, kw, tdb) in runs
        out = mktempdir()
        extra = tdb === nothing ? Dict() : Dict("noise_menu" => true)
        name == "ess" && (kw = merge(kw, Dict("init" => truth_init())))
        cfg = rj_config(out; sampler = name, kwargs = kw, transdim = tdb, extra_model = extra)
        s = run_job(cfg)
        @test s["status"] == "ok"
        @test isfile(joinpath(out, "chains.nc"))
        NCDataset(joinpath(out, "chains.nc")) do ds
            λ = vec(Array(ds["lambda_k1"]))
            @test !isempty(λ) && all(isfinite, λ)
            # the sampler's own log-posterior at its last draw is the target's:
            # it saw the maps
            if haskey(ds, "lp")
                data, rvn, pmn, _ = Nereus._build_data(cfg["data"];
                                                       obliquity = cfg["model"]["obliquity"])
                p, tgt, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()), rvn, pmn)
                x = [Float64(vec(Array(ds[nm]))[end]) for nm in p.layout.unfrozen_names
                     if haskey(ds, nm)]
                # pt_emcee, pt_whitening and nuts store the bounded log-posterior;
                # with the maps missing it would be off by thousands of nats.
                if length(x) == length(p.layout.unfrozen_names) &&
                   name in ("pt_emcee", "pt_whitening", "nuts")
                    @test vec(Array(ds["lp"]))[end] ≈ Nereus.logdensity_bounded(tgt, x) rtol = 1e-8
                end
            end
        end
    end

    # Every sampler that assembles its own likelihood (with workspaces) must
    # include the residual maps. They did not: only NereusTarget's logdensity
    # had the map term, so pt_emcee, the nested family, pa/smc, pt_whitening,
    # rjmcmc, ESS and the trans-dim samplers sampled a tomographic target as if
    # it had no maps. Checked through the samplers' own likelihood closures:
    # the map term is the difference between a target with maps and without.
    @testset "every sampler's likelihood includes the maps" begin
        cfg = rj_config(mktempdir(); fit = "shadow")
        data, rvn, pmn, _ = Nereus._build_data(cfg["data"]; obliquity = cfg["model"]["obliquity"])
        p, tgt, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()), rvn, pmn)
        th = Theta{Float64}(p)
        for (i, s) in enumerate(p.layout.unfrozen_idx)
            q = p.layout.unfrozen_priors[i]
            th.values[s] = (q.lo + q.hi) / 2
        end
        Nereus.set_param!(th, "lambda_k1", OS_LAM)
        lmap = Nereus.tomogram_log_likelihood(th, data)
        @test isfinite(lmap) && lmap != 0
        ws = Nereus.PTWorkspace(p, 1, length(p.config.noise_models); n_obs = 0, n_phot = 0)
        @test Nereus._eval_ll(th, data, nothing) ≈ lmap
        @test Nereus._eval_ll(th, data, nothing, ws) ≈ lmap
    end
end
