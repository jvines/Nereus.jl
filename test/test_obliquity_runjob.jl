# An obliquity fit is a run_job config like any other: RM nights and
# line-profile stacks are data blocks, the model is `model.obliquity`, the
# priors are the usual `priors` block. This pins the schema checks and the
# end-to-end run with checkpoint / resume; that EVERY Nereus sampler runs the
# target is test_obliquity_runjob_samplers_{1,2,3}.jl.
using Test
@isdefined(rj_config) || include(joinpath(@__DIR__, "fixtures", "obliquity_runjob.jl"))

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
        # Re-pinned: a WrappedUniform is no longer `is_wrapped` (now: a polar
        # angle folded at its poles); it is the circular Uniform, walled at its
        # chart edges and re-cut by the samplers (src/circular.jl).
        @test p.config.priors["lambda_k1"].dist isa Nereus.WrappedUniform &&
              p.config.priors["lambda_k1"].lo == 0.0
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

    # The sampler runs live in test_obliquity_runjob_samplers_{1,2,3}.jl.
    @testset "the three sampler files cover every sampler once" begin
        names = reduce(vcat, collect.(RJ_SAMPLER_GROUPS))
        @test sort(names) == sort(first.(RJ_RUNS))
        @test allunique(names)
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
