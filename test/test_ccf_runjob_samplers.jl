# Every Nereus engine runs a CCF line-component target through run_job
# (briefly -- the plumbing, not convergence), and the log-posterior an engine
# stores is the target's, CCF term included. The trans-dimensional engines get
# RVs as well and a static-only CCF, so that toggling the planet is legal.
using Test
using Nereus
using NCDatasets
using Random
@isdefined(cs_ccfs) || include(joinpath(@__DIR__, "fixtures", "ccf_synth.jl"))

const CR_RUNS = [
    ("pt_emcee",       Dict("n_temps" => 3, "n_walkers" => 60, "n_steps" => 30, "n_burnin" => 10)),
    ("pt",             Dict("n_rounds" => 3, "n_chains" => 4)),
    ("pt_whitening",   Dict("n_temps" => 2, "n_walkers" => 60, "n_steps" => 30, "n_burnin" => 10,
                            "warmup_swaps" => 5)),
    ("ensemble",       Dict("n_walkers" => 60, "n_steps" => 30, "n_burnin" => 10)),
    ("ess",            Dict("n_samples" => 30, "n_burnin" => 10)),
    ("nuts",           Dict("n_samples" => 10, "n_warmup" => 10, "n_chains" => 1)),
    ("pt_hmc",         Dict("n_temps" => 2, "n_sweeps" => 6, "n_warmup" => 6,
                            "adapt_ladder" => false, "warm_start" => false)),
    ("nested",         Dict("n_live" => 60, "dlogz" => 50.0)),
    ("nested_ins",     Dict("n_live" => 60, "dlogz" => 50.0, "max_iter" => 400)),
    ("nested_dynamic", Dict("n_live_init" => 60, "n_live_batch" => 30, "dlogz_init" => 50.0,
                            "max_iter" => 400)),
    ("pa",             Dict("n_replicas" => 60, "n_mcmc" => 2, "max_steps" => 20)),
    ("smc",            Dict("n_replicas" => 60, "n_mcmc" => 2, "max_steps" => 20)),
]
# `map` is a point estimator that run_job refuses by design (sample_map is
# exercised on CCF targets in test_ccf_components.jl); `ofti` samples
# relative astrometry only.
const CR_TD_RUNS = [
    ("transdim_pt_emcee", Dict("n_temps" => 2, "n_walkers" => 60, "n_steps" => 30,
                               "n_burnin" => 10)),
    ("rjmcmc",         Dict("n_samples" => 60, "n_warmup" => 20)),
    ("moms",           Dict("n_samples" => 60, "n_warmup" => 20)),
    ("daedalus",       Dict("n_live" => 40, "dlogz" => 50.0, "max_iter" => 300)),
]

const _CR_OUT = Dict{String, Any}("plots" => String[], "ppc" => false,
                                  "detection_limits" => false)
_cr_ccf(comps) = cs_ccf_block(; comps, synth = (n_epochs = 5, scale_sd = 0.0, cont_sd = 0.0),
                              extra = Dict{String, Any}("jitter" => "shared",
                                                        "continuum" => "shared",
                                                        "depth_scale" => "none"))

function cr_config(out, name, kw; transdim = false)
    if !transdim
        return Dict{String, Any}("output_dir" => out, "seed" => 5,
            "data" => Dict{String, Any}("ccf" => [_cr_ccf(CS_BLEND)]),
            "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"]),
            "priors" => cs_prior_cfg(),
            "sampler" => Dict{String, Any}("name" => name, "kwargs" => kw),
            "output" => _CR_OUT)
    end
    rng = MersenneTwister(2)
    trv = sort!(CS_TC .+ 50 .* rand(rng, 20))
    rv = 30 .* cs_kep.(trv) .+ 2 .* randn(rng, 20)
    static = [("A", 0, 10.0, 0.0, 12.0, 0.15), ("C", 0, 30.0, 0.0, 8.0, 0.05)]
    return Dict{String, Any}("output_dir" => out, "seed" => 5,
        "data" => Dict{String, Any}("ccf" => [_cr_ccf(static)],
            "rv" => Dict("values" => Dict("bjd" => trv, "rv" => rv, "rv_err" => fill(2.0, 20),
                                          "instrument" => fill("SIM", 20)))),
        "model" => Dict{String, Any}("max_kplanet" => 1, "planet_modes" => ["RV_ONLY"]),
        "priors" => Dict{String, Any}("P_k1" => Dict("type" => "loguniform", "args" => [3.0, 6.0])),
        "transdim" => Dict{String, Any}("max_kplanet" => 1, "planets" => true),
        "sampler" => Dict{String, Any}("name" => name, "kwargs" => kw),
        "output" => _CR_OUT)
end

function cr_check(cfg, name, out)
    s = run_job(cfg)
    @test s["status"] == "ok"
    @test isfile(joinpath(out, "chains.nc"))
    NCDataset(joinpath(out, "chains.nc")) do ds
        @test haskey(ds, "ccf_gamma_SIM_A")
        g = vec(Array(ds["ccf_gamma_SIM_A"]))
        @test !isempty(g) && all(isfinite, g)
        if haskey(ds, "lp") && name in ("pt_emcee", "pt_whitening", "nuts")
            data, rvn, pmn, _ = Nereus._build_data(cfg["data"])
            p, tgt, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()), rvn, pmn)
            x = [Float64(vec(Array(ds[nm]))[end]) for nm in p.layout.unfrozen_names]
            # the stored log-posterior includes the CCF term (it is thousands in ln p)
            @test vec(Array(ds["lp"]))[end] ≈ Nereus.logdensity_bounded(tgt, x) rtol = 1e-8
        end
    end
end

@testset "CCF target through run_job, every engine" begin
    @testset "engine $name" for (name, kw) in CR_RUNS
        out = mktempdir()
        cr_check(cr_config(out, name, kw), name, out)
    end
    @testset "trans-dimensional engine $name" for (name, kw) in CR_TD_RUNS
        out = mktempdir()
        cr_check(cr_config(out, name, kw; transdim = true), name, out)
    end
end
