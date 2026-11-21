# Shared by test_obliquity_runjob.jl and test_obliquity_runjob_samplers_{1,2,3}.jl:
# the synthetic data written once, the run_job config, and the every-sampler
# loop. The loop was one file of 805 s -- the longest unit of the suite, and so
# the floor under any number of shards -- and is split three ways by
# RJ_SAMPLER_GROUPS, which must cover every entry of RJ_RUNS exactly once.
using Test
using Nereus
using JSON3
# Narrow import: a bare `using NCDatasets` also exports its `bounds`, which from
# Julia 1.12 makes every later bare `bounds` in Main ambiguous with Nereus's
# (src/io.jl narrows its import for the same reason).
using NCDatasets: NCDataset
using Nereus: Theta
using Random, Statistics

@isdefined(OS_P) || include(joinpath(@__DIR__, "obliquity_synthetic.jl"))

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

# Every sampler, briefly, through run_job. `transdim` samplers select the
# noise per night from the obliquity menu.
const RJ_TD = Dict("max_kplanet" => 1, "planets" => false, "noise" => true)
const RJ_RUNS = [
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
                               "n_burnin" => 10), RJ_TD),
    ("rjmcmc",         Dict("n_samples" => 60, "n_warmup" => 20), RJ_TD),
    ("moms",           Dict("n_samples" => 60, "n_warmup" => 20), RJ_TD),
    ("daedalus",       Dict("n_live" => 40, "dlogz" => 50.0, "max_iter" => 300), RJ_TD),
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

# Balanced on the seconds each took through run_job (arm64, 3 threads,
# 2026-11-21): nested 221, pt_hmc 194, ensemble 91, nuts 36, daedalus 29, the
# rest under 16 -- plus ~100 s of compiling the run_job path, which every file
# pays once (pt_emcee's 122 s ran first and carried it).
const RJ_SAMPLER_GROUPS = (
    ["nested", "nested_ins", "nested_dynamic", "daedalus"],
    ["pt_hmc", "nuts", "pt", "pa", "smc", "ess"],
    ["pt_emcee", "ensemble", "pt_whitening", "transdim_pt_emcee", "rjmcmc", "moms"],
)

"""Every sampler in `names`, briefly, through run_job."""
function rj_sampler_testsets(names)
    @testset "sampler $name" for (name, kw, tdb) in filter(r -> r[1] in names, RJ_RUNS)
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
                # with the maps missing it would be off by thousands in ln p.
                if length(x) == length(p.layout.unfrozen_names) &&
                   name in ("pt_emcee", "pt_whitening", "nuts")
                    @test vec(Array(ds["lp"]))[end] ≈ Nereus.logdensity_bounded(tgt, x) rtol = 1e-8
                end
            end
        end
    end
end
