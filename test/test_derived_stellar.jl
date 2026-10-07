# The star's uncertainties reach the derived table.
#
# A job's star block carries sigma_M_s, sigma_R_s and sigma_T_eff, and
# `science_derived` draws M★, R★ and T_eff per posterior sample from them. But
# run_job never read the σ's -- nor T_eff, the magnitudes or the albedo -- so
# every derived error bar was built on the ASSUMED σ's (10% of M★, 5% of R★,
# 150 K), and T_eq, insolation, TSM and ESM were never computed at all. Found by
# rebuilding an NGTS-33 summary by hand. The fit_* entry points had the same
# hole, behind a read of `config.T_eff`, a field the config does not have.

using Test, Nereus, Random, Statistics
include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))

# Half the 68% interval over the median: the relative 1σ of a derived entry.
_rel(e) = (e["err_lo"] + e["err_hi"]) / 2 / e["value"]

# M_p sin i ∝ M★^(2/3), so with the orbit pinned its relative 1σ is that of
# M★^(2/3) under M★ ~ N(1, σ): ~0.2 for σ = 0.3, ~0.067 for the assumed 10%.
const _SIG_M = 0.3

@testset "star block σ's: read, validated" begin
    st = Nereus._build_star(Dict("M_s" => 1.0, "sigma_M_s" => 0.3,
                                 "sigma_T_eff" => 120))
    @test st.sigma_M_s === 0.3 && st.sigma_T_eff === 120.0
    @test st.sigma_R_s === nothing
    @test_throws ErrorException Nereus._build_star(Dict("sigma_M_s" => -0.1))
    @test_throws ErrorException Nereus._build_star(Dict("sigma_R_s" => NaN))
    kw = Nereus._star_science_kwargs(Nereus._build_star(
        Dict("M_s" => 1, "T_eff" => 5800, "J_mag" => 9.1, "sigma_R_s" => 0.02)))
    @test kw.M_s === 1.0 && kw.T_eff === 5800.0 && kw.J_mag === 9.1
    @test kw.sigma_R_s === 0.02 && kw.sigma_M_s === nothing && kw.Ab === 0.3
end

@testset "run_job: the star block's σ's are in the derived errors" begin
    cfg = _easy_rv_cfg()
    # The orbit pinned, so the stellar σ's dominate the derived spread.
    cfg["priors"] = Dict(
        "P_k1" => Dict("type" => "UniformPrior", "args" => [12.29, 12.31]),
        "K_k1" => Dict("type" => "UniformPrior", "args" => [11.95, 12.05]))
    cfg["sampler"]["kwargs"] = Dict("show_progress" => false, "n_temps" => 2,
                                    "n_walkers" => 32, "n_steps" => 300,
                                    "n_burnin" => 100)
    cfg["star"] = Dict("M_s" => 1.0, "R_s" => 1.0, "T_eff" => 5800.0,
                       "sigma_M_s" => _SIG_M, "sigma_R_s" => 0.1,
                       "sigma_T_eff" => 400.0)
    res = Nereus.run_job(cfg)
    s = res isa AbstractDict ? res : res.summary

    stellar = s["run_info"]["stellar"]
    for (k, σ) in (("M_s", _SIG_M), ("R_s", 0.1), ("T_eff", 400.0))
        @test stellar[k]["sigma"] == σ
        @test stellar[k]["assumed"] == false
    end
    d = s["derived"]["parameters"]
    # ~0.2 from M★ alone; the assumed 10% would give ~0.07
    @test 0.15 < _rel(d["msini_earth_k1"]) < 0.3
    # T_eq needs T_eff, which run_job never passed
    @test haskey(d, "Teq_k1")
    @test _rel(d["Teq_k1"]) > 0.09         # T_eff 6.9%, R★^(1/2) 5%, M★^(-1/6) 5%
end

@testset "fit_rv: the same keywords, the same propagation" begin
    rng = MersenneTwister(5)
    t = sort!(200 .* rand(rng, 40))
    rv = 15 .* sin.(2π .* t ./ 12.3) .+ randn(rng, 40)
    pri = Dict("P_k1" => UniformPrior(12.29, 12.31), "K_k1" => UniformPrior(14.9, 15.1))
    r = fit_rv(Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.0, 40)));
               planets = 1, priors = pri, M_s = 1.0, R_s = 1.0, T_eff = 5800.0,
               sigma_M_s = _SIG_M, sigma_T_eff = 400.0,
               engine = Dict("engine" => "pt_emcee",
                             "options" => Dict("n_temps" => 2, "n_walkers" => 16,
                                               "n_steps" => 200, "n_burnin" => 50)),
               output_dir = mktempdir())
    s = r.summary
    @test s["run_info"]["stellar"]["M_s"]["sigma"] == _SIG_M
    @test s["run_info"]["stellar"]["M_s"]["assumed"] == false
    @test s["run_info"]["stellar"]["R_s"]["assumed"] == true    # not given
    d = s["derived"]["parameters"]
    @test 0.15 < _rel(d["msini_earth_k1"]) < 0.3
    @test haskey(d, "Teq_k1")
    # a bad σ fails before the engine runs, not after
    @test_throws ErrorException fit_rv(Dict("SIM" => (t = t, rv = rv, rv_err = fill(1.0, 40)));
                                       M_s = 1.0, sigma_M_s = -1.0)
end
