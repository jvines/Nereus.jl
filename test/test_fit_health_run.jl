# Run-summary fit-health checks (`assess_run_summary`): the signs of a cold
# chain trapped in a wrong mode that the chain-only checks of `assess_fit`
# cannot see, because a population collapsed into one wrong basin is converged,
# well mixed and off every prior bound.
#
# Reproduces HIP 63762 (nereus I, 2026-11-24): pt_emcee, RV + Hipparcos/Gaia
# astrometry, two Keplerians. The trapped runs (k2_ast_ms, k2_ast_v4, k2_ast_v5)
# showed: rungs near β = 1 at within-move acceptance 0.002 with the hot half of
# the ladder frozen (1e-5) and swapping at 6e-6; a bridge-sampling evidence error
# of ±4.3–23 nats; mode-Laplace 30–160 nats above bridge; HIRES jitter 109–279
# m/s on 1.1 m/s errors (1.4 m/s in the right mode); lnZ ~70 nats below a nested
# simpler model. k2_ast_ms passed every check of `assess_fit` with overall "ok".
# The healthy runs of the same programme (k1_v3, k1_ast_v5) are the controls.

using Nereus, Random, Statistics, Test, MCMCChains
using Random: MersenneTwister

_rcheck(cs, name) = cs[findfirst(c -> c.name === name, cs)]

# k2_ast_ms, as its summary.json reported it (sampler, evidence, ppc), with the
# per-instrument PPC noise block the check reads.
function _trapped_summary()
    Dict{String, Any}(
        "sampler" => Dict{String, Any}(
            "acceptance_within" => [0.0641, 0.0602, 0.0540, 0.0366, 0.0047, 0.0043,
                                    0.0034, 0.0024, 0.0029, 0.0019, 0.0023, 1.5e-4,
                                    1.2e-5, 1.4e-5, 1.4e-5, 1.5e-5],
            "acceptance_swap"   => [0.44, 0.41, 0.31, 0.069, 0.25, 0.24, 0.26, 0.24,
                                    0.22, 0.18, 1.1e-4, 1.1e-4, 1.0e-4, 9.8e-5, 5.8e-6]),
        "evidence" => Dict{String, Any}(
            "bridge" => Dict{String, Any}("log_z" => -189.85, "se" => 4.35,
                                          "overlap" => 0.82),
            "mode_laplace" => -176.52,
            "mode_laplace_check" => Dict{String, Any}("minus_bridge" => 13.33),
            "reported" => "bridge"),
        "ppc" => Dict{String, Any}(
            "rv_err_formal_median_per_inst" => Dict("ELODIE_OHP" => 15.0,
                                                    "HIRES_POST" => 1.10),
            "rv_jitter_median_per_inst"     => Dict("ELODIE_OHP" => 2.4,
                                                    "HIRES_POST" => 109.0),
            "rv_resid_rms_per_inst"         => Dict("ELODIE_OHP" => 46.0,
                                                    "HIRES_POST" => 121.0),
            "rv_red_chi2_formal_per_inst"   => Dict("ELODIE_OHP" => 11.0,
                                                    "HIRES_POST" => 1.7e4)),
        "fit_health" => Dict{String, Any}(
            "overall" => "ok",
            "checks"  => Dict{String, Any}(
                "convergence" => Dict("status" => "ok", "message" => "converged"))))
end

# k1_ast_v5, the right mode of the same star.
function _healthy_summary()
    Dict{String, Any}(
        "sampler" => Dict{String, Any}(
            "acceptance_within" => [0.030, 0.019, 0.009, 0.005, 0.003, 0.003, 0.004,
                                    0.006, 0.011, 0.023, 0.076, 0.134, 0.151],
            "acceptance_swap"   => [0.31, 0.45, 0.52, 0.58, 0.61, 0.66, 0.70, 0.72,
                                    0.75, 0.80, 0.84, 0.87]),
        "evidence" => Dict{String, Any}(
            "bridge" => Dict{String, Any}("log_z" => -142.81, "se" => 0.32,
                                          "overlap" => 0.999),
            "mode_laplace" => -134.51,
            "mode_laplace_check" => Dict{String, Any}("minus_bridge" => 8.31),
            "reported" => "bridge"),
        "ppc" => Dict{String, Any}(
            "rv_err_formal_median_per_inst" => Dict("ELODIE_OHP" => 15.0,
                                                    "HIRES_POST" => 1.10),
            "rv_jitter_median_per_inst"     => Dict("ELODIE_OHP" => 2.5,
                                                    "HIRES_POST" => 1.37),
            "rv_resid_rms_per_inst"         => Dict("ELODIE_OHP" => 21.0,
                                                    "HIRES_POST" => 2.0),
            "rv_red_chi2_formal_per_inst"   => Dict("ELODIE_OHP" => 2.0,
                                                    "HIRES_POST" => 3.3)),
        "fit_health" => Dict{String, Any}(
            "overall" => "ok",
            "checks"  => Dict{String, Any}(
                "convergence" => Dict("status" => "ok", "message" => "converged"))))
end

@testset "assess_run_summary — trapped-mode signs" begin

    @testset "healthy run: every run-summary check ok" begin
        cs = assess_run_summary(_healthy_summary())
        @test all(c -> c.status === :ok, cs)
        @test Set(c.name for c in cs) ==
              Set([:cold_acceptance, :tempering_ladder, :evidence_precision, :rv_jitter])
    end

    @testset "HIP 63762 k2_ast_ms: flagged, warnings only" begin
        cs = assess_run_summary(_trapped_summary())
        @test all(c -> c.status in (:ok, :warn), cs)       # never a hard fail
        @test _rcheck(cs, :tempering_ladder).status === :warn
        @test occursin("swap", _rcheck(cs, :tempering_ladder).message)
        @test occursin("frozen", _rcheck(cs, :tempering_ladder).message)
        @test _rcheck(cs, :evidence_precision).status === :warn
        @test occursin("bridge", _rcheck(cs, :evidence_precision).message)
        j = _rcheck(cs, :rv_jitter)
        @test j.status === :warn
        @test occursin("HIRES_POST", j.message)
        @test !occursin("ELODIE_OHP", j.message)
        @test any(p -> p.first == "HIRES_POST", j.details)
        # The cold rung itself moved at 6%: not the pathological-acceptance case.
        @test _rcheck(cs, :cold_acceptance).status === :ok
    end

    @testset "pathologically low cold-chain acceptance" begin
        s = _healthy_summary()
        s["sampler"]["acceptance_within"][1] = 0.002
        c = _rcheck(assess_run_summary(s), :cold_acceptance)
        @test c.status === :warn
        @test occursin("0.002", c.message)
    end

    @testset "evidence: mode-Laplace far from bridge, estimators disagreeing" begin
        s = _healthy_summary()
        s["evidence"]["mode_laplace"] = -112.0          # 30.8 above bridge
        s["evidence"]["mode_laplace_check"]["minus_bridge"] = 30.81
        c = _rcheck(assess_run_summary(s), :evidence_precision)
        @test c.status === :warn
        @test occursin("mode-Laplace", c.message)

        s = _healthy_summary()
        s["evidence"]["ti_plus"] = Dict("log_z" => -150.0, "se" => 0.2)
        c = _rcheck(assess_run_summary(s), :evidence_precision)
        @test c.status === :warn
        @test occursin("ti_plus", c.message)
        # Agreeing within the tolerance is fine.
        s["evidence"]["ti_plus"] = Dict("log_z" => -143.5, "se" => 0.2)
        @test _rcheck(assess_run_summary(s), :evidence_precision).status === :ok
        # A bare-number bridge (before the bootstrap) is read, not an error.
        s = _healthy_summary()
        s["evidence"]["bridge"] = -142.81
        @test _rcheck(assess_run_summary(s), :evidence_precision).status === :ok
    end

    @testset "jitter that dwarfs the residuals: an instrument switched off" begin
        s = _healthy_summary()
        s["ppc"]["rv_jitter_median_per_inst"]["HIRES_POST"] = 300.0
        s["ppc"]["rv_resid_rms_per_inst"]["HIRES_POST"] = 40.0
        s["ppc"]["rv_red_chi2_formal_per_inst"]["HIRES_POST"] = 1300.0
        c = _rcheck(assess_run_summary(s), :rv_jitter)
        @test c.status === :warn
        @test occursin("ignored", c.message)
    end

    @testset "missing blocks are not assessed, not errors" begin
        cs = assess_run_summary(Dict{String, Any}())
        @test all(c -> c.status === :ok, cs)
        @test all(c -> occursin("not assessed", c.message), cs)
    end

    @testset "merged into summary.json fit_health as warnings" begin
        s = _trapped_summary()
        Nereus._fit_health_run_checks!(s)
        fh = s["fit_health"]
        @test fh["overall"] == "warn"
        @test fh["suspect_wrong_mode"] === true
        @test fh["checks"]["rv_jitter"]["status"] == "warn"
        @test fh["checks"]["evidence_precision"]["status"] == "warn"
        @test haskey(fh["checks"], "convergence")          # existing checks kept
        @test "rv_jitter" in fh["suspect_wrong_mode_checks"]

        s = _healthy_summary()
        Nereus._fit_health_run_checks!(s)
        @test s["fit_health"]["overall"] == "ok"
        @test s["fit_health"]["suspect_wrong_mode"] === false

        # A fail from the chain checks is not downgraded.
        s = _trapped_summary()
        s["fit_health"]["overall"] = "fail"
        Nereus._fit_health_run_checks!(s)
        @test s["fit_health"]["overall"] == "fail"

        # fit_health switched off (or skipped after an error): nothing is added.
        s = _trapped_summary(); delete!(s, "fit_health")
        Nereus._fit_health_run_checks!(s)
        @test !haskey(s, "fit_health")
        s = _trapped_summary()
        s["fit_health"] = Dict{String, Any}("status" => "skipped", "error" => "x")
        Nereus._fit_health_run_checks!(s)
        @test s["fit_health"] == Dict{String, Any}("status" => "skipped", "error" => "x")
    end
end

# The PPC records the per-instrument noise block the jitter check reads.
@testset "posterior_predictive_check — per-instrument jitter vs formal errors" begin
    rng = MersenneTwister(11)
    n_a, n_b = 30, 12
    n = n_a + n_b
    t = sort!(200.0 .* rand(rng, n))
    inst = vcat(ones(Int, n_a), fill(2, n_b))
    err = vcat(fill(5.0, n_a), fill(1.0, n_b))
    data0 = Data(; t_rv = t, rv = zeros(n), rv_err = err, rv_inst = inst)
    params = Params(; parametrization = ParametrizationConfig(time = :Mo),
        max_kplanet = 1, planet_modes = [RV_ONLY],
        instruments = InstrumentConfig(rv = ["A", "B"]), data = data0, M_s = 1.0)
    truth = Dict(:P_k1 => 30.0, :K_k1 => 50.0, :sesinw_k1 => 0.2, :secosw_k1 => 0.2,
                 :Mo_k1 => 1.0, :gamma_A => 0.0, :gamma_B => 0.0,
                 :sigma_A => 1.0, :sigma_B => 1.0)
    th = Theta{Float64}(params)
    for (k, v) in truth; set_param!(th, String(k), v); end
    clean, _ = rv_predictions(th, data0)
    data = Data(; t_rv = t, rv = clean .+ err .* randn(rng, n), rv_err = err,
                  rv_inst = inst)

    # The "trapped" posterior: K at 40% of truth, with B's jitter blown up to
    # absorb what the wrong orbit leaves in B.
    trapped = merge(truth, Dict(:K_k1 => 20.0, :sigma_B => 50.0))
    names_ = Symbol.(params.layout.unfrozen_names)
    arr = zeros(200, length(names_), 1)
    for (j, nm) in enumerate(names_)
        arr[:, j, 1] .= get(trapped, nm, 0.0) .+ 1e-7 .* randn(rng, 200)
    end
    ppc = posterior_predictive_check(Chains(arr, names_), params, data;
                                      n_draws = 200, rng = MersenneTwister(0))
    s = ppc.summary
    @test s["rv_err_formal_median_per_inst"]["A"] ≈ 5.0
    @test s["rv_err_formal_median_per_inst"]["B"] ≈ 1.0
    @test isapprox(s["rv_jitter_median_per_inst"]["B"], 50.0; rtol = 1e-4)
    @test isapprox(s["rv_jitter_median_per_inst"]["A"], 1.0; rtol = 1e-4)
    rB = ppc.rv_resid_med[inst .== 2]
    @test s["rv_resid_rms_per_inst"]["B"] ≈ sqrt(mean(abs2, rB))
    @test s["rv_red_chi2_formal_per_inst"]["B"] ≈ sum(abs2, rB) / (n_b - 2)
    @test s["rv_red_chi2_formal_per_inst"]["B"] > 30
    # And the check reads it: B's 50 m/s on 1 m/s errors is flagged, A is not.
    c = _rcheck(assess_run_summary(Dict{String, Any}("ppc" => s)), :rv_jitter)
    @test c.status === :warn
    @test occursin("B", c.message)
    @test !any(p -> p.first == "A", c.details)
end
