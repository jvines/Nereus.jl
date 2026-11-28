# Reporting on the planet count of a trans-dim run, the one convergence verdict
# and the reliability flag of the evidence (items 79-83 of the trans-dim
# reporting round). Nothing here runs a sampler: each piece reads draws, move
# counters or a summary dictionary, and the end-to-end check at the bottom is a
# short fit that only has to produce the blocks.
using Test, Nereus, Random, Statistics, MCMCChains, Logging
using Nereus: transdim_k_prior, transdim_k_report, transdim_occupancy_health,
              transdim_move_counts, _convergence_verdict, _apply_convergence_verdict!,
              _evidence_reliability!, _finalize_run_reporting!, _worst

# (iteration x walker) planet counts of walkers that each hop between counts as
# a Markov chain with the stationary law `p` (Metropolis, uniform proposal).
function _hop_chain(p::Vector{Float64}, n_iter::Int, n_walkers::Int; seed = 1)
    rng = MersenneTwister(seed)
    cum = cumsum(p ./ sum(p))
    draw() = something(findfirst(>=(rand(rng)), cum), length(p)) - 1
    M = Matrix{Int}(undef, n_iter, n_walkers)
    for w in 1:n_walkers
        k = draw()
        for i in 1:n_iter
            kn = rand(rng, 0:(length(p) - 1))        # uniform proposal
            rand(rng) < p[kn + 1] / p[k + 1] && (k = kn)
            M[i, w] = k
        end
    end
    return M
end

# The flat layout of the walker ensembles: WALKER is the fast index.
_flat(M) = vec(permutedims(M))

@testset "trans-dim planet-count report (items 79-83)" begin

    @testset "prior on k: (p/(1-p))^k / k!, per planet mode, normalised" begin
        w3 = [1.0, 1.0, 0.5, 1 / 6]
        @test transdim_k_prior(3) ≈ w3 ./ sum(w3)
        w = transdim_k_prior(3; inclusion_prior = 0.25)       # r = 1/3
        wr = [1.0, 1 / 3, (1 / 3)^2 / 2, (1 / 3)^3 / 6]
        @test w ≈ wr ./ sum(wr)
        @test sum(w) ≈ 1
        @test transdim_k_prior(0) == [1.0]
        # two planet modes (2 + 1 slots): the 1/k_g! factors multiply
        g = transdim_k_prior(3; group_sizes = [2, 1])
        a = [1.0, 1.0, 0.5]; b = [1.0, 1.0]
        c = [a[1]*b[1], a[2]*b[1] + a[1]*b[2], a[3]*b[1] + a[2]*b[2], a[3]*b[2]]
        @test g ≈ c ./ sum(c)
        @test_throws ArgumentError transdim_k_prior(3; group_sizes = [2, 2])
        @test_throws ArgumentError transdim_k_prior(2; inclusion_prior = 1.0)
    end

    @testset "ln B between adjacent k, from occupancy and the prior" begin
        p = [0.05, 0.60, 0.30, 0.05]
        M = _hop_chain(p, 4000, 12)
        rep = transdim_k_report(_flat(M), 3; nchains = 1, n_walkers = 12)
        pr = transdim_k_prior(3)
        @test rep["prior"]["p_k"] ≈ pr
        @test rep["prior"]["inclusion_prior"] == 0.5
        occ = [rep["occupancy"][string(k)] for k in 0:3]
        @test sum(occ) ≈ 1
        for r in rep["ln_bayes_factor_adjacent"]
            k = r["k_lo"]
            @test r["ln_bayes_factor"] ≈ log(occ[k + 2] / occ[k + 1]) - log(pr[k + 2] / pr[k + 1])
            truth = log(p[k + 2] / p[k + 1]) - log(pr[k + 2] / pr[k + 1])
            @test abs(r["ln_bayes_factor"] - truth) < 4 * r["se"] + 0.05
            @test r["estimable"]
            @test r["se"] > 0
        end
        # a non-uniform prior is divided out
        rep2 = transdim_k_report(_flat(M), 3; n_walkers = 12, inclusion_prior = 0.25)
        r2 = rep2["ln_bayes_factor_adjacent"][1]
        pr2 = transdim_k_prior(3; inclusion_prior = 0.25)
        @test r2["ln_bayes_factor"] ≈ log(occ[2] / occ[1]) - log(pr2[2] / pr2[1])
        @test r2["ln_prior_ratio"] ≈ log(pr2[2] / pr2[1])
    end

    @testset "a k with too few draws is flagged, not quoted" begin
        np = vcat(fill(1, 3000), fill(2, 5), fill(3, 0))
        rep = transdim_k_report(np, 3; n_walkers = 1)
        r12 = rep["ln_bayes_factor_adjacent"][2]            # k = 2 vs 1
        @test !r12["estimable"]
        @test occursin("effective draws", r12["reason"])
        r23 = rep["ln_bayes_factor_adjacent"][3]            # k = 3 has no draw
        @test !r23["estimable"]
        @test isnan(r23["ln_bayes_factor"])
        @test occursin("no draws", r23["reason"])
        # and it warns, naming the pair, for the one that has draws
        @test_logs (:warn, r"ln B\(k = 2 vs 1\) is not quotable") match_mode = :any begin
            Nereus._warn_transdim_k(rep)
        end
    end

    @testset "mixture evidence: the engine's, with k-conditional evidences" begin
        M = _hop_chain([0.2, 0.5, 0.3], 3000, 10)
        rep = transdim_k_report(_flat(M), 2; n_walkers = 10, log_evidence = -123.5,
                                evidence_method = "ti_plus")
        me = rep["mixture_evidence"]
        @test me["available"]
        @test me["log_z"] == -123.5
        @test me["method"] == "ti_plus"
        @test !isempty(me["how"])
        pr3 = transdim_k_prior(2)
        occ = [rep["occupancy"][string(k)] for k in 0:2]
        for k in 0:2
            @test me["ln_z_k"][string(k)] ≈ -123.5 + log(occ[k + 1]) - log(pr3[k + 1])
        end
        # the mixture is the prior-weighted sum of the k evidences
        @test log(sum(exp(me["ln_z_k"][string(k)]) * pr3[k + 1] for k in 0:2)) ≈ -123.5 atol = 1e-9
        none = transdim_k_report(_flat(M), 2; n_walkers = 10)["mixture_evidence"]
        @test !none["available"]
        @test isnan(none["log_z"])
        @test occursin("no quotable evidence", none["reason"])
    end

    @testset "occupancy health: walkers and separate visits" begin
        ok = transdim_k_report(_flat(_hop_chain([0.3, 0.5, 0.2], 2000, 12)), 2; n_walkers = 12)
        @test ok["occupancy_health"]["ok"]
        # k = 2 reached by one walker only: its mass is one excursion
        M = fill(1, 2000, 12)
        M[1:600, 1] .= 2
        bad = transdim_k_report(vec(permutedims(M)), 2; n_walkers = 12)
        h = bad["occupancy_health"]
        @test !h["ok"]
        @test any(occursin("k = 2", s) && occursin("walker", s) for s in h["problems"])
        # three walkers each sitting in k = 2 once for half the run: visits are few
        M2 = fill(1, 2000, 12)
        M2[1:1000, 1:3] .= 2        # three walkers, one long visit each
        h2 = transdim_k_report(vec(permutedims(M2)), 2; n_walkers = 12)["occupancy_health"]
        @test !h2["ok"]
        @test any(occursin("separate visit", s) for s in h2["problems"])
        # a single dominant k is healthy
        h3 = transdim_k_report(fill(1, 2400), 2; n_walkers = 12)["occupancy_health"]
        @test h3["ok"]
    end

    @testset "P(k = k_max) > 0.5 warns, says to raise k_max" begin
        trunc = transdim_k_report(_flat(_hop_chain([0.1, 0.2, 0.7], 2000, 10)), 2;
                                  n_walkers = 10)
        @test trunc["k_max_truncation"]["warn"]
        @test occursin("raise k_max", trunc["k_max_truncation"]["advice"])
        @test_logs (:warn, r"Raise k_max") match_mode = :any Nereus._warn_transdim_k(trunc)
        fine = transdim_k_report(_flat(_hop_chain([0.5, 0.4, 0.1], 2000, 10)), 2;
                                 n_walkers = 10)
        @test !fine["k_max_truncation"]["warn"]
        @test_logs min_level = Logging.Warn Nereus._warn_transdim_k(fine)
        # a fixed-count run (planets = false) never claims truncation
        fixed = transdim_k_report(fill(2, 500), 2; planets = false)
        @test !fixed["k_max_truncation"]["warn"]
    end

    @testset "accepted k transitions, per move type" begin
        res = (; planet_birth_proposed = [10, 20, 30], planet_birth_accepted = [4, 5, 6],
                 planet_death_proposed = [10, 20, 30], planet_death_accepted = [3, 5, 7],
                 noise_td_proposed = [0, 2, 4], noise_td_accepted = [0, 1, 1],
                 planet_noise_swap = Dict("proposed" => 12, "accepted" => 3))
        mv = transdim_move_counts(res)
        @test mv["birth"]["proposed"] == 60 && mv["birth"]["accepted"] == 15
        @test mv["birth"]["cold_accepted"] == 4 && mv["birth"]["cold_rate"] ≈ 0.4
        @test mv["death"]["accepted"] == 15
        @test mv["noise_toggle_swap"]["accepted"] == 2
        @test mv["planet_noise_swap"]["accepted"] == 3
        @test transdim_move_counts(nothing) === nothing
        @test transdim_move_counts((; chains = nothing)) === nothing
    end

    @testset "the convergence fields cannot disagree" begin
        for conv_pass in (true, false), fh_status in ("ok", "fail", "warn"),
            occ_ok in (true, false, nothing), stopped in (nothing, "steps", "max_evals")
            s = Dict{String, Any}(
                "run_info" => Dict{String, Any}(
                    "convergence" => Dict{String, Any}("assessed" => true, "pass" => conv_pass,
                        "worst_rhat" => 1.3, "worst_rhat_param" => "P_k1",
                        "min_ess" => 50.0, "min_ess_param" => "K_k1", "n_fail" => 2)),
                "fit_health" => Dict{String, Any}("overall" => fh_status == "fail" ? "fail" : "ok",
                    "checks" => Dict{String, Any}("convergence" =>
                        Dict{String, Any}("status" => fh_status, "message" => "m"))))
            occ_ok === nothing || (s["run_info"]["transdim"] = Dict{String, Any}(
                "occupancy_health" => Dict{String, Any}("ok" => occ_ok,
                    "problems" => occ_ok ? String[] : ["k = 2 visited by 1 walker"])))
            stopped === nothing || (s["stopped"] = stopped)
            _apply_convergence_verdict!(s)
            want = conv_pass && fh_status != "fail" && occ_ok !== false && stopped != "max_evals"
            conv = s["run_info"]["convergence"]
            chk = s["fit_health"]["checks"]["convergence"]
            @test s["converged"] == want
            @test conv["pass"] == want
            @test conv["converged"] == want
            @test s["fit_health"]["converged"] == want
            # run_info and fit_health agree: the check reads "ok" only if converged
            @test (chk["status"] == "ok") == (want && fh_status == "ok")
            @test s["convergence_verdict"]["converged"] == want
            want || @test !isempty(s["convergence_verdict"]["failed"])
            # idempotent
            before = deepcopy(s)
            _apply_convergence_verdict!(s)
            @test s == before
        end
        # nothing assessed: unknown, and the fields are left alone
        s = Dict{String, Any}("run_info" => Dict{String, Any}(
            "convergence" => Dict{String, Any}("assessed" => false)))
        _apply_convergence_verdict!(s)
        @test s["converged"] === nothing
        @test !haskey(s["run_info"]["convergence"], "pass")
    end

    @testset "evidence of an unconverged fit is reported, flagged, warned about" begin
        mk(conv, rep) = begin
            s = Dict{String, Any}(
                "log_z" => -42.0,
                "evidence" => Dict{String, Any}("ti_plus" => Dict("log_z" => -42.0, "se" => 0.5),
                                                "reported" => rep),
                "run_info" => Dict{String, Any}("log_evidence" => -42.0,
                    "convergence" => Dict{String, Any}("assessed" => true, "pass" => conv,
                        "worst_rhat" => 1.4, "worst_rhat_param" => "P_k1",
                        "min_ess" => 30.0, "min_ess_param" => "K_k1", "n_fail" => 3)))
            _apply_convergence_verdict!(s)
            s
        end
        s = mk(false, "ti_plus")
        logs = Test.collect_test_logs() do
            _evidence_reliability!(s)
        end
        msgs = [string(l.message) for l in logs[1] if l.level == Logging.Warn]
        @test any(occursin("UNRELIABLE", m) && occursin("R-hat", m) for m in msgs)
        @test s["log_z"] == -42.0                                  # never withheld
        @test s["evidence_reliable"] === false
        @test s["evidence"]["evidence_reliable"] === false
        @test s["run_info"]["evidence_reliable"] === false
        @test any(occursin("R-hat/ESS gate", r) for r in s["run_info"]["evidence_unreliable_reasons"])
        # the Laplace fallback is named
        s2 = mk(false, "mode_laplace")
        _evidence_reliability!(s2)
        @test any(occursin("mode-Laplace fallback", r) for r in s2["evidence_unreliable_reasons"])
        # a converged fit is flagged reliable and says nothing
        s3 = mk(true, "ti_plus")
        @test_logs min_level = Logging.Warn _evidence_reliability!(s3)
        @test s3["evidence_reliable"] === true
        @test s3["run_info"]["evidence_reliable"] === true
        @test !haskey(s3, "evidence_unreliable_reasons")
        # no evidence, nothing to flag
        s4 = Dict{String, Any}("log_z" => NaN, "converged" => false)
        _evidence_reliability!(s4)
        @test !haskey(s4, "evidence_reliable")
        # a MAP fit whose optimiser did not converge: its Laplace log_z is flagged
        s5 = Dict{String, Any}("log_z" => -10.0,
                               "map" => Dict{String, Any}("converged" => false))
        _apply_convergence_verdict!(s5)
        _evidence_reliability!(s5)
        @test s5["evidence_reliable"] === false
        @test any(occursin("MAP optimiser", r) for r in s5["evidence_unreliable_reasons"])
        # a run stopped early by a budget
        s6 = Dict{String, Any}("log_z" => -10.0, "stopped" => "max_seconds")
        _apply_convergence_verdict!(s6); _evidence_reliability!(s6)
        @test s6["evidence_reliable"] === false
    end

    @testset "end to end: a short trans-dim fit carries every block" begin
        include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))
        cfg = _easy_rv_cfg()
        tcfg = merge(cfg, Dict{String, Any}(
            "output_dir" => mktempdir(),
            "model" => Dict("max_kplanet" => 2, "planet_modes" => ["RV_ONLY", "RV_ONLY"]),
            "transdim" => Dict("max_kplanet" => 2),
            "sampler" => Dict("name" => "transdim_pt_emcee",
                              "kwargs" => Dict("show_progress" => false, "n_temps" => 4,
                                               "n_walkers" => 40, "n_steps" => 800,
                                               "n_burnin" => 300))))
        s = Nereus.run_job(tcfg)
        td = s["run_info"]["transdim"]
        @test td["prior"]["inclusion_prior"] == 0.5
        @test td["prior"]["p_k"] ≈ transdim_k_prior(2)
        @test haskey(td, "ln_bayes_factor_adjacent") && length(td["ln_bayes_factor_adjacent"]) == 2
        @test haskey(td, "mixture_evidence") && haskey(td["mixture_evidence"], "available")
        @test haskey(td, "occupancy_health") && haskey(td, "k_max_truncation")
        mv = td["k_transitions"]
        @test mv["birth"]["proposed"] > 0 && mv["death"]["proposed"] > 0
        @test s["diagnostics"]["k_transitions"]["birth"]["proposed"] == mv["birth"]["proposed"]
        @test haskey(s["model_selection"], "n_planets_prior")
        # both convergence fields come from the one verdict
        @test s["converged"] isa Bool
        @test s["run_info"]["convergence"]["pass"] == s["converged"]
        @test s["fit_health"]["converged"] == s["converged"]
        chk = s["fit_health"]["checks"]["convergence"]
        @test (chk["status"] == "ok") == (s["converged"] && chk["own_status"] == "ok")
        # the evidence flag sits beside the evidence whenever there is one
        isfinite(s["log_z"]) && @test s["run_info"]["evidence_reliable"] == s["converged"]
    end
end
