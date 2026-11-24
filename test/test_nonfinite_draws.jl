# A draw whose model prediction is not finite (a transit with an unphysical
# geometry, a NaN parameter) must not take the posterior-predictive check or
# LOO/WAIC down with it: it is left out, counted, and warned about. Before, one
# NaN in the per-draw χ² made `quantile` throw, and one NaN prediction made
# `compute_loo` refuse the whole chain. The same code ran for an RV, a transit
# and a joint fit, so each is injected here.
using Test, Nereus, Logging, MCMCChains, Random

@testset "non-finite draws in PPC and LOO" begin
    fx = Nereus._fit_precompile_inputs()
    quiet(f) = Logging.with_logger(Logging.NullLogger()) do
        redirect_stdio(f; stdout = devnull, stderr = devnull)
    end
    eng(n) = Dict("engine" => "pt_emcee", "options" => Dict(
        "n_steps" => n, "n_burnin" => n ÷ 2, "n_walkers" => 16, "n_temps" => 3,
        "seed" => 1, "show_progress" => false))

    # `kind => (channels, planets, kwargs, fit)` -- the target the fit builds and
    # a chain from the fit itself.
    rvch = Dict("source" => "RV", "data" => fx.data)
    cases = [
        "rv"      => (Nereus._target_from([rvch],
                                          Nereus._planet_spec(1; block = Nereus.default_rv_planet(
                                              fx.data["HARPS"].t, fx.data["HARPS"].rv));
                                          time_anchor = :phic),
                      () -> fit_rv(fx.data; engine = eng(60), output_dir = mktempdir(),
                                   science = false)),
        "transit" => (Nereus._target_from([fx.phj], 1; time_anchor = :phic,
                                          priors = fx.priors, M_s = 1.0, R_s = 1.0),
                      () -> fit_transit(fx.phot; time_anchor = :phic, priors = fx.priors,
                                        M_s = 1.0, R_s = 1.0, engine = eng(100),
                                        output_dir = mktempdir(), science = false)),
        "joint"   => (Nereus._target_from([fx.rvj, fx.phj], fx.planets; time_anchor = :phic,
                                          M_s = 1.0, R_s = 1.0),
                      () -> fit_joint(fx.rvj, fx.phj; planets = fx.planets, time_anchor = :phic,
                                      M_s = 1.0, R_s = 1.0, engine = eng(100),
                                      output_dir = mktempdir(), science = false)),
    ]
    for (label, (tgt, fit)) in cases
        @testset "$label" begin
            r = quiet(fit)
            ch = r.chains
            params, data = tgt.params, tgt.data
            nms = names(ch)
            A = Array(ch.value.data)
            n_iter, _, n_chain = size(A)
            col = findfirst(==(label == "transit" ? :offset_T : :P_k1), nms)
            @test col !== nothing
            clean = ch
            # One non-finite draw in every few: NaN period.
            A2 = copy(A)
            nbad = 0
            for it in 1:7:n_iter, c in 1:n_chain
                A2[it, col, c] = NaN
                nbad += 1
            end
            bad = MCMCChains.Chains(A2, nms)
            n_all = n_iter * n_chain

            # PPC: all draws examined, the bad ones left out, counted, warned.
            ppc = @test_logs (:warn, r"non-finite prediction") match_mode = :any begin
                posterior_predictive_check(bad, params, data; n_draws = n_all,
                                           rng = MersenneTwister(1))
            end
            @test ppc.summary["n_nonfinite_draws"] == nbad
            @test ppc.summary["n_draws"] == n_all - nbad
            @test all(isfinite, ppc.rv_chi2_per_draw) && all(isfinite, ppc.phot_chi2_per_draw)
            key = data.t_rv |> isempty ? "phot_red_chi2_per_draw_p50" : "rv_red_chi2_per_draw_p50"
            @test isfinite(ppc.summary[key])
            ppc0 = posterior_predictive_check(clean, params, data; n_draws = n_all,
                                              rng = MersenneTwister(1))
            @test ppc0.summary["n_nonfinite_draws"] == 0

            # LOO/WAIC: the same.
            loo = @test_logs (:warn, r"non-finite") match_mode = :any begin
                compute_loo(bad, params, data; n_draws = n_all, rng = MersenneTwister(1))
            end
            @test loo.n_nonfinite_draws == nbad
            @test loo.n_draws == n_all - nbad
            @test isfinite(loo.elpd_waic)
            # The draws left out change nothing: the same chain without them gives
            # the same WAIC (a sum over draws, so the order does not matter).
            keep = setdiff(1:n_iter, 1:7:n_iter)
            sub = MCMCChains.Chains(A[keep, :, :], nms)
            loo_sub = compute_loo(sub, params, data; n_draws = length(keep) * n_chain,
                                  rng = MersenneTwister(1))
            @test loo.elpd_waic ≈ loo_sub.elpd_waic rtol = 1e-10
            ppc_sub = posterior_predictive_check(sub, params, data;
                          n_draws = length(keep) * n_chain, rng = MersenneTwister(1))
            @test sort(vcat(ppc.rv_chi2_per_draw, ppc.phot_chi2_per_draw)) ≈
                  sort(vcat(ppc_sub.rv_chi2_per_draw, ppc_sub.phot_chi2_per_draw))
            loo0 = compute_loo(clean, params, data; n_draws = n_all, rng = MersenneTwister(1))
            @test loo0.n_nonfinite_draws == 0

            # Every draw non-finite: nothing to check, an ArgumentError that says why.
            A3 = copy(A); A3[:, col, :] .= NaN
            allbad = MCMCChains.Chains(A3, nms)
            @test_throws ArgumentError posterior_predictive_check(allbad, params, data;
                                                                  n_draws = 20)
            @test_throws ArgumentError compute_loo(allbad, params, data; n_draws = 20)
        end
    end

    @testset "a transit with no a/R★ source says so" begin
        @test_logs (:warn, r"nothing sets its a/R★") match_mode = :any begin
            Nereus._target_from([fx.phj], 1; time_anchor = :phic, priors = fx.priors)
        end
        @test_logs min_level = Logging.Warn begin
            Nereus._target_from([fx.phj], 1; time_anchor = :phic, priors = fx.priors,
                                M_s = 1.0, R_s = 1.0)
        end
    end

    @testset "the workload's fits leave no PPC or LOO failure" begin
        dir = mktempdir()
        try
            quiet() do
                Nereus._fit_precompile_run(fx, dir)
            end
            for sub in ("fit_rv", "fit_transit", "fit_joint", "run_job")
                s = Nereus.JSON3.read(read(joinpath(dir, sub, "summary.json"), String))
                @test !haskey(s.ppc, :status)
                @test !haskey(s.loo, :status)
                @test s.ppc.n_nonfinite_draws == 0
            end
        finally
            rm(dir; recursive = true, force = true)
        end
    end
end
