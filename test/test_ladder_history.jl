# The ladder history: what a tempered run did to its own temperature ladder,
# step by step. `sample_pt_emcee` and `sample_transdim_pt_emcee` used to keep
# only the run-total swap acceptance and the final β, so the two figures every
# reddemcee user reads first -- T / swap rate / swap mean distance against
# step, and E[log L] against β -- could not be drawn from a Nereus run at all.
#
# These tests pin the recorded quantities to numbers the run already reports
# (the per-step swap rate must average to `acceptance_swap`; the per-rung
# mean log L must integrate to the TI the evidence report quotes), so a
# history that drifts from the run it describes fails here.

using Test, Nereus, Random, Statistics

@testset "ladder history" begin

    Random.seed!(7)
    n = 60; t = sort!(400 .* rand(n))
    rv = 40.0 .* sin.(2π .* t ./ 4.23) .+ 1.5 .* randn(n)
    mk_target() = build_target(
        planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                        Mo = UniformPrior(0.0, 2pi)),),
        rv = (SIM = (data = (t = t, rv = rv, rv_err = fill(1.5, n)),
                     sigma = LogUniformPrior(0.5, 10.0)),))
    n_temps, n_steps, n_burnin = 6, 300, 150

    @testset "swap-distance scale is the prior width on the move scale" begin
        priors = [UniformPrior(0.0, 10.0), LogUniformPrior(1.0, 100.0),
                  NormalPrior(3.0, 2.0), ModJeffreysPrior(1.0, 99.0)]
        shift = Union{Nothing,Float64}[Nereus.log_scale_shift(p) for p in priors]
        D = Nereus._swap_distance_scale(priors, shift)
        @test D[1] ≈ 10.0
        @test D[2] ≈ log(100.0)
        # No hard bounds: the central 99.73% of the prior, i.e. ±3σ.
        @test D[3] ≈ 12.0 rtol = 1e-3
        # ModJeffreys is flat in log(x + knee): log(99 + 1) - log(0 + 1).
        @test D[4] ≈ log(100.0)
        @test all(isfinite, D) && all(>(0), D)
    end

    @testset "trailing running mean" begin
        @test Nereus._running_mean([1.0, 2.0, 3.0, 4.0], 2) ≈ [1.0, 1.5, 2.5, 3.5]
        @test Nereus._running_mean([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]
        @test Nereus._running_mean([2.0, 4.0, 6.0], 10) ≈ [2.0, 3.0, 4.0]
    end

    @testset "pt_emcee, fixed ladder" begin
        target = mk_target()
        r = sample_pt_emcee(target, target.data; n_temps = n_temps, n_walkers = 20,
                            n_steps = n_steps, n_burnin = n_burnin, seed = 42,
                            show_progress = false)
        @test hasproperty(r, :ladder)
        L = r.ladder
        @test L isa Nereus.LadderHistory
        @test size(L.betas) == (n_steps, n_temps)
        @test size(L.swap_rate) == (n_steps, n_temps - 1)
        @test size(L.swap_distance) == (n_steps, n_temps - 1)
        @test L.n_burnin == n_burnin

        # Not adapting: every step ran on the ladder the run reports.
        @test all(L.betas[s, :] == r.betas for s in 1:n_steps)

        # Every step proposes the same number of swaps per pair, so the mean
        # of the per-step rates IS the run total.
        @test vec(mean(L.swap_rate; dims = 1)) ≈ r.acceptance_swap atol = 1e-12
        @test all(0 .<= L.swap_rate .<= 1)

        # reddemcee's SMD: rejected swaps count as zero distance, so it is
        # zero exactly where nothing was accepted; and with every dimension
        # scaled by its prior width no accepted swap can travel further than
        # the unit hypercube's diagonal.
        n_dim = length(target.params.layout.unfrozen_idx)
        @test all(L.swap_distance .>= 0)
        @test all((L.swap_distance .== 0) .== (L.swap_rate .== 0))
        @test all(L.swap_distance .<= L.swap_rate .* sqrt(n_dim) .+ 1e-12)
        @test any(>(0), L.swap_distance)

        # The per-rung <log L> is the array the tempered evidence integrates.
        @test length(L.mean_logL) == n_temps
        @test all(isfinite, L.mean_logL)
        @test L.mean_logL[1] > L.mean_logL[end]          # cold fits better than hot
        @test Nereus.ti_trapezoidal(L.mean_logL, r.betas) ≈ r.evidence.ti[1]
    end

    @testset "pt_emcee, adaptive ladder" begin
        target = mk_target()
        r = sample_pt_emcee(target, target.data; n_temps = n_temps, n_walkers = 20,
                            n_steps = n_steps, n_burnin = n_burnin, seed = 42,
                            adapt_ladder = true, show_progress = false)
        L = r.ladder
        # Recorded after each step's adaptation: the last row is the final ladder.
        @test L.betas[end, :] == r.betas
        # It moved during burn-in ...
        @test L.betas[1, :] != L.betas[n_burnin, :]
        # ... the endpoints never do ...
        @test all(==(1.0), L.betas[:, 1])
        @test all(==(L.betas[1, end]), L.betas[:, end])
        # ... and nothing adapts once the recorded chain has started.
        @test all(L.betas[s, :] == r.betas for s in n_burnin:n_steps)
        @test all(issorted(L.betas[s, :]; rev = true) for s in 1:n_steps)
    end

    @testset "trans-dim sibling records the same history" begin
        Random.seed!(11)
        t_rv = sort!(200 .* rand(40))
        ic   = InstrumentConfig(rv = ["HARPS"])
        data = Data(; t_rv = t_rv,
                    rv = 30.0 .* sin.(2π .* t_rv ./ 7.3) .+ randn(40),
                    rv_err = ones(length(t_rv)))
        params = Params(max_kplanet = 2, planet_modes = [RV_ONLY, RV_ONLY],
                        instruments = ic, data = data, M_s = 1.0)
        target = NereusTarget(params, data; unconstrained = false)
        td     = TransDimConfig(max_kplanet = 2)
        res = sample_transdim_pt_emcee(target, data; td = td, n_temps = 4,
                                       n_walkers = 16, n_steps = 200, n_burnin = 100,
                                       seed = 7, show_progress = false)
        @test hasproperty(res, :ladder)
        L = res.ladder
        @test size(L.betas) == (200, 4)
        @test size(L.swap_rate) == (200, 3)
        @test size(L.swap_distance) == (200, 3)
        # This engine adapts its ladder by default, during burn-in only: the
        # history ends on the reported ladder and is frozen from n_burnin on.
        @test L.betas[end, :] == res.betas
        @test L.betas[1, :] != res.betas
        @test all(L.betas[s, :] == res.betas for s in 100:200)
        @test all(==(1.0), L.betas[:, 1])
        @test vec(mean(L.swap_rate; dims = 1)) ≈ res.acceptance_swap atol = 1e-12
        @test all(L.swap_distance .>= 0)
        @test all(isfinite, L.swap_distance)
        @test all((L.swap_distance .== 0) .| (L.swap_rate .> 0))
        @test length(L.mean_logL) == 4
        @test Nereus.ti_trapezoidal(L.mean_logL, res.betas) ≈ res.evidence_report.ti[1]
    end

    @testset "figures" begin
        target = mk_target()
        r = sample_pt_emcee(target, target.data; n_temps = n_temps, n_walkers = 20,
                            n_steps = n_steps, n_burnin = n_burnin, seed = 42,
                            adapt_ladder = true, show_progress = false)
        chains, params, data = r.chains, target.params, target.data

        out = mktempdir()
        plot_ladder_rates(r; output = out)
        plot_beta_ladder(r; output = out)
        @test isfile(joinpath(out, "betas", "rates.png"))
        @test isfile(joinpath(out, "betas", "beta_ladder.png"))

        # Through the runner: named kinds, and part of "auto" when -- and only
        # when -- the engine returned a ladder history.
        out2 = mktempdir()
        @test Nereus._dispatch_plot("ladder_rates", chains, params, data, out2,
                                    Dict{Symbol,Any}(); result = r) == "betas/rates.png"
        @test Nereus._dispatch_plot("beta_ladder", chains, params, data, out2,
                                    Dict{Symbol,Any}(); result = r) == "betas/beta_ladder.png"
        @test isfile(joinpath(out2, "betas", "rates.png"))
        @test isfile(joinpath(out2, "betas", "beta_ladder.png"))
        @test Nereus._dispatch_plot("ladder_rates", chains, params, data, mktempdir(),
                                    Dict{Symbol,Any}()) === nothing
        @test Nereus._dispatch_plot("beta_ladder", chains, params, data, mktempdir(),
                                    Dict{Symbol,Any}(); result = (; chains)) === nothing

        auto = Nereus._auto_plot_kinds(chains, params, data; result = r)
        @test "ladder_rates" in auto && "beta_ladder" in auto
        bare = Nereus._auto_plot_kinds(chains, params, data)
        @test !("ladder_rates" in bare) && !("beta_ladder" in bare)

        # And end to end through `_make_plots`, which is what run_job and the
        # fit_* API both call.
        out3 = mktempdir()
        cfg = Dict("output" => Dict("plots" => ["ladder_rates", "beta_ladder"],
                                    "show_progress" => false))
        made = Nereus._make_plots(cfg, chains, params, data, out3; result = r)
        @test "betas/rates.png" in made && "betas/beta_ladder.png" in made
        @test isfile(joinpath(out3, "plots", "betas", "rates.png"))
        @test isfile(joinpath(out3, "plots", "betas", "beta_ladder.png"))
    end
end
