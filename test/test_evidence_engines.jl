# The evidence engines on what made each give wrong or poor answers on an RV +
# Hipparcos-IAD companion:
#
# - Daedalus replaced a dead point by a short walk that stayed by its donor
#   (log Z over 150-169 across 8 seeds at i = 5°, where bridge sampling gives
#   165.5), and its live points all started at the data's γ. The walk now moves
#   every coordinate at once, shaped by the live points, at a scale adapted
#   between walks, for as many steps as decorrelate it, and the live points
#   start from the prior.
# - pt_hmc's quadratic ladder from β = 0 left the integrand's climb through
#   four decades to one segment (TI+ at -5 and -624 where log Z is 164). Its
#   ladder is now β = 0 and a geometric run from beta_min, the tempered value
#   SS+, quoted only when its rungs overlap; and SS+ is fed every walker's
#   log L, not their mean.
# - Population annealing resampled at every step and stopped mutating after a
#   count of accepted moves; it now resamples on the ESS, steps β on the
#   conditional ESS, and mutates until the replicas have decorrelated.

using Nereus, MCMCChains, Random, Test
using Logging: Logging
using Statistics: mean, std

# A Gaussian prior and likelihood, so log Z is exact; the likelihood 200 times
# narrower than the prior in each of 3 dimensions, so ⟨log L⟩ climbs from -6e4
# at the prior, as on the companion.
const _EE_D = 3
const _EE_S0 = 10.0
const _EE_S1 = 0.05
const _EE_MU = [1.0, -2.0, 0.5]
_ee_parts(y) = (-_EE_D / 2 * log(2π * _EE_S0^2) - sum(abs2, y) / (2 * _EE_S0^2),
                -_EE_D / 2 * log(2π * _EE_S1^2) - sum(abs2, y .- _EE_MU) / (2 * _EE_S1^2))
const _EE_LOGZ = -_EE_D / 2 * log(2π * (_EE_S0^2 + _EE_S1^2)) -
                 sum(abs2, _EE_MU) / (2 * (_EE_S0^2 + _EE_S1^2))

function _ee_core(βs, W; seed = 1)
    rng = MersenneTwister(seed)
    inits = [_EE_S0 .* randn(rng, _EE_D) for _ in 1:(length(βs) * W)]
    return Nereus._pt_hmc_core(_ee_parts, _EE_D, inits, βs; n_sweeps = 600, n_warmup = 200,
                               swap_interval = 1, target_accept = 0.8, W = W,
                               progress = false, rng = rng)
end

@testset "pt_hmc: the default ladder and SS+ recover an exact log Z" begin
    βs = Nereus._pt_hmc_ladder(8, 1e-4)
    @test βs[1] == 0 && βs[2] ≈ 1e-4 && βs[end] == 1
    @test all(≈(1e-4^(-1 / 6)), βs[3:end] ./ βs[2:(end - 1)])
    @test Nereus._pt_hmc_ladder(2, 1e-4) == [0.0, 1.0]
    # Measured, 3 seeds: SS+ within 0.3 of the exact -9.69 with one walker per
    # rung and with two; TI+ 1.7 to 2.1 low, from the first segment. With two
    # walkers SS+ was fed their mean log L and came out 1.9 to 2.3 low.
    for W in (1, 2)
        r = _ee_core(βs, W)
        @test abs(r.report.ss_plus[1] - _EE_LOGZ) < 0.6
        ok, why = Nereus._pt_hmc_tempered_ok(r.report, βs, r.swap_acc, 0.05)
        @test ok
        @test why == ""
        @test length(r.swap_acc) == length(βs) - 1 && all(0 .<= r.swap_acc .<= 1)
    end
    # The old ladder: no overlap between β = 0 and its second rung (TI+ near
    # -500, SS+ -12 to -68), which the gate reports instead of a number.
    quad = Float64[((i - 1) / 7)^2 for i in 1:8]
    r = _ee_core(quad, 1)
    @test abs(r.report.ti_plus[1] - _EE_LOGZ) > 100
    ok, why = Nereus._pt_hmc_tempered_ok(r.report, quad, r.swap_acc, 0.05)
    @test !ok
    @test occursin("swapped", why)
    # A ladder that does not reach β = 0 leaves the head out of the integral.
    r = _ee_core(βs[2:end], 1)
    ok, why = Nereus._pt_hmc_tempered_ok(r.report, βs[2:end], r.swap_acc, 0.05)
    @test !ok
    @test occursin("head", why)
end

# A small RV target for the end-to-end checks.
function _ee_rv(; P = NormalPrior(4.16, 0.05, 4.0, 4.3))
    rng = MersenneTwister(7)
    t = sort(rand(rng, 40) .* 30.0)
    rv = 5.0 .+ 12.0 .* sin.(2π .* t ./ 4.16) .+ 1.5 .* randn(rng, 40)
    data = Data(; t_rv = t, rv = rv, rv_err = fill(1.5, 40))
    params = Params(max_kplanet = 1, planet_modes = [RV_ONLY],
                    instruments = InstrumentConfig(rv = ["X"]), data = data, M_s = 1.0,
                    priors = Dict{String,PriorSpec}("P_k1" => P))
    return NereusTarget(params, data; unconstrained = false), data
end

@testset "pt_hmc: a number only when it can be quoted" begin
    tg, _ = _ee_rv()
    ch, lz, rep, ev = sample_pt_hmc(tg; n_temps = 5, n_sweeps = 60, n_warmup = 60,
                                    adapt_ladder = false, bridge_n = 0, seed = 2,
                                    progress = false)
    @test ev.betas == Nereus._pt_hmc_ladder(5, 1e-4)
    @test isnan(ev.bridge) && ev.reported != "bridge"
    if ev.tempered_ok
        @test lz == ev.tempered == rep.ss_plus[1] && ev.reported == "ss_plus"
    else
        @test isnan(lz) && ev.reported == "none" && !isempty(ev.tempered_why)
    end
    # Two rungs, 0 and 1, cannot overlap on this target: no number, and a warning.
    ch, lz, rep, ev = @test_logs (:warn, r"no quotable log-evidence") match_mode = :any sample_pt_hmc(
        tg; n_temps = 2, n_sweeps = 60, n_warmup = 60, adapt_ladder = false, bridge_n = 0,
        seed = 2, progress = false)
    @test isnan(lz)
    @test !ev.tempered_ok
end

@testset "pt_hmc: the warm-start pre-search reports no evidence of its own" begin
    # The pre-search, a short pt_emcee, computed and warned about its own log Z,
    # and pt_hmc's output on the companion carried "sample_pt_emcee: bridge and
    # mode-Laplace disagree by 154.0 nats ... bridge = 13.74" beside the 165.51
    # it reported. Two rungs make a pre-search's tempered stack disagree.
    tg, data = _ee_rv()
    tu = NereusTarget(tg.params, data; unconstrained = true)
    tl = Test.TestLogger(min_level = Logging.Warn)
    pts = Logging.with_logger(tl) do
        Nereus._warmstart_points(tu, 4, MersenneTwister(1); n_temps = 2,
                                 n_steps = 200, n_burnin = 100)
    end
    @test length(pts) == 4 && all(p -> all(isfinite, p), pts)
    @test !any(r -> occursin("sample_pt_emcee", string(r.message)), tl.logs)
end

@testset "daedalus: a replacement walk forgets its donor" begin
    tg, data = _ee_rv()
    params = tg.params
    td = TransDimConfig(max_kplanet = 1, planets = false)
    strategy = Nereus.MoMSBirth(params; rng = MersenneTwister(1))
    live = [Nereus._daedalus_init_point(tg, data, strategy, td, 0.5, MersenneTwister(k))
            for k in 1:120]
    d = length(params.layout.unfrozen_idx)
    # The live points start as prior draws, γ included: it was the data's median
    # RV at every one of them, which the walks then could not move.
    g = [p.values[params.layout.name_to_idx["gamma_X"]] for p in live]
    @test allunique(g)
    # No constraint (L_thr = -Inf): the constrained prior is the prior, and the
    # live points are draws from it. The walk's scale and length adapt over 40
    # walks as in sample_daedalus, then 150 walks from one donor must spread
    # like the live points do. The single-coordinate walk at 1/50 of each
    # prior width kept them within a fraction of that.
    scale = Ref(2.38 / sqrt(d)); jump = Ref(NaN)
    for k in 1:40
        c = Ref((0, 0, 0.0))
        nw = Nereus._daedalus_walk_length(jump, 25, d, 16.0)
        Nereus._daedalus_evolve(live[k], tg, data, strategy, td, -Inf, 25, 0.5,
                                MersenneTwister(100 + k); live, scale = scale[],
                                n_walk = nw, counts = c)
        a, r, j = c[]
        Nereus._daedalus_update_scale!(scale, a, r, d)
        jump[] = isfinite(jump[]) ? 0.9 * jump[] + 0.1 * j / (a + r) : j / (a + r)
    end
    nw = Nereus._daedalus_walk_length(jump, 25, d, 16.0)
    @test 25 <= nw <= 1000
    donor = live[1]
    out = [Nereus._daedalus_evolve(donor, tg, data, strategy, td, -Inf, 25, 0.5,
                                   MersenneTwister(1000 + s); live, scale = scale[],
                                   n_walk = nw) for s in 1:150]
    for idx in params.layout.unfrozen_idx
        ref = [p.values[idx] for p in live]
        got = [p.values[idx] for p in out]
        @test std(got) > 0.6 * std(ref)
        @test abs(mean(got) - mean(ref)) < 0.4 * std(ref)
    end
    # And the run returns its log Z's standard error, √(H / n_live).
    ch, lz, _, lze = sample_daedalus(tg, data; td, n_live = 40, seed = 3,
                                     show_progress = false)
    @test isfinite(lz)
    @test 0 < lze < 2
end

@testset "population annealing: resampled on the ESS, mutated to decorrelation" begin
    # P on LogUniform(1, 20) and e free. Measured, seeds 1-3: log Z -98.9,
    # -97.6, -97.1, where nested INS gives -97.2 and -96.9 (n_live = 400); the
    # sampler resampling at every step, mutating to 5 accepted moves a replica,
    # gave -107.3, -107.3 and -113.6.
    tg, data = _ee_rv(P = LogUniformPrior(1.0, 20.0))
    r = sample_pa(tg, data; n_replicas = 120, seed = 2, show_progress = false)
    ref = sample_nested_ins(tg, data; n_live = 400, seed = 1, show_progress = false).log_z_ns
    @test abs(r.log_evidence - ref) < 3
    n = length(r.beta_history) - 1
    @test r.beta_history[end] == 1.0
    @test length(r.resampled) == length(r.sweep_history) == length(r.corr_history) == n
    @test r.resampled[end]
    @test count(r.resampled) < n
    # Every step either decorrelated or ran into the cap of 30 × n_mcmc
    # sweeps (the e-ω-M₀ ridge of a free e near 0 does not decorrelate).
    @test all((r.corr_history .<= 0.2) .| (r.sweep_history .== 300))
    # The decorrelation measure: rank correlation, so 1 for a monotone map of
    # the same positions and 0 for positions the start does not predict.
    X0 = reshape(collect(1.0:10.0), 10, 1)
    W = fill(0.1, 10)
    @test Nereus._pa_max_corr(X0, exp.(X0), W) ≈ 1
    @test Nereus._pa_max_corr(X0, reshape([3.0, 1, 4, 1, 5, 9, 2, 6, 5, 3], 10, 1), W) < 0.5
    @test Nereus._pa_ranks([2.0, 1.0, 2.0]) == [2.5, 1.0, 2.5]
end

@testset "sample_smc: no step cap unless betas is given" begin
    tg, data = _ee_rv()
    r = @test_logs (:info, r"routing through sample_pa") match_mode = :any sample_smc(
        tg, data; n_replicas = 60, n_mcmc = 2, seed = 1, show_progress = false)
    @test r.beta_history[end] == 1.0
    @test isfinite(r.log_evidence)
end
