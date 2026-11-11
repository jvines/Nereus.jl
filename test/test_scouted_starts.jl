# Scouted starts (src/samplers/scouts.jl). An RV + Hipparcos-IAD companion on a
# circular orbit, 0.05 mas per scan, a on LogUniform(0.8, 2) AU and M_sec on
# LogUniform(0.003, 0.03): a chain that starts at the wrong period settles with
# M_sec on the lower edge of its prior, 23,000+ nats below the mode, and no
# local move leaves. rjmcmc, moms and cold NUTS started each chain at one prior
# draw, and the ensemble every walker; 3 of 8 single chains of each of the
# first three stranded at i = 45°, two of four NUTS chains, and one ensemble of
# four at i = 5° (at 161°, 1,500 nats down). Cold pt_hmc froze its β = 1
# chain, and Pathfinder's runs all stopped on the plateau, the same way.
#
# Pinned: the scouts the bound drops are the stranded ones and only those; a
# target without a signal, where the plateau IS the posterior, keeps every
# scout; and every chain of each local sampler comes back in the mode.

using Nereus, MCMCChains, Random, Test
using Statistics: median, mean, std
using Nereus: Theta, set_param!, IADData

const _SC_N = 220
const _SC_RNG = MersenneTwister(11)
const _SC_T = sort(57000 .+ 1800 .* rand(_SC_RNG, _SC_N))
const _SC_PSI = 2π .* rand(_SC_RNG, _SC_N)
const _SC_TRV = sort(57000 .+ 1800 .* rand(_SC_RNG, 40))

function _sc_build(w, rv; unconstrained, efree = false)
    dt = (_SC_T .- mean(_SC_T)) ./ 365.25
    pf = sin.(2π .* (_SC_T .- 57000) ./ 365.25 .- _SC_PSI)
    iad = IADData(t = _SC_T, abscissa = w, abscissa_err = fill(0.05, _SC_N), psi = _SC_PSI,
                  parallax_factor = pf, pm_factor = dt)
    blk = (a = LogUniformPrior(0.8, 2.0), M_sec = LogUniformPrior(0.003, 0.03),
           sesinw = efree ? UniformPrior(-1.0, 1.0) : FixedPrior(0.0),
           secosw = efree ? UniformPrior(-1.0, 1.0) : FixedPrior(0.0), inc = SinePrior(),
           Omega = UniformPrior(0.0, 2π), Mo = UniformPrior(0.0, 2π))
    tg = build_target(M_pri = 0.644, planets = (b = blk,), iad = iad,
                      rv = (HARPS = (data = (t = _SC_TRV, rv = rv, rv_err = fill(5.0, 40)),
                                     sigma = 0.0, gamma = UniformPrior(-50.0, 50.0)),),
                      plx = 13.6, M_s = 0.644)
    return NereusTarget(tg.params, tg.data; unconstrained)
end

# The companion at inclination `inc_deg` (signal = false: noise alone; efree:
# √e sin ω and √e cos ω sampled on U(-1, 1), the truth still circular).
function _sc_target(inc_deg; unconstrained = false, signal = true, efree = false)
    tg0 = _sc_build(zeros(_SC_N), zeros(40); unconstrained = false)
    th = Theta{Float64}(tg0.params)
    for (nm, v) in ("a_k1" => 1.18, "M_sec_k1" => 0.0105, "inc_k1" => deg2rad(inc_deg),
                    "Omega_k1" => 3.07, "Mo_k1" => 0.5, "gamma_HARPS" => 0.0)
        set_param!(th, nm, v)
    end
    _, orbs, Ms = Nereus._iad_active_orbits(th, Nereus.astrom_M_pri(th),
                                             Nereus.astrom_plx(th), tg0.data.t_ref)
    r = zeros(_SC_N)
    Nereus._iad_residuals!(r, tg0.data.iad, orbs, Ms, Nereus.astrom_plx(th))
    rvm, _ = Nereus.rv_predictions(th, tg0.data)
    rng = MersenneTwister(12)
    w = (signal ? -r : zeros(_SC_N)) .+ 0.05 .* randn(rng, _SC_N)
    rv = (signal ? rvm : zeros(40)) .+ 5.0 .* randn(rng, 40)
    return _sc_build(w, rv; unconstrained, efree)
end

_sc_draw!(tg) = (th, r) -> (th.values[th.params.layout.unfrozen_idx] .=
                                Nereus._draw_from_prior(tg, r); th)

# Every chain's draws in the mode: folded inclination within `tol` degrees of
# `inc`, and M_sec off the plateau's edge.
function _sc_in_mode(ch, inc; tol = 3.0)
    for c in 1:size(ch, 3)
        i = Nereus.fold_inclination.(vec(Array(ch[:, :inc_k1, c])))
        m = vec(Array(ch[:, :M_sec_k1, c]))
        @test abs(rad2deg(median(i)) - inc) < tol
        @test minimum(m) > 0.006
    end
end

@testset "scouted starts" begin
    @testset "the bound drops the stranded scouts and only those" begin
        tg = _sc_target(45.0)
        p = tg.params
        th = Theta{Float64}(p)
        L = p.layout
        surv = Nereus._scout_survivors(th, tg.data, MersenneTwister(1),
                                       Nereus._prior_widths(L); draw! = _sc_draw!(tg))
        # 11 of 16 at the mode (log L ≈ 192-195); the other 5 settled on the
        # plateau, log L ≈ -23,000, or near it.
        @test 6 <= length(surv) < 16
        ii = findfirst(==("inc_k1"), L.unfrozen_names)
        im = findfirst(==("M_sec_k1"), L.unfrozen_names)
        for v in surv
            x = v[L.unfrozen_idx]
            th.values .= v
            @test Nereus._eval_ll(th, tg.data) > 150
            @test abs(rad2deg(Nereus.fold_inclination(x[ii])) - 44.6) < 3
            @test x[im] > 0.006
        end
    end

    @testset "no signal: the plateau is the posterior and every scout stays" begin
        tg = _sc_target(45.0; signal = false)
        p = tg.params
        surv = Nereus._scout_survivors(Theta{Float64}(p), tg.data, MersenneTwister(1),
                                       Nereus._prior_widths(p.layout); draw! = _sc_draw!(tg))
        @test length(surv) == 16
    end

    @testset "rjmcmc and moms: every chain in the mode" begin
        tg = _sc_target(45.0)
        td = Nereus.TransDimConfig(max_kplanet = 1, planets = false)
        ch, _ = sample_rjmcmc(tg, tg.data; td, n_samples = 2000, n_warmup = 1000,
                              n_chains = 4, seed = 3, show_progress = false)
        _sc_in_mode(ch, 44.6)
        ch, _, _ = Nereus.sample_moms(tg, tg.data; td, n_samples = 2000, n_warmup = 1000,
                                      n_chains = 4, seed = 3, show_progress = false)
        _sc_in_mode(ch, 44.6)
    end

    @testset "cold NUTS: every chain in the mode" begin
        tg = _sc_target(89.5; unconstrained = true)
        ch = sample_nuts(tg; n_samples = 200, n_warmup = 300, n_chains = 4,
                         warm_start = false, seed = 3, progress = false)
        _sc_in_mode(ch, 88.7; tol = 4.0)
    end

    # Each rung's kernel is frozen at the end of its warm-up: one run on the
    # plateau left the β = 1 chain unable to move once a swap brought it to
    # the mode (one point for all 1000 sweeps, seeds 1 and 2).
    @testset "cold pt_hmc: the β = 1 chain in the mode, and moving" begin
        tg = _sc_target(5.0; unconstrained = true)
        ch, _, _ = sample_pt_hmc(tg; n_temps = 4, n_sweeps = 300, n_warmup = 200,
                                 warm_start = false, seed = 1, progress = false)
        _sc_in_mode(ch, 4.9; tol = 1.5)
        @test std(vec(Array(ch[:, :inc_k1, 1]))) > deg2rad(0.1)
    end

    # Pathfinder's own starts: all 8 runs on the plateau at e = 0.885.
    @testset "pathfinder: the draws in the mode" begin
        tg = _sc_target(89.5; unconstrained = true, efree = true)
        pf = pathfinder_init(tg; n_runs = 8, n_draws = 200, seed = 1)
        L = tg.params.layout
        X = reduce(hcat, [Nereus.transform_inverse(pf.draws[:, j], tg.transform)
                          for j in axes(pf.draws, 2)])
        ii = findfirst(==("inc_k1"), L.unfrozen_names)
        e = X[findfirst(==("sesinw_k1"), L.unfrozen_names), :] .^ 2 .+
            X[findfirst(==("secosw_k1"), L.unfrozen_names), :] .^ 2
        @test abs(rad2deg(median(Nereus.fold_inclination.(X[ii, :]))) - 88.7) < 3
        @test median(e) < 0.05
    end

    @testset "ensemble: every chain in the mode" begin
        tg = _sc_target(5.0; unconstrained = true)
        ch = sample_ensemble(tg; n_walkers = 32, n_steps = 1500, n_burnin = 1000,
                             n_chains = 4, seed = 3, show_progress = false)
        _sc_in_mode(ch, 4.9; tol = 1.5)
    end
end
