# pt_emcee on an easy target with default settings.
#
# Reported: a single-planet RV fit with the default settings failed Nereus's own
# health check (R-hat 1.4-1.9), with walkers stranded across the 0.1-3000 d
# period prior. Three causes, each tested here:
#   - run_job defaulted to `Tp` over the whole baseline: baseline/P copies of
#     the orbit that an ensemble cannot mix across;
#   - walkers moved in linear P over a log-uniform prior spanning 4.5 decades;
#   - a stretch move cannot bring a walker from the plateau into a narrow mode,
#     so a found mode filled its cold rung at ~5 walkers per 100 steps.

using Test, Nereus, Random, Statistics, MCMCChains
using LinearAlgebra: svdvals
using Nereus: _pt_prune_stranded!, _warn_repeating_time_window
include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))

# One rung of `n_mode` walkers in a Gaussian mode (σ = `width`, centred on 0.5)
# and `n_flat` spread over the unit box `gap` nats of log-likelihood below it.
# Uniform prior on [0, 1]^d, so log π = 0 and ∫ π = 1.
function _rung(; d = 3, n_mode = 70, n_flat = 30, gap = 60.0, seed = 1, width = 0.01)
    rng = MersenneTwister(seed)
    n = n_mode + n_flat
    state = Array{Float64,3}(undef, 1, n, d)
    logL = Matrix{Float64}(undef, 1, n)
    for w in 1:n_mode
        x = 0.5 .+ width .* randn(rng, d)
        state[1, w, :] = x
        logL[1, w] = _mode_logL(x, gap, width)
    end
    for w in (n_mode + 1):n
        state[1, w, :] = rand(rng, d)
        logL[1, w] = -gap
    end
    return state, zeros(1, n), logL
end

# The density `_rung` samples, for the pruning's own evaluations: the mode on a
# plateau `gap` nats down, zero outside the unit box.
_mode_logL(x, gap, width) = max(-0.5 * sum(abs2, (x .- 0.5) ./ width), -gap)
_mode_eval(gap; width = 0.01) =
    y -> all(v -> 0 <= v <= 1, y) ? (0.0, _mode_logL(y, gap, width)) : (-Inf, -Inf)
# The pruning as the samplers run it: `n` passes sharing one record, the walkers
# unchanged in between but for what it moves. Returns each pass's count on rung
# 1. The first pass only records: a walker moves once it has been stranded at
# two passes running and has not climbed in between.
function _passes!(state, lπ, lL, βs; n = 2, kw...)
    mem = Nereus.PruneMemory(size(state)...)
    return [_pt_prune_stranded!(state, lπ, lL, βs, MersenneTwister(1 + i);
                                memory = mem, kw...)[1] for i in 1:n]
end
# A record of each rung's median log L at the previous prune that says the rung
# has stopped climbing: the same median as now.
_settled(lL) = [median(filter(isfinite, lL[t, :])) for t in axes(lL, 1)]
# Passes until one moves nothing. No pass moves more walkers than the rung's
# survivors (A, and the climbed top), so none moves more than half the rung.
function _settle!(f; n = 16)
    h = f(n)
    @test h[1] == 0
    @test maximum(h) <= 50
    @test h[end] == 0
    return h
end

@testset "pt_emcee: stranded walkers and default convergence" begin

    @testset "prune moves a plateau that provably weighs nothing" begin
        # Z_mode ≈ (2π)^{3/2} 0.01³ → log ≈ -11.1. The plateau's whole region
        # weighs at most e^-60, so it is 49 nats below: moved.
        state, lπ, lL = _rung(gap = 60.0)
        before = copy(state)
        # Nothing on the first pass, which only records; and nothing without a
        # record at all.
        @test _pt_prune_stranded!(copy(state), copy(lπ), copy(lL), [1.0],
                                  MersenneTwister(2)) == [0]
        @test _passes!(state, lπ, lL, [1.0]) == [0, 30]
        @test state[1, 1:70, :] == before[1, 1:70, :]          # mode untouched
        mode_rows = [before[1, w, :] for w in 1:70]
        @test all(w -> state[1, w, :] in mode_rows, 71:100)  # copies of mode walkers
        @test all(lL[1, 71:100] .> -60)
    end

    @testset "prune leaves a plateau that could carry mass" begin
        # 18 nats down is past the Gaussian allowance (~15 for d = 3), but the
        # plateau fills the whole prior and the mode's Occam factor is 11 nats:
        # the plateau may weigh e^-7 of the mode. Not provably nothing: kept.
        state, lπ, lL = _rung(gap = 18.0)
        before = copy(state)
        @test _passes!(state, lπ, lL, [1.0]) == [0, 0]
        @test state == before
    end

    @testset "tempering shrinks the gap, so hot rungs keep their plateau" begin
        state, lπ, lL = _rung(gap = 60.0)
        @test _passes!(state, lπ, lL, [0.1]) == [0, 0]
    end

    @testset "a walker moves only once stranded at two passes running" begin
        # At the previous pass the 30 were in the mode; they have just fallen
        # to the plateau. Only at the next pass are they stranded.
        state, lπ, lL = _rung(gap = 60.0)
        mem = Nereus.PruneMemory(size(state)...)
        s0, l0 = copy(state), copy(lL)
        s0[1, 71:100, :] = s0[1, 1:30, :]; l0[1, 71:100] = l0[1, 1:30]
        _pt_prune_stranded!(s0, copy(lπ), l0, [1.0], MersenneTwister(1); memory = mem)
        @test _pt_prune_stranded!(state, lπ, lL, [1.0], MersenneTwister(2);
                                  memory = mem) == [0]
        @test _pt_prune_stranded!(state, lπ, lL, [1.0], MersenneTwister(3);
                                  memory = mem) == [30]
    end

    @testset "a walker climbing fast enough to clear the cut is not moved" begin
        # The cut sits ~21 nats down. Walker 71 climbed from 120 to 60 nats
        # down since the previous pass: as fast again, it clears the cut by
        # the next, and stays. Walker 72 climbed from 70 to 60: it would take
        # four more passes, and may never; it goes with the 28 that sat still.
        state, lπ, lL = _rung(gap = 60.0)
        mem = Nereus.PruneMemory(size(state)...)
        l0 = copy(lL); l0[1, 71] = -120.0; l0[1, 72] = -70.0
        _pt_prune_stranded!(copy(state), copy(lπ), l0, [1.0], MersenneTwister(1);
                            memory = mem)
        before = copy(state)
        @test _pt_prune_stranded!(state, lπ, lL, [1.0], MersenneTwister(2);
                                  memory = mem) == [29]
        @test state[1, 71, :] == before[1, 71, :]
        @test state[1, 72, :] != before[1, 72, :]
    end

    @testset "a separate mode the bound keeps: moved by its own mass" begin
        # 90 walkers in the main mode (σ = 0.001, log Z ≈ -18) and 10 in a
        # second, 20 nats down, centred on 0.2. Its walkers sit above the cut
        # (log Z_A - 10 ≈ -28): the bound gives their level of L the whole
        # prior to fill. At σ = 0.001 the second mode weighs e^-38, 20 nats
        # under the first, and its 10 walkers move. At σ = 0.05 it weighs
        # e^-26, which may matter, and they stay.
        function twomode(wl; d = 3, w = 0.001, depth = 20.0)
            L(x) = all(v -> 0 <= v <= 1, x) ?
                   max(-0.5 * sum(abs2, (x .- 0.5) ./ w),
                       -depth - 0.5 * sum(abs2, (x .- 0.2) ./ wl), -1000.0) : -Inf
            rng = MersenneTwister(3)
            state = Array{Float64,3}(undef, 1, 100, d)
            lL = zeros(1, 100)
            for k in 1:100
                x = k <= 90 ? 0.5 .+ w .* randn(rng, d) : 0.2 .+ wl .* randn(rng, d)
                state[1, k, :] = x
                lL[1, k] = L(x)
            end
            ev = y -> (r = L(y); isfinite(r) ? (0.0, r) : (-Inf, -Inf))
            return state, zeros(1, 100), lL, ev
        end
        state, lπ, lL, ev = twomode(0.001)
        @test all(lL[1, 91:100] .> -28)
        @test _passes!(state, lπ, lL, [1.0]; n = 3, evalf = ev) == [0, 10, 0]
        @test all(abs.(state[1, :, :] .- 0.5) .< 0.01)
        state, lπ, lL, ev = twomode(0.05)
        before = copy(state)
        @test _passes!(state, lπ, lL, [1.0]; n = 3, evalf = ev) == [0, 0, 0]
        @test state == before
    end

    @testset "a walker on the mode's own flank, not climbing, is moved" begin
        # Walker 71 sits 6σ out on the mode's flank (log L -18): outside the
        # allowance (15.3) and the mode as measured at its top, above the cut
        # (-21). Its climb ends in the mode itself, so it moves with the plateau.
        state, lπ, lL = _rung(gap = 60.0)
        x = 0.5 .+ (6 * 0.01 / sqrt(3)) .* ones(3)
        state[1, 71, :] = x
        lL[1, 71] = _mode_logL(x, 60.0, 0.01)
        @test lL[1, 71] ≈ -18.0
        s0, l0 = copy(state), copy(lL)
        @test _passes!(state, lπ, lL, [1.0]; evalf = _mode_eval(60.0)) == [0, 30]
        @test all(abs.(state[1, :, :] .- 0.5) .< 0.05)
        # Without an evaluator it cannot be followed, and stays.
        state, lπ, lL = copy(s0), zeros(1, 100), copy(l0)
        @test _passes!(state, lπ, lL, [1.0]) == [0, 29]
        @test state[1, 71, :] == x
        # Climbing back fast enough to reach the allowance by the next pass
        # (from -40 to -18 since the last), it stays too.
        state, lπ, lL = copy(s0), zeros(1, 100), copy(l0)
        mem = Nereus.PruneMemory(size(state)...)
        lp = copy(l0); lp[1, 71] = -40.0
        _pt_prune_stranded!(copy(state), copy(lπ), lp, [1.0], MersenneTwister(1);
                            memory = mem, evalf = _mode_eval(60.0))
        @test _pt_prune_stranded!(state, lπ, lL, [1.0], MersenneTwister(2); memory = mem,
                                  evalf = _mode_eval(60.0)) == [29]
        @test state[1, 71, :] == x
    end

    @testset "a mode too new for a covariance: volume from the density" begin
        # Four walkers have found the mode, fewer than the d + 2 a covariance
        # needs. Gaia-4 b joint (d = 14) kept 97 of 100 cold walkers 200 nats
        # down at every prune for this reason. Without an evaluator the rung is
        # left alone, as before.
        state, lπ, lL = _rung(n_mode = 4, n_flat = 96, gap = 60.0)
        @test _passes!(copy(state), copy(lπ), copy(lL), [1.0]) == [0, 0]

        # Nor while the rung is still climbing: its best walker may be only the
        # fastest climber, and the rest would be herded onto it. Here the rung's
        # median log L rose 2 nats since the previous pass (d/2 = 1.5).
        mem = Nereus.PruneMemory(size(state)...)
        _pt_prune_stranded!(copy(state), copy(lπ), lL .- 2.0, [1.0], MersenneTwister(1);
                            evalf = _mode_eval(60.0), memory = mem)
        @test _pt_prune_stranded!(copy(state), copy(lπ), copy(lL), [1.0],
                                  MersenneTwister(2); evalf = _mode_eval(60.0),
                                  memory = mem) == [0]

        # Settled, with an evaluator: the drops along each axis about the best
        # walker give widths of at most σ = 0.01, the plateau provably weighs
        # nothing, and all 96 move: at each pass onto the survivors, the upper
        # half of A and the top of the mode, at most as many as there are in A
        # and the top, so the mode at most doubles per pass.
        before = copy(state)
        h = _settle!(n -> _passes!(state, lπ, lL, [1.0]; n, evalf = _mode_eval(60.0)))
        @test sum(h) == 96
        @test h[2] <= 4 + 1
        @test state[1, 1:4, :] == before[1, 1:4, :]
        # Drawn about the survivors and evaluated, not copied: the ensemble
        # keeps full rank, which copies of four walkers would not.
        X = state[1, :, :]
        @test length(unique(eachrow(X))) == 100
        @test minimum(svdvals(X .- sum(X; dims = 1) ./ 100)) > 1e-3
        @test all(abs.(X[5:end, :] .- 0.5) .< 0.1)
        @test lL[1, 5:end] ≈ [_mode_logL(X[w, :], 60.0, 0.01) for w in 5:100]
        @test all(lL[1, 5:end] .> -30)

        # The same guarantee as with a covariance: 18 nats down could still
        # carry mass, so it stays.
        state, lπ, lL = _rung(n_mode = 4, n_flat = 96, gap = 18.0)
        @test _passes!(state, lπ, lL, [1.0]; n = 3, evalf = _mode_eval(18.0)) == [0, 0, 0]

        # A spike is not a mode: at σ = 1e-6 the volume is 41 nats smaller and a
        # plateau 40 nats down may outweigh it. At σ = 0.01 it may not.
        for (width, n) in ((1e-6, 0), (0.01, 96))
            state, lπ, lL = _rung(n_mode = 4, n_flat = 96, gap = 40.0, width = width)
            h = _settle!(k -> _passes!(state, lπ, lL, [1.0]; n = k,
                                       evalf = _mode_eval(40.0; width)))
            @test sum(h) == n
        end
    end

    @testset "a lone walker on the flank of its mode climbs before it is measured" begin
        # One walker has found the mode, 7.5σ out on its flank (log L -28), and
        # 99 sit on a plateau 50 nats down. Measured where it landed, the bound
        # cannot clear the plateau: the drop along each axis is steep with the
        # gradient, so the widths come out ~0.2σ and the cut sits ~55 nats
        # down. At the top of the mode it sits 21 nats down. Gaia-4 b's lone
        # walkers sat 200-280 nats below their peak, and the plateau stayed.
        d, gap, width = 3, 50.0, 0.01
        state, lπ, lL = _rung(n_mode = 0, n_flat = 100, gap = gap)
        x0 = 0.5 .+ (7.5 * width / sqrt(d)) .* ones(d)
        state[1, 1, :] = x0
        lL[1, 1] = _mode_logL(x0, gap, width)
        ev = _mode_eval(gap)
        ℓat = x -> (r = ev(x); isfinite(r[1]) ? r[1] + r[2] : -Inf)
        h0 = [0.1 * Nereus._pt_spread(view(state, 1, :, j), x0[j], false) for j in 1:d]
        σ0 = Nereus._pt_axis_widths(ℓat, x0, lL[1, 1], h0)
        @test all(σ0 .< 0.3 * width)
        @test -gap >= lL[1, 1] + 0.5 * d * log(2π) + sum(log, σ0) - 10.0

        # The climb reaches the top, deterministically.
        yc, ℓc = Nereus._pt_climb(ℓat, x0, lL[1, 1], σ0)
        @test ℓc > -0.1
        @test maximum(abs.(yc .- 0.5)) < 0.1 * width
        @test Nereus._pt_climb(ℓat, x0, lL[1, 1], σ0) == (yc, ℓc)
        @test Nereus._pt_climb(ℓat, fill(0.5, d), 0.0, σ0) == (fill(0.5, d), 0.0)

        # So the whole rung moves there, the flank walker with it: one walker
        # at the first pass that can move any, about the top alone; then the
        # mode at most doubles per pass.
        h = _settle!(n -> _passes!(state, lπ, lL, [1.0]; n, evalf = ev))
        @test h[2] == 1
        @test sum(h) == 100
        X = state[1, :, :]
        @test all(abs.(X .- 0.5) .< 5 * width)
        @test lL[1, :] ≈ [_mode_logL(X[w, :], gap, width) for w in 1:100]
        @test all(lL[1, :] .> -20)
        @test minimum(svdvals(X .- sum(X; dims = 1) ./ 100)) > 1e-3

        # The trans-dim prune climbs too, and the 99 take the planet along.
        state, lπ, lL = _rung(n_mode = 0, n_flat = 100, gap = gap)
        state[1, 1, :] = x0
        lL[1, 1] = _mode_logL(x0, gap, width)
        tds = [Nereus.TransDimState(w == 1 ? 1 : 0, BitVector([w == 1]),
                                    BitVector(), BitVector([true]))
               for _ in 1:1, w in 1:100]
        tev = (x, td) -> td.planet_active[1] ? ev(x) : (0.0, -gap)
        shift = Union{Nothing,Float64}[nothing, nothing, nothing]
        moved = Nereus._td_prune_stranded!(state, tds, lπ, lL, [1.0], shift,
                                           [1, 1, 1], 0, MersenneTwister(2);
                                           evalf = tev, med_logL = _settled(lL))
        @test moved == [100]
        @test all(w -> tds[1, w].planet_active[1], 1:100)
        @test all(abs.(state[1, :, :] .- 0.5) .< 5 * width)
        @test all(lL[1, :] .> -20)
    end

    @testset "the circular re-cut runs with every pruning pass" begin
        # Halfway, with every prune (every 50 steps through 3/4 of burn-in), and
        # at the end.
        @test [s for s in 1:1500 if Nereus._pt_recut_due(s, 1500)] == [50:50:1100; 1500]
        @test [s for s in 1:130 if Nereus._pt_recut_due(s, 130)] == [50, 65, 130]
        # What it catches: a mode found after the halfway cut, across the seam.
        # Gaia-4 b (seed 6) found Mo ≈ 0.6 approaching from 6.1, and its cold
        # rung piled up against the wall at 2π until the end of burn-in. The
        # next cut puts the seam where the pile and the mode beyond it are not.
        pile = collect(range(6.1, 2π - 1e-4; length = 100))
        @test 0.6 < mod(Nereus.circular_cut(pile, 0.0), 2π) < 6.1
    end

    @testset "a mode across the seam of a circle is measured on the circle" begin
        # Dim 3 is a full-circle angle, U(0, 2π), and the mode sits on its seam:
        # half the walkers near 0, half near 2π. Read as a line, Σ_A spans the
        # window and overstates Z_A by ~4 nats; a plateau 19 nats down, which
        # may carry mass, would move.
        state, lπ, lL = _rung(gap = 19.0)
        state[1, :, 3] .= mod.(2π .* (state[1, :, 3] .- 0.5), 2π)
        lπ .= -log(2π)
        @test _passes!(copy(state), copy(lπ), copy(lL), [1.0]) == [0, 30]
        @test _passes!(state, lπ, lL, [1.0]; circ = [3]) == [0, 0]
    end

    @testset "trans-dim prune moves the model too" begin
        # 70 walkers with planet 1 active in the mode, 30 without it on the
        # no-planet plateau 60 nats down. All three dims belong to planet 1.
        state, lπ, lL = _rung(gap = 60.0)
        tds = [Nereus.TransDimState(w <= 70 ? 1 : 0, BitVector([w <= 70]),
                                    BitVector(), BitVector([true]))
               for _ in 1:1, w in 1:100]
        shift = Union{Nothing,Float64}[nothing, nothing, nothing]
        moved = Nereus._td_prune_stranded!(state, tds, lπ, lL, [1.0], shift,
                                           [1, 1, 1], 0, MersenneTwister(2))
        @test moved == [30]
        @test all(w -> tds[1, w].planet_active[1] && tds[1, w].n_planets_active == 1, 71:100)
        @test tds[1, 71].planet_active !== tds[1, 1].planet_active   # copied, not aliased

        # 18 nats down could still carry mass: left alone, model and all.
        state, lπ, lL = _rung(gap = 18.0)
        tds = [Nereus.TransDimState(w <= 70 ? 1 : 0, BitVector([w <= 70]),
                                    BitVector(), BitVector([true]))
               for _ in 1:1, w in 1:100]
        @test Nereus._td_prune_stranded!(state, tds, lπ, lL, [1.0], shift,
                                         [1, 1, 1], 0, MersenneTwister(2)) == [0]
        @test count(w -> tds[1, w].planet_active[1], 1:100) == 70

        # Four walkers have the planet: the volume comes from the density about
        # the best one, and the 96 without it are drawn there, planet and all.
        state, lπ, lL = _rung(n_mode = 4, n_flat = 96, gap = 60.0)
        tds = [Nereus.TransDimState(w <= 4 ? 1 : 0, BitVector([w <= 4]),
                                    BitVector(), BitVector([true]))
               for _ in 1:1, w in 1:100]
        ev = (x, td) -> td.planet_active[1] ? _mode_eval(60.0)(x) : (0.0, -60.0)
        @test Nereus._td_prune_stranded!(state, tds, lπ, lL, [1.0], shift,
                                         [1, 1, 1], 0, MersenneTwister(2)) == [0]
        @test Nereus._td_prune_stranded!(state, tds, lπ, lL, [1.0], shift,
                                         [1, 1, 1], 0, MersenneTwister(2);
                                         evalf = ev) == [0]
        moved = Nereus._td_prune_stranded!(state, tds, lπ, lL, [1.0], shift,
                                           [1, 1, 1], 0, MersenneTwister(2);
                                           evalf = ev, med_logL = _settled(lL))
        @test moved == [96]
        @test all(w -> tds[1, w].planet_active[1], 1:100)
        @test length(unique(eachrow(state[1, :, :]))) == 100
        @test all(lL[1, 5:end] .> -30)
    end

    @testset "convergence_report: label switching is not non-convergence" begin
        # One planet, two exchangeable RV slots. Walkers 1-10 hold it in slot 2
        # for the whole run, the rest in slot 1 (births sort, deaths do not
        # compact). Flat chain, walker = fast index, as transdim_pt_emcee saves.
        rng = MersenneTwister(5)
        nw, ns = 100, 400
        n = nw * ns
        w_of(i) = mod1(i, nw)
        in2 = [w_of(i) <= 10 for i in 1:n]
        P = 12.3 .+ 0.001 .* randn(rng, n); K = 12.0 .+ 0.3 .* randn(rng, n)
        cols = Dict(
            :P_k1 => ifelse.(in2, 900.0, P), :K_k1 => ifelse.(in2, 3.0, K),
            :P_k2 => ifelse.(in2, P, 450.0), :K_k2 => ifelse.(in2, K, 7.0),
            :planet_active_1 => Float64.(.!in2), :planet_active_2 => Float64.(in2),
            :n_planets => ones(n))
        nm = [:P_k1, :K_k1, :P_k2, :K_k2, :n_planets, :planet_active_1, :planet_active_2]
        ch = Chains(hcat((cols[s] for s in nm)...), nm)
        params = Params(max_kplanet = 2, planet_modes = [RV_ONLY, RV_ONLY], stability = :none,
                        instruments = InstrumentConfig(rv = ["HARPS"]),
                        data = Data(; t_rv = collect(0.0:10.0:990.0),
                                    rv = zeros(100), rv_err = ones(100),
                                    rv_inst = ones(Int, 100)))
        r = Nereus.convergence_report(ch, nw; model_params = params, io = devnull)
        @test r.pass
        # The planet itself still has to mix: ten walkers stuck off-mode make
        # the canonical slot-1 period fail.
        cols[:P_k1][[i for i in 1:n if 50 <= w_of(i) <= 59]] .= 13.0
        ch2 = Chains(hcat((cols[s] for s in nm)...), nm)
        @test !Nereus.convergence_report(ch2, nw; model_params = params, io = devnull).pass
    end

    @testset "Tp/Tc windows wider than a period warn" begin
        pri = Dict("P_k1" => LogUniformPrior(0.1, 3000.0),
                   "Tp_k1" => UniformPrior(0.0, 929.0))
        @test_logs (:warn, r"identical copies") _warn_repeating_time_window(pri, 1, :Tp)
        pri["Tp_k1"] = UniformPrior(0.0, 0.05)
        @test_logs _warn_repeating_time_window(pri, 1, :Tp)
        @test_logs _warn_repeating_time_window(pri, 1, :Mo)
    end

    # The reported symptom, end to end: run_job with no parametrization and no
    # sampler settings, on an easy 12.3 d planet over a 1000 d baseline.
    cfg = _easy_rv_cfg()   # test/fixtures/easy_rv_target.jl

    @testset "run_job defaults converge on an easy target" begin
        res = Nereus.run_job(cfg)
        s = res isa AbstractDict ? res : res.summary
        checks = s["fit_health"]["checks"]
        @test checks["convergence"]["status"] == "ok"
        @test checks["multimodality"]["status"] == "ok"
        # The reported false positive: P = 12.3 d on LogUniform(0.1, 3000).
        @test !occursin("P_k1", checks["prior_rail"]["message"])
        # sigma_HARPS sits at 0 here (the errors are exact): a jitter can be
        # zero, and fit_health must not cry FAIL over it.
        @test checks["prior_rail"]["status"] == "ok"
        @test s["fit_health"]["overall"] == "ok"
        ch, _ = Nereus.load_chains(joinpath(cfg["output_dir"], "chains.nc"))
        @test :phic_k1 in names(ch)                     # the default is the transit phase now
        Pd = vec(Array(ch[:P_k1]))
        @test mean(abs.(Pd .- 12.3) ./ 12.3 .< 0.01) == 1.0   # no stranded draws
    end

    # The transdim_pt_emcee defaults test lives in
    # test_transdim_pt_emcee_defaults.jl, so the two full fits run in parallel.

    @testset "draws and :lp are recorded in x, not on the move scale" begin
        data, irv, ipm, ias = Nereus._build_data(cfg["data"])
        _, target, _ = Nereus._build_model(cfg, data, Nereus._build_star(Dict()),
                                           irv, ipm, ias)
        r = sample_pt_emcee(target, data; n_temps = 6, n_walkers = 30,
                            n_steps = 60, n_burnin = 30, seed = 3,
                            show_progress = false)
        nms = Symbol.(target.params.layout.unfrozen_names)
        X = hcat([vec(Array(r.chains[p])) for p in nms]...)
        lp = vec(Array(r.chains[:lp]))
        @test all(X[:, findfirst(==(:P_k1), nms)] .>= 0.1)
        for i in round.(Int, range(1, size(X, 1); length = 20))
            @test lp[i] ≈ Nereus.logdensity_bounded(target, X[i, :]) rtol = 1e-9
        end
    end
end
