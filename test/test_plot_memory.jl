# The plotting stage's memory (src/plotting). A figure draws a bounded number of
# draws -- a thinned corner, 1500 points a walker, a 60,000-draw band -- and what
# it allocates beyond those must not grow with the chain. A 1M-draw joint fit's
# plots peaked gigabytes above the fit before this was bounded: whole-column and
# whole-sub-chain copies in every fold, every draw of every corner panel held by
# its hexbin, and a boxed float per grid point per band draw.
using Test, Nereus, MCMCChains, Random
using CairoMakie

const _PLOTMEM_MAKIE = Nereus.CairoMakie.Makie

# A one-planet, two-instrument RV fit, and a chain of `n` draws (`n_walk` walkers)
# scattered around its truth, with an `lp` that peaks there.
function _plotmem_fixture()
    rng = MersenneTwister(11)
    n = 40
    t = sort(300.0 .* rand(rng, n))
    inst = [isodd(i) ? 1 : 2 for i in 1:n]
    data = Data(; t_rv = t, rv = zeros(n), rv_err = fill(2.0, n), rv_inst = inst)
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                       instruments = InstrumentConfig(rv = ["A", "B"]),
                       data = data, M_s = 1.0)
    vals = Dict("P_k1" => 11.3, "K_k1" => 25.0, "sesinw_k1" => 0.2,
                "secosw_k1" => -0.1, "Mo_k1" => 0.7, "gamma_A" => 0.0,
                "gamma_B" => 3.0, "sigma_A" => 2.0, "sigma_B" => 2.0)
    theta = Theta{Float64}(params)
    for (k, v) in vals
        set_param!(theta, k, v)
    end
    data.rv .= rv_predictions(theta, data)[1] .+ 2.0 .* randn(rng, n)
    return params, data, vals
end

function _plotmem_chain(params, vals, n, n_walk; seed = 5)
    rng = MersenneTwister(seed)
    nm = params.layout.unfrozen_names
    n_iter = n ÷ n_walk
    arr = zeros(n_iter, length(nm) + 1, n_walk)
    for w in 1:n_walk, i in 1:n_iter
        z2 = 0.0
        for (j, x) in enumerate(nm)
            z = randn(rng)
            z2 += z^2
            arr[i, j, w] = vals[x] + 0.01 * max(abs(vals[x]), 0.1) * z
        end
        arr[i, end, w] = -0.5 * z2
    end
    return Chains(arr, vcat(Symbol.(nm), [:lp]))
end

_plotmem_stage(chains, params, data) = Nereus._make_plots(
    Dict("output" => Dict("plots" => ["auto"], "show_progress" => false)),
    chains, params, data, mktempdir())

@testset "plotting memory is bounded" begin
    params, data, vals = _plotmem_fixture()

    @testset "the corner's hexbins hold one point per hexagon" begin
        # `_hexbin_reduce` stands in for every draw: Makie's hexbin of the reduced
        # points is the hexbin of the draws, hexagon for hexagon, in the same order.
        rng = MersenneTwister(3)
        for _ in 1:20
            x = randn(rng, 20_000) .* 10.0^rand(rng, -3:3)
            y = randn(rng, 20_000) .^ 3
            bins = (rand(rng, 7:40), rand(rng, 7:40))
            fig = Figure()
            full = hexbin!(Axis(fig[1, 1]), x, y; bins, colorscale = log10)
            pts, w = Nereus._hexbin_reduce(x, y, bins)
            red = hexbin!(Axis(fig[1, 2]), pts; weights = w, bins, colorscale = log10)
            @test red.points[] == full.points[]
            @test red.count_hex[] == full.count_hex[]
            @test _PLOTMEM_MAKIE.data_limits(red) == _PLOTMEM_MAKIE.data_limits(full)
            @test length(pts) == length(full.points[]) + 4
        end
        @test Nereus._hexbin_reduce([1.0, NaN], [1.0, 2.0], (10, 10)) === nothing

        # The corner's contours are PairPlots' own, from bandwidths computed once
        # per column instead of twice per panel.
        PP = Nereus.PairPlots
        nt = (; a = randn(rng, 20_000), b = randn(rng, 20_000) .^ 3,
                c = randn(rng, 20_000) .+ 0.1 .* (1:20_000) ./ 20_000)
        contour_lines(fig) = [p[1][] for c in fig.content if c isa Axis
                              for p in c.scene.plots if p isa _PLOTMEM_MAKIE.Lines]
        ref = contour_lines(PP.pairplot(nt => (PP.Contour(color = :black),)))
        @test !isempty(ref)
        @test contour_lines(PP.pairplot(nt => (Nereus._CornerContour(color = :black),))) == ref

        ch = _plotmem_chain(params, vals, 40_000, 4)
        fig = plot_corner(ch, params)
        held = [length(p[1][]) for c in fig.content if c isa Axis
                for p in c.scene.plots if p isa _PLOTMEM_MAKIE.Hexbin]
        @test !isempty(held)
        @test maximum(held) < 2_000                  # hexagons, not 40,000 draws
    end

    @testset "a band draw allocates no boxed floats" begin
        # The planet's elements came out of an abstractly typed block and were
        # boxed at every grid point: 230 bytes per point per draw, 14 GB of
        # garbage for a 60,000-draw, 1000-point fold band.
        ch = _plotmem_chain(params, vals, 4_000, 4)
        ph = collect(range(-0.5, 0.5; length = 1000))
        band() = Nereus.compute_ci_bands(ch, params, data, ph; planet = 1,
                                         n_draws = 2_000, fold_phase = ph)
        band()
        bytes = @allocated band()
        @test bytes < 4 * (8 * 2_000 * 1000)         # ~ the 16 MB band matrix
    end

    @testset "a fold band from a pool equals the band from its sub-chain" begin
        # The folds used to copy their credible pool out as a sub-chain of every
        # column; `pool` passes its indices instead, and draws the same band.
        ch = _plotmem_chain(params, vals, 8_000, 4)
        pool = collect(1:3:8_000)
        sub = Chains(reduce(hcat, [Nereus._draws(ch, s)[pool] for s in names(ch)]),
                     names(ch))
        ph = collect(range(-0.5, 0.5; length = 200))
        Random.seed!(9)
        a = Nereus.compute_ci_bands(sub, params, data, ph; planet = 1, n_draws = 1_500,
                                    bf_cutoff = Inf, fold_phase = ph)
        Random.seed!(9)
        b = Nereus.compute_ci_bands(ch, params, data, ph; planet = 1, n_draws = 1_500,
                                    pool, fold_phase = ph)
        @test a == b
    end

    @testset "a large chain's figures collect once enough is allocated" begin
        # `_save_plot` collects (when `_make_plots` turns it on) only once
        # `_RELEASE_EVERY` bytes have been allocated since the last collection.
        at = Nereus._RELEASED_AT[]
        try
            t0 = Base.gc_bytes()
            Nereus._RELEASED_AT[] = t0
            Nereus._release_memory_if_due()
            @test Nereus._RELEASED_AT[] == t0             # not due: no collection
            due = Base.gc_bytes() - Nereus._RELEASE_EVERY - 1
            Nereus._RELEASED_AT[] = due
            Nereus._release_memory_if_due()
            @test Nereus._RELEASED_AT[] > due             # due: collected
        finally
            Nereus._RELEASED_AT[] = at
        end
        @test Nereus._large_chain(_plotmem_chain(params, vals, 2_000_000, 8))
        @test !Nereus._large_chain(_plotmem_chain(params, vals, 4_000, 8))
        Nereus._RELEASE_MEMORY[] = false
        _plotmem_stage(_plotmem_chain(params, vals, 4_000, 8), params, data)
        @test Nereus._RELEASE_MEMORY[] == false        # restored after the stage
    end

    @testset "the plotting stage allocates little per draw" begin
        _plotmem_stage(_plotmem_chain(params, vals, 4_000, 8), params, data)   # compile
        n1, n2 = 100_000, 300_000                    # both above the 60,000-draw band
        c1 = _plotmem_chain(params, vals, n1, 8)
        c2 = _plotmem_chain(params, vals, n2, 8)
        a1 = @allocated _plotmem_stage(c1, params, data)
        a2 = @allocated _plotmem_stage(c2, params, data)
        per_draw = (a2 - a1) / (n2 - n1)
        row = 8 * size(c2, 2)                        # one draw of every column
        # Measured: 9 rows a draw, most of it the corner's KDEs over its thinned
        # draws. Before it was bounded the stage took 46: a copy of every column
        # in each figure, a sub-chain per fold, two effective sample sizes per
        # corner panel.
        @test per_draw < 15 * row
    end
end
