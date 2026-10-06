# plot_trace thins every line to `max_points` and rasterizes multi-walker
# figures (src/plotting/diagnostics.jl): a pt_emcee run of 100 walkers x 30000
# steps otherwise draws 3M vertices per figure.
using Test
using Nereus
using MCMCChains
using Random
using CairoMakie

@testset "plot_trace thins and rasterizes" begin
    rng = MersenneTwister(3)
    tg = build_target(planets = (b = (P = LogUniformPrior(4.0, 4.5),
                                      K = LogUniformPrior(10.0, 90.0)),),
        rv = (SIM = (data = (t = collect(1.0:30.0), rv = randn(rng, 30),
                             rv_err = fill(1.5, 30)),
                     sigma = LogUniformPrior(0.5, 10.0)),))
    nm = tg.params.layout.unfrozen_names
    n_iter, n_walk = 5000, 8
    figs = plot_trace(Chains(randn(rng, n_iter, length(nm), n_walk), Symbol.(nm)),
                      tg.params; output = mktempdir(), max_points = 100)
    @test Set(keys(figs)) == Set(nm)
    ls = filter(p -> p isa Lines, figs[nm[1]].content[1].scene.plots)
    @test length(ls) == n_walk                       # every walker is drawn
    pts = ls[1][1][]
    @test length(pts) <= 100
    @test first(pts)[1] == 1 && last(pts)[1] > n_iter - n_iter / 100   # true iterations
    @test all(p -> p.rasterize[] == 2, ls)

    # One chain: thinned, not rasterized.
    one = plot_trace(Chains(randn(rng, n_iter, length(nm), 1), Symbol.(nm)),
                     tg.params; max_points = 100)
    l1 = only(filter(p -> p isa Lines, one[nm[1]].content[1].scene.plots))
    @test length(l1[1][]) <= 100
end
