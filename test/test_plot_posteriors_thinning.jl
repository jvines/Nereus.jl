# plot_posteriors draws at most `max_points` samples per panel: unthinned, a
# pt_emcee run of 100 walkers x 47000 steps took 25 minutes on this one figure set.
using Test
using Nereus
using MCMCChains
using Random
using CairoMakie

@testset "plot_posteriors thins" begin
    rng = MersenneTwister(5)
    tg = build_target(planets = (b = (P = LogUniformPrior(4.0, 4.5),
                                      K = LogUniformPrior(10.0, 90.0)),),
        rv = (SIM = (data = (t = collect(1.0:30.0), rv = randn(rng, 30),
                             rv_err = fill(1.5, 30)),
                     sigma = LogUniformPrior(0.5, 10.0)),))
    nm = tg.params.layout.unfrozen_names
    n_iter, n_walk = 5000, 8
    ch = Chains(randn(rng, n_iter, length(nm) + 1, n_walk), [Symbol.(nm); :lp])
    figs = plot_posteriors(ch, tg.params; output = mktempdir(), max_points = 1000)
    @test Set(keys(figs)) == Set(nm)
    sc = only(filter(p -> p isa Scatter, figs[nm[1]].content[1].scene.plots))
    @test 500 <= length(sc[1][]) <= 1000
end
