# plot_corner plots at most `max_draws` draws (every k-th) and then pins each
# panel to the full sample's range (src/plotting/diagnostics.jl): a corner of
# 3M draws took 37 minutes on the NGTS-33 global fit.
using Test
using Nereus
using MCMCChains
using Random
using CairoMakie

@testset "plot_corner thins with full-sample limits" begin
    rng = MersenneTwister(4)
    tg = build_target(planets = (b = (P = LogUniformPrior(4.0, 4.5),
                                      K = LogUniformPrior(10.0, 90.0)),),
        rv = (SIM = (data = (t = collect(1.0:30.0), rv = randn(rng, 30),
                             rv_err = fill(1.5, 30)),
                     sigma = LogUniformPrior(0.5, 10.0)),))
    n_iter, n_walk = 10_000, 2
    cube = randn(rng, n_iter, 2, n_walk)
    cube[2, 1, 1] = 40.0            # an outlier the every-20th subset skips
    ch = Chains(cube, [:P_k1, :K_k1])
    p = ["P_k1", "K_k1"]

    @test Nereus._corner_lims([0.0, 10.0]) == (; lims = (; low = -0.5, high = 10.5))
    @test Nereus._corner_lims([1.0, 1.0]) == (;)

    xmax(fig) = (save(tempname() * ".png", fig);
                 maximum(a -> a.finallimits[].origin[1] + a.finallimits[].widths[1],
                         filter(c -> c isa Axis, fig.content)))
    thinned = plot_corner(ch, tg.params; params_to_plot = p, max_draws = 1000)
    @test xmax(thinned) ≈ 40.0 + 0.05 * (40.0 - minimum(cube[:, 1, :]))   # pinned
    full = plot_corner(ch, tg.params; params_to_plot = p, max_draws = typemax(Int))
    @test xmax(full) >= 40.0                                              # autolimits
end
