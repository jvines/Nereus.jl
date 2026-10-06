# What sample_pt_emcee reports while it runs: the progress line and the
# progress log. A long fit is watched through these alone, so they must name
# the parameter that is holding it back, and they must not touch the chain.
using Test
using Nereus
using Printf
using Random
using Statistics

# A toy ensemble, (step, parameter, walker), where parameter 2 is deliberately
# slow: each walker drifts on a random walk from its own starting point, so
# the walkers disagree (high R-hat) and every draw repeats the last (low ESS).
# The other three are independent draws from one normal.
function _slow_cube(; n_step = 400, n_w = 16, rng = MersenneTwister(3))
    cube = randn(rng, n_step, 4, n_w)
    for w in 1:n_w
        x = 3.0 * randn(rng)
        for s in 1:n_step
            x += 0.05 * randn(rng)
            cube[s, 2, w] = x
        end
    end
    return cube
end

@testset "pt_emcee readout: worst parameter by name" begin
    names = ["P_k1", "K_k1", "secosw_k1", "sesinw_k1"]
    cube = _slow_cube()
    d = Nereus._pt_cube_diag(cube)
    @test d.worst_rhat == 2 && d.worst_tail == 2
    @test d.max_rhat == maximum(d.rhat) && d.max_rhat > 1.5
    @test d.min_tail == minimum(d.tail_ess)
    @test all(<(1.05), d.rhat[[1, 3, 4]])
    rs, es = Nereus._pt_diag_strings(d, names)
    @test endswith(rs, "(K_k1)") && endswith(es, "(K_k1)")
    @test startswith(rs, @sprintf("%.3f/%.3f", d.mean_rhat, d.max_rhat))

    # The same draws through the flat buffer the sampler keeps (each step's
    # walkers stacked), and only the science columns: the position returned is
    # within `science_cols`, so a nuisance column in front shifts nothing.
    n_step, _, n_w = size(cube)
    flat = Matrix{Float64}(undef, n_step * n_w, 5)
    for s in 1:n_step, w in 1:n_w
        flat[(s - 1) * n_w + w, 1] = randn()               # a nuisance
        flat[(s - 1) * n_w + w, 2:5] .= cube[s, :, w]
    end
    d2 = Nereus._pt_convergence(flat, n_step * n_w, n_w, [2, 3, 4, 5])
    @test d2.worst_rhat == 2 && d2.rhat ≈ d.rhat
    @test Nereus._pt_convergence(flat, 3 * n_w, n_w, [2, 3, 4, 5]) === nothing
end
