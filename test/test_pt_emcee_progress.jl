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

@testset "pt_emcee readout: ladder and cold log-posterior" begin
    s = Nereus._pt_swap_summary([40.0, 9.0, 30.0], [100, 100, 100])
    @test s.rates == [0.4, 0.09, 0.3] && s.worst == 2
    @test s.mean ≈ 0.79 / 3 && s.min == 0.09
    @test Nereus._pt_swap_str(s) == "0.263/0.090(2-3)"
    @test Nereus._pt_swap_str(Nereus._pt_swap_summary(Float64[], Int[])) == "—"

    # Walkers move on log(x + s) for a log-uniform prior; the readout is the
    # :lp of the recorded draws, so that Jacobian comes back off.
    state = reshape([log(2.0), log(4.0), 0.5, 0.7], 1, 2, 2)   # (rung, walker, dim)
    logπ = [-1.0 -2.0]; logL = [10.0 20.0]
    med, mx = Nereus._pt_cold_lp(state, logπ, logL,
                                 Union{Nothing,Float64}[0.0, nothing])
    @test mx ≈ -2.0 - log(4.0) + 20.0
    @test med ≈ ((-1.0 - log(2.0) + 10.0) + mx) / 2
    @test Nereus._pt_lp_str(74390.4) == "74390" && Nereus._pt_lp_str(-152.34) == "-152.3"
    @test Nereus._pt_count_str(950) == "950" && Nereus._pt_count_str(1_234_567) == "1.2M"
end
