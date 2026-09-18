# The photometry log-likelihood must be bit-reproducible.
#
# It used to accumulate into `local_total[Threads.threadid()]` under a
# :dynamic `Threads.@threads`, which both lost updates (a task can migrate
# between the read and the write of `+=`) and varied the floating-point
# summation order run to run. The resulting ~1e-3 log-unit wobble is far below
# MCMC noise for a fixed-dimension fit, but it is NOT below the level that
# flips trans-dimensional birth/death decisions: occupancies were not
# reproducible at fixed seed because of it.
#
# The reduction now sums FIXED-SIZE chunks into per-chunk slots and combines
# them in index order, so the result depends on neither the task schedule nor
# nthreads(). Verified manually at 1/2/4/8 threads: identical to all 17
# significant digits.
using Test
using Nereus
using Random

@testset "photometry log-likelihood is deterministic" begin
    Random.seed!(11)
    n = 20_000                       # spans several _PHOT_REDUCE_CHUNK blocks
    @test n > 2 * Nereus._PHOT_REDUCE_CHUNK
    t = collect(range(0.0, 20.0; length = n))
    P0, T0, dur, dep = 3.11, 1.2, 0.11, 0.006
    ph = @. abs(mod(t - T0 + P0 / 2, P0) - P0 / 2)
    flux = 1.0 .+ 2.5e-4 .* randn(n); flux[ph .< dur / 2] .-= dep
    ferr = fill(2.5e-4, n)

    target = build_target(
        planets = (b = (P  = UniformPrior(P0 - 0.05, P0 + 0.05),
                        Tc = UniformPrior(T0 - 0.05, T0 + 0.05),
                        b  = UniformPrior(0.0, 0.9),
                        rr = UniformPrior(0.01, 0.2),
                        sesinw = UniformPrior(-1.0, 1.0),
                        secosw = UniformPrior(-1.0, 1.0)),),
        phot = (TESS = (data = (t = t, flux = flux, flux_err = ferr),),),
        M_s = 1.0, R_s = 1.0,
    )
    th = Nereus.Theta{Float64}(target.params)
    truth = Dict("P_k1" => P0, "Tc_k1" => T0, "b_k1" => 0.3, "rr_k1" => 0.078)
    for nm in target.params.layout.unfrozen_names
        Nereus.set_param!(th, nm, get(truth, nm, 0.0))
    end

    vals = [Nereus.transit_log_likelihood(th, target.data) for _ in 1:16]
    @test isfinite(vals[1])
    # BIT-identical, not approximately equal. `≈` would pass on the very bug
    # this guards against.
    @test length(unique(vals)) == 1
    @test all(==(vals[1]), vals)

    # The partition must not depend on nthreads(), or two machines disagree.
    @test Nereus._PHOT_REDUCE_CHUNK isa Integer
    @test Nereus._PHOT_REDUCE_CHUNK > 0
end
