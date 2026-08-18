# assess_fit — post-fit "silent-wrong" guard. Verifies the three real
# failure modes the validation matrix caught all get flagged:
#   (a) clean unimodal Gaussian chain          => overall :ok
#   (b) bimodal chain (two disjoint modes)     => :fail w/ multimodality msg
#   (c) chain railed against a prior bound      => :fail w/ rail msg
# plus log-posterior sanity, ensemble mode, and the MAP-consistency
# convenience method.

using Nereus, Random, Test, MCMCChains
using Random: MersenneTwister

# Locate a check by name in a FitHealthReport.
_find_check(r, name) = r.checks[findfirst(c -> c.name === name, r.checks)]

@testset "assess_fit — silent-wrong guard" begin

    @testset "(a) clean unimodal Gaussian => :ok" begin
        rng = MersenneTwister(7)
        # 4 well-mixed chains around the same mode for two params.
        arr = zeros(Float64, 2000, 2, 4)
        for k in 1:4
            arr[:, 1, k] .= 10.0 .+ 1.0 .* randn(rng, 2000)   # K ~ N(10,1)
            arr[:, 2, k] .= 4.14 .+ 0.01 .* randn(rng, 2000)  # P ~ N(4.14,0.01)
        end
        ch = Chains(arr, [:K, :P])
        r = assess_fit(ch)

        @test r isa FitHealthReport
        @test r.overall === :ok
        @test _find_check(r, :convergence).status === :ok
        @test _find_check(r, :multimodality).status === :ok
        # show() must not throw and should print the OK icon.
        io = IOBuffer()
        show(io, MIME"text/plain"(), r)
        @test occursin("✅", String(take!(io)))
    end

    @testset "(b) bimodal — two disjoint modes => :fail (multimodality)" begin
        rng = MersenneTwister(8)
        # Two chains frozen at -8, two at +8: the nuts failure mode.
        arr = zeros(Float64, 2000, 1, 4)
        arr[:, 1, 1] .= -8.0 .+ 0.3 .* randn(rng, 2000)
        arr[:, 1, 2] .= -8.0 .+ 0.3 .* randn(rng, 2000)
        arr[:, 1, 3] .=  8.0 .+ 0.3 .* randn(rng, 2000)
        arr[:, 1, 4] .=  8.0 .+ 0.3 .* randn(rng, 2000)
        ch = Chains(arr, [:K])
        r = assess_fit(ch)

        @test r.overall === :fail
        mm = _find_check(r, :multimodality)
        @test mm.status === :fail
        @test occursin("disjoint modes", mm.message)
        @test occursin("merged CI is not a posterior", mm.message)
        # The merged marginal brackets 0 even though no chain visits it.
        @test !isempty(mm.details)
        # LOUD banner on overall fail.
        io = IOBuffer()
        show(io, MIME"text/plain"(), r)
        out = String(take!(io))
        @test occursin("DO NOT TRUST", out)
        @test occursin("❌", out)
    end

    @testset "(c) railed against prior bound => :fail (rail)" begin
        rng = MersenneTwister(9)
        # Eccentricity piled against the hard lower bound at 0.
        arr = zeros(Float64, 2000, 1, 4)
        for k in 1:4
            arr[:, 1, k] .= abs.(0.002 .* randn(rng, 2000))   # ~0, bound = 0
        end
        ch = Chains(arr, [:e])
        r = assess_fit(ch; prior_bounds = Dict(:e => (0.0, 1.0)))

        @test r.overall === :fail
        rail = _find_check(r, :prior_rail)
        @test rail.status === :fail
        @test occursin("rail", lowercase(rail.message))
        @test occursin("e", rail.message)
        # Without prior_bounds the rail check is a no-op (:ok), so the
        # rail itself — not convergence — must be what fails here.
        r_nobounds = assess_fit(ch)
        @test _find_check(r_nobounds, :prior_rail).status === :ok
    end

    @testset "prior-edge mass (no median rail) still flags" begin
        rng = MersenneTwister(10)
        # Median is safely interior, but a fat tail piles >5% of draws
        # against the upper bound at 1.0.
        arr = zeros(Float64, 4000, 1, 4)
        for k in 1:4
            v = 0.6 .+ 0.5 .* randn(rng, 4000)
            clamp!(v, 0.0, 1.0)
            arr[:, 1, k] .= v
        end
        ch = Chains(arr, [:e])
        r = assess_fit(ch; prior_bounds = Dict(:e => (0.0, 1.0)))
        @test _find_check(r, :prior_rail).status === :fail
    end

    @testset "log-posterior sanity catches corrupt :lp" begin
        rng = MersenneTwister(11)
        arr = zeros(Float64, 1000, 2, 3)
        for k in 1:3
            arr[:, 1, k] .= 1.0 .+ randn(rng, 1000)
            arr[:, 2, k] .= 2.0 .+ randn(rng, 1000)
        end
        # Corrupt log-posterior column: absurdly negative values.
        lp = fill(-1e9, 1000, 1, 3)
        ch = Chains(cat(arr, lp; dims = 2), [:a, :b, :lp])
        r = assess_fit(ch)
        lpc = _find_check(r, :logpost)
        @test lpc.status === :fail
        @test occursin("corrupt log-posterior", lpc.message)
        @test r.overall === :fail

        # Sane lp => :ok and lp is not convergence-tested as a parameter.
        lp_ok = -50.0 .+ randn(rng, 1000, 1, 3)
        ch_ok = Chains(cat(arr, lp_ok; dims = 2), [:a, :b, :lp])
        r_ok = assess_fit(ch_ok)
        @test _find_check(r_ok, :logpost).status === :ok
        @test r_ok.overall === :ok
    end

    @testset "ensemble-aware convergence on a single ensemble" begin
        rng = MersenneTwister(12)
        # One ensemble of 20 walkers, each its own short trace — well
        # mixed within and across walkers.
        arr = zeros(Float64, 1000, 1, 20)
        for w in 1:20
            arr[:, 1, w] .= 5.0 .+ 1.0 .* randn(rng, 1000)
        end
        ch = Chains(arr, [:K])
        r = assess_fit(ch; ensemble = true)
        @test _find_check(r, :convergence).status === :ok
        @test r.overall === :ok
    end

    @testset "MAP-consistency convenience method" begin
        rng = MersenneTwister(13)
        arr = zeros(Float64, 2000, 2, 4)
        for k in 1:4
            arr[:, 1, k] .= 10.0 .+ 1.0 .* randn(rng, 2000)
            arr[:, 2, k] .= 4.14 .+ 0.01 .* randn(rng, 2000)
        end
        ch = Chains(arr, [:K, :P])

        # MAP inside the bulk => :ok.
        r_in = assess_fit(Dict(:K => 10.1, :P => 4.141), ch; nsigma = 4.0)
        @test _find_check(r_in, :map_consistency).status === :ok
        @test r_in.overall === :ok

        # MAP far outside the bulk (railed/ spurious mode) => :fail.
        r_out = assess_fit(Dict(:K => 30.0, :P => 4.14), ch; nsigma = 4.0)
        mc = _find_check(r_out, :map_consistency)
        @test mc.status === :fail
        @test occursin("outside posterior bulk", mc.message)
        @test r_out.overall === :fail
    end
end
