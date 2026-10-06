# Prior dicts, the form the Python client and the fit_* API send.

using Nereus, Test, Distributions

@testset "prior dicts: a normal takes optional bounds" begin
    free = Nereus._as_prior(Dict("type" => "normal", "mu" => 2.2, "sigma" => 0.02))
    @test free.lo == -Inf && free.hi == Inf

    # An inclination needs a bounded prior: the bare normal reaches below 0.
    inc = Nereus._as_prior(Dict("type" => "normal", "mu" => 2.2, "sigma" => 0.02,
                                "lo" => 0.0, "hi" => π))
    @test (inc.lo, inc.hi) == (0.0, Float64(π))
    @test inc.dist isa Distributions.Truncated

    half = Nereus._as_prior(Dict("type" => "normal", "mu" => 1.0, "sigma" => 0.5, "lo" => 0.0))
    @test (half.lo, half.hi) == (0.0, Inf)

    @test_throws ArgumentError Nereus._as_prior(
        Dict("type" => "normal", "mu" => 1.0, "sigma" => 0.5, "lo" => 2.0, "hi" => 1.0))
end
