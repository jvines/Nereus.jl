# vet_candidate: the periodogram-based vetoes on short baselines and short periods.

using Nereus, Random, Test
using Random: MersenneTwister

@testset "vet_candidate: a baseline shorter than 1 d leaves the periodogram vetoes untested" begin
    rng = MersenneTwister(1)
    t = sort!(0.5 .* rand(rng, 30))
    rv = randn(rng, 30)
    v = vet_candidate(t, rv, ones(30), 3.0, 5.0, 0.1;
                      indicators = Dict("bis" => randn(rng, 30)))
    @test v.activity_indicator === :inapplicable
    @test v.alias === :inapplicable
    @test v.decision !== :candidate
end

@testset "vet_candidate: the indicator veto reaches periods below 1 d" begin
    # an indicator varying at P = 0.6 d; the search used to start at 1 d
    rng = MersenneTwister(2)
    t = sort!(60.0 .* rand(rng, 120))
    P = 0.6
    rv  = 5.0 .* sin.(2π .* t ./ P) .+ randn(rng, 120)
    bis = sin.(2π .* t ./ P) .+ 0.1 .* randn(rng, 120)
    v = vet_candidate(t, rv, ones(120), P, 5.0, 0.0; indicators = Dict("bis" => bis))
    @test v.activity_indicator === :fail
    @test v.detail["indicators_flagged"] == ["bis"]
    @test v.decision === :reject
end

@testset "vet_candidate: the alias veto sees the window's peak" begin
    # four visits every 10 d: the spectral window peaks at 10 d
    rng = MersenneTwister(3)
    t = sort!(vcat([10.0 * k .+ 0.3 .* rand(rng, 4) for k in 0:30]...))
    n = length(t)
    v = vet_candidate(t, randn(rng, n), ones(n), 10.0, 5.0, 0.0)
    @test v.alias === :fail
    @test any(occursin("window-function peak", r) for r in v.reasons)
    # a period away from the window's peaks passes
    v2 = vet_candidate(t, randn(rng, n), ones(n), 13.7, 5.0, 0.0)
    @test v2.alias === :pass
end
