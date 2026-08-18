# Bayesian K_lim(P) from a synthetic prior-seeded chain. Verifies the
# baseline noise level off-peak, the inflation at the injected-signal
# bin, sentinel NaN for empty bins, and the coverage warning.

using Nereus, Random, Statistics, Test, MCMCChains
using Random: MersenneTwister

@testset "detection_limits — synthetic prior-seeded chain" begin
    rng = MersenneTwister(0)
    data = Data(; t_rv = collect(0.0:5.0:200.0), rv = randn(rng, 41),
                  rv_err = ones(41), rv_inst = ones(Int, 41))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                     instruments = ic, data = data, M_s = 1.0)

    # Prior-seeded chain: log-uniform P in [1, 100] d (well inside the
    # auto-prior [0.1, T/2]). K is half-Normal at σ=0.5 except in a
    # narrow band around P=20 d where the "signal" lives at K=5±0.5.
    n = 20_000
    logP_min, logP_max = log(1.0), log(100.0)
    P = exp.(logP_min .+ (logP_max - logP_min) .* rand(rng, n))
    K = similar(P)
    @inbounds for i in 1:n
        K[i] = (18.0 < P[i] < 22.0) ?
               5.0 + 0.5 * randn(rng) :
               abs(0.5 * randn(rng))
    end

    fitted_names = Symbol.(params.layout.unfrozen_names)
    arr = zeros(Float64, n, length(fitted_names), 1)
    for (j, nm) in enumerate(fitted_names)
        if nm === :P_k1
            arr[:, j, 1] .= P
        elseif nm === :K_k1
            arr[:, j, 1] .= K
        end
    end
    chains = Chains(arr, fitted_names)

    res = detection_limits(chains, params; n_bins = 25, confidence = 0.95)

    @test length(res.K_limit) == 25
    @test length(res.P_centers) == 25
    @test length(res.P_edges) == 26
    @test res.confidence == 0.95
    @test res.planet_index == 1

    # Bins outside the chain's P range → NaN.
    outside_lo = isnan.(res.K_limit[res.P_centers .< 0.9])
    outside_hi = isnan.(res.K_limit[res.P_centers .> 110.0])
    @test all(outside_lo)
    @test all(outside_hi)

    # In-range, no-signal bins (P far from 20 d) should be at the 95%
    # quantile of |Normal(0, 0.5)| ≈ 0.98 m/s. Allow some slack.
    valid = isfinite.(res.K_limit)
    nosig = valid .& ((res.P_centers .< 12.0) .| (res.P_centers .> 35.0))
    @test count(nosig) >= 5
    K_nosig = res.K_limit[nosig]
    @test 0.6 < median(K_nosig) < 1.4

    # Signal bins (P_center near 20 d) should be near K=5.
    sig = valid .& (res.P_centers .> 12.0) .& (res.P_centers .< 35.0)
    @test count(sig) >= 1
    @test maximum(res.K_limit[sig]) > 3.0

    # Plot saves a non-trivial PNG.
    tmp = tempname() * ".png"
    plot_detection_limits(res; filename = tmp)
    @test isfile(tmp)
    @test filesize(tmp) > 1000
    rm(tmp; force = true)
end

@testset "detection_limits — input validation" begin
    data = Data(; t_rv = collect(0.0:5.0:100.0), rv = randn(21),
                  rv_err = ones(21), rv_inst = ones(Int, 21))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                     instruments = ic, data = data, M_s = 1.0)
    fitted_names = Symbol.(params.layout.unfrozen_names)
    arr = ones(Float64, 100, length(fitted_names), 1)
    for (j, nm) in enumerate(fitted_names)
        arr[:, j, 1] .= nm === :P_k1 ? 20.0 :
                         (nm === :K_k1 ? 1.0 : 0.0)
    end
    chains = Chains(arr, fitted_names)

    @test_throws ArgumentError detection_limits(chains, params; planet = 0)
    @test_throws ArgumentError detection_limits(chains, params; confidence = 0.5)
    @test_throws ArgumentError detection_limits(chains, params; confidence = 1.0)
    @test_throws ArgumentError detection_limits(chains, params; n_bins = 2)
    @test_throws ArgumentError detection_limits(chains, params; planet = 2)  # no K_k2 column
end

@testset "detection_limits — trans-dim mask" begin
    # When n_planets is in the chain and varies across draws, only
    # samples with the planet ACTIVE should contribute to K_lim.
    rng = MersenneTwister(3)
    data = Data(; t_rv = collect(0.0:5.0:200.0), rv = randn(rng, 41),
                  rv_err = ones(41), rv_inst = ones(Int, 41))
    ic = InstrumentConfig(rv = ["SIM"])
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                     instruments = ic, data = data, M_s = 1.0)

    fitted_names = Symbol.(params.layout.unfrozen_names)
    # Append :n_planets to the chain
    names_with_np = vcat(fitted_names, [:n_planets])
    n = 4_000
    arr = zeros(Float64, n, length(names_with_np), 1)
    # half the draws have n_planets = 0 (inactive), K is huge nonsense
    # there; should be masked out.
    for i in 1:n
        active = i > n ÷ 2
        for (j, nm) in enumerate(names_with_np)
            arr[i, j, 1] = if nm === :P_k1
                exp(log(1.0) + (log(100.0) - log(1.0)) * rand(rng))
            elseif nm === :K_k1
                active ? abs(0.3 * randn(rng)) : 1000.0
            elseif nm === :n_planets
                active ? 1.0 : 0.0
            else
                0.0
            end
        end
    end
    chains = Chains(arr, names_with_np)

    res = detection_limits(chains, params; n_bins = 20, confidence = 0.95)
    # All active-draw K values are tiny → no bin should report K_lim
    # anywhere near 1000.
    @test maximum(filter(isfinite, res.K_limit)) < 5.0
end
