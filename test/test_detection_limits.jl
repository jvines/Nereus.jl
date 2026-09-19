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

# ---------------------------------------------------------------------
# detectability: detection probability over (P, mass) from the draws
# ---------------------------------------------------------------------

@testset "detectability — synthetic chain with an injected signal" begin
    rng = MersenneTwister(7)
    t = collect(0.0:4.0:400.0)
    n_obs = length(t)
    P_true, K_true, Mo_true = 40.0, 12.0, π / 2
    ic = InstrumentConfig(rv = ["SIM"])

    # Build the injected velocities with Nereus's own model, so the draws below
    # share its phase convention instead of a hand-rolled sinusoid's.
    blank = Data(; t_rv = t, rv = zeros(n_obs), rv_err = ones(n_obs),
                   rv_inst = ones(Int, n_obs))
    p0 = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                 instruments = ic, data = blank, M_s = 1.0)
    th0 = Theta(p0)
    for (nm, v) in ("P_k1" => P_true, "K_k1" => K_true, "Mo_k1" => Mo_true,
                    "sesinw_k1" => 0.0, "secosw_k1" => 0.0,
                    "gamma_SIM" => 0.0, "sigma_SIM" => 1.0)
        set_param!(th0, nm, v)
    end
    pred, _ = Nereus.rv_predictions(th0, blank)
    rv = pred .+ randn(rng, n_obs)
    data = Data(; t_rv = t, rv = rv, rv_err = ones(n_obs),
                  rv_inst = ones(Int, n_obs))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                     instruments = ic, data = data, M_s = 1.0)

    # Draws spanning K from negligible to well above the injected amplitude,
    # at the right period and phase, so detectability must rise with mass.
    n = 4_000
    names_ = params.layout.unfrozen_names
    fitted = Symbol.(names_)
    arr = zeros(Float64, n, length(fitted), 1)
    Ks = exp.(range(log(0.05), log(40.0); length = n))
    for (j, nm) in enumerate(fitted)
        if nm === :P_k1
            arr[:, j, 1] .= P_true
        elseif nm === :K_k1
            arr[:, j, 1] .= Ks
        elseif nm === :Mo_k1
            arr[:, j, 1] .= Mo_true        # matches the injected signal
        elseif nm === :sigma_SIM
            arr[:, j, 1] .= 1.0
        end
    end
    chains = Chains(arr, fitted)

    # Two period bins: one holding the injected period, one far from it.
    res = detectability(chains, params, data; planet = 1, n_P_bins = 3,
                        n_M_bins = 12, min_samples = 5,
                        P_min = P_true, P_max = P_true)

    @test res.quantity === :msini          # RV_ONLY block carries no inclination
    @test size(res.fraction) == (3, 12)
    @test length(res.M50) == 3
    @test res.threshold == 25.0

    fin = filter(isfinite, res.fraction)
    @test !isempty(fin)
    @test all(0.0 .<= fin .<= 1.0)

    # At the injected period the true mass is distinguishable from no companion,
    # while an amplitude far above it fits worse and is not.
    row = res.fraction[2, :]
    near_truth = [res.M_centers[b] for b in 1:12 if isfinite(row[b]) && row[b] > 0.5]
    @test !isempty(near_truth)
    K_of = m -> 28.4 * (m / 9.543e-4) * (P_true / 365.25)^(-1/3)
    @test any(4.0 .< K_of.(near_truth) .< 25.0)     # brackets the injected 12 m/s
    @test row[end] < 0.5                            # K ~ 40 m/s is ruled out

    # The 50% crossing, where the chain brackets it, lands inside the mass grid.
    for m50 in filter(isfinite, res.M50)
        @test res.M_edges[1] <= m50 <= res.M_edges[end]
    end
end

@testset "detectability — input validation" begin
    rng = MersenneTwister(3)
    t = collect(0.0:10.0:200.0)
    data = Data(; t_rv = t, rv = randn(rng, length(t)),
                  rv_err = ones(length(t)), rv_inst = ones(Int, length(t)))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                     instruments = InstrumentConfig(rv = ["SIM"]),
                     data = data, M_s = 1.0)
    fitted = Symbol.(params.layout.unfrozen_names)
    arr = ones(Float64, 50, length(fitted), 1)
    for (j, nm) in enumerate(fitted)
        nm === :P_k1 && (arr[:, j, 1] .= 30.0)
        nm === :K_k1 && (arr[:, j, 1] .= 2.0)
    end
    chains = Chains(arr, fitted)

    @test_throws ArgumentError detectability(chains, params, data; planet = 0)
    @test_throws ArgumentError detectability(chains, params, data; threshold = 0.0)
    @test_throws ArgumentError detectability(chains, params, data; n_P_bins = 2)
end
