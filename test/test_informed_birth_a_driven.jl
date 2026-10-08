# Informed births under the :a_driven parametrization, where the period slot
# holds a (AU). They read that slot's bounds as a period range in days: for
# a ∈ [0.3, 3] AU the RV periodogram ran over 0.3–3 d (~10⁵ frequencies, which
# dominated a trans-dim burn-in) and every "informed" a was a sub-3-day peak
# written into an AU slot, so a birth never landed on the planet the RVs show.
using Test
using Random
using Nereus
using Nereus: propose_planet_birth, InformedBirth, TransDimState, Theta, planet_P

function informed_params(mass::Symbol, P_true::Float64)
    t_rv = collect(range(55000.0, 57000.0; length = 80))
    hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                    pmra = (0.0, 0.0, 0.0), pmdec = (0.0, 0.0, 0.0),
                    sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
                    plx = 10.0, plx_err = 0.1, hip_id = 1)
    data = Data(t_rv = t_rv, rv = 30.0 .* sin.(2π .* t_rv ./ P_true),
                rv_err = ones(length(t_rv)), hgca = hgca)
    slot = mass === :a_driven ? "a_k1" => LogUniformPrior(0.3, 3.0) :
                                 "P_k1" => LogUniformPrior(60.0, 1900.0)
    params = Params(max_kplanet = 1, planet_modes = [RVAS],
                    instruments = InstrumentConfig(rv = ["X"]), data = data,
                    stability = :none, M_s = 1.0,
                    parametrization = ParametrizationConfig(mass = mass),
                    priors = Dict{String, PriorSpec}(slot,
                        "M_sec_k1" => LogUniformPrior(1e-4, 0.01),
                        "M_pri" => FixedPrior(1.0)))
    return params, data
end

# Periods of the companions `n` informed births propose, and their slot values.
function informed_draws(mass::Symbol, P_true::Float64; n::Int = 400, seed::Int = 7)
    params, data = informed_params(mass, P_true)
    theta = Theta{Float64}(params; td = TransDimState(max_planets = 1))
    rng = MersenneTwister(seed)
    slot = params.layout.planet_blocks[1].P
    periods, slots, log_qs = Float64[], Float64[], Float64[]
    for _ in 1:n
        born, log_q = propose_planet_birth(theta, rng, InformedBirth(); data = data)
        push!(periods, planet_P(born, 1))
        push!(slots, born.values[slot])
        push!(log_qs, log_q)
    end
    return periods, slots, log_qs
end

near(periods, P) = count(p -> abs(log(p / P)) < 0.15, periods) / length(periods)

@testset "informed births find the RV period under every mass parametrization" begin
    P_true = 300.0
    periods, a, log_q = informed_draws(:a_driven, P_true)
    @test all(isfinite, log_q)
    @test all(x -> 0.3 - 1e-12 <= x <= 3.0 + 1e-12, a)
    @test near(periods, P_true) > 0.4

    # The parametrizations whose slot IS the period are untouched by the map.
    periods_P, P_slot, log_q_P = informed_draws(:M_sec_driven, P_true)
    @test all(isfinite, log_q_P)
    @test periods_P == P_slot
    @test near(periods_P, P_true) > 0.4
end
