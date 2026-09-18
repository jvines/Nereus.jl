# PT donor-buffer capacity — regression for the n_chains ≥ 22 BoundsError.
#
# `PTWorkspace` used to allocate the donor-birth scratch buffer at a
# hardcoded 20 ("# max chains"), while the trans-dim PT population loop
# counted EVERY eligible donor and then built
# `view(ws.population, 1:ws.n_pop)` from the unclamped counter. With
# n_chains ≥ 22, once ≥21 replicas held an active planet the view went out
# of bounds:
#
#   BoundsError: attempt to access 20-element Vector{Theta{Float64}}
#                at index [1:21]
#
# It failed loud rather than truncating silently, so no past result is
# suspect — but it capped `sample_pt` trans-dim at 21 chains.
#
# Two invariants are locked here:
#   1. the buffer is sized on request (sample_pt asks for n_chains − 1), and
#   2. `max_kplanet = 0` still allocates nothing (no planet births to seed).

using Test
using Nereus
using Nereus: PTWorkspace

@testset "PT donor buffer sizes to the chain count" begin
    data = Data(; t_rv = collect(0.0:2.0:60.0),
                  rv = zeros(31), rv_err = fill(1.0, 31),
                  rv_inst = ones(Int, 31))
    params = Params(; max_kplanet = 3, planet_modes = fill(RV_ONLY, 3),
                      instruments = InstrumentConfig(rv = ["I1"]), data = data,
                      M_s = 1.0)

    # Default preserves the historical capacity for every non-PT caller.
    @test length(PTWorkspace(params, 3; n_obs = 31).population) == 20

    # Explicit request wins — this is what sample_pt passes as n_chains − 1.
    for cap in (1, 19, 20, 21, 29, 64)
        ws = PTWorkspace(params, 3; n_obs = 31, max_donors = cap)
        @test length(ws.population) == cap
    end

    # The exact configuration that used to BoundsError: 30 chains => 29
    # donors, which must fit.
    @test length(PTWorkspace(params, 3; n_obs = 31,
                             max_donors = 30 - 1).population) == 29

    # No planets => no donor births => no buffer, regardless of max_donors.
    params0 = Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
                       instruments = InstrumentConfig(rv = ["I1"]), data = data)
    @test isempty(PTWorkspace(params0, 0; n_obs = 31, max_donors = 40).population)

    # n_pop starts empty, so an unpopulated workspace yields no donor view.
    @test PTWorkspace(params, 3; n_obs = 31, max_donors = 29).n_pop == 0
end
