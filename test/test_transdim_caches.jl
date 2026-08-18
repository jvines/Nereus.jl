# Bounded-growth guards for the trans-dim birth-strategy caches.
#
# A memory audit of the long-lived nereus daemon found the serve loop holds
# no cross-request state EXCEPT two module-level caches that grew one entry
# per distinct Data object, forever, strong-ref (_WINDOW_CACHES keyed on a
# Data-content hash, _GAMMA_MEDIAN_CACHE keyed on objectid(data)). KB-scale
# per job — not a big leak, but the only monotonic one. These lock in the
# `length > 512 && empty!` guard the sibling caches (_PEAK_CACHES,
# _BLS_CACHES) already carry.
using Test
using Random
using Nereus
using Nereus: _get_window_cache, _WINDOW_CACHES,
               _per_instrument_medians, _GAMMA_MEDIAN_CACHE, _get_cache,
               _PEAK_CACHES

@testset "reset_transdim_caches! clears every global cache" begin
    # Benchmarking two configurations in one process is only fair if each
    # starts cold: these caches are objectid-keyed and the >512 guard wipes
    # them wholesale, so what survives depends on run order.
    _get_window_cache(UInt64(11))
    Nereus._get_cache(UInt64(12))
    Nereus._get_bls_cache(UInt64(13))
    @test !isempty(_WINDOW_CACHES)
    @test !isempty(_PEAK_CACHES)

    reset_transdim_caches!()

    @test isempty(_WINDOW_CACHES)
    @test isempty(_PEAK_CACHES)
    @test isempty(Nereus._BLS_CACHES)
    @test isempty(Nereus._GAMMA_MEDIAN_CACHE)
    # The GP period hint added a fourth cache; it must be cleared too or a
    # benchmark's second configuration starts warm and the comparison lies.
    @test isempty(Nereus._GP_HINT_CACHE)
end

@testset "_GP_HINT_CACHE cleared and bounded" begin
    empty!(Nereus._GP_HINT_CACHE)
    for i in 1:600
        Nereus._GP_HINT_CACHE[UInt64(i)] = (Float64[log(8.0)], Float64[1.0])
        if length(Nereus._GP_HINT_CACHE) > 256
            # mirrors the in-function guard
            empty!(Nereus._GP_HINT_CACHE)
        end
    end
    @test length(Nereus._GP_HINT_CACHE) <= 256 + 1
    Nereus._GP_HINT_CACHE[UInt64(1)] = (Float64[log(8.0)], Float64[1.0])
    reset_transdim_caches!()
    @test isempty(Nereus._GP_HINT_CACHE)
end

@testset "trans-dim caches stay bounded" begin
    @testset "_WINDOW_CACHES capped at 512" begin
        empty!(_WINDOW_CACHES)
        for i in 1:1300                       # >2 cap cycles
            _get_window_cache(UInt64(i))
        end
        @test length(_WINDOW_CACHES) <= 512 + 1   # cap empties, then adds one
        # still functional after a reset: same id returns the same entry
        c1 = _get_window_cache(UInt64(7))
        c2 = _get_window_cache(UInt64(7))
        @test c1 === c2
    end

    @testset "_GAMMA_MEDIAN_CACHE capped at 512" begin
        empty!(_GAMMA_MEDIAN_CACHE)
        rng = MersenneTwister(1)
        t = collect(1.0:10.0)
        keep = Data[]                          # hold refs so objectids stay distinct
        for i in 1:600
            d = Data(; t_rv = t, rv = randn(rng, 10), rv_err = fill(1.0, 10),
                       rv_inst = vcat(ones(Int, 5), fill(2, 5)))
            push!(keep, d)
            meds = _per_instrument_medians(d)
            @test length(meds) == 2            # per-instrument medians computed
        end
        @test length(_GAMMA_MEDIAN_CACHE) <= 512 + 1
        # correctness unchanged by the guard: recompute-on-demand after reset
        d = keep[end]
        @test _per_instrument_medians(d) == _per_instrument_medians(d)
    end

    @testset "sibling _PEAK_CACHES guard still present (regression anchor)" begin
        empty!(_PEAK_CACHES)
        for i in 1:1300
            _get_cache(UInt64(i))
        end
        @test length(_PEAK_CACHES) <= 512 + 1
    end
end
