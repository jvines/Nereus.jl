using Test
using Statistics
using Random
using Nereus   # detrend_locor, detrend_notch, save_detrend_result, load_lightcurve

@testset "save_detrend_result :locor round-trip" begin
    # Fast-rotator synthetic (same recipe as the test_locor.jl headline test):
    # P_rot = 0.8 d, 10-min cadence, ~12 d baseline, small noise.
    P_rot_true = 0.8
    cad = 10 / 60 / 24            # 10 min in days
    t = collect(0.0:cad:12.0)
    n = length(t)
    rng = MersenneTwister(20260524)
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot_true) .+ 3e-4 .* randn(rng, n)
    flux_err = fill(3e-4, n)

    res = Nereus.detrend_locor(t, flux, flux_err; P_rot = 0.8)

    p = tempname() * ".nc"
    try
        Nereus.save_detrend_result(p, t, flux, flux_err, res; method = :locor)
        lc = Nereus.load_lightcurve(p)

        # Core arrays round-trip.
        @test lc.flux_detrended ≈ res.flux_detrended
        @test lc.trend ≈ res.trend

        # Method attribute recorded.
        @test lc.attrs["detrend_method"] == "locor"

        # P_rot_used persisted as a (finite) attr.
        @test haskey(lc.attrs, "P_rot_used")
        @test lc.attrs["P_rot_used"] ≈ res.P_rot_used

        # LOCoR per-obs diagnostics persisted as extra columns (coerced to
        # Float64 by save_lightcurve — load returns them as fields named after
        # the saved variables).
        @test hasproperty(lc, :phase)
        @test hasproperty(lc, :cycle_id)
        @test lc.phase ≈ res.phase
        @test lc.cycle_id ≈ Float64.(res.cycle_id)
        @test hasproperty(lc, :outlier)
        @test hasproperty(lc, :used_notch)
    finally
        isfile(p) && rm(p)
    end
end

@testset "save_detrend_result :locor sector_id round-trip" begin
    # Two-sector synthetic (>0.5 d gap between chunks); sector_id is user
    # metadata that must survive detrend + save + load.
    P_rot_true = 0.8
    cad = 10 / 60 / 24
    t1 = collect(0.0:cad:12.0)
    t2 = collect(20.0:cad:32.0)
    t = vcat(t1, t2)
    n1 = length(t1); n2 = length(t2); n = length(t)
    sid = [fill(1, n1); fill(2, n2)]

    rng = MersenneTwister(99)
    flux = 1.0 .+ 0.02 .* sin.(2π .* t ./ P_rot_true) .+ 3e-4 .* randn(rng, n)
    flux_err = fill(3e-4, n)

    res = Nereus.detrend_locor(t, flux, flux_err; P_rot = 0.8, sector_id = sid)

    p = tempname() * ".nc"
    try
        Nereus.save_detrend_result(p, t, flux, flux_err, res; method = :locor)
        lc = Nereus.load_lightcurve(p)
        @test hasproperty(lc, :sector_id)
        @test lc.sector_id == sid
    finally
        isfile(p) && rm(p)
    end
end

@testset "save_detrend_result :notch still round-trips" begin
    # Quick smoke that the pre-existing :notch path is untouched.
    cad = 10 / 60 / 24
    t = collect(0.0:cad:5.0)
    n = length(t)
    rng = MersenneTwister(7)
    flux = 1.0 .+ 3e-4 .* randn(rng, n)
    flux_err = fill(3e-4, n)

    res = Nereus.detrend_notch(t, flux, flux_err)

    p = tempname() * ".nc"
    try
        Nereus.save_detrend_result(p, t, flux, flux_err, res; method = :notch)
        lc = Nereus.load_lightcurve(p)
        @test lc.flux_detrended ≈ res.flux_detrended
        @test lc.trend ≈ res.trend
        @test lc.attrs["detrend_method"] == "notch"
        @test hasproperty(lc, :delta_bic)
    finally
        isfile(p) && rm(p)
    end
end
