using Test
using Nereus

@testset "LightCurve construction" begin
    n = 50
    t = collect(range(0.0, 5.0, length=n))
    flux = ones(n)
    flux_err = fill(0.01, n)

    lc = Nereus.LightCurve(t, flux, flux_err)
    @test length(lc.t) == n
    @test length(lc.flux) == n
    @test length(lc.flux_err) == n
    @test eltype(lc.t) === Float64
    @test all(lc.sector_id .== 1)
    @test lc.flux_detrended === nothing
    @test lc.trend === nothing
    @test lc.is_transit === nothing
    @test lc.method === nothing

    # length mismatch
    @test_throws ArgumentError Nereus.LightCurve(t, flux[1:end-1], flux_err)
    # non-positive flux_err
    bad_err = copy(flux_err); bad_err[3] = 0.0
    @test_throws ArgumentError Nereus.LightCurve(t, flux, bad_err)
    bad_err2 = copy(flux_err); bad_err2[5] = -0.1
    @test_throws ArgumentError Nereus.LightCurve(t, flux, bad_err2)
end

@testset "LightCurve multi-sector construction" begin
    # three sectors with DISTINCT, non-overlapping, out-of-order time ranges
    lc1 = Nereus.LightCurve(collect(range(10.0, 12.0, length=20)), ones(20), fill(0.01, 20))
    lc2 = Nereus.LightCurve(collect(range(0.0, 2.0, length=15)),   ones(15), fill(0.01, 15))
    lc3 = Nereus.LightCurve(collect(range(20.0, 22.0, length=25)), ones(25), fill(0.01, 25))

    lc = Nereus.LightCurve([lc1, lc2, lc3])
    @test length(lc.t) == 20 + 15 + 25
    @test length(lc.sector_id) == length(lc.t)
    # time sorted ascending
    @test issorted(lc.t)
    # sector_id assigned by position: lc1->1, lc2->2, lc3->3
    @test count(==(1), lc.sector_id) == 20
    @test count(==(2), lc.sector_id) == 15
    @test count(==(3), lc.sector_id) == 25
    # since ranges don't overlap, after sort the first block (t∈0..2) is sector 2
    @test lc.sector_id[1] == 2
    @test lc.sector_id[end] == 3
    @test lc.flux_detrended === nothing
end

# Build a synthetic single-sector LC with an injected box transit
function _synthetic_transit_lc(; n=400)
    t = collect(range(0.0, 27.0, length=n))
    flux = ones(n)
    # mild slow trend
    flux .+= 0.002 .* sin.(2π .* t ./ 13.0)
    # injected box transit
    P = 3.5; t0 = 1.5; dur = 0.12; depth = 0.01
    phase = mod.(t .- t0 .+ 0.5P, P) .- 0.5P
    in_transit = abs.(phase) .< (dur / 2)
    flux[in_transit] .-= depth
    flux_err = fill(0.0008, n)
    return Nereus.LightCurve(t, flux, flux_err), in_transit
end

@testset "clean_lightcurve notch" begin
    lc, in_transit = _synthetic_transit_lc()
    n = length(lc.t)
    res = Nereus.clean_lightcurve(lc; method=:notch)
    @test res.method === :notch
    @test length(res.flux_detrended) == n
    @test all(isfinite, res.flux_detrended)
    @test res.is_transit isa Vector{Bool}
    @test length(res.is_transit) == n
    @test res.sector_id == lc.sector_id
    @test any(res.is_transit)
    # flagged cadences should land near the injected transit
    @test any(res.is_transit .& in_transit)
end

@testset "clean_lightcurve savgol no transit flag" begin
    lc, _ = _synthetic_transit_lc()
    n = length(lc.t)
    res = Nereus.clean_lightcurve(lc; method=:savgol, window_length=101)
    @test res.method === :savgol
    @test res.is_transit isa Vector{Bool}
    @test !any(res.is_transit)
    @test length(res.flux_detrended) == n
end

@testset "sectors split + roundtrip" begin
    lc1 = Nereus.LightCurve(collect(range(0.0, 5.0, length=60)),  ones(60), fill(0.01, 60))
    lc2 = Nereus.LightCurve(collect(range(10.0, 15.0, length=70)), ones(70), fill(0.01, 70))
    lc3 = Nereus.LightCurve(collect(range(20.0, 25.0, length=80)), ones(80), fill(0.01, 80))
    multi = Nereus.LightCurve([lc1, lc2, lc3])
    clean = Nereus.clean_lightcurve(multi; method=:savgol, window_length=21)

    secs = Nereus.sectors(clean)
    @test length(secs) == 3
    for s in secs
        @test s.flux_detrended !== nothing
        @test s.trend !== nothing
        @test s.is_transit !== nothing
        @test s.method === clean.method
    end
    # concatenating sectors back reproduces the cleaned arrays in time order
    rebuilt_t = vcat([s.t for s in secs]...)
    rebuilt_fd = vcat([s.flux_detrended for s in secs]...)
    @test rebuilt_t == clean.t
    @test rebuilt_fd == clean.flux_detrended
end

@testset "clean_lightcurve errors" begin
    lc, _ = _synthetic_transit_lc(n=100)
    @test_throws ArgumentError Nereus.clean_lightcurve(lc; method=:bogus)
    @test_throws ArgumentError Nereus.clean_lightcurve(lc; method=:gp)
end
