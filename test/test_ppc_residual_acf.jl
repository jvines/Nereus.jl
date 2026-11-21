# The PPC residual autocorrelation on sparse, gapped sampling.
#
# Empty bins of the uniform grid were filled by linear interpolation, so on a
# typical RV campaign (short runs of nightly points, long gaps) the series was
# mostly straight lines and lag 1 came out near 1 for white residuals: 0.81 on
# the 36 NGTS-33 RVs, whose time-ordered lag-1 correlation is 0.02. The
# estimator now averages over pairs of occupied bins only.

using Nereus, Test, Random, Statistics

@testset "residual ACF uses occupied bins only" begin
    rng = MersenneTwister(7)
    # 20 seasons of 8 consecutive nights, one point a night, seasons 30 d apart
    t = sort(vec([30.0 * s + n + 0.05 * randn(rng) for n in 0:7, s in 0:19]))
    lags, ac = Nereus._residual_acf(t, randn(rng, length(t)))
    @test ac[1] ≈ 1.0
    @test lags[2] ≈ 1.0 atol = 0.1                 # median cadence: one night
    @test abs(ac[2]) < 0.25                         # white residuals: no lag-1 memory
    @test isnan(ac[findfirst(>(12.0), lags)])       # 12-25 d: no pair of points there

    # A real AR(1) process with gaps keeps its lag-1 correlation
    n = 4000; φ = 0.9
    x = zeros(n); x[1] = randn(rng)
    for i in 2:n; x[i] = φ * x[i-1] + sqrt(1 - φ^2) * randn(rng); end
    keep = [i <= 1500 || i > 2500 for i in 1:n]     # a 1000-sample gap
    lags2, ac2 = Nereus._residual_acf(collect(1.0:n)[keep], x[keep])
    @test ac2[2] ≈ φ atol = 0.05
end

# The grid used to run one bin per cadence from the first observation to the
# last. TESS sectors in BTJD pooled with NGTS/LCO in BJD spanned 2.46e6 d at
# 200 s, 1.06e9 bins, and took a NEREUS daemon past 79 GB in the PPC. Long gaps
# are now squeezed, and every lag returned keeps its full-grid value.
function _acf_bruteforce(t, r, dt, L)
    b = round.(Int, (t .- minimum(t)) ./ dt) .+ 1
    occ = sort(unique(b))
    v = [sum(r[b .== k]) / count(==(k), b) for k in occ]
    v .-= sum(v) / length(v)
    ac = fill(NaN, L + 1)
    for ℓ in 0:L
        s = 0.0; n = 0
        for i in eachindex(occ)
            for j in i:length(occ)
                d = occ[j] - occ[i]
                d > ℓ && break
                d == ℓ && (s += v[i] * v[j]; n += 1)
            end
        end
        n > 0 && (ac[ℓ + 1] = s / n)
    end
    return ac ./ ac[1]
end

@testset "residual ACF: long gaps squeezed, lags exact" begin
    rng = MersenneTwister(11)
    # three runs of 60 unit-cadence points, 5000 bins apart
    t = vcat(collect(0.0:59), collect(5000.0:5059), collect(10_000.0:10_059))
    r = randn(rng, length(t))
    lags, ac = Nereus._residual_acf(t, r; max_lag_bins = 100)
    @test length(lags) == 101
    ref = _acf_bruteforce(t, r, 1.0, 100)
    @test isequal(isnan.(ac), isnan.(ref))
    ok = .!isnan.(ref)
    @test ac[ok] ≈ ref[ok] atol = 1e-10
    @test all(isnan, ac[62:end])          # 60-100: no pair within a run reaches it
end

@testset "residual ACF: BTJD mixed with BJD stays bounded" begin
    rng = MersenneTwister(3)
    cad = 200 / 86400
    t = vcat(1570.0 .+ cad .* (0:9999), 2.4601e6 .+ cad .* (0:9999))
    r = randn(rng, length(t))
    Nereus._residual_acf(t, r)                        # compile
    bytes = @allocated lags, ac = Nereus._residual_acf(t, r)
    @test bytes < 200 * 2^20
    @test length(lags) == Nereus._ACF_MAX_LAG_BINS + 1
    @test ac[1] ≈ 1.0
    @test abs(ac[2]) < 0.05
end

@testset "mixed time bases are refused" begin
    kw = Dict{Symbol, Any}(
        :t_rv => [2.4601e6, 2.4602e6], :rv_inst => [1, 1],
        :t_phot => vcat(fill(1570.0, 3), fill(2.4601e6, 3)),
        :phot_inst => [1, 1, 1, 2, 2, 2])
    err = try
        Nereus._check_time_bases(kw, ["FEROS"], ["TESS_S10", "NGTS"]); nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("TESS_S10", err.msg) && occursin("NGTS", err.msg)
    kw[:t_phot] = fill(2.4601e6, 6)
    @test Nereus._check_time_bases(kw, ["FEROS"], ["TESS_S10", "NGTS"]) === nothing
    kw[:t_rv] = [10.0, 20.0]; kw[:t_phot] = fill(1570.0, 6)   # all reduced: fine
    @test Nereus._check_time_bases(kw, ["FEROS"], ["TESS_S10", "NGTS"]) === nothing
end
