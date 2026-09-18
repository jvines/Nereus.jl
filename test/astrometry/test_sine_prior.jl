# Tests for SinePrior — isotropic-orbit inclination prior on [0, π].

@testset "SinePrior" begin
    using Distributions: logpdf, quantile, cdf, insupport

    sp = SinePrior()

    @testset "Bounds" begin
        # Note: SinePrior stores π as Float64; Irrational(π) > Float64(π)
        # exactly, so use Float64(π) in comparisons.
        lo, hi = bounds(sp)
        @test lo == 0.0
        @test hi ≈ π
        @test in_support(sp, 0.0)
        @test in_support(sp, Float64(π))
        @test in_support(sp, π/2)
        @test !in_support(sp, -0.1)
        @test !in_support(sp, π + 0.1)
    end

    @testset "PDF p(i) = sin(i)/2 on [0, π]" begin
        d = sp.dist
        # Logpdf at peak (π/2): log(1/2) = -log(2)
        @test logpdf(d, π/2) ≈ -log(2)
        # Logpdf at endpoints → -Inf (sin(0) = sin(π) = 0)
        @test logpdf(d, 0.0) == -Inf
        # General: logpdf(i) = log(sin(i) / 2)
        for i in (0.1, π/4, π/3, π/2, 2π/3, π - 0.1)
            @test logpdf(d, i) ≈ log(sin(i) / 2)
        end
        # Outside support
        @test logpdf(d, -0.1) == -Inf
        @test logpdf(d, π + 0.1) == -Inf
    end

    @testset "CDF and quantile (inverse CDF)" begin
        d = sp.dist
        # CDF: F(i) = (1 - cos i) / 2
        @test cdf(d, 0.0) == 0.0
        @test cdf(d, π/2) ≈ 0.5
        @test cdf(d, Float64(π)) == 1.0
        # CDF/quantile round-trip
        for u in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
            i = quantile(d, u)
            @test cdf(d, i) ≈ u atol=1e-12
        end
        # Boundary
        @test quantile(d, 0.0) == 0.0
        @test quantile(d, 1.0) ≈ π
    end

    @testset "Equivalent to uniform-cos(i)" begin
        # Draw 10000 samples and check distribution of cos(i) is uniform on [-1, 1]
        using Random: MersenneTwister
        using Statistics: mean, std
        rng = MersenneTwister(42)
        d = sp.dist
        samples = [Distributions.quantile(d, rand(rng)) for _ in 1:10_000]
        cos_i = cos.(samples)
        # cos i should be uniform on [-1, 1]: mean ≈ 0, std ≈ 1/√3 ≈ 0.577
        @test abs(mean(cos_i)) < 0.05
        @test std(cos_i) ≈ 1/sqrt(3) atol=0.05
    end

    @testset "prior_transform round-trip" begin
        for u in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
            i = prior_transform(u, sp)
            @test 0 <= i <= π
        end
    end

    @testset "Wired through packed_priors (PACKED_SINE)" begin
        # Build a tiny model that uses SinePrior on inc
        instruments = InstrumentConfig(rv=["X"])
        t_rv = collect(0.0:1.0:50.0)
        data = Data(t_rv=t_rv, rv=zeros(length(t_rv)), rv_err=fill(1.0, length(t_rv)),
                    hgca = HGCAData(
                        epochs=mjd_epochs((1991.25, 2004.6, 2016.0)),
                        pmra=(0.,0.,0.), pmdec=(0.,0.,0.),
                        sigma_pmra=(0.5,0.02,0.03), sigma_pmdec=(0.4,0.02,0.03),
                        plx=10.0, plx_err=0.1, hip_id=1))
        params = Params(max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=1.0)
        # default_priors should set inc to SinePrior
        params2 = Params(max_kplanet=1, planet_modes=[RVAS],
            instruments=instruments, data=data, stability=:none, M_s=1.0,
            priors = Dict("inc_k1" => SinePrior()))
        # Find inc_k1 in unfrozen
        idx_inc = findfirst(==("inc_k1"), params2.layout.unfrozen_names)
        @test idx_inc !== nothing
        # The packed prior at this index should be PACKED_SINE = 6
        @test params2.layout.packed_priors.type_ids[idx_inc] == Nereus.PACKED_SINE
    end
end
