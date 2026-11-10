# Where a Gaia DR3 parallax sits among the default `plx` priors:
# user prior > Gaia > HGCA > G23H > LogUniform, and no slot without astrometry.

using Nereus, Test, Distributions

@testset "default plx prior precedence" begin
    # (mean, sigma) of a truncated-Normal prior, or nothing if it is another kind
    normal_of(pr) = pr.dist isa Distributions.Truncated && pr.dist.untruncated isa Distributions.Normal ?
        (pr.dist.untruncated.μ, pr.dist.untruncated.σ) : nothing
    begin
        hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                        pmra = (10.0, 11.0, 12.5), pmdec = (-3.0, -2.5, -2.0),
                        sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
                        corr_pmra_pmdec = (0.2, -0.1, 0.3),
                        plx = 20.0, plx_err = 0.5, hip_id = 1)
        cov = zeros(10, 10)
        for i in 1:10; cov[i, i] = 0.01; end
        g23h = G23HData(epochs = mjd_epochs((1991.25, 2004.6, 2015.0, 2015.5, 2016.0)),
                        pmra = (5.0, 4.95, 4.92, 4.94, 4.9),
                        pmdec = (-3.0, -3.05, -3.07, -3.06, -3.1),
                        cov = cov, plx = 30.0, plx_err = 0.4, hip_id = 1)
        rv = (t_rv = [58000.0, 58100.0, 58200.0], rv = [0.0, 1.0, 2.0],
              rv_err = [1.0, 1.0, 1.0])
        gaia = (plx = 14.36, plx_err = 0.02)
        function plxprior(; data, gaia_plx = nothing, priors = nothing, modes = [RVAS])
            ic = InstrumentConfig(rv = ["RV"])
            p = Params(; max_kplanet = length(modes), planet_modes = modes,
                       instruments = ic, data, priors, gaia_plx, stability = :none)
            return p.config.priors["plx"]
        end
        d_h  = Data(; rv..., hgca = hgca)
        d_g  = Data(; rv..., g23h = g23h)
        d_hg = Data(; rv..., hgca = hgca, g23h = g23h)
        d_0  = Data(; rv...)

        pr = plxprior(data = d_hg, gaia_plx = gaia)
        @test normal_of(pr) == (14.36, 0.02)
        @test bounds(pr) == (14.36 - 0.2, 14.36 + 0.2)
        pr = plxprior(data = d_hg)
        @test normal_of(pr)[1] == 20.0            # HGCA beats G23H
        pr = plxprior(data = d_g)
        @test normal_of(pr) == (30.0, 0.4)    # G23H when it is all there is
        @test plxprior(data = d_0).dist isa Distributions.LogUniform
        pr = plxprior(data = d_hg, gaia_plx = gaia,
                      priors = Dict{String, PriorSpec}("plx" => NormalPrior(5.0, 0.1, 4.0, 6.0)))
        @test normal_of(pr)[1] == 5.0             # the user's prior wins

        # A planet without astrometry has no plx slot; gaia_plx changes nothing.
        ic = InstrumentConfig(rv = ["RV"])
        p = Params(; max_kplanet = 1, planet_modes = [RV_ONLY], instruments = ic,
                   data = d_0, gaia_plx = gaia, stability = :none)
        @test !haskey(p.config.priors, "plx")
    end
end
