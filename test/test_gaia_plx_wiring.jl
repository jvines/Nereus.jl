# Wiring of the Gaia DR3 parallax lookup: fit_* (`_target_from`) and run_job.
# No network: the cache directory is seeded.

using Nereus, Test, Distributions

@testset "Gaia DR3 parallax wiring" begin
    normal_of(pr) = pr.dist isa Distributions.Truncated && pr.dist.untruncated isa Distributions.Normal ?
        (pr.dist.untruncated.μ, pr.dist.untruncated.σ) : nothing
    @testset "star block keys" begin
        s = Nereus._build_star(Dict("gaia_id" => "Gaia DR3 77", "name" => "X"))
        @test s.gaia_id == 77 && s.name == "X"
        s = Nereus._build_star(Dict())
        @test s.gaia_id === nothing && s.name === nothing
    end

    @testset "fit_* path: _target_from" begin
        mktempdir() do dir
            withenv("NEREUS_DATA_CACHE" => dir) do
                gdir = Nereus.nereus_cache_dir("gaia")
                write(joinpath(gdir, "dr3_55.json"),
                      """{"source_id":55,"plx":8.5,"plx_err":0.03,"release":"DR3"}""")
                hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                                pmra = (10.0, 11.0, 12.5), pmdec = (-3.0, -2.5, -2.0),
                                sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
                                corr_pmra_pmdec = (0.2, -0.1, 0.3),
                                plx = 20.0, plx_err = 0.5, hip_id = 1)
                rv = Dict("source" => "RV", "data" => Dict("RV" => (
                    t = [58000.0, 58100.0, 58200.0], rv = [0.0, 1.0, 2.0],
                    rv_err = [1.0, 1.0, 1.0])))
                # `hip_scans = false`: otherwise the row's hip_id fetches the
                # Hipparcos scans its Hipparcos epoch is fitted through, over the
                # network into this empty cache.
                as = Dict{String,Any}("source" => "AS", "hgca" => hgca, "hip_scans" => false)
                rep = Any[]
                tgt = Nereus._target_from([rv, as], 1; gaia_id = "Gaia DR3 55", gaia_report = rep)
                @test normal_of(tgt.params.config.priors["plx"])[1] == 8.5
                @test length(rep) == 1 && rep[1].resolved_from == "gaia_id"
                # the HGCA value stays the default without a Gaia key
                tgt = Nereus._target_from([rv, as], 1)
                @test normal_of(tgt.params.config.priors["plx"])[1] == 20.0
                # a plx prior of the caller's: Gaia is not consulted
                rep = Any[]
                tgt = Nereus._target_from([rv, as], 1; gaia_id = 56, gaia_report = rep,
                    priors = Dict("plx" => NormalPrior(5.0, 0.1, 4.0, 6.0)))
                @test normal_of(tgt.params.config.priors["plx"])[1] == 5.0 && isempty(rep)
                # no astrometry: no lookup (id 56 is not cached and would need the network)
                tgt = Nereus._target_from([rv], (k1 = default_rv_planet([58000.0, 58100.0], [0.0, 1.0]),); gaia_id = 56)
                @test !haskey(tgt.params.config.priors, "plx")
            end
        end
    end

    @testset "run_job: query only when it can matter" begin
        mktempdir() do dir
            withenv("NEREUS_DATA_CACHE" => dir) do
                gdir = Nereus.nereus_cache_dir("gaia")
                write(joinpath(gdir, "dr3_55.json"),
                      """{"source_id":55,"plx":8.5,"plx_err":0.03,"release":"DR3"}""")
                star = Nereus._build_star(Dict("gaia_id" => 55))
                cfg(modes; priors = Dict()) = Dict("model" => Dict("planet_modes" => modes),
                                                   "priors" => priors)
                g = Nereus._resolve_gaia_plx(cfg(["RVAS"]), star)
                @test g.plx == 8.5 && g.resolved_from == "gaia_id"
                @test Nereus._gaia_summary(g)["source_id"] == 55
                # no astrometric planet: nothing to look up
                @test Nereus._resolve_gaia_plx(cfg(["RV_ONLY"]), star) === nothing
                # user plx prior: no query (an id with no cache entry would fail)
                star2 = Nereus._build_star(Dict("gaia_id" => 56))
                @test Nereus._resolve_gaia_plx(
                    cfg(["RVAS"]; priors = Dict("plx" => Dict("type" => "normal"))), star2) === nothing
                # requested and unavailable: throws, no silent fallback
                down(args...) = error("network down")
                e = try Nereus._resolve_gaia_plx(cfg(["RVAS"]), star2; http_get = down,
                                                 verbose = false); nothing
                    catch err; sprint(showerror, err) end
                @test e !== nothing && occursin("prior on plx", e)
                # no star keys: nothing
                @test Nereus._resolve_gaia_plx(cfg(["RVAS"]), Nereus._build_star(Dict())) === nothing
            end
        end
    end
end
