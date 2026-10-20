# HGCA and GOST as the Python client sends them, inside an astrometry channel:
# the run_job block forms, and the field dicts the client's `load_hgca` /
# `load_gost` return. Both used to reach `Data` as raw dicts and die there.

using Nereus, Test, JSON3

# What the wire does to a value: the daemon writes a struct's fields with
# JSON3, Python reads and re-sends them, and the daemon reads them back into
# plain Dicts and Vector{Any} (daemon.jl `_j`).
_plain(x::AbstractDict)   = Dict{String,Any}(String(k) => _plain(v) for (k, v) in x)
_plain(x::AbstractVector) = Any[_plain(v) for v in x]
_plain(x) = x
wire(x) = _plain(JSON3.read(JSON3.write(x)))
fields(s) = Dict{String,Any}(String(f) => getfield(s, f) for f in fieldnames(typeof(s)))

@testset "fit_* channels: HGCA and GOST from JSON blocks" begin
    hgca = HGCAData(epochs = mjd_epochs((1991.25, 2004.6, 2016.0)),
                    pmra = (10.0, 11.0, 12.5), pmdec = (-3.0, -2.5, -2.0),
                    sigma_pmra = (0.5, 0.02, 0.03), sigma_pmdec = (0.4, 0.02, 0.03),
                    corr_pmra_pmdec = (0.2, -0.1, 0.3),
                    plx = 50.0, plx_err = 0.05, hip_id = 85653)
    gost = GOSTData(t = [57000.0, 57100.0, 57250.0], psi = [0.1, 1.2, 2.3],
                    parallax_factor = [0.5, -0.3, 0.8],
                    along_scan_pos = [1.0, 2.0, 3.0])
    same_hgca(h) = h isa HGCAData && h.epochs == hgca.epochs &&
        h.pmra == hgca.pmra && h.pmdec == hgca.pmdec && h.cov_ep == hgca.cov_ep &&
        h.plx == hgca.plx && h.plx_err == hgca.plx_err && h.hip_id == hgca.hip_id

    rv = Dict("source" => "RV", "data" => Dict("RV" => (t = [58000.0, 58100.0, 58200.0],
                                                     rv = [0.0, 1.0, 2.0], rv_err = [1.0, 1.0, 1.0])))
    priors = Dict("plx" => Dict("type" => "normal", "mu" => 50.0, "sigma" => 0.05),
                  "M_pri" => Dict("type" => "normal", "mu" => 1.0, "sigma" => 0.05))
    as(; kw...) = Dict{String,Any}("source" => "AS", (String(k) => v for (k, v) in kw)...)

    @testset "loader field dicts round-trip into Data" begin
        # The struct carries `hip_id`; the values must win over an auto-fetch.
        tgt = Nereus._target_from([rv, as(hgca = wire(fields(hgca)), gost = wire(fields(gost)))],
                                  1; priors)
        @test same_hgca(tgt.data.hgca)
        g = tgt.data.gost
        @test g isa GOSTData
        @test g.t == gost.t && g.psi == gost.psi
        @test g.parallax_factor == gost.parallax_factor
        @test g.along_scan_pos == gost.along_scan_pos
    end

    @testset "run_job `values` blocks" begin
        vals = Dict("t" => collect(hgca.epochs), "pmra" => collect(hgca.pmra),
                    "pmdec" => collect(hgca.pmdec),
                    "cov_ep" => [[c[1, :], c[2, :]] for c in hgca.cov_ep],   # nested rows
                    "plx" => hgca.plx, "plx_err" => hgca.plx_err, "hip_id" => hgca.hip_id)
        gvals = Dict("t" => gost.t, "psi" => gost.psi, "parallax_factor" => gost.parallax_factor)
        tgt = Nereus._target_from([rv, as(hgca = wire(Dict("values" => vals)),
                                          gost = wire(Dict("values" => gvals)))], 1; priors)
        @test same_hgca(tgt.data.hgca)
        @test tgt.data.gost.psi == gost.psi
        @test tgt.data.gost.along_scan_pos == zeros(3)   # absent -> GOSTData's default
    end

    @testset "structs pass through untouched" begin
        tgt = Nereus._target_from([rv, as(hgca = hgca, gost = gost)], 1; priors)
        @test tgt.data.hgca === hgca
        @test tgt.data.gost === gost
    end

    @testset "a covariance keeps its orientation through JSON3" begin
        # HGCA covariances are symmetric, so a transposed read would pass the
        # tests above; an asymmetric one pins the order JSON3 writes a Matrix.
        m = [1.0 2.0; 3.0 4.0]
        blk = wire(Dict("epochs" => [1.0, 2.0, 3.0], "pmra" => [0.0, 0.0, 0.0],
                        "pmdec" => [0.0, 0.0, 0.0], "cov_ep" => (m, m, m),
                        "plx" => 1.0, "plx_err" => 0.1, "hip_id" => 1))
        @test Nereus._parse_hgca_block(blk).cov_ep[1] == m
        @test Nereus._hgca_cov([[1.0, 2.0], [3.0, 4.0]]) == m
        @test_throws ErrorException Nereus._hgca_cov([1.0, 2.0, 3.0])
    end

    @testset "malformed blocks are refused" begin
        # `csv` named a loader that never existed; with hip_id beside it the
        # block must not fall through to the auto-fetch.
        @test_throws ErrorException Nereus._parse_hgca_block(
            Dict("csv" => "hgca.csv", "hip_id" => 85653))
        @test_throws ErrorException Nereus._parse_hgca_block(Dict("fits" => "HGCA_vEDR3.fits"))
        @test_throws ErrorException Nereus._parse_hgca_block(Dict{String,Any}())
        two = Dict("epochs" => [1.0, 2.0], "pmra" => [0.0, 0.0], "pmdec" => [0.0, 0.0],
                   "cov_ep" => [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
        @test_throws ErrorException Nereus._parse_hgca_block(two)
    end
end
