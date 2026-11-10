# Runtime Gaia DR3 parallax: fetch_gaia_parallax (id and name, cached, errors).
# No network: the transport is a function the tests supply, and the cache is a
# temp directory.

using Nereus, Test, JSON3

@testset "Gaia DR3 parallax at runtime" begin
    gaia_body(rows) = """{"metadata":[{"name":"source_id"},{"name":"parallax"},{"name":"parallax_error"}],"data":$rows}"""
    simbad_body(ids) = JSON3.write((; metadata = [(; name = "id")], data = [[i] for i in ids]))

    calls = String[]
    function fake_http(url, adql)
        push!(calls, adql)
        if occursin("gaiadr3.gaia_source", adql)
            occursin("source_id = 1234567890123456789", adql) &&
                return gaia_body("[[1234567890123456789, 14.36, 0.021]]")
            occursin("source_id = 42", adql) && return gaia_body("[[42, null, null]]")
            return gaia_body("[]")
        end
        # Simbad: every identifier of the object whose name matches
        occursin("'HIP 1'", adql) &&
            return simbad_body(["Gaia DR3 1234567890123456789"])
        occursin("'Two Gaia'", adql) &&
            return simbad_body(["Gaia DR3 1", "Gaia DR3 2"])
        return simbad_body(String[])
    end

    mktempdir() do dir
        @testset "by id, cached" begin
            empty!(calls)
            g = fetch_gaia_parallax(1234567890123456789; cache_dir = dir,
                                    http_get = fake_http, verbose = false)
            @test g.source_id == 1234567890123456789
            @test g.plx == 14.36 && g.plx_err == 0.021
            @test g.release == "DR3" && g.name === nothing
            @test length(calls) == 1
            # second call: no transport at all
            boom(args...) = error("network touched")
            g2 = fetch_gaia_parallax(1234567890123456789; cache_dir = dir,
                                     http_get = boom, verbose = false)
            @test g2 == g
            @test_throws ErrorException fetch_gaia_parallax(1234567890123456789;
                cache_dir = dir, http_get = boom, force_refetch = true, verbose = false)
        end

        @testset "by name resolves through Simbad" begin
            empty!(calls)
            g = fetch_gaia_parallax("HIP 1"; cache_dir = dir, http_get = fake_http,
                                    verbose = false)
            @test g.source_id == 1234567890123456789 && g.plx == 14.36
            @test g.name == "HIP 1"
            @test any(c -> occursin("ident", c) && occursin("'HIP 1'", c), calls)
            # name and parallax both cached now
            boom(args...) = error("network touched")
            @test fetch_gaia_parallax("HIP 1"; cache_dir = dir, http_get = boom,
                                      verbose = false).plx == 14.36
            # a string that is already an id skips Simbad
            empty!(calls)
            g3 = fetch_gaia_parallax("Gaia DR3 1234567890123456789"; cache_dir = dir,
                                     http_get = boom, verbose = false)
            @test g3.source_id == 1234567890123456789
        end

        @testset "names as people and catalogues spell them" begin
        # Simbad's match is exact on its own spelling; the fetcher must rewrite.
        known = Dict("HIP 4618" => "Gaia DR3 11", "HIP 103836" => "Gaia DR3 12",
                     "HD 114762" => "Gaia DR3 13", "GJ 15A" => "Gaia DR3 14",
                     "* eps Eri" => "Gaia DR3 15")
        asked = String[]
        simbad(url, adql) = begin
            m = match(r"id1\.id = '([^']*)'", adql)
            push!(asked, m.captures[1])
            haskey(known, m.captures[1]) ? simbad_body([known[m.captures[1]]]) : simbad_body(String[])
        end
        for (given, want) in ["HIP4618" => 11, "hip 4618" => 11, "  HIP   4618 " => 11,
                              "hip 103836" => 12, "HD114762" => 13, "hd 114762" => 13,
                              "GJ 15 A" => 14, "gj15 a" => 14, "eps Eri" => 15,
                              "EPS ERI" => 15]
            @test Nereus._gaia_name_to_id(given, mktempdir(), false, simbad, false) == want
        end
        @test "HIP 4618" in Nereus._simbad_name_candidates("HIP4618")
        @test Nereus._simbad_name_candidates("HIP 4618")[1] == "HIP 4618"   # as given first
        # the whole path, parallax included, for an unspaced name
        mktempdir() do dir
            both(url, adql) = occursin("gaiadr3", adql) ?
                gaia_body("[[11, 4.5, 0.02]]") : simbad(url, adql)
            g = fetch_gaia_parallax("HIP4618"; cache_dir = dir, http_get = both, verbose = false)
            @test g.source_id == 11 && g.plx == 4.5 && g.name == "HIP4618"
        end
    end

    @testset "errors say what to do" begin
            e = try fetch_gaia_parallax("Nobody"; cache_dir = dir, http_get = fake_http,
                                        verbose = false); nothing
                catch err; sprint(showerror, err) end
            @test e !== nothing && occursin("no Gaia DR3 identifier", e) &&
                  occursin("plx", e)
            e = try fetch_gaia_parallax("Two Gaia"; cache_dir = dir, http_get = fake_http,
                                        verbose = false); nothing
                catch err; sprint(showerror, err) end
            @test e !== nothing && occursin("several", e)
            # 2-parameter solution: no parallax
            e = try fetch_gaia_parallax(42; cache_dir = dir, http_get = fake_http,
                                        verbose = false); nothing
                catch err; sprint(showerror, err) end
            @test e !== nothing && occursin("42", e) && occursin("no usable parallax", e)
            # absent source
            e = try fetch_gaia_parallax(7; cache_dir = dir, http_get = fake_http,
                                        verbose = false); nothing
                catch err; sprint(showerror, err) end
            @test e !== nothing && occursin("not found", e)
            # transport failure names the way out
            bad(args...) = error("timeout")
            e = try fetch_gaia_parallax(99; cache_dir = dir, http_get = bad,
                                        verbose = false); nothing
                catch err; sprint(showerror, err) end
            @test e !== nothing && occursin("prior on plx", e)
        end
    end

    @testset "star.gaia_id forms" begin
        p = Nereus._parse_gaia_id
        @test p(123) == 123
        @test p("123") == 123
        @test p("Gaia DR3 123") == 123
        @test p(" gaia dr3  123 ") == 123
        @test_throws ArgumentError p("Gaia DR2 123")
        @test_throws ArgumentError p("HIP 5")
        @test_throws ArgumentError p(-1)
    end
end
