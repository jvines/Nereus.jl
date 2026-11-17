using Test, Nereus, Printf, Random, Logging

# Hipparcos-2 residual records as the 2014 Java tool writes them (the bundle's
# README): header lines 7 (isol_n), 9 and 11 (reference solution), then
# IORB EPOCH PARF CPSI SPSI RES SRES. `rows` are (EPOCH, PARF, CPSI, SPSI, RES,
# SRES), already rounded to the printed precision so they read back exactly.
function _hip2_record(path, rows; isol_n, plx = 10.82, pmra = 33.46, pmdec = -67.91,
                      acc = nothing, jrk = nothing, ups = nothing, var = nothing,
                      isol_line = true, ref_line = true)
    f(x) = x === nothing ? "---" : @sprintf("%.2f", x)
    e(x) = x === nothing ? "---" : "1.00"
    a = something(acc, (nothing, nothing))
    j = something(jrk, (nothing, nothing))
    u = something(ups, (nothing, nothing))
    open(path, "w") do io
        println(io, "# This file contains residual records, extracted from the Hipparcos 2")
        println(io, "#")
        if isol_line
            println(io, "# HIP    MCE    NRES NC isol_n SCE  F2     F1")
            @printf(io, "# %6d %6d %3d  %1d  %3d    %1d    %5.2f  %2d \n",
                    104148, 103791, length(rows), 1, isol_n, 0, 1.10, 0)
        end
        println(io, "# Hp      B-V    VarAnn NOB NR")
        @printf(io, "# %6.4f  %5.3f  %1d      %3d %2d \n", 5.856, 1.047, 0, length(rows), 0)
        if ref_line
            println(io, "# RAdeg        DEdeg        Plx      pm_RA    pm_DE    e_RA   e_DE   " *
                        "e_Plx  e_pmRA e_pmDE dpmRA  dpmDE  e_dpmRA  e_dpmDE  ddpmRA  ddpmDE  " *
                        "e_ddpmRA  e_ddpmDE  upsRA   upsDE   e_upsRA  e_upsDE  var   ")
            println(io, "# 316.50468086 -30.12495411 " * join([f(plx), f(pmra), f(pmdec),
                        "0.60", "0.49", "0.64", "0.77", "0.52",
                        f(a[1]), f(a[2]), e(a[1]), e(a[2]), f(j[1]), f(j[2]), e(j[1]), e(j[2]),
                        f(u[1]), f(u[2]), e(u[1]), e(u[2]), f(var)], "  "))
        end
        println(io, "#")
        println(io, "#   IORB  EPOCH    PARF    CPSI    SPSI     RES   SRES")
        for (k, r) in enumerate(rows)
            @printf(io, "  %4d %7.4f %7.4f %7.4f %7.4f %7.2f %6.2f\n", 100 + k, r...)
        end
    end
    return path
end

# Scans over the mission, rounded to the record's precision.
function _hip2_rows(rng, n; res = zeros(n))
    ep = sort(round.(-1.2 .+ 2.6 .* rand(rng, n); digits = 4))
    θ  = 2π .* rand(rng, n)
    cp, sp = round.(sin.(θ); digits = 4), round.(cos.(θ); digits = 4)
    pf = round.(0.8 .* sin.(2π .* ep .+ 0.3); digits = 4)
    return [(ep[i], pf[i], cp[i], sp[i], round(res[i]; digits = 2), 1.50) for i in 1:n]
end

@testset "Hipparcos-2 records: every solution type keeps its reference" begin
    rng = MersenneTwister(104148)
    rows = _hip2_rows(rng, 60; res = 0.4 .* randn(rng, 60))
    ep  = first.(rows); res = getindex.(rows, 5)
    # the along-scan unit vector, as ψ = atan(CPSI, SPSI) gives it
    ψ = atan.(getindex.(rows, 3), getindex.(rows, 4)); sψ, cψ = sin.(ψ), cos.(ψ)
    mktempdir() do dir
        read(isol_n; kw...) = @test_logs min_level = Logging.Warn Nereus._parse_van_leeuwen_iad(
            _hip2_record(joinpath(dir, "H104148_$(isol_n).d"), rows; isol_n, kw...))

        # 5p and stochastic: RES is against the five-parameter sky path as is.
        for (isol, kw) in ((5, (;)), (15, (;)), (1, (var = 0.17,)))
            d = read(isol; kw...)
            @test d.ref_params == [(0.0, 0.0, 10.82, 33.46, -67.91)]
            @test d.abscissa_kind == [:residual]
            @test d.abscissa == res
        end

        # 7p: the catalogue's acceleration goes back along scan, g·t²/2, so the
        # abscissae are O−C against the five-parameter path like every other
        # source's. Read as given, the curvature was missing from the data.
        g = (-6.47, -10.16)
        d7 = read(7; acc = g)
        @test d7.ref_params == [(0.0, 0.0, 10.82, 33.46, -67.91)]
        @test d7.abscissa ≈ res .+ (g[1] .* sψ .+ g[2] .* cψ) .* ep .^ 2 ./ 2 rtol = 1e-12
        @test read(57; acc = g).abscissa == d7.abscissa          # isol_n mod 10

        # 9p: plus the jerk, ġ·t³/6.
        j = (30.90, -1.98)
        d9 = read(9; acc = g, jrk = j)
        @test d9.abscissa ≈ res .+ (g[1] .* sψ .+ g[2] .* cψ) .* ep .^ 2 ./ 2 .+
                                    (j[1] .* sψ .+ j[2] .* cψ) .* ep .^ 3 ./ 6 rtol = 1e-12

        # VIM: the brightness-driven photocentre term stays removed; the record
        # has no per-transit magnitude to restore it with.
        dv = read(3; ups = (-22.15, -12.46))
        @test dv.ref_params == [(0.0, 0.0, 10.82, 33.46, -67.91)]
        @test dv.abscissa == res
    end
end

@testset "Hipparcos-2 records: restored terms are the ladder's coefficients" begin
    # A catalogue fit leaves RES orthogonal to its own design. With RES = 0 the
    # restored abscissae are exactly the higher-order path, and the closed-form
    # 7p/9p solution, in `astrom_design`'s Taylor basis and ψ convention, must
    # give back the catalogue's dpmRA/dpmDE (and ddpm) with nothing else.
    rng = MersenneTwister(9)
    rows = _hip2_rows(rng, 80)
    wide(k) = fill(1e6, k)
    mktempdir() do dir
        g, j = (-6.47, -10.16), (30.90, -1.98)
        d7 = Nereus._parse_van_leeuwen_iad(_hip2_record(joinpath(dir, "a.d"), rows;
                                                        isol_n = 7, acc = g))
        s7 = astrom_solution(d7, 7; prior_sigma = wide(7))
        @test s7.q[6:7] ≈ collect(g) atol = 1e-6
        @test maximum(abs, s7.q[1:5]) < 1e-6
        d9 = Nereus._parse_van_leeuwen_iad(_hip2_record(joinpath(dir, "b.d"), rows;
                                                        isol_n = 9, acc = g, jrk = j))
        s9 = astrom_solution(d9, 9; prior_sigma = wide(9))
        @test s9.q[6:9] ≈ [g..., j...] atol = 1e-6
        @test maximum(abs, s9.q[1:5]) < 1e-6
    end
end

@testset "Hipparcos-2 records: what cannot be read is refused" begin
    rows = _hip2_rows(MersenneTwister(3), 20)
    mktempdir() do dir
        p(name; kw...) = _hip2_record(joinpath(dir, name), rows; kw...)
        @test_throws "no catalogue reference solution" Nereus._parse_van_leeuwen_iad(
            p("noref.d"; isol_n = 5, ref_line = false))
        @test_throws "no solution type" Nereus._parse_van_leeuwen_iad(
            p("noisol.d"; isol_n = 5, isol_line = false))
        @test_throws "not a Hipparcos-2 solution type" Nereus._parse_van_leeuwen_iad(
            p("type4.d"; isol_n = 4))
        @test_throws "dpmRA and dpmDE" Nereus._parse_van_leeuwen_iad(
            p("noacc.d"; isol_n = 7))
        @test_throws "ddpmRA and ddpmDE" Nereus._parse_van_leeuwen_iad(
            p("nojerk.d"; isol_n = 9, acc = (1.0, 2.0)))
    end
end

@testset "a fit refuses residual IAD with no reference solution" begin
    # The likelihood samples the parallax and models ϖ − ref_params[3]: against
    # an all-zero reference the catalogue parallax would be subtracted twice.
    path = tempname() * ".iad"
    open(path, "w") do io
        for k in 0:9
            @printf(io, "%.4f %.3f %.3f %.2f 1.00 %.3f\n",
                    1990.0 + 0.3k, 0.6 * sin(k), -1.2 + 0.3k, 0.1k, 0.7k)
        end
    end
    iad = load_hip_iad(path)                          # reads: residual, no reference
    @test iad.abscissa_kind == [:residual] && iad.ref_params == [ntuple(_ -> 0.0, 5)]
    rv = (t_rv = [1.0, 2.0, 3.0], rv = zeros(3), rv_err = ones(3))
    @test_throws "all-zero ref_params" Data(; rv..., iad = iad)
    @test Data(; rv..., iad = load_hip_iad(path; ref_params = (0.0, 0.0, 10.82, 33.46, -67.91))) isa Data
    @test Data(; rv..., iad = load_hip_iad(path; abscissa_kind = :absolute)) isa Data
    rm(path)
end
