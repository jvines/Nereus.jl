# The hoisted Thiele-Innes reflex kernel against `star_reflex_offset`.
#
# `_reflex_kernel`/`_reflex_offset` (src/astrometry/projection.jl) exist only to
# avoid re-deriving one orbit's geometry at every abscissa of an IAD likelihood.
# They reach INTO PlanetOrbits' `Visual{KepOrbit}` -- its field names, its mean
# anomaly convention, its AU->mas factor and its reflex mass scaling -- so if
# PlanetOrbits ever changes any of those, nothing else in Nereus will notice.
# This file is that tripwire: it is the ONLY thing standing between an upstream
# change and a silently wrong astrometric orbit. If it fails, do not adjust the
# tolerance -- re-derive the kernel.

using PlanetOrbits
using Random
using ForwardDiff

@testset "Reflex kernel == star_reflex_offset" begin

    @testset "agreement over the parameter space these fits reach" begin
        rng = MersenneTwister(20260922)
        worst_rel = 0.0
        worst_abs = 0.0
        n_checked = 0
        for _ in 1:400
            P     = exp(log(30.0) + (log(30_000.0) - log(30.0)) * rand(rng))  # 30 d - 80 yr
            e     = 0.999 * rand(rng)
            ω     = 2π * rand(rng)
            Ω     = 2π * rand(rng)
            inc   = acos(2 * rand(rng) - 1)          # isotropic
            M_pri = 0.2 + 1.8 * rand(rng)
            M_sec = exp(log(1e-4) + (log(0.5) - log(1e-4)) * rand(rng))
            plx   = 1.0 + 99.0 * rand(rng)
            tp    = 50_000.0 + 5_000.0 * rand(rng)
            orb = Nereus.build_orbit(P, e, ω, Ω, inc, M_pri, M_sec, tp, plx)
            k   = Nereus._reflex_kernel(orb, M_sec)
            @test k isa Nereus.ReflexKernel          # fast path, not the fallback
            for _ in 1:25
                t = 40_000.0 + 25_000.0 * rand(rng)  # covers Hipparcos through DR4
                ra_ref, dec_ref = star_reflex_offset(orb, t, M_sec)
                ra_k,   dec_k   = Nereus._reflex_offset(k, t)
                scale = max(abs(ra_ref), abs(dec_ref), 1e-12)
                worst_rel = max(worst_rel, abs(ra_k - ra_ref) / scale,
                                           abs(dec_k - dec_ref) / scale)
                worst_abs = max(worst_abs, abs(ra_k - ra_ref), abs(dec_k - dec_ref))
                n_checked += 1
            end
        end
        @info "reflex kernel vs star_reflex_offset" n_checked worst_rel worst_abs
        # Rounding only. The Kepler solve is PlanetOrbits' own and E is
        # bit-identical; only the order of operations after it differs.
        @test worst_rel < 1e-10
    end

    @testset "circular and near-parabolic edges" begin
        for e in (0.0, 1e-8, 0.9, 0.99, 0.9999)
            orb = Nereus.build_orbit(1183.0, e, 1.1, 2.4, 1.05, 0.644, 0.01, 51_000.0, 13.628)
            k = Nereus._reflex_kernel(orb, 0.01)
            for t in (40_000.0, 48_349.0, 51_000.0, 57_388.0, 60_000.0)
                a_ref = star_reflex_offset(orb, t, 0.01)
                a_k   = Nereus._reflex_offset(k, t)
                @test a_k[1] ≈ a_ref[1] rtol=1e-10 atol=1e-14
                @test a_k[2] ≈ a_ref[2] rtol=1e-10 atol=1e-14
            end
        end
    end

    @testset "zero companion mass gives zero reflex" begin
        orb = Nereus.build_orbit(1183.0, 0.4, 1.1, 2.4, 1.05, 0.644, 0.0, 51_000.0, 13.628)
        k = Nereus._reflex_kernel(orb, 0.0)
        @test all(iszero, Nereus._reflex_offset(k, 55_000.0))
    end

    @testset "fallback path stays exact" begin
        # Anything the closed form does not own must route back to the reference
        # and agree with it EXACTLY, not to a tolerance.
        orb = Nereus.build_orbit(1183.0, 0.4, 1.1, 2.4, 1.05, 0.644, 0.01, 51_000.0, 13.628)
        fb = Nereus.ReflexFallback(orb, 0.01)
        for t in (45_000.0, 55_000.0)
            @test Nereus._reflex_offset(fb, t) === star_reflex_offset(orb, t, 0.01)
        end
        # A bare (non-Visual) orbit has no parallax, so no mas conversion: the
        # closed form must decline it rather than reach for `.parent`/`.dist`.
        bare = PlanetOrbits.KepOrbit(1.0, 0.3, 1.0, 0.5, 0.7, 51_000.0, 1.0)
        @test !Nereus._reflex_fast_applicable(bare)
        # A hyperbolic orbit likewise: sqrt(1 - e^2) does not exist there.
        hyp = Nereus.build_orbit(1183.0, 1.4, 1.1, 2.4, 1.05, 0.644, 0.01, 51_000.0, 13.628)
        @test !Nereus._reflex_fast_applicable(hyp)
        # ...and an ordinary bound orbit must be accepted, or the fast path is
        # silently dead and this whole file is testing nothing.
        @test Nereus._reflex_fast_applicable(orb)
    end

    @testset "_markley_sc returns upstream's E bit-for-bit" begin
        # `_markley_sc` is PlanetOrbits' Markley body with the sin/cos it
        # already computes handed back instead of thrown away. Every operation
        # feeding E is unchanged and in the same order, so E must come out
        # BIT-IDENTICAL to `kepler_solver(M, e, Markley())`. This is the
        # tripwire: if upstream changes the corrector, this fails loudly rather
        # than letting Nereus' copy drift into a silently different orbit.
        rng = MersenneTwister(4242)
        worst_sc = 0.0
        n_exact = 0
        n_tot = 0
        for _ in 1:20_000
            e = 0.9999 * rand(rng)
            M = (2 * rand(rng) - 1) * 40π          # well outside [-π, π]
            E_up = PlanetOrbits.kepler_solver(M, e, PlanetOrbits.Markley())
            E, sE, cE = Nereus._markley_sc(M, e)
            E === E_up && (n_exact += 1)
            n_tot += 1
            s_up, c_up = sincos(E_up)
            worst_sc = max(worst_sc, abs(sE - s_up), abs(cE - c_up))
            # and they really are the sine and cosine of the returned E
            @test sE^2 + cE^2 ≈ 1.0 atol=1e-14
        end
        @info "_markley_sc vs upstream" n_exact n_tot worst_sc
        @test n_exact == n_tot                      # E bit-identical, all of them
        # sin/cos come from the angle-sum reconstruction rather than a direct
        # `sincos(E)`, so they agree to a few ulp (measured worst 1.6e-15, ~7
        # ulp at unit magnitude), not bit-for-bit. Anything above this is a
        # real divergence, not rounding.
        @test worst_sc < 1e-14
    end

    @testset "grouped epoch solve == per-abscissa solve" begin
        # Real Gaia epoch data is per-CCD: 8-9 abscissae inside one ~40 s
        # field-of-view transit. `_iad_residuals_kernels!` solves Kepler once per
        # group and refines the siblings; `_iad_residuals_generic!` solves every
        # one. They must agree to rounding, or the fast path is quietly fitting a
        # different orbit. The synthetic fixture in test_iad_multi_instrument.jl
        # has its transits 15 d apart and therefore NO groups, so it cannot catch
        # this — hence a grouped fixture here.
        rng = MersenneTwister(31337)
        n_grp, per_grp = 40, 9
        t = Float64[]
        for g in 1:n_grp, i in 1:per_grp
            push!(t, 57000.0 + 12.0 * g + 4.6e-4 * (i - 1) / (per_grp - 1))
        end
        n = length(t)
        iad = IADData(t = t, abscissa = randn(rng, n), abscissa_err = fill(0.05, n),
                      psi = 2π .* rand(rng, n),
                      parallax_factor = sin.(2π .* (1:n) ./ 365.25),
                      pm_factor = (t .- t[1]) ./ 365.25)
        # the fixture really is grouped the way the fast path assumes
        @test length(unique(iad.grp_head)) == n_grp

        worst = 0.0
        r1 = Vector{Float64}(undef, n); r2 = similar(r1)
        for _ in 1:150
            P     = exp(log(0.8) + (log(20_000.0) - log(0.8)) * rand(rng))  # incl. ~1 d
            e     = 0.999 * rand(rng)
            M_sec = exp(log(1e-3) + (log(0.4) - log(1e-3)) * rand(rng))
            orb = Nereus.build_orbit(P, e, 2π*rand(rng), 2π*rand(rng),
                                     acos(2*rand(rng) - 1), 0.9, M_sec,
                                     57000.0 + 500*rand(rng), 20.0)
            ks = [Nereus._reflex_kernel(orb, M_sec)]
            Nereus._iad_residuals_kernels!(r1, iad, ks, 20.0)   # grouped
            Nereus._iad_residuals_generic!(r2, iad, ks, 20.0)   # per-abscissa
            worst = max(worst, maximum(abs.(r1 .- r2)))
        end
        @info "grouped vs per-abscissa residuals" worst
        # Well below any astrometric σ (Gaia along-scan is ~0.05 mas). Short
        # periods and e→1 are included above precisely because that is where the
        # neighbour refinement would fail; the residual guard sends those to a
        # full solve instead.
        @test worst < 1e-9
    end

    @testset "ForwardDiff flows through the kernel" begin
        # The IAD likelihood is differentiated (pt_hmc / NUTS / Pathfinder), so
        # the kernel has to carry Dual element types, not just Float64.
        f = v -> begin
            P, e, ω, Ω, inc, M_sec, plx = v
            orb = Nereus.build_orbit(P, e, ω, Ω, inc, 0.644, M_sec, 51_000.0, plx)
            k = Nereus._reflex_kernel(orb, M_sec)
            Δra, Δdec = Nereus._reflex_offset(k, 55_123.0)
            Δra^2 + Δdec^2
        end
        v0 = [1183.0, 0.4, 1.1, 2.4, 1.05, 0.01, 13.628]
        g = ForwardDiff.gradient(f, v0)
        @test all(isfinite, g)
        # Same gradient as the reference path, to rounding.
        f_ref = v -> begin
            P, e, ω, Ω, inc, M_sec, plx = v
            orb = Nereus.build_orbit(P, e, ω, Ω, inc, 0.644, M_sec, 51_000.0, plx)
            Δra, Δdec = star_reflex_offset(orb, 55_123.0, M_sec)
            Δra^2 + Δdec^2
        end
        @test g ≈ ForwardDiff.gradient(f_ref, v0) rtol=1e-8
    end
end
