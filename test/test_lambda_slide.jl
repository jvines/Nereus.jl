# The λ slide (src/samplers/lambda_slide.jl).
#
# At low e an RV orbit pins λ = Mo + ω, not either angle, and under
# (:Mo, :sesinw) that degenerate direction is a helix the linear stretch can
# only cut chords across. The move slides along it: (Mo, ω) → (Mo + δ, ω − δ).
#
# The assertions pin the things that make it a valid MCMC move rather than a
# plausible-looking one:
#   * the map preserves λ and e EXACTLY, which is the whole premise;
#   * T_{−δ} ∘ T_δ = id, the involutive pairing detailed balance needs;
#   * the Jacobian is 1 (a rotation composed with a translation);
#   * the window gate is ONE-SIDED. A window LONGER than a period is refused,
#     because there the 2π wrap folds two points onto one, T_{−δ} is no longer
#     the inverse, and detailed balance genuinely fails. A window SHORT of a
#     period is admitted, because `mod` is injective there and a proposal in
#     the gap is out of support and rejected, which is reversible. Getting this
#     one-sided is the bug this file exists to catch;
#   * a window that is not a circle at all is refused. (Mo under TTVs or
#     obs_prior is refused too, by delegating to `is_circular`, which carries
#     its own tests -- see src/circular.jl.)

using Nereus, Random, Statistics, Test
using LinearAlgebra: det
using Nereus: LambdaSlide, lambda_slides, lambda_slide!, CIRCULAR_PERIOD

@testset "λ slide" begin

    @testset "T preserves λ and e, exactly" begin
        rng = MersenneTwister(7)
        s = LambdaSlide(1, 1, 2, 3)
        for _ in 1:200
            e, ω, Mo = 0.9 * rand(rng), 2π * rand(rng), 2π * rand(rng)
            x = [Mo, sqrt(e) * sin(ω), sqrt(e) * cos(ω)]
            λ0 = mod2pi(x[1] + atan(x[2], x[3]))
            e0 = x[2]^2 + x[3]^2
            δ = 4 * (rand(rng) - 0.5) * π
            buf = copy(x)
            lambda_slide!(buf, s, δ, 0.0)
            @test buf[2]^2 + buf[3]^2 ≈ e0 atol = 1e-14
            @test mod2pi(buf[1] + atan(buf[2], buf[3])) ≈ λ0 atol = 1e-12
        end
    end

    @testset "T_{-δ} ∘ T_δ = id on a full-circle window" begin
        rng = MersenneTwister(8)
        s = LambdaSlide(1, 1, 2, 3)
        for lo in (0.0, -π, 4.21)                     # incl. a re-cut window
            for _ in 1:100
                e, ω = 0.5 * rand(rng), 2π * rand(rng)
                x = [lo + CIRCULAR_PERIOD * rand(rng),
                     sqrt(e) * sin(ω), sqrt(e) * cos(ω)]
                δ = 4 * (rand(rng) - 0.5) * π
                buf = copy(x)
                lambda_slide!(buf, s, δ, lo)
                lambda_slide!(buf, s, -δ, lo)
                @test buf ≈ x atol = 1e-12
            end
        end
    end

    @testset "the Jacobian is 1" begin
        s = LambdaSlide(1, 1, 2, 3)
        for δ in (0.3, -1.7, 2.9)
            J = zeros(3, 3)
            x = [1.0, 0.2, -0.3]
            h = 1e-6
            for j in 1:3
                xp = copy(x); xp[j] += h; lambda_slide!(xp, s, δ, 0.0)
                xm = copy(x); xm[j] -= h; lambda_slide!(xm, s, δ, 0.0)
                J[:, j] = (xp .- xm) ./ (2h)
            end
            @test abs(det(J)) ≈ 1.0 atol = 1e-6
        end
    end

    # ---- the window gate ------------------------------------------------
    # Built through the public model builder so the layout, names and priors
    # are the real ones.
    _blk(Mo) = (P = LogUniformPrior(1.0, 100.0), K = LogUniformPrior(0.1, 100.0),
                sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                Mo = Mo)
    _params(Mo) = build_target(
        planets = (b = _blk(Mo),),
        rv = (HARPS = (data = (t = collect(1.0:20.0), rv = zeros(20),
                               rv_err = ones(20)),
                       sigma = LogUniformPrior(0.01, 10.0)),)).params

    @testset "a full-circle window is admitted" begin
        pr = _params(UniformPrior(0.0, 2π))
        sl = lambda_slides(pr)
        @test length(sl) == 1
        @test pr.layout.unfrozen_names[sl[1].mo] == "Mo_k1"
        @test pr.layout.unfrozen_names[sl[1].se] == "sesinw_k1"
        @test pr.layout.unfrozen_names[sl[1].sc] == "secosw_k1"
    end

    @testset "a window longer than a period cannot reach the gate" begin
        # The model builder caps Mo at exactly 2π, so a window LONGER than a
        # period never reaches `lambda_slides` through the public API. That
        # matters because on such a window the 2π wrap would be a one-way fold
        # (its image is [lo, lo+2π), so the sliver past it is in the support but
        # never proposed into) and detailed balance would genuinely fail.
        @test_throws ArgumentError _params(UniformPrior(0.0, CIRCULAR_PERIOD + 3e-4))
    end

    @testset "a re-cut window over by ulps is still admitted" begin
        # The gate is one-sided but carries eps slop, because a re-cut window
        # stores `hi = lo + span` (src/circular.jl) and `(lo + 2π) - lo` is not
        # bit-exact: at lo = 4.21 it reads back 2π + 8.9e-16. A flat `<=` would
        # silently drop the move on a legitimate full-circle window, which is a
        # worse bug than the one it would be fixing.
        pr = _params(UniformPrior(0.0, 2π))
        i  = findfirst(==("Mo_k1"), pr.layout.unfrozen_names)
        Nereus.set_circular_window!(pr, i, 4.21)
        nlo, nhi = Nereus.bounds(pr.layout.unfrozen_priors[i])
        @test nhi - nlo >= CIRCULAR_PERIOD          # over by ulps, or exact
        @test length(lambda_slides(pr)) == 1
    end

    @testset "a window a hair SHORT of a period is admitted" begin
        # `mod` is injective on an interval shorter than a period, and a
        # proposal landing in the gap is out of support and rejected — which is
        # the reversible outcome. Refusing these would be a silent loss.
        short = UniformPrior(0.0, CIRCULAR_PERIOD - 3e-4)
        @test length(lambda_slides(_params(short))) == 1
    end

    @testset "a window that is not a circle at all is refused" begin
        @test isempty(lambda_slides(_params(UniformPrior(0.0, 3.0))))
    end

end
