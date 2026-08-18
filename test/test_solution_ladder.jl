# Exact evidences for the 5/7/9-parameter astrometric solution ladder.
#
# The whole value of this is that the answer is CLOSED FORM — no sampling, no
# thermodynamic integration, no estimator error bar of the kind that turned out
# to dominate the noise-menu gates. So the tests have to verify the formula
# against something independent, not against itself.
#
#   1. brute-force numerical integration in a low-dimensional case;
#   2. recovery of an injected acceleration — the 7p rung must win when the data
#      really do curve, and must NOT when they don't;
#   3. the Occam direction: widening a prior can only ever LOWER that model's
#      evidence.
using Test
using Nereus
using Nereus: IADData, astrom_design, astrom_logZ, ladder_probabilities,
              default_ladder_prior, n_iad
using LinearAlgebra, Random, Statistics

# Synthetic scanning: psi sweeps so the 2-D information is actually there, and
# dt spans a mission-like baseline.
function fake_iad(; n = 120, seed = 7, accel = 0.0, jerk = 0.0, σ = 0.3,
                    da = 0.4, dd = -0.3, plx = 2.0, mua = 1.5, mud = -0.8)
    rng = MersenneTwister(seed)
    t   = collect(range(0.0, 1800.0; length = n))
    dt  = (t .- mean(t)) ./ 365.25
    psi = 2π .* rand(rng, n)
    pf  = cos.(2π .* t ./ 365.25)
    w = similar(dt)
    for j in 1:n
        s, c = sincos(psi[j])
        w[j] = da*s + dd*c + plx*pf[j] + mua*s*dt[j] + mud*c*dt[j] +
               accel*s*dt[j]^2/2 + jerk*s*dt[j]^3/6 + σ*randn(rng)
    end
    return IADData(t = t, abscissa = w, abscissa_err = fill(σ, n),
                   psi = psi, parallax_factor = pf, pm_factor = dt)
end

@testset "astrometric solution ladder" begin

    @testset "design matrix shape and columns" begin
        iad = fake_iad(n = 40)
        for o in (5, 7, 9)
            X = astrom_design(iad, o)
            @test size(X) == (40, o)
            @test all(isfinite, X)
        end
        X9 = astrom_design(iad, 9)
        X5 = astrom_design(iad, 5)
        @test X9[:, 1:5] ≈ X5           # nested: lower rungs are prefixes
        s, c = sincos(iad.psi[3]); dt = iad.pm_factor[3]
        @test X9[3, 6] ≈ s * dt^2 / 2
        @test X9[3, 9] ≈ c * dt^3 / 6
        @test_throws ArgumentError astrom_design(iad, 6)
    end

    # ---- 1. the formula, against brute force ----------------------------
    @testset "matches numerical integration" begin
        # Two free parameters only, so the integral is tractable on a grid.
        # Any error in the Occam terms (log|S|, log|M|) shows up here.
        n = 25
        rng = MersenneTwister(3)
        t = collect(range(0, 900; length = n)); dt = (t .- mean(t)) ./ 365.25
        psi = fill(π/2, n)                        # sinψ = 1, cosψ = 0
        pf = cos.(2π .* t ./ 365.25)
        σ = 0.5
        q_true = [1.2, 0.7]                       # (da*, plx)
        y = q_true[1] .+ q_true[2] .* pf .+ σ .* randn(rng, n)
        iad = IADData(t = t, abscissa = y, abscissa_err = fill(σ, n), psi = psi,
                      parallax_factor = pf, pm_factor = dt)

        ps = [3.0, 3.0]
        # analytic, restricted to the two columns that matter
        X = hcat(ones(n), pf)
        w = fill(1/σ^2, n)
        A = X' * (X .* w); v = X' * (y .* w)
        M = A + Diagonal(1 ./ ps.^2)
        yWy = sum(w .* y.^2)
        lz_analytic = -0.5*n*log(2π) + 0.5*sum(log, w) -
                      0.5*(yWy - dot(v, M \ v)) - sum(log, ps) - 0.5*logdet(M)

        # brute force: integrate prior x likelihood on a grid
        g = range(-6, 6; length = 601)
        lp = fill(-Inf, length(g), length(g))
        for (i, q1) in enumerate(g), (j, q2) in enumerate(g)
            r = y .- q1 .- q2 .* pf
            ll = -0.5*n*log(2π) + 0.5*sum(log, w) - 0.5*sum(w .* r.^2)
            pr = -0.5*log(2π*ps[1]^2) - q1^2/(2ps[1]^2) +
                 -0.5*log(2π*ps[2]^2) - q2^2/(2ps[2]^2)
            lp[i, j] = ll + pr
        end
        mx = maximum(lp); h = step(g)
        lz_numeric = mx + log(sum(exp.(lp .- mx)) * h^2)
        @test lz_analytic ≈ lz_numeric atol = 1e-3
    end

    # ---- 2. does it detect real curvature, and not imagine it? ----------
    @testset "prefers 7p only when there IS acceleration" begin
        flat = fake_iad(accel = 0.0, seed = 11)
        r = ladder_probabilities(flat)
        @test r.best == 5                       # no curvature ⇒ simplest wins
        @test r.prob[1] > 0.5

        curved = fake_iad(accel = 6.0, seed = 11)
        r2 = ladder_probabilities(curved)
        @test r2.best in (7, 9)                 # curvature ⇒ a higher rung
        @test r2.log_z[2] > r2.log_z[1]         # 7p beats 5p
        @test sum(r2.prob) ≈ 1.0
    end

    @testset "prefers 9p when there is jerk" begin
        j = fake_iad(accel = 4.0, jerk = 40.0, seed = 5)
        r = ladder_probabilities(j)
        @test r.log_z[3] > r.log_z[1]           # 9p beats 5p
        @test r.best in (7, 9)
    end

    # ---- 3. Occam direction ---------------------------------------------
    @testset "widening a prior lowers that model's evidence" begin
        # Data with NO acceleration, deliberately. Widening a prior is only an
        # Occam penalty when the prior already contains the posterior mass; if
        # the tight prior EXCLUDES the truth, widening lets the model finally
        # fit and evidence goes UP. That is not a bug, it is what a prior is —
        # but it means the default accel width (2*a0_max/accel_yr^2 = 0.1
        # mas/yr^2 for the defaults) is a real physical commitment about how
        # curved a long-period companion could plausibly make the track, and a
        # source outside it will be pushed onto the wrong rung. Sensitivity
        # analysis on this prior is mandatory, not optional.
        iad = fake_iad(accel = 0.0, seed = 2)
        tight = astrom_logZ(iad, 7; prior_sigma = default_ladder_prior(iad, 7))
        wide  = astrom_logZ(iad, 7;
                    prior_sigma = 100 .* default_ladder_prior(iad, 7))
        @test wide < tight
        # and the penalty is the log prior-volume ratio, asymptotically
        @test isfinite(wide) && isfinite(tight)
    end

    @testset "argument checking" begin
        iad = fake_iad(n = 12)
        @test_throws ArgumentError astrom_logZ(iad, 7; prior_sigma = ones(5))
        @test_throws ArgumentError astrom_logZ(iad, 7; prior_sigma = -ones(7))
        tiny = fake_iad(n = 6)
        @test_throws ArgumentError astrom_logZ(tiny, 9)
    end

    @testset "residual override" begin
        iad = fake_iad(accel = 5.0, seed = 9)
        # feeding the abscissae explicitly must reproduce the default
        @test astrom_logZ(iad, 7; residual = iad.abscissa) ≈ astrom_logZ(iad, 7)
        # pure noise as "residual" ⇒ no curvature to find
        rng = MersenneTwister(1)
        noise = 0.3 .* randn(rng, n_iad(iad))
        r = ladder_probabilities(iad; residual = noise)
        @test r.best == 5
    end
end
