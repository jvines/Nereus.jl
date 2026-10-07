# The ReverseDiff gradient sample_nuts integrates (src/samplers/nuts.jl).
#
# sample_nuts used to compile a ReverseDiff tape by default (`compile_tape =
# true`), recorded at zeros(dim). A compiled tape replays the operations made
# at the point where it was recorded, branches included, and every Nereus log
# density branches on the parameter values: Kepler's solver stops once it has
# converged, a point outside the support returns -Inf, transit windows and
# clamps choose what is computed. On the eccentric orbit below, zeros(dim) is
# e = 0, where Kepler's initial guess is already exact and the solver takes no
# Newton step; the tape then took none anywhere, and its values and gradients
# were wrong wherever e > 0.
using Test
using Nereus
using Random
using Logging
using ForwardDiff
import ReverseDiff, LogDensityProblems, LogDensityProblemsAD
using Statistics: median

const _RD_N = 40
const _RD_T, _RD_RV = let r = MersenneTwister(7)
    t = sort!(100 .* rand(r, _RD_N))
    t, [Nereus.rv_keplerian(ti, 4.23, 40.0, 0.4, 1.0, 0.0, median(t)) for ti in t] .+
       1.5 .* randn(r, _RD_N)
end

_rd_target() = build_target(
    planets = (b = (P = LogUniformPrior(4.0, 4.5), K = LogUniformPrior(10.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _RD_T, rv = _RD_RV, rv_err = fill(1.5, _RD_N)),
                 sigma = LogUniformPrior(0.5, 10.0)),))

# Points of the unconstrained space away from zeros(dim), eccentric all.
function _rd_points(d)
    r = MersenneTwister(11)
    return [0.8 .* randn(r, d) for _ in 1:5]
end

@testset "sample_nuts ReverseDiff: gradients right away from zeros(dim)" begin
    tg = _rd_target()
    d = LogDensityProblems.dimension(tg)
    f(y) = LogDensityProblems.logdensity(tg, y)
    ℓ, ∂ℓ = Nereus._nuts_logdensity_fns(tg, :ReverseDiff)
    for y in _rd_points(d)
        v, g = ∂ℓ(y)
        gf = ForwardDiff.gradient(f, y)
        @test v ≈ f(y) rtol = 1e-12
        @test ℓ(y) == f(y)
        @test maximum(abs.(g .- gf)) <= 1e-9 * maximum(abs, gf)
    end

    # Why a compiled tape is refused: recorded at zeros(dim), as
    # LogDensityProblemsAD records one, it is wrong at each of these points.
    ct = LogDensityProblemsAD.ADgradient(:ReverseDiff, tg; compile = Val(true))
    for y in _rd_points(d)
        gc = LogDensityProblems.logdensity_and_gradient(ct, y)[2]
        gf = ForwardDiff.gradient(f, y)
        @test maximum(abs.(gc .- gf)) > 1e-3 * maximum(abs, gf)
    end
end

@testset "sample_nuts ReverseDiff: no compiled tape" begin
    tg = _rd_target()
    # P, K, sesinw, secosw, Mo, gamma, sigma
    x0 = [4.23, 40.0, sqrt(0.4) * sin(1.0), sqrt(0.4) * cos(1.0), 0.3, 0.0, 1.5]
    kw = (n_warmup = 30, n_samples = 30, n_chains = 1, warm_start = false,
          init = x0, progress = false)
    err = try
        sample_nuts(tg; kw..., ad_backend = :ReverseDiff, compile_tape = true, seed = 3)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError && occursin("compile_tape", err.msg)
    # With the default the chain moves on the gradient of the log density: the
    # same as ForwardDiff's, so the same draws from the same seed.
    rd, fd = with_logger(ConsoleLogger(stderr, Logging.Error)) do
        (sample_nuts(tg; kw..., ad_backend = :ReverseDiff, seed = 3),
         sample_nuts(tg; kw..., ad_backend = :ForwardDiff, seed = 3))
    end
    @test rd.value.data ≈ fd.value.data rtol = 1e-6
end
