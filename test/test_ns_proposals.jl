# The vendored NestedSamplers proposals that `sample_nested` uses. Each one
# replaces a dead point with a chain of steps from a copied live point, and the
# chain has to move: every slice (Slice, RSlice) or walk step (RWalk, RStagger)
# starts where the previous one ended. The fork used to start every step from
# the copied point, so the new point was one step from it whatever
# `slices`/`walks` said, and the live set lost its spread (an RM obliquity fit
# came out with λ sd 1.6–2.1° against pt_emcee's 2.1–2.5°). The walks also drew
# again whenever a step left the unit cube, which favoured the middle of the
# cube; a step out of it is now a rejected step.
using Test, Random, Statistics, Distributions, Logging, NestedSamplers

const _NSP = NestedSamplers.Proposals
const _NSB = NestedSamplers.Bounds

@testset "nested proposals chain their steps" begin
    # Flat likelihood on the unit square: the constrained region is the whole
    # square, so a long chain forgets its start in the corner and ends uniform
    # (mean 1/2, sd 1/√12 ≈ 0.289). Before the fix: RSlice mean 0.21, Slice
    # 0.26, RWalk 0.36 with sd 0.21.
    flat = NestedModel(x -> 0.0, [Uniform(0, 1), Uniform(0, 1)])
    nb = _NSB.fit(_NSB.NoBounds, rand(MersenneTwister(0), 2, 4))
    for (name, mk) in (("RSlice", () -> _NSP.RSlice(slices = 30)),
                       ("Slice", () -> _NSP.Slice(slices = 10)),
                       ("RWalk", () -> _NSP.RWalk(walks = 200)),
                       ("RStagger", () -> _NSP.RStagger(walks = 200)),
                       ("HSlice", () -> _NSP.HSlice(slices = 10)))
        @testset "$name" begin
            rng = MersenneTwister(7)
            prop = mk()
            xs = with_logger(NullLogger()) do
                [prop(rng, [0.02, 0.02], -1.0, nb, flat)[1] for _ in 1:2000]
            end
            for j in 1:2
                c = getindex.(xs, j)
                @test all(x -> 0 < x < 1, c)
                @test mean(c) ≈ 0.5 atol = 0.03          # 4.6 standard errors
                @test std(c) ≈ 1 / sqrt(12) atol = 0.015
            end
        end
    end

    # The whole sampler on a correlated Gaussian inside the unit cube, whose
    # evidence is 1 (log Z = 0) and whose sd is 0.03 per coordinate. Before the
    # fix RWalk gave log Z +2.7 to +3.1 and sd 0.86–0.90 of the truth (seeds
    # 1–3); after it, |log Z| ≤ 0.24 and sd 0.97–1.02 for RWalk, RSlice and
    # Slice. The log Z error of one run is about 0.2 here.
    d = 6
    Σ = [0.03^2 * 0.8^abs(i - j) for i in 1:d, j in 1:d]
    mvn = MvNormal(fill(0.5, d), Σ)
    model = NestedModel(x -> logpdf(mvn, x), fill(Uniform(0, 1), d))
    for (name, mk) in (("RWalk", () -> _NSP.RWalk(walks = 25)),
                       ("RSlice", () -> _NSP.RSlice(slices = 2d)),
                       ("HSlice", () -> _NSP.HSlice(slices = d)))
        @testset "Gaussian, $name" begin
            spl = Nested(d, 300; bounds = _NSB.MultiEllipsoid, proposal = mk())
            ch, st = with_logger(NullLogger()) do
                sample(MersenneTwister(1), model, spl; dlogz = 0.1, progress = false)
            end
            w = vec(Array(ch[:weights]))
            w ./= sum(w)
            X = Array(ch)[:, 1:d]
            sds = [(m = sum(w .* X[:, j]); sqrt(sum(w .* (X[:, j] .- m) .^ 2))) for j in 1:d]
            @test abs(st.logz) < 0.6
            @test 0.9 < mean(sds) / 0.03 < 1.1
        end
    end
end

@testset "nested refits its bounds as it goes" begin
    # The bounds are refit once `update_interval` likelihood calls have passed
    # since the last fit. The check used to ask for an exact multiple of the
    # interval, which a count that grows by a whole proposal at a time almost
    # never hits: a run on this Gaussian refit 10 times with RWalk and twice
    # with RSlice, each time because a live point fell outside the stale bound,
    # and an RSlice run could stop with "Slice sampling appears to be stuck".
    # Now about 110 refits with RWalk and 40 with RSlice.
    d = 6
    Σ = [0.03^2 * 0.8^abs(i - j) for i in 1:d, j in 1:d]
    mvn = MvNormal(fill(0.5, d), Σ)
    model = NestedModel(x -> logpdf(mvn, x), fill(Uniform(0, 1), d))
    for (name, mk, atleast) in (("RWalk", () -> _NSP.RWalk(walks = 25), 60),
                                ("RSlice", () -> _NSP.RSlice(slices = 2d), 25))
        @testset "$name" begin
            nfit = Ref(0)
            seen = Ref{Any}(nothing)
            cb = (rng, m, spl, smp, st, i; kw...) -> begin
                b = st.active_bound
                b !== nothing && b !== seen[] && (nfit[] += 1)
                seen[] = b
            end
            spl = Nested(d, 300; bounds = _NSB.MultiEllipsoid, proposal = mk())
            ch, st = with_logger(NullLogger()) do
                sample(MersenneTwister(1), model, spl; dlogz = 0.1, progress = false, callback = cb)
            end
            @test nfit[] ≥ atleast
            @test abs(st.logz) < 0.6
        end
    end
end

@testset "HSlice leaves the constrained uniform distribution unchanged" begin
    # Points drawn uniformly inside a curved region (a banana in the unit
    # square) must stay uniform after one move, whatever the move does, if the
    # move is valid. The region is curved so reflections off it happen at
    # every angle, and the bound is a fitted, rotated ellipse so the velocities'
    # metric matters. The HSlice this replaced turned its slices along the
    # likelihood contour at the current point, a direction that depends on the
    # state: after one move from 6000 such points the mean of u2 moved by 6
    # standard errors and the sd of u1 by +0.018 (+0.025 after three slices).
    banana(u) = -(((u[1] - 0.5) / 0.3)^2 + ((u[2] - 0.3 - 1.5 * (u[1] - 0.5)^2) / 0.08)^2)
    model = NestedModel(banana, [Uniform(0, 1), Uniform(0, 1)])
    lstar = -1.0
    function inside(rng, m)
        out = Vector{Vector{Float64}}()
        while length(out) < m
            u = rand(rng, 2)
            banana(u) ≥ lstar && push!(out, u)
        end
        return out
    end
    rng = MersenneTwister(11)
    ref = inside(rng, 200_000)
    starts = inside(rng, 20_000)
    ell = _NSB.scale!(_NSB.fit(_NSB.Ellipsoid, reduce(hcat, inside(MersenneTwister(2), 500))), 1.25)
    M = length(starts)
    col(xs, j) = getindex.(xs, j)
    for slices in (1, 3)
        prop = _NSP.HSlice(slices = slices)
        out = [prop(MersenneTwister(1000 + i), copy(starts[i]), lstar, ell, model)[1] for i in 1:M]
        @test all(u -> banana(u) ≥ lstar, out)
        for j in 1:2
            sref = std(col(ref, j))
            @test abs(mean(col(out, j)) - mean(col(ref, j))) < 4 * sref / sqrt(M)
            @test abs(std(col(out, j)) - sref) < 4 * sref / sqrt(2M)
        end
        @test abs(cor(col(out, 1), col(out, 2)) - cor(col(ref, 1), col(ref, 2))) < 4 / sqrt(M)
    end
end

# A proposal that counts the starts it is handed below the likelihood threshold,
# then draws by rejection. Counted in a global: `sample_parallel` deep-copies
# the proposal for each thread.
const _NSP_STARTS = Ref((0, 0))
struct _NSPRecordStart <: _NSP.AbstractProposal end
NestedSamplers.default_update_interval(::_NSPRecordStart, ndims) = 1.5
function (::_NSPRecordStart)(rng, point, logl_star, bound, model; kwargs...)
    _, l = NestedSamplers.prior_transform_and_loglikelihood(model, point)
    n, below = _NSP_STARTS[]
    _NSP_STARTS[] = (n + 1, below + (l < logl_star))
    return _NSP.Rejection()(rng, point, logl_star, bound, model)
end

@testset "batch nested starts its walks inside the constraint" begin
    # Each batch replaces the K lowest live points with K walks constrained
    # above the K-th; a walk must start at a live point inside that constraint.
    # The batch path drew starts among all live points, the K-1 below the
    # threshold included: (K-1)/N of the walks, 14% here, started outside.
    model = NestedModel(x -> -sum(abs2, (x .- 0.5) ./ 0.01) / 2, fill(Uniform(0, 1), 2))
    spl = Nested(2, 50; bounds = _NSB.MultiEllipsoid, proposal = _NSPRecordStart())
    _NSP_STARTS[] = (0, 0)
    with_logger(NullLogger()) do
        NestedSamplers.sample_parallel(MersenneTwister(3), model, spl; batch_size = 8,
                                       parallel = false, dlogz = 0.5)
    end
    n, below = _NSP_STARTS[]
    @test n > 200
    @test below == 0
end
