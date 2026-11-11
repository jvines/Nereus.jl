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
                       ("RStagger", () -> _NSP.RStagger(walks = 200)))
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
                       ("RSlice", () -> _NSP.RSlice(slices = 2d)))
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
