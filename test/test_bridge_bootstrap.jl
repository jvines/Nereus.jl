# The bridge equation's solver and its bootstrap (src/samplers/bridge.jl),
# against a verbatim copy of the serial fixed-point iteration they replaced
# (Nereus 0.8.8).
#
# The solver is now safeguarded Newton on the same equation, so it agrees with
# the old iteration to within the old iteration's own stopping error, not to
# the bit: those comparisons are on the residual of the equation and on a
# tolerance a few orders below any reported standard error. The proposal's
# moments are now streamed off the chain rather than taken from a matrix of
# the draws, so they agree with mean/cov to rounding. What must still be exact
# is everything about scheduling: the bootstrap's resamples come off the same
# MersenneTwister in the same order (now as counts per draw), so its
# replicates are the same bits at any worker count, and `===` is used there.
using Test, Nereus, Random, Statistics, LinearAlgebra, MCMCChains, SpecialFunctions
using Nereus: _bridge_iterate, _bridge_g, _bridge_wextreme, _bridge_bootstrap,
              _bridge_draw_counts!, _bridge_skip_indices!, _bridge_boot_workers,
              _bridge_boot_bytes, _BridgeBootBuffers, _BridgeDraws, _bridge_draw!,
              _bridge_moments, _BridgeLogq, _bridge_logq!, _bridge_ratios,
              _logaddexp, _logsumexp, _SharedProgress, _tick!, ProgressBar, update!

# ---- the 0.8.8 code, verbatim -------------------------------------------------

function _ref_bridge_iterate(l1::Vector{Float64}, l2::Vector{Float64},
                             max_iter::Int, tol::Float64)
    n1, n2 = length(l1), length(l2)
    (n1 == 0 || n2 == 0) && return (NaN, 0, false)
    ls1 = log(n1 / (n1 + n2))
    ls2 = log(n2 / (n1 + n2))
    log_r = 0.5 * (mean(l1) + mean(l2))
    iters = 0
    converged = false
    for it in 1:max_iter
        iters = it
        num = _logsumexp(Float64[l2[j] - _logaddexp(ls1 + l2[j], ls2 + log_r)
                                 for j in 1:n2]) - log(n2)
        den = _logsumexp(Float64[-_logaddexp(ls1 + l1[i], ls2 + log_r)
                                 for i in 1:n1]) - log(n1)
        new = num - den
        isfinite(new) || break
        if abs(new - log_r) < tol
            log_r = new; converged = true; break
        end
        log_r = new
    end
    return log_r, iters, converged
end

# The bootstrap loop of 0.8.8's bridge_evidence, keeping every replicate's
# result (it pushed only the converged, finite ones into `ses`).
function _ref_bootstrap(l1, l2, seed, n_bootstrap, max_iter, tol)
    rb = MersenneTwister(seed + 1)
    out = Tuple{Float64, Int, Bool}[]
    for _ in 1:n_bootstrap
        b1 = l1[rand(rb, 1:length(l1), length(l1))]
        b2 = l2[rand(rb, 1:length(l2), length(l2))]
        push!(out, _ref_bridge_iterate(b1, b2, max_iter, Float64(tol)))
    end
    return out
end

# 0.8.8's matrix of the draws in the target's space, one column a draw.
function _ref_draws(target, chains::MCMCChains.Chains)
    names = target.params.layout.unfrozen_names
    d = length(names)
    cols = [vec(Array(chains[:, Symbol(nm), :])) for nm in names]
    N = length(cols[1])
    pt = target.transform
    Y_all = Matrix{Float64}(undef, d, N)
    @inbounds for i in 1:N
        xi = Float64[cols[j][i] for j in 1:d]
        Y_all[:, i] = pt === nothing ? xi : Nereus.transform_forward(xi, pt)
    end
    return Y_all
end

# 0.8.8's log ratios; only `N1` when there are too few draws.
function _ref_ratios(target, chains::MCMCChains.Chains;
                     n_proposal::Int = 20_000, seed::Int = 1, proposal::Symbol = :student)
    d = length(target.params.layout.unfrozen_names)
    Y_all = _ref_draws(target, chains)
    keep = vec(all(isfinite, Y_all; dims = 1))
    Y = Y_all[:, keep]
    N1 = size(Y, 2)
    N1 >= 2d + 2 || return (; N1)
    evs = [Nereus._BridgeEvaluator(target) for _ in 1:Nereus._bridge_ntasks(max(N1, n_proposal))]
    μ = vec(mean(Y; dims = 2))
    Σ = cov(Y; dims = 2) + 1e-10 * I(d)
    Lc = cholesky(Symmetric(Σ)).L
    ν  = 4.0
    function logq(y)
        z = Lc \ (y .- μ)
        q = sum(abs2, z)
        ld = logdet(Lc)
        if proposal === :gaussian
            return -0.5 * q - 0.5 * d * log(2π) - ld
        else
            return loggamma((ν + d) / 2) - loggamma(ν / 2) - 0.5 * d * log(ν * π) -
                   ld - 0.5 * (ν + d) * log1p(q / ν)
        end
    end
    rng = MersenneTwister(seed)
    nu_i = Int(ν)
    draw() = proposal === :gaussian ? μ .+ Lc * randn(rng, d) :
             μ .+ Lc * (randn(rng, d) / sqrt(sum(abs2, randn(rng, nu_i)) / ν))
    v1 = Vector{Float64}(undef, N1)
    Nereus._bridge_eval!(v1) do c, i
        y = @view Y[:, i]
        Nereus._bridge_logdensity!(evs[c], y) - logq(y)
    end
    l1 = Float64[v for v in v1 if isfinite(v)]
    Yp = Matrix{Float64}(undef, d, n_proposal)
    @inbounds for i in 1:n_proposal
        Yp[:, i] = draw()
    end
    v2 = Vector{Float64}(undef, n_proposal)
    Nereus._bridge_eval!(v2) do c, i
        Nereus._bridge_logdensity!(evs[c], @view Yp[:, i])
    end
    l2 = Float64[]
    n_finite = 0
    @inbounds for i in 1:n_proposal
        y = @view Yp[:, i]
        v = v2[i]
        if isfinite(v)
            n_finite += 1
            push!(l2, v - logq(y))
        end
    end
    return (; N1, l1, l2, overlap = n_finite / n_proposal)
end

function _ref_bridge_evidence(target, chains::MCMCChains.Chains;
                              n_proposal::Int = 20_000, seed::Int = 1,
                              proposal::Symbol = :student, max_iter::Int = 1_000,
                              tol::Real = 1e-10, n_bootstrap::Int = 200)
    d = length(target.params.layout.unfrozen_names)
    d >= 1 || return (; log_z = NaN, se = NaN, n_post = 0, n_prop = 0,
                        iters = 0, converged = false, overlap = NaN,
                        proposal = proposal)
    rr = _ref_ratios(target, chains; n_proposal, seed, proposal)
    N1 = rr.N1
    haskey(rr, :l1) || return (; log_z = NaN, se = NaN, n_post = N1, n_prop = 0,
                                 iters = 0, converged = false, overlap = NaN,
                                 proposal = proposal)
    l1, l2, overlap = rr.l1, rr.l2, rr.overlap
    (isempty(l1) || isempty(l2)) &&
        return (; log_z = NaN, se = NaN, n_post = N1, n_prop = length(l2),
                  iters = 0, converged = false, overlap = overlap,
                  proposal = proposal)
    log_r, iters, converged = _ref_bridge_iterate(l1, l2, max_iter, Float64(tol))
    ses = Float64[]
    if n_bootstrap > 0 && converged && isfinite(log_r)
        rb = MersenneTwister(seed + 1)
        for _ in 1:n_bootstrap
            b1 = l1[rand(rb, 1:length(l1), length(l1))]
            b2 = l2[rand(rb, 1:length(l2), length(l2))]
            r, _, ok = _ref_bridge_iterate(b1, b2, max_iter, Float64(tol))
            ok && isfinite(r) && push!(ses, r)
        end
    end
    se = length(ses) >= 2 ? std(ses) : NaN
    return (; log_z = log_r, se = se, n_post = length(l1), n_prop = length(l2),
              iters = iters, converged = converged, overlap = overlap,
              proposal = proposal)
end

# ---- ratio fixtures -------------------------------------------------------------

# Log ratios of a d-dimensional Gaussian posterior against the Student-t
# (ν = 4) proposal fitted to its draws, as bridge_evidence builds it. A
# fraction `stuck` of the posterior draws sits in a tight clump `shift` sigma
# out: walkers that never mixed. Each step of the fixed point then moves log r
# by nearly the same amount as the last, and a solve takes hundreds of steps
# instead of five.
function _bb_ratios(n1, n2; d = 6, stuck = 0.0, shift = 8.0, seed = 1, logZ = -321.0)
    rng = MersenneTwister(seed)
    A = randn(rng, d, d); S = A * A' / d + 0.5I
    L = cholesky(Symmetric(S)).L; ld = logdet(Symmetric(S))
    X = L * randn(rng, d, n1)
    ns = round(Int, stuck * n1)
    ns > 0 && (X[:, 1:ns] .= L * (0.2 .* randn(rng, d, ns) .+ shift .* normalize(ones(d))))
    logp(Y) = (q = vec(sum(abs2, L \ Y; dims = 1)); @. logZ - 0.5q - 0.5d * log(2π) - 0.5ld)
    μ = vec(mean(X; dims = 2)); Lc = cholesky(Symmetric(cov(X; dims = 2) + 1e-10I)).L
    c = loggamma((4 + d) / 2) - loggamma(2) - 0.5d * log(4π) - logdet(Lc)
    logq(Y) = (q = vec(sum(abs2, Lc \ (Y .- μ); dims = 1)); @. c - 0.5(4 + d) * log1p(q / 4))
    P = μ .+ Lc * (randn(rng, d, n2) ./ [sqrt(sum(abs2, randn(rng, 4)) / 4) for _ in 1:n2]')
    return logp(X) .- logq(X), logp(P) .- logq(P)
end

@testset "bridge bootstrap" begin
    nt = Threads.nthreads()
    fast = _bb_ratios(6_000, 1_500)
    # 50 posterior draws per proposal draw, as in production (5.8M : 20k is 290):
    # the stuck clump then sets the pace, 160 steps a solve.
    slow = _bb_ratios(20_000, 400; stuck = 0.02, shift = 6.0, seed = 2)
    # Extreme ratios: thousands of nats apart on both sides.
    ext = let (a, b) = _bb_ratios(3_000, 800; seed = 3)
        a[1:7:end] .+= 900.0; a[2:11:end] .-= 1200.0; b[1:5:end] .-= 700.0
        a, b
    end
    unequal = _bb_ratios(20_000, 50; seed = 4)      # N2 << N1, as in production

    # One solver pass, (g, g'), at x = log r: on the ratios, or on a resample
    # given as counts.
    function _gpass(l1, l2, x; w1 = nothing, w2 = nothing)
        n1 = w1 === nothing ? length(l1) : Int(sum(w1))
        n2 = w2 === nothing ? length(l2) : Int(sum(w2))
        return _bridge_g(l1, l2, w1, w2, log(n1 / (n1 + n2)), log(n2 / (n1 + n2)),
                         log(n1), log(n2), _bridge_wextreme(min, l1, w1, Inf),
                         _bridge_wextreme(max, l2, w2, -Inf), x)
    end
    # F(x) = g(x) - x, the residual of the bridge equation at x = log r.
    _residual(l1, l2, x; kw...) = _gpass(l1, l2, x; kw...)[1] - x

    @testset "solver: the old iteration's root, in a few passes" begin
        for (l1, l2) in (fast, slow, ext, unequal, reverse(unequal))
            x, it, ok = _bridge_iterate(l1, l2, 1_000, 1e-10)
            xr, itr, okr = _ref_bridge_iterate(l1, l2, 1_000, 1e-10)
            @test ok && okr
            @test it <= 10
            @test abs(_residual(l1, l2, x)) < 1e-9
            # The old iteration stopped within about its last step / (1 - |g'|)
            # of the root.
            @test abs(x - xr) < 1e-8
        end
        @test _ref_bridge_iterate(slow..., 1_000, 1e-10)[2] > 100   # the old way crawled
        @test _bridge_iterate(slow..., 1_000, 1e-10)[2] <= 10

        # g' as computed in the pass is the derivative of g.
        for (l1, l2) in (fast, slow, ext), x in (-330.0, -321.0, -300.0)
            _, dg = _gpass(l1, l2, x)
            h = 1e-5
            fd = (_gpass(l1, l2, x + h)[1] - _gpass(l1, l2, x - h)[1]) / 2h
            @test -1 < dg < 1
            @test abs(dg - fd) < 1e-6
        end

        # The pass shifts each logsumexp by the term at the smallest l1 and the
        # largest l2, found before the pass, instead of by the largest term
        # computed in it: the same g and g' as two passes and a buffer.
        function _gpass_ref(l1, l2, x)
            n1, n2 = length(l1), length(l2)
            ls1, ls2 = log(n1 / (n1 + n2)), log(n2 / (n1 + n2))
            c = ls2 + x
            t2 = [l2[j] - _logaddexp(ls1 + l2[j], c) for j in 1:n2]
            t1 = [-_logaddexp(ls1 + l1[i], c) for i in 1:n1]
            m1, m2 = maximum(t1), maximum(t2)
            e2 = exp.(t2 .- m2); e1 = exp.(t1 .- m1)
            g = (m2 + log(sum(e2)) - log(n2)) - (m1 + log(sum(e1)) - log(n1))
            dg = sum(e1 .* exp.(c .+ t1)) / sum(e1) - sum(e2 .* exp.(c .+ t2 .- l2)) / sum(e2)
            return g, dg
        end
        for (l1, l2) in (fast, slow, ext, unequal), x in (-330.0, -321.0, -300.0)
            g, dg = _gpass(l1, l2, x)
            gr, dgr = _gpass_ref(l1, l2, x)
            @test abs(g - gr) < 1e-12 * max(1, abs(gr)) && abs(dg - dgr) < 1e-12
        end
        # Far from the root, on both sides of where the pass leaves its
        # one-exp form for the log-space one (`_BRIDGE_G_FAST`): the same g.
        thr = Nereus._BRIDGE_G_FAST
        for (l1, l2) in (fast, ext, unequal)
            n1, n2 = length(l1), length(l2)
            δ = log(n1 / (n1 + n2)) - log(n2 / (n1 + n2))
            x1 = δ + minimum(l1) - thr      # the l1 term at the edge
            x2 = δ + maximum(l2) + thr      # the l2 term at the edge
            for x in (x1 - 1, x1 + 1, x1 - 900, x2 - 1, x2 + 1, x2 + 900)
                g, dg = _gpass(l1, l2, x)
                gr, dgr = _gpass_ref(l1, l2, x)
                @test abs(g - gr) < 1e-12 * max(1, abs(gr)) && abs(dg - dgr) < 1e-12
            end
            # Started out there, the same root.
            x0 = _bridge_iterate(l1, l2, 1_000, 1e-10)[1]
            for s in (x1 - 2000, x2 + 2000)
                xs, its, oks = _bridge_iterate(l1, l2, 1_000, 1e-10; start = s)
                @test oks && abs(xs - x0) < 1e-10
            end
        end

        # Counts: entry i standing for w[i] draws is the resample that repeats
        # it w[i] times.
        let (l1, l2) = fast, rng = MersenneTwister(7)
            w1 = UInt16.(rand(rng, 0:3, length(l1))); w2 = UInt16.(rand(rng, 0:2, length(l2)))
            r1 = reduce(vcat, (fill(l1[i], w1[i]) for i in eachindex(l1)))
            r2 = reduce(vcat, (fill(l2[j], w2[j]) for j in eachindex(l2)))
            for x in (-330.0, -321.0, -300.0)
                g, dg = _gpass(l1, l2, x; w1, w2)
                gr, dgr = _gpass(r1, r2, x)
                @test abs(g - gr) < 1e-12 * max(1, abs(gr)) && abs(dg - dgr) < 1e-12
            end
            x, it, ok = _bridge_iterate(l1, l2, 1_000, 1e-10; w1, w2)
            xr, _, okr = _bridge_iterate(r1, r2, 1_000, 1e-10)
            @test ok && okr && abs(x - xr) < 1e-10
            # Started at another point: the same root.
            xs, its, oks = _bridge_iterate(l1, l2, 1_000, 1e-10; w1, w2, start = x + 0.05)
            @test oks && abs(xs - x) < 1e-10 && its <= 4
            @test_throws DimensionMismatch _bridge_iterate(l1, l2, 5, 1e-10; w1 = w1[1:end-1])
        end

        # Many posteriors, clean and with stuck walkers of every size: the
        # safeguard keeps Newton converging, always in a handful of passes.
        worst = 0
        for seed in 1:40, stuck in (0.0, 0.002, 0.02, 0.1), shift in (4.0, 10.0)
            l1, l2 = _bb_ratios(4_000, 200; d = 3 + seed % 5, stuck = stuck, shift = shift,
                                seed = 100 + seed)
            x, it, ok = _bridge_iterate(l1, l2, 1_000, 1e-10)
            worst = max(worst, it)
            @test ok
            @test abs(_residual(l1, l2, x)) < 1e-9
        end
        @test worst <= 12

        # Capped: unconverged at max_iter.
        @test _bridge_iterate(slow..., 1, 1e-10)[2:3] == (1, false)
        # Empty sides return a non-answer, as before.
        @test isequal(_bridge_iterate(Float64[], [1.0], 10, 1e-8), (NaN, 0, false))
        @test isequal(_bridge_iterate([1.0], Float64[], 10, 1e-8), (NaN, 0, false))
        # Non-finite ratios: a NaN start, stopped unconverged, as the old
        # iteration did.
        nf = ([0.0, 1.0, Inf], [0.5, -Inf])
        @test isequal(_bridge_iterate(nf..., 50, 1e-10), _ref_bridge_iterate(nf..., 50, 1e-10))
        # A NaN ratio stops it, unconverged; with counts, counted or not.
        @test _bridge_iterate([0.0, NaN], [0.5, 1.0], 50, 1e-10)[3] == false
        for w1 in (UInt16[1, 1], UInt16[2, 0]), w2 in (UInt16[1, 1],)
            @test _bridge_iterate([0.0, NaN], [0.5, 1.0], 50, 1e-10; w1, w2)[3] == false
            @test _bridge_iterate([0.5, 1.0], [0.0, NaN], 50, 1e-10; w1 = w2, w2 = w1)[3] == false
        end

        # No buffers, and nothing allocated per pass, with counts or without.
        l1, l2 = slow
        w1 = ones(UInt16, length(l1)); w2 = ones(UInt16, length(l2))
        _bridge_iterate(l1, l2, 5, 1e-300); _bridge_iterate(l1, l2, 5, 1e-300; w1, w2)
        a5 = @allocated _bridge_iterate(l1, l2, 5, 1e-300)
        a500 = @allocated _bridge_iterate(l1, l2, 500, 1e-300)
        @test a500 == a5                    # nothing per pass
        @test a5 < 1_000
        aw5 = @allocated _bridge_iterate(l1, l2, 5, 1e-300; w1, w2)
        aw500 = @allocated _bridge_iterate(l1, l2, 500, 1e-300; w1, w2)
        @test aw500 == aw5 && aw5 < 1_000
        # Counts of one each are the ratios themselves.
        @test _bridge_iterate(l1, l2, 1_000, 1e-10; w1, w2)[1] ≈
              _bridge_iterate(l1, l2, 1_000, 1e-10)[1] atol = 1e-12

        # `stop` ends it at the next pass, unconverged.
        stop = Threads.Atomic{Bool}(true)
        @test _bridge_iterate(l1, l2, 500, 1e-10; stop = stop)[2:3] == (0, false)
        seen = Int[]
        _bridge_iterate(l1, l2, 2, 1e-300; on_iter = (it, δ) -> push!(seen, it))
        @test seen == 1:2
    end

    @testset "indices drawn in chunks, counted, are the bulk draw" begin
        chunk = Nereus._BRIDGE_IDX_CHUNK
        for (m, n) in ((1, 1), (7, 3), (chunk - 1, 10), (chunk, 10^6), (chunk + 1, 17),
                       (3chunk + 5, 3chunk + 5), (40_000, 9_800_000))
            ra = MersenneTwister(99); rb = MersenneTwister(99); rc = MersenneTwister(99)
            bulk = rand(ra, 1:n, m)
            want = zeros(UInt16, n)
            for j in bulk; want[j] += 1; end
            cnt = fill(0x0007, n)                          # cleared first
            idx = Vector{Int}(undef, min(chunk, m))
            _bridge_draw_counts!(cnt, idx, rb, m)
            @test cnt == want
            # Skipping takes the same numbers off the generator.
            _bridge_skip_indices!(idx, rc, m, n)
            u = rand(ra, UInt64)
            @test u == rand(rb, UInt64) == rand(rc, UInt64)   # same stream position
        end
        # Past what a count holds: an error, not a wrapped count.
        @test_throws OverflowError _bridge_draw_counts!(zeros(UInt16, 1), Vector{Int}(undef, 16),
                                                        MersenneTwister(1), 70_000)
    end

    @testset "replicates: the same bits at every worker count, the old resamples" begin
        for (name, (l1, l2), nb) in (("fast", fast, 24), ("slow", slow, 9),
                                     ("extreme", ext, 16), ("unequal", unequal, 12))
            # The serial loop's resamples, solved the old way: the same roots.
            ref = _ref_bootstrap(l1, l2, 41, nb, 1_000, 1e-10)
            r1, o1, i1 = _bridge_bootstrap(l1, l2, MersenneTwister(42), nb, 1_000, 1e-10;
                                           workers = 1)
            @test all(i -> o1[i] == ref[i][3] && abs(r1[i] - ref[i][1]) < 1e-8, 1:nb)
            @test all(<=(10), i1)
            for w in unique((2, 3, 8, nt, nt + 3, nb + 5))
                res, ok, its = _bridge_bootstrap(l1, l2, MersenneTwister(42), nb, 1_000, 1e-10;
                                                 workers = w)
                @test all(i -> (res[i], its[i], ok[i]) === (r1[i], i1[i], o1[i]), 1:nb)
            end
            # The counts are the old resamples: replicate k solved on the copy
            # 0.8.8 made of it, by the same solver, gives the same root.
            rb = MersenneTwister(42)
            for k in 1:min(nb, 4)
                b1 = l1[rand(rb, 1:length(l1), length(l1))]
                b2 = l2[rand(rb, 1:length(l2), length(l2))]
                @test abs(_bridge_iterate(b1, b2, 1_000, 1e-10)[1] - r1[k]) < 1e-10
            end
            # Started at the primary root, as bridge_evidence does: the same
            # roots in fewer passes, still the same at every worker count.
            x0 = _bridge_iterate(l1, l2, 1_000, 1e-10)[1]
            rw, ow, iw = _bridge_bootstrap(l1, l2, MersenneTwister(42), nb, 1_000, 1e-10;
                                           workers = 1, start = x0)
            @test ow == o1 && all(i -> abs(rw[i] - r1[i]) < 1e-10, 1:nb)
            @test all(i -> iw[i] <= i1[i], 1:nb) && sum(iw) < sum(i1)
            res, ok, its = _bridge_bootstrap(l1, l2, MersenneTwister(42), nb, 1_000, 1e-10;
                                             workers = 3, start = x0)
            @test all(i -> (res[i], its[i], ok[i]) === (rw[i], iw[i], ow[i]), 1:nb)
        end
        # Capped at max_iter: replicates come back unconverged and are kept in
        # their slots, not dropped or reordered.
        _, okc, itc = _bridge_bootstrap(slow..., MersenneTwister(42), 8, 2, 1e-10; workers = 4)
        @test !any(okc) && all(==(2), itc)
        # Zero replicates: nothing drawn, nothing solved.
        rng = MersenneTwister(5)
        r0 = _bridge_bootstrap(fast..., rng, 0, 100, 1e-10; workers = 4)
        @test isempty(r0[1]) && rand(rng) == rand(MersenneTwister(5))
        r1, o1, _ = _bridge_bootstrap(fast..., MersenneTwister(42), 1, 100, 1e-10; workers = 4)
        rr = _ref_bootstrap(fast..., 41, 1, 100, 1e-10)[1]
        @test o1[1] == rr[3] && abs(r1[1] - rr[1]) < 1e-8
    end

    @testset "memory: one set of buffers per worker, not per replicate" begin
        l1, l2 = fast
        @test _bridge_boot_bytes(9_800_000, 20_000) == 2 * 9_820_000 + 8 * Nereus._BRIDGE_IDX_CHUNK
        @test _bridge_boot_workers(9_800_000, 20_000, 200, 10^12) ==
              min(nt, 200, Int(Sys.total_memory() ÷ 4) ÷ _bridge_boot_bytes(9_800_000, 20_000))
        @test _bridge_boot_workers(9_800_000, 20_000, 200, 0) == 1      # never zero
        @test _bridge_boot_workers(100, 10, 3, 10^12) <= 3
        f(nb) = @allocated _bridge_bootstrap(l1, l2, MersenneTwister(1), nb, 1_000, 1e-10; workers = 1)
        f(2)
        a10, a40 = f(10), f(40)
        # The counts, the index chunk, the generator copy and the result slots:
        # well under one Float64 per draw.
        buf = sizeof(Float64) * (length(l1) + length(l2))
        @test a10 < buf + 8 * Nereus._BRIDGE_IDX_CHUNK + 50_000
        # A replicate allocates a few small objects, not a copy of the draws
        # (the serial loop allocated more than `buf` per replicate and step).
        @test (a40 - a10) / 30 < 2_000
    end

    @testset "a failing worker stops the rest and propagates" begin
        l1, l2 = slow
        calls = Threads.Atomic{Int}(0)
        steps_after = Threads.Atomic{Int}(0)
        function bad(l1, l2, c1, c2, mi, tol; start, stop)
            k = Threads.atomic_add!(calls, 1) + 1
            k == 5 && error("replicate failed")
            # A solve after the fifth waits for the flag. A warm Newton solve is
            # quicker than an exception unwinds, so without this the other
            # workers could run any number of solves before the flag went up;
            # with it, each holds at most one, whatever a solve costs.
            if k > 5
                t0 = time()
                while !stop[] && time() - t0 < 60
                    yield()
                end
            end
            # Steps taken once the flag is up: at most the one in progress.
            on = (it, δ) -> (stop[] && Threads.atomic_add!(steps_after, 1))
            return _bridge_iterate(l1, l2, mi, tol; w1 = c1, w2 = c2, start, stop,
                                   on_iter = on)
        end
        for w in unique((1, 4, nt))
            calls[] = 0; steps_after[] = 0
            @test_throws Exception _bridge_bootstrap(l1, l2, MersenneTwister(1), 200, 1_000,
                                                     1e-10; workers = w, solve = bad)
            n_calls = calls[]
            # Four solves were under way, or done, when the fifth threw; each
            # other worker can hold one more, waiting for the flag. Not 200.
            @test 5 <= n_calls <= 4 + w
            @test steps_after[] <= w - 1
            # Every worker has returned: nothing solves after the call is over.
            sleep(0.2)
            @test calls[] == n_calls
        end
    end

    @testset "shared progress from many threads" begin
        out = mktemp() do path, io
            redirect_stdout(io) do
                pb = ProgressBar("unit"; total = 1_000, print_every = 1e9)
                sp = _SharedProgress(pb; fields = () -> (:k => 1,))
                Threads.@threads for _ in 1:1_000
                    _tick!(sp)
                end
                # forced updates are never throttled
                update!(pb; n_done = 1_000, fields = (:stage => "x",), force = true)
            end
            flush(io); read(path, String)
        end
        lines = filter(!isempty, split(out, r"[\r\n]+"))
        @test count(l -> occursin("1000/1000 (100.0%)", l) && occursin("k=1", l), lines) == 1
        @test occursin("stage=x", lines[end])
    end

    # End to end: bridge_evidence against 0.8.8's, on a posterior it converges on.
    rng = MersenneTwister(20261025)
    n = 60
    t = sort!(900 .* rand(rng, n))
    rv = 40.0 .* sin.(2π .* t ./ 5.31) .+ 2.0 .* randn(rng, n)
    tg = build_target(
        planets = (b = (P = LogUniformPrior(5.0, 5.6), K = LogUniformPrior(10.0, 100.0),
                        sesinw = UniformPrior(-0.7, 0.7), secosw = UniformPrior(-0.7, 0.7),
                        Mo = UniformPrior(0.0, 2π)),),
        rv = (SIM = (data = (t = t, rv = rv, rv_err = fill(2.0, n)),
                     sigma = LogUniformPrior(0.5, 20.0)),))
    res = sample_pt_emcee(tg, tg.data; n_temps = 4, n_walkers = 24, n_steps = 900,
                          n_burnin = 500, seed = 3, bridge_headline = false,
                          show_progress = false)
    ch = res.chains

    @testset "draws streamed off the chain: no copy, the same numbers" begin
        # Draw i is column i of 0.8.8's matrix, to the bit.
        Y = _ref_draws(tg, ch)
        D = _BridgeDraws(ch, tg.params.layout.unfrozen_names, tg.transform)
        @test D.N == size(Y, 2) == size(ch, 1) * size(ch, 3)
        y = Vector{Float64}(undef, size(Y, 1))
        @test all(i -> _bridge_draw!(y, D, i) == Y[:, i] &&
                       all(j -> y[j] === Y[j, i], eachindex(y)), 1:D.N)
        # Moments: mean and cov of that matrix, to rounding.
        m = _bridge_moments(D)
        @test m.keep == vec(all(isfinite, Y; dims = 1)) && m.n == D.N
        @test norm(m.μ - vec(mean(Y; dims = 2))) <= 1e-12 * norm(vec(mean(Y; dims = 2)))
        @test norm(m.Σ - cov(Y; dims = 2)) <= 1e-12 * norm(cov(Y; dims = 2))
        @test issymmetric(m.Σ)

        # Many blocks, a ragged last one, non-finite draws in several, more
        # than one chain: still mean and cov of the finite draws, and the
        # same at any thread count (blocks are fixed by the draw count).
        B = Nereus._BRIDGE_MOMENT_BLOCK
        rng = MersenneTwister(11)
        A = randn(rng, cld(3B + 123, 3), 5, 3) .* [1e-3 1.0 50.0 1.0 1.0] .+ [7.0 -2.0 1e4 0.0 0.0]
        A[17, 2, 1] = NaN; A[B + 3, 1, 2] = Inf; A[end, 3, 3] = -Inf; A[5, 5, 2] = NaN
        cs = Chains(A, [:a, :b, :c, :d, :e])
        Ds = _BridgeDraws(cs, ["c", "a", "b", "d"], nothing)    # e, with a NaN, unused
        Ys = permutedims(reduce(hcat, [vec(Array(cs[:, s, :])) for s in (:c, :a, :b, :d)]))
        @test size(Ys) == (4, Ds.N)
        ks = vec(all(isfinite, Ys; dims = 1))
        ms = _bridge_moments(Ds)
        @test ms.keep == ks && ms.n == count(ks) == Ds.N - 3
        Yk = Ys[:, ks]
        @test norm(ms.μ - vec(mean(Yk; dims = 2))) <= 1e-12 * norm(vec(mean(Yk; dims = 2)))
        @test norm(ms.Σ - cov(Yk; dims = 2)) <= 1e-12 * norm(cov(Yk; dims = 2))
        @test_throws ArgumentError _BridgeDraws(cs, ["a", "nope"], nothing)

        # log q without allocating and without BLAS: the closure it replaces,
        # to rounding (its forward substitution is BLAS's, but not its kernel).
        d = size(Y, 1)
        μ = vec(mean(Y; dims = 2))
        Lc = cholesky(Symmetric(cov(Y; dims = 2) + 1e-10 * I(d))).L
        ν = 4.0
        for proposal in (:student, :gaussian)
            function logq(y)
                z = Lc \ (y .- μ)
                q = sum(abs2, z)
                ld = logdet(Lc)
                if proposal === :gaussian
                    return -0.5 * q - 0.5 * d * log(2π) - ld
                else
                    return loggamma((ν + d) / 2) - loggamma(ν / 2) - 0.5 * d * log(ν * π) -
                           ld - 0.5 * (ν + d) * log1p(q / ν)
                end
            end
            lq = _BridgeLogq(μ, Lc, proposal, ν)
            z = Vector{Float64}(undef, d)
            @test all(i -> isapprox(_bridge_logq!(z, lq, view(Y, :, i)), logq(view(Y, :, i));
                                    rtol = 1e-14, atol = 1e-12), 1:37:size(Y, 2))
            # Far out in the tails too, where q is large.
            @test all(s -> isapprox(_bridge_logq!(z, lq, μ .+ s .* Y[:, 3]),
                                    logq(μ .+ s .* Y[:, 3]); rtol = 1e-13), (1e2, 1e4, -1e3))
            # Through a function barrier, so the call is compiled for these
            # types and the test's own boxing of the result is not counted.
            alloc(z, lq, y) = @allocated _bridge_logq!(z, lq, y)
            yy = Y[:, 5]
            alloc(z, lq, yy)
            @test alloc(z, lq, yy) == 0
            v9 = view(Y, :, 9)
            alloc(z, lq, v9)
            @test alloc(z, lq, v9) == 0
        end

        # The log ratios: 0.8.8's, but for the rounding of the moments.
        for (kw, use_lp) in (((n_proposal = 3_000, seed = 42, proposal = :student), false),
                             ((n_proposal = 2_000, seed = 9, proposal = :gaussian), false),
                             ((n_proposal = 3_000, seed = 42, proposal = :student), true))
            rr = _ref_ratios(tg, ch; kw...)
            rs = _bridge_ratios(tg, ch, kw.n_proposal, kw.seed, kw.proposal, false, nothing;
                                use_lp)
            @test rs.n_post == rr.N1 && length(rs.l1) == length(rr.l1) &&
                  length(rs.l2) == length(rr.l2) && rs.overlap == rr.overlap
            tol = use_lp ? 1e-6 : 1e-12              # :lp is log p* to ~1e-9 nats
            @test maximum(abs.(rs.l1 .- rr.l1) ./ max.(1, abs.(rr.l1))) < tol
            @test maximum(abs.(rs.l2 .- rr.l2) ./ max.(1, abs.(rr.l2))) < 1e-12
        end
    end

    # The same estimate as 0.8.8: every count and the overlap exactly, log Z
    # and its standard error to far below that error. Only the solver, the
    # rounding of the proposal's moments (streamed, not from a matrix) and,
    # with the chain's :lp, the rounding of each log p* differ.
    function same_estimate(b, r)
        (b.converged == r.converged && b.n_post == r.n_post && b.n_prop == r.n_prop &&
         isequal(b.overlap, r.overlap) && b.proposal === r.proposal) || return false
        lz_ok = isnan(r.log_z) ? isnan(b.log_z) : abs(b.log_z - r.log_z) < 1e-8
        se_ok = isnan(r.se) ? isnan(b.se) : abs(b.se - r.se) < 1e-7
        return lz_ok && se_ok
    end

    @testset "bridge_evidence: the 0.8.8 estimate" begin
        ref = _ref_bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, n_bootstrap = 30)
        @test ref.converged && isfinite(ref.se)
        tm = Dict{String, Any}()
        got = bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, n_bootstrap = 30,
                              timings = tm)
        @test same_estimate(got, ref)
        @test got.iters <= 10
        # pt_emcee's :lp is log prior + log L in the bounded space: it matched
        # a full evaluation and stood in for one at every draw.
        @test tm["lp_reused"] === true
        @test tm["lp_check_max_abs"] < 1e-6
        # Evaluating every draw instead: the same terms as 0.8.8, to rounding.
        full = bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, n_bootstrap = 30,
                               use_lp = false)
        @test same_estimate(full, ref)
        @test abs(full.log_z - got.log_z) < 1e-9
        # One worker or many, a budget that fits one or all: the same bits.
        for mem in (0, 10^12)
            @test isequal(bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42,
                                          n_bootstrap = 30, bootstrap_memory = mem), got)
        end
        # The stage record.
        for k in ("transform", "posterior_evals", "proposal_evals", "primary_solve", "bootstrap")
            @test tm[k]["seconds"] >= 0 && tm[k]["maxrss_bytes"] > 0
        end
        @test tm["n_draws"] == tm["n_post"] == size(ch, 1) * size(ch, 3)
        @test tm["n_post_finite"] == got.n_post && tm["n_prop_finite"] == got.n_prop
        @test tm["iters"] == got.iters && tm["converged"] === true
        @test tm["bootstrap_ok"] == 30 && 1 <= tm["bootstrap_workers"] <= nt

        # An :lp that is not this target's log posterior is not used. Off by a
        # constant (a log joint on another scale, a tempered density) or
        # missing: every draw is evaluated, and the answer is the full one.
        A = copy(ch.value.data)
        jlp = findfirst(==(:lp), names(ch))
        A[:, jlp, :] .+= 0.5
        shifted = Chains(A, names(ch))
        tms = Dict{String, Any}()
        @test isequal(bridge_evidence(tg, shifted; n_proposal = 3_000, seed = 42,
                                      n_bootstrap = 30, timings = tms), full)
        @test tms["lp_reused"] === false && tms["lp_check_max_abs"] ≈ 0.5
        nolp = ch[:, filter(!=(:lp), names(ch)), :]
        tmn = Dict{String, Any}()
        @test isequal(bridge_evidence(tg, nolp; n_proposal = 3_000, seed = 42,
                                      n_bootstrap = 30, timings = tmn), full)
        @test tmn["lp_reused"] === false

        # Zero and one replicate: no standard error, the same estimate.
        for nb in (0, 1)
            b = bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, n_bootstrap = nb)
            @test isnan(b.se) && b.log_z === got.log_z
            @test same_estimate(b, _ref_bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42,
                                                        n_bootstrap = nb))
        end

        # A non-converged primary skips the bootstrap entirely.
        tm2 = Dict{String, Any}()
        b = bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, max_iter = 1,
                            timings = tm2)
        @test !b.converged && isnan(b.se) && !haskey(tm2, "bootstrap")

        # Proposal draws outside the support (a Gaussian proposal on a skewed
        # posterior loses some), and too few draws: the old early returns.
        @test same_estimate(bridge_evidence(tg, ch; n_proposal = 2_000, seed = 9,
                                            proposal = :gaussian, n_bootstrap = 12),
                            _ref_bridge_evidence(tg, ch; n_proposal = 2_000, seed = 9,
                                                 proposal = :gaussian, n_bootstrap = 12))
        # Non-finite posterior draws are dropped before anything else, as
        # before, and the :lp of the others still lines up with them.
        A = copy(ch.value.data)
        j = findfirst(==(Symbol(tg.params.layout.unfrozen_names[1])), names(ch))
        A[3, j, 1] = NaN
        A[7, j, 2] = NaN                   # (an Inf can map to a finite point)
        bad = Chains(A, names(ch))
        tmb = Dict{String, Any}()
        gb = bridge_evidence(tg, bad; n_proposal = 3_000, seed = 42, n_bootstrap = 30,
                             timings = tmb)
        @test same_estimate(gb, _ref_bridge_evidence(tg, bad; n_proposal = 3_000, seed = 42,
                                                     n_bootstrap = 30))
        @test tmb["lp_reused"] === true
        @test tmb["n_post"] == tmb["n_draws"] - 2 == gb.n_post
        @test gb.converged && isfinite(gb.se)
        few = ch[1:1, :, 1:2]
        @test isequal(bridge_evidence(tg, few), _ref_bridge_evidence(tg, few))
    end

    @testset "bridge_evidence progress lines" begin
        out = mktemp() do path, io
            redirect_stdout(io) do
                bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, n_bootstrap = 10,
                                show_progress = true)
            end
            flush(io); read(path, String)
        end
        lines = filter(!isempty, split(out, r"[\r\n]+"))
        npost = size(ch, 1) * size(ch, 3)
        @test any(l -> startswith(l, "bridge posterior") && occursin("$npost/$npost (100.0%)", l), lines)
        @test any(l -> startswith(l, "bridge proposal") && occursin("3000/3000", l), lines)
        @test any(l -> startswith(l, "bridge solve") && occursin("converged=true", l), lines)
        @test any(l -> startswith(l, "bridge bootstrap") && occursin("10/10 (100.0%)", l) &&
                       occursin("workers=", l), lines)
        # In that order.
        first_of(p) = findfirst(l -> startswith(l, p), lines)
        @test issorted([first_of("bridge posterior"), first_of("bridge proposal"),
                        first_of("bridge solve"), first_of("bridge bootstrap")])
    end
end
