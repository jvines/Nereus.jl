# The bridge fixed point and its bootstrap (src/samplers/bridge.jl), against a
# verbatim copy of the serial algorithm they replaced (Nereus 0.8.8).
#
# The bootstrap now solves its replicates in parallel, with buffers reused
# across fixed-point steps and replicates. Neither may change a bit of what
# bridge_evidence returns: the resamples come off the same MersenneTwister in
# the same order, every term is the same expression, and every reduction runs
# over the same vector. So the comparisons here are `===` / `isequal`, not
# tolerances, at every worker count.
using Test, Nereus, Random, Statistics, LinearAlgebra, MCMCChains, SpecialFunctions
using Nereus: _bridge_iterate, _bridge_iterate!, _bridge_bootstrap, _bridge_draw_indices!,
              _bridge_boot_workers, _bridge_boot_bytes, _BridgeBootBuffers, _logaddexp,
              _logsumexp, _SharedProgress, _tick!, ProgressBar, update!

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

function _ref_bridge_evidence(target, chains::MCMCChains.Chains;
                              n_proposal::Int = 20_000, seed::Int = 1,
                              proposal::Symbol = :student, max_iter::Int = 1_000,
                              tol::Real = 1e-10, n_bootstrap::Int = 200)
    names = target.params.layout.unfrozen_names
    d = length(names)
    d >= 1 || return (; log_z = NaN, se = NaN, n_post = 0, n_prop = 0,
                        iters = 0, converged = false, overlap = NaN,
                        proposal = proposal)
    cols = [vec(Array(chains[:, Symbol(nm), :])) for nm in names]
    N = length(cols[1])
    pt = target.transform
    Y_all = Matrix{Float64}(undef, d, N)
    @inbounds for i in 1:N
        xi = Float64[cols[j][i] for j in 1:d]
        Y_all[:, i] = pt === nothing ? xi : Nereus.transform_forward(xi, pt)
    end
    keep = vec(all(isfinite, Y_all; dims = 1))
    Y = Y_all[:, keep]
    N1 = size(Y, 2)
    N1 >= 2d + 2 || return (; log_z = NaN, se = NaN, n_post = N1, n_prop = 0,
                              iters = 0, converged = false, overlap = NaN,
                              proposal = proposal)
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
    overlap = n_finite / n_proposal
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

    @testset "fixed point: the 0.8.8 bits, no allocation per step" begin
        for (l1, l2) in (fast, slow, ext, unequal, reverse(unequal))
            for (mi, tol) in ((1_000, 1e-10), (3, 1e-10), (1_000, 1e-300))
                @test _bridge_iterate(l1, l2, mi, tol) === _ref_bridge_iterate(l1, l2, mi, tol)
            end
        end
        @test _bridge_iterate(slow..., 1_000, 1e-10)[2] > 100     # the slow case is slow
        @test _bridge_iterate(fast..., 1_000, 1e-10)[2] < 20
        # Empty sides return a non-answer, as before.
        @test isequal(_bridge_iterate(Float64[], [1.0], 10, 1e-8), (NaN, 0, false))
        @test isequal(_bridge_iterate([1.0], Float64[], 10, 1e-8), (NaN, 0, false))
        # Non-finite step: stops, unconverged, exactly as the reference does.
        nf = ([0.0, 1.0, Inf], [0.5, -Inf])
        @test isequal(_bridge_iterate(nf..., 50, 1e-10), _ref_bridge_iterate(nf..., 50, 1e-10))

        l1, l2 = slow
        t1 = similar(l1); t2 = similar(l2)
        _bridge_iterate!(t1, t2, l1, l2, 5, 1e-10)
        a5 = @allocated _bridge_iterate!(t1, t2, l1, l2, 5, 1e-10)
        a500 = @allocated _bridge_iterate!(t1, t2, l1, l2, 500, 1e-10)
        @test a500 == a5                    # nothing per step
        @test a5 < 1_000
        @test_throws DimensionMismatch _bridge_iterate!(t1[1:end-1], t2, l1, l2, 5, 1e-10)

        # `stop` ends it at the next step, unconverged.
        stop = Threads.Atomic{Bool}(true)
        @test _bridge_iterate!(t1, t2, l1, l2, 500, 1e-10; stop = stop)[2:3] == (0, false)
        seen = Int[]
        _bridge_iterate!(t1, t2, l1, l2, 4, 1e-300; on_iter = (it, δ) -> push!(seen, it))
        @test seen == 1:4
    end

    @testset "indices drawn in chunks are the bulk draw" begin
        chunk = Nereus._BRIDGE_IDX_CHUNK
        for (m, n) in ((1, 1), (7, 3), (chunk - 1, 10), (chunk, 10^6), (chunk + 1, 17),
                       (3chunk + 5, 3chunk + 5), (40_000, 9_800_000))
            ra = MersenneTwister(99); rb = MersenneTwister(99)
            bulk = rand(ra, 1:n, m)
            dst = Vector{Float64}(undef, m)
            _bridge_draw_indices!(dst, Vector{Int}(undef, min(chunk, m)), rb, n)
            @test dst == bulk
            @test all(Float64(bulk[i]) === dst[i] for i in 1:m)
            @test rand(ra, UInt64) == rand(rb, UInt64)     # same stream position
        end
    end

    @testset "replicates: the serial loop's, at every worker count" begin
        for (name, (l1, l2), nb, mi) in (("fast", fast, 24, 1_000), ("slow", slow, 9, 1_000),
                                         ("extreme", ext, 16, 1_000), ("unequal", unequal, 12, 1_000),
                                         ("capped", slow, 8, 20))
            ref = _ref_bootstrap(l1, l2, 41, nb, mi, 1e-10)
            for w in unique((1, 2, 3, 8, nt, nt + 3, nb + 5))
                res, ok, its = _bridge_bootstrap(l1, l2, MersenneTwister(42), nb, mi, 1e-10;
                                                 workers = w)
                @test all(i -> (res[i], its[i], ok[i]) === ref[i], 1:nb)
            end
        end
        # The capped case hits max_iter: replicates come back unconverged and
        # are kept in their slots, not dropped or reordered.
        _, okc, itc = _bridge_bootstrap(slow..., MersenneTwister(42), 8, 20, 1e-10; workers = 4)
        @test !any(okc) && all(==(20), itc)
        # Zero replicates: nothing drawn, nothing solved.
        rng = MersenneTwister(5)
        r0 = _bridge_bootstrap(fast..., rng, 0, 100, 1e-10; workers = 4)
        @test isempty(r0[1]) && rand(rng) == rand(MersenneTwister(5))
        r1, o1, _ = _bridge_bootstrap(fast..., MersenneTwister(42), 1, 100, 1e-10; workers = 4)
        @test (r1[1], o1[1]) == _ref_bootstrap(fast..., 41, 1, 100, 1e-10)[1][[1, 3]]
    end

    @testset "memory: one set of buffers per worker, not per replicate" begin
        l1, l2 = fast
        @test _bridge_boot_bytes(9_800_000, 20_000) == 16 * 9_820_000 + 8 * Nereus._BRIDGE_IDX_CHUNK
        @test _bridge_boot_workers(9_800_000, 20_000, 200, 10^12) ==
              min(nt, 200, Int(Sys.total_memory() ÷ 4) ÷ _bridge_boot_bytes(9_800_000, 20_000))
        @test _bridge_boot_workers(9_800_000, 20_000, 200, 0) == 1      # never zero
        @test _bridge_boot_workers(100, 10, 3, 10^12) <= 3
        f(nb) = @allocated _bridge_bootstrap(l1, l2, MersenneTwister(1), nb, 1_000, 1e-10; workers = 1)
        f(2)
        a10, a40 = f(10), f(40)
        buf = sizeof(Float64) * 2 * (length(l1) + length(l2))
        @test a10 < 2buf                    # the buffers and the result slots
        # A replicate allocates a few small objects, not a copy of the draws
        # (the serial loop allocated more than `buf` per replicate and step).
        @test (a40 - a10) / 30 < 2_000
    end

    @testset "a failing worker stops the rest and propagates" begin
        l1, l2 = slow                       # each solve is hundreds of steps
        calls = Threads.Atomic{Int}(0)
        steps_after = Threads.Atomic{Int}(0)
        function bad!(t1, t2, b1, b2, mi, tol; stop)
            Threads.atomic_add!(calls, 1) + 1 == 5 && error("replicate failed")
            # Steps taken once the flag is up: at most the one in progress.
            on = (it, δ) -> (stop[] && Threads.atomic_add!(steps_after, 1))
            return _bridge_iterate!(t1, t2, b1, b2, mi, tol; stop = stop, on_iter = on)
        end
        for w in unique((1, 4, nt))
            calls[] = 0; steps_after[] = 0
            @test_throws Exception _bridge_bootstrap(l1, l2, MersenneTwister(1), 200, 1_000,
                                                     1e-10; workers = w, solve! = bad!)
            n_calls = calls[]
            # Four solves were under way, or done, when the fifth threw; at most
            # one more per worker can start before the flag is up. Not 200.
            @test 5 <= n_calls <= 4 + 2w
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

    @testset "bridge_evidence: the 0.8.8 numbers" begin
        ref = _ref_bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, n_bootstrap = 30)
        @test ref.converged && isfinite(ref.se)
        tm = Dict{String, Any}()
        got = bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, n_bootstrap = 30,
                              timings = tm)
        @test isequal(got, ref)
        # One worker or many, a budget that fits one or all: the same answer.
        for mem in (0, 10^12)
            @test isequal(bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42,
                                          n_bootstrap = 30, bootstrap_memory = mem), ref)
        end
        # The stage record.
        for k in ("transform", "posterior_evals", "proposal_evals", "primary_solve", "bootstrap")
            @test tm[k]["seconds"] >= 0 && tm[k]["maxrss_bytes"] > 0
        end
        @test tm["n_draws"] == tm["n_post"] == size(ch, 1) * size(ch, 3)
        @test tm["n_post_finite"] == got.n_post && tm["n_prop_finite"] == got.n_prop
        @test tm["iters"] == got.iters && tm["converged"] === true
        @test tm["bootstrap_ok"] == 30 && 1 <= tm["bootstrap_workers"] <= nt

        # Zero and one replicate: no standard error, the same estimate.
        for nb in (0, 1)
            b = bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, n_bootstrap = nb)
            @test isnan(b.se) && b.log_z === ref.log_z
            @test isequal(b, _ref_bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42,
                                                  n_bootstrap = nb))
        end

        # A non-converged primary skips the bootstrap entirely.
        tm2 = Dict{String, Any}()
        b = bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42, max_iter = 1,
                            timings = tm2)
        @test !b.converged && isnan(b.se) && !haskey(tm2, "bootstrap")
        @test isequal(b, _ref_bridge_evidence(tg, ch; n_proposal = 3_000, seed = 42,
                                              max_iter = 1))

        # Proposal draws outside the support (a Gaussian proposal on a skewed
        # posterior loses some), and too few draws: the old early returns.
        @test isequal(bridge_evidence(tg, ch; n_proposal = 2_000, seed = 9, proposal = :gaussian,
                                      n_bootstrap = 12),
                      _ref_bridge_evidence(tg, ch; n_proposal = 2_000, seed = 9,
                                           proposal = :gaussian, n_bootstrap = 12))
        # Non-finite posterior draws are dropped before anything else, as
        # before: the one path on which the transformed draws are copied.
        A = copy(ch.value.data)
        j = findfirst(==(Symbol(tg.params.layout.unfrozen_names[1])), names(ch))
        A[3, j, 1] = NaN
        A[7, j, 2] = NaN                   # (an Inf can map to a finite point)
        bad = Chains(A, names(ch))
        tmb = Dict{String, Any}()
        gb = bridge_evidence(tg, bad; n_proposal = 3_000, seed = 42, n_bootstrap = 30,
                             timings = tmb)
        @test isequal(gb, _ref_bridge_evidence(tg, bad; n_proposal = 3_000, seed = 42,
                                               n_bootstrap = 30))
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
