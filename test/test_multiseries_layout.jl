# Data-only layout of the semiseparable multi-series GP (src/noise/multiseries_gp.jl).
#
# The time sort, the sorted series ids and the distinct epochs depend on
# (t_all, series_id) only, so multiseries_loglike builds them once per data set
# and looks them up afterwards, and evaluates the generators at the distinct
# epochs (the ActivityGP joint puts C channels on each RV epoch). Both must be
# invisible in the result: these tests compare against the row-by-row
# evaluation bit for bit, and check the lookup never hands out a layout built
# for different data. The cache keeps a layout while it is in use, so a fit
# cycling through more layouts than it holds must not rebuild and evict on
# every call (it did with 4 slots and 5 TESS sectors, each with its own GP).
using Nereus, Test
using Nereus: multiseries_loglike, _ms_layout, _ms_layout_build, _ms_fill_solve,
              _MSLayout, _MSLayoutCache, _MS_LAYOUT_CACHE, _generators, k0, mk2_0,
              SSSHO, SSESP, SSMEP, SSES, SSMatern32

# The evaluation before epoch sharing: generators on every sorted row, a zero
# gap between tied rows.
function rowwise_loglike(t_all, y_all, var_all, series_id, α, β, k; jitter = zeros(length(α)))
    ord = sortperm(t_all); t = Float64.(t_all[ord]); N = length(t)
    L = _MSLayout(Vector{Float64}(t_all), Vector{Int}(series_id), ord, t,
                  Vector{Int}(series_id[ord]), t, N > 1 ? diff(t) : Float64[],
                  Int[], false)
    G = _generators(k, t, L.dtu)
    return _ms_fill_solve(L, Vector{Float64}(y_all[ord]), var_all[ord], α, β, jitter,
                          Float64(k0(k)), Float64(mk2_0(k)), G..., G[5])
end

@testset "multiseries GP: data-only layout" begin
    # ActivityGP-like input: 4 channels on 40 shared epochs (C-fold ties), plus a
    # fifth series on its own grid, interleaved and unsorted.
    n = 40
    ep = [5.0 + 3.1 * i + 1.7 * sin(2.3 * i) for i in 1:n]
    perm = [mod(7i, n) + 1 for i in 0:(n - 1)]                  # a fixed shuffle
    own = [2.2 * i + 0.4 for i in 1:15]
    t = vcat(ep[perm], ep, ep[perm], ep, own)
    sid = vcat(fill(1, n), fill(2, n), fill(3, n), fill(4, n), fill(5, 15))
    N = length(t)
    y = [sin(0.29 * i) + 0.3 * cos(1.1 * i) for i in 1:N]
    v = [0.15 + 0.1 * abs(cos(0.7 * i)) for i in 1:N]
    α = [1.0, 0.6, -0.4, 0.2, 0.8]; β = [0.3, -0.2, 0.5, 0.0, 0.1]
    jit = [0.1, 0.0, 0.05, 0.2, 0.0]

    @testset "layout" begin
        L = _ms_layout_build(t, sid)
        @test L.ord == sortperm(t) && L.t == t[L.ord] && L.sid == sid[L.ord]
        @test L.has_ties && length(L.tu) == n + 15
        @test issorted(L.tu) && allunique(L.tu) && L.dtu == diff(L.tu)
        @test all(i -> L.t[i] === L.tu[L.row[i]], 1:N)
        # without ties the epochs are the sorted times themselves
        tt = t .+ 1e-3 .* (1:N)
        Ln = _ms_layout_build(tt, sid)
        @test !Ln.has_ties && Ln.tu === Ln.t && isempty(Ln.row) && Ln.dtu == diff(Ln.t)
        # a one-call layout carries no keys and is otherwise the same
        for (x, Lk) in ((t, L), (tt, Ln))
            Lu = _ms_layout_build(x, sid, false)
            @test isempty(Lu.t_all) && isempty(Lu.series_id)
            @test Lk.t_all == x && Lk.series_id == sid
            @test all(f -> getfield(Lu, f) == getfield(Lk, f),
                      (:ord, :t, :sid, :tu, :dtu, :row, :has_ties))
        end
    end

    kernels = (SSMEP(1.1, 12.0, 8.0, 0.5), SSESP(1.2, 15.0, 8.0, 0.6; nharm = 3),
               SSSHO(1.0, 9.0, 3.0), SSSHO(1.0, 9.0, 0.3), SSES(1.3, 9.0),
               SSMatern32(1.2, 8.0))
    @testset "bit-identical to row-by-row: $(nameof(typeof(k)))" for k in kernels
        ref = rowwise_loglike(t, y, v, sid, α, β, k; jitter = jit)
        @test isfinite(ref)
        @test multiseries_loglike(t, y, v, sid, α, β, k; jitter = jit) === ref
        @test multiseries_loglike(copy(t), y, v, copy(sid), α, β, k; jitter = jit) === ref
        # no ties at all, and one epoch only
        tt = t .+ 1e-3 .* (1:N)
        @test multiseries_loglike(tt, y, v, sid, α, β, k) === rowwise_loglike(tt, y, v, sid, α, β, k)
        t1 = fill(3.0, 6); s1 = [1, 2, 3, 1, 2, 3]
        @test multiseries_loglike(t1, y[1:6], v[1:6], s1, α[1:3], β[1:3], k) ===
              rowwise_loglike(t1, y[1:6], v[1:6], s1, α[1:3], β[1:3], k)
        @test multiseries_loglike([2.0], [0.3], [0.2], [1], [1.0], [0.0], k) ===
              rowwise_loglike([2.0], [0.3], [0.2], [1], [1.0], [0.0], k)
    end

    @testset "lookup is keyed on the bits of both inputs" begin
        # a cache of its own: what the shared one holds depends on earlier tests
        C0 = _MSLayoutCache()
        L1 = _ms_layout(C0, t, sid)
        @test _ms_layout(C0, copy(t), copy(sid)) === L1      # same data, new arrays
        @test _ms_layout(C0, view(copy(t), :), sid) === L1   # a view of the same bits
        sid2 = copy(sid); sid2[1], sid2[n + 1] = sid2[n + 1], sid2[1]
        L2 = _ms_layout(C0, t, sid2)
        @test L2 !== L1 && _ms_layout(C0, t, sid2) === L2   # same times, other series
        tz = [0.0, -0.0, 1.0]; L0 = _ms_layout(C0, tz, [1, 1, 1])
        Lz = _ms_layout(C0, [-0.0, 0.0, 1.0], [1, 1, 1])
        @test Lz !== L0 && !isempty(Lz.t_all)               # signed zeros differ
        @test length(L0.tu) == 3                             # and are distinct epochs
        @test length(@atomic C0.entries) == 4
        k = SSMEP(1.1, 12.0, 8.0, 0.5)
        @test multiseries_loglike(tz, [0.1, 0.2, 0.3], fill(0.2, 3), [1, 1, 1], [1.0], [0.2], k) ===
              rowwise_loglike(tz, [0.1, 0.2, 0.3], fill(0.2, 3), [1, 1, 1], [1.0], [0.2], k)
        C = _MSLayoutCache(slots = 4)
        for j in 1:12
            _ms_layout(C, t .+ j, sid)
        end
        @test length(@atomic C.entries) == 4
        @test length(@atomic _MS_LAYOUT_CACHE.entries) <= _MS_LAYOUT_CACHE.slots
        # non-Float64 / non-Int inputs are not cached but give the same answer
        ti = collect(1:30); si = Int32.(mod1.(1:30, 3))
        k2 = SSMatern32(1.2, 8.0)
        @test multiseries_loglike(ti, y[1:30], v[1:30], si, α[1:3], β[1:3], k2) ===
              multiseries_loglike(Float64.(ti), y[1:30], v[1:30], Int.(si), α[1:3], β[1:3], k2)
    end

    @testset "concurrent callers on alternating data sets" begin
        k = SSMEP(1.1, 12.0, 8.0, 0.5)
        sets = [(t .+ 0.5j, y .* (1 + 0.1j)) for j in 1:(_MS_LAYOUT_CACHE.slots + 4)]
        want = [rowwise_loglike(ts, ys, v, sid, α, β, k) for (ts, ys) in sets]
        got = Matrix{Float64}(undef, length(sets), 24)
        Threads.@threads for c in 1:24
            for (j, (ts, ys)) in enumerate(sets)
                got[j, c] = multiseries_loglike(ts, ys, v, sid, α, β, k)
            end
        end
        @test all(c -> got[:, c] == want && all(got[:, c] .=== want), 1:24)
    end

    # Keys of a cache's entries, in order.
    keys_of(C) = [E.L.t_all for E in (@atomic C.entries)]
    same_layout(A, B) = all(f -> getfield(A, f) == getfield(B, f),
                            (:ord, :t, :sid, :tu, :dtu, :row, :has_ties))

    @testset "rotation through more layouts than slots: no churn" begin
        C = _MSLayoutCache(slots = 4, idle = 1000)
        sets = [t .+ 0.25j for j in 1:5]                    # 5 layouts, 4 slots
        first = [_ms_layout(C, x, sid) for x in sets]
        E0 = @atomic C.entries
        @test length(E0) == 4 && keys_of(C) == sets[1:4]
        @test isempty(first[5].t_all)                        # the fifth went uncached
        for cyc in 1:50, (j, x) in enumerate(sets)
            L = _ms_layout(C, x, sid)
            j <= 4 ? (@test L === first[j]) :                # cached ones keep hitting
                     (@test L !== first[5] && isempty(L.t_all) && same_layout(L, first[5]))
        end
        @test (@atomic C.entries) === E0                     # never republished
        # at most slots: 4 layouts in turn all hit after the first round
        C4 = _MSLayoutCache(slots = 4, idle = 1000)
        got = [_ms_layout(C4, x, sid) for x in sets[1:4]]
        @test all(cyc -> all(j -> _ms_layout(C4, sets[j], sid) === got[j], 1:4), 1:20)
    end

    @testset "an idle entry gives way; one in use does not" begin
        C = _MSLayoutCache(slots = 3, idle = 30)
        A = [t .+ 0.5j for j in 1:3]; B = [t .- 0.5j for j in 1:3]
        foreach(x -> _ms_layout(C, x, sid), A)
        @test keys_of(C) == A
        # B arrives; nothing of A is idle yet, so B goes uncached
        @test all(x -> isempty(_ms_layout(C, x, sid).t_all), B)
        @test keys_of(C) == A
        # A1 stays in use while B cycles; A2 and A3 go idle and give way
        for _ in 1:15, x in (A[1], B...)
            _ms_layout(C, x, sid)
        end
        K = keys_of(C)
        @test length(K) == 3 && A[1] in K && count(k -> k in B, K) == 2
        @test !(A[2] in K) && !(A[3] in K)
    end

    @testset "row budget" begin
        C = _MSLayoutCache(slots = 8, rows = 2 * N + 10, idle = 1000)
        big = repeat(t, 3); sbig = repeat(sid, 3)          # 3N rows > budget
        @test isempty(_ms_layout(C, big, sbig).t_all)
        @test isempty(@atomic C.entries)
        foreach(j -> _ms_layout(C, t .+ j, sid), 1:4)
        @test length(@atomic C.entries) == 2                 # 2N rows fit, not 3N
        @test sum(E -> length(E.L.t), @atomic C.entries) <= C.rows
    end

    @testset "an uncached call allocates no key copies" begin
        n2 = 20_000
        tb = sort(collect(range(0.0, 3000.0; length = n2)) .+ 1e-3 .* sin.(1:n2))
        sb = ones(Int, n2)
        C = _MSLayoutCache(slots = 1, idle = 10^9)
        _ms_layout(C, tb .+ 1.0, sb)                        # fill the slot
        _ms_layout(C, tb, sb); _ms_layout_build(tb, sb, true)   # compile
        a_unc = @allocated _ms_layout(C, tb, sb)
        a_key = @allocated _ms_layout_build(tb, sb, true)
        # ord, t, sid, dtu: 4 vectors; the keyed build adds the 2 key copies
        @test a_unc < 5 * 8 * n2
        @test a_key - a_unc > 2 * 8 * n2 - 1024
    end

    @testset "global cache: >slots layouts in rotation, values exact" begin
        k = SSMEP(1.1, 12.0, 8.0, 0.5)
        K = _MS_LAYOUT_CACHE.slots + 3
        sets = [(t .+ 0.75j, y .* (1 - 0.02j)) for j in 1:K]
        want = [rowwise_loglike(ts, ys, v, sid, α, β, k; jitter = jit) for (ts, ys) in sets]
        ok = true
        for cyc in 1:4, (j, (ts, ys)) in enumerate(sets)
            ok &= multiseries_loglike(ts, ys, v, sid, α, β, k; jitter = jit) === want[j]
        end
        @test ok
    end

    @testset "concurrent lookups on a small cache" begin
        C = _MSLayoutCache(slots = 3, idle = 40)
        sets = [t .+ 0.3j for j in 1:7]
        ref = [_ms_layout_build(x, sid) for x in sets]
        bad = Threads.Atomic{Int}(0)
        Threads.@threads for c in 1:32
            for r in 1:60
                j = mod1(c + r, length(sets))
                L = _ms_layout(C, sets[j], sid)
                same_layout(L, ref[j]) || Threads.atomic_add!(bad, 1)
            end
        end
        @test bad[] == 0
        E = @atomic C.entries
        @test length(E) <= 3 && allunique([e.L.t_all for e in E])
        @test all(e -> e.L.t_all in sets, E)
    end
end
