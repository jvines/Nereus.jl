# Data-only layout of the semiseparable multi-series GP (src/noise/multiseries_gp.jl).
#
# The time sort, the sorted series ids and the distinct epochs depend on
# (t_all, series_id) only, so multiseries_loglike builds them once per data set
# and looks them up afterwards, and evaluates the generators at the distinct
# epochs (the ActivityGP joint puts C channels on each RV epoch). Both must be
# invisible in the result: these tests compare against the row-by-row
# evaluation bit for bit, and check the lookup never hands out a layout built
# for different data.
using Nereus, Test
using Nereus: multiseries_loglike, _ms_layout, _ms_layout_build, _ms_fill_solve,
              _MSLayout, _MS_LAYOUT_CACHE, _MS_LAYOUT_SLOTS, _generators, k0, mk2_0,
              SSSHO, SSESP, SSMEP, SSES, SSMatern32

# The evaluation before epoch sharing: generators on every sorted row, a zero
# gap between tied rows.
function rowwise_loglike(t_all, y_all, var_all, series_id, α, β, k; jitter = zeros(length(α)))
    ord = sortperm(t_all); t = Float64.(t_all[ord]); N = length(t)
    L = _MSLayout(Vector{Float64}(t_all), Vector{Int}(series_id), ord, t,
                  Vector{Int}(series_id[ord]), t, N > 1 ? diff(t) : Float64[],
                  collect(1:N), false)
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
        L1 = _ms_layout(t, sid)
        @test _ms_layout(copy(t), copy(sid)) === L1          # same data, new arrays
        sid2 = copy(sid); sid2[1], sid2[n + 1] = sid2[n + 1], sid2[1]
        @test _ms_layout(t, sid2) !== L1                     # same times, other series
        tz = [0.0, -0.0, 1.0]; L0 = _ms_layout(tz, [1, 1, 1])
        @test _ms_layout([-0.0, 0.0, 1.0], [1, 1, 1]) !== L0 # signed zeros differ
        @test length(L0.tu) == 3                             # and are distinct epochs
        k = SSMEP(1.1, 12.0, 8.0, 0.5)
        @test multiseries_loglike(tz, [0.1, 0.2, 0.3], fill(0.2, 3), [1, 1, 1], [1.0], [0.2], k) ===
              rowwise_loglike(tz, [0.1, 0.2, 0.3], fill(0.2, 3), [1, 1, 1], [1.0], [0.2], k)
        for j in 1:(2 * _MS_LAYOUT_SLOTS)
            _ms_layout(t .+ j, sid)
        end
        @test length(@atomic _MS_LAYOUT_CACHE.entries) == _MS_LAYOUT_SLOTS
        # non-Float64 / non-Int inputs are not cached but give the same answer
        ti = collect(1:30); si = Int32.(mod1.(1:30, 3))
        k2 = SSMatern32(1.2, 8.0)
        @test multiseries_loglike(ti, y[1:30], v[1:30], si, α[1:3], β[1:3], k2) ===
              multiseries_loglike(Float64.(ti), y[1:30], v[1:30], Int.(si), α[1:3], β[1:3], k2)
    end

    @testset "concurrent callers on alternating data sets" begin
        k = SSMEP(1.1, 12.0, 8.0, 0.5)
        sets = [(t .+ 0.5j, y .* (1 + 0.1j)) for j in 1:(_MS_LAYOUT_SLOTS + 2)]
        want = [rowwise_loglike(ts, ys, v, sid, α, β, k) for (ts, ys) in sets]
        got = Matrix{Float64}(undef, length(sets), 24)
        Threads.@threads for c in 1:24
            for (j, (ts, ys)) in enumerate(sets)
                got[j, c] = multiseries_loglike(ts, ys, v, sid, α, β, k)
            end
        end
        @test all(c -> got[:, c] == want && all(got[:, c] .=== want), 1:24)
    end
end
