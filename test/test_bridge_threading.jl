# How bridge sampling spreads its posterior evaluations over the threads
# (src/samplers/bridge.jl, `_bridge_eval!`).
#
# `bridge_evidence` evaluates the posterior at every kept draw plus every
# proposal draw: 1.5M evaluations after an HD 18599 production run. There are
# far more of them than threads, so each thread's task is marked with
# `_serial_inner_loops!` and the photometry's own threaded reduction runs
# serially inside it, instead of spawning one task per thread on every call
# that can only queue behind the other threads' evaluations. That reduction
# sums fixed-size chunks and combines them in order, so running it serially
# must not change a single bit.
using Test, Nereus, Random, MCMCChains
using Nereus: _bridge_eval!, _inner_loops_serial, _logdensity_parts

Random.seed!(23)
const _BT_NRV = 40
const _BT_TRV = sort!(300 .* rand(_BT_NRV))
const _BT_RV = 30.0 .* sin.(2π .* (_BT_TRV .- 1.2) ./ 3.11) .+ 2.0 .* randn(_BT_NRV)
const _BT_NPH = 10_000            # more than two photometry reduction chunks
const _BT_TPH = collect(range(0.0, 25.0; length = _BT_NPH))
const _BT_FLUX = let ph = @. abs(mod(_BT_TPH - 1.2 + 3.11 / 2, 3.11) - 3.11 / 2)
    f = 1.0 .+ 3e-4 .* randn(_BT_NPH); f[ph .< 0.05] .-= 0.006; f
end

_bt_target() = build_target(
    planets = (b = (P = UniformPrior(3.06, 3.16), Tc = UniformPrior(1.15, 1.25),
                    K = UniformPrior(5.0, 60.0), b = UniformPrior(0.0, 0.9),
                    rr = UniformPrior(0.02, 0.15), sesinw = UniformPrior(-0.5, 0.5),
                    secosw = UniformPrior(-0.5, 0.5)),),
    rv = (SIM = (data = (t = _BT_TRV, rv = _BT_RV, rv_err = fill(2.0, _BT_NRV)),
                 sigma = LogUniformPrior(0.5, 10.0)),),
    phot = (TESS = (data = (t = _BT_TPH, flux = _BT_FLUX, flux_err = fill(3e-4, _BT_NPH)),),),
    M_s = 1.0, R_s = 1.0)

# Bounded-space draws scattered about a point near the truth, `n` of them.
function _bt_draws(tg, n; rng = MersenneTwister(5))
    L = tg.params.layout
    near = Dict("P_k1" => 3.11, "Tc_k1" => 1.2, "K_k1" => 30.0, "b_k1" => 0.3,
                "rr_k1" => 0.075, "sesinw_k1" => 0.0, "secosw_k1" => 0.0)
    X = Matrix{Float64}(undef, n, length(L.unfrozen_names))
    for (j, nm) in enumerate(L.unfrozen_names)
        ps = L.unfrozen_priors[j]
        c = get(near, nm, isfinite(ps.lo) && isfinite(ps.hi) ? (ps.lo + ps.hi) / 2 : 0.0)
        w = isfinite(ps.lo) && isfinite(ps.hi) ? 0.01 * (ps.hi - ps.lo) : 0.01
        for i in 1:n
            X[i, j] = clamp(c + w * randn(rng), ps.lo + 1e-9, ps.hi - 1e-9)
        end
    end
    return X
end

@testset "bridge evaluation threading" begin
    nt = Threads.nthreads()
    blk = Nereus._BRIDGE_EVAL_BLOCK

    @testset "every index once, in place" begin
        for n in (0, 1, 3, blk, blk + 1, 7blk + 5, 1000)
            out = fill(NaN, n)
            calls = zeros(Int, n)
            task_of = zeros(Int, n)
            _bridge_eval!(out) do c, i
                calls[i] += 1
                task_of[i] = c
                2.0 * i
            end
            @test out == 2.0 .* (1:n)
            @test all(==(1), calls)
            # The task number indexes per-task scratch: always in range.
            @test all(c -> 1 <= c <= Nereus._bridge_ntasks(n), task_of)
        end
    end

    @testset "tasks are marked only when they fill the threads" begin
        n = 4 * nt * blk                      # a task for every thread
        marked = fill(false, n)
        _bridge_eval!(fill(NaN, n)) do _, i
            marked[i] = _inner_loops_serial()
            0.0
        end
        @test all(marked)
        # The mark is task-local: the caller is never marked.
        @test !_inner_loops_serial()
        if nt > 1
            # Fewer blocks than threads: threads are left over for the
            # likelihood's own loops, so those keep threading.
            n_small = blk
            marked_small = fill(true, n_small)
            _bridge_eval!(fill(NaN, n_small)) do _, i
                marked_small[i] = _inner_loops_serial()
                0.0
            end
            @test !any(marked_small)
        end
    end

    tg = _bt_target()
    @test length(tg.data.t_phot) > 2 * Nereus._PHOT_REDUCE_CHUNK
    X = _bt_draws(tg, 64)
    Y = reduce(hcat, [Nereus.transform_forward(X[i, :], tg.transform) for i in 1:size(X, 1)])
    logp(y) = (a = _logdensity_parts(tg, y); a[1] + a[2])

    @testset "marked tasks give the caller's bits" begin
        # In the caller's task the photometry reduction threads its chunks;
        # inside `_bridge_eval!` it runs them serially. Identical, not close.
        ref = [logp(view(Y, :, i)) for i in 1:size(Y, 2)]
        @test all(isfinite, ref)
        got = Vector{Float64}(undef, size(Y, 2))
        _bridge_eval!((_, i) -> logp(view(Y, :, i)), got)
        @test all(got .=== ref)
    end

    @testset "bridge_evidence is reproducible" begin
        X2 = _bt_draws(tg, 600; rng = MersenneTwister(9))
        ch = Chains(reshape(X2, size(X2, 1), size(X2, 2), 1),
                    Symbol.(tg.params.layout.unfrozen_names))
        b1 = bridge_evidence(tg, ch; n_proposal = 400, seed = 3, n_bootstrap = 10)
        b2 = bridge_evidence(tg, ch; n_proposal = 400, seed = 3, n_bootstrap = 10)
        @test b1.n_post == 600
        @test isfinite(b1.log_z)
        @test isequal(b1, b2)
    end
end
