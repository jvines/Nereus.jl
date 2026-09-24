# The thread-slot contract (src/threading.jl).
#
# Julia >= 1.12 starts an interactive thread by default and numbers it FIRST, so
# a stock `julia` with no -t flag reports Threads.nthreads() == 1 while every
# Threads.@threads loop runs on thread id 2. Scratch allocated with
# `nthreads()` slots and indexed by `threadid()` therefore reads out of bounds
# on its first iteration — reported against ofti_sample on 1.13 as
#
#     BoundsError: attempt to access 1-element Vector{Theta{Float64}} at index [2]
#
# Nothing about it needs many threads: `julia -t 1,1` reproduces it on every
# version since 1.9. These tests pin the three things that keep it fixed — the
# chunk partition, the thread-id bound, and (in a child process that actually
# has an interactive thread) the samplers themselves.

# Its own imports: under sharding (test/shards.jl) this file can run in a process
# where no earlier file has loaded Nereus.
using Test, Nereus
using Nereus: _chunk_ranges, _nthread_slots, _nthread_chunks

@testset "thread slots" begin

    @testset "_chunk_ranges partitions exactly" begin
        for n in (0, 1, 2, 3, 5, 7, 8, 16, 17, 100, 1000)
            for k in (1, 2, 3, 4, 7, 16, 64)
                ch = _chunk_ranges(n, k)
                # Never more chunks than slots — this is what keeps the chunk
                # index in bounds of a k-long buffer array.
                @test length(ch) <= k
                @test length(ch) == min(n, k)
                @test all(!isempty, ch)
                # Exactly covers 1:n, in order, no gaps and no overlap.
                @test reduce(vcat, ch; init = Int[]) == collect(1:n)
                # Balanced to within one element.
                if !isempty(ch)
                    @test maximum(length, ch) - minimum(length, ch) <= 1
                end
            end
        end
    end

    @testset "_chunk_ranges honours an offset range" begin
        ch = _chunk_ranges(11:30, 3)
        @test length(ch) == 3
        @test first(first(ch)) == 11
        @test last(last(ch)) == 30
        @test reduce(vcat, ch) == collect(11:30)

        # A one-element range is one chunk however many slots are offered.
        @test _chunk_ranges(5:5, 8) == [5:5]
        # An empty range is no chunks at all: `for c in 1:0` then does nothing.
        @test isempty(_chunk_ranges(1:0, 4))
        @test isempty(_chunk_ranges(0, 4))
    end

    @testset "_nthread_slots bounds every thread id in use" begin
        # The invariant the reported bug violated. Threads.nthreads() does NOT
        # satisfy it: it counts the default pool while the id spans all pools.
        @test _nthread_slots() >= Threads.maxthreadid()
        @test _nthread_slots() >= Threads.threadid()
        seen = fill(false, _nthread_slots())
        Threads.@threads :static for _ in 1:(8 * max(1, Threads.nthreads()))
            seen[Threads.threadid()] = true   # BoundsError if the bound is wrong
        end
        @test any(seen)
        @test _nthread_chunks() >= 1
    end

    @testset "threadid() is named only where the scheduling is not ours" begin
        # Source guard for the whole class, over everything that ships: src/ and
        # the vendored NestedSamplers fork. A first cut matched only
        # `buf[Threads.threadid()]` written inline and so caught nothing —
        # every real site binds `tid = Threads.threadid()` first. So this one
        # is blunt on purpose: NAMING threadid() outside the allowlist fails,
        # whether it is used as an index one line later or ten.
        root = normpath(joinpath(@__DIR__, ".."))
        hits = String[]
        for sub in ("src", "vendor")
            for (dir, _, files) in walkdir(joinpath(root, sub)), f in files
                endswith(f, ".jl") || continue
                path = joinpath(dir, f)
                for line in eachline(path)
                    s = strip(line)
                    (isempty(s) || startswith(s, "#")) && continue
                    occursin("Threads.threadid()", s) &&
                        push!(hits, replace(relpath(path, root), "\\" => "/"))
                end
            end
        end
        # Everything else is chunk-keyed (`_chunk_ranges`), so it has no reason
        # to name a thread id at all. A new entry here needs a reason and
        # `_nthread_slots()` sizing.
        allowed = Set([
            "src/threading.jl",                       # the helper, documenting the rule
            "src/samplers/nested.jl",                 # NestedSamplers owns the scheduling
            "src/samplers/nuts.jl",                   # diagnostic message, not an index
            "vendor/NestedSamplers/src/parallel.jl",  # upstream fork, thread-keyed walks
        ])
        @test setdiff(Set(hits), allowed) == Set(String[])

        # And the two that do index by thread id size themselves by the real
        # bound on that id, never by the default-pool count.
        nested = read(joinpath(root, "src", "samplers", "nested.jl"), String)
        @test occursin("_nthread_slots()", nested)
        @test !occursin("for _ in 1:max(1, Threads.nthreads())", nested)
        vendored = read(joinpath(root, "vendor", "NestedSamplers", "src", "parallel.jl"), String)
        @test occursin("max(Threads.maxthreadid(), K)", vendored)
    end

    # ------------------------------------------------------------------
    # End-to-end: a child process with a non-empty interactive pool.
    #
    # `-t 1,1` puts the interactive thread at id 1 and the single worker at id
    # 2 on every Julia >= 1.9, which is exactly the topology a stock `julia`
    # has on >= 1.12. Every sampler below crashed with a BoundsError under it
    # before the chunk-keyed rewrite.
    # ------------------------------------------------------------------
    @testset "samplers run with an interactive thread present" begin
        script = joinpath(mktempdir(), "thread_slots_child.jl")
        write(script, raw"""
        using Nereus, Random, Statistics
        @assert Threads.nthreads() == 1
        @assert Threads.nthreads(:interactive) == 1

        function rv_target(seed = 2027; n_obs = 30)
            rng = MersenneTwister(seed)
            t   = sort(rand(rng, n_obs) .* 40.0)
            rv  = 12.0 .* sin.(2π .* t ./ 8.0) .+ randn(rng, n_obs)
            data = Data(; t_rv = t, rv = rv, rv_err = fill(1.0, n_obs))
            params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                             instruments = InstrumentConfig(rv = ["SIM"]),
                             data = data, M_s = 1.0)
            return NereusTarget(params, data; unconstrained = false), data
        end

        # Astrometry-only planet: Omega spans the full circle, so the node-flip
        # sweep actually fires (it is a no-op on the RV target).
        function as_target(seed = 11; n = 60)
            rng = MersenneTwister(seed)
            t   = sort(57000 .+ 1800 .* rand(rng, n))
            dt  = (t .- mean(t)) ./ 365.25
            psi = 2π .* rand(rng, n)
            pf  = sin.(2π .* (t .- 57000) ./ 365.25 .- psi)
            iad = Nereus.IADData(t = t, abscissa = 0.3 .* randn(rng, n),
                                 abscissa_err = fill(0.1, n), psi = psi,
                                 parallax_factor = pf, pm_factor = dt)
            blk = (a = LogUniformPrior(0.3, 4.0), M_sec = LogUniformPrior(0.001, 0.05),
                   sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                   inc = SinePrior(), Omega = UniformPrior(0.0, 2π),
                   Mo = UniformPrior(0.0, 2π))
            return build_target(M_pri = 0.644, planets = (b = blk,), iad = iad,
                                plx = NormalPrior(13.6, 0.02), M_s = 0.644)
        end

        function ofti_target()
            relast = RelAstromData(t = [55000.0, 55365.0, 55730.0],
                                   ra_off = [10.0, 12.0, 8.0], dec_off = [-5.0, -3.0, -7.0],
                                   ra_err = fill(1.0, 3), dec_err = fill(1.0, 3))
            data = Data(t_rv = [55000.0, 55300.0], rv = [0.0, 0.0],
                        rv_err = [5.0, 5.0], relastrom = relast)
            params = Params(max_kplanet = 1, planet_modes = [RVAS],
                instruments = InstrumentConfig(rv = ["X"]), data = data,
                stability = :none, M_s = 1.0,
                parametrization = ParametrizationConfig(mass = :a_driven),
                priors = Dict{String,PriorSpec}(
                    "n_p" => FixedPrior(1.0), "a_k1" => LogUniformPrior(0.5, 50.0),
                    "M_sec_k1" => LogUniformPrior(1e-5, 0.1),
                    "sesinw_k1" => UniformPrior(-1.0, 1.0),
                    "secosw_k1" => UniformPrior(-1.0, 1.0),
                    "Mo_k1" => UniformPrior(0.0, 2π), "inc_k1" => SinePrior(),
                    "Omega_k1" => UniformPrior(0.0, 2π),
                    "sigma_X" => LogUniformPrior(0.1, 50.0)))
            return NereusTarget(params, data)
        end

        tg, data = rv_target()
        ofti_sample(ofti_target(); n_attempts = 5_000, n_calibrate = 500,
                    buffer = 1.0, seed = 7, show_progress = false)
        sample_pt_emcee(tg, data; n_temps = 4, n_walkers = 16, n_steps = 60,
                        n_burnin = 30, seed = 1, show_progress = false,
                        bridge_headline = false)
        sample_pa(tg, data; n_replicas = 8, n_mcmc = 2, max_steps = 4,
                  seed = 0, show_progress = false)
        sample_nested_ins(tg, data; n_live = 30, dlogz = 5.0, max_iter = 150, seed = 1)
        sample_nested_dynamic(tg, data; n_live_init = 40, n_live_batch = 20,
                              dlogz_init = 5.0, max_iter = 150, seed = 1)
        sample_nested(tg, data; n_live = 30, dlogz = 5.0, seed = 1)
        sample_pt_whitening(tg, data; n_temps = 4, n_steps = 60, n_burnin = 30,
                            warmup_swaps = 20, seed = 1, show_progress = false)

        # The threaded BLS in the pre-fit transit search — same bug class,
        # reached with `threaded = true` from find_transits.
        let n = 600
            rng  = MersenneTwister(4)
            tt   = collect(range(0.0, 12.0; length = n))
            flux = 1.0 .+ 3e-4 .* randn(rng, n)
            half = (2.5 / 24) / 2
            flux[(tt .>= 6.0 .- half) .& (tt .<= 6.0 .+ half)] .*= 0.99
            Nereus.find_transits(tt, flux, fill(3e-4, n); detrend = :none,
                                 n_periods = 200, period_min = 0.5, period_max = 5.0)
        end

        # Node flip + trans-dim, on the astrometry target.
        ast = as_target()
        sample_pt_emcee(ast, ast.data; n_temps = 4, n_walkers = 16, n_steps = 60,
                        n_burnin = 30, seed = 1, show_progress = false,
                        bridge_headline = false)
        sample_pt_whitening(ast, ast.data; n_temps = 4, n_steps = 60, n_burnin = 30,
                            warmup_swaps = 20, seed = 1, show_progress = false)
        sample_transdim_pt_emcee(ast, ast.data; td = TransDimConfig(max_kplanet = 1),
                                 n_temps = 4, n_walkers = 16, n_steps = 60,
                                 n_burnin = 30, seed = 1, show_progress = false)
        println("THREAD_SLOTS_CHILD_OK")
        """)

        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=1,1 --project=$(Base.active_project()) $script`
        out = IOBuffer()
        ok  = success(pipeline(cmd; stdout = out, stderr = out))
        txt = String(take!(out))
        ok && occursin("THREAD_SLOTS_CHILD_OK", txt) ||
            @info "thread-slot child process output" txt
        @test ok
        @test occursin("THREAD_SLOTS_CHILD_OK", txt)
    end
end
