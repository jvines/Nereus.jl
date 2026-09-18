# Sampler reproducibility must not depend on how many threads Julia was
# started with. This regressed once already: every ensemble sampler drew
# from a PER-THREAD RNG (`thread_rngs[threadid()]`), so `-t 2` and `-t 4`
# produced different chains at the same seed. On 51 Peg that moved the
# reported log-evidence by ~3.5 nats, which was briefly mistaken for a
# disagreement between two evidence estimators.
#
# The fix is to key every random stream on the TASK (walker / replica /
# attempt) index instead of the thread, so a stream is consumed by exactly
# one task in its own deterministic order.

using Test, Nereus, Random, Statistics

@testset "sampler determinism" begin

    # --- Tripwire: no sampler may draw randomness keyed on the thread ---
    # Cheap, and it fires the moment the anti-pattern is reintroduced —
    # unlike the subprocess check below, which only covers ptemcee.
    @testset "no per-thread RNG streams" begin
        srcdir  = joinpath(@__DIR__, "..", "src", "samplers")
        offenders = String[]
        for f in filter(endswith(".jl"), readdir(srcdir))
            txt = read(joinpath(srcdir, f), String)
            for (i, line) in enumerate(split(txt, '\n'))
                # a per-thread RNG vector, or any rng indexed by the tid
                if occursin("thread_rngs", line) ||
                   occursin(r"rngs?\[\s*tid\s*\]", line) ||
                   occursin(r"rngs?\[\s*Threads\.threadid\(\)\s*\]", line)
                    push!(offenders, "$f:$i: $(strip(line))")
                end
            end
        end
        if !isempty(offenders)
            @info "per-thread RNG streams found" offenders
        end
        @test isempty(offenders)
    end

    # --- The real thing: same seed, different -t, identical output ------
    @testset "ptemcee invariant to thread count" begin
        script = """
        using Nereus, Random, Printf
        Random.seed!(7)
        n = 60; t = sort!(400 .* rand(n))
        rv = 40.0 .* sin.(2π .* t ./ 4.23) .+ 1.5 .* randn(n)
        target = build_target(
            planets=(b=(P=LogUniformPrior(4.0,4.5), K=LogUniformPrior(10.0,90.0),
                        sesinw=UniformPrior(-1.0,1.0), secosw=UniformPrior(-1.0,1.0),
                        Mo=UniformPrior(0.0,2pi)),),
            rv=(SIM=(data=(t=t,rv=rv,rv_err=fill(1.5,n)),
                     sigma=LogUniformPrior(0.5,10.0)),))
        r = sample_ptemcee(target, target.data; n_temps=6, n_walkers=20,
                           n_steps=300, n_burnin=150, seed=42,
                           show_progress=false)
        K = vec(Array(r.chains[:, :K_k1, :]))
        @printf("%.17g %.17g %.17g\\n", sum(K), K[1], r.log_evidence_bridge)
        """
        sf = joinpath(mktempdir(), "det.jl"); write(sf, script)
        proj = dirname(@__DIR__)
        out = map((1, 2)) do nt
            s = read(`$(Base.julia_cmd()) --project=$proj -t $nt $sf`, String)
            strip(split(strip(s), '\n')[end])
        end
        @test out[1] == out[2]
    end
end
