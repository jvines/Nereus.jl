# Progress lines of the nested samplers. `sample_nested` advances its bar from
# a callback that AbstractMCMC calls on the serial path and the vendored fork's
# `sample_parallel` calls on the batch path; `sample_nested_dynamic` prints one
# bar per pass and `sample_nested_ins` one bar. Every line carries the
# iteration, logz, the remaining evidence dlogz, the stage (how far log dlogz has
# come toward the stopping threshold) and ncall.
using Test, Nereus, Random, NestedSamplers, Distributions

function _np_target(seed::Int = 2027; n_obs::Int = 30)
    rng  = MersenneTwister(seed)
    t    = sort(rand(rng, n_obs) .* 40.0)
    rv   = 12.0 .* sin.(2π .* t ./ 8.0) .+ randn(rng, n_obs)
    data = Data(; t_rv = t, rv = rv, rv_err = fill(1.0, n_obs))
    params = Params(; max_kplanet = 1, planet_modes = [RV_ONLY],
                    instruments = InstrumentConfig(rv = ["SIM"]), data = data, M_s = 1.0)
    return NereusTarget(params, data; unconstrained = false), data
end

# Everything `f` prints to stdout, one entry per line. Off a TTY the bar prints
# whole lines, so each update is one entry.
function _np_capture(f)
    out = mktemp() do path, io
        redirect_stdout(io) do
            f()
        end
        flush(io); read(path, String)
    end
    return filter(!isempty, split(out, r"[\r\n]+"))
end

_np_line(label) = Regex("^" * label * raw" \| iter (\d+) \| logz=\S+ \| dlogz=\S+ \| stage=(\d+)% \| ncall=\d+ \| elapsed ")
_np_lines(lines, label) = filter(l -> occursin(_np_line(label), l), lines)
_np_stage(l) = parse(Int, match(_np_line("nested"), l)[2])

const _NP_MODEL = NestedModel(x -> -0.5 * sum(abs2, (x .- 0.5) ./ 0.1),
                              [Uniform(0, 1), Uniform(0, 1)])

@testset "nested progress" begin
    target, data = _np_target()

    @testset "stage measure" begin
        s = Nereus._NestedStage(0.1)
        @test Nereus._stage!(s, -1e300, 1e300) == 0.0   # failed-likelihood sentinel
        @test Nereus._stage!(s, -Inf, Inf) == 0.0
        @test isnan(s.log_start)                         # neither one starts the scale
        @test Nereus._stage!(s, -500.0, 1000.0) == 0.0   # this one does
        @test Nereus._stage!(s, -100.0, 10.0) ≈ 0.5      # halfway in log Δ
        @test Nereus._stage!(s, -95.0, 100.0) ≈ 0.25     # Δ grew again: the stage falls
        @test Nereus._stage!(s, -90.0, 0.1) ≈ 1.0
        @test Nereus._stage!(s, -90.0, 0.0) == 1.0       # Δ underflowed
        @test Nereus._stage!(Nereus._NestedStage(0.5), -90.0, 0.2) == 1.0  # done at the start
        f = Nereus._nested_fields(Nereus._NestedStage(0.1), -100.0, 0.1, 7)
        @test first.(f) == (:logz, :dlogz, :stage, :ncall)
        @test last(f[3]) == "100%"
    end

    @testset "a full run goes from stage 0% to 100%" begin
        lines = _np_capture() do
            pb = ProgressBar("nested"; total = 0, print_every = 0.0)
            NestedSamplers.sample_parallel(MersenneTwister(1), _NP_MODEL, Nested(2, 50);
                                           batch_size = 3, parallel = false, dlogz = 1.0,
                                           callback = Nereus._nested_progress(pb, 1.0))
            Nereus.finish!(pb)
        end
        stages = _np_stage.(_np_lines(lines, "nested"))
        @test length(stages) > 2
        @test first(stages) == 0 && last(stages) == 100
        @test count(==(100), stages) == 1                # 100% only once the run stops
    end

    @testset "sample_parallel calls the callback after every batch" begin
        model = _NP_MODEL
        its, state_its = Int[], Int[]
        cb = (rng, model, sampler, sample, state, i) -> (push!(its, i); push!(state_its, state.it))
        NestedSamplers.sample_parallel(MersenneTwister(1), model, Nested(2, 50);
                                       batch_size = 3, parallel = false, dlogz = 1.0,
                                       callback = cb)
        @test length(its) > 2
        @test its == state_its
        @test first(its) == 1               # the bootstrap step
        @test all(==(3), diff(its))         # then one call per batch of 3
    end

    @testset "sample_nested, serial path" begin
        lines = _np_capture() do
            sample_nested(target, data; n_live = 40, dlogz = 2.0, proposal = :rwalk,
                          parallel = false, seed = 1)
        end
        hits = _np_lines(lines, "nested")
        @test !isempty(hits)
        @test match(_np_line("nested"), hits[1])[1] == "1"
    end

    @testset "sample_nested, batch path" begin
        lines = _np_capture() do
            sample_nested(target, data; n_live = 40, dlogz = 2.0, proposal = :rwalk,
                          parallel = true, batch_size = 2, seed = 1)
        end
        @test !isempty(_np_lines(lines, "nested"))
        @test isempty(_np_lines(lines, "nested serial"))   # no fallback
    end

    @testset "sample_nested_dynamic labels its two passes" begin
        lines = _np_capture() do
            sample_nested_dynamic(target, data; n_live_init = 80, n_live_batch = 50,
                                  dlogz_init = 2.0, max_iter = 1200, seed = 1)
        end
        base  = findall(l -> occursin(_np_line("nested baseline"), l), lines)
        batch = findall(l -> occursin(_np_line("nested batch 1"), l), lines)
        @test !isempty(base) && !isempty(batch)
        @test maximum(base) < minimum(batch)
    end

    @testset "sample_nested_ins" begin
        lines = _np_capture() do
            sample_nested_ins(target, data; n_live = 30, dlogz = 5.0, max_iter = 150, seed = 1)
        end
        @test !isempty(_np_lines(lines, "nested INS"))
    end

    @testset "show_progress = false prints nothing" begin
        lines = _np_capture() do
            sample_nested(target, data; n_live = 40, dlogz = 2.0, proposal = :rwalk,
                          parallel = false, seed = 1, show_progress = false)
            sample_nested_dynamic(target, data; n_live_init = 80, n_live_batch = 50,
                                  dlogz_init = 2.0, max_iter = 1200, seed = 1,
                                  show_progress = false)
            sample_nested_ins(target, data; n_live = 30, dlogz = 5.0, max_iter = 150,
                              seed = 1, show_progress = false)
        end
        @test !any(l -> startswith(l, "nested"), lines)
    end
end
