# pt_whitening checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any step and continued with `resume = true` is
# bit-identical to one that ran straight through, so a run that has not mixed
# can simply be extended. The split points below cross everything that carries
# state between steps: the whitening ring buffer part-filled (step 12) and
# wrapped (window 20, step 25), identity swaps giving way to whitened ones (step
# 31), the (μ, σ) refresh cadence (every 7 steps from 31, so step 43 stops
# between refreshes and the saved (μ, σ) lag the ring), the burn-in re-cut of
# the circular Mo (steps 60 and 120: the ring relabelled, the window moved), the
# end of burn-in, and a finished run extended past its own end. A second target,
# with an astrometry-only planet, carries the node-flip RNG streams across a
# split. Both check what a resume refuses: different data in any field of
# `Data` (`t_ref` and the IAD records included, which `run_fingerprint` does not
# hash), or a different setting, named by its keyword.
using Test
using Nereus
using Nereus: IADData, Data
using Random

# An eccentric planet whose mean anomaly sits 0.02 rad from the 0/2π seam, with
# priors narrow enough that the cold walkers gather there within burn-in, so the
# re-cut has a seam-centred Mo to move (as in test_circular.jl). Measured on
# arm64 under seed 42: it moves at both re-cuts, and the second move's hi comes
# out an ulp off lo + 2π, the case the exact window restore is there for (with
# it off, the windows check below fails).
const _WS_E, _WS_W = 0.3, 1.0
_ws_target(t, rv) = build_target(
    planets = (b = (P = UniformPrior(12.25, 12.35), K = UniformPrior(20.0, 30.0),
                    sesinw = UniformPrior(sqrt(_WS_E) * sin(_WS_W) - 0.1,
                                          sqrt(_WS_E) * sin(_WS_W) + 0.1),
                    secosw = UniformPrior(sqrt(_WS_E) * cos(_WS_W) - 0.1,
                                          sqrt(_WS_E) * cos(_WS_W) + 0.1),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (HARPS = (data = (t = t, rv = rv, rv_err = fill(2.0, length(t))),
                   sigma = LogUniformPrior(0.5, 10.0)),))
const _WS_T, _WS_RV = let rng = MersenneTwister(42), n = 60
    t = sort(55000 .+ 200 .* rand(rng, n))
    th = Theta{Float64}(_ws_target(t, zeros(n)).params)
    for (nm, v) in ("P_k1" => 12.3, "K_k1" => 25.0,
                    "sesinw_k1" => sqrt(_WS_E) * sin(_WS_W),
                    "secosw_k1" => sqrt(_WS_E) * cos(_WS_W), "Mo_k1" => 0.02,
                    "gamma_HARPS" => 0.0, "sigma_HARPS" => 0.01)
        set_param!(th, nm, v)
    end
    model, _ = rv_predictions(th, _ws_target(t, zeros(n)).data)
    t, model .+ 2.0 .* randn(rng, n)
end

const _WS_KW = (n_temps = 4, n_walkers = 20, n_burnin = 120, warmup_swaps = 30,
                whiten_window = 20, whiten_refresh = 7, seed = 42,
                show_progress = false)

# A fresh target per run: the re-cut moves circular windows on the target itself.
# `data = f` runs on `f(tg.data)` in place of the target's own data.
_ws_run(n_steps; rv = _WS_RV, data = identity, kw...) = (tg = _ws_target(_WS_T, rv);
    sample_pt_whitening(tg, data(tg.data); _WS_KW..., n_steps, kw...))
# `d` with some fields replaced.
_ws_with(d::Data; kw...) =
    Data((get(kw, f, getfield(d, f)) for f in fieldnames(Data))...)
# The message of the ArgumentError `f()` throws, or nothing if it throws none.
_ws_refusal(f) = try f(); nothing catch e; e isa ArgumentError ? e.msg : rethrow() end
# (step, parameter, walker); `Array(chains)` would stack the walkers.
_ws_cube(r) = r.chains.value.data

# Every field of the result, bit for bit.
function _ws_same(a, b)
    @test _ws_cube(a) == _ws_cube(b)
    @test names(a.chains) == names(b.chains)
    @test isequal(a.log_evidence, b.log_evidence)
    @test isequal(a.evidence, b.evidence)
    @test a.acceptance_within == b.acceptance_within
    @test a.acceptance_swap == b.acceptance_swap
    @test a.betas == b.betas
    @test a.n_evals == b.n_evals
    @test a.whitening_active_after == b.whitening_active_after
end

@testset "pt_whitening resume" begin
    tg_full = _ws_target(_WS_T, _WS_RV)
    full = sample_pt_whitening(tg_full, tg_full.data; _WS_KW..., n_steps = 190)
    # The run does cross what the splits are placed to cross.
    @test full.whitening_active_after == 31
    @test first(Nereus.circular_windows(tg_full.params)["Mo_k1"]) != 0.0

    path = joinpath(mktempdir(), "pt_whitening_state.jls")
    _ws_run(12; checkpoint = path)                  # ring part-filled
    @test isfile(path) && !isfile(path * ".tmp")
    _ws_run(25; checkpoint = path, resume = true)   # ring wrapped, identity swaps
    _ws_run(43; checkpoint = path, resume = true)   # whitened, between refreshes
    _ws_run(61; checkpoint = path, resume = true)   # just past the first re-cut
    # End of burn-in, nothing kept. `node_flip` changed: nothing here can flip,
    # so it does not touch the chain and is not refused.
    _ws_run(120; checkpoint = path, resume = true, node_flip = 0.0)
    mid = _ws_run(160; checkpoint = path, resume = true)   # a finished run ...
    tg_ext = _ws_target(_WS_T, _WS_RV)                     # ... extended
    ext = sample_pt_whitening(tg_ext, tg_ext.data; _WS_KW..., n_steps = 190,
                              checkpoint = path, resume = true)

    # Identical to the uninterrupted run, bit for bit, and left in the same
    # chart: run_job and fit_health read the windows the engine sampled in.
    _ws_same(ext, full)
    @test Nereus.circular_windows(tg_ext.params) == Nereus.circular_windows(tg_full.params)
    # The finished 160-step run kept the first 40 post-burn-in steps of it.
    @test _ws_cube(mid) == _ws_cube(full)[1:40, :, :]

    # Resuming at the checkpoint's own step runs nothing and returns the same.
    _ws_same(_ws_run(190; checkpoint = path, resume = true), full)

    # Refusals: no checkpoint, another sampler's, a different run, a total
    # shorter than the saved.
    @test_throws ArgumentError _ws_run(190; resume = true)
    @test_throws ArgumentError _ws_run(190; checkpoint = path * ".none", resume = true)
    other = joinpath(mktempdir(), "pt_emcee_state.jls")
    Nereus.write_checkpoint(other, "pt_emcee", (;), (; step = 1))
    err = try _ws_run(190; checkpoint = other, resume = true); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("pt_emcee", err.msg)
    for (k, v) in ((:n_walkers, 24), (:whiten_window, 30), (:whiten_refresh, 5),
                   (:warmup_swaps, 40))
        err = try _ws_run(200; checkpoint = path, resume = true, k => v); nothing
              catch e; e end
        @test err isa ArgumentError && occursin(String(k), err.msg)
    end
    err = try _ws_run(200; checkpoint = path, resume = true, rv = _WS_RV .+ 1.0); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("data", err.msg)
    @test_throws ArgumentError _ws_run(150; checkpoint = path, resume = true)

    # A different ladder is named by the keyword that sets it.
    msg = _ws_refusal(() -> _ws_run(200; checkpoint = path, resume = true,
                                    betas = [1.0, 0.5, 0.25, 0.125]))
    @test msg !== nothing && occursin("betas", msg)
    # `t_ref` is a scalar, which `run_fingerprint` skips; it sets the epoch Mo
    # is measured from, so it changes the chain.
    msg = _ws_refusal(() -> _ws_run(200; checkpoint = path, resume = true,
                                    data = d -> _ws_with(d; t_ref = d.t_ref + 1.0)))
    @test msg !== nothing && occursin("data_content", msg)
end

# Base's hash of an array of 8192 or more entries samples about log(n) of them,
# so most single-entry changes in a long light curve leave it unchanged; the
# content hash reads every entry.
@testset "pt_whitening fingerprint reads every data entry" begin
    n = 10_000
    rng = MersenneTwister(5)
    t, rv = sort(55000 .+ 200 .* rand(rng, n)), randn(rng, n)
    d0 = Data(t_rv = t, rv = rv, rv_err = fill(2.0, n))
    i = findfirst(1:n) do i
        rv2 = copy(rv); rv2[i] += 1e-9
        hash(rv2) == hash(rv)
    end
    @test i !== nothing                    # Base misses this entry ...
    rv2 = copy(rv); rv2[i] += 1e-9
    d1 = Data(t_rv = t, rv = rv2, rv_err = fill(2.0, n))
    @test Nereus._ptw_data_hash(d1) != Nereus._ptw_data_hash(d0)    # ... this does not
    # Equal content in new arrays hashes the same: the hash is of content, not
    # of object identity, so it is the same in the process that resumes.
    @test Nereus._ptw_data_hash(deepcopy(d0)) == Nereus._ptw_data_hash(d0)
end

# An astrometry-only planet, so the node flip (and its per-walker RNG streams)
# is live; the split falls inside burn-in after whitening has started.
const _WS_IAD = let rng = MersenneTwister(11), n = 40
    t   = sort(57000 .+ 1800 .* rand(rng, n))
    psi = 2pi .* rand(rng, n)
    IADData(t = t, abscissa = 0.1 .* randn(rng, n), abscissa_err = fill(0.1, n),
            psi = psi, parallax_factor = sin.(2pi .* (t .- 57000) ./ 365.25 .- psi),
            pm_factor = (t .- sum(t) / n) ./ 365.25)
end
_ws_as_target(iad = _WS_IAD) = build_target(M_pri = 0.644, iad = iad,
    planets = (b = (a = LogUniformPrior(0.3, 4.0), M_sec = LogUniformPrior(0.001, 0.05),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    inc = SinePrior(), Omega = UniformPrior(0.0, 2pi),
                    Mo = UniformPrior(0.0, 2pi)),),
    plx = NormalPrior(13.6, 0.02), M_s = 0.644)

const _WS_AS_KW = (n_temps = 3, n_walkers = 16, n_burnin = 30, warmup_swaps = 10,
                   whiten_window = 10, whiten_refresh = 3, seed = 3,
                   show_progress = false)
_ws_as_run(n_steps; iad = _WS_IAD, kw...) = (tg = _ws_as_target(iad);
    sample_pt_whitening(tg, tg.data; _WS_AS_KW..., n_steps, kw...))
# The same IAD with every abscissa moved by `δ`.
_ws_iad_shift(δ) = IADData(t = _WS_IAD.t, abscissa = _WS_IAD.abscissa .+ δ,
    abscissa_err = _WS_IAD.abscissa_err, psi = _WS_IAD.psi,
    parallax_factor = _WS_IAD.parallax_factor, pm_factor = _WS_IAD.pm_factor)

@testset "pt_whitening resume carries the node-flip streams" begin
    tg = _ws_as_target()
    @test !isempty(Nereus.node_flips(tg.params, tg.data))
    full = _ws_as_run(60)
    path = joinpath(mktempdir(), "pt_whitening_state.jls")
    _ws_as_run(20; checkpoint = path)

    # Shifted abscissae change the chain, and `run_fingerprint` alone does not
    # see them: the IAD is a record, not an array. The resume is refused.
    shifted = _ws_iad_shift(0.05)
    @test _ws_cube(_ws_as_run(60; iad = shifted)) != _ws_cube(full)
    tg_a, tg_b = _ws_as_target(), _ws_as_target(shifted)
    @test isequal(Nereus.run_fingerprint(tg_a.params, tg_a.data),
                  Nereus.run_fingerprint(tg_b.params, tg_b.data))
    msg = _ws_refusal(() -> _ws_as_run(60; checkpoint = path, resume = true,
                                       iad = shifted))
    @test msg !== nothing && occursin("data_content", msg)
    # Here a planet can flip, so a different rate is refused, and turning the
    # flip off is too.
    for nf in (0.2, 0.0)
        msg = _ws_refusal(() -> _ws_as_run(60; checkpoint = path, resume = true,
                                           node_flip = nf))
        @test msg !== nothing && occursin("node_flip", msg)
    end

    # The same IAD in new arrays (as a resume in another process sees it) is
    # accepted, and the continuation is the uninterrupted run.
    _ws_same(_ws_as_run(60; checkpoint = path, resume = true,
                        iad = deepcopy(_WS_IAD)), full)
end
