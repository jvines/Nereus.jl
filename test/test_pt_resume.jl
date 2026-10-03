# sample_pt checkpoints and resume (src/checkpoint.jl).
#
# The contract: a run stopped at any iteration and continued with
# `resume = true` is bit-identical to one that ran straight through, so a run
# that has not mixed can simply be extended. The split points below cross
# everything that carries state between iterations. Fixed-dim, slice kernel,
# warmup of 4 rounds (iterations 1-30): a kill inside the last warmup round
# while the circular trace records (iteration 22), a run that ends with its
# warmup and so leaves the Mo re-cut to the resume (30), a kill inside the
# evidence rounds (45), a finished run (126) extended past its end (254).
# Trans-dim with prior births, RWM kernel: a kill while the step sizes adapt
# (4), and an early stop at the end of warmup resumed past it. Informed births
# (the default td) are not checkpointed and their resume is refused.
using Test
using Nereus
using Random

_pr_target(t, rv) = build_target(
    planets = (b = (P = LogUniformPrior(12.0, 12.6), K = LogUniformPrior(1.0, 100.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2π)),),
    rv = (HARPS = (data = (t = t, rv = rv, rv_err = fill(2.0, length(t))),
                   sigma = LogUniformPrior(0.01, 10.0)),))

# An eccentric planet (so Mo is not degenerate with ω) whose mean anomaly sits
# 0.02 rad from the seam, as in test_circular.jl: the cold chain straddles
# 0 ≡ 2π during warmup, so the re-cut moves the window.
const _PR_T, _PR_RV = let rng = MersenneTwister(42), N = 60
    t = sort(55000 .+ 200 .* rand(rng, N))
    tg = _pr_target(t, zeros(N))
    th = Theta{Float64}(tg.params)
    e, ω = 0.3, 1.0
    for (nm, v) in ("P_k1" => 12.3, "K_k1" => 25.0, "sesinw_k1" => sqrt(e) * sin(ω),
                    "secosw_k1" => sqrt(e) * cos(ω), "Mo_k1" => 0.02,
                    "gamma_HARPS" => 0.0, "sigma_HARPS" => 0.01)
        set_param!(th, nm, v)
    end
    model, _ = rv_predictions(th, tg.data)
    t, model .+ sqrt(2.0^2 + 3.0^2) .* randn(rng, N)
end
const _PR_N = length(_PR_T)

# A fresh target per run: the re-cut moves circular windows on the target itself.
_pr_target(rv = _PR_RV) = _pr_target(_PR_T, rv)

# No n_warmup_rounds: the first run (n_rounds = 8) has 4 by default, and every
# resume keeps the checkpoint's although its own n_rounds would give 2 or 3.
const _PR_KW = (n_chains = 6, seed = 1, show_report = false)

_pr_run(n_rounds; kw...) = sample_pt(_pr_target(); _PR_KW..., n_rounds, kw...)
# The killed run: sample_pt's inner loop with its test hook, which writes the
# checkpoint after iteration `at` and throws as a kill would. Fixed-dim is the
# null trans-dim config, exactly as sample_pt builds it.
function _pr_kill(n_rounds, at; kw...)
    tg = _pr_target()
    td = TransDimConfig(; max_kplanet = tg.params.config.max_kplanet,
                          planets = false, noise = false, transdim_fraction = 0.0)
    Nereus._sample_pt_transdim(tg, td; _PR_KW..., n_rounds, halt_after = at, kw...)
end
_raw(ch) = ch.value.data

# Trans-dim: two RV-only slots over the same data, RWM within-model kernel.
# Prior births only: informed births keep process-global periodogram caches a
# checkpoint cannot hold, and are refused (last testset).
function _td_target()
    data = Data(; t_rv = _PR_T, rv = _PR_RV, rv_err = fill(2.0, _PR_N))
    params = Params(; max_kplanet = 2, planet_modes = [RV_ONLY, RV_ONLY],
                      instruments = InstrumentConfig(rv = ["HARPS"]), data = data,
                      M_s = 1.0)
    return NereusTarget(params, data; unconstrained = false)
end
const _TD = TransDimConfig(max_kplanet = 2, birth_strategies = [PriorBirth()],
                           birth_weights = [1.0])
const _TD_KW = (n_chains = 4, seed = 7, within_model = :rwm, n_warmup_rounds = 3,
                show_report = false)
_td_run(n_rounds; kw...) = sample_pt(_td_target(); td = _TD, _TD_KW..., n_rounds, kw...)

@testset "sample_pt resume" begin
    @testset "fixed-dim: kills, the warmup re-cut, an extension" begin
        tg_full = _pr_target()
        full = sample_pt(tg_full; _PR_KW..., n_rounds = 7, n_warmup_rounds = 4)
        lo, hi = Nereus.circular_windows(tg_full.params)["Mo_k1"]
        @test lo != 0.0                      # the re-cut moved the seam: it is tested

        path = joinpath(mktempdir(), "pt_state.jls")
        @test_throws InterruptException _pr_kill(8, 22; checkpoint = path)
        @test isfile(path) && !isfile(path * ".tmp")

        # Ends with its warmup: no re-cut, so its Mo draws are still in the old
        # chart; everything else is the uninterrupted run's first 30 rows.
        w4, _ = _pr_run(4; checkpoint = path, resume = true)
        mo = findfirst(==(:Mo_k1), names(w4))
        others = setdiff(1:size(_raw(w4), 2), mo)
        @test size(w4, 1) == 30
        @test _raw(w4)[:, others, :] == _raw(full[1])[1:30, others, :]
        @test Nereus.circular_relabel.(_raw(w4)[:, mo, 1], lo, hi) ==
              _raw(full[1])[1:30, mo, 1]

        # Re-cut on resume, killed again inside the evidence rounds.
        @test_throws InterruptException _pr_kill(7, 45; checkpoint = path, resume = true)
        mid = _pr_run(6; checkpoint = path, resume = true)     # a finished run ...
        ext = _pr_run(7; checkpoint = path, resume = true)     # ... extended

        # Identical to the uninterrupted run, bit for bit.
        @test _raw(ext[1]) == _raw(full[1])
        @test isequal(ext[2], full[2])
        @test _raw(mid[1]) == _raw(full[1])[1:126, :, :]

        # Resuming at the checkpoint's own iteration runs nothing and returns the same.
        same = _pr_run(7; checkpoint = path, resume = true)
        @test _raw(same[1]) == _raw(full[1])
        @test isequal(same[2], full[2])

        # Refusals: no checkpoint, a different run, a total shorter than the saved.
        @test_throws ArgumentError _pr_run(7; resume = true)
        @test_throws ArgumentError _pr_run(7; checkpoint = path * ".none", resume = true)
        n_uf = length(tg_full.params.layout.unfrozen_idx)
        for (kw, field) in (((n_chains = 8,), "n_chains"),
                            ((n_warmup_rounds = 3,), "warmup_rounds"),
                            ((within_model = :rwm,), "within_model"),
                            ((init = zeros(n_uf, 6),), "init"))
            err = try _pr_run(8; checkpoint = path, resume = true, kw...); nothing
                  catch e; e end
            @test err isa ArgumentError && occursin(field, err.msg)
        end
        err = try sample_pt(_pr_target(_PR_RV .+ 1.0); _PR_KW..., n_rounds = 8,
                            checkpoint = path, resume = true); nothing
              catch e; e end
        @test err isa ArgumentError && occursin("data", err.msg)
        @test_throws ArgumentError _pr_run(6; checkpoint = path, resume = true)
    end

    @testset "trans-dim, RWM: a kill while adapting, an early stop" begin
        full = _td_run(6)

        path = joinpath(mktempdir(), "pt_state.jls")
        @test_throws InterruptException Nereus._sample_pt_transdim(_td_target(), _TD;
            _TD_KW..., n_rounds = 6, checkpoint = path, halt_after = 4)
        res = _td_run(6; checkpoint = path, resume = true)
        @test _raw(res[1]) == _raw(full[1])
        @test isequal(res[2], full[2])
        @test res[3] == full[3]

        # Stops at round 3, the end of warmup (Δmax < 1 always), checkpointing
        # first; the resume, with the check off, runs the rest.
        path = joinpath(mktempdir(), "pt_state.jls")
        es = _td_run(6; checkpoint = path, early_stop_thresh = 1.0,
                     early_stop_min_rounds = 2)
        @test size(es[1], 1) == 14
        res = _td_run(6; checkpoint = path, resume = true)
        @test _raw(res[1]) == _raw(full[1])
        @test isequal(res[2], full[2])
        @test res[3] == full[3]
    end

    @testset "informed births: not checkpointed, resume refused" begin
        td = TransDimConfig(max_kplanet = 2)          # PriorBirth + InformedBirth
        path = joinpath(mktempdir(), "pt_state.jls")
        @test_logs (:warn, r"not checkpointing") match_mode = :any sample_pt(
            _td_target(); td, _TD_KW..., n_rounds = 2, checkpoint = path)
        @test !isfile(path)
        err = try sample_pt(_td_target(); td, _TD_KW..., n_rounds = 2,
                            checkpoint = path, resume = true); nothing
              catch e; e end
        @test err isa ArgumentError && occursin("InformedBirth", err.msg)
    end
end
