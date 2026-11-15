# The post-hoc burn-in trim (`trim_burnin`, src/samplers/burnin_trim.jl).
#
# Walkers that reach the main mode only during the recorded run put their climb
# into the posterior. The settling test runs on every run and warns; with
# `trim_burnin = true` the draws up to the last step a walker was below are
# dropped, and the run is reported as if recording had begun there.
#
# The rule is tested on synthetic log densities first. The samplers then get a
# settling run set up as in test_burnin_extend.jl: one planet fitted to two
# sinusoids, P2 a local mode ~1900 below P1 in ln p, a checkpoint after the
# first step rewritten with every walker in P2 but β = 1 walker 1, in P1. Six
# rungs here, the DE move on and the pruning off, so the hotter rungs feed P1
# states down and the β = 1 walkers cross over during the recorded run (all by
# recorded step ~230 of 1496). pt_whitening and sample_ensemble start every
# walker in P1 with K at half its value instead, a climb rather than a crossing,
# which the test sees the same way.
using Test, Nereus, Random, Serialization, Statistics, Distributions, MCMCChains, JSON3
using Logging
using Nereus: TransDimConfig, reset_transdim_caches!, _trim_detect, _trim_rows,
              _trim_report, _prune_allow

Random.seed!(20261115)
const _BT_N = 60
const _BT_T = sort!(400 .* rand(_BT_N))
const _BT_RV = 40.0 .* sin.(2π .* _BT_T ./ 4.23) .+
               10.0 .* sin.(2π .* _BT_T ./ 4.40 .+ 1.0) .+ 1.5 .* randn(_BT_N)

_bt_target(Plo = 4.0, Phi = 4.6) = build_target(
    planets = (b = (P = LogUniformPrior(Plo, Phi), K = LogUniformPrior(1.0, 90.0),
                    sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
                    Mo = UniformPrior(0.0, 2pi)),),
    rv = (SIM = (data = (t = _BT_T, rv = _BT_RV, rv_err = fill(1.5, _BT_N)),
                 sigma = LogUniformPrior(0.5, 3.0)),))

# Synthetic levels: a d-dimensional posterior's draws sit χ²_d / 2 below its peak.
_bt_levels(rng, n, nw, d) = -rand(rng, Chisq(d), n, nw) ./ 2

# A report from a detection, as a sampler makes it.
_bt_report(rec; trim = false, cuts = nothing, kw...) =
    _trim_report("pt_emcee", rec; trim, rows = _trim_rows(rec; cuts),
                 n_trimmed = trim ? max(_trim_rows(rec; cuts), 0) : 0,
                 n_burnin_used = 1000, extend = false, kw...)

@testset "the settling test" begin
    rng = MersenneTwister(1)
    d, n, nw = 10, 400, 20

    @testset "a clean run: nothing flagged, nothing said" begin
        rec = _trim_detect(_bt_levels(rng, n, nw, d), [d])
        @test rec.verdict === :clean && rec.cut == 0 && _trim_rows(rec) == 0
        @test rec.halfwidth == 10 && rec.final == 40
        t = @test_logs _bt_report(rec)
        @test t.status == "ok" && !t.applied && t.verdict == "clean"
    end

    @testset "walkers settling: the cut is the last step one was below" begin
        L = _bt_levels(rng, n, nw, d)
        for w in 1:5
            L[1:(20w), w] .-= 1000             # walker w crosses after step 20w
        end
        rec = _trim_detect(L, [d])
        @test rec.verdict === :settling
        @test rec.cut == 100
        @test rec.n_settling == 5
        @test 990 < rec.delta < 1015
        @test _trim_rows(rec) == 100
        @test _trim_rows(rec; cuts = 0:30:n) == 120          # up to the next block
        @test _trim_rows(rec; cuts = [0, 350]) == -1         # under a quarter left
        t = @test_logs (:warn, r"pt_emcee: 5 of 20 walkers at β = 1 were still settling into the main mode until recorded step 100 of 400 .*Set trim_burnin = true to drop them, or rerun with n_burnin ≥ 1100 or extend_burnin = true\.") _bt_report(rec)
        @test t.status == "warn" && !t.applied && t.n_trimmed == 0
        @test t.cut_step == 100 && t.n_walkers_settling == 5 && t.n_recorded == 400
        # Applied: an info line with show_progress, the record says so.
        t = @test_logs (:info, r"trim_burnin dropped the first 120 of 400") _bt_report(
            rec; trim = true, cuts = 0:30:n)
        @test t.status == "ok" && t.applied && t.n_trimmed == 120
        t = @test_logs _bt_report(rec; trim = true, show_progress = false)
        @test t.applied && t.n_trimmed == 100
        # A trim that would leave under a quarter is refused, and said so.
        t = @test_logs (:warn, r"fewer than a quarter.*trim_burnin = true has dropped nothing\. Rerun with n_burnin ≥ 1100 \(and n_steps raised by as much\), or with extend_burnin = true\.") _bt_report(
            rec; trim = true, cuts = [0, 350])
        @test t.status == "warn" && !t.applied && t.verdict == "short_tail"
    end

    @testset "the cut falls in the last quarter: refused" begin
        L = _bt_levels(rng, n, nw, d)
        L[1:350, 3] .-= 1000
        rec = _trim_detect(L, [d])
        @test rec.verdict === :short_tail && rec.cut == 350 && _trim_rows(rec) == 0
        t = @test_logs (:warn, r"fewer than a quarter") _bt_report(rec; trim = true)
        @test !t.applied && t.n_trimmed == 0
    end

    @testset "modes of similar height: no cut, no warning" begin
        d34 = 34                               # the allowance is 44.2
        @test _prune_allow(d34) ≈ 44.19 atol = 0.01
        for off in (8.0, 30.0)
            L = _bt_levels(rng, n, nw, d34)
            L[:, 1:(nw ÷ 2)] .-= off           # half the walkers in the lower mode
            rec = _trim_detect(L, [d34])
            @test rec.verdict === :clean && rec.cut == 0
            t = @test_logs _bt_report(rec; trim = true)
            @test !t.applied && t.status == "ok"
        end
    end

    @testset "excursions after reaching the main mode are posterior mass" begin
        # A stationary posterior whose walkers wander into a heavy tail, past the
        # allowance, for far longer than the window: mid-run, late, and one still
        # out at the end. Nothing settled, so no cut and no warning.
        n2 = 2000                              # half-width 25
        L = _bt_levels(rng, n2, nw, d)
        @test _prune_allow(d) < 30
        L[500:524, 2] .-= 1000                 # 25 draws, under half the window
        L[1:7:n2, 4] .-= 1000                  # every 7th draw
        L[800:1000, 5] .-= 60                  # 201 draws mid-run
        L[1700:1850, 6] .-= 60                 # 151 draws late
        L[1900:n2, 7] .-= 60                   # out at the end: 101 of the final 200
        L[1200:n2, 8] .-= 1000                 # out for the last 40% of the run
        rec = _trim_detect(L, [d])
        @test rec.halfwidth == 25
        @test rec.verdict === :clean && rec.cut == 0 && rec.n_settling == 0
        t = @test_logs _bt_report(rec; trim = true)
        @test !t.applied && t.status == "ok"
        # The same stretch at the start of recording, before the walker first
        # reached the main mode, is settling: cut at its last draw.
        L[1:61, 3] .-= 1000
        rec = _trim_detect(L, [d])
        @test rec.verdict === :settling && rec.cut == 61 && rec.n_settling == 1
        @test 990 < rec.delta < 1015
    end

    @testset "an excursion just after recording begins is not settling" begin
        # The first windows are cut short and reach forward only: a walker in the
        # band for its first 4 draws, then below for 15, has its first 4 windows
        # mostly below. Its draws before its first entry sit in the band, so it
        # was not climbing.
        L = _bt_levels(rng, 2000, nw, d)
        L[5:19, 9] .-= 60
        rec = _trim_detect(L, [d])
        @test rec.verdict === :clean && rec.cut == 0
        # Below from the first draw, the same walker is settling.
        L[1:4, 9] .-= 60
        rec = _trim_detect(L, [d])
        @test rec.verdict === :settling && rec.cut > 0 && rec.n_settling == 1
    end

    @testset "walkers never in the main mode: split, refused" begin
        L = _bt_levels(rng, n, nw, d)
        L[:, 1:3] .-= 1000
        rec = _trim_detect(L, [d])
        @test rec.verdict === :split && rec.stuck == [1, 2, 3] && rec.n_settling == 3
        @test _trim_rows(rec) == 0
        t = @test_logs (:warn, r"3 of 20 walkers at β = 1 are still out of the main mode at the end of the run .*were never in it.*no trim can rescue it; trim_burnin = true has dropped nothing\. Rerun with a longer n_burnin or extend_burnin = true, or narrow the prior of the parameter that separates the modes\.") _bt_report(
            rec; trim = true)
        @test !t.applied && t.status == "warn" && t.verdict == "split"
        # With the separating parameter named.
        t = @test_logs (:warn, r"in P_k1 \(.*\)\. Narrow the prior of P_k1 to leave the other mode out, or rerun with a longer n_burnin or extend_burnin = true\.$") _bt_report(
            rec; split = (" The two groups sit furthest apart, against their own " *
                          "spread, in P_k1 (12 times it).", "P_k1"))
    end

    @testset "too short, or a non-number: not assessed" begin
        @test _trim_detect(_bt_levels(rng, 19, nw, d), [d]).verdict === :not_assessed
        L = _bt_levels(rng, n, nw, d)
        L[5, 5] = NaN
        rec = _trim_detect(L, [d])
        @test rec.verdict === :not_assessed
        t = @test_logs _bt_report(rec; trim = true)
        @test !t.applied && t.status == "ok"
    end

    @testset "trans-dim: models are not compared with each other" begin
        # Two models, the smaller one 100 lower in ln p, walkers hopping between
        # them every 37 steps.
        n3 = 600
        model = [1 + ((i ÷ 37 + w) % 2) for i in 1:n3, w in 1:nw]
        L = [model[i, w] == 1 ? -100 - rand(rng, Chisq(5)) / 2 : -rand(rng, Chisq(8)) / 2
             for i in 1:n3, w in 1:nw]
        @test _trim_detect(L, [5, 8]; model).verdict === :clean
        @test _trim_detect(L, [8]).verdict !== :clean    # as one model it would flag
        # A third model seen only early and far below: no reference, no flag.
        M3 = copy(model); L3 = copy(L)
        M3[1:100, 1:4] .= 3
        L3[1:100, 1:4] .= -5000
        @test _trim_detect(L3, [5, 8, 3]; model = M3).verdict === :clean
        # Settling within one model is still found, at that model's last draw below.
        L4 = copy(L)
        for i in 1:150
            model[i, 1] == 2 && (L4[i, 1] -= 1000)
        end
        rec = _trim_detect(L4, [5, 8]; model)
        @test rec.verdict === :settling
        @test rec.cut == maximum(i for i in 1:150 if model[i, 1] == 2)
        @test rec.n_settling == 1
    end
end

@testset "evidence in blocks" begin
    rng = MersenneTwister(2)
    βs = [1.0, 0.5, 0.25, 0.0]
    nT, b_end, thin, n_steps = 4, 7, 3, 400
    eb = Nereus.EvidenceBlocks(thin)
    @test eb.len == 51                         # 50 rounded up to a multiple of thin
    acc, tail = Nereus.EvidenceAccumulator(nT), Nereus.EvidenceAccumulator(nT)
    for step in (b_end + 1):(b_end + n_steps)
        blk = Nereus._evidence_block!(eb, step, b_end, nT)
        for _ in 1:5
            ll = -100 .* (1 .- βs) .- 10 .* rand(rng, nT)
            step == b_end + 60 && (ll[4] = -Inf)          # a dropped draw
            Nereus.update_evidence!(acc, ll, βs)
            Nereus.update_evidence!(blk, ll, βs)
            step >= b_end + 1 + 2 * 51 && Nereus.update_evidence!(tail, ll, βs)
        end
    end
    @test eb.start == collect((b_end + 1):51:(b_end + n_steps))
    @test Nereus._evidence_cuts(eb, b_end, thin) == collect(0:17:(17 * 7))
    same(a, b) = a.n == b.n && a.n_nonfinite == b.n_nonfinite && a.started == b.started &&
                 isapprox(a.sum_logL, b.sum_logL; rtol = 1e-12) &&
                 isapprox(a.sum_logL2, b.sum_logL2; rtol = 1e-12) &&
                 isapprox(a.ss_lse_fwd, b.ss_lse_fwd; rtol = 1e-12) &&
                 isapprox(a.ss_lse_bwd, b.ss_lse_bwd; rtol = 1e-12)
    # All the blocks give the run's own sums; from the third on, a fresh
    # accumulator fed from there.
    @test same(Nereus._evidence_from(eb, b_end + 1, nT), acc)
    @test same(Nereus._evidence_from(eb, eb.start[3], nT), tail)
    @test same(Nereus._evidence_from(eb, eb.start[3] - 10, nT), tail)
    r1 = Nereus.evidence_report(Nereus._evidence_from(eb, b_end + 1, nT), βs)
    r2 = Nereus.evidence_report(acc, βs)
    @test r1.ti[1] ≈ r2.ti[1] && r1.ss_plus[1] ≈ r2.ss_plus[1] && r1.hybrid[1] ≈ r2.hybrid[1]
    # Restored from a checkpoint with blocks, or from one written before them.
    e2 = Nereus._evidence_blocks_restore!(Nereus.EvidenceBlocks(thin),
                                          (; evidence_blocks = eb), acc, b_end)
    @test e2.start == eb.start && e2.acc !== eb.acc
    e3 = Nereus._evidence_blocks_restore!(Nereus.EvidenceBlocks(thin), (; step = 1), acc,
                                          b_end)
    @test e3.start == [b_end + 1] && same(e3.acc[1], acc)
    e4 = Nereus._evidence_blocks_restore!(Nereus.EvidenceBlocks(thin), (; step = 1),
                                          Nereus.EvidenceAccumulator(nT), b_end)
    @test isempty(e4.start)
end

# ------------------------------------------------------------------
# The samplers.
# ------------------------------------------------------------------

# Posterior draws (x, rows) of the one-planet model with P confined to one mode.
function _bt_mode_draws(Plo, Phi)
    tg = _bt_target(Plo, Phi)
    r = sample_pt_emcee(tg, tg.data; n_temps = 4, n_walkers = 32, n_steps = 400,
                        n_burnin = 300, seed = 3, show_progress = false,
                        bridge_headline = false)
    nms = Symbol.(tg.params.layout.unfrozen_names)
    X = hcat([vec(Array(r.chains[p])) for p in nms]...)
    iM = findfirst(==(:Mo_k1), nms)
    X[:, iM] .= mod2pi.(X[:, iM])          # into the window of a fresh target
    return X
end

const _BT_X = (_bt_mode_draws(4.15, 4.30), _bt_mode_draws(4.35, 4.45))
const _BT_IK = 2                               # K_k1

# Where an injection puts walker w of rung t: `:cross`, P2 draws for all but
# β = 1 walker 1 (a P1 draw); `:climb`, P1 draws with K halved.
function _bt_position(kind, t, w, rng)
    X1, X2 = _BT_X
    kind === :cross && return (t, w) == (1, 1) ? X1[end, :] : X2[rand(rng, 1:size(X2, 1)), :]
    x = X1[rand(rng, 1:size(X1, 1)), :]
    x[_BT_IK] /= 2
    return x
end

# (step, parameter, walker); `Array(chains)` would stack the walkers.
_bt_cube(r) = r.chains.value.data

const _BT_PT = (n_temps = 6, n_walkers = 32, seed = 7, de_move = 0.5,
                show_progress = false, bridge_headline = false, prune_stranded = false)

# pt_emcee's checkpoint at step 1, rewritten: positions on the flat scale the
# walkers move on, their log π with its Jacobian, and log L.
function _bt_pt_checkpoint(n_burnin)
    path = joinpath(mktempdir(), "split.jls")
    tg = _bt_target()
    sample_pt_emcee(tg, tg.data; _BT_PT..., n_burnin, n_steps = 1, checkpoint = path)
    ck = open(deserialize, path)
    st = ck.state
    S, LP, LL = copy(st.state), copy(st.logπ), copy(st.logL)
    shift = [Nereus.log_scale_shift(p) for p in tg.params.layout.unfrozen_priors]
    bounded = Nereus.NereusTarget(tg.params, tg.data; unconstrained = false)
    rng = MersenneTwister(5)
    for t in axes(S, 1), w in axes(S, 2)
        x = _bt_position(:cross, t, w, rng)
        lp, ll = Nereus._logdensity_parts(bounded, x)
        for j in eachindex(x)
            s = shift[j]
            S[t, w, j] = s === nothing ? x[j] : log(x[j] + s)
            s === nothing || (lp += S[t, w, j])
        end
        LP[t, w], LL[t, w] = lp, ll
    end
    open(io -> serialize(io, merge(ck, (state = merge(st, (state = S, logπ = LP,
                                                           logL = LL)),))), path, "w")
    return path
end

function _bt_pt_resume(src, n_steps; n_burnin = 4, kw...)
    path = joinpath(mktempdir(), "state.jls")
    cp(src, path)
    tg = _bt_target()
    r = sample_pt_emcee(tg, tg.data; _BT_PT..., n_burnin, n_steps, checkpoint = path,
                        resume = true, kw...)
    return r, path
end

@testset "trim_burnin (pt_emcee)" begin
    src = _bt_pt_checkpoint(4)
    off, poff = @test_logs (:warn, r"pt_emcee: 31 of 32 walkers at β = 1 were still settling into the main mode until recorded step \d+ of 1496 .*Set trim_burnin = true to drop them, or rerun with n_burnin ≥ \d+ or extend_burnin = true\.") match_mode = :any begin
        _bt_pt_resume(src, 1500)
    end
    on, pon = _bt_pt_resume(src, 1500; trim_burnin = true)
    to, tn = off.burnin_trim, on.burnin_trim

    @testset "off: detected, warned, nothing dropped" begin
        @test to.verdict == "settling" && !to.applied && to.status == "warn"
        @test to.n_walkers_settling == 31 && to.n_walkers == 32
        @test 100 < to.cut_step < 1496 ÷ 2
        @test to.delta_lnp > 100
        @test off.n_trimmed == 0 && off.n_burnin_used == 4
        @test size(_bt_cube(off), 1) == 1496
    end

    @testset "on: the draws from the next 50-step block on" begin
        @test tn.applied && tn.status == "ok" && tn.verdict == "settling"
        @test tn.cut_step == to.cut_step
        @test on.n_trimmed == 50 * cld(to.cut_step, 50) == tn.n_trimmed
        @test on.n_burnin_used == 4 + on.n_trimmed
        @test _bt_cube(on) == _bt_cube(off)[(on.n_trimmed + 1):end, :, :]
        @test on.n_evals == off.n_evals
        @test on.ladder.n_burnin == on.n_burnin_used
        @test on.stop.n_draws == size(_bt_cube(on), 1) * 32
        @test !isequal(on.ladder.mean_logL, off.ladder.mean_logL)
        @test !isequal(on.evidence.ti, off.evidence.ti)
    end

    @testset "on is the run that began recording at the cut" begin
        # No pruning, no adaptation: burn-in and recording draw alike, so the
        # same run with n_burnin at the cut draws the same states, and its
        # evidence is that of the same steps.
        ref, _ = _bt_pt_resume(_bt_pt_checkpoint(on.n_burnin_used), 1500;
                               n_burnin = on.n_burnin_used)
        @test ref.burnin_trim.verdict == "clean"
        @test _bt_cube(ref) == _bt_cube(on)
        @test ref.n_evals == on.n_evals
        @test ref.ladder.mean_logL ≈ on.ladder.mean_logL rtol = 1e-12
        @test ref.evidence.ti[1] ≈ on.evidence.ti[1] rtol = 1e-12
        @test ref.evidence.ss_plus[1] ≈ on.evidence.ss_plus[1] rtol = 1e-12
        @test ref.evidence.hybrid[1] ≈ on.evidence.hybrid[1] rtol = 1e-12
        @test isequal(ref.log_evidence, on.log_evidence) ||
              ref.log_evidence ≈ on.log_evidence
    end

    @testset "resume: the checkpoint carries no trim" begin
        # Through a checkpoint mid-run, the same as straight through.
        path = joinpath(mktempdir(), "state.jls")
        cp(src, path)
        tg = _bt_target()
        sample_pt_emcee(tg, tg.data; _BT_PT..., n_burnin = 4, n_steps = 800,
                        checkpoint = path, resume = true, trim_burnin = true)
        tg = _bt_target()
        rr = sample_pt_emcee(tg, tg.data; _BT_PT..., n_burnin = 4, n_steps = 1500,
                             checkpoint = path, resume = true, trim_burnin = true)
        @test _bt_cube(rr) == _bt_cube(on)
        @test rr.burnin_trim == on.burnin_trim
        @test isequal(rr.log_evidence, on.log_evidence)
        @test isequal(rr.ladder.mean_logL, on.ladder.mean_logL)
        # A finished run resumed at its own n_steps: trims (on) or does not (off)
        # whichever way the first run went.
        tg = _bt_target()
        rz = sample_pt_emcee(tg, tg.data; _BT_PT..., n_burnin = 4, n_steps = 1500,
                             checkpoint = poff, resume = true, trim_burnin = true)
        @test _bt_cube(rz) == _bt_cube(on) && rz.burnin_trim == on.burnin_trim
        @test isequal(rz.log_evidence, on.log_evidence) && rz.n_evals == on.n_evals
        tg = _bt_target()
        rb = Logging.with_logger(Logging.NullLogger()) do
            sample_pt_emcee(tg, tg.data; _BT_PT..., n_burnin = 4, n_steps = 1500,
                            checkpoint = pon, resume = true)
        end
        @test _bt_cube(rb) == _bt_cube(off) && rb.burnin_trim == off.burnin_trim
        @test isequal(rb.log_evidence, off.log_evidence)
    end

    @testset "the summary and fit_health" begin
        s = Nereus._burnin_to_summary!(Dict{String,Any}(), on)
        @test s["n_burnin_used"] == on.n_burnin_used
        b = s["burnin_trim"]
        @test b["applied"] === true && b["n_trimmed"] == on.n_trimmed
        @test b["cut_step"] == tn.cut_step && b["n_recorded"] == 1496
        @test b["n_walkers_settling"] == 31 && b["verdict"] == "settling"
        @test Nereus._burnin_trim_check(off.burnin_trim).status === :warn
        @test Nereus._burnin_trim_check(on.burnin_trim).status === :ok
        @test Nereus._burnin_trim_check(nothing) === nothing
    end

    @testset "a split run: refused, the parameter named" begin
        # Two rungs, every walker but one in P2, stretch moves only: the split
        # holds to the end (test_burnin_extend.jl).
        tg = _bt_target()
        kw = (n_temps = 2, betas = [1.0, 0.5], n_walkers = 32, n_burnin = 100, seed = 7,
              de_move = 0, show_progress = false, bridge_headline = false,
              prune_stranded = false)
        path = joinpath(mktempdir(), "split.jls")
        sample_pt_emcee(tg, tg.data; kw..., n_steps = 1, checkpoint = path)
        ck = open(deserialize, path)
        st = ck.state
        S, LP, LL = copy(st.state), copy(st.logπ), copy(st.logL)
        shift = [Nereus.log_scale_shift(p) for p in tg.params.layout.unfrozen_priors]
        bounded = Nereus.NereusTarget(tg.params, tg.data; unconstrained = false)
        rng = MersenneTwister(5)
        for t in axes(S, 1), w in axes(S, 2)
            x = _bt_position(:cross, t, w, rng)
            lp, ll = Nereus._logdensity_parts(bounded, x)
            for j in eachindex(x)
                s = shift[j]
                S[t, w, j] = s === nothing ? x[j] : log(x[j] + s)
                s === nothing || (lp += S[t, w, j])
            end
            LP[t, w], LL[t, w] = lp, ll
        end
        open(io -> serialize(io, merge(ck, (state = merge(st, (state = S, logπ = LP,
                                                               logL = LL)),))), path, "w")
        tg = _bt_target()
        r = @test_logs (:warn, r"pt_emcee: 31 of 32 walkers at β = 1 are still out of the main mode.*no trim can rescue it; trim_burnin = true has dropped nothing\. The two groups sit furthest apart.* in P_k1 .*Narrow the prior of P_k1") match_mode = :any begin
            sample_pt_emcee(tg, tg.data; kw..., n_steps = 400, checkpoint = path,
                            resume = true, trim_burnin = true)
        end
        @test r.burnin_trim.verdict == "split" && !r.burnin_trim.applied
        @test r.n_trimmed == 0 && r.n_burnin_used == 100
        @test size(_bt_cube(r), 1) == 300
    end

    @testset "trim off: the run is the run" begin
        # A run with nothing to trim draws the same with the option on as off.
        tg = _bt_target(4.15, 4.30)
        kw = (n_temps = 3, n_walkers = 16, n_burnin = 600, n_steps = 800, seed = 11,
              show_progress = false, bridge_headline = false)
        a = sample_pt_emcee(tg, tg.data; kw...)
        tg = _bt_target(4.15, 4.30)
        b = sample_pt_emcee(tg, tg.data; kw..., trim_burnin = true)
        @test a.burnin_trim.verdict == "clean" == b.burnin_trim.verdict
        @test _bt_cube(a) == _bt_cube(b) && a.n_evals == b.n_evals
        @test isequal(a.log_evidence, b.log_evidence) && b.n_trimmed == 0
        @test b.n_burnin_used == 600
    end
end

@testset "trim_burnin (pt_whitening)" begin
    kw = (n_temps = 4, n_walkers = 32, n_burnin = 4, seed = 7, show_progress = false)
    src = joinpath(mktempdir(), "split.jls")
    tg = _bt_target()
    sample_pt_whitening(tg, tg.data; kw..., n_steps = 1, checkpoint = src)
    ck = open(deserialize, src)
    st = ck.state
    S, LP, LL = copy(st.state), copy(st.logπ), copy(st.logL)
    bounded = Nereus.NereusTarget(tg.params, tg.data; unconstrained = false)
    rng = MersenneTwister(5)
    for t in axes(S, 1), w in axes(S, 2)
        x = _bt_position(:climb, t, w, rng)
        lp, ll = Nereus._logdensity_parts(bounded, x)
        S[t, w, :] .= x
        LP[t, w], LL[t, w] = lp, ll
    end
    open(io -> serialize(io, merge(ck, (state = merge(st, (state = S, logπ = LP,
                                                           logL = LL)),))), src, "w")
    go(; o...) = (p = joinpath(mktempdir(), "s.jls"); cp(src, p); tg = _bt_target();
                  sample_pt_whitening(tg, tg.data; kw..., n_steps = 1000, checkpoint = p,
                                      resume = true, o...))
    off = @test_logs (:warn, r"pt_whitening: 32 of 32 walkers at β = 1 were still settling.*rerun with n_burnin ≥ \d+\.$") match_mode = :any go()
    on = go(trim_burnin = true)
    @test off.burnin_trim.verdict == "settling" && !off.burnin_trim.applied
    @test off.n_burnin_used == 4 && off.n_trimmed == 0
    @test on.burnin_trim.applied
    @test on.n_trimmed == 50 * cld(off.burnin_trim.cut_step, 50)
    @test on.n_burnin_used == 4 + on.n_trimmed
    @test _bt_cube(on) == _bt_cube(off)[(on.n_trimmed + 1):end, :, :]
    @test !isequal(on.evidence.ti, off.evidence.ti)
    s = Nereus._burnin_to_summary!(Dict{String,Any}(), on)
    @test s["burnin_trim"]["applied"] && s["n_burnin_used"] == on.n_burnin_used
end

@testset "trim_burnin (transdim_pt_emcee)" begin
    kw = (td = TransDimConfig(max_kplanet = 1), n_temps = 6, n_walkers = 32, n_burnin = 4,
          seed = 7, de_move = 0.5, show_progress = false, prune_stranded = false)
    # Every walker in the one-planet model, positions in x, and log π that of a
    # one-planet walker of the run but for the change of the prior between their
    # positions (the model's own share is the same constant).
    src = joinpath(mktempdir(), "split.jls")
    reset_transdim_caches!()
    tg = _bt_target()
    sample_transdim_pt_emcee(tg, tg.data; kw..., n_steps = 1, checkpoint = src)
    ck = open(deserialize, src)
    st = ck.state
    S, LP, LL, TDS = copy(st.state), copy(st.logπ), copy(st.logL), deepcopy(st.td_states)
    bounded = Nereus.NereusTarget(tg.params, tg.data; unconstrained = false)
    ref = findfirst(tds -> tds.planet_active[1], TDS)
    @assert ref !== nothing
    c = LP[ref] - first(Nereus._logdensity_parts(bounded, S[ref[1], ref[2], :]))
    rng = MersenneTwister(5)
    for t in axes(S, 1), w in axes(S, 2)
        x = _bt_position(:cross, t, w, rng)
        lp, ll = Nereus._logdensity_parts(bounded, x)
        S[t, w, :] .= x
        LP[t, w], LL[t, w] = lp + c, ll
        TDS[t, w].n_planets_active = 1
        TDS[t, w].planet_active[1] = true
    end
    open(io -> serialize(io, merge(ck, (state = merge(st, (state = S, logπ = LP, logL = LL,
                                                           td_states = TDS)),))), src, "w")
    go(; o...) = (p = joinpath(mktempdir(), "s.jls"); cp(src, p); reset_transdim_caches!();
                  tg = _bt_target();
                  sample_transdim_pt_emcee(tg, tg.data; kw..., n_steps = 1200,
                                           checkpoint = p, resume = true, o...))
    off = @test_logs (:warn, r"td-pt_emcee: 31 of 32 walkers at β = 1 were still settling") match_mode = :any go()
    on = go(trim_burnin = true)
    @test off.burnin_trim.verdict == "settling" && off.n_trimmed == 0
    @test on.burnin_trim.applied
    @test on.n_trimmed == 50 * cld(off.burnin_trim.cut_step, 50)
    @test on.n_burnin_used == 4 + on.n_trimmed
    # Rows: steps × walkers.
    @test _bt_cube(on) == _bt_cube(off)[(on.n_trimmed * 32 + 1):end, :, :]
    @test on.ladder.n_burnin == on.n_burnin_used
    @test !isequal(on.ladder.mean_logL, off.ladder.mean_logL)
    # Resumed mid-run, the same.
    p = joinpath(mktempdir(), "s.jls")
    cp(src, p)
    reset_transdim_caches!()
    tg = _bt_target()
    sample_transdim_pt_emcee(tg, tg.data; kw..., n_steps = 600, checkpoint = p,
                             resume = true, trim_burnin = true)
    reset_transdim_caches!()
    tg = _bt_target()
    rr = sample_transdim_pt_emcee(tg, tg.data; kw..., n_steps = 1200, checkpoint = p,
                                  resume = true, trim_burnin = true)
    @test _bt_cube(rr) == _bt_cube(on) && rr.burnin_trim == on.burnin_trim
    @test isequal(rr.ladder.mean_logL, on.ladder.mean_logL)
end

const _BT_ENS = (n_walkers = 32, n_burnin = 3, seed = 7, show_progress = false,
                 prune_stranded = false)

# The ensemble's checkpoint (one rung): positions in its sampling coordinates and
# their log-densities as `logpost` gives them.
function _bt_ens_inject!(path, tg)
    ck = open(deserialize, path)
    st = ck.state
    x, ll = copy(st.x), copy(st.lastll)
    rng = MersenneTwister(5)
    for w in axes(x, 2)
        xb = _bt_position(:climb, 1, w, rng)
        y = tg.transform === nothing ? xb : Nereus.transform_forward(xb, tg.transform)
        x[:, w] .= y
        ll[w] = Nereus.LogDensityProblems.logdensity(tg, y)
    end
    open(io -> serialize(io, merge(ck, (state = merge(st, (x = x, lastll = ll)),))),
         path, "w")
    return path
end

# A run that ends before its burn-in throws, after writing the checkpoint.
function _bt_ens_checkpoint(n_burnin)
    path = joinpath(mktempdir(), "split.jls")
    tg = _bt_target()
    @test_throws ArgumentError sample_ensemble(tg; _BT_ENS..., n_burnin, n_steps = 2,
                                               checkpoint = path)
    return _bt_ens_inject!(path, tg)
end

function _bt_ens_resume(src, n_steps; n_burnin = 3, kw...)
    path = joinpath(mktempdir(), "state.jls")
    cp(src, path)
    return sample_ensemble(_bt_target(); _BT_ENS..., n_burnin, n_steps, checkpoint = path,
                           resume = true, kw...)
end

@testset "trim_burnin (ensemble)" begin
    src = _bt_ens_checkpoint(3)
    off = @test_logs (:warn, r"ensemble: 32 of 32 walkers were still settling into the main mode until recorded step \d+ of 997 .*rerun with n_burnin ≥ \d+ or extend_burnin = true\.$") match_mode = :any begin
        _bt_ens_resume(src, 1000)
    end
    on = _bt_ens_resume(src, 1000; trim_burnin = true)
    to = off.info.burnin_trim
    @test to.verdict == "settling" && !to.applied && off.info.n_trimmed == 0
    @test on.info.burnin_trim.applied
    # Saved draws, not blocks: the cut as detected.
    @test on.info.n_trimmed == to.cut_step
    @test on.info.n_burnin_used == 3 + to.cut_step
    @test Array(on.value.data) == off.value.data[(to.cut_step * 32 + 1):end, :, :]
    # The same run with n_burnin at the cut.
    ref = _bt_ens_resume(_bt_ens_checkpoint(3 + to.cut_step), 1000;
                         n_burnin = 3 + to.cut_step)
    @test ref.info.burnin_trim.verdict == "clean"
    @test ref.value.data == on.value.data
    # Resumed mid-recording: the log densities come back with the checkpoint.
    path = joinpath(mktempdir(), "state.jls")
    cp(src, path)
    sample_ensemble(_bt_target(); _BT_ENS..., n_steps = 500, checkpoint = path,
                    resume = true, trim_burnin = true)
    rr = sample_ensemble(_bt_target(); _BT_ENS..., n_steps = 1000, checkpoint = path,
                         resume = true, trim_burnin = true)
    @test rr.value.data == on.value.data && rr.info.burnin_trim == on.info.burnin_trim
    # The summary from the chain, as run_job and fit_* see it.
    s = Nereus._burnin_to_summary!(Dict{String,Any}(), on)
    @test s["burnin_trim"]["n_trimmed"] == to.cut_step
    @test s["n_burnin_used"] == on.info.n_burnin_used
    @test Nereus._burnin_trim_of((chains = on,)) == on.info.burnin_trim

    @testset "two chains: tested together, trimmed alike" begin
        function two(; kw...)
            path = joinpath(mktempdir(), "ens_state.jls")
            p1, p2 = (Nereus._chain_checkpoint_path(path, c) for c in 1:2)
            @test_throws Exception sample_ensemble(_bt_target(); _BT_ENS..., n_chains = 2,
                                                   n_steps = 2, checkpoint = path)
            @test timedwait(() -> isfile(p1) && isfile(p2), 60.0) === :ok
            _bt_ens_inject!(p1, _bt_target())
            _bt_ens_inject!(p2, _bt_target())
            return sample_ensemble(_bt_target(); _BT_ENS..., n_chains = 2, n_steps = 1000,
                                   checkpoint = path, resume = true, kw...)
        end
        o2 = @test_logs (:warn, r"ensemble: \d+ of 64 walkers \(all 2 chains\) were still settling") match_mode = :any two()
        n2 = two(trim_burnin = true)
        t2 = o2.info.burnin_trim
        @test t2.verdict == "settling" && t2.n_walkers == 64
        @test n2.info.n_trimmed == t2.cut_step && n2.info.burnin_trim.applied
        @test n2.info.n_burnin_used == 3 + t2.cut_step
        @test size(n2, 3) == 2
        @test n2.value.data == o2.value.data[(t2.cut_step * 32 + 1):end, :, :]
    end
end

# run_job passes `trim_burnin` through from sampler.kwargs; summary.json carries
# the burnin_trim block and fit_health the burnin_settling check, either way.
@testset "trim_burnin through run_job" begin
    include(joinpath(@__DIR__, "fixtures", "easy_rv_target.jl"))
    for trim in (false, true)
        cfg = _easy_rv_cfg()
        cfg["sampler"]["kwargs"] = Dict("show_progress" => false, "n_temps" => 4,
            "n_walkers" => 20, "n_steps" => 300, "n_burnin" => 100, "trim_burnin" => trim)
        res = Logging.with_logger(Logging.NullLogger()) do
            Nereus.run_job(cfg)
        end
        disk = JSON3.read(read(joinpath(cfg["output_dir"], "summary.json"), String);
                          allow_inf = true)
        b = disk["burnin_trim"]
        @test b["verdict"] in ("clean", "settling", "short_tail", "split",
                               "not_assessed")
        @test b["n_recorded"] == 200 && b["n_walkers"] == 20
        @test b["applied"] == (b["n_trimmed"] > 0)
        @test disk["n_burnin_used"] == 100 + b["n_trimmed"]
        @test disk["n_draws_kept"] == (300 - disk["n_burnin_used"]) * 20
        chk = disk["fit_health"]["checks"]["burnin_settling"]
        @test chk["status"] == (b["verdict"] in ("split", "short_tail") ||
                                (b["verdict"] == "settling" && !b["applied"]) ? "warn" : "ok")
    end
end
