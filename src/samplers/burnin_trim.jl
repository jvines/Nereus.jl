# ------------------------------------------------------------------
# Walkers still settling into the main mode during the recorded run, and the
# post-hoc trim of their draws (`trim_burnin`; off by default, when the
# detection still runs and only warns).
#
# A joint RV + 11 light-curve fit (pt_emcee, 24 rungs, 150 walkers, n_steps
# 30000, n_burnin 1000, P on U(0.1, 30) d) ended burn-in with 149 of its 150
# β = 1 walkers in the 2P alias of P = 9.4832 d, about 965 below the main mode
# in ln p. They crossed over during the recorded run -- none at P at recorded
# step 5000, 67 at 10000, 137 at 15000, all 150 by 20000 -- and the posterior
# came out a mixture (R-hat 1.92, a kinked median model). The tail was clean.
# R-hat cannot place the cut: the worst R-hat of that run sat at 1.09-1.14 from
# recorded step 18000 to 26000 because a limb-darkening coefficient mixes
# slowly, and a cut chosen by R-hat would have dropped nearly everything.
#
# The rule (`_trim_detect`), on the log densities the walkers sample, as the
# stranded-walker pruning measures them:
#
# - Allowance: the pruning's own (`_prune_allow`, the 1 - 1e-6 quantile of a
#   d-dimensional Gaussian's drop from its peak), over the free dimension d less
#   any held coordinate, as the pruning counts it. The trim and the pruning then
#   agree on what "far below" means. Derived columns do not count.
# - Reference: each walker's median over the final tenth of the recorded steps,
#   and the largest of those over the walkers. A median is a typical level, not
#   a peak, and so is every level it is compared with below; taking the best
#   walker's keeps the reference where it should be however many walkers end
#   the run in another mode.
# - Band: a walker is in the main band at a recorded step when most of the draws
#   in a centred window about it are within the allowance of the reference, i.e.
#   when the window's median is (`_trim_halfwidth` gives the window).
# - Settling: a walker's draws from the start of recording up to, not including,
#   its first recorded step in the band; none if it starts in it, or if those
#   draws sit, on median, in the band themselves (a window cut short at the
#   start reaches forward only, into a tail excursion just after it).
# - Cut: the last settling draw of any walker.
#
# Time below the band after a walker's first entry is posterior mass, not
# burn-in, however long it lasts and whether or not the walker is back by the
# end: a correct kernel takes a walker out of the main mode into a region of
# negligible mass with negligible probability, so where it goes from there the
# posterior goes. A stationary run whose walkers wander into a heavy tail for
# longer than the window, late in the run as much as early, has nothing to trim.
# (A low, wide mode -- a weak signal's undetected plateau -- can carry real mass
# far below the peak; the pruning's mass bound exists for that case, and this is
# the trim's version of it.) Modes of comparable height sit within the
# allowance, so neither is below the other.
#
# Two outcomes are refused, with a warning. Walkers never in the band that are
# below it at the end of the run: the run ends split, and dropping draws cannot
# rescue it. A cut that would leave fewer than a quarter of the recorded steps.
#
# Trans-dim (`sample_transdim_pt_emcee`): log densities of different models are
# never compared. A model with fewer planets sits lower in ln p and still carries
# mass through the Occam factor, so the rule runs within each model (the
# pattern of active planet slots and noise models), with that model's free
# dimension and its own reference, per walker and model: a draw in a model is
# settling when it comes before that walker's first entry into that model's
# band. A model absent from the final tenth has no reference and flags nothing.
#
# The trimmed result is the run as if recording had started at the cut: the
# draws, log densities and derived columns, the evidence (bridge and Laplace
# from the kept draws; thermodynamic integration from per-block sums of each
# rung's log L, `EvidenceBlocks`), and every diagnostic computed from them. A PT
# cut is rounded up to the next block boundary (50 steps, or the next multiple
# of `thin` above), so the draws and the evidence share one window. The run's
# own state, and its checkpoint, are untouched: a resumed run trims at its end on
# everything recorded.
# ------------------------------------------------------------------

# Half-width, in recorded steps, of the window a walker's level is the median
# of: 25 (a 51-draw window) in a long run, n_rec ÷ 40 in a shorter one, never
# under 2. A dip has to hold more than half the window, 26 draws, to count, so a
# healthy walker's excursion into its posterior's tail, shorter than that, is
# never one. The cut lands within that many draws of a walker's actual
# crossing, and at most a fortieth of the run is smeared at either end.
_trim_halfwidth(n_rec::Integer) = clamp(n_rec ÷ 40, 2, 25)

# The final window the reference is taken over: the last tenth of the recorded
# steps, at least 10.
_trim_final(n_rec::Integer) = min(n_rec, max(cld(n_rec, 10), 10))

# Fewer recorded steps than this, and nothing is assessed.
const _TRIM_MIN_ROWS = 20

"""
    _trim_detect(L, ds; model = nothing) -> NamedTuple

The settling test on `L`, the log densities of the recorded draws (recorded
step × walker), on the scale the walkers sample. `ds[m]` is model `m`'s free
dimension and `model` (same shape as `L`) each draw's model, or `nothing` for
one model. Returns `(; verdict, cut, n_rec, n_walkers, n_settling, delta,
stuck, halfwidth, final)`:

- `verdict` -- `:settling` (walkers reached the main band only during the
  recorded run, the last at `cut`, with at least a quarter of the recorded
  steps after it), `:short_tail` (the same, with fewer after it), `:split`
  (walkers never in the band and below it at the end; `stuck` lists them and
  `stuck_model` the model each is stuck in), `:clean`, or `:not_assessed`
  (fewer than `_TRIM_MIN_ROWS` recorded steps, or a non-number in `L`).
- `cut` -- the last settling draw's recorded step: the latest, over walkers
  (and models), of the step before a walker's first entry into the band (0 if
  every walker starts in it). A walker whose draws before that entry sit, on
  median, in the band has none.
- `n_settling` -- the walkers with settling draws (`:split`: the stuck ones).
- `delta` -- how far below the reference, in ln p: the median over the
  settling draws (`:split`: over the stuck walkers' final medians).
"""
function _trim_detect(L::AbstractMatrix{<:Real}, ds::AbstractVector{<:Integer};
                      model::Union{Nothing,AbstractMatrix{<:Integer}} = nothing)
    n_rec, n_w = size(L)
    h = _trim_halfwidth(max(n_rec, 1))
    nf = _trim_final(max(n_rec, 1))
    out(verdict; cut = 0, n_settling = 0, delta = NaN, stuck = Int[],
        stuck_model = Int[]) =
        (; verdict, cut, n_rec, n_walkers = n_w, n_settling, delta, stuck,
         stuck_model, halfwidth = h, final = nf)
    (n_rec >= _TRIM_MIN_ROWS && n_w >= 1) || return out(:not_assessed)
    any(isnan, L) && return out(:not_assessed)
    M = length(ds)
    f0 = n_rec - nf + 1
    # Draws in a model that a walker needs in the final window for a median.
    nmin = max(3, cld(nf, 4))
    rows = [Int[] for _ in 1:M]
    function bucket!(w)
        foreach(empty!, rows)
        for i in 1:n_rec
            push!(rows[model === nothing ? 1 : Int(model[i, w])], i)
        end
        return rows
    end
    # Each walker's final-window median in each model, and the reference.
    medf = fill(NaN, M, n_w)
    ref = fill(-Inf, M)
    buf = Float64[]
    for w in 1:n_w
        bucket!(w)
        for m in 1:M
            empty!(buf)
            for i in rows[m]
                i >= f0 && push!(buf, L[i, w])
            end
            length(buf) >= nmin || continue
            medf[m, w] = median(buf)
            medf[m, w] > ref[m] && (ref[m] = medf[m, w])
        end
    end
    thr = [isfinite(ref[m]) ? ref[m] - _prune_allow(max(Int(ds[m]), 1)) : -Inf
           for m in 1:M]
    # Each walker along its own draws in each model: settling up to its first
    # draw whose window is not mostly below the band (its first entry), never
    # after it.
    cut = 0                               # the last settling draw's step
    set_w = falses(n_w)                   # walkers with settling draws
    drops = Float64[]                     # reference minus log density, settling
    stuck = Tuple{Int,Int}[]
    cs = Int[]
    for w in 1:n_w
        bucket!(w)
        for m in 1:M
            isfinite(ref[m]) || continue
            I = rows[m]
            n = length(I)
            n == 0 && continue
            resize!(cs, n)
            c = 0
            for j in 1:n
                c += !(L[I[j], w] >= thr[m])
                cs[j] = c
            end
            f = n + 1                     # the first entry into the band
            for j in 1:n
                lo, hi = max(1, j - h), min(n, j + h)
                if 2 * (cs[hi] - (lo > 1 ? cs[lo - 1] : 0)) <= hi - lo + 1
                    f = j
                    break
                end
            end
            f == 1 && continue            # in the band from the start
            # Never in the band and below it at the end: the run is split.
            if f > n && medf[m, w] < thr[m]
                push!(stuck, (w, m))
                continue
            end
            # Its draws before the first entry sitting, on median, in the band:
            # not a climb but a tail excursion that began just after recording
            # did, which the first windows, cut short at the start, reach
            # forward into.
            empty!(buf)
            for j in 1:(f - 1)
                push!(buf, L[I[j], w])
            end
            median(buf) >= thr[m] && continue
            set_w[w] = true
            cut = max(cut, I[f - 1])
            append!(drops, ref[m] .- buf)
        end
    end
    if !isempty(stuck)
        ws = unique(first.(stuck))
        delta = median([ref[m] - medf[m, w] for (w, m) in stuck])
        return out(:split; n_settling = length(ws), delta, stuck = first.(stuck),
                   stuck_model = last.(stuck))
    end
    cut == 0 && return out(:clean)
    verdict = 4 * (n_rec - cut) < n_rec ? :short_tail : :settling
    return out(verdict; cut, n_settling = count(set_w), delta = median(drops))
end

# The recorded log densities as a matrix (recorded step × walker), from a
# vector stacked walker by walker within each step, as the PT samplers record.
_trim_matrix(v::AbstractVector{<:Real}, n_w::Integer) =
    permutedims(reshape(collect(Float64, v), n_w, :))

# ------------------------------------------------------------------
# Evidence sums in blocks of recorded steps, so the tempered evidence can be
# taken over the steps a trim keeps. Each block is an `EvidenceAccumulator` fed
# exactly as the run's own (which is left as it was: a run that trims nothing
# reports the run's numbers bit for bit). A new block starts at every `len`-th
# step after the end of burn-in, `len` a multiple of `thin`, so block starts are
# recorded steps. A checkpoint written before the blocks existed resumes with
# its accumulator as the first block, which then runs on to the next boundary.
# ------------------------------------------------------------------
mutable struct EvidenceBlocks
    len::Int
    start::Vector{Int}                    # the first step of each block
    acc::Vector{EvidenceAccumulator}
end

EvidenceBlocks(thin::Integer) = EvidenceBlocks(thin * cld(50, thin), Int[],
                                               EvidenceAccumulator[])

# The block step `step` feeds (recording from `b_end + 1`), started if due.
function _evidence_block!(eb::EvidenceBlocks, step::Integer, b_end::Integer,
                          n_chains::Integer)
    if isempty(eb.acc) || ((step - b_end - 1) % eb.len == 0 && eb.start[end] != step)
        push!(eb.start, step)
        push!(eb.acc, EvidenceAccumulator(n_chains))
    end
    return eb.acc[end]
end

# From a checkpoint: its blocks, or, from one written before they existed, its
# accumulator as one block from the start of recording.
function _evidence_blocks_restore!(eb::EvidenceBlocks, ck, evidence_acc, b_end::Integer)
    if hasproperty(ck, :evidence_blocks)
        c = ck.evidence_blocks
        append!(eb.start, c.start); append!(eb.acc, deepcopy(c.acc))
    elseif evidence_acc.started
        push!(eb.start, b_end + 1); push!(eb.acc, deepcopy(evidence_acc))
    end
    return eb
end

# The recorded-step offsets a cut may take: those at which a block starts.
_evidence_cuts(eb::EvidenceBlocks, b_end::Integer, thin::Integer) =
    Int[(s - b_end - 1) ÷ thin for s in eb.start if (s - b_end - 1) % thin == 0]

# The evidence of the blocks from the one starting at step `from` on.
function _evidence_from(eb::EvidenceBlocks, from::Integer, n_chains::Integer)
    out = EvidenceAccumulator(n_chains)
    for (s, a) in zip(eb.start, eb.acc)
        s >= from || continue
        out.n .+= a.n
        out.sum_logL .+= a.sum_logL
        out.sum_logL2 .+= a.sum_logL2
        out.n_nonfinite .+= a.n_nonfinite
        out.ss_lse_fwd .= _logaddexp.(out.ss_lse_fwd, a.ss_lse_fwd)
        out.ss_lse_bwd .= _logaddexp.(out.ss_lse_bwd, a.ss_lse_bwd)
        out.started |= a.started
    end
    return out
end

# ------------------------------------------------------------------
# The decision, the record and the one message.
# ------------------------------------------------------------------

# The recorded steps a trim of `rec` drops: 0 unless it is `:settling`; else the
# detected cut rounded up to the first of `cuts` (allowed offsets; every step
# when `nothing`) at or above it, or -1 when that would leave fewer than a
# quarter of the recorded steps (refused).
function _trim_rows(rec; cuts = nothing)
    rec.verdict === :settling || return 0
    c = cuts === nothing ? rec.cut :
        minimum((k for k in cuts if k >= rec.cut); init = typemax(Int))
    (c < typemax(Int) && 4 * (rec.n_rec - c) >= rec.n_rec) || return -1
    return c
end

_trim_fmt(v) = isfinite(v) ? string(round(Int, v)) : string(v)

"""
    _trim_report(sampler, rec; trim, rows, n_trimmed, n_burnin_used, thin = 1,
                 extend = nothing, who = "walkers at β = 1", split = ("", nothing),
                 show_progress = true) -> NamedTuple

Emits the one message of the burn-in trim (a warning when something needs
doing, an info line, with `show_progress`, when the trim was applied) and
returns the record for the result and summary.json: `(; applied, verdict,
cut_step, n_recorded, n_walkers, n_walkers_settling, delta_lnp, n_trimmed,
status, message)`. `rows` is what `_trim_rows` gives (-1: a trim would leave
under a quarter), `n_trimmed` the recorded steps dropped; `n_burnin_used` the
steps before recording began, counted as the sampler counts `n_burnin`, its
recorded steps `thin` apart; `extend` whether `extend_burnin` was on
(`nothing` where the sampler has none); `split` what `_trim_split_text` gives,
for a split run.
"""
function _trim_report(sampler::AbstractString, rec; trim::Bool, rows::Integer,
                      n_trimmed::Integer, n_burnin_used::Integer, thin::Integer = 1,
                      extend::Union{Nothing,Bool} = nothing,
                      who::AbstractString = "walkers at β = 1",
                      split::Tuple = ("", nothing), show_progress::Bool = true)
    v = rec.verdict
    v === :settling && rows < 0 && (v = :short_tail)
    k, n, nr, cut = rec.n_settling, rec.n_walkers, rec.n_rec, rec.cut
    Δ = _trim_fmt(rec.delta)
    at = n_burnin_used + cut * thin             # the step of the last draw below
    ext = extend === false ? " or extend_burnin = true" : ""
    status, msg = "ok", ""
    settled = "$k of $n $who were still settling into the main mode until " *
              "recorded step $cut of $nr (Δln p ≈ $Δ below it), so the draws up to " *
              "it are not yet from the posterior"
    if v === :settling && n_trimmed > 0
        msg = "$sampler: trim_burnin dropped the first $n_trimmed of $nr recorded " *
              "steps: $k of $n $who were still settling into the main mode until " *
              "recorded step $cut (Δln p ≈ $Δ below it). The draws, diagnostics and " *
              "evidence are those of the last $(nr - n_trimmed) steps, and " *
              "n_burnin_used counts the dropped ones."
    elseif v === :settling
        status = "warn"
        msg = "$sampler: $settled. Set trim_burnin = true to drop them, or rerun " *
              "with n_burnin ≥ $at$ext."
    elseif v === :short_tail
        status = "warn"
        msg = "$sampler: $settled, and dropping them would leave fewer than a " *
              "quarter of the recorded steps$(trim ? ", so trim_burnin = true has " *
              "dropped nothing" : ""). Rerun with n_burnin ≥ $at (and n_steps " *
              "raised by as much)$(extend === false ? ", or with extend_burnin = true" : "")."
    elseif v === :split
        status = "warn"
        sentence, nm = split
        todo = nm === nothing ?
            " Rerun with a longer n_burnin$ext, or narrow the prior of the parameter " *
            "that separates the modes." :
            " Narrow the prior of $nm to leave the other mode out, or rerun with a " *
            "longer n_burnin$ext."
        msg = "$sampler: $k of $n $who are still out of the main mode at the end " *
              "of the run (their median over the last $(rec.final) recorded steps " *
              "sits Δln p ≈ $Δ below the best walker's) and were never in it, so the " *
              "posterior mixes two modes and no trim can rescue it" *
              "$(trim ? "; trim_burnin = true has dropped nothing" : "")." *
              sentence * todo
    elseif v === :clean
        msg = "$sampler: no $who were settling into the main mode during the " *
              "recorded run."
    else
        msg = "$sampler: settling into the main mode not assessed " *
              "($(nr < _TRIM_MIN_ROWS ? "fewer than $_TRIM_MIN_ROWS recorded steps" :
                  "no log density for some recorded draws"))."
    end
    if status == "warn"
        @warn msg
    elseif v === :settling && n_trimmed > 0 && show_progress
        @info msg
    end
    return (; applied = n_trimmed > 0, verdict = String(v), cut_step = cut,
            n_recorded = nr, n_walkers = n, n_walkers_settling = k,
            delta_lnp = Float64(rec.delta), n_trimmed, status, message = msg)
end

# The parameter the stuck walkers `out` and the rest sit furthest apart in
# (`_split_coordinate`), and the sentence saying so: `(sentence, name)`, or
# `("", nothing)`. `ℓ` the walkers' levels, `Y` (walker × coordinate) their
# positions on the scale they move on, `dims` the coordinates compared,
# `tox(j, v)` a value back to the parameter's own.
function _trim_split_text(ℓ, out, Y, dims, names, circ, tox; cmp = out)
    jb, sb, vo, vi = isempty(ℓ) ? (0, -Inf, NaN, NaN) :
                     _split_coordinate(ℓ, out, Y, dims, circ; cmp)
    jb > 0 || return ("", nothing)
    nm = String(names[jb])
    return (" The two groups sit furthest apart, against their own spread, in $nm " *
            "($(_burnin_fmt(sb)) times it; median $(_burnin_fmt(tox(jb, vo))) for the " *
            "walkers outside, $(_burnin_fmt(tox(jb, vi))) in the main mode).", nm)
end

# ------------------------------------------------------------------
# The fixed-dimension PT samplers (pt_emcee, pt_whitening): rows of `samples`
# (x) and `lp` stacked walker by walker within each recorded step. `flat_shift`
# maps x to the scale the walkers move on (`nothing`: they move in x), where the
# log density gains Σ log(x + s) over the log-scale coordinates. Returns the
# record and the recorded steps to drop.
# ------------------------------------------------------------------
function _pt_trim(sampler::AbstractString, samples::AbstractMatrix{Float64},
                  lp::AbstractVector{Float64}, n_w::Integer, d::Integer;
                  flat_shift = nothing, trim::Bool, cuts, n_burnin_used::Integer,
                  thin::Integer, extend, circ, names, show_progress::Bool = true)
    n_keep = length(lp)
    ℓ = copy(lp)
    if flat_shift !== nothing
        for (j, s) in enumerate(flat_shift)
            s === nothing && continue
            @inbounds for r in 1:n_keep
                x = samples[r, j] + s
                ℓ[r] += x > 0 ? log(x) : -Inf
            end
        end
    end
    rec = (n_w > 0 && n_keep % n_w == 0) ? _trim_detect(_trim_matrix(ℓ, n_w), [d]) :
          _trim_detect(zeros(0, max(n_w, 1)), [d])
    rows = _trim_rows(rec; cuts)
    n_trim = trim ? max(rows, 0) : 0
    split = ("", nothing)
    if rec.verdict === :split
        # Positions at the last recorded step, on the walkers' own scale.
        last = (n_keep - n_w + 1):n_keep
        Y = Float64[(s = flat_shift === nothing ? nothing : flat_shift[j];
                     s === nothing ? samples[r, j] : log(samples[r, j] + s))
                    for r in last, j in 1:size(samples, 2)]
        lev = vec(median(_trim_matrix(ℓ, n_w)[(rec.n_rec - rec.final + 1):end, :];
                         dims = 1))
        tox = (j, v) -> (flat_shift === nothing || flat_shift[j] === nothing) ? v :
                        exp(v) - flat_shift[j]
        split = _trim_split_text(lev, rec.stuck, Y, 1:size(samples, 2), names, circ, tox)
    end
    t = _trim_report(sampler, rec; trim, rows, n_trimmed = n_trim, n_burnin_used,
                     thin, extend, split, show_progress)
    return t, n_trim
end

# ------------------------------------------------------------------
# The trans-dim sampler: rows of `samples` hold the parameters (x, `n_dim`
# columns), then n_planets, the `max_k` slot indicators and the indicators of
# the toggled noise models `toggled` (indices into the noise models). A draw's
# model is its pattern of indicators; its active coordinates are those `owner`
# gives to an always-on component (0), an active slot (k > 0) or an active noise
# model (-j; one never toggled has the activity `noise_on` gives it all run).
# The log density gains Σ log(x + s) over the active log-scale coordinates, as
# `_td_prune_stranded!` measures it, and a model's free dimension is its active
# coordinates less the held ones.
# ------------------------------------------------------------------
function _td_trim(samples::AbstractMatrix{Float64}, lp::AbstractVector{Float64},
                  n_w::Integer, n_dim::Integer, max_k::Integer,
                  toggled::AbstractVector{Int}, noise_on::AbstractVector{Bool},
                  owner::AbstractVector{Int}, flat_shift, held; trim::Bool, cuts,
                  n_burnin_used::Integer, thin::Integer, extend, circ, names,
                  show_progress::Bool = true)
    n_keep = length(lp)
    n_ind = max_k + length(toggled)
    ids = Vector{Int}(undef, n_keep)
    pats = Dict{Vector{Bool},Int}()
    act = Vector{Vector{Int}}()               # active coordinates per model
    for r in 1:n_keep
        p = Bool[samples[r, n_dim + 1 + k] > 0.5 for k in 1:n_ind]
        ids[r] = get!(pats, p) do
            na = collect(Bool, noise_on)
            for (j, nm) in enumerate(toggled)
                na[nm] = p[max_k + j]
            end
            push!(act, [d for d in 1:n_dim
                        if (o = owner[d]; o == 0 || (o > 0 ? p[o] : na[-o]))])
            length(act)
        end
    end
    ℓ = copy(lp)
    for r in 1:n_keep, d in act[ids[r]]
        s = flat_shift[d]
        s === nothing && continue
        x = samples[r, d] + s
        ℓ[r] += x > 0 ? log(x) : -Inf
    end
    ds = [count(d -> !(d in held), a) for a in act]
    ok = n_w > 0 && n_keep % n_w == 0
    rec = ok ? _trim_detect(_trim_matrix(ℓ, n_w), ds;
                            model = permutedims(reshape(ids, n_w, :))) :
          _trim_detect(zeros(0, max(n_w, 1)), [1])
    rows = _trim_rows(rec; cuts)
    n_trim = trim ? max(rows, 0) : 0
    split = ("", nothing)
    if rec.verdict === :split
        # Within the model most stuck walkers are in: the walkers in it at the
        # last recorded step, on the flat scale of its active coordinates.
        m = argmax(k -> count(==(k), rec.stuck_model), unique(rec.stuck_model))
        last = (n_keep - n_w + 1):n_keep
        sub = [w for w in 1:n_w if ids[last[w]] == m]
        out = findall(in(Set(rec.stuck[rec.stuck_model .== m])), sub)
        Y = Float64[(s = flat_shift[j]; s === nothing ? samples[last[w], j] :
                     log(samples[last[w], j] + s)) for w in sub, j in 1:n_dim]
        tox = (j, v) -> flat_shift[j] === nothing ? v : exp(v) - flat_shift[j]
        split = isempty(sub) ? _trim_split_text(Float64[], Int[], zeros(0, 0), 1:0,
                                                names, circ, tox) :
                _trim_split_text(ℓ[last[sub]], out, Y,
                                 [d for d in act[m] if !(d in held)], names, circ, tox)
    end
    t = _trim_report("td-pt_emcee", rec; trim, rows, n_trimmed = n_trim, n_burnin_used,
                     thin, extend, split, show_progress)
    return t, n_trim
end

# ------------------------------------------------------------------
# `sample_ensemble`: `L` (recorded draw × walker) the log densities in its
# sampling coordinates, as its pruning measures them, and `X` (walker ×
# parameter) the walkers' last recorded positions in the parameters' own
# values. Counted in saved draws, like its `n_burnin`.
# ------------------------------------------------------------------
function _ens_trim(L::AbstractMatrix{Float64}, X::AbstractMatrix{Float64}, d::Integer,
                   params; trim::Bool, n_burnin_used::Integer, extend,
                   who::AbstractString = "walkers", show_progress::Bool = true)
    rec = _trim_detect(L, [d])
    rows = _trim_rows(rec)
    n_trim = trim ? max(rows, 0) : 0
    split = ("", nothing)
    if rec.verdict === :split
        shift = [log_scale_shift(p) for p in params.layout.unfrozen_priors]
        Y = Float64[shift[j] === nothing ? X[w, j] : log(X[w, j] + shift[j])
                    for w in axes(X, 1), j in axes(X, 2)]
        lev = vec(median(L[(rec.n_rec - rec.final + 1):end, :]; dims = 1))
        tox = (j, v) -> shift[j] === nothing ? v : exp(v) - shift[j]
        split = _trim_split_text(lev, rec.stuck, Y, axes(X, 2),
                                 params.layout.unfrozen_names, circular_indices(params),
                                 tox)
    end
    t = _trim_report("ensemble", rec; trim, rows, n_trimmed = n_trim, n_burnin_used,
                     extend, who, split, show_progress)
    return t, n_trim
end

# `summary["burnin_trim"]` from a result's record (`burnin_trim`, or a chain's
# info as `sample_ensemble` returns it): written whether or not the trim was
# applied. Shared by run_job and the fit_* route.
function _burnin_trim_of(result)
    t = hasproperty(result, :burnin_trim) ? getproperty(result, :burnin_trim) :
        result isa MCMCChains.Chains ? get(result.info, :burnin_trim, nothing) :
        hasproperty(result, :chains) && result.chains isa MCMCChains.Chains ?
            get(result.chains.info, :burnin_trim, nothing) : nothing
    return t isa NamedTuple ? t : nothing
end

function _burnin_trim_summary(t::NamedTuple)
    return Dict{String,Any}(
        "applied" => t.applied, "verdict" => t.verdict, "cut_step" => t.cut_step,
        "n_recorded" => t.n_recorded, "n_walkers" => t.n_walkers,
        "n_walkers_settling" => t.n_walkers_settling, "delta_lnp" => t.delta_lnp,
        "n_trimmed" => t.n_trimmed)
end

# The fit_health check of the same finding (`_run_fit_health!`).
function _burnin_trim_check(t)
    t === nothing && return nothing
    return FitHealthCheck(:burnin_settling, t.status == "warn" ? :warn : :ok, t.message)
end
