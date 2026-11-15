# ------------------------------------------------------------------
# Burn-in that runs on while the β = 1 rung is split (`extend_burnin`; off by
# default, when nothing here runs).
#
# The stranded-walker pruning (`_pt_prune_stranded!`) works through the first
# three quarters of burn-in and no further, so a burn-in too short for it ends
# with the cold rung split, and every draw of the walkers' later migration is
# recorded. A joint RV + 12 light-curve fit (P = 9.4832 d on U(0.1, 30) d, 150
# walkers, 24 rungs, n_burnin = 1000 of 30000 steps) ended burn-in with 149 of
# its 150 cold walkers in the 2P alias, down ~965 in ln p, and one at P. They
# crossed over during the recorded run -- none at P at kept step 5000, all 150
# only by kept step 20000 -- and the posterior came out a mixture (R-hat 1.92);
# the last 10000 steps alone were clean.
#
# With `extend_burnin`, the end of burn-in is a test instead. At step n_burnin,
# and every 50 steps after while it fails, the pruning's own pass is run at the
# β = 1 rung on copies of the ensemble (`_burnin_split_pass!`), and the rung is
# split if that pass would move any of its walkers. No threshold of its own: a
# walker counts only as the pruning counts it. It sits more than the allowance
# below the best walker at this pass and at the last (the record taken 50 steps
# earlier, kept fresh at n_burnin by a record-only pass at n_burnin - 50), and
# it is below the mass bound and not climbing fast enough to clear it by the
# next pass, or in a separate mode whose own mass is under e^-tau of the best
# walker's, or far down that mode's own flank; and while too few walkers have
# found the mode for a covariance, only once the rung's median log L has risen
# by less than d/2 since the last pass. What the pruning leaves alone -- a
# healthy posterior's tail, a separate mode the bound keeps, a plateau that
# could carry mass -- extends nothing.
#
# Split: the pass is taken, at every rung, and burn-in goes on for 50 steps
# more. Not split: nothing changes (no walker, no random number, no record),
# and recording starts at the next step, so a run whose cold rung is in one
# mode at n_burnin draws exactly what it does without the option. A pass moves
# at most as many walkers as the best mode holds (one more for its climbed
# top), so the mode grows about twofold per pass at most: 149 stranded
# walkers against one leave in about eight passes, 400 steps. The walkers
# moved by the last pass get the 50 steps to the next check, the interval the
# pruning itself leaves a moved walker before judging it again; they are drawn
# about survivors with the mode's widths and evaluated, not copied.
#
# The circular re-cut, and the fold of cyclic inclinations before it, run with
# every check as they run with every nominal pass: walkers moved onto a mode
# found late land wherever the seam happens to be, which is what the re-cut
# with every pass is for (`_pt_recut_due`), and the check that ends burn-in
# fixes the chart for every recorded step. The ladder adaptation, `burnin_hold`
# and the trans-dim birth tuning stay keyed to n_burnin: none of them is about
# which mode the walkers are in.
#
# Capped at `max_burnin`, by default n_burnin plus half the steps the run would
# have recorded, so at least half of those are kept. Still split at the cap,
# the run records anyway and warns with the split it measured. Every step the
# extension takes comes out of the recorded ones: n_steps stays the total.
# ------------------------------------------------------------------

# Whether the extended burn-in runs. Its split test is the stranded-walker
# pruning's own pass, so with the pruning off there is nothing to test with,
# and asking for both is a contradiction, not a request to do nothing.
function _burnin_ext_on(engine::AbstractString, extend_burnin::Bool, prune_stranded::Bool)
    extend_burnin && !prune_stranded && throw(ArgumentError(
        "$engine: extend_burnin = true needs prune_stranded = true. The test " *
        "for walkers split between modes at the end of burn-in is the " *
        "stranded-walker pruning's own pass, so with the pruning off nothing " *
        "would extend. Turn prune_stranded back on, or drop extend_burnin."))
    return extend_burnin
end

# The last step an extended burn-in may run to: `max_burnin`, or by default
# n_burnin plus half the steps after it. Not cut at n_steps: a run that ends
# inside its extension has decided nothing, and a resume with more steps
# carries the extension on, as a run ending inside its nominal burn-in does.
function _burnin_cap(n_burnin::Integer, n_steps::Integer, max_burnin)
    cap = max_burnin === nothing ? n_burnin + max(n_steps - n_burnin, 0) ÷ 2 :
                                   Int(max_burnin)
    cap >= n_burnin || throw(ArgumentError(
        "max_burnin = $cap is below n_burnin = $n_burnin: the extended burn-in " *
        "starts at n_burnin and can only run on from there"))
    return cap
end

# One check of an extended burn-in. `pass!(arrays..., rng, memory, rungs)` is
# the stranded-walker pruning over `rungs`, returning the walkers it moved at
# each rung. It runs first at the β = 1 rung alone, on copies of `arrays`, the
# RNG and the record. If it moves none there, nothing has changed. If it moves
# some and `commit`, the copies' β = 1 rung and RNG are taken and the pass runs
# on at the other rungs with the real record: the same pass as one over every
# rung, since a rung's moves read only that rung, the hotter rungs draw after
# the cold one, and the record is taken at every rung at the end. Returns the
# walkers moved at each rung (at β = 1 only, without `commit`).
function _burnin_split_pass!(pass!, arrays::Tuple, rng, memory, n_temps::Integer;
                             commit::Bool = true)
    dry = map(deepcopy, arrays)
    r1 = copy(rng)
    mv1 = pass!(dry..., r1, deepcopy(memory), 1:1)
    (mv1[1] > 0 && commit) || return mv1
    foreach(_rung1_take!, arrays, dry)
    copy!(rng, r1)
    mv = pass!(arrays..., rng, memory, 2:n_temps)
    mv[1] = mv1[1]
    return mv
end

# The β = 1 rung of `src` into `dst`, in place.
_rung1_take!(dst::Array{Float64,3}, src) = (dst[1, :, :] .= @view src[1, :, :]; dst)
_rung1_take!(dst::Matrix{Float64}, src) = (dst[1, :] .= @view src[1, :]; dst)
function _rung1_take!(dst::AbstractMatrix{TransDimState}, src)
    for w in axes(dst, 2)
        a, b = dst[1, w], src[1, w]
        a.n_planets_active = b.n_planets_active
        copyto!(a.planet_active, b.planet_active)
        copyto!(a.noise_active, b.noise_active)
        copyto!(a.as_active, b.as_active)
    end
    return dst
end

# Where burn-in ends after a resume at step `step0`, from the checkpoint's
# `burnin_ext` = (end, last pass, passes): returns the same three. Recording
# that had begun goes on; an extension under way continues from its last pass
# under this run's settings (`cap`, `on`; its next check on a multiple of
# `thin`, as `sample_ensemble` keeps them); a checkpoint written before the
# option existed recorded from n_burnin + 1.
function _burnin_resume(ck, step0::Integer, n_burnin::Integer, cap::Integer, on::Bool;
                        thin::Integer = 1)
    hasproperty(ck, :burnin_ext) || return (n_burnin, 0, 0)
    be, last, np = ck.burnin_ext
    step0 >= be && return (be, last, np)
    step0 < n_burnin && return (n_burnin, last, np)
    return (on ? max(step0, min(cld(last + 50, thin) * thin, cap)) : step0, last, np)
end

_burnin_fmt(v) = isfinite(v) ? string(round(v; sigdigits = 6)) : string(v)
# The scaled median absolute deviation of `x` about its median `m`.
_burnin_mad(x, m) = 1.4826 * median(abs.(x .- m))

# The warning of an extended burn-in that reached its cap with the β = 1 rung
# still split. `ℓ` is the rung's log densities as the pruning measures them,
# `out` the walkers outside the best walker's mode, `Y` (walker × coordinate)
# the positions on the scale the walkers move on, `dims` the coordinates to
# compare there, `tox(j, v)` a value back to the parameter's own, `circ` the
# full-circle angles (compared as offsets on the circle). The coordinate named
# is the one along which the two groups' medians sit furthest apart against
# their own spread (the larger of the two scaled MADs), over the walkers `cmp`
# of `out` (all of them by default): the parameter the modes are told apart by
# is the one each mode pins far more tightly than they differ, a period above
# all, where a share of the prior width would name an amplitude or a phase.
# `extra` follows the measured split.
function _burnin_split_warn(sampler::AbstractString, step::Integer, n_burnin::Integer,
                            ℓ::AbstractVector{Float64}, out::AbstractVector{Int},
                            Y::AbstractMatrix{Float64}, dims, names, circ, tox;
                            extra::AbstractString = "", cmp = out)
    n = length(ℓ)
    b = argmax(ℓ)
    inn = setdiff(1:n, out)
    k = length(out)
    drop = k > 0 ? ℓ[b] - median(ℓ[out]) : NaN
    jb, sb, vo, vi = 0, -Inf, NaN, NaN
    if !isempty(cmp) && !isempty(inn)
        for j in dims
            off = [j in circ ? rem2pi(Y[w, j] - Y[b, j], RoundNearest) : Y[w, j] - Y[b, j]
                   for w in 1:n]
            mo, mi = median(off[cmp]), median(off[inn])
            sd = max(_burnin_mad(off[cmp], mo), _burnin_mad(off[inn], mi))
            s = abs(mo - mi) / sd
            if isfinite(s) && s > sb
                jb, sb, vo, vi = j, s, Y[b, j] + mo, Y[b, j] + mi
            end
        end
    end
    todo = if jb > 0
        nm = names[jb]
        " The two groups sit furthest apart, against their own spread, in $nm " *
        "($(_burnin_fmt(sb)) times it; " *
        "median $(_burnin_fmt(tox(jb, vo))) for the walkers outside, " *
        "$(_burnin_fmt(tox(jb, vi))) in the best mode). Narrow the prior of $nm " *
        "to leave the other mode out, or raise n_burnin (and max_burnin) so the " *
        "pruning has the time to move them."
    else
        " Raise n_burnin (and max_burnin) so the pruning has the time to move " *
        "them, or narrow the prior of the parameter that separates the modes."
    end
    @warn "$sampler: the β = 1 rung is still split between modes at step $step, " *
          "where the extended burn-in stops (max_burnin; n_burnin = $n_burnin): " *
          "$k of its $n walkers sit outside the best walker's mode, a median " *
          "Δln p = $(_burnin_fmt(drop)) below it, and the stranded-walker pruning " *
          "would still move them.$extra Recording anyway, so the posterior mixes " *
          "the modes and its R-hat will show it.$todo"
    return nothing
end

# The one line an extension prints at the end of the run (`n_done` steps),
# and a warning if the run ended inside it, with nothing recorded.
function _burnin_extend_info(sampler::AbstractString, n_burnin::Integer,
                             b_end::Integer, passes::Integer, n_steps::Integer;
                             n_done::Integer = n_steps, quiet::Bool = false)
    b_end > n_burnin || return nothing
    if n_done <= b_end
        @warn "$sampler: the run ended at step $n_done inside its extended burn-in " *
              "(the β = 1 rung was still split between modes; next check at step " *
              "$b_end), so nothing was recorded. Resume it with a larger n_steps to " *
              "carry the extension on, or raise n_burnin."
        return nothing
    end
    quiet && return nothing
    @info "$sampler: the β = 1 rung was still split between modes at step " *
          "$n_burnin: burn-in ran on to step $b_end ($passes stranded-walker " *
          "pruning pass(es) after n_burnin), and the recording started there, " *
          "$(max(n_steps - b_end, 0)) steps instead of $(max(n_steps - n_burnin, 0))."
    return nothing
end

# `_burnin_split_warn` for `sample_pt_emcee`: the β = 1 rung on the flat scale
# the walkers move on, the walkers outside its best walker's allowance over the
# `d` coordinates not held, compared with those inside.
function _pt_split_warn(state::Array{Float64,3}, logπ_arr::Matrix{Float64},
                        logL_arr::Matrix{Float64}, step::Integer, n_burnin::Integer,
                        layout, flat_shift, circ, d::Integer)
    ℓ = Float64[(v = logπ_arr[1, w] + logL_arr[1, w]; isfinite(v) ? v : -Inf)
                for w in axes(state, 2)]
    out = findall(<(maximum(ℓ) - _prune_allow(max(d, 1))), ℓ)
    tox = (j, v) -> flat_shift[j] === nothing ? v : exp(v) - flat_shift[j]
    _burnin_split_warn("pt_emcee", step, n_burnin, ℓ, out, state[1, :, :],
                       1:size(state, 3), layout.unfrozen_names, circ, tox)
end

# The same for `sample_transdim_pt_emcee`, whose walkers live in x: the β = 1
# rung's densities on the flat scale of each walker's active coordinates, as
# `_td_prune_stranded!` measures them, and outside the best walker's mode
# meaning in another model or more than the allowance below it. The coordinate
# named is compared over the walkers outside that share the best model, if any.
function _td_split_warn(state::Array{Float64,3}, td_states::AbstractMatrix{TransDimState},
                        logπ_arr::Matrix{Float64}, logL_arr::Matrix{Float64},
                        step::Integer, n_burnin::Integer, layout, flat_shift,
                        owner::Vector{Int}, circ, held)
    n_w, n_dim = size(state, 2), size(state, 3)
    active(tds, o) = o == 0 || (o > 0 ? tds.planet_active[o] : tds.noise_active[-o])
    same(a, b) = a.planet_active == b.planet_active &&
                 a.noise_active == b.noise_active && a.as_active == b.as_active
    flat(x, s) = s === nothing ? x : (x + s > 0 ? log(x + s) : -Inf)
    ℓ = Vector{Float64}(undef, n_w)
    for w in 1:n_w
        v = logπ_arr[1, w] + logL_arr[1, w]
        for d in 1:n_dim
            s = flat_shift[d]
            (s === nothing || !active(td_states[1, w], owner[d])) && continue
            v += flat(state[1, w, d], s)
        end
        ℓ[w] = isfinite(v) ? v : -Inf
    end
    b = argmax(ℓ)
    tb = td_states[1, b]
    dims = [d for d in 1:n_dim if active(tb, owner[d]) && !(d in held)]
    allow = _prune_allow(max(length(dims), 1))
    out = [w for w in 1:n_w if !(ℓ[w] >= ℓ[b] - allow && same(td_states[1, w], tb))]
    cmp = [w for w in out if same(td_states[1, w], tb)]
    n_other = length(out) - length(cmp)
    Y = [flat(state[1, w, d], flat_shift[d]) for w in 1:n_w, d in 1:n_dim]
    tox = (j, v) -> flat_shift[j] === nothing ? v : exp(v) - flat_shift[j]
    extra = n_other > 0 ? " $n_other of them are in another model than the best " *
                          "walker's ($(tb.n_planets_active) planet(s))." : ""
    _burnin_split_warn("td-pt_emcee", step, n_burnin, ℓ, out, Y, dims,
                       layout.unfrozen_names, circ, tox; extra, cmp)
end

# `summary["n_burnin_used"]`: the steps before recording began, from a result
# that reports them (`n_burnin_used`, or a chain's info as `sample_ensemble`
# returns it, in saved draws as its `n_burnin` counts them); the summary is left as it was otherwise. Shared by run_job and
# the fit_* route.
function _burnin_to_summary!(summary::AbstractDict, result)
    v = hasproperty(result, :n_burnin_used) ? getproperty(result, :n_burnin_used) :
        result isa MCMCChains.Chains ? get(result.info, :n_burnin_used, nothing) :
        hasproperty(result, :chains) && result.chains isa MCMCChains.Chains ?
            get(result.chains.info, :n_burnin_used, nothing) : nothing
    v === nothing || (summary["n_burnin_used"] = v)
    return summary
end

# The same for `sample_ensemble`, one rung at β = 1: its walkers `x` (coordinate
# × walker) in its sampling coordinates, their densities there as its pruning
# measures them, and the coordinates compared in the parameters' own values, on
# the scale each prior is flat on. `step` and `n_burnin` count saved draws.
function _ensemble_split_warn(x::Matrix{Float64}, target, step::Integer, n_burnin::Integer)
    nd, nw = size(x)
    tr = target.transform
    shift = [log_scale_shift(p) for p in target.params.layout.unfrozen_priors]
    ℓ = Vector{Float64}(undef, nw)
    Y = Matrix{Float64}(undef, nw, nd)
    for w in 1:nw
        lp, ll = _logdensity_parts(target, x[:, w])
        ℓ[w] = isfinite(lp + ll) ? lp + ll : -Inf
        xb = tr === nothing ? x[:, w] : transform_inverse(x[:, w], tr)
        for j in 1:nd
            s = shift[j]
            Y[w, j] = s === nothing ? xb[j] : log(xb[j] + s)
        end
    end
    out = findall(<(maximum(ℓ) - _prune_allow(nd)), ℓ)
    tox = (j, v) -> shift[j] === nothing ? v : exp(v) - shift[j]
    _burnin_split_warn("ensemble", step, n_burnin, ℓ, out, Y, 1:nd,
                       target.params.layout.unfrozen_names,
                       circular_indices(target.params), tox)
end
