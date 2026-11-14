# Ending a production run early, on purpose, without losing the draws.
#
# Four things can end a PT ensemble run before `n_steps`, and all four funnel
# into one path: the loop stops at the last fully recorded step, the sampler
# finishes normally (chains, evidence, diagnostics), and the result says how it
# ended in a `stop` field -- `(; stopped, step, n_steps, n_draws, stop_file)`.
#
#   "converged"    the sampler's own R-hat / ESS gate (`convergence_stop`)
#   "interrupted"  an InterruptException (Ctrl-C, a notebook interrupt) after burn-in
#   "requested"    the stop file appeared (`stop_file`)
#   "max_seconds"  the wall-clock limit passed (`max_seconds`)
#   "max_evals"    the likelihood-evaluation limit passed (`max_evals`)
#   "steps"        none of the above: the run went to `n_steps`
#
# Before the end of burn-in there are no usable draws, so a stop request there
# is an error, not a result.

"How often, in wall-clock seconds, the stop file is looked for."
const STOP_POLL_SECONDS = 5.0

"""
    StopControl

The out-of-band stops one run watches: a wall-clock limit, an evaluation limit
and a stop file. Built once per run; [`_stop_check!`](@ref) is cheap enough to
call every step because the file system is only touched every
`STOP_POLL_SECONDS` (or when forced, at the diagnostic cadence).
"""
mutable struct StopControl
    t0::Float64
    max_seconds::Float64
    max_evals::Float64
    stop_file::Union{Nothing,String}
    next_poll::Float64
end

function StopControl(; max_seconds::Real = Inf, max_evals::Real = Inf,
                     stop_file = nothing)
    max_seconds > 0 || throw(ArgumentError(
        "max_seconds must be positive; got $max_seconds"))
    max_evals > 0 || throw(ArgumentError(
        "max_evals must be positive; got $max_evals"))
    sf = (stop_file === nothing || isempty(String(stop_file))) ? nothing :
         String(stop_file)
    return StopControl(time(), Float64(max_seconds), Float64(max_evals), sf, 0.0)
end

"""
    _stop_check!(sc, n_evals; force=false) -> Union{Nothing,String}

The reason a limit or the stop file asks the run to end now, or `nothing`.
"""
function _stop_check!(sc::StopControl, n_evals::Integer; force::Bool = false)
    now = time()
    if isfinite(sc.max_seconds) && now - sc.t0 >= sc.max_seconds
        return "max_seconds"
    end
    isfinite(sc.max_evals) && n_evals >= sc.max_evals && return "max_evals"
    if sc.stop_file !== nothing && (force || now >= sc.next_poll)
        sc.next_poll = now + STOP_POLL_SECONDS
        isfile(sc.stop_file) && return "requested"
    end
    return nothing
end

"The error for a stop asked for while there is still nothing to keep."
function _stop_in_burnin_error(sampler::AbstractString, reason::AbstractString,
                               sc::StopControl, step::Int, n_burnin::Int)
    what = reason == "requested" ?
        "a stop was requested (stop file $(sc.stop_file))" :
        reason == "max_seconds" ?
        "max_seconds = $(sc.max_seconds) elapsed" :
        "max_evals = $(Int(sc.max_evals)) was reached"
    return ErrorException(
        "$sampler: $what at step $step, before the end of burn-in " *
        "($n_burnin steps): no production draws existed yet, so nothing was " *
        "kept. Raise the limit, or let the run get past burn-in.")
end

"The one progress line a graceful stop prints, on the progress bar's stream."
function _announce_stop(reason::AbstractString, step::Int, n_steps::Int,
                        n_draws::Int)
    why = reason == "converged"   ? "converged" :
          reason == "interrupted" ? "interrupted" :
          reason == "requested"   ? "stop requested" :
          reason == "max_seconds" ? "max_seconds reached" :
          reason == "max_evals"   ? "max_evals reached" : reason
    println(stdout, "stopping early: $why at step $step/$n_steps " *
                    "($n_draws draws kept)")
    flush(stdout)
    return nothing
end

"The `stop` record a sampler result carries."
_stop_record(reason::AbstractString, step::Int, n_steps::Int, n_draws::Int,
             sc::StopControl) =
    (; stopped = String(reason), step, n_steps, n_draws,
       stop_file = sc.stop_file)

"""
Let the threads of an interrupted `Threads.@threads` half-step finish the walker
they are on and take no more, so nothing is still writing the ensemble while the
run is wound up. `@threads` leaves them running when the waiting task is
interrupted.
"""
function _drain_workers!(next_task)
    next_task[] = 1 << 40
    sleep(0.5)
    return nothing
end

"Names of the engines whose sampler declares keyword `kw`."
_engines_with(kw::Symbol) = sort!(String[n for (n, f) in ENGINES
    if any(m -> kw in Base.kwarg_decl(m), methods(f))])

"Add a result's `stop` record to a summary (no-op when the engine has none)."
function _stop_to_summary!(summary::AbstractDict, result)
    hasproperty(result, :stop) || return summary
    s = getproperty(result, :stop)
    summary["stopped"] = s.stopped
    summary["stopped_at_step"] = s.step
    summary["n_steps"] = s.n_steps
    summary["n_draws_kept"] = s.n_draws
    s.stop_file === nothing || (summary["stop_file"] = s.stop_file)
    return summary
end

# A function run at the end of every step with the step number and the run's
# `StopControl`; `nothing` outside tests. It exists so a test can deliver an
# interrupt, create the stop file or move the clock at an exact step instead of
# racing the sampler with a timer.
const _STEP_HOOK = Ref{Union{Nothing,Function}}(nothing)
_step_hook!(step::Int, sc::StopControl) =
    (h = _STEP_HOOK[]; h === nothing || h(step, sc); nothing)
