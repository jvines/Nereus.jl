# Juliacall compatibility helpers.
#
# When Nereus is loaded into a Python process via juliacall (every
# exoautomata worker today, before the subprocess+sysimage migration),
# concurrent Julia code from Python is explicitly unsupported by
# PythonCall:
#
#   https://juliapy.github.io/PythonCall.jl/stable/faq/#Is-PythonCall/JuliaCall-thread-safe?
#
# We use this detection helper in *exactly one place* — the watchdog
# `Threads.@spawn` in `runner.jl::_run_sampler_with_timeout`. Under
# juliacall, that @spawn + sleep-poll pattern deadlocks (the spawned
# task never gets scheduled, main task loops `sleep(1.0)` forever).
# The fix is to skip the watchdog and run synchronously, letting
# Python-side timeout enforcement (subprocess.kill / signal.alarm)
# tear down the worker if needed.
#
# Note: we previously also had a `@maybe_threaded` macro that ran
# `Threads.@threads` loops serially under juliacall. It was removed:
# silent serial execution masked the real performance regression
# and didn't fix the underlying architecture problem. The real
# answer is subprocess + sysimage isolation for the worker path
# (full Julia threading, no GIL), with this detection helper only
# guarding the timeout watchdog. Don't bring `@maybe_threaded` back.

"""
    _is_under_juliacall() -> Bool

True when Nereus is loaded into a Python process via juliacall.
Detection priority:

  1. `PYTHON_JULIACALL_HANDLE_SIGNALS` env var — set unconditionally
     by `nereus-py`'s `__init__.py` before importing juliacall.
  2. `JULIA_PYTHONCALL_LIBPTR` — set by juliacall itself when it
     loads Julia.
  3. `PythonCall` module loaded — backup for entry points that
     bypass (1) and (2).

Recomputed on each call (~ns) so env-var flips in tests / harnesses
work.
"""
function _is_under_juliacall()
    haskey(ENV, "PYTHON_JULIACALL_HANDLE_SIGNALS") && return true
    haskey(ENV, "JULIA_PYTHONCALL_LIBPTR")       && return true
    pkg_id = Base.PkgId(Base.UUID("6099a3de-0909-46bc-b1f4-468b9a2dfc0d"),
                         "PythonCall")
    return haskey(Base.loaded_modules, pkg_id)
end
