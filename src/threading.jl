# Thread-slot helpers — how scratch buffers get indexed in this package.
#
# `Threads.threadid()` is a GLOBAL thread id spanning every threadpool, while
# `Threads.nthreads()` reports the size of the *default* pool only. Julia ≥1.12
# starts one interactive thread by default AND numbers it first, so on a stock
# `julia` with no `-t` flag at all:
#
#     Threads.nthreads()                      # 1   (default pool)
#     Threads.@threads :static for ...        # runs on tid 2
#
# Any buffer built as `[T() for _ in 1:Threads.nthreads()]` and indexed by
# `Threads.threadid()` is then read out of bounds on the very first iteration.
# That is the `BoundsError: attempt to access 1-element Vector{Theta{Float64}}
# at index [2]` reported against `ofti_sample` on Julia 1.13. It is not an OFTI
# bug and not a thread-count bug: it fires with *one* worker thread, because
# the id and the count come from different namespaces.
#
# Two rules, in order of preference:
#
#  1. For a `Threads.@threads` loop we own, key the scratch by *chunk*: split
#     the index range with `_chunk_ranges` and let the chunk number index the
#     buffers. It is an ordinary loop counter — in bounds by construction,
#     independent of the thread count, and immune to the task migration that
#     forced `:static` on these loops in the first place.
#
#  2. Where the scheduling belongs to someone else (a callback handed to
#     NestedSamplers, say) the buffer has to be thread-keyed. Size it with
#     `_nthread_slots()`, never `Threads.nthreads()`.

"""
    _nthread_slots() -> Int

Upper bound on `Threads.threadid()` — the length a buffer must have to be
indexed by thread id. `Threads.nthreads()` is *not* that bound (it counts the
default pool; the id spans all pools).
"""
_nthread_slots() = Threads.maxthreadid()

"""
    _chunk_ranges(range_or_n, nchunks) -> Vector{UnitRange{Int}}

Split an index range into at most `nchunks` contiguous, non-empty chunks, sizes
differing by at most one. The returned count is `min(length(range), nchunks)`,
so a per-chunk buffer allocated with `nchunks` slots is always big enough and
chunk `c` is always a valid index into it.

Used to give each task of a `Threads.@threads` loop its own scratch without
touching `Threads.threadid()`:

    chunks = _chunk_ranges(n_tasks, n_slots)
    Threads.@threads for c in 1:length(chunks)
        buf = bufs[c]
        for i in chunks[c]
            ...
        end
    end
"""
function _chunk_ranges(rng::AbstractUnitRange{<:Integer}, nchunks::Integer)
    n = length(rng)
    n <= 0 && return UnitRange{Int}[]
    k = clamp(Int(nchunks), 1, n)
    base, rem = divrem(n, k)
    out   = Vector{UnitRange{Int}}(undef, k)
    start = Int(first(rng))
    @inbounds for c in 1:k
        len    = base + (c <= rem ? 1 : 0)
        out[c] = start:(start + len - 1)
        start += len
    end
    return out
end

_chunk_ranges(n::Integer, nchunks::Integer) = _chunk_ranges(1:Int(n), nchunks)

"""
    _nthread_chunks() -> Int

Default number of chunks for a threaded loop: one per worker thread in the
current pool, at least one. Buffer counts are sized from this, and the actual
chunk count from `_chunk_ranges` never exceeds it.
"""
_nthread_chunks() = max(1, Threads.nthreads())

const _SERIAL_INNER_KEY = :nereus_serial_inner_loops

"""
    _serial_inner_loops!()

Marks the current task as one of a sampler loop that already gives every thread
its own task, such as a `pt_emcee` half-step with at least `nthreads()` walker
updates. Likelihood loops written with `@_threads_unless_nested` then run
serially in this task. The mark is task-local: the caller's task, and the tasks
of a loop that leaves threads idle (`sample_pt`'s `@threads` over 10 chains on
90 threads), keep threading their inner loops.
"""
_serial_inner_loops!() = (task_local_storage(_SERIAL_INNER_KEY, true); nothing)

"""
    _inner_loops_serial() -> Bool

Whether `_serial_inner_loops!` marked the current task. Allocation-free, and
false for a task with no task-local storage at all.
"""
function _inner_loops_serial()
    s = current_task().storage
    return s isa IdDict{Any,Any} && get(s, _SERIAL_INNER_KEY, false) === true
end

"""
    @_threads_unless_nested for ... end

`Threads.@threads`, or the same loop run serially when the current task is
marked by `_serial_inner_loops!`. For loops inside a likelihood: a nested
`@threads` spawns one task per thread whatever its trip count (Julia 1.11
`Base.Threads.threading_run` spawns `threadpoolsize()` tasks per call). Under
`pt_emcee` on 90 threads the NGTS-33 photometry reduction, five 4096-point
chunks, spawned 90 tasks per likelihood call: 91 KB per call against 9.5 KB
serial, a stop-the-world GC every ~0.4 s, and each walker's call waiting on
chunks queued behind the other walkers. The loop body must be correct under
either schedule, as any `@threads` body already is. Both branches run the loop
in a closure: inlined into the NGTS-33 photometry likelihood, the plain loop
passed every per-point call boxed arguments (2.3 MB per likelihood call); inside
a closure, as `@threads` puts it, it allocates nothing per point.
"""
macro _threads_unless_nested(loop)
    (loop isa Expr && loop.head === :for) ||
        throw(ArgumentError("@_threads_unless_nested expects a `for` loop"))
    return esc(quote
        if $(GlobalRef(@__MODULE__, :_inner_loops_serial))()
            (() -> $loop)()
        else
            Base.Threads.@threads $loop
        end
    end)
end
