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
