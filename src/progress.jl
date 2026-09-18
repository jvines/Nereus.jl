# Progress reporting — single-line, in-place updates inspired by dynesty.
#
# Usage:
#   pb = ProgressBar("PT trans-dim"; total = n_total)
#   for i in 1:n_total
#       # ... sampler work ...
#       update!(pb; n_done = i,
#                fields = (:logZ => logZ, :acc => acc_rate))
#   end
#   finish!(pb)
#
# Output style (TTY):
#   PT trans-dim ▕████████░░░░░░░░░░░░▏ 4096/16384 (25.0%) | logZ=-1234.5 | acc=0.65 | ETA 1m45s
#
# Off-TTY (log file): each `update!` is throttled to once per
# `print_every` seconds and emitted as a full newline-terminated line so
# log readers stay sane.

using Printf

mutable struct ProgressBar
    label::String
    total::Int                # 0 means "unknown total"
    n_done::Int
    t_start::Float64
    t_last_print::Float64
    print_every::Float64       # seconds between TTY refreshes
    width::Int                 # bar character width
    use_tty::Bool
    finished::Bool
    enabled::Bool
end

"""
    ProgressBar(label; total=0, print_every=0.5, width=24, enabled=true)

Construct a single-line progress bar. `total=0` switches to an
indeterminate format (just iter count + elapsed + extra fields). When
stdout is not a TTY, output is throttled to once per `print_every`
seconds and emitted as full lines instead of carriage-return refreshes,
so it survives in log files. Pass `enabled=false` to make all
`update!` / `finish!` calls no-ops.
"""
function ProgressBar(label::AbstractString;
                       total::Integer = 0,
                       print_every::Real = 0.5,
                       width::Integer = 24,
                       enabled::Bool = true)
    use_tty = enabled && _stdout_is_tty()
    now = time()
    return ProgressBar(String(label), Int(total), 0, now, 0.0,
                        Float64(print_every), Int(width), use_tty,
                        false, enabled)
end

# Stdout-is-TTY check that handles both terminal and pipe-to-file.
_stdout_is_tty() = try
    isa(stdout, Base.TTY) ||
        (hasmethod(Base.isopen, Tuple{typeof(stdout)}) &&
         hasmethod(Base.isreadable, Tuple{typeof(stdout)}) &&
         Base.isopen(stdout) && Base.isatty(stdout))
catch
    false
end

"""
    update!(pb; n_done=pb.n_done+1, fields=())

Advance the bar to `n_done` iters and refresh the display. `fields` is
an iterable of `Symbol => value` pairs that get formatted into the
status line (numbers get `%g`-style formatting; strings pass through).
"""
function update!(pb::ProgressBar;
                  n_done::Integer = pb.n_done + 1,
                  fields = ())
    pb.enabled || return pb
    pb.n_done = Int(n_done)
    now = time()
    # Throttle except on the final tick
    if (now - pb.t_last_print < pb.print_every) &&
       !(pb.total > 0 && pb.n_done >= pb.total)
        return pb
    end
    pb.t_last_print = now
    line = _render(pb, fields)
    if pb.use_tty
        # \r returns to start, \033[K clears to end of line, then we
        # re-print without trailing newline so the next update overwrites
        # cleanly.
        print(stdout, "\r\033[K", line)
        flush(stdout)
    else
        # Off-TTY (log file): each update is a full line so it's grep-able.
        println(stdout, line)
        flush(stdout)
    end
    return pb
end

"""
    finish!(pb; final="")

Terminate the bar: emit a newline (so the next message starts on a
fresh row) and optionally print `final` underneath.
"""
function finish!(pb::ProgressBar; final::AbstractString = "")
    pb.enabled || return pb
    pb.finished && return pb
    pb.finished = true
    if pb.use_tty
        println(stdout)  # close the carriage-return line
    end
    if !isempty(final)
        println(stdout, final)
    end
    flush(stdout)
    return pb
end

# --- internals --------------------------------------------------------

function _render(pb::ProgressBar, fields)
    elapsed = time() - pb.t_start
    parts = String[]
    push!(parts, pb.label)

    if pb.total > 0
        frac = clamp(pb.n_done / pb.total, 0.0, 1.0)
        push!(parts, _bar(frac, pb.width))
        push!(parts, @sprintf("%d/%d (%.1f%%)",
                              pb.n_done, pb.total, 100 * frac))
    else
        push!(parts, @sprintf("iter %d", pb.n_done))
    end

    for (k, v) in fields
        push!(parts, @sprintf("%s=%s", String(k), _fmt(v)))
    end

    push!(parts, @sprintf("elapsed %s", _time_fmt(elapsed)))

    if pb.total > 0 && pb.n_done > 0 && pb.n_done < pb.total
        rate = pb.n_done / max(elapsed, 1e-9)
        eta = (pb.total - pb.n_done) / rate
        push!(parts, @sprintf("ETA %s", _time_fmt(eta)))
    end

    return join(parts, " | ")
end

function _bar(frac::Float64, width::Int)
    filled = round(Int, frac * width)
    filled = clamp(filled, 0, width)
    return "▕" * "█"^filled * "░"^(width - filled) * "▏"
end

function _time_fmt(s::Real)
    s < 0 && (s = 0)
    if s < 60
        return @sprintf("%.0fs", s)
    elseif s < 3600
        return @sprintf("%dm%02ds", floor(Int, s / 60),
                         round(Int, mod(s, 60)))
    else
        return @sprintf("%dh%02dm", floor(Int, s / 3600),
                         floor(Int, mod(s, 3600) / 60))
    end
end

_fmt(x::Integer) = string(x)
_fmt(x::AbstractString) = String(x)
function _fmt(x::Real)
    isnan(x) && return "NaN"
    isinf(x) && return x > 0 ? "Inf" : "-Inf"
    ax = abs(x)
    if ax == 0.0
        return "0"
    elseif ax < 1e-3 || ax >= 1e5
        return @sprintf("%.3g", x)
    elseif ax < 1
        return @sprintf("%.4f", x)
    else
        return @sprintf("%.3f", x)
    end
end
_fmt(x) = string(x)
