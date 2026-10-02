# Sampler checkpoints: a run's whole state on disk, so a killed run, or a
# finished one that has not mixed, can be continued instead of started over.
#
# A sampler that supports this takes `checkpoint` (a path), `checkpoint_interval`
# (seconds) and `resume`. It writes `write_checkpoint` during the run and at its
# end, and with `resume = true` restores from `read_checkpoint` instead of
# initialising. What goes in the state is the sampler's business; the rule is
# that the continuation is bit-identical to an uninterrupted run, so everything
# the step loop reads or writes is saved: positions, RNG streams, adaptation
# state, counters, histories and the draws kept so far.
#
# The fingerprint guards against continuing the wrong run. It carries what fixes
# the target (parameter names, priors, data) plus every sampler setting the
# chain depends on; `read_checkpoint` refuses a mismatch and names the fields.
# Settings that only change how long the run is (`n_steps` and the like) are
# deliberately left out: extending a run is the point.

using Serialization

const _CHECKPOINT_FORMAT = 1

"""
    run_fingerprint(params, data; settings...) -> NamedTuple

What a checkpoint must match before a run may continue from it: the unfrozen
parameter names, the priors, the rest of the model (`params.config`: external
priors, stability, instruments, sharing, ...), all of the data, the two
process-wide informed-birth switches, and the sampler `settings` passed as
keywords. Priors and model are hashed separately only so a refusal can say which.
"""
function run_fingerprint(params::Params, data::Data; settings...)
    return (; names = copy(params.layout.unfrozen_names),
              priors = content_hash(params.config.priors),
              model = content_hash(params.config), data = content_hash(data),
              informed_birth = (GP_INFORMED_BIRTH[], AD_INFORMED_BIRTH[]),
              settings...)
end

"""
    content_hash(x) -> UInt

A hash of `x` by content, the same in every process: numbers, strings and
symbols by value, arrays and tuples every element (not a sample, as Base.hash
takes of a long array), dictionaries and sets
in sorted order, any other struct field by field under its bare type name.
Not `repr`, whose module prefixes depend on what the caller imported (`using
Nereus` prints `ActivityDecorrelation(...)`, `import Nereus` prints
`Nereus.ActivityDecorrelation(...)`), and not the default `hash`, which falls
back to `objectid` for mutable structs. Functions count by name only.
"""
content_hash(x) = _content_hash(x, zero(UInt), IdDict{Any,Nothing}())

function _content_hash(x, h::UInt, seen::IdDict{Any,Nothing})
    x isa Union{Number, AbstractString, AbstractChar, Nothing, Missing} && return hash(x, h)
    x isa Symbol && return hash(String(x), hash("Symbol", h))
    x isa Enum && return hash(Int(x), hash(string(nameof(typeof(x))), h))
    x isa Type && return hash(x isa DataType ? string(nameof(x)) : string(x), h)
    x isa Function && return hash(string(nameof(x)), h)
    if x isa AbstractArray{<:Union{Number, Bool}}
        # Every element: Base.hash reads only a sample of an array of 8192 or more,
        # so one changed point of a long light curve would leave it unchanged.
        h = hash(size(x), h)
        for v in x
            h = hash(v, h)
        end
        return h
    end
    if ismutable(x)
        haskey(seen, x) && return hash(:cycle, h)
        seen[x] = nothing
    end
    h = hash(string(nameof(typeof(x))), h)
    if x isa AbstractDict
        for k in sort!(collect(keys(x)); by = string)
            h = _content_hash(x[k], _content_hash(k, h, seen), seen)
        end
    elseif x isa AbstractSet
        for e in sort!([_content_hash(e, zero(UInt), seen) for e in x])
            h = hash(e, h)
        end
    elseif x isa Union{AbstractArray, Tuple}
        h = hash(x isa Tuple ? length(x) : size(x), h)
        for e in x
            h = _content_hash(e, h, seen)
        end
    elseif x isa NamedTuple
        for (k, v) in pairs(x)
            h = _content_hash(v, hash(k, h), seen)
        end
    elseif fieldcount(typeof(x)) == 0
        h = hash(string(x), h)
    else
        for f in fieldnames(typeof(x))
            isdefined(x, f) || continue
            h = _content_hash(getfield(x, f), hash(f, h), seen)
        end
    end
    return h
end

"""
    write_checkpoint(path, sampler, fingerprint, state::NamedTuple)

Write `state` to `path`, replacing any previous checkpoint atomically: the new
file is written beside it and renamed over it, so a run killed mid-write leaves
the last complete checkpoint in place.
"""
function write_checkpoint(path::AbstractString, sampler::AbstractString,
                          fingerprint::NamedTuple, state::NamedTuple)
    mkpath(dirname(abspath(path)))
    tmp = string(path, ".tmp")
    open(tmp, "w") do io
        serialize(io, (format = _CHECKPOINT_FORMAT, sampler = String(sampler),
                       fingerprint = fingerprint, state = state))
    end
    mv(tmp, path; force = true)
    return path
end

"""
    read_checkpoint(path, sampler, fingerprint) -> NamedTuple

The `state` saved at `path` by `sampler`. Throws if there is no checkpoint, if
it was written by another sampler or format, or if its fingerprint differs from
`fingerprint`, listing the fields that differ.
"""
function read_checkpoint(path::AbstractString, sampler::AbstractString,
                         fingerprint::NamedTuple)
    isfile(path) || throw(ArgumentError(
        "resume = true, but there is no checkpoint at $path"))
    ck = open(deserialize, path)
    (ck isa NamedTuple && get(ck, :format, nothing) == _CHECKPOINT_FORMAT) ||
        throw(ArgumentError("$path is not a Nereus checkpoint of format " *
                            "$_CHECKPOINT_FORMAT"))
    ck.sampler == sampler || throw(ArgumentError(
        "$path was written by $(ck.sampler), not $sampler"))
    saved = ck.fingerprint
    diffs = String[string(k) for k in union(keys(fingerprint), keys(saved))
                   if !isequal(get(fingerprint, k, missing), get(saved, k, missing))]
    isempty(diffs) || throw(ArgumentError(
        "$path was written by a different run; these differ: " * join(diffs, ", ")))
    return ck.state
end
