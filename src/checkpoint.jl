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
parameter names, a hash of every prior, a hash of every data array, and the
sampler `settings` passed as keywords.
"""
function run_fingerprint(params::Params, data::Data; settings...)
    pri = sort!([(String(k), prior_to_dict(v)) for (k, v) in params.config.priors];
                by = first)
    dh = zero(UInt)
    for f in fieldnames(typeof(data))
        v = getfield(data, f)
        if v isa AbstractArray{<:Number} || v isa AbstractDict
            dh = hash(v, hash(f, dh))
        end
    end
    return (; names = copy(params.layout.unfrozen_names), priors = hash(pri),
              data = dh, settings...)
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
