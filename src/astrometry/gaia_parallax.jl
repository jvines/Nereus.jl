# Runtime Gaia DR3 parallax lookup: the astrometric fits need the star's
# parallax and this fetches it, by Gaia DR3 source_id (ESA archive TAP) or by
# name (resolved to a source_id through the Simbad TAP service first).
#
# Both lookups are cached as small JSON files, so a rerun is offline and gives
# the same prior. The HTTP layer is a keyword (`http_get`) so tests can run
# without network.

using Downloads
using JSON3
using SHA

export fetch_gaia_parallax

const GAIA_TAP_URL   = "https://gea.esac.esa.int/tap-server/tap/sync"
# Simbad's TAP service, Strasbourg first and the Harvard mirror after it: the
# primary has been unreachable for long stretches, and the mirror answers the
# same ADQL.
const SIMBAD_TAP_URLS = ("https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
                         "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync")

# Percent-encode everything outside the RFC 3986 unreserved set.
function _urlencode(s::AbstractString)
    io = IOBuffer()
    for b in codeunits(s)
        c = Char(b)
        if isletter(c) && b < 0x80 || isdigit(c) || c in ('-', '_', '.', '~')
            write(io, b)
        else
            write(io, '%', uppercase(string(b; base = 16, pad = 2)))
        end
    end
    return String(take!(io))
end

"""
    _tap_get(base_url, adql; timeout = 60, retries = 4) -> String

Synchronous TAP query (GET, JSON output); the response body. A 5xx answer or a
transport failure is retried with a growing pause: the ESA archive rejects
synchronous jobs with a 503 whenever its queue is full, and asking again a few
seconds later is what it says to do. Other statuses fail at once.
"""
function _tap_get(base_url::AbstractString, adql::AbstractString;
                  timeout::Real = 120, retries::Integer = 6)
    url = string(base_url, "?REQUEST=doQuery&LANG=ADQL&FORMAT=json&QUERY=", _urlencode(adql))
    msg = ""
    for attempt in 1:(retries + 1)
        io = IOBuffer()
        resp = try
            Downloads.request(url; output = io, timeout = timeout,
                              headers = ["Accept" => "application/json"])
        catch err
            err
        end
        if resp isa Downloads.Response && 200 <= resp.status < 300
            return String(take!(io))
        end
        retryable = !(resp isa Downloads.Response) || resp.status >= 500
        msg = resp isa Downloads.Response ? "HTTP $(resp.status)" : sprint(showerror, resp)
        (retryable && attempt <= retries) || break
        sleep(5.0 * attempt)
    end
    error("TAP query to $base_url failed ($msg)")
end

"""Rows of a TAP JSON response (`{"metadata": [...], "data": [[...], ...]}`)."""
function _tap_rows(body::AbstractString)
    obj = try
        JSON3.read(body)
    catch err
        error("TAP response is not valid JSON ($(sprint(showerror, err)))")
    end
    haskey(obj, :data) || error("TAP response has no `data` member")
    return [collect(r) for r in obj[:data]]
end

"""
    _parse_gaia_id(x) -> Int

A Gaia DR3 source_id from an integer, a string of digits, or `"Gaia DR3 <id>"`.
"""
function _parse_gaia_id(x)
    if x isa Integer
        x > 0 || throw(ArgumentError("Gaia source_id must be positive, got $x"))
        return Int(x)
    elseif x isa AbstractString
        m = match(r"^\s*(?:Gaia\s+DR3\s+)?(\d+)\s*$"i, x)
        m === nothing && throw(ArgumentError(
            "cannot read a Gaia DR3 source_id from $(repr(x)); give an " *
            "integer, a string of digits, or \"Gaia DR3 <id>\""))
        return parse(Int, m.captures[1])
    end
    throw(ArgumentError("Gaia source_id must be an integer or string, got $(typeof(x))"))
end

_sql_quote(s::AbstractString) = replace(s, "'" => "''")

# The spellings Simbad stores a name under, from the ones people and catalogues
# use. Simbad's `ident.id` match is exact, so "HIP4618", "hip 103836",
# "HD114762" and "GJ 15 A" must be rewritten to "HIP 4618", "HIP 103836",
# "HD 114762" and "GJ 15A"; Bayer/Flamsteed names ("eps Eri") live under a
# "* " prefix. Tried in order, the name as given first.
function _simbad_name_candidates(name::AbstractString)
    n = String(strip(replace(name, r"\s+" => " ")))
    out = String[n]
    m = match(r"^([A-Za-z][A-Za-z]*)[\s_]*(\d.*)$", n)
    if m !== nothing
        pre, rest = uppercase(m.captures[1]), strip(m.captures[2])
        push!(out, "$pre $rest")
        c = match(r"^(\S*\d)\s+([A-Za-z]{1,2})$", rest)       # "GJ 15 A" -> "GJ 15A"
        c === nothing || push!(out, "$pre $(c.captures[1])$(uppercase(c.captures[2]))")
    else
        w = split(n)
        # Bayer / Flamsteed: Greek-letter abbreviation lower case, constellation capitalised
        if length(w) >= 2
            rest = titlecase.(lowercase.(w[2:end]))
            bayer = join([lowercase(w[1]); rest], " ")
            mixed = join([w[1]; rest], " ")          # "AU Mic"
            push!(out, "* " * n, "* " * bayer, "* " * mixed,
                  "V* " * n, "V* " * mixed)
        end
        push!(out, "NAME " * n)
    end
    return unique(out)
end

function _simbad_dr3_ids(cand::AbstractString, http_get, verbose)
    adql = "SELECT id2.id FROM ident AS id1 JOIN ident AS id2 ON id1.oidref = id2.oidref " *
           "WHERE id1.id = '$(_sql_quote(cand))' AND id2.id LIKE 'Gaia DR3 %'"
    last_err = nothing
    for url in SIMBAD_TAP_URLS
        try
            rows = _tap_rows(http_get(url, adql))
            ids = unique([String(r[1]) for r in rows if !isempty(r) && r[1] !== nothing])
            return filter(s -> occursin(r"^Gaia DR3 \d+$", s), ids), nothing
        catch err
            last_err = err
            verbose && @warn "Simbad query to $url failed: $(sprint(showerror, err))"
        end
    end
    return String[], last_err
end

function _gaia_name_to_id(name::AbstractString, cache_dir, force_refetch, http_get, verbose)
    key = bytes2hex(sha1("name=" * lowercase(strip(name))))[1:16]
    path = joinpath(cache_dir, "name_$(key).json")
    if isfile(path) && !force_refetch
        d = JSON3.read(read(path, String))
        verbose && @info "Gaia name cache hit: $name -> $(d[:source_id])"
        return Int(d[:source_id])
    end
    verbose && @info "Simbad: resolving $(repr(name)) to a Gaia DR3 source_id"
    ids = String[]
    last_err = nothing
    answered = false              # at least one candidate got a reply from Simbad
    for cand in _simbad_name_candidates(name)
        got, err = _simbad_dr3_ids(cand, http_get, verbose)
        err === nothing ? (answered = true) : (last_err = err)
        isempty(got) || (ids = got; break)
    end
    if isempty(ids) && !answered
        error("could not resolve $(repr(name)) through Simbad: $(sprint(showerror, last_err)). " *
              "Pass the Gaia DR3 source_id (star.gaia_id) or a prior on plx instead.")
    end
    isempty(ids) && error(
        "Simbad has no Gaia DR3 identifier for $(repr(name)) (name not found, or " *
        "the object has no DR3 counterpart). Pass the Gaia DR3 source_id " *
        "(star.gaia_id) or a prior on plx instead.")
    length(ids) > 1 && error(
        "$(repr(name)) matches several Gaia DR3 sources in Simbad ($(join(ids, ", "))); " *
        "pass the source_id you mean as star.gaia_id.")
    sid = _parse_gaia_id(only(ids))
    write(path, JSON3.write((; name = String(name), source_id = sid)))
    return sid
end

"""
    fetch_gaia_parallax(source_id::Integer; ...) -> NamedTuple
    fetch_gaia_parallax(name::AbstractString; ...) -> NamedTuple

Gaia DR3 parallax of a star, for the astrometric fits' default `plx` prior.
Returns `(source_id, plx, plx_err, release = "DR3", name)` with `plx` and
`plx_err` in mas; `name` is the name the lookup started from (`nothing` when
called with an id).

By id, the ESA Gaia archive's TAP service is queried for
`gaiadr3.gaia_source`. By name, Simbad's TAP service first resolves the name
to its `Gaia DR3 <id>` identifier, then the id is queried as above. An error
says so if the name does not resolve or has no Gaia DR3 identifier, and if the
source has no parallax (2-parameter solutions carry none).

The value is the raw catalogue parallax: no zero-point correction is applied.

Results are cached as JSON in `cache_dir` (`nereus_cache_dir("gaia")`), one
file per source_id and one per name, so reruns are offline;
`force_refetch = true` bypasses the cache. `http_get(url, adql) -> String` is
the transport (a TAP GET returning the JSON body); replace it to run offline.
"""
function fetch_gaia_parallax(source_id::Integer;
                             cache_dir::AbstractString = nereus_cache_dir("gaia"),
                             force_refetch::Bool = false,
                             verbose::Bool = true,
                             http_get = _tap_get,
                             _name::Union{Nothing, AbstractString} = nothing)
    sid = _parse_gaia_id(source_id)
    mkpath(cache_dir)
    path = joinpath(cache_dir, "dr3_$(sid).json")
    nm = _name === nothing ? nothing : String(_name)
    if isfile(path) && !force_refetch
        d = JSON3.read(read(path, String))
        verbose && @info "Gaia parallax cache hit: DR3 $sid"
        return (; source_id = sid, plx = Float64(d[:plx]), plx_err = Float64(d[:plx_err]),
                release = "DR3", name = nm)
    end
    adql = "SELECT source_id, parallax, parallax_error FROM gaiadr3.gaia_source " *
           "WHERE source_id = $sid"
    verbose && @info "Gaia archive: querying DR3 $sid"
    rows = try
        _tap_rows(http_get(GAIA_TAP_URL, adql))
    catch err
        error("Gaia archive query for DR3 $sid failed: $(sprint(showerror, err)). " *
              "Pass a prior on plx instead.")
    end
    isempty(rows) && error("Gaia DR3 source $sid not found in gaiadr3.gaia_source")
    r = rows[1]
    plx = r[2] === nothing ? NaN : Float64(r[2])
    err = r[3] === nothing ? NaN : Float64(r[3])
    (isfinite(plx) && isfinite(err) && err > 0) || error(
        "Gaia DR3 source $sid has no usable parallax (parallax = $(r[2]), " *
        "parallax_error = $(r[3])); a 2-parameter solution carries none. " *
        "Pass a prior on plx instead.")
    write(path, JSON3.write((; source_id = sid, plx = plx, plx_err = err, release = "DR3")))
    return (; source_id = sid, plx = plx, plx_err = err, release = "DR3", name = nm)
end

function fetch_gaia_parallax(name::AbstractString;
                             cache_dir::AbstractString = nereus_cache_dir("gaia"),
                             force_refetch::Bool = false,
                             verbose::Bool = true,
                             http_get = _tap_get)
    # A string that is already a source_id needs no name resolution.
    if occursin(r"^\s*(?:Gaia\s+DR3\s+)?\d+\s*$"i, name)
        return fetch_gaia_parallax(_parse_gaia_id(name); cache_dir, force_refetch,
                                   verbose, http_get)
    end
    mkpath(cache_dir)
    sid = _gaia_name_to_id(name, cache_dir, force_refetch, http_get, verbose)
    return fetch_gaia_parallax(sid; cache_dir, force_refetch, verbose, http_get,
                               _name = String(strip(name)))
end

# What `star.gaia_id` / `star.name` ask for, resolved to a lookup result with
# `resolved_from` ("gaia_id" or "name") added; `nothing` when neither is given.
# `gaia_id` wins when both are present.
function _gaia_lookup_from_star(star; http_get = _tap_get, kw...)
    if star.gaia_id !== nothing
        r = fetch_gaia_parallax(_parse_gaia_id(star.gaia_id); http_get, kw...)
        return merge(r, (; resolved_from = "gaia_id"))
    elseif star.name !== nothing
        r = fetch_gaia_parallax(String(star.name); http_get, kw...)
        return merge(r, (; resolved_from = "name"))
    end
    return nothing
end

# The default `plx` prior a Gaia lookup stands for: the same ±10σ truncated
# Gaussian `default_priors` builds from a catalogue parallax.
function _gaia_plx_prior(g)
    σ = max(g.plx_err, 1e-3)
    return NormalPrior(g.plx, σ, max(0.001, g.plx - 10σ), g.plx + 10σ)
end

# What a fit's summary records about the lookup behind its parallax prior.
_gaia_summary(g) = Dict{String, Any}(
    "source_id" => g.source_id, "plx" => g.plx, "plx_err" => g.plx_err,
    "release" => g.release, "resolved_from" => g.resolved_from,
    "name" => g.name)
