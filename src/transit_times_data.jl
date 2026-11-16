# Externally measured mid-transit times (`data.transit_times`).
#
# One `TransitTimeSet` per planet: integer epochs, mid-transit times in BJD_TDB,
# their 1σ errors and a provenance string per point. The epochs are labels:
# cycle counts from an epoch_zero someone else chose. The likelihood
# (src/transit_times_likelihood.jl) reads only their differences, never the
# reference ephemeris that produced them.

"""
    TransitTimeSet(; planet, epoch, tc, tc_err, source = nothing, use = nothing,
                   name = "", time_system = "BJD_TDB", reference = nothing,
                   exclude_sources = String[], jitter = false)

Measured mid-transit times of planet `planet` (1-based index into the model's
planets), passed to `Data(; transit_times = [...])`.

- `epoch` -- integer cycle numbers. Only their differences are used: the
  likelihood numbers transits from the model's own transit time and checks the
  labels against it.
- `tc` -- mid-transit times, BJD_TDB (days). `tc_err` -- their 1σ errors (days),
  finite and > 0.
- `source` -- one provenance string per point, carried to the outputs.
- `use` -- per-point mask; `false` keeps the point out of the likelihood (it is
  still reported, with its model time and residual).
- `exclude_sources` -- sources to leave out: an entry drops a point whose source
  equals it, or starts with it followed by `:` (`"EXOCLOCK"` drops every
  `"EXOCLOCK:..."` point).
- `time_system` -- must be `"BJD_TDB"`.
- `reference` -- the ephemeris the epochs were counted with,
  `(t0, period, epoch_zero)`, or `nothing`. Kept for provenance; the
  likelihood never reads it.
- `jitter` -- give the planet a `sigma_tt_k<planet>` jitter (days), added in
  quadrature to `tc_err`. Off by default.

Throws an `ArgumentError` for a time system other than BJD_TDB, arrays of
different lengths, a non-positive or non-finite `tc_err`, a non-finite `tc`, a
non-integer epoch, or epochs that disagree with the times by more than a
quarter period (a mislabelled point).
"""
struct TransitTimeSet
    planet::Int
    name::String
    epoch::Vector{Int}
    tc::Vector{Float64}
    tc_err::Vector{Float64}
    source::Vector{String}
    # "" for a point in the likelihood, else why it is not: "use" (the
    # per-point mask), "exclude_sources", or "photometry_overlap" (its transit
    # is in the fitted photometry; see `_drop_photometry_overlap`).
    excluded::Vector{String}
    used::Vector{Int}       # indices of the points in the likelihood
    reference::Union{Nothing, NamedTuple{(:t0, :period, :epoch_zero),
                                         Tuple{Float64, Float64, Int}}}
    jitter::Bool
end

# An epoch label as an Int: JSON may deliver 12.0 for 12.
function _tt_epoch_int(x, planet)
    (x isa Real && isfinite(x) && isinteger(x)) || throw(ArgumentError(
        "transit times of planet $planet: epoch $x is not an integer. Epochs are " *
        "cycle numbers; give them as integers."))
    return Int(x)
end

"""
    _tt_source_excluded(source, exclude_sources) -> Bool

Whether `source` is dropped by `exclude_sources`: an entry matches the source
itself, or any source that starts with the entry followed by `:`.
"""
_tt_source_excluded(source::AbstractString, exclude_sources) =
    any(e -> source == e || startswith(source, e * ":"), exclude_sources)

function TransitTimeSet(; planet::Integer, epoch, tc, tc_err, source = nothing,
                        use = nothing, name::AbstractString = "",
                        time_system::AbstractString = "BJD_TDB", reference = nothing,
                        exclude_sources = String[], jitter::Bool = false,
                        excluded = nothing)
    p = Int(planet)
    p >= 1 || throw(ArgumentError(
        "transit times: planet index $p is not a planet; it is 1-based (1 is the " *
        "first entry of the model's planets)"))
    time_system == "BJD_TDB" || throw(ArgumentError(
        "transit times of planet $p are in time_system \"$time_system\"; only " *
        "\"BJD_TDB\" is accepted. Convert them to BJD_TDB, the time system of the " *
        "photometry and the RVs."))
    n = length(tc)
    n > 0 || throw(ArgumentError("transit times of planet $p: no points. Drop the block."))
    lens = ["epoch" => length(epoch), "tc" => n, "tc_err" => length(tc_err)]
    source === nothing || push!(lens, "source" => length(source))
    use === nothing || push!(lens, "use" => length(use))
    all(l -> last(l) == n, lens) || throw(ArgumentError(
        "transit times of planet $p: the value arrays have different lengths (" *
        join(("$k = $v" for (k, v) in lens), ", ") * "); give one entry per transit " *
        "in each."))
    ep = Int[_tt_epoch_int(x, p) for x in epoch]
    t = Float64.(collect(tc))
    σ = Float64.(collect(tc_err))
    bad_t = findall(!isfinite, t)
    isempty(bad_t) || throw(ArgumentError(
        "transit times of planet $p: tc is not finite at epoch(s) $(ep[bad_t]). Fix " *
        "or remove those points."))
    bad = findall(s -> !(isfinite(s) && s > 0), σ)
    isempty(bad) || throw(ArgumentError(
        "transit times of planet $p: tc_err must be finite and > 0; it is not at " *
        "epoch(s) $(ep[bad]) ($(σ[bad])). Fix or remove those points."))
    src = source === nothing ? fill("", n) : String[string(s) for s in source]
    ref = if reference === nothing
        nothing
    else
        r = reference
        t0 = Float64(r isa NamedTuple ? r.t0 : r["t0"])
        per = Float64(r isa NamedTuple ? r.period : r["period"])
        ez = r isa NamedTuple ? (hasproperty(r, :epoch_zero) ? r.epoch_zero : 0) :
                                get(r, "epoch_zero", 0)
        (isfinite(t0) && isfinite(per) && per > 0) || throw(ArgumentError(
            "transit times of planet $p: reference_ephemeris needs a finite t0 and a " *
            "period > 0; got t0 = $t0, period = $per. Fix it or set it to null."))
        (t0 = t0, period = per, epoch_zero = _tt_epoch_int(ez, p))
    end
    exc = excluded === nothing ? fill("", n) : String[string(x) for x in excluded]
    length(exc) == n || throw(ArgumentError("excluded must have one entry per point"))
    ex_src = String[string(s) for s in exclude_sources]
    for i in 1:n
        isempty(exc[i]) || continue
        if use !== nothing && !Bool(use[i])
            exc[i] = "use"
        elseif !isempty(ex_src) && _tt_source_excluded(src[i], ex_src)
            exc[i] = "exclude_sources"
        end
    end
    used = findall(isempty, exc)
    _tt_check_labels(p, ep, t, σ, src, used)
    return TransitTimeSet(p, String(name), ep, t, σ, src, exc, used, ref, jitter)
end

"""
    n_tt(set::TransitTimeSet) -> Int

Number of transit times of `set` in the likelihood (excluded points left out).
"""
n_tt(s::TransitTimeSet) = length(s.used)

"""
    _tt_linear_fit(epoch, tc, tc_err) -> NamedTuple or nothing

Weighted (1/σ²) least-squares ephemeris `tc = t0 + P·epoch` through the points:
`(t0, P, P_err, rms, span)`, with `rms` the unweighted RMS of the residuals and
`span` the range of the epochs. `nothing` with fewer than two distinct epochs.
"""
function _tt_linear_fit(epoch::AbstractVector{<:Integer}, tc::AbstractVector{<:Real},
                        tc_err::AbstractVector{<:Real})
    length(unique(epoch)) >= 2 || return nothing
    w = 1 ./ Float64.(tc_err) .^ 2
    x = Float64.(epoch)
    # Offsets keep BJD digits: the fit is of tc - tc[1] against epoch - epoch[1].
    x0, y0 = x[1], Float64(tc[1])
    xs, ys = x .- x0, Float64.(tc) .- y0
    S, Sx, Sy = sum(w), sum(w .* xs), sum(w .* ys)
    Sxx, Sxy = sum(w .* xs .^ 2), sum(w .* xs .* ys)
    D = S * Sxx - Sx^2
    P = (S * Sxy - Sx * Sy) / D
    a = (Sy - P * Sx) / S                 # tc at epoch x0, minus y0
    res = ys .- (a .+ P .* xs)
    return (t0 = y0 + a - P * x0, P = P, P_err = sqrt(S / D),
            rms = sqrt(sum(abs2, res) / length(res)),
            span = maximum(epoch) - minimum(epoch), residuals = res)
end

_tt_linear_fit(s::TransitTimeSet) =
    _tt_linear_fit(s.epoch[s.used], s.tc[s.used], s.tc_err[s.used])

# The epochs must agree with the times: the likelihood refuses any ephemeris
# more than a quarter period from a point (src/transit_times_likelihood.jl), so a
# mislabelled point would leave no ephemeris it accepts and the fit would have
# nowhere to start. Checked against the least-squares ephemeris of the points in
# the likelihood; needs three distinct epochs (two always fit exactly).
function _tt_check_labels(p, ep, t, σ, src, used)
    length(unique(ep[used])) >= 3 || return nothing
    fit = _tt_linear_fit(ep[used], t[used], σ[used])
    P = abs(fit.P)
    j = argmax(abs.(fit.residuals))
    r = fit.residuals[j]
    abs(r) <= P / 4 && return nothing
    i = used[j]
    throw(ArgumentError(
        "transit times of planet $p: the epochs disagree with the times. The point " *
        "at epoch $(ep[i])" * (isempty(src[i]) ? "" : " (source $(src[i]))") *
        " is $(round(r; sigdigits = 4)) d, $(round(r / P; sigdigits = 3)) periods, " *
        "from the least-squares ephemeris through the points (P = " *
        "$(round(P; sigdigits = 10)) d), and the likelihood accepts no ephemeris more " *
        "than a quarter period from a point. Check that point's epoch label, or set " *
        "its `use` to false."))
end

# The same set with the points flagged in `drop` (and in the likelihood) taken
# out for `reason`.
function _tt_exclude(s::TransitTimeSet, drop::AbstractVector{Bool}, reason::String)
    exc = copy(s.excluded)
    for i in eachindex(exc)
        (drop[i] && isempty(exc[i])) && (exc[i] = reason)
    end
    return TransitTimeSet(s.planet, s.name, s.epoch, s.tc, s.tc_err, s.source, exc,
                          findall(isempty, exc), s.reference, s.jitter)
end

"""
    _drop_photometry_overlap(set, t_phot) -> (set, n_dropped)

Take out of the likelihood every transit time whose transit is in the fitted
photometry, which already constrains it: counting both would count that
transit twice. A transit is in the photometry when a cadence lies within
`h = min(0.5 d, P/10)` of its mid-time, `P` from the least-squares ephemeris of
the set (one transit time: no estimate, `h = 0.5 d`). Only called for a planet
whose transit the photometry models (`:PM` in its mode).
"""
function _drop_photometry_overlap(s::TransitTimeSet, t_phot::AbstractVector{<:Real})
    isempty(t_phot) && return s, 0
    fit = _tt_linear_fit(s)
    h = fit === nothing ? 0.5 : min(0.5, abs(fit.P) / 10)
    ts = sort(Float64.(t_phot))
    drop = falses(length(s.tc))
    for i in s.used
        j = searchsortedfirst(ts, s.tc[i] - h)
        drop[i] = j <= length(ts) && ts[j] <= s.tc[i] + h
    end
    n = count(drop)
    return (n == 0 ? s : _tt_exclude(s, drop, "photometry_overlap")), n
end
