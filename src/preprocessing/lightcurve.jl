# lightcurve.jl — canonical LightCurve container + clean_lightcurve one-call API.
#
# One type, one call: noisy photometry in (one sector or many), detrended
# photometry out, sector identity preserved. clean_lightcurve picks the
# detrender (default :notch) and forwards per-cadence sector_id so the
# underlying detrenders stay per-orbit/per-segment internally.
#
# Included into the Nereus module after preprocessing/locor.jl so all the
# detrend_* functions are in scope.

"""
    LightCurve

Canonical photometry container. One `LightCurve` holds 1+ sectors; cadences are
concatenated and time-ordered, sectors tracked per-cadence by `sector_id`.

Fields:
- `t`, `flux`, `flux_err` — per-cadence arrays (Float64).
- `sector_id` — per-cadence sector tag (Int).
- `flux_detrended`, `trend`, `is_transit`, `method` — `nothing` before
  [`clean_lightcurve`](@ref); filled afterwards. `is_transit[i]` marks cadences
  the notch detrender favors as transit-bearing (`delta_bic > 0`).

Construct with `LightCurve(t, flux, flux_err)` (single sector), with explicit
tags `LightCurve(t, flux, flux_err, sector_id)`, or from a vector of sectors
`LightCurve([lc1, lc2, ...])` where vector position = sector number.
"""
struct LightCurve
    t              ::Vector{Float64}
    flux           ::Vector{Float64}
    flux_err       ::Vector{Float64}
    sector_id      ::Vector{Int}
    flux_detrended ::Union{Nothing,Vector{Float64}}
    trend          ::Union{Nothing,Vector{Float64}}
    is_transit     ::Union{Nothing,Vector{Bool}}
    method         ::Union{Nothing,Symbol}

    # All-field inner constructor with validation.
    function LightCurve(t, flux, flux_err, sector_id,
                        flux_detrended, trend, is_transit, method)
        tt  = Vector{Float64}(t)
        ff  = Vector{Float64}(flux)
        fe  = Vector{Float64}(flux_err)
        sid = Vector{Int}(sector_id)
        n = length(tt)
        (length(ff) == n && length(fe) == n && length(sid) == n) ||
            throw(ArgumentError(
                "t/flux/flux_err/sector_id length mismatch: " *
                "$(n)/$(length(ff))/$(length(fe))/$(length(sid))"))
        all(>(0), fe) || throw(ArgumentError("all flux_err must be > 0"))
        return new(tt, ff, fe, sid, flux_detrended, trend, is_transit, method)
    end
end

# Single sector: sector_id .= 1, detrend fields nothing.
LightCurve(t, flux, flux_err) =
    LightCurve(t, flux, flux_err, ones(Int, length(t)), nothing, nothing, nothing, nothing)

# Explicit per-cadence sector tags, detrend fields nothing.
LightCurve(t, flux, flux_err, sector_id::AbstractVector{<:Integer}) =
    LightCurve(t, flux, flux_err, sector_id, nothing, nothing, nothing, nothing)

"""
    LightCurve(lcs::AbstractVector{LightCurve}) -> LightCurve

Concatenate multiple sectors. Vector position is authoritative: element `k`
becomes sector `k` for all its cadences, ignoring any pre-existing `sector_id`.
All cadences are then stably sorted by time, carrying `sector_id` along. The
result's detrend fields are `nothing`.
"""
function LightCurve(lcs::AbstractVector{LightCurve})
    isempty(lcs) && throw(ArgumentError("cannot build a LightCurve from an empty vector"))
    t   = Float64[]
    fl  = Float64[]
    fe  = Float64[]
    sid = Int[]
    for (k, lc) in enumerate(lcs)
        append!(t, lc.t)
        append!(fl, lc.flux)
        append!(fe, lc.flux_err)
        append!(sid, fill(k, length(lc.t)))
    end
    perm = sortperm(t)  # sortperm is stable by default
    return LightCurve(t[perm], fl[perm], fe[perm], sid[perm])
end

"""
    trim_lightcurve(lc::LightCurve, mask::AbstractVector{Bool}) -> LightCurve

Remove the cadences flagged `true` in `mask` (length == number of cadences)
**entirely**: the returned `LightCurve` carries no trace of them in ANY field —
raw `t`/`flux`/`flux_err`, `sector_id`, or the detrend arrays
(`flux_detrended`/`trend`/`is_transit`) when present. `mask[i] == true` ⇒ cadence
`i` is trimmed (dropped); `mask[i] == false` ⇒ kept.

This is a hard trim, not a flag: masked cadences will not appear in downstream
plots, periodograms, or saved files. Use it to drop bad/unwanted cadences (e.g.
flares, scattered-light ramps, momentum dumps) before detrending or saving.
"""
function trim_lightcurve(lc::LightCurve, mask::AbstractVector{Bool})
    length(mask) == length(lc.t) || throw(ArgumentError(
        "trim mask length $(length(mask)) != n cadences $(length(lc.t))"))
    keep = .!mask
    any(keep) || throw(ArgumentError("trim mask would remove ALL cadences"))
    sl(x) = x === nothing ? nothing : x[keep]
    return LightCurve(lc.t[keep], lc.flux[keep], lc.flux_err[keep], lc.sector_id[keep],
                      sl(lc.flux_detrended), sl(lc.trend), sl(lc.is_transit), lc.method)
end

# Robust running median over a centered cadence window (edge-truncated).
function _running_median(flux::AbstractVector{<:Real}, w::Int)
    n = length(flux); rm = Vector{Float64}(undef, n); h = w ÷ 2
    @inbounds for i in 1:n
        rm[i] = median(@view flux[max(1, i - h):min(n, i + h)])
    end
    return rm
end

"""
    sigma_clip_mask(t, flux; sigma=5.0, window=101, iters=5, sector_id=nothing,
                    gap_size=0.5, protect_mask=nothing) -> Vector{Bool}

Flag extreme point-outliers statistically (complementary to a time-based
`trim_mask`): within each gap-separated segment, sigma-clip the residuals of a
robust **running median** (window in cadences). `mask[i] == true` ⇒ cadence `i`
is an outlier. Robust scatter = 1.4826·MAD, re-estimated from un-flagged points
over up to `iters` passes. Cadences where `protect_mask` is `true` are **never**
flagged — pass the in-transit set here so deep transits aren't mistaken for
outliers. Asymmetric clipping is intentionally avoided; the running-median
baseline + transit protection keep real dips safe.
"""
function sigma_clip_mask(t::AbstractVector{<:Real}, flux::AbstractVector{<:Real};
                          sigma::Real=5.0, window::Integer=101, iters::Integer=5,
                          sector_id::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                          gap_size::Real=0.5,
                          protect_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    n = length(flux)
    length(t) == n || throw(ArgumentError("t and flux length mismatch"))
    outlier = falses(n)
    prot = protect_mask === nothing ? falses(n) : protect_mask
    length(prot) == n || throw(ArgumentError("protect_mask length != n cadences"))
    segs = find_segments(t; gap_size=gap_size)
    for seg in segs
        ns = length(seg)
        ns >= 5 || continue
        f  = @view flux[seg]
        pr = @view prot[seg]
        resid = f .- _running_median(f, min(Int(window), ns))
        flagged = falses(ns)
        for _ in 1:iters
            keepers = [(!flagged[k]) & (!pr[k]) for k in 1:ns]
            any(keepers) || break
            r0  = resid[keepers]
            med = median(r0)
            scat = 1.4826 * median(abs.(r0 .- med))
            scat > 0 || break
            newflag = [(!pr[k]) & (abs(resid[k] - med) > sigma * scat) for k in 1:ns]
            newflag == flagged && break
            flagged = newflag
        end
        @inbounds for (k, i) in enumerate(seg)
            outlier[i] = flagged[k]
        end
    end
    return outlier
end

"""
    sigma_clip_lightcurve(lc::LightCurve; sigma=5.0, window=101, iters=5,
                          protect_transit=true) -> LightCurve

Sigma-clip a `LightCurve` and TRIM the flagged outliers (removes them entirely,
like [`trim_lightcurve`](@ref) — they won't appear in the output, raw or
detrended). When `protect_transit=true` and `lc.is_transit` is present, in-transit
cadences are protected from clipping. Uses [`sigma_clip_mask`](@ref) per segment.
"""
function sigma_clip_lightcurve(lc::LightCurve; sigma::Real=5.0, window::Integer=101,
                                iters::Integer=5, protect_transit::Bool=true)
    prot = (protect_transit && lc.is_transit !== nothing) ? lc.is_transit : nothing
    mask = sigma_clip_mask(lc.t, lc.flux; sigma=sigma, window=window, iters=iters,
                            sector_id=lc.sector_id, protect_mask=prot)
    any(mask) || return lc
    return trim_lightcurve(lc, mask)
end

"""
    clean_lightcurve(lc::LightCurve; method::Symbol=:notch,
                     trim_mask=nothing, kwargs...) -> LightCurve

Foolproof one-call detrender. Dispatches to the matching `detrend_*`, forwarding
the per-cadence `sector_id` and any `kwargs`. Returns a new `LightCurve` with
`flux_detrended`, `trend`, `is_transit`, and `method` filled.

`trim_mask::Union{Nothing,AbstractVector{Bool}}` — if given (length == cadences),
the flagged cadences (`true`) are TRIMMED via [`trim_lightcurve`](@ref) **before**
detrending, so the returned `LightCurve` (raw and detrended alike) contains none
of them. This differs from a `detrend_*` `transit_mask` kwarg, which keeps masked
cadences but protects them from the detrender.

`method ∈ (:notch, :locor, :savgol, :gp)`. Default `:notch` (robust across
rotation regimes, no reliable `P_rot` needed, and yields the transit flag via
`delta_bic > 0`). The other methods don't find transits, so `is_transit` is all
`false`. For `:gp` you must pass the kernel as `kernel=CeleriteRotation()` etc.

On detrender error this `@warn`s and returns a `LightCurve` with `method=:failed`,
`flux_detrended=copy(flux)`, `trend=ones`, `is_transit=falses` (never rethrows).
"""
function clean_lightcurve(lc::LightCurve; method::Symbol=:notch,
                          trim_mask::Union{Nothing,AbstractVector{Bool}}=nothing,
                          sigma_clip::Union{Nothing,Real,NamedTuple}=nothing,
                          kwargs...)
    method in (:notch, :locor, :savgol, :gp) || throw(ArgumentError(
        "method must be one of (:notch, :locor, :savgol, :gp), got :$method"))

    kw = Dict{Symbol,Any}(kwargs)

    # Hard-trim flagged cadences BEFORE detrending so the returned LightCurve
    # (raw + detrended) never carries them. Two complementary sources:
    #   * trim_mask    — user-supplied, time/cadence-based (flares, ramps, dumps)
    #   * sigma_clip   — statistical: extreme point-outliers (cosmic rays, hot
    #                    pixels) flagged via running-median residuals (MAD),
    #                    protecting in-transit cadences (the transit_mask kwarg).
    # Both are unioned and trimmed once; an overlapping transit_mask is kept
    # aligned (overlapping cadences are trimmed, surviving entries re-indexed).
    combined_trim = trim_mask === nothing ? nothing : collect(Bool, trim_mask)
    if sigma_clip !== nothing
        scp  = sigma_clip isa Real ? (sigma = Float64(sigma_clip),) : sigma_clip
        prot = get(kw, :transit_mask, nothing)
        sc_mask = sigma_clip_mask(lc.t, lc.flux;
            sigma   = Float64(get(scp, :sigma, 5.0)),
            window  = Int(get(scp, :window, 101)),
            iters   = Int(get(scp, :iters, 5)),
            sector_id = lc.sector_id,
            protect_mask = prot === nothing ? nothing : Vector{Bool}(prot))
        combined_trim = combined_trim === nothing ? sc_mask : (combined_trim .| sc_mask)
    end
    if combined_trim !== nothing && any(combined_trim)
        keep = .!combined_trim
        lc = trim_lightcurve(lc, combined_trim)
        if haskey(kw, :transit_mask) && kw[:transit_mask] !== nothing
            kw[:transit_mask] = kw[:transit_mask][keep]
        end
    end

    # :gp needs a positional kernel supplied as a kwarg.
    kernel = nothing
    if method === :gp
        haskey(kw, :kernel) || throw(ArgumentError(
            "method=:gp requires a kernel kwarg, e.g. clean_lightcurve(lc; method=:gp, kernel=CeleriteRotation())"))
        kernel = pop!(kw, :kernel)
    end

    n = length(lc.t)
    try
        res = if method === :notch
            detrend_notch(lc.t, lc.flux, lc.flux_err; sector_id=lc.sector_id, kw...)
        elseif method === :locor
            detrend_locor(lc.t, lc.flux, lc.flux_err; sector_id=lc.sector_id, kw...)
        elseif method === :savgol
            detrend_savgol(lc.t, lc.flux, lc.flux_err; sector_id=lc.sector_id, kw...)
        else # :gp
            detrend_gp(lc.t, lc.flux, lc.flux_err, kernel; sector_id=lc.sector_id, kw...)
        end

        is_transit = method === :notch ? (res.delta_bic .> 0) : falses(n)
        return LightCurve(lc.t, lc.flux, lc.flux_err, lc.sector_id,
                          res.flux_detrended, res.trend, Vector{Bool}(is_transit), method)
    catch err
        @warn "clean_lightcurve: detrending failed (method=:$method); returning raw flux" exception=(err, catch_backtrace())
        return LightCurve(lc.t, lc.flux, lc.flux_err, lc.sector_id,
                          copy(lc.flux), ones(Float64, n), falses(n), :failed)
    end
end

"""
    sectors(lc::LightCurve) -> Vector{LightCurve}

Split a `LightCurve` into one `LightCurve` per unique sector (sorted by sector
number), carrying the detrend fields (sliced) if present. Within-sector cadence
order is preserved.
"""
function sectors(lc::LightCurve)
    out = LightCurve[]
    for s in sort(unique(lc.sector_id))
        idx = findall(==(s), lc.sector_id)
        fd  = lc.flux_detrended === nothing ? nothing : lc.flux_detrended[idx]
        tr  = lc.trend          === nothing ? nothing : lc.trend[idx]
        it  = lc.is_transit     === nothing ? nothing : lc.is_transit[idx]
        push!(out, LightCurve(lc.t[idx], lc.flux[idx], lc.flux_err[idx],
                              lc.sector_id[idx], fd, tr, it, lc.method))
    end
    return out
end
