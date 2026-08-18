# Gaia epoch astrometry (DR4 format) — reader + IADData adapter.
#
# Gaia DR4 (2 Dec 2026) publishes per-CCD along-scan (AL) epoch astrometry: for
# each field-of-view transit, the AL centroid position `centroid_pos_al` (mas)
# of the image relative to a reference point (ra0, dec0), together with the scan
# position angle θ and the AL parallax factor. The AL abscissa model is
#
#     w_i = Δα*·sinθ_i + Δδ·cosθ_i + ϖ·p_AL,i
#           + μα*·sinθ_i·Δt_i + μδ·cosθ_i·Δt_i        (+ photocenter orbit)
#
# with Δt_i the time relative to the DR4 reference epoch J2017.5 in Julian
# years. This is *identical* in structure to the Hipparcos IAD abscissa model
# that `IADData` + `iad_log_likelihood` already implement (design vector
# `(sinψ, cosψ, plx_factor, sinψ·pm_factor, cosψ·pm_factor)`, 5 linear nuisance
# parameters marginalised analytically). So a Gaia epoch source drops straight
# into the existing likelihood by mapping ψ → deg2rad(scan_pos_angle),
# parallax_factor → parallax_factor_al, pm_factor → Δt[yr], abscissa →
# centroid_pos_al. No new likelihood is required.
#
# The data is distributed as a VOTable with a BINARY2 (base64) payload and
# variable-length per-CCD arrays. There is no Julia VOTable/BINARY2 reader in
# the General registry, so — as with `_parse_van_leeuwen_iad` — we decode the
# specific schema by hand. The reader is validated against ESA's `gaiasupdate`
# reference fit to machine precision (test/validation/validate_gaia_dr4_epoch.jl).
#
# Reference epochs (ESA gaiasupdate/constants.py):
#   TCB time origin  = 2010-01-01T00:00:00 TCB  (JD 2455197.5 = MJD 55197.0)
#   DR4 ref epoch    = J2017.5 (TCB)
#
# Pre-release (June 2026): epoch astrometry for 12 illustrative sources,
# https://www.cosmos.esa.int/web/gaia/dr4-prerelease

using Base64
using Downloads

export GaiaEpochSource, read_gaia_epoch_votable, fetch_gaia_dr4_prerelease

# --- BINARY2 field schema for the DR4 epoch-astrometry table, in stream order.
# (name, element type, is_variable_length). Variable-length arrays hold the
# per-CCD values of a field-of-view transit.
const _GEP_LONG, _GEP_DBL, _GEP_FLT, _GEP_SHORT, _GEP_BOOL = :long, :double, :float, :short, :bool
const _GAIA_EPOCH_SCHEMA = [
    (:solution_id, _GEP_LONG, false), (:source_id, _GEP_LONG, false),
    (:transit_id, _GEP_LONG, false), (:ra0, _GEP_DBL, false), (:dec0, _GEP_DBL, false),
    (:agis_source_excess_noise, _GEP_FLT, false),
    (:obs_time_tcb, _GEP_LONG, true), (:obs_time_bary_corr, _GEP_FLT, false),
    (:scan_pos_angle, _GEP_DBL, true), (:zeta, _GEP_FLT, false),
    (:parallax_factor_al, _GEP_FLT, false), (:parallax_factor_ac, _GEP_FLT, false),
    (:colour_factor_al, _GEP_FLT, true), (:colour_factor_ac, _GEP_FLT, true),
    (:nu_eff_used_in_astrometry, _GEP_FLT, false), (:nu_eff_error, _GEP_FLT, false),
    (:centroid_pos_al, _GEP_DBL, true), (:centroid_pos_ac, _GEP_DBL, true),
    (:calculated_pos_ac, _GEP_DBL, true),
    (:centroid_pos_error_al, _GEP_FLT, true), (:centroid_pos_error_ac, _GEP_FLT, true),
    (:used_by_agis_al, _GEP_BOOL, true), (:used_by_agis_ac, _GEP_BOOL, true),
    (:transit_acq_flags, _GEP_SHORT, false), (:transit_proc_flags, _GEP_SHORT, false),
    (:ccd_proc_flags, _GEP_SHORT, true), (:multipeak, _GEP_BOOL, false),
    (:blended, _GEP_BOOL, false), (:ipd_error_al, _GEP_FLT, true),
    (:ipd_error_ac, _GEP_FLT, true), (:g_mag, _GEP_FLT, false),
    (:g_class, _GEP_SHORT, false), (:gates, _GEP_SHORT, true),
    (:source_dist_to_last_ci, _GEP_FLT, true), (:ac_rate, _GEP_FLT, false),
    (:sub_pixel_coord, _GEP_FLT, true), (:mu, _GEP_FLT, true),
]

# time constants
const _GEP_NS_TO_YR  = 1e-9 / (365.25 * 86400.0)   # nanoseconds → Julian years
const _GEP_NS_TO_DAY = 1e-9 / 86400.0              # nanoseconds → days
const _GEP_MJD_ORIGIN = 55197.0                    # MJD of 2010-01-01.0 TCB
const _GEP_TCB_ORIGIN_JYEAR = 2010.0               # jyear of the TCB origin
const _GEP_DR4_REF_JYEAR    = 2017.5               # DR4 reference epoch

"""
    GaiaEpochSource

One source's Gaia epoch astrometry, parsed from the DR4 (pre-release) VOTable.

- `source_id`  : Gaia DR3/DR4 source identifier.
- `ra0`,`dec0` : reference point (deg) that the AL centroids are measured from.
- `g_mag`      : representative G magnitude.
- `iad`        : [`IADData`](@ref) holding the per-CCD along-scan abscissae
                 (`used_by_agis_al == true` only), ready for
                 [`iad_log_likelihood`](@ref). `psi = deg2rad(scan_pos_angle)`,
                 `parallax_factor = parallax_factor_al`, `pm_factor = Δt[yr]`
                 relative to J2017.5, `t` in MJD (TCB, barycentric-corrected).
"""
struct GaiaEpochSource
    source_id::Int64
    ra0::Float64
    dec0::Float64
    g_mag::Float64
    iad::IADData
end

function Base.show(io::IO, s::GaiaEpochSource)
    print(io, "GaiaEpochSource(source_id=", s.source_id, ", N_ccd=", length(s.iad.t),
          ", G=", round(s.g_mag, digits=2), ")")
end

# --- BINARY2 stream extraction + decode -------------------------------------

_gep_read_scalar(io, t) =
    t === _GEP_LONG  ? ntoh(read(io, Int64)) :
    t === _GEP_DBL   ? ntoh(read(io, Float64)) :
    t === _GEP_FLT   ? ntoh(read(io, Float32)) :
    t === _GEP_SHORT ? ntoh(read(io, Int16)) :
                       read(io, UInt8)          # boolean: 'T'/'F'/'?' byte

function _gep_read_array(io, t)
    n = ntoh(read(io, Int32))
    out = Vector{Any}(undef, n)
    @inbounds for i in 1:n
        out[i] = _gep_read_scalar(io, t)
    end
    return out
end

# pull the base64 <STREAM> payload out of the VOTable and decode to bytes
function _gaia_epoch_stream_bytes(path::AbstractString)
    xml = read(path, String)
    i = findfirst("<STREAM", xml)
    i === nothing && throw(ArgumentError("no <STREAM> in $path — not a BINARY2 VOTable?"))
    gt = first(i) + first(findfirst(">", xml[first(i):end])) - 1     # '>' closing the opening tag
    e  = first(i) + first(findfirst("</STREAM>", xml[first(i):end])) - 1
    b64 = replace(xml[gt+1:e-1], r"\s" => "")
    return base64decode(b64)
end

# decode all rows into a column-oriented Dict (Symbol => Vector)
function _decode_gaia_epoch(path::AbstractString)
    io = IOBuffer(_gaia_epoch_stream_bytes(path))
    nmaskbytes = cld(length(_GAIA_EPOCH_SCHEMA), 8)
    cols = Dict{Symbol, Vector{Any}}(f[1] => Any[] for f in _GAIA_EPOCH_SCHEMA)
    nrows = 0
    while !eof(io)
        read(io, nmaskbytes)                     # BINARY2 null mask (all values encoded regardless)
        for (name, t, isvar) in _GAIA_EPOCH_SCHEMA
            push!(cols[name], isvar ? _gep_read_array(io, t) : _gep_read_scalar(io, t))
        end
        nrows += 1
    end
    return cols, nrows
end

_gep_isT(b) = (b == UInt8('T') || b == UInt8('t') || b == UInt8('1'))

# assemble one source's IADData from the exploded per-CCD transits
function _gep_build_source(cols, rows_idx)
    # Guard against duplicated transit blocks (cf. the Octofitter G23H audit —
    # a user catalog with repeated per-target scan blocks double-counts data).
    # A repeated transit_id would double-count that transit's abscissae; keep the
    # first occurrence and warn.
    tids = cols[:transit_id]
    seen = Set{Int64}(); kept = Int[]; n_dup = 0
    for r in rows_idx
        tid = Int64(tids[r])
        if tid in seen
            n_dup += 1
        else
            push!(seen, tid); push!(kept, r)
        end
    end
    n_dup > 0 && @warn "read_gaia_epoch_votable: dropped $n_dup duplicate transit " *
                       "block(s) (repeated transit_id) — double-counting guard."
    rows_idx = kept

    t_mjd = Float64[]; abscissa = Float64[]; abscissa_err = Float64[]
    psi = Float64[]; plx_fac = Float64[]; pm_fac = Float64[]
    for r in rows_idx
        times = cols[:obs_time_tcb][r]
        n = length(times)
        bc  = Float64(cols[:obs_time_bary_corr][r])           # per-transit scalar (ns)
        pAL = Float64(cols[:parallax_factor_al][r])           # per-transit scalar
        θs  = cols[:scan_pos_angle][r]
        ws  = cols[:centroid_pos_al][r]
        σs  = cols[:centroid_pos_error_al][r]
        us  = cols[:used_by_agis_al][r]
        @inbounds for j in 1:n
            _gep_isT(us[j]) || continue
            w = Float64(ws[j]); σ = Float64(σs[j])
            (isfinite(w) && isfinite(σ) && σ > 0) || continue
            tns = Float64(times[j]) + bc                       # barycentric-corrected ns since 2010.0
            push!(t_mjd,        _GEP_MJD_ORIGIN + tns * _GEP_NS_TO_DAY)
            push!(pm_fac,       _GEP_TCB_ORIGIN_JYEAR + tns * _GEP_NS_TO_YR - _GEP_DR4_REF_JYEAR)
            push!(psi,          deg2rad(Float64(θs[j])))
            push!(plx_fac,      pAL)
            push!(abscissa,     w)
            push!(abscissa_err, σ)
        end
    end
    return IADData(; t = t_mjd, abscissa = abscissa, abscissa_err = abscissa_err,
                   psi = psi, parallax_factor = plx_fac, pm_factor = pm_fac)
end

"""
    read_gaia_epoch_votable(path) -> Dict{Int64, GaiaEpochSource}
    read_gaia_epoch_votable(path, source_id) -> GaiaEpochSource

Parse a Gaia DR4-format epoch-astrometry VOTable (BINARY2). With no `source_id`,
returns every source keyed by identifier; with one, returns just that source.

Only CCD transits flagged `used_by_agis_al == true` (and with finite, positive
error) are retained. The resulting [`IADData`](@ref) plugs directly into
[`iad_log_likelihood`](@ref) — the star's 5 astrometric parameters are treated
as marginalised nuisances and any planetary/binary photocentre orbit is the
signal.
"""
function read_gaia_epoch_votable(path::AbstractString)
    cols, _ = _decode_gaia_epoch(path)
    sids = cols[:source_id]
    out = Dict{Int64, GaiaEpochSource}()
    for sid in unique(sids)
        idx = findall(==(sid), sids)
        out[Int64(sid)] = GaiaEpochSource(Int64(sid),
            Float64(cols[:ra0][idx[1]]), Float64(cols[:dec0][idx[1]]),
            Float64(cols[:g_mag][idx[1]]), _gep_build_source(cols, idx))
    end
    return out
end

function read_gaia_epoch_votable(path::AbstractString, source_id::Integer)
    cols, _ = _decode_gaia_epoch(path)
    idx = findall(==(Int64(source_id)), cols[:source_id])
    isempty(idx) && throw(ArgumentError("source_id $source_id not found in $path"))
    return GaiaEpochSource(Int64(source_id),
        Float64(cols[:ra0][idx[1]]), Float64(cols[:dec0][idx[1]]),
        Float64(cols[:g_mag][idx[1]]), _gep_build_source(cols, idx))
end

# --- pre-release downloader --------------------------------------------------

const GAIA_DR4_PRERELEASE_URL =
    "https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/" *
    "gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip"
const GAIA_DR4_PRERELEASE_XML = "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"

"""
    fetch_gaia_dr4_prerelease(; cache_dir = nereus_cache_dir("GaiaDR4"), force = false)
        -> String

Download and unzip the June-2026 Gaia DR4 pre-release epoch-astrometry bundle
(12 illustrative sources, incl. the orbital systems HD 114762, Gaia-4, Gaia BH3)
and return the path to the extracted VOTable. Cached; pass `force=true` to
re-download. Feed the result to [`read_gaia_epoch_votable`](@ref).
"""
function fetch_gaia_dr4_prerelease(; cache_dir::AbstractString = nereus_cache_dir("GaiaDR4"),
                                    force::Bool = false)
    mkpath(cache_dir)
    xml_path = joinpath(cache_dir, GAIA_DR4_PRERELEASE_XML)
    (!force && isfile(xml_path)) && return xml_path
    zip_path = joinpath(cache_dir, basename(GAIA_DR4_PRERELEASE_URL))
    if force || !isfile(zip_path)
        @info "Downloading Gaia DR4 pre-release epoch astrometry" url = GAIA_DR4_PRERELEASE_URL
        Downloads.download(GAIA_DR4_PRERELEASE_URL, zip_path)
    end
    success(`unzip -o -q -d $cache_dir $zip_path $GAIA_DR4_PRERELEASE_XML`) ||
        throw(ErrorException("failed to unzip $zip_path"))
    isfile(xml_path) || throw(ErrorException("$GAIA_DR4_PRERELEASE_XML not found after unzip"))
    return xml_path
end
