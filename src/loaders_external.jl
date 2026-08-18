# Loaders for external observational data — TESS / Kepler photometry,
# Vizier-published RV time series.
#
# These keep the Julia side thin: the Python sidecar
# (`scripts/fetch_tess_lcs.py`) downloads the .fits files and writes
# normalized CSVs alongside; Nereus just ingests the CSVs. Per spec
# decision D5, the package never touches MAST / lightkurve / astroquery
# directly.

using DelimitedFiles
using Printf
using Statistics: median
using FITSIO
using NCDatasets

"""
    load_tess_lc(csv_path; trim_window=nothing) -> NamedTuple

Read a TESS LC CSV produced by `scripts/fetch_tess_lcs.py`. Columns:
`bjd_tdb,pdcsap_flux_norm,pdcsap_flux_err_norm` (NaN-stripped, median-
normalized to 1).

Returns `(t, flux, flux_err)` ready to drop into a `Data(...)`
photometry block as `(t_phot, flux, flux_err)`.

# Optional trimming

- `trim_window::Union{Nothing, NTuple{2, Real}}=nothing`  — `(t_lo,
  t_hi)` BJD bounds; only points within the window are returned. Useful
  to pick a contiguous slab when a sector spans a downlink gap.
"""
function load_tess_lc(csv_path::AbstractString;
                      trim_window::Union{Nothing, NTuple{2, <:Real}} = nothing)
    isfile(csv_path) || throw(ArgumentError("LC CSV not found: $csv_path"))
    # `comments=true` skips `# key = value` metadata blocks emitted by
    # `save_lightcurve`'s CSV writer, so cleaned LCs round-trip cleanly.
    raw = readdlm(csv_path, ',', Any, '\n';
                   header = true, comments = true, comment_char = '#')
    rows = raw[1]
    t        = Float64.(rows[:, 1])
    flux     = Float64.(rows[:, 2])
    flux_err = Float64.(rows[:, 3])

    if trim_window !== nothing
        lo, hi = Float64(trim_window[1]), Float64(trim_window[2])
        keep = (t .>= lo) .& (t .<= hi)
        t        = t[keep]
        flux     = flux[keep]
        flux_err = flux_err[keep]
    end

    return (t = t, flux = flux, flux_err = flux_err)
end

"""
    load_vizier_rv(csv_path; t_col=:bjd, rv_col=:rv, err_col=:rv_err,
                   inst_col=:inst) -> NamedTuple

Read an RV CSV (from Vizier or a paper data release). Default column
names are `bjd, rv, rv_err, inst` (instrument string). Returns
`(t, rv, rv_err, rv_inst)` with `rv_inst` as 1-based integers (sorted
unique instrument strings define the ordering).

Pass column-name overrides as `Symbol` kwargs when the source uses
different headers.

# Example

    rv = load_vizier_rv("data/HD209458/HD209458_rv.csv")
    target = build_target(M_pri = 1.148, planets = (b = (...),),
                          rv = (HARPS = (data = rv, sigma = ...),))

If the file has a single instrument (no `inst_col`), pass
`inst_col = nothing`; all rows get instrument index 1.
"""
function load_vizier_rv(csv_path::AbstractString;
                        t_col::Symbol  = :bjd,
                        rv_col::Symbol = :rv,
                        err_col::Symbol = :rv_err,
                        inst_col::Union{Symbol, Nothing} = :inst)
    isfile(csv_path) || throw(ArgumentError("RV CSV not found: $csv_path"))
    raw = readdlm(csv_path, ',', Any, '\n'; header = true)
    rows = raw[1]
    headers = Symbol.(strip.(string.(vec(raw[2]))))

    function col_idx(s::Symbol)
        idx = findfirst(==(s), headers)
        idx === nothing &&
            throw(ArgumentError("RV CSV $csv_path missing column `$s`; " *
                                "headers are $(headers)"))
        return idx
    end

    t      = Float64.(rows[:, col_idx(t_col)])
    rv     = Float64.(rows[:, col_idx(rv_col)])
    rv_err = Float64.(rows[:, col_idx(err_col)])

    if inst_col === nothing
        rv_inst = ones(Int, length(t))
        inst_names = ["INST"]
    else
        inst_strs = String.(strip.(string.(rows[:, col_idx(inst_col)])))
        inst_names = sort!(unique(inst_strs))
        inst_to_idx = Dict(name => i for (i, name) in enumerate(inst_names))
        rv_inst = [inst_to_idx[s] for s in inst_strs]
    end

    return (t = t, rv = rv, rv_err = rv_err, rv_inst = rv_inst,
            instruments = inst_names)
end

# =====================================================================
# Light curve saving — multi-format
# =====================================================================

"""
    save_lightcurve(path, t, flux, flux_err; metadata=Dict(), format=:auto,
                    time_col=:bjd_tdb, flux_col=:flux, flux_err_col=:flux_err,
                    overwrite=true)

Write a (cleaned) light curve to disk in CSV, NetCDF, or FITS format.
Format is auto-detected from the file extension when `format=:auto`:
`.csv` → `:csv`, `.nc` → `:nc`, `.fits` → `:fits`. Pass an explicit
`format` symbol to override.

# Format details

- **CSV**: header `bjd_tdb,flux,flux_err` (configurable via column-name
  kwargs), one row per cadence. Metadata is written as a leading
  comment-line block (`# key = value`) before the header.
- **NetCDF**: a single dataset with `time`, `flux`, `flux_err` variables
  along the `cadence` dimension. `metadata` becomes attributes (string,
  numeric, or vector). Compatible with `xarray.open_dataset`.
- **FITS**: primary HDU with metadata as header keywords, plus a
  `LIGHTCURVE` binary table HDU with three columns
  (`TIME`, `FLUX`, `FLUX_ERR`).

`metadata` accepts string, numeric, or vector values. NetCDF and FITS
formats preserve types; CSV serialises everything to text.
"""
function save_lightcurve(path::AbstractString,
                          t::AbstractVector{<:Real},
                          flux::AbstractVector{<:Real},
                          flux_err::AbstractVector{<:Real};
                          metadata::AbstractDict = Dict{String, Any}(),
                          format::Symbol = :auto,
                          time_col::Symbol = :bjd_tdb,
                          flux_col::Symbol = :flux,
                          flux_err_col::Symbol = :flux_err,
                          overwrite::Bool = true)
    n = length(t)
    n == length(flux) == length(flux_err) ||
        throw(ArgumentError("t, flux, flux_err must have the same length"))

    fmt = format === :auto ? _detect_lc_format(path) : format
    !overwrite && isfile(path) && throw(ArgumentError(
        "$path exists and overwrite=false"))
    overwrite && isfile(path) && rm(path)
    mkpath(dirname(abspath(path)))

    tt = collect(Float64, t)
    ff = collect(Float64, flux)
    ee = collect(Float64, flux_err)
    meta_str = Dict{String, Any}(string(k) => v for (k, v) in metadata)

    if fmt === :csv
        _save_lc_csv(path, tt, ff, ee, meta_str,
                     time_col, flux_col, flux_err_col)
    elseif fmt === :nc
        _save_lc_nc(path, tt, ff, ee, meta_str)
    elseif fmt === :fits
        _save_lc_fits(path, tt, ff, ee, meta_str)
    else
        throw(ArgumentError("save_lightcurve: unknown format `:$fmt` " *
                             "(supported: :csv, :nc, :fits, or :auto)"))
    end
    return path
end

function _detect_lc_format(path::AbstractString)
    ext = lowercase(splitext(path)[2])
    ext == ".csv" && return :csv
    ext == ".nc"  && return :nc
    (ext == ".fits" || ext == ".fit") && return :fits
    throw(ArgumentError(
        "save_lightcurve: cannot infer format from extension `$ext` " *
        "(supported: .csv, .nc, .fits) — pass `format=` explicitly"))
end

function _save_lc_csv(path, t, flux, flux_err, meta,
                       tc::Symbol, fc::Symbol, fec::Symbol)
    open(path, "w") do io
        for k in sort(collect(keys(meta)))
            v = meta[k]
            v_str = v isa AbstractVector ? join(v, ";") : string(v)
            write(io, "# $k = $v_str\n")
        end
        write(io, "$(string(tc)),$(string(fc)),$(string(fec))\n")
        for i in eachindex(t)
            @printf(io, "%.10f,%.8e,%.8e\n", t[i], flux[i], flux_err[i])
        end
    end
end

function _save_lc_nc(path, t, flux, flux_err, meta)
    NCDatasets.Dataset(path, "c") do ds
        defDim(ds, "cadence", length(t))
        v_t = defVar(ds, "time", Float64, ("cadence",))
        v_f = defVar(ds, "flux", Float64, ("cadence",))
        v_e = defVar(ds, "flux_err", Float64, ("cadence",))
        v_t[:] = t
        v_f[:] = flux
        v_e[:] = flux_err
        v_t.attrib["units"] = "days"
        v_t.attrib["description"] = "Barycentric Julian Date (TDB)"
        v_f.attrib["description"] = "Cleaned/normalised flux"
        v_e.attrib["description"] = "Per-cadence flux uncertainty"
        for (k, v) in meta
            ds.attrib[k] = v isa AbstractVector ? collect(v) : v
        end
    end
end

function _save_lc_fits(path, t, flux, flux_err, meta)
    # Build a header with metadata keywords. FITS keywords are limited
    # to 8-char names + ≤70-char string values; long values are
    # truncated. For the primary HDU we write a 1-element zero array
    # (FITSIO requires at least one element) and attach the header.
    hdr_keys = String[]
    hdr_vals = Any[]
    hdr_cmts = String[]
    for k in sort(collect(keys(meta)))
        push!(hdr_keys, _fits_keyword(k))
        push!(hdr_vals, _fits_value(meta[k]))
        push!(hdr_cmts, "")
    end
    header = FITSHeader(hdr_keys, hdr_vals, hdr_cmts)

    f = FITS(path, "w")
    try
        write(f, [0.0]; header = header)   # primary HDU placeholder
        cols = Dict("TIME" => t, "FLUX" => flux, "FLUX_ERR" => flux_err)
        write(f, cols; hdutype = TableHDU, name = "LIGHTCURVE")
    finally
        close(f)
    end
end

"""Sanitize a metadata key for use as a FITS keyword (≤ 8 ASCII chars)."""
function _fits_keyword(k::AbstractString)
    s = uppercase(replace(k, r"[^A-Za-z0-9_]" => ""))
    isempty(s) && (s = "META")
    length(s) > 8 ? s[1:8] : s
end

"""FITS header values must be a primitive type or short string. Vectors
get joined with ';' separators and truncated to fit the FITS limit."""
_fits_value(v::AbstractVector) = first_72_chars(join(v, ";"))
_fits_value(v::AbstractString) = first_72_chars(v)
_fits_value(v) = v
first_72_chars(s::AbstractString) = length(s) > 70 ? s[1:70] : String(s)
