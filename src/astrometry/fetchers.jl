# Programmatic fetchers for Hipparcos IAD (van Leeuwen 2007 / 2014 Java Tool
# re-distribution) and Gaia GOST scanning-law predictions. Both ESA tools
# distribute their data without a public REST API — the IAD ships as a
# single 350 MB zip bundle, and GOST runs through a JSP form that we
# replay by hand (session GET → form POST → async result poll → CSV
# export). These fetchers wrap both and cache the results locally so
# subsequent fetches are instant.
#
# Cache layout (default `~/.nereus/`):
#   ~/.nereus/IAD/ResRec_JavaTool_2014.zip           — full 350 MB bundle
#   ~/.nereus/IAD/ResRec_JavaTool_2014/Hxxx/Hxxxyyy.d — extracted per-HIP file
#   ~/.nereus/GOST/<query-hash>.csv                   — per-query CSV
#
# Both functions use system-level `curl` and `unzip` rather than pulling
# in `HTTP.jl` and `ZipFile.jl` dependencies; the network handling is
# robust at the curl layer (retries, timeouts, redirects).

using Downloads
using SHA
using Dates

export fetch_hip_iad, fetch_gost, fetch_hgca

const HIP_IAD_BUNDLE_URL =
    "https://www.cosmos.esa.int/documents/532822/6470227/ResRec_JavaTool_2014.zip/" *
    "a58ad12e-cffb-f959-0ed5-2ae26899f61a?t=1631109433177&download=true"

const HIP_IAD_BUNDLE_SIZE_MB = 350  # rough — for user-facing messages

# Brandt 2021 HGCA (eDR3) — the canonical Hipparcos-Gaia 3-PM catalog that
# orvara / orbitize! / Octofitter all consume. A single ~20 MB FITS table.
const HGCA_EDR3_URL      = "https://physics.ucsb.edu/~tbrandt/HGCA_vEDR3.fits"
const HGCA_EDR3_SIZE_MB  = 20

const GOST_BASE_URL = "https://gaia.esac.esa.int/gost"

# Current GOST queryable interval (UTC ISO, TCB scale). This range
# advances with each Gaia data release; the fetcher detects out-of-range
# errors at POST time and reports the server's authoritative bounds.
const GOST_DEFAULT_FROM = "2014-07-26T00:00:00"
const GOST_DEFAULT_TO   = "2025-01-15T00:00:00"


"""
    nereus_cache_dir(subdir::AbstractString = "") -> String

Resolve the Nereus local cache directory. Falls back to
`~/.nereus` when `\$NEREUS_DATA_CACHE` is unset. Creates the
directory if it doesn't exist.
"""
function nereus_cache_dir(subdir::AbstractString = "")
    base = get(ENV, "NEREUS_DATA_CACHE",
                joinpath(homedir(), ".nereus"))
    dir = isempty(subdir) ? base : joinpath(base, subdir)
    mkpath(dir)
    return dir
end


# =====================================================================
# Hipparcos IAD bundle fetcher
# =====================================================================

"""
    fetch_hip_iad(hip_id::Integer;
                   catalogue_pos, catalogue_pm, parallax_mas,
                   epoch_ref = 1991.25,
                   cache_dir = nereus_cache_dir("IAD"),
                   force_redownload = false,
                   verbose = true) -> IADData

Fetch the van Leeuwen 2007 Hipparcos IAD for a specific HIP source and
return the parsed `IADData` ready to drop into
`Data(; iad = ...)`.

The first call downloads the full 350 MB ResRec_JavaTool_2014 bundle
from ESA's CDN to `cache_dir/ResRec_JavaTool_2014.zip` and extracts
the requested HIP file. Subsequent calls hit the local cache (instant).
The bundle is frozen 2014 vintage of the 2007 reduction — no need to
refresh; the cached version is permanent.

# Arguments
- `hip_id` — Hipparcos source ID (1..120 416).
- `catalogue_pos = (ra_deg, dec_deg)` — required to reconstruct the
  along-scan projection vectors.
- `catalogue_pm = (pmra_mas_per_yr, pmde_mas_per_yr)` — required.
- `parallax_mas` — required.
- `epoch_ref` — reference epoch (default 1991.25, the Hipparcos epoch).
- `cache_dir` — where to land the bundle. Defaults to
  `\$NEREUS_DATA_CACHE/IAD` or `~/.nereus/IAD`.
- `force_redownload` — re-fetch the bundle even if cached.

# Disk cost
- 350 MB for the full bundle (one-time)
- ~5 KB per extracted source

To free disk space after extracting only a handful of sources, you
can safely `rm -rf cache_dir/ResRec_JavaTool_2014/` and re-fetch on
demand.
"""
function fetch_hip_iad(hip_id::Integer;
                       cache_dir::AbstractString = nereus_cache_dir("IAD"),
                       force_redownload::Bool = false,
                       verbose::Bool = true)
    hip_id_int = Int(hip_id)
    1 <= hip_id_int <= 120_416 || throw(ArgumentError(
        "hip_id out of valid Hipparcos range [1, 120_416]: $hip_id_int"))

    # File path inside the extracted bundle: HXXX/HXXXYYY.d where
    # XXX = first three digits of the zero-padded 6-digit HIP ID.
    hipstr  = lpad(hip_id_int, 6, '0')
    subdir  = "H" * hipstr[1:3]
    fname   = "H" * hipstr * ".d"
    extract_root = joinpath(cache_dir, "ResRec_JavaTool_2014")
    target_path  = joinpath(extract_root, subdir, fname)

    if !isfile(target_path) || force_redownload
        zip_path = joinpath(cache_dir, "ResRec_JavaTool_2014.zip")
        if !isfile(zip_path) || force_redownload
            verbose && @info "Downloading Hipparcos IAD bundle (~$(HIP_IAD_BUNDLE_SIZE_MB) MB) " *
                              "from ESA CDN to $zip_path"
            Downloads.download(HIP_IAD_BUNDLE_URL, zip_path;
                                timeout = 600.0)
        end
        member_path = "ResRec_JavaTool_2014/$subdir/$fname"
        verbose && @info "Extracting $member_path from local bundle"
        ok = success(`unzip -o -q -d $cache_dir $zip_path $member_path`)
        ok && isfile(target_path) || throw(ErrorException(
            "Failed to extract $member_path from $zip_path; " *
            "check that hip_id=$hip_id_int is in the catalogue."))
    end

    return _parse_van_leeuwen_iad(target_path)
end


"""
    _parse_van_leeuwen_iad(path) -> IADData

Parse the ESA Hipparcos van Leeuwen 2007 IAD ASCII format directly.
This is the format ESA distributes via the ResRec_JavaTool zip; it
differs from `load_hip_iad`'s default 6-column synthetic format and
needs its own parser.

Column layout (whitespace-separated, lines starting with `#` are
header / comments):
    IORB  EPOCH  PARF  CPSI  SPSI  RES  SRES

- `IORB`  : satellite orbit number (informational, not used).
- `EPOCH` : time relative to 1991.25 in Julian years.
- `PARF`  : along-scan parallax factor.
- `CPSI`, `SPSI` : cos / sin of the along-scan position angle.
- `RES`   : along-scan abscissa residual (mas).
- `SRES`  : 1-sigma uncertainty on the residual (mas).

The returned `IADData` carries `psi = atan2(CPSI, SPSI)` (= htof's
scan angle θ, so the along-scan unit vector is (sin ψ, cos ψ) =
(CPSI, SPSI)) derived on-the-fly, `pm_factor = EPOCH` (Nereus
convention: time baseline
in years from epoch_ref), and time converted from years-since-1991.25
to MJD via `jyear_to_mjd(1991.25 + EPOCH)`.
"""
function _parse_van_leeuwen_iad(path::AbstractString)
    isfile(path) || throw(ArgumentError("IAD file not found: $path"))

    t        = Float64[]
    pxf      = Float64[]
    pmf      = Float64[]
    abscissa = Float64[]
    abs_err  = Float64[]
    psi_vec  = Float64[]

    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        toks = split(s)
        # Need exactly 7 columns per the van Leeuwen 2007 format
        length(toks) >= 7 || continue
        try
            epoch  = parse(Float64, toks[2])
            parf   = parse(Float64, toks[3])
            cpsi   = parse(Float64, toks[4])
            spsi   = parse(Float64, toks[5])
            res    = parse(Float64, toks[6])
            sres   = parse(Float64, toks[7])
            # van Leeuwen 2007 convention: SRES < 0 marks scans that
            # were rejected during the catalogue reduction. Skip them.
            sres > 0 || continue
            push!(t,        jyear_to_mjd(1991.25 + epoch))
            push!(pxf,      parf)
            push!(pmf,      epoch)             # time baseline in years
            push!(abscissa, res)
            push!(abs_err,  sres)
            # Scan angle θ such that the along-scan unit vector in
            # (RA*, Dec) is (sin θ, cos θ) = (CPSI, SPSI). Verified against
            # htof (special_parse.to_along_scan_basis + the parser comment
            # `CPSI = sin θ`, `SPSI = cos θ`): θ = atan2(CPSI, SPSI). The
            # along-scan projection Δη = Δra·sin θ + Δdec·cos θ then maps
            # RA→CPSI, Dec→SPSI. Using atan2(SPSI, CPSI) here (= π/2 − θ)
            # silently swaps RA↔Dec in the reflex projection — invisible to
            # self-consistent forward-model tests but biases IAD masses low.
            push!(psi_vec,  atan(cpsi, spsi))
        catch
            continue   # skip malformed lines silently
        end
    end
    isempty(t) && throw(ArgumentError(
        "No IAD records parsed from $path — file format may not be " *
        "van Leeuwen 2007 (expected 7 whitespace columns: IORB EPOCH " *
        "PARF CPSI SPSI RES SRES)."))

    return IADData(;
        t              = t,
        abscissa       = abscissa,
        abscissa_err   = abs_err,
        psi            = psi_vec,
        parallax_factor = pxf,
        pm_factor      = pmf,
    )
end

# ---------------------------------------------------------------------
# HGCA (Brandt 2021 eDR3) — Hipparcos-Gaia 3-PM catalog fetcher
# ---------------------------------------------------------------------

"""
    fetch_hgca(hip_id; cache_dir = nereus_cache_dir("HGCA"),
               force_redownload = false, verbose = true) -> HGCAData

Fetch the Brandt 2021 HGCA (eDR3) three-epoch proper-motion row for a
Hipparcos source and return an `HGCAData` ready to drop into
`Data(; hgca = ...)`. This is the VALIDATED Hipparcos–Gaia astrometry
path (`hgca_log_likelihood`); pair it with `fetch_gost` to upgrade the
Gaia epoch to the GOST-window-averaged Mode B (Brandt 2018 §4), which
matters when the orbital period is comparable to Gaia's ~3-yr window.

Downloads and caches the full `HGCA_vEDR3.fits` (~$(HGCA_EDR3_SIZE_MB) MB —
the same file orvara / orbitize! / Octofitter consume) on first call;
subsequent fetches read the row straight from the cached table.

# Cache
    \$NEREUS_DATA_CACHE/HGCA/HGCA_vEDR3.fits   (or ~/.nereus/HGCA/…)

# Example
```julia
hgca = fetch_hgca(85653)                       # HD 159062 (HIP 85653)
gost = fetch_gost(263.0908, 8.9083)            # Mode-B Gaia epoch
data = Data(; t_rv=…, rv=…, rv_err=…, hgca=hgca, gost=gost)
```
"""
function fetch_hgca(hip_id::Integer;
                    cache_dir::AbstractString = nereus_cache_dir("HGCA"),
                    force_redownload::Bool = false,
                    verbose::Bool = true)
    hip_id_int = Int(hip_id)
    1 <= hip_id_int <= 120_416 || throw(ArgumentError(
        "hip_id out of valid Hipparcos range [1, 120_416]: $hip_id_int"))

    mkpath(cache_dir)
    fits_path = joinpath(cache_dir, "HGCA_vEDR3.fits")
    if !isfile(fits_path) || force_redownload
        verbose && @info "Downloading HGCA eDR3 catalog (~$(HGCA_EDR3_SIZE_MB) MB) " *
                          "from $HGCA_EDR3_URL to $fits_path"
        Downloads.download(HGCA_EDR3_URL, fits_path; timeout = 600.0)
    end
    return load_hgca_row(fits_path, hip_id_int)
end


# =====================================================================
# Gaia GOST fetcher
# =====================================================================

"""
    fetch_gost(name_or_ra::Union{AbstractString, Real},
               dec::Union{Real, Nothing} = nothing;
               from = "$GOST_DEFAULT_FROM",
               to   = "$GOST_DEFAULT_TO",
               epoch_ref = 2016.0,
               cache_dir = nereus_cache_dir("GOST"),
               force_refetch = false,
               verbose = true) -> GOSTData

Fetch Gaia Observation Forecast Tool (GOST) per-scan predictions for a
target and return the parsed `GOSTData` ready to drop into
`Data(; gost = ...)`.

There are two query modes:
1. **By name** (Simbad-resolved): `fetch_gost("eps Eri")` — pass a
   single string argument; the name is sent to GOST's Simbad resolver.
2. **By coordinates** (decimal degrees): `fetch_gost(53.232665, -9.458250)`
   — pass two `Real` arguments.

# Arguments
- `name_or_ra`, `dec` — target identifier or RA/Dec (deg, ICRS).
- `from`, `to` — UTC ISO strings bounding the prediction range. Must
  fall within GOST's queryable window (currently
  `[2014-07-25T10:31:26, 2025-01-15T06:16:32]` — bounds advance each
  data release). On out-of-range request, GOST returns an error that
  this fetcher surfaces.
- `epoch_ref` — reference epoch for the constructed `GOSTData` (default
  J2016.0, the Gaia DR3 reference epoch).
- `cache_dir` — per-query CSVs are stored here keyed by SHA1 of the
  query parameters.
- `force_refetch` — re-fetch even if cached.

# Cost
- One HTTP session per query (~5-15 seconds wall): GET index, POST
  form, poll async job, download CSV.
- Cached CSV: ~30 KB per target × decade.

# Implementation note
This is **not** a stable API — GOST is a JSP form, not a REST
service. If ESA changes the form fields or async job protocol, this
fetcher will need updating. The fall-back path is the interactive web
tool at https://gaia.esac.esa.int/gost/.
"""
function fetch_gost(name_or_ra::Union{AbstractString, Real},
                    dec::Union{Real, Nothing} = nothing;
                    from::AbstractString = GOST_DEFAULT_FROM,
                    to::AbstractString = GOST_DEFAULT_TO,
                    cache_dir::AbstractString = nereus_cache_dir("GOST"),
                    force_refetch::Bool = false,
                    verbose::Bool = true)
    # Build a stable cache key
    if dec === nothing
        # Name mode — name_or_ra must be string
        name_or_ra isa AbstractString || throw(ArgumentError(
            "Single-argument fetch_gost requires a name string; pass " *
            "(ra, dec) as two Reals for coordinate mode."))
        key_input = "name=$(String(name_or_ra))|from=$from|to=$to"
        target_label = String(name_or_ra)
    else
        ra_deg = Float64(name_or_ra)
        dec_deg = Float64(dec)
        key_input = "ra=$ra_deg|dec=$dec_deg|from=$from|to=$to"
        target_label = "coords_$(ra_deg)_$(dec_deg)"
    end
    key = bytes2hex(sha1(key_input))[1:16]
    csv_path = joinpath(cache_dir, "gost_$(key).csv")

    if isfile(csv_path) && !force_refetch
        verbose && @info "GOST cache hit: $csv_path"
        return load_gost(csv_path)
    end

    # JSP form replay: session GET → POST → poll → CSV export.
    cookie_path = tempname() * ".cookies"
    index_path  = tempname() * ".html"
    submit_path = tempname() * ".html"
    result_path = tempname() * ".html"

    try
        # 1. Session GET
        verbose && @info "GOST: querying for $target_label"
        run(pipeline(`curl -sS -c $cookie_path -o $index_path
                      $GOST_BASE_URL/index.jsp`,
                      stderr = devnull))

        # 2. Form POST
        post_args = String[
            "-sS", "-b", cookie_path, "-c", cookie_path, "-X", "POST",
            "-o", submit_path,
            "-F", "csvfilename=",
            "-F", "serviceCode=1",
            "-F", "from=$from",
            "-F", "to=$to",
        ]
        if dec === nothing
            push!(post_args, "-F", "inputmode=resolver")
            push!(post_args, "-F", "srcname=$(target_label)")
        else
            push!(post_args, "-F", "inputmode=single")
            push!(post_args, "-F", "srcname=")
            push!(post_args, "-F", "srcra=$(Float64(name_or_ra))")
            push!(post_args, "-F", "srcdec=$(Float64(dec))")
        end
        push!(post_args, "$GOST_BASE_URL/GostServlet")
        run(pipeline(`curl $post_args`, stderr = devnull))

        submit_text = read(submit_path, String)
        # Server-side error?
        if occursin("error occured", lowercase(submit_text)) ||
           occursin("error occurred", lowercase(submit_text))
            throw(ErrorException(
                "GOST returned an error on submit:\n$(strip(submit_text))"))
        end
        m = match(r"Submitted with id (\d+)", submit_text)
        m === nothing && throw(ErrorException(
            "Could not extract job ID from GOST submit response:\n" *
            "$(strip(submit_text))"))
        job_id = m.captures[1]

        # 3. Poll result.jsp for the export link
        max_polls = 30
        result_ready = false
        for poll in 1:max_polls
            sleep(poll == 1 ? 4 : 2)
            run(pipeline(`curl -sS -b $cookie_path -c $cookie_path
                          -o $result_path $GOST_BASE_URL/result.jsp`,
                          stderr = devnull))
            txt = read(result_path, String)
            if occursin("export.jsp?id=", txt) &&
               occursin("&format=csv", txt)
                result_ready = true
                break
            end
        end
        result_ready || throw(ErrorException(
            "GOST job $job_id did not complete within " *
            "$(max_polls * 2 + 2) s; try the web tool at $GOST_BASE_URL"))

        # 4. Extract the session ID from cookie file (lazy parse; JSESSIONID line)
        sid_line = ""
        for ln in eachline(cookie_path)
            if occursin("JSESSIONID", ln)
                sid_line = ln; break
            end
        end
        # Netscape cookie format: cols are domain, flag, path, secure, expires, name, value
        sid = split(strip(sid_line), '\t')[end]

        # 5. Download CSV
        export_url = "$GOST_BASE_URL/export.jsp?id=$(sid)/$(job_id)&format=csv"
        run(pipeline(`curl -sS -b $cookie_path -o $csv_path $export_url`,
                      stderr = devnull))
        isfile(csv_path) && filesize(csv_path) > 200 || throw(ErrorException(
            "GOST CSV download produced empty/missing file at $csv_path"))
        verbose && @info "GOST: wrote $(filesize(csv_path)) bytes to $csv_path"
    finally
        for p in (cookie_path, index_path, submit_path, result_path)
            isfile(p) && rm(p; force = true)
        end
    end

    return load_gost(csv_path)
end
