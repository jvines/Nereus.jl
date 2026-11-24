# Data — pre-staged observational data.
#
# Per spec decision D5 (see `julia_port_architecture.md` §8), the Julia
# package never touches instrument-specific code. A Python worker in
# the ecosystem handles data discovery, BJD conversion, detrending,
# outlier rejection, instrument-specific systematics, and any other
# per-mission quirks, then hands Nereus a set of plain numeric arrays
# with integer instrument indices.
#
# The `Data` struct here is therefore a thin container. No loaders,
# no detrenders, no reducers. Just arrays + validation that they're
# mutually consistent.
#
# Scope note: this pass provides RV fields only. Photometric fields
# (t_phot, flux, flux_err, phot_inst) will be added when the transit
# likelihood lands — the struct will gain additional fields without
# breaking the current public API.

using Statistics: median

# Astrometry data containers — defined in src/astrometry/data.jl. The
# include order in Nereus.jl loads astrometry/data.jl before this file,
# so `RelAstromData` and `HGCAData` are already in scope here.

"""
    Data

Container for pre-staged observational data. Constructed via the
keyword constructor:

```julia
Data(;
    t_rv   = [...],          # observation times (days)
    rv     = [...],          # RV measurements (m/s)
    rv_err = [...],          # measurement uncertainties (m/s)
    rv_inst = [...],         # instrument index per obs (1-based)
    t_ref  = median(t_rv),   # reference epoch for Mo parametrization
)
```

# Fields
- `t_rv::Vector{Float64}`    : observation times, any unit consistent with
                                the `P`-parameter unit (typically days)
- `rv::Vector{Float64}`      : RV measurements, any unit consistent with
                                `K`-parameter unit (typically m/s)
- `rv_err::Vector{Float64}`  : measurement uncertainties, same unit as `rv`
- `rv_inst::Vector{Int}`     : 1-based instrument index per observation;
                                must match the ordering in
                                `InstrumentConfig.rv_names`
- `t_ref::Float64`           : reference epoch for the `M₀` time anchor;
                                defaults to the median observation time

# Invariants
- `t_rv`, `rv`, `rv_err`, `rv_inst` all have matching length.
- `rv_err` entries are strictly positive.
- `rv_inst` entries are strictly positive (1-based).

Photometry fields are not present in this revision. They will be added
when the transit likelihood is implemented.
"""
struct Data
    # RV
    t_rv::Vector{Float64}
    rv::Vector{Float64}
    rv_err::Vector{Float64}
    rv_inst::Vector{Int}
    rv_comp::Vector{Int}    # stellar component each RV point measures: 1=primary (A),
                            # 2=secondary (B) for an SB2 double-lined fit. All-ones for
                            # the normal single-star case. ORTHOGONAL to rv_inst (which
                            # stays per-spectrograph for γ/σ/activity) so the barycentric
                            # γ is shared across A and B.
    t_ref::Float64
    indicators::Dict{String, Vector{Float64}}
    indicator_errs::Dict{String, Vector{Float64}}     # per-indicator 1σ uncertainty —
                                                       # required by ActivityGP / Rajpaul-style
                                                       # multivariate-GP likelihoods. Missing
                                                       # for indicators where errors weren't
                                                       # provided.
    indicator_derivs::Dict{String, Vector{Float64}}   # per-instrument finite-difference
                                                       # ∂indicator/∂t — used by FF'-style
                                                       # ActivityDecorrelation(derivative=true).
    # Photometry
    t_phot::Vector{Float64}
    flux::Vector{Float64}
    flux_err::Vector{Float64}
    phot_inst::Vector{Int}
    exposure_times::Vector{Float64}   # per-cadence exposure time in DAYS (same
                                       # units as t_phot). Empty ⇒ instantaneous
                                       # model evaluation (no finite-exposure
                                       # supersampling). Length matches t_phot
                                       # when present.
    # Astrometry (Phase 1: RelAstromData + HGCA EDR3 + G23H;
    #            Phase 2 scaffolding: Hipparcos IAD + Gaia GOST)
    relastrom::Union{Nothing, RelAstromData}
    hgca::Union{Nothing, HGCAData}
    iad::Union{Nothing, IADData}
    gost::Union{Nothing, GOSTData}
    g23h::Union{Nothing, G23HData}
    gaia_dr3::Union{Nothing, GaiaDR3Data}
    # Doppler tomography: one residual map per transit night. Empty when no
    # tomography is supplied, which is the common case — the tomographic term
    # then contributes exactly zero and costs nothing.
    tomo::Vector{TomoNight}
    # Hipparcos scan geometry for the Hipparcos epoch of the HGCA / G23H
    # models (`_catalogue_epoch`): epochs, scan angles, parallax factors and
    # abscissa errors of the IAD scans, never their abscissae. The `iad` itself
    # when the fit has Hipparcos IAD, an explicit `hip_scans` otherwise,
    # `nothing` when neither (the Hipparcos epoch is then instantaneous).
    hip_scans::Union{Nothing, IADData}
    # Externally measured mid-transit times, one set per planet
    # (src/transit_times_data.jl). Empty unless given; the transit-time term
    # then contributes exactly zero.
    transit_times::Vector{TransitTimeSet}
    # `t_phot` in time order (the same vector when it already is): the free TTV
    # offsets are numbered from the first cadence near a transit, found by a
    # scan of it at every evaluation (`_ttv_first_transit`, src/ttv.jl).
    t_phot_sorted::Vector{Float64}
    # CCF profiles fitted as Gaussian line components (src/ccf_data.jl), one
    # set per instrument. Empty unless given; the CCF term is then exactly zero.
    ccf::Vector{CCFSet}
end

"""
    Data(; t_rv, rv, rv_err, rv_inst=nothing, t_ref=nothing)

Keyword constructor for `Data`. Validates:

- all RV arrays have matching length
- `rv_err` entries are strictly positive
- `rv_inst` entries are strictly positive (1-based)

`rv_inst` defaults to all-ones (single-instrument case). `t_ref`
defaults to `median(t_rv)`, which is the standard choice for Mo-anchor
parametrizations (centers the mean-anomaly zero near the middle of
the baseline, improving sampling conditioning).

# Hipparcos scan geometry (`hip_scans`)

The Hipparcos epoch of the HGCA and G23H models is a five-parameter fit of the
reflex through the Hipparcos scans, so it needs their geometry: epochs, scan
angles, parallax factors and abscissa errors. With `iad` holding Hipparcos IAD
(at least five scans inside `HIPPARCOS_WINDOW`) those scans are used, and
nothing else is needed. Without it, pass `hip_scans = fetch_hip_iad(hip_id)`
(an `IADData`; its abscissae are never read, so it adds nothing to the
likelihood). Passing both is allowed only when they are the same scans. With
neither, `Data` warns and the Hipparcos epoch is modelled instantaneously.
`run_job` and the `fit_*` functions fetch the scans by the catalogue's
`hip_id` themselves.
"""
function Data(;
    t_rv::AbstractVector{<:Real} = Float64[],
    rv::AbstractVector{<:Real} = Float64[],
    rv_err::AbstractVector{<:Real} = Float64[],
    rv_inst::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    rv_comp::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    t_ref::Union{Nothing, Real} = nothing,
    indicators::Union{Nothing, Dict{String, <:AbstractVector{<:Real}}} = nothing,
    indicator_errs::Union{Nothing, Dict{String, <:AbstractVector{<:Real}}} = nothing,
    normalize_indicators::Bool = true,
    t_phot::AbstractVector{<:Real} = Float64[],
    flux::AbstractVector{<:Real} = Float64[],
    flux_err::AbstractVector{<:Real} = Float64[],
    phot_inst::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    exposure_times::AbstractVector{<:Real} = Float64[],
    relastrom::Union{Nothing, RelAstromData} = nothing,
    hgca::Union{Nothing, HGCAData} = nothing,
    iad::Union{Nothing, IADData} = nothing,
    gost::Union{Nothing, GOSTData} = nothing,
    g23h::Union{Nothing, G23HData} = nothing,
    gaia_dr3::Union{Nothing, GaiaDR3Data} = nothing,
    tomo::Vector{TomoNight} = TomoNight[],
    hip_scans::Union{Nothing, IADData} = nothing,
    transit_times::AbstractVector{TransitTimeSet} = TransitTimeSet[],
    ccf::AbstractVector{CCFSet} = CCFSet[],
)
    n_rv = length(t_rv)
    n_phot = length(t_phot)
    tt_sets = Vector{TransitTimeSet}(transit_times)
    tt_planets = [s.planet for s in tt_sets]
    length(unique(tt_planets)) == length(tt_planets) || throw(ArgumentError(
        "transit times: more than one set for planet(s) " *
        "$(sort!(unique(filter(p -> count(==(p), tt_planets) > 1, tt_planets)))). " *
        "Merge each planet's transit times into one set: the epochs of a set are " *
        "counted from one epoch_zero."))
    # Mid-transit times in the likelihood, every planet pooled.
    tt_used_t = reduce(vcat, (s.tc[s.used] for s in tt_sets); init = Float64[])
    has_astrom = relastrom !== nothing || hgca !== nothing ||
                 iad !== nothing || gost !== nothing ||
                 g23h !== nothing || gaia_dr3 !== nothing
    # Residual maps are data in their own right: a Doppler-shadow fit has no
    # velocities, and giving it a dummy RV point to get past this check left a
    # constant offset in the likelihood and three prior-only parameters.
    ccf_sets = Vector{CCFSet}(ccf)
    tags = [s.tag for s in ccf_sets]
    length(unique(tags)) == length(tags) || throw(ArgumentError(
        "CCF sets must have unique tags; got $(tags). They name the parameters"))
    (n_rv > 0 || n_phot > 0 || has_astrom || !isempty(tomo) || !isempty(tt_used_t) ||
     !isempty(ccf_sets)) ||
        throw(ArgumentError(
        "Data requires at least one RV, photometric, astrometric, tomographic, " *
        "transit-time or CCF observation"))

    # --- RV validation ---
    if n_rv > 0
        length(rv) == n_rv || throw(ArgumentError(
            "rv length ($(length(rv))) must match t_rv length ($n_rv)"))
        length(rv_err) == n_rv || throw(ArgumentError(
            "rv_err length ($(length(rv_err))) must match t_rv length ($n_rv)"))
        all(e -> e > 0, rv_err) || throw(ArgumentError(
            "rv_err entries must be strictly positive"))
    end

    if rv_inst === nothing
        rv_inst_vec = ones(Int, n_rv)
    else
        length(rv_inst) == n_rv || throw(ArgumentError(
            "rv_inst length ($(length(rv_inst))) must match t_rv length ($n_rv)"))
        rv_inst_vec = Vector{Int}(rv_inst)
    end
    if n_rv > 0
        all(i -> i > 0, rv_inst_vec) || throw(ArgumentError(
            "rv_inst entries must be 1-based positive integers"))
    end

    # rv_comp: stellar component per RV point (1=primary, 2=secondary). Defaults to
    # all-primary (the single-star case); only an SB2 fit sets component 2.
    if rv_comp === nothing
        rv_comp_vec = ones(Int, n_rv)
    else
        length(rv_comp) == n_rv || throw(ArgumentError(
            "rv_comp length ($(length(rv_comp))) must match t_rv length ($n_rv)"))
        rv_comp_vec = Vector{Int}(rv_comp)
        all(c -> c == 1 || c == 2, rv_comp_vec) || throw(ArgumentError(
            "rv_comp entries must be 1 (primary) or 2 (secondary)"))
    end

    # t_ref: use RV median if available, else photometry median, else the
    # median measured transit time, else fall back to astrometry epoch
    # (IAD → GOST → relAST → HGCA → G23H).
    # The numeric value matters only as the zero-point for Mo/Tp time
    # anchors; for astrometry-only fits any astrometric epoch suffices.
    if t_ref !== nothing
        t_ref_val = Float64(t_ref)
    elseif n_rv > 0
        t_ref_val = Float64(median(t_rv))
    elseif n_phot > 0
        t_ref_val = Float64(median(t_phot))
    elseif !isempty(tt_used_t)
        t_ref_val = Float64(median(tt_used_t))
    elseif iad !== nothing && n_iad(iad) > 0
        t_ref_val = Float64(median(iad.t))
    elseif gost !== nothing && n_gost(gost) > 0
        t_ref_val = Float64(median(gost.t))
    elseif relastrom !== nothing && n_relast(relastrom) > 0
        t_ref_val = Float64(median(relastrom.t))
    elseif hgca !== nothing
        t_ref_val = Float64(hgca.epochs[2])  # HG midpoint
    elseif g23h !== nothing
        t_ref_val = Float64(g23h.epochs[3])  # DR2 epoch (~mid)
    elseif gaia_dr3 !== nothing
        t_ref_val = Float64(gaia_dr3.t_ref)
    elseif !isempty(tomo)
        t_ref_val = Float64(median(reduce(vcat, (nt.t for nt in tomo))))
    elseif !isempty(ccf_sets)
        t_ref_val = Float64(median(reduce(vcat, (s.t for s in ccf_sets))))
    else
        throw(ArgumentError("no observations to derive t_ref from"))
    end

    # --- Indicator validation + per-instrument time derivatives ---
    ind_dict = Dict{String, Vector{Float64}}()
    if indicators !== nothing
        for (name, vals) in indicators
            length(vals) == n_rv || throw(ArgumentError(
                "indicator \"$name\" length ($(length(vals))) must match t_rv length ($n_rv)"))
            ind_dict[name] = Vector{Float64}(vals)
        end
    end
    ind_err_dict = Dict{String, Vector{Float64}}()
    if indicator_errs !== nothing
        for (name, errs) in indicator_errs
            haskey(ind_dict, name) || throw(ArgumentError(
                "indicator_errs entry \"$name\" has no matching indicator values"))
            length(errs) == n_rv || throw(ArgumentError(
                "indicator_errs \"$name\" length ($(length(errs))) must match t_rv length ($n_rv)"))
            all(e -> e > 0, errs) || throw(ArgumentError(
                "indicator_errs \"$name\" entries must be strictly positive"))
            ind_err_dict[name] = Vector{Float64}(errs)
        end
    end
    # Per-instrument central-difference derivatives for FF'-style
    # ActivityDecorrelation. Computed once at construction; consumed by
    # ActivityDecorrelation(derivative=true). NaN-safe: if any neighbor
    # is NaN, falls back to one-sided; if both are NaN, derivative is NaN.
    # ---- Activity-indicator normalisation (DEFAULT) -------------------------
    #
    # Per instrument: subtract the median, divide by the RMS. This is the
    # convention of Vines et al. 2023 (MNRAS 518, 2627), Table 7 footnote:
    # "Activity indices were mean subtracted and normalized to their RMS."
    #
    # It matters because the ActivityDecorrelation term is pred += C * indicator
    # on the RAW value, so without it C carries the indicator's units and its
    # prior means something different for every instrument and every channel.
    # Measured on HD 18599, scaling each indicator to its instrument's RV
    # amplitude instead of its own RMS moved the recovered semi-amplitude from
    # 11.9 m/s to 5.9 -- it let a 7-point instrument's regression absorb the
    # planet. With unit-RMS regressors C is simply the m/s amplitude of the
    # activity term and is comparable across instruments and channels.
    #
    # normalize_indicators=false restores the raw values for a caller who has
    # already scaled them.
    # The SAME per-instrument RMS must be applied to `indicator_errs`. The
    # values and their uncertainties live in one unit system: rescaling the
    # values alone silently changes the signal-to-noise of the indicator
    # block. ActivityDecorrelation never reads the errors so it cannot see
    # the difference, but ActivityGP scores the indicators as a data block
    # with their own noise (sigma^2 = err^2 + jitter^2) and is corrupted by
    # it -- an indicator whose RMS is 0.05 gets its values multiplied by 20
    # while its errors stay put, so the GP is asked to fit 20x-inflated data
    # at the original precision.
    if normalize_indicators && !isempty(ind_dict)
        for (name, vals) in ind_dict
            v = Vector{Float64}(vals)
            e = get(ind_err_dict, name, nothing)
            for ins in unique(rv_inst_vec)
                idx = findall(==(ins), rv_inst_vec)
                fin = filter(k -> isfinite(v[k]), idx)
                isempty(fin) && continue
                mu = median(@view v[fin])
                for k in fin; v[k] -= mu; end
                rms = sqrt(sum(abs2, @view v[fin]) / length(fin))
                rms > 0 || continue
                for k in fin; v[k] /= rms; end
                # Divide the errors by the same factor. Scale only -- the
                # median subtraction is a shift and does not affect them.
                e === nothing || for k in idx; e[k] /= rms; end
            end
            ind_dict[name] = v
            e === nothing || (ind_err_dict[name] = e)
        end
    end

    deriv_dict = Dict{String, Vector{Float64}}()
    for (name, vals) in ind_dict
        d = fill(NaN, n_rv)
        for ins in unique(rv_inst_vec)
            idx_ins = findall(==(ins), rv_inst_vec)
            length(idx_ins) > 1 || continue
            ord = sortperm(Float64.(t_rv[idx_ins]))
            sorted_idx = idx_ins[ord]
            ts = Float64.(t_rv[sorted_idx])
            vs = vals[sorted_idx]
            n_s = length(sorted_idx)
            for k in 1:n_s
                isfinite(vs[k]) || continue
                if k == 1
                    isfinite(vs[2]) || continue
                    d[sorted_idx[k]] = (vs[2] - vs[k]) / (ts[2] - ts[1])
                elseif k == n_s
                    isfinite(vs[k-1]) || continue
                    d[sorted_idx[k]] = (vs[k] - vs[k-1]) / (ts[k] - ts[k-1])
                elseif isfinite(vs[k-1]) && isfinite(vs[k+1])
                    d[sorted_idx[k]] = (vs[k+1] - vs[k-1]) / (ts[k+1] - ts[k-1])
                elseif isfinite(vs[k+1])
                    d[sorted_idx[k]] = (vs[k+1] - vs[k]) / (ts[k+1] - ts[k])
                elseif isfinite(vs[k-1])
                    d[sorted_idx[k]] = (vs[k] - vs[k-1]) / (ts[k] - ts[k-1])
                end
            end
        end
        deriv_dict[name] = d
    end

    # --- Photometry validation ---
    if n_phot > 0
        length(flux) == n_phot || throw(ArgumentError(
            "flux length ($(length(flux))) must match t_phot length ($n_phot)"))
        length(flux_err) == n_phot || throw(ArgumentError(
            "flux_err length ($(length(flux_err))) must match t_phot length ($n_phot)"))
        all(e -> e > 0, flux_err) || throw(ArgumentError(
            "flux_err entries must be strictly positive"))
    end

    if phot_inst === nothing
        phot_inst_vec = ones(Int, n_phot)
    else
        length(phot_inst) == n_phot || throw(ArgumentError(
            "phot_inst length must match t_phot length"))
        phot_inst_vec = Vector{Int}(phot_inst)
    end

    # Exposure times (days): empty ⇒ instantaneous model. When provided, must be
    # one-per-cadence and positive (a non-finite/zero entry just disables
    # supersampling for that point downstream).
    if isempty(exposure_times)
        exp_vec = Float64[]
    else
        length(exposure_times) == n_phot || throw(ArgumentError(
            "exposure_times length ($(length(exposure_times))) must match t_phot length ($n_phot)"))
        exp_vec = Vector{Float64}(exposure_times)
    end

    # Fail-loud: GOST is a scan-geometry AUXILIARY, not a standalone
    # measurement — `gost_log_likelihood` returns 0, and GOST only does
    # anything when it window-averages the Gaia epoch of a paired product
    # (HGCA / Gaia DR3 / G23H). Supplied alone (or only with IAD, whose Gaia
    # joint needs `gaia_dr3`), it contributes NOTHING. Warn so it can't be
    # silently inert.
    # An IAD source that stores O−C residuals must say what they are
    # residuals AGAINST before it is fitted. The likelihood samples the
    # parallax and models ϖ − ref_params[3] (`_iad_residuals!`), so against an
    # all-zero reference the catalogue parallax, already subtracted from the
    # data, is subtracted again and the IAD drags ϖ toward zero. Next to the
    # published Gaia five-parameter solution, which plays the role of a
    # second, absolute instrument, a shared ϖ and μ would also compare a
    # correction to a full quantity, and the fit silently absorbs the ~μ·Δt
    # difference into the orbit. Measured on a 180 mas/yr star: ln L of
    # −9.2e6 where −109 is correct, and a spurious mass.
    if iad !== nothing
        for m in 1:n_iad_inst(iad)
            (iad.abscissa_kind[m] === :residual &&
             all(==(0.0), iad.ref_params[m])) && throw(ArgumentError(
                "Data: IAD instrument $m stores :residual abscissae with an " *
                "all-zero ref_params. Residuals are fitted against the " *
                "catalogue solution they were formed against: the likelihood " *
                "models ϖ − ref_params[3], so without it the catalogue " *
                "parallax is subtracted twice. Supply that solution " *
                "(`load_hip_iad(path; ref_params = (0, 0, ϖ, μα*, μδ))`, or " *
                "`ref_params` in a config's `values`), or mark the abscissae " *
                "`:absolute` if nothing was subtracted. `fetch_hip_iad` reads " *
                "it from the record's own header."))
        end
    end

    # A published Gaia five-parameter solution is a SUMMARY of the very
    # transits a Gaia epoch-astrometry instrument supplies. Fitting both
    # counts that data twice and tightens the posterior by a factor nothing
    # earned. The joint path also ties the DR3 frame to IAD instrument 1,
    # which is the Hipparcos frame in the intended two-mission setup.
    if iad !== nothing && n_iad_inst(iad) > 1 && gaia_dr3 !== nothing
        @warn("multi-instrument IAD ($(n_iad_inst(iad)) instruments) supplied " *
              "together with a published Gaia DR3 five-parameter solution. If " *
              "one of those instruments is Gaia epoch astrometry its transits " *
              "are now counted TWICE — once per transit and once through the " *
              "catalogue solution that was fitted to them. Drop `gaia_dr3` " *
              "unless the epoch instruments are from a different mission. The " *
              "DR3 constraint is tied to IAD instrument 1's zero point.",
              maxlog = 1)
    end

    if gost === nothing && (hgca !== nothing || g23h !== nothing)
        @info("HGCA/G23H without a GOST scan plan: the Gaia epoch is averaged " *
              "over a UNIFORM NOMINAL scan grid across the release's observing " *
              "window (DR3 2014-07-25 to 2017-05-28; DR2 to 2016-05-23), every " *
              "$(NOMINAL_GAIA_SCAN_STEP_DAYS) d. Companions with P shorter than " *
              "about twice that step average out. GOST (`fetch_gost`) gives the " *
              "exact scan law; supply `gost` for it.", maxlog = 1)
    end

    if gost !== nothing && hgca === nothing && gaia_dr3 === nothing && g23h === nothing
        @warn("GOST provided without a Gaia-epoch measurement (HGCA / Gaia DR3 / " *
              "G23H) — GOST is INERT here (it only window-averages those). The fit " *
              "will use IAD/other sources only; add `hgca=fetch_hgca(hip_id)` (or a " *
              "Gaia DR3 solution) to actually constrain the Gaia epoch.", maxlog = 1)
    end

    # The Gaia-epoch models fit only the transits inside the release's own
    # window (GAIA_DR3_WINDOW; G23H's DR2 epoch GAIA_DR2_WINDOW). Fewer than
    # five there and `gost_5param_fit` returns zeros — a zero orbital reflex
    # at the Gaia epoch, which reads as a measurement. Refuse it here.
    if gost !== nothing && (hgca !== nothing || gaia_dr3 !== nothing || g23h !== nothing)
        _require_gost_window(gost, GAIA_DR3_WINDOW, "DR3")
        g23h !== nothing && _require_gost_window(gost, GAIA_DR2_WINDOW, "DR2")
    end

    hip_geom = _resolve_hip_scans(iad, hip_scans, hgca !== nothing || g23h !== nothing)

    t_phot_vec = Vector{Float64}(t_phot)
    return Data(
        Vector{Float64}(t_rv), Vector{Float64}(rv),
        Vector{Float64}(rv_err), rv_inst_vec, rv_comp_vec,
        t_ref_val, ind_dict, ind_err_dict, deriv_dict,
        t_phot_vec, Vector{Float64}(flux),
        Vector{Float64}(flux_err), phot_inst_vec, exp_vec,
        relastrom, hgca, iad, gost, g23h, gaia_dr3, tomo, hip_geom, tt_sets,
        issorted(t_phot_vec) ? t_phot_vec : sort(t_phot_vec),
        ccf_sets,
    )
end

"""
    _resolve_hip_scans(iad, hip_scans, needed) -> IADData or nothing

The Hipparcos scan geometry the HGCA / G23H models fit their Hipparcos epoch
through (`needed` says whether either catalogue is present).

- `iad` with at least five scans inside `HIPPARCOS_WINDOW` is used as it is:
  a fit that has the Hipparcos IAD in its likelihood models the catalogue's
  Hipparcos epoch through the very same scans. The window keeps a merged Gaia
  instrument's transits out of that fit.
- Otherwise an explicit `hip_scans`, which must have five scans in the window.
- Both: allowed only when they are the same scans (epochs and angles inside
  the window); two different scan sets for one mission is a mistake.
- Neither: `nothing`, with a warning — the Hipparcos epoch is then
  instantaneous, which is not what the catalogue measured.
"""
function _resolve_hip_scans(iad, hip_scans, needed::Bool)
    if !needed
        hip_scans === nothing || @warn("`hip_scans` given without an HGCA or G23H " *
            "row — the Hipparcos scan geometry is INERT here (it only models the " *
            "Hipparcos epoch of those catalogues) and adds nothing to the likelihood.",
            maxlog = 1)
        return hip_scans
    end
    win = HIPPARCOS_WINDOW
    iad_ok = iad !== nothing && _n_in_window(iad.t, win) >= 5
    if hip_scans !== nothing
        n_in = _n_in_window(hip_scans.t, win)
        n_in >= 5 || throw(ArgumentError(
            "hip_scans has $n_in scan(s) inside the Hipparcos mission (MJD " *
            "$(win[1]) to $(win[2])); the Hipparcos epoch is a five-parameter fit " *
            "through them and needs at least 5. Pass `fetch_hip_iad(hip_id)`."))
        if iad_ok
            inwin(d) = findall(t -> win[1] <= t <= win[2], d.t)
            a, b = inwin(iad), inwin(hip_scans)
            (iad.t[a] == hip_scans.t[b] && iad.psi[a] == hip_scans.psi[b]) ||
                throw(ArgumentError(
                    "`iad` and `hip_scans` carry DIFFERENT Hipparcos scans. The " *
                    "Hipparcos epoch of the HGCA/G23H model is fitted through the " *
                    "IAD's own scans when the fit has IAD; drop `hip_scans`."))
            return iad
        end
        return hip_scans
    end
    iad_ok && return iad
    @warn("HGCA / G23H supplied without Hipparcos scan geometry (no `iad` with " *
          "Hipparcos scans, no `hip_scans`): the Hipparcos proper motion and the " *
          "Hipparcos position in the Hipparcos–Gaia difference are modelled as the " *
          "INSTANTANEOUS reflex at the Hipparcos epoch, not as the catalogue's " *
          "five-parameter fit over the mission. Pass `hip_scans = " *
          "fetch_hip_iad(hip_id)` to model them as measured.", maxlog = 1)
    return nothing
end

"""
    n_rv(data) -> Int

Number of RV observations.
"""
n_rv(data::Data) = length(data.t_rv)
"""
    n_phot(data::Data) -> Int

Number of photometric data points across every instrument, at native
cadence (the light curve is never binned).
"""
n_phot(data::Data) = length(data.t_phot)
"""
    n_transit_times(data::Data) -> Int

Number of measured mid-transit times in the likelihood, every planet together
(points excluded by `use`, `exclude_sources` or the photometry overlap left out).
"""
n_transit_times(data::Data) = sum(n_tt, data.transit_times; init = 0)

"""
    transit_time_set(data, k) -> Union{TransitTimeSet, Nothing}

Planet `k`'s measured transit times, or `nothing` when it has none.
"""
function transit_time_set(data::Data, k::Integer)
    for s in data.transit_times
        s.planet == k && return s
    end
    return nothing
end
function _require_gost_window(gost::GOSTData, win::Tuple{Real, Real}, release::String)
    n_in = count(t -> win[1] <= t <= win[2], gost.t)
    n_in >= 5 || throw(ArgumentError(
        "GOST has $n_in transit(s) inside the Gaia $release window (MJD $(win[1]) to " *
        "$(win[2])); the Gaia $release proper motion is fitted through that window " *
        "alone and needs at least 5. Fetch the scan plan over it (`fetch_gost`'s " *
        "default range covers it)."))
    return nothing
end

"""
    has_astrometry(data) -> Bool

True if the `Data` carries any astrometry payload (relative, HGCA,
Hipparcos IAD, or Gaia GOST).
"""
has_astrometry(data::Data) = data.relastrom !== nothing ||
                             data.hgca !== nothing ||
                             data.iad !== nothing ||
                             data.gost !== nothing ||
                             data.g23h !== nothing ||
                             data.gaia_dr3 !== nothing
