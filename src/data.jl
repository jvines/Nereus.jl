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
)
    n_rv = length(t_rv)
    n_phot = length(t_phot)
    has_astrom = relastrom !== nothing || hgca !== nothing ||
                 iad !== nothing || gost !== nothing ||
                 g23h !== nothing || gaia_dr3 !== nothing
    (n_rv > 0 || n_phot > 0 || has_astrom) || throw(ArgumentError(
        "Data requires at least one RV, photometric, or astrometric observation"))

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

    # t_ref: use RV median if available, else photometry median, else
    # fall back to astrometry epoch (IAD → GOST → relAST → HGCA → G23H).
    # The numeric value matters only as the zero-point for Mo/Tp time
    # anchors; for astrometry-only fits any astrometric epoch suffices.
    if t_ref !== nothing
        t_ref_val = Float64(t_ref)
    elseif n_rv > 0
        t_ref_val = Float64(median(t_rv))
    elseif n_phot > 0
        t_ref_val = Float64(median(t_phot))
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
    if gost !== nothing && hgca === nothing && gaia_dr3 === nothing && g23h === nothing
        @warn("GOST provided without a Gaia-epoch measurement (HGCA / Gaia DR3 / " *
              "G23H) — GOST is INERT here (it only window-averages those). The fit " *
              "will use IAD/other sources only; add `hgca=fetch_hgca(hip_id)` (or a " *
              "Gaia DR3 solution) to actually constrain the Gaia epoch.", maxlog = 1)
    end

    return Data(
        Vector{Float64}(t_rv), Vector{Float64}(rv),
        Vector{Float64}(rv_err), rv_inst_vec, rv_comp_vec,
        t_ref_val, ind_dict, ind_err_dict, deriv_dict,
        Vector{Float64}(t_phot), Vector{Float64}(flux),
        Vector{Float64}(flux_err), phot_inst_vec, exp_vec,
        relastrom, hgca, iad, gost, g23h, gaia_dr3, tomo,
    )
end

"""
    n_rv(data) -> Int

Number of RV observations.
"""
n_rv(data::Data) = length(data.t_rv)
n_phot(data::Data) = length(data.t_phot)
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
