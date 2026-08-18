"""
    _locor_cycles(t, P_rot, t0) -> NamedTuple

Assign each timestamp in `t` to a rotation cycle and a within-cycle phase,
as used by the LOCoR detrender (Rizzuto et al. 2017).

Given a rotation period `P_rot` and a reference epoch `t0`, returns a
`NamedTuple` `(; phase, cycle_id)` where

  - `phase[i] = mod(t[i] - t0, P_rot) / P_rot`  lies in `[0, 1)`, the
    fractional position within the rotation cycle.
  - `cycle_id[i] = floor(Int, (t[i] - t0) / P_rot)`  is the integer index of
    the rotation cycle the sample falls in (0 for the cycle starting at `t0`).
"""
function _locor_cycles(t::AbstractVector{<:Real}, P_rot::Real, t0::Real)
    phase = @. mod(t - t0, P_rot) / P_rot
    cycle_id = @. floor(Int, (t - t0) / P_rot)
    return (; phase, cycle_id)
end

"""
    _interp_constant(xq, x, y) -> Vector{Float64}

Linear interpolation of `(x, y)` evaluated at the query points `xq`, with
**constant extrapolation** outside the range of `x`: query points below
`x[1]` take `y[1]`, and query points above `x[end]` take `y[end]`.

`x` is assumed sorted ascending (donor phases are sorted in `[0, 1)`). No
`Interpolations.jl` dependency is used — a base `searchsortedlast` plus a
manual linear blend is sufficient and keeps the preprocessing layer
dependency-free.
"""
function _interp_constant(xq::AbstractVector{<:Real}, x::AbstractVector{<:Real},
                          y::AbstractVector{<:Real})
    n = length(x)
    out = Vector{Float64}(undef, length(xq))
    @inbounds for (k, q) in enumerate(xq)
        if q <= x[1]
            out[k] = y[1]
        elseif q >= x[n]
            out[k] = y[n]
        else
            j = searchsortedlast(x, q)          # x[j] <= q < x[j+1]
            x0, x1 = x[j], x[j+1]
            w = (q - x0) / (x1 - x0)
            out[k] = (1 - w) * y[j] + w * y[j+1]
        end
    end
    return out
end

# Build the phase-aligned donor matrix `olc` (`n_donors × n_pts`): each donor
# cycle's `phase → flux` curve is interpolated (constant extrapolation, see
# `_interp_constant`) onto the target cycle's native phase samples
# `target_phase`. Shared by `_rcomb_cycle` and `_rcomb_cycle_clipped`.
function _build_olc(target_phase::Vector{Float64},
                    donor_phases::Vector{Vector{Float64}},
                    donor_fluxes::Vector{Vector{Float64}})
    n_donors = length(donor_phases)
    n_pts = length(target_phase)
    olc = Matrix{Float64}(undef, n_donors, n_pts)
    @inbounds for d in 1:n_donors
        olc[d, :] = _interp_constant(target_phase, donor_phases[d], donor_fluxes[d])
    end
    return olc
end

"""
    _rcomb_cycle(target_phase, target_flux, donor_phases, donor_fluxes) -> baseline

Predict the rotation-baseline of a single target cycle as a least-squares
linear combination of the "donor" cycles, following the LOCoR `rcomb`
construction (Rizzuto et al. 2017; arizzuto/Notch_and_LOCoR).

Each donor cycle (its own `phase → flux` curve) is interpolated onto the
target cycle's native phase samples `target_phase` (constant extrapolation
where a donor does not cover a target phase). Stacking the phase-aligned
donors as the rows of an `n_donors × n_pts` matrix `olc`, the donor weights
solve the normal equations and the baseline is the weighted donor sum:

    A       = olc * olc'
    b       = olc * target_flux
    coeffs  = A \\ b               # backslash, not inv(A): same math, better conditioned
    baseline = olc' * coeffs

Returns `baseline`, a length-`n_pts` vector aligned with `target_phase`.

Assumes the caller has already median-normalized each cycle's flux
(both target and donors); normalization is handled by a later task.
"""
function _rcomb_cycle(target_phase::Vector{Float64}, target_flux::Vector{Float64},
                      donor_phases::Vector{Vector{Float64}},
                      donor_fluxes::Vector{Vector{Float64}})
    olc = _build_olc(target_phase, donor_phases, donor_fluxes)
    A = olc * olc'
    b = olc * target_flux
    coeffs = A \ b
    baseline = olc' * coeffs
    return baseline
end

"""
    _rcomb_cycle_clipped(target_phase, target_flux, donor_phases, donor_fluxes;
                         niter=3, clip_sigma=3.5, init_outlier=nothing)
        -> (; baseline, outlier)

Iterating, MAD sigma-clipping variant of [`_rcomb_cycle`](@ref), robust to
transits/flares contaminating either the target or the donor cycles
(following LOCoR `rcomb`; arizzuto/Notch_and_LOCoR).

`init_outlier`, when given (a length-`n_pts` `Bool` vector), pre-seeds the
outlier mask: those target points are excluded from the LSQ fit from the very
first iteration (the `cleanmask`/in-transit defense — see [`detrend_locor`](@ref)
`transit_mask`). They are OR-ed with the σ-clip results, so they stay flagged
but still receive a predicted baseline (the clean donor extrapolation) and
appear in `flux_detrended`. With `init_outlier=nothing` the behavior is
identical to the original (all points start active).

Starting from all (non-pre-seeded) target points active, each iteration:

  1. Solves the donor-weight normal equations using only the **active**
     (non-outlier) target columns of `olc` and `target_flux[active]`.
  2. Evaluates `baseline = olc' * coeffs` over **all** target points.
  3. Forms residuals over active points, `r = target_flux ./ baseline .- 1`,
     a robust scale `mad = median(abs.(r_active)) / 0.6745`, and flags any
     active point with `abs(r) > clip_sigma * mad` as a new outlier.

Points once flagged stay flagged. Iteration stops early when no new outlier
appears, or after `niter` iterations. Donors are always used at full length;
only the target points participating in the LSQ fit are clipped.

**Underdetermination guard:** if the number of active target points drops
below `n_donors`, the normal equations are underdetermined; iteration stops
and the last good full-length `baseline` is returned (no crash).

Returns `(; baseline, outlier)` where `baseline` is the length-`n_pts`
predicted rotation baseline aligned with `target_phase`, and `outlier` is a
`Vector{Bool}` over the target points (`true` = clipped).

Assumes each cycle's flux has already been median-normalized.
"""
function _rcomb_cycle_clipped(target_phase::Vector{Float64}, target_flux::Vector{Float64},
                              donor_phases::Vector{Vector{Float64}},
                              donor_fluxes::Vector{Vector{Float64}};
                              niter::Int = 3, clip_sigma::Real = 3.5,
                              init_outlier::Union{Nothing,AbstractVector{Bool}} = nothing)
    n_donors = length(donor_phases)
    n_pts = length(target_phase)

    olc = _build_olc(target_phase, donor_phases, donor_fluxes)

    outlier = falses(n_pts)
    if init_outlier !== nothing
        length(init_outlier) == n_pts || throw(ArgumentError(
            "init_outlier length $(length(init_outlier)) != n_pts $n_pts"))
        outlier .|= init_outlier
    end

    # Initial fit over all points (guards the under-determined edge case for
    # the very first solve too).
    function fit_baseline(active)
        Aolc = olc[:, active]
        A = Aolc * Aolc'
        b = Aolc * target_flux[active]
        coeffs = A \ b
        return olc' * coeffs
    end

    # Initial fit excludes any pre-seeded (in-transit) points already in
    # `outlier`, so the first baseline is driven only by clean target points.
    initial_active = .!outlier
    baseline = count(initial_active) >= n_donors ? fit_baseline(initial_active) :
                                                   fit_baseline(trues(n_pts))

    for _ in 1:niter
        active = .!outlier
        # Underdetermination guard: need at least n_donors active points.
        count(active) < n_donors && break

        baseline = fit_baseline(active)

        r = target_flux ./ baseline .- 1.0
        r_active = r[active]
        mad = median(abs.(r_active)) / 0.6745

        # Degenerate scale (perfect fit) → nothing more to clip.
        mad <= 0.0 && break

        new_outlier = active .& (abs.(r) .> clip_sigma * mad)
        any(new_outlier) || break   # converged: no new outliers
        outlier .|= new_outlier
    end

    return (; baseline, outlier)
end

"""
    _locor_segment!(flux_detrended, trend, outlier, cycle_id_out, used_notch,
                    t, flux, flux_err, seg, P_rot, t0,
                    max_donor_cycles, niter, clip_sigma, gap_size;
                    transit_mask=nothing)

Detrend a single contiguous photometric `seg`ment (a `UnitRange` into the
full-length arrays) with LOCoR, writing results in place at the `seg`
indices of the full-length output arrays `flux_detrended`, `trend`,
`outlier`, `cycle_id_out`, `used_notch`.

**Normalization-cancels reasoning.** LOCoR operates on per-cycle
median-normalized flux: for a cycle, `norm_flux = flux ./ median(flux_cycle)`.
`_rcomb_cycle_clipped` predicts the normalized rotation baseline
`norm_baseline`. The detrended flux is then simply
`flux_detrended = norm_flux ./ norm_baseline` — the cycle median appears in
both numerator and denominator and cancels, so we never multiply the median
back. For *reporting only*, the raw rotation baseline (written to the output
`trend` array) is reconstructed as `trend = norm_baseline .* median(flux_cycle)`.

**Segment-level fallback rule.** A meaningful, stable rotational combination
needs at least one target cycle plus two donor cycles, i.e.
`length(unique(cycle_id)) >= 3`. If the segment has fewer cycles, the *whole*
segment is detrended with [`detrend_notch`](@ref) instead (`used_notch[seg]
.= true`); otherwise the segment takes the LOCoR path (`used_notch[seg]
.= false`).

**Per-cycle degenerate fallback.** Within a LOCoR segment, the donor count for
a target cycle is capped at `n_target_pts - 4` (to keep the LSQ
over-determined) and at `length(ucyc) - 1`. A target cycle with ≤ 4 points
(e.g. a 1-point trailing partial cycle at the end of a segment) cannot be
given even one donor; rather than discard the whole segment, that single cycle
is passed through flat (`flux_detrended = norm_flux`, `trend =
median(flux_cycle)`, `outlier = false`) while the rest of the segment is still
detrended with LOCoR. This keeps a normal segment fully on the LOCoR path even
when its length is not an exact integer number of rotation cycles.

**`transit_mask` (cleanmask).** When supplied (per-cadence `Bool` over the full
LC, viewed onto `seg`), it implements the commensurate-period transit defense
described in [`detrend_locor`](@ref): each donor cycle's in-transit points are
dropped before its curve enters `_build_olc`/`_interp_constant` (a donor with
<2 surviving points is skipped; if all donors collapse, the target cycle is
passed through flat), and the target cycle's in-transit points are pre-seeded as
outliers via `init_outlier` so they do not drive the per-cycle LSQ fit. They
still receive a predicted baseline and are reported in `outlier`.
"""
function _locor_segment!(flux_detrended, trend, outlier, cycle_id_out, used_notch,
                         t, flux, flux_err, seg::UnitRange,
                         P_rot, t0, max_donor_cycles, niter, clip_sigma, gap_size;
                         transit_mask::Union{Nothing,AbstractVector{Bool}} = nothing)
    tv = view(t, seg)
    fluxv = view(flux, seg)
    errv = view(flux_err, seg)
    # Per-segment in-transit mask (Bool over `seg` points), or all-false.
    tmaskv = transit_mask === nothing ? falses(length(seg)) :
                                        collect(Bool, view(transit_mask, seg))

    cyc = _locor_cycles(tv, P_rot, t0)
    cycle_id = cyc.cycle_id
    phase = cyc.phase
    ucyc = unique(cycle_id)

    # --- fallback helper: detrend the whole segment with notch ---
    function notch_fallback!()
        res = detrend_notch(collect(tv), collect(fluxv), collect(errv); gap_size = gap_size)
        @inbounds for (k, i) in enumerate(seg)
            flux_detrended[i] = res.flux_detrended[k]
            trend[i] = res.trend[k]
            outlier[i] = res.outlier[k]
            cycle_id_out[i] = cycle_id[k]
            used_notch[i] = true
        end
        return nothing
    end

    # FALLBACK CONDITION: need ≥1 target + ≥2 donors for a stable combination.
    if length(ucyc) < 3
        notch_fallback!()
        return nothing
    end

    # Per-cycle median-normalize. Store normalized flux aligned with `seg`,
    # plus the per-cycle median (for raw-baseline reporting).
    norm_flux = similar(fluxv, Float64)
    cyc_median = Dict{Int,Float64}()
    for c in ucyc
        idx = findall(==(c), cycle_id)
        m = median(@view fluxv[idx])
        cyc_median[c] = m
        @inbounds for j in idx
            norm_flux[j] = fluxv[j] / m
        end
    end

    # Per-cycle phase grids in normalized units (needed as donor curves),
    # plus the per-cycle in-transit mask (aligned with cyc_phase/cyc_norm).
    cyc_phase = Dict(c => Float64.(phase[findall(==(c), cycle_id)]) for c in ucyc)
    cyc_norm = Dict(c => norm_flux[findall(==(c), cycle_id)] for c in ucyc)
    cyc_mask = Dict(c => tmaskv[findall(==(c), cycle_id)] for c in ucyc)

    cap = max_donor_cycles === nothing ? typemax(Int) : max_donor_cycles

    # LOCoR path.
    for c in ucyc
        target_idx_local = findall(==(c), cycle_id)        # indices into `seg`
        target_phase = cyc_phase[c]
        target_norm = cyc_norm[c]
        n_target_pts = length(target_phase)

        # Donor cycles: the other cycles, nearest in cycle number first.
        # Cap so the LSQ stays over-determined (`n_target_pts - 4`) and to the
        # number of available other cycles.
        others = sort(filter(!=(c), ucyc); by = d -> abs(d - c))
        n_allowed = min(cap, n_target_pts - 4, length(ucyc) - 1)

        if n_allowed < 1
            # Degenerate target cycle (too few points, e.g. a 1-point trailing
            # partial cycle): it cannot get a donor. Rather than discard the
            # whole segment, pass this cycle through flat — its normalized flux
            # is already ≈ 1, so flux_detrended = norm_flux and baseline = the
            # cycle median. This keeps the rest of the segment on the LOCoR path.
            @inbounds for (k, jloc) in enumerate(target_idx_local)
                iglob = seg[jloc]
                flux_detrended[iglob] = target_norm[k]
                trend[iglob] = cyc_median[c]
                outlier[iglob] = false
                cycle_id_out[iglob] = c
            end
            continue
        end

        donors = others[1:n_allowed]

        # Donor cleaning: drop each donor's in-transit points BEFORE building
        # `olc`. `_interp_constant` then interpolates the donor ACROSS its
        # transit gap, so the donor contributes its rotational baseline rather
        # than its (commensurate) transit dip. A donor left with <2 points
        # after cleaning is skipped entirely.
        donor_phases = Vector{Float64}[]
        donor_fluxes = Vector{Float64}[]
        for d in donors
            keepd = .!cyc_mask[d]
            count(keepd) < 2 && continue
            push!(donor_phases, cyc_phase[d][keepd])
            push!(donor_fluxes, cyc_norm[d][keepd])
        end

        if isempty(donor_phases)
            # All donors collapsed under cleaning: pass this cycle through flat
            # rather than crash (no usable rotational template).
            @inbounds for (k, jloc) in enumerate(target_idx_local)
                iglob = seg[jloc]
                flux_detrended[iglob] = target_norm[k]
                trend[iglob] = cyc_median[c]
                outlier[iglob] = cyc_mask[c][k]
                cycle_id_out[iglob] = c
            end
            continue
        end

        # Target cleaning: pre-seed the target's in-transit points as outliers
        # so they never drive the LSQ fit; the baseline at those phases becomes
        # the clean donor extrapolation. They still receive a baseline and
        # appear in flux_detrended.
        target_mask = cyc_mask[c]
        init_outlier = any(target_mask) ? target_mask : nothing

        res = _rcomb_cycle_clipped(target_phase, target_norm,
                                   donor_phases, donor_fluxes;
                                   niter = niter, clip_sigma = clip_sigma,
                                   init_outlier = init_outlier)
        norm_baseline = res.baseline

        # Normalization cancels: flux_detrended = norm_flux / norm_baseline.
        # Raw baseline reconstructed only for reporting.
        @inbounds for (k, jloc) in enumerate(target_idx_local)
            iglob = seg[jloc]
            flux_detrended[iglob] = target_norm[k] / norm_baseline[k]
            trend[iglob] = norm_baseline[k] * cyc_median[c]
            outlier[iglob] = res.outlier[k]
            cycle_id_out[iglob] = c
        end
    end

    @inbounds for i in seg
        used_notch[i] = false
    end
    return nothing
end

"""
    detrend_locor(t, flux, flux_err; P_rot=nothing, max_donor_cycles=nothing,
                  niter=3, clip_sigma=3.5, gap_size=0.5,
                  transit_mask=nothing) -> NamedTuple

Locally Optimized Combination of Rotations detrending (LOCoR; Rizzuto+ 2017,
AJ 154 224). Phase-folds the light curve on the stellar rotation period and,
cycle by cycle, predicts each cycle's rotation baseline as a least-squares
linear combination of the neighbouring ("donor") cycles, then divides it out.
Built for fast rotators where a transit spans a small fraction of a rotation
cycle: dividing by a rotation template flattens the spot modulation while
leaving the transit dip intact. Returns a NamedTuple with:

- `flux_detrended` — rotation-flattened flux (multiplicative convention; a
   transit survives as a residual dip ≈ `1 − depth`)
- `trend` — reconstructed raw rotation baseline (the divided-out curve) at each
   cadence; named `trend` for consistency with the other Nereus detrenders
- `phase` — within-cycle phase in `[0, 1)` (global, from `_locor_cycles`)
- `cycle_id` — integer rotation-cycle index per cadence (`-1` where notch fell back)
- `outlier` — `Bool` outlier flag (set during per-cycle MAD σ-clipping)
- `segments` — segment ranges (gaps split via [`find_segments`](@ref))
- `P_rot_used` — the rotation period actually used (days)
- `used_notch` — `Bool` per-cadence flag: `true` where the LOCoR path was
   unavailable and [`detrend_notch`](@ref) was used as a fallback

# Arguments

- `t, flux, flux_err` — light curve arrays (BJD, normalized flux, error).
   Coerced to `Vector{Float64}`.
- `P_rot=nothing` — stellar rotation period in days. When `nothing`, it is
   estimated automatically via `find_rotation_period(t, flux, flux_err).P_rot`
   (autocorrelation). Pass it explicitly for reproducibility — the ACF
   estimate can be off for short or noisy baselines.
- `max_donor_cycles=nothing` — cap on the number of donor cycles per target
   cycle. `nothing` ⇒ uncapped (subject to the over-determination guard in
   [`_locor_segment!`](@ref)).
- `niter::Int=3` — per-cycle MAD σ-clip passes in [`_rcomb_cycle_clipped`](@ref).
- `clip_sigma::Real=3.5` — σ-clip threshold for the per-cycle robust fit.
- `gap_size::Real=0.5` — segment break threshold (days), passed to
   [`find_segments`](@ref).
- `transit_mask=nothing` — optional per-cadence `Bool` over the FULL light
   curve (`true` = in-transit), validated to `length == length(t)`. LOCoR is
   **mask-free by default**. Pass an ephemeris-derived mask to prevent
   *commensurate-period self-destruction*: when a transit's orbital period is
   commensurate with `P_rot`, the dip recurs at the same rotation phase across
   cycles, so the donor cycles contain it and the LSQ learns it into the
   baseline — absorbing/destroying the transit. With the mask supplied, each
   donor's in-transit points are dropped before the donor curve is built (so
   `_interp_constant` interpolates the rotational baseline ACROSS the gap), and
   the target's in-transit points are pre-seeded inactive in the per-cycle fit
   (so they never drive the LSQ); both effects make the baseline the clean
   rotational extrapolation while the dip survives in `flux_detrended`. In-
   transit points are still reported in `outlier`.

   !!! note
       The **whole-LC notch fallback** path (below) ignores `transit_mask`:
       [`detrend_notch`](@ref) is mask-free and takes no mask, so when LOCoR
       falls back to notch the mask has no effect there.

# Whole-LC notch fallback

LOCoR needs at least three rotation cycles somewhere in the data. If `P_rot`
is non-finite, non-positive, or longer than one third of the total baseline
(`P_rot > (t[end] − t[1]) / 3`, so ≥3 cycles can never form), the whole light
curve is detrended with [`detrend_notch`](@ref) instead. In that case
`phase` is all `NaN`, `cycle_id` is all `-1`, and `used_notch` is all `true`.
Per-segment fallback (segments with <3 cycles) is handled inside
[`_locor_segment!`](@ref).
"""
function detrend_locor(t::AbstractVector{<:Real}, flux::AbstractVector{<:Real},
                       flux_err::AbstractVector{<:Real};
                       P_rot = nothing,
                       max_donor_cycles = nothing,
                       niter::Int = 3,
                       clip_sigma::Real = 3.5,
                       gap_size::Real = 0.5,
                       sector_id::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                       transit_mask::Union{Nothing, AbstractVector{Bool}} = nothing)
    t = collect(Float64, t)
    flux = collect(Float64, flux)
    flux_err = collect(Float64, flux_err)
    n = length(t)
    n == length(flux) == length(flux_err) || throw(ArgumentError(
        "t, flux, flux_err must have the same length"))
    sector_id === nothing || length(sector_id) == n || throw(ArgumentError(
        "sector_id length $(length(sector_id)) != length(t) $n"))
    transit_mask === nothing || length(transit_mask) == n || throw(ArgumentError(
        "transit_mask length $(length(transit_mask)) != length(t) $n"))
    tmask = transit_mask === nothing ? nothing : collect(Bool, transit_mask)
    sid_out = sector_id === nothing ? nothing : Vector{Int}(sector_id)

    if P_rot === nothing
        P_rot = find_rotation_period(t, flux, flux_err).P_rot
    end
    P_rot = Float64(P_rot)
    P_rot_used = P_rot

    t0 = t[1]

    # Whole-LC notch fallback: cannot form ≥3 cycles anywhere.
    if !isfinite(P_rot) || P_rot <= 0 || P_rot > (t[end] - t[1]) / 3
        res = detrend_notch(t, flux, flux_err; gap_size = gap_size)
        return (; flux_detrended = res.flux_detrended,
                trend = res.trend,
                phase = fill(NaN, n),
                cycle_id = fill(-1, n),
                outlier = res.outlier,
                segments = res.segments,
                P_rot_used,
                used_notch = trues(n),
                sector_id = sid_out)
    end

    segments = find_segments(t; gap_size = gap_size)

    flux_detrended = similar(flux)
    trend = similar(flux)
    outlier = falses(n)
    cycle_id_out = fill(-1, n)
    used_notch = falses(n)

    for seg in segments
        _locor_segment!(flux_detrended, trend, outlier, cycle_id_out, used_notch,
                        t, flux, flux_err, seg, P_rot, t0,
                        max_donor_cycles, niter, clip_sigma, gap_size;
                        transit_mask = tmask)
    end

    # Global phase for output (cycle_id comes from the segment helper).
    phase = _locor_cycles(t, P_rot, t0).phase

    return (; flux_detrended, trend, phase, cycle_id = cycle_id_out,
            outlier, segments, P_rot_used, used_notch, sector_id = sid_out)
end

"""
    detrend_rotation(t, flux, flux_err; method=:auto, P_rot_threshold=2.0,
                     P_rot=nothing, kwargs...) -> NamedTuple

Auto-selecting rotation detrender that dispatches between [`detrend_locor`](@ref)
(LOCoR; Rizzuto+ 2017) and [`detrend_notch`](@ref) (Notch; Rizzuto+ 2017)
depending on the stellar rotation period.

The two methods cover complementary regimes:

  - **LOCoR** divides out a per-cycle rotation template and excels for *fast*
    rotators, where a transit spans only a small fraction of a rotation cycle.
  - **Notch** fits a local trend-plus-notch window and is the workhorse for
    *slow* rotators (and the safe fallback when too few cycles exist).

# Arguments

- `t, flux, flux_err` — light curve arrays (BJD, normalized flux, error).
- `method::Symbol=:auto` — one of `:auto`, `:locor`, `:notch`:
    - `:locor` → always [`detrend_locor`](@ref) (with the resolved `P_rot`).
    - `:notch` → always [`detrend_notch`](@ref) (notch ignores `P_rot`).
    - `:auto`  → route fast rotators (`isfinite(P_rot) && P_rot < P_rot_threshold`)
      to LOCoR, otherwise to notch.
  Any other value throws `ArgumentError`.
- `P_rot_threshold::Real=2.0` — period (days) below which `:auto` selects LOCoR.
- `P_rot=nothing` — stellar rotation period (days). When `nothing`, it is
  estimated via `find_rotation_period(t, flux, flux_err).P_rot`.
- `kwargs...` — forwarded verbatim to the chosen detrender.

Returns whatever the chosen detrender returns; the schemas are intentionally
**not** unified here. Both share `flux_detrended` and `trend`, which is what
downstream callers inspect — branch on `method`/`P_rot_used` if you need the
method-specific fields (`cycle_id`/`P_rot_used` for LOCoR, `delta_bic`/`depth`
for notch).

!!! warning "Auto-routing caveat for fast rotators"
    `find_rotation_period`'s ACF peak-finder is tuned for *slow* rotators and
    currently returns `NaN` for many fast rotators. Since `:auto` requires
    `isfinite(P_rot)` to pick LOCoR, an auto-resolved `P_rot` of `NaN` routes
    the target to **notch** even when LOCoR would be the right tool. For
    guaranteed LOCoR on a known fast rotator, use `method=:locor` or pass
    `P_rot` explicitly. Robust fast-rotator `P_rot` detection is a separate
    follow-up.
"""
function detrend_rotation(t::AbstractVector{<:Real}, flux::AbstractVector{<:Real},
                          flux_err::AbstractVector{<:Real};
                          method::Symbol = :auto,
                          P_rot_threshold::Real = 2.0,
                          P_rot = nothing,
                          kwargs...)
    method in (:auto, :locor, :notch) || throw(ArgumentError(
        "method must be one of :auto, :locor, :notch (got $(repr(method)))"))

    if P_rot === nothing
        P_rot = find_rotation_period(t, flux, flux_err).P_rot
    end
    P_rot = Float64(P_rot)

    if method === :notch
        return detrend_notch(collect(Float64, t), collect(Float64, flux),
                             collect(Float64, flux_err); kwargs...)
    elseif method === :locor
        return detrend_locor(t, flux, flux_err; P_rot = P_rot, kwargs...)
    else  # :auto
        if isfinite(P_rot) && P_rot < P_rot_threshold
            return detrend_locor(t, flux, flux_err; P_rot = P_rot, kwargs...)
        else
            return detrend_notch(collect(Float64, t), collect(Float64, flux),
                                 collect(Float64, flux_err); kwargs...)
        end
    end
end
