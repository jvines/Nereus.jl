# Candidate promotion gate for the blind RV catalog.
#
# A candidate is a signal that survives every way we know to kill it. This
# assembles the conjunction of vetoes — each aimed at one false-positive mode —
# into a single verdict (:candidate / :flagged / :reject). It's deliberately
# decoupled from the trans-dim chain format: the caller extracts the fitted
# (P, K, e), the occupancy/evidence, and the activity-subtracted residual from
# whatever sampler was used and passes them in. The vetoes reuse existing
# primitives (`gls_periodogram`, `coherence_discriminant`); the only judgement
# lives in the thresholds, which the calibration harness sets to a target
# false-positive rate rather than by taste.

"""
    CandidateVerdict

Per-veto verdicts + the overall promotion decision from [`vet_candidate`](@ref).
Each veto is `:pass`, `:fail`, or `:inapplicable` (data absent). `decision` is
`:candidate` (all applicable vetoes pass), `:reject` (a hard veto fails), or
`:flagged` (ambiguous — e.g. the coherence test is toothless on a short
baseline). `reasons` lists the failing/ambiguous vetoes.
"""
struct CandidateVerdict
    decision::Symbol
    detection::Symbol
    activity_indicator::Symbol
    rotation::Symbol
    alias::Symbol
    coherence::Symbol
    physicality::Symbol
    reasons::Vector{String}
    detail::Dict{String, Any}
end

# period matches P_ref, or its low-order harmonics/aliases, within frac
function _period_matches(P::Real, P_ref::Real; frac::Real = 0.03,
                          harmonics = (1//1, 2//1, 1//2, 3//2, 2//3))
    for h in harmonics
        Pm = P_ref * float(h)
        abs(P - Pm) / Pm <= frac && return true
    end
    return false
end

"""
    vet_candidate(t, rv, rv_err, P, K, e; kwargs...) -> CandidateVerdict

Run the promotion gate on one candidate signal at period `P` (semi-amplitude
`K`, eccentricity `e`).

# Key kwargs
- `residual`         : RV with the OTHER planets + γ/trend removed — the series
                       the coherence test runs on. Defaults to `rv` (use the raw
                       series only for a single-planet system).
- `indicators`       : `Dict(name => values)` of activity indicators (BIS,
                       S-index, FWHM, Hα, CCF-PCs) sampled at `t`.
- `P_rot`            : measured rotation period (days), or `nothing`.
- `occupancy`,`dlnZ` : trans-dim occupancy / Δln Z of the planet's inclusion.
- `occ_min`,`dlnZ_min` : detection thresholds (**calibrate to a target FPR**).
- `ind_fap`,`P_tol`  : indicator-peak FAP cut and fractional period tolerance.
- `k_floor`          : minimum semi-amplitude to be physical (e.g. a few × the
                       per-point error / √N).
"""
function vet_candidate(t::AbstractVector{<:Real}, rv::AbstractVector{<:Real},
                        rv_err::AbstractVector{<:Real}, P::Real, K::Real, e::Real;
                        residual::Union{Nothing, AbstractVector{<:Real}} = nothing,
                        indicators = Dict{String, Vector{Float64}}(),
                        P_rot::Union{Nothing, Real} = nothing,
                        occupancy::Union{Nothing, Real} = nothing,
                        dlnZ::Union{Nothing, Real} = nothing,
                        occ_min::Real = 0.95, dlnZ_min::Real = 5.0,
                        ind_fap::Real = 0.01, P_tol::Real = 0.03,
                        k_floor::Real = 0.0, e_max::Real = 0.99,
                        coherence_Δtol::Real = 2.0)
    resid = residual === nothing ? Float64.(collect(rv)) : Float64.(collect(residual))
    reasons = String[]
    detail  = Dict{String, Any}()

    # 1. Detection — occupancy and/or evidence over the null. Skipped if neither
    #    supplied (the caller asserts the peak was detected).
    detection = :inapplicable
    if occupancy !== nothing || dlnZ !== nothing
        occ_ok = occupancy === nothing || occupancy >= occ_min
        znz_ok = dlnZ === nothing || dlnZ >= dlnZ_min
        detection = (occ_ok && znz_ok) ? :pass : :fail
        detection === :fail && push!(reasons, "detection below threshold " *
            "(occ=$(occupancy), ΔlnZ=$(dlnZ))")
    end

    # 2. Activity-indicator veto — a significant peak at P (or harmonics) in ANY
    #    indicator kills it. Presence disproves; absence is not proof (a veto).
    activity_indicator = :inapplicable
    flagged_inds = String[]
    if !isempty(indicators)
        activity_indicator = :pass
        for (nm, vals) in indicators
            v = Float64.(collect(vals))
            length(v) == length(t) || continue
            pg = gls_periodogram(collect(Float64, t), v, Float64.(collect(rv_err));
                                  fap_levels = [ind_fap], fap_method = :analytic,
                                  samples_per_peak = 10)
            for pk in pg.peaks
                if _period_matches(pk.period, P; frac = P_tol) &&
                   isfinite(pk.fap) && pk.fap <= ind_fap
                    push!(flagged_inds, nm); break
                end
            end
        end
        if !isempty(flagged_inds)
            activity_indicator = :fail
            push!(reasons, "period matches an activity indicator: " *
                join(flagged_inds, ", "))
        end
    end
    detail["indicators_flagged"] = flagged_inds

    # 3. Rotation veto — P at P_rot or its harmonics. Inapplicable when P_rot
    #    unknown (falls through to indicators + coherence).
    rotation = :inapplicable
    if P_rot !== nothing
        rotation = _period_matches(P, P_rot; frac = P_tol) ? :fail : :pass
        rotation === :fail && push!(reasons, "period at rotation P_rot=$(P_rot) d")
    end

    # 4. Alias veto — power at P in the spectral WINDOW ⇒ P may be a sampling
    #    alias of another signal, not intrinsic.
    win = ones(length(t))
    pgw = gls_periodogram(collect(Float64, t), win, ones(length(t));
                           fap_levels = [ind_fap], fap_method = :analytic,
                           samples_per_peak = 10)
    alias = :pass
    for pk in pgw.peaks
        if _period_matches(pk.period, P; frac = P_tol) && pk.period > 1.5
            alias = :fail
            push!(reasons, "period coincides with a window-function peak")
            break
        end
    end

    # 5. Coherence — the SHO-Q test. :incoherent kills (activity); :no_signal is
    #    toothless (short baseline / no residual power) → don't reject on it.
    cr = coherence_discriminant(collect(Float64, t), resid,
                                 Float64.(collect(rv_err)), P; Δtol = coherence_Δtol)
    coherence = cr.verdict === :coherent ? :pass :
                cr.verdict === :incoherent ? :fail : :inapplicable
    detail["coherence"] = cr
    coherence === :fail && push!(reasons,
        "signal decoheres within the baseline (Q_ml=$(round(cr.Q_ml, digits=1)) ≪ " *
        "Q_coh=$(round(cr.Q_coh, digits=1)))")

    # 6. Physicality — eccentricity not railed, amplitude above the floor.
    physicality = (e < e_max && K > k_floor) ? :pass : :fail
    physicality === :fail && push!(reasons,
        "unphysical (e=$(round(e, digits=3)) or K=$(round(K, digits=2)) ≤ floor)")

    # Overall: hard vetoes are activity-indicator, rotation, alias, coherence,
    # physicality, and (when tested) detection. A hard :fail ⇒ :reject. All-pass
    # ⇒ :candidate. Otherwise (only inapplicable/toothless items unresolved) ⇒
    # :flagged.
    hard = (detection, activity_indicator, rotation, alias, coherence, physicality)
    decision = if any(==(:fail), hard)
        :reject
    elseif all(v -> v === :pass, (activity_indicator, rotation, alias,
                                   coherence, physicality)) &&
           detection !== :fail
        :candidate
    else
        :flagged
    end

    return CandidateVerdict(decision, detection, activity_indicator, rotation,
                            alias, coherence, physicality, reasons, detail)
end
