# Default trans-dim noise menu (NEREUS_NOISE_MENU.md §3).
#
# Assembles the canonical toggleable noise set + mutual-exclusion groups so a
# user doesn't hand-wire the whole thing. Role-based SINGLE-WINNER structure —
# physically meaningful, one winner per role, not one-model-total:
#
#   • Correlated-noise DESCRIPTION (single winner): CeleriteSHO (oscillatory) ·
#     CeleriteRotation (rotation) · HarmonicBlock (coherent rotation) ·
#     MaternGP (short-memory) · ActivityGP (quasi-periodic, indicator-coupled).
#     These describe the SAME stellar signal → at most one active.
#   • Indicator-activity (single winner): ActivityDecorrelation linear vs FF′ vs
#     ActivityGP — the indicator-driven models. AD competes with ActivityGP
#     (same hypothesis, different fidelity) but COMPOSES with the pure-RV GPs
#     (it regresses external indicator data — different information).
#   • White treatment (single winner): ErrorScale vs StudentT; plain jitter is
#     the always-on baseline (neither active ⇒ jitter). StudentT's
#     incompatibility with any covariance is enforced at eval (−Inf), so no
#     extra groups are needed for it.
#   • Calibration: NightlyOffset is OFF by default for RV — a per-night
#     marginalized offset is degenerate with a Keplerian when P ≫ 1 night (it
#     absorbs the planet's per-night-sampled signal) and collapses to jitter for
#     1-exposure nights. Opt in (`include_nightly=true`) only for photometry /
#     dense-cadence RV. When on, it composes on the activity winner except
#     ActivityGP (joint path can't absorb an additive term).
#
# When ActivityGP is in the menu a `:qp` IndicatorFloor is added as an
# ALWAYS-ON (non-toggleable) model: AGP scores p(RV, indicators) while every
# other state scores p(RV)·p(indicators|floor) — the SAME data either way, so
# the occupancy is a valid P(M|D) (see the AD↔AGP comparability post-mortem).

"""
    default_noise_menu(data; indicators, include_matern=true, include_studentt=true,
                       agp_latent=:mep, nharm=3)
        -> (noise_models, toggleable, exclusion_groups)

Canonical trans-dim noise menu. Pass `noise_models` to `Params(...)` and
`toggleable` + `exclusion_groups` to `TransDimConfig(noise=true, ...)`:

    menu = default_noise_menu(data)
    params = Params(...; noise_models = menu.noise_models, transdim_noise = true)
    td = TransDimConfig(noise = true, toggleable = menu.toggleable,
                        noise_exclusion_groups = menu.exclusion_groups)

Indicator-driven members (ActivityGP, ActivityDecorrelation) are included only
when indicators are present in `data`; ActivityGP only for indicator channels
it supports (`:bis, :fwhm, :logrhk, :halpha`).
"""
function default_noise_menu(data::Data;
                            indicators::Vector{String} = collect(keys(data.indicators)),
                            include_matern::Bool = false,   # free-period interpolator on sparse RV
                            include_studentt::Bool = false,     # never fires; +eval-incompat headache
                            include_harmonic::Bool = false,     # redundant with the rotation GP
                            include_ad_ffprime::Bool = false,   # redundant with AD-linear, +12 dims
                            include_nightly::Bool = false,
                            include_activity_gp::Bool = true,  # single-channel AGP can hijack (see below)
                            agp_latent::Symbol = :mep,
                            nharm::Int = 3)
    toggleable = NoiseModel[]
    always_on  = NoiseModel[]
    groups     = Vector{NoiseModel}[]

    # White treatment. Plain per-instrument jitter is the always-on floor.
    # ErrorScale (white error inflation) and StudentT (heavy-tailed white) are
    # NOT independent axes that compose with a correlated model: you describe the
    # excess scatter as WHITE **or** as CORRELATED, never both. Composing them is
    # exactly what let ErrorScale inflate/discard points (FEROS ×10) while a
    # rotation GP simultaneously injected at P_rot/2 ≈ P_orb. So ErrorScale and
    # StudentT join the SINGLE noise-treatment exclusion group below as the white
    # competitors against the correlated GPs and the AD decorrelation.
    st = nothing
    es = ErrorScale(); push!(toggleable, es)
    include_studentt && (st = StudentT(); push!(toggleable, st))

    # Calibration. OFF by default for RV: a per-night marginalized offset is
    # degenerate with a Keplerian when P ≫ 1 night (each night samples ~one
    # orbital phase, so marginalizing the per-night mean removes the planet's
    # signal at that phase) and collapses to plain jitter for 1-exposure nights.
    # Opt in (`include_nightly=true`) only for photometry / dense-cadence RV
    # where within-night sampling separates a shared calibration shift from both
    # jitter and the signal.
    nightly = nothing
    if include_nightly
        nightly = NightlyOffset(); push!(toggleable, nightly)
    end

    # NOISE / ACTIVITY TREATMENT — a SINGLE winner (one exclusion group). Every
    # member is a competing description of the SAME excess scatter, so at most one
    # is active and the occupancy picks. WHITE and CORRELATED descriptions compete
    # HERE — they never compose. Members:
    #   • ErrorScale        — WHITE: per-instrument error inflation (no covariance).
    #   • CeleriteRotation  — rotation GP, period ANCHORED at P_rot (2 SHO terms).
    #   • HarmonicBlock     — coherent rotation harmonics (anchored).
    #   • ActivityGP        — quasi-periodic, indicator-coupled (anchored).
    #   • ActivityDecorrelation linear / FF′ — linear indicator regression.
    # AD COMPETES with the GPs here — it does NOT compose. Composing AD+GP
    # double-models the activity and dilutes the planet (HD 18599: AD-alone
    # K≈11, AD+GP K≈17). The bare CeleriteSHO and MaternGP are EXCLUDED from the
    # default RV menu: a GP with no period anchor (free-period SHO, or short-
    # memory Matérn) slides onto / interpolates the planet on sparse RV and
    # inflates K (SHO parked at 2.6 d → K≈23; Matérn best-fit → K≈34). Only
    # period-anchored covariance descriptions are safe. StudentT joins as the
    # "heavy-tailed white, no correlation" option (this also encodes its
    # eval-incompatibility with any covariance as exclusion).
    # ErrorScale (white) is the first competitor in the single noise-treatment
    # group: white-noise XOR correlated-GP XOR indicator-decorrelation.
    activity = NoiseModel[es]
    rot = CeleriteRotation(); push!(toggleable, rot); push!(activity, rot)
    include_harmonic && (h = HarmonicBlock(nharm = nharm); push!(toggleable, h); push!(activity, h))
    include_matern && (mat = MaternGP(); push!(toggleable, mat); push!(activity, mat))

    valid_ind = filter(k -> haskey(data.indicators, k), indicators)
    agp_channels = filter(k -> Symbol(k) in keys(_ACTIVITY_GP_COEFFS), valid_ind)
    if !isempty(valid_ind)
        adL = ActivityDecorrelation(indicators = valid_ind, derivative = false, label = "lin")
        push!(toggleable, adL); push!(activity, adL)
        if include_ad_ffprime
            adF = ActivityDecorrelation(indicators = valid_ind, derivative = true, label = "ffp")
            push!(toggleable, adF); push!(activity, adF)
        end
        # include_activity_gp=false drops the joint RV+indicator GP from the
        # race. Needed because AGP's shared latent is only as constrained as
        # the channel set: with a SINGLE selected channel that itself carries
        # power near the orbital period (HD 18599, BIS at P_rot/2 ≈ P_orb),
        # the latent aligns with the planet phase and Vc·G absorbs the planet
        # (measured: AGP occupancy 1.0, K 11 → 4.2 on bis-only; 0.004 occupancy
        # with all four channels constraining the latent). The IndicatorFloor
        # stays always-on regardless — every surviving state must still score
        # the indicator data, or occupancies compare different datasets.
        if !isempty(agp_channels)
            if include_activity_gp
                agp = ActivityGP(channels = Symbol.(agp_channels), latent_kernel = agp_latent)
                push!(toggleable, agp); push!(activity, agp)
                nightly === nothing || push!(groups, NoiseModel[agp, nightly])
            end
            # Always-on floor so an AD-active state (RV only) and an AGP-active
            # state (RV + joint indicators) score the SAME data — honest occupancy.
            push!(always_on, IndicatorFloor(channels = Symbol.(agp_channels), kernel = :qp))
        end
    end

    st === nothing || (push!(activity, st);
                       nightly === nothing || push!(groups, NoiseModel[st, nightly]))

    push!(groups, activity)

    noise_models = vcat(toggleable, always_on)
    return (noise_models = noise_models, toggleable = toggleable,
            exclusion_groups = groups)
end
