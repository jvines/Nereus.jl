# Parameter name generation and validation for noise models.
# Loaded after model.jl (needs InstrumentConfig).

"""
    noise_param_names(model, instruments) -> Vector{String}

Return the parameter names this noise model contributes to the layout.
"""
function noise_param_names end

function noise_param_names(m::ActivityDecorrelation, instruments::InstrumentConfig;
                            data::Union{Data, Nothing}=nothing)
    names = String[]
    suf = _ad_suffix(m)
    if m.per_instrument
        for (ins_idx, ins) in enumerate(instruments.rv_names)
            for ind in m.indicators
                # Skip instruments without finite indicator values
                if data !== nothing && haskey(data.indicators, ind)
                    ind_vals = data.indicators[ind]
                    has_data = any(i -> data.rv_inst[i] == ins_idx && isfinite(ind_vals[i]),
                                   eachindex(ind_vals))
                    has_data || continue
                end
                push!(names, "C_$(ind)_$(ins)$suf")
                # FF'-style: companion derivative coefficient per inst.
                m.derivative && push!(names, "Cdot_$(ind)_$(ins)$suf")
            end
        end
    else
        for ind in m.indicators
            push!(names, "C_$(ind)$suf")
            m.derivative && push!(names, "Cdot_$(ind)$suf")
        end
    end
    return names
end

function noise_param_names(m::MAModel, instruments::InstrumentConfig; data=nothing)
    names = String[]
    s = _channel_suffix(m.channel)
    chan_inst = m.channel === :rv ? instruments.rv_names : instruments.pm_names
    if m.per_instrument
        for ins in chan_inst
            for j in 1:m.order
                push!(names, "ma_omega_$(j)_$(ins)$s")
                push!(names, "ma_beta_$(j)_$(ins)$s")
            end
        end
    else
        for j in 1:m.order
            push!(names, "ma_omega_$j$s")
            push!(names, "ma_beta_$j$s")
        end
    end
    return names
end

function noise_param_names(m::ARModel, instruments::InstrumentConfig; data=nothing)
    names = String[]
    s = _channel_suffix(m.channel)
    chan_inst = m.channel === :rv ? instruments.rv_names : instruments.pm_names
    if m.per_instrument
        for ins in chan_inst
            for j in 1:m.order
                push!(names, "ar_phi_$(j)_$(ins)$s")
                push!(names, "ar_alpha_$(j)_$(ins)$s")
            end
        end
    else
        for j in 1:m.order
            push!(names, "ar_phi_$j$s")
            push!(names, "ar_alpha_$j$s")
        end
    end
    return names
end

function noise_param_names(m::ActivityJitter, instruments::InstrumentConfig; data=nothing)
    names = String[]
    for ins in instruments.rv_names
        # Include the indicator: multiple ActivityJitter models can target the
        # same instrument (one per indicator), and bare jit_base_$(ins) names
        # would collide → duplicate layout entries → NetCDF defVar -42 on write.
        push!(names, "jit_base_$(m.indicator)_$(ins)")
        push!(names, "jit_act_$(m.indicator)_$(ins)")
    end
    return names
end

function noise_param_names(m::IndicatorFloor, ::InstrumentConfig; data=nothing)
    if m.kernel === :qp
        # Quasi-periodic GP floor: shared (period, λ_e, λ_p) kernel
        # hyperparameters + per-channel amplitude and white jitter.
        names = ["ind_floor_period", "ind_floor_lambda_e", "ind_floor_lambda_p"]
        for ch in m.channels
            push!(names, "ind_floor_$(ch)_amp")
            push!(names, "ind_floor_$(ch)_jit")
        end
        return names
    end
    # :white — one floor σ per indicator channel.
    return ["ind_floor_$(ch)" for ch in m.channels]
end

function noise_param_names(m::StudentT, ::InstrumentConfig; data=nothing)
    return ["studentt_nu"]
end

function noise_param_names(m::ErrorScale, instruments::InstrumentConfig; data=nothing)
    insts = isempty(m.instruments) ? instruments.rv_names : m.instruments
    return ["errscale_$(ins)" for ins in insts]
end

function noise_param_names(m::NightlyOffset, instruments::InstrumentConfig; data=nothing)
    s = _channel_suffix(m.channel)
    chan_inst = m.channel === :rv ? instruments.rv_names : instruments.pm_names
    insts = isempty(m.instruments) ? chan_inst : m.instruments
    return ["night_sigma_$(ins)$s" for ins in insts]
end

function noise_param_names(m::HarmonicBlock, instruments::InstrumentConfig; data=nothing)
    s = _channel_suffix(m.channel)
    chan_inst = m.channel === :rv ? instruments.rv_names : instruments.pm_names
    insts = isempty(m.instruments) ? chan_inst : m.instruments
    # With an external frequency comb there is no rotation period to fit; the
    # only free parameters are the per-instrument amplitudes, since the cos/sin
    # coefficients (hence the phases) are marginalised analytically.
    names = isempty(m.freqs) ? ["harm_period$s"] : String[]
    for ins in insts
        push!(names, "harm_amp_$(ins)$s")
    end
    return names
end

"""Suffix appended to GP hyperparameter names for non-default channels.
Empty for `:rv` (backward compatible — old layouts have bare names);
`"_phot"` for photometry-channel kernels so they don't collide with an
RV-channel kernel of the same family in the same fit.

Kept for callers that only care about the channel; new code should
prefer `_gp_suffix(nm)` which also accounts for per-instrument scoping."""
_channel_suffix(ch::Symbol) = ch === :rv ? "" : "_$ch"

"""Full parameter-name suffix for a `CovarianceNoise` GP — combines
the channel tag with the per-instrument scope.

Examples (with `instruments=String[]` ⇒ global GP, the back-compat case):
  - `:rv`,   []                   → `""`
  - `:phot`, []                   → `"_phot"`
With per-instrument scoping (`instruments=["HARPS"]` etc.):
  - `:rv`,   ["HARPS"]            → `"_HARPS"`
  - `:rv`,   ["HARPS", "FEROS"]   → `"_HARPS+FEROS"`
  - `:phot`, ["TESS"]             → `"_phot_TESS"`

The leading `_` is included only when the suffix is non-empty so the
back-compat global-RV layout still uses bare `gp_log_amp` etc."""
function _gp_suffix(nm::CovarianceNoise)
    parts = String[]
    nm.channel === :rv || push!(parts, String(nm.channel))
    isempty(nm.instruments) || push!(parts, join(nm.instruments, "+"))
    isempty(parts) ? "" : "_" * join(parts, "_")
end

# ActivityGP is inherently multi-channel (RV + indicator list), so it
# has no single `channel` field. Suffix is empty for the bare global
# case; per-instrument scoping appends the instrument list.
function _gp_suffix(nm::ActivityGP)
    isempty(nm.instruments) ? "" : "_" * join(nm.instruments, "+")
end

function noise_param_names(nm::CeleriteSHO, ::InstrumentConfig; data=nothing)
    s = _gp_suffix(nm)
    return ["gp_log_S0$s", "gp_log_Q$s", "gp_log_omega0$s"]
end

function noise_param_names(nm::MaternGP, ::InstrumentConfig; data=nothing)
    s = _gp_suffix(nm)
    return ["matern_sigma$s", "matern_rho$s"]
end

function noise_param_names(nm::CeleriteRotationFM17, ::InstrumentConfig; data=nothing)
    s = _gp_suffix(nm)
    return ["gp_log_amp$s", "gp_log_timescale$s",
            "gp_log_period$s", "gp_log_factor$s"]
end

function noise_param_names(nm::CeleriteRotation, ::InstrumentConfig; data=nothing)
    s = _gp_suffix(nm)
    return ["gp_sigma$s", "gp_period$s", "gp_Q0$s", "gp_dQ$s", "gp_f$s"]
end

# Activity multivariate-GP (Rajpaul+ 2015): kernel hyperparameters +
# per-channel coupling coefficients on G and dG/dt. RV is always
# included as an active channel; user controls the indicator list.
function noise_param_names(nm::ActivityGP, ::InstrumentConfig; data=nothing)
    s = _gp_suffix(nm)
    # NO kernel amplitude: Rajpaul+ 2015 defines G(t) as a UNIT-VARIANCE
    # latent GP — all scale lives in the per-channel couplings (Vc, Vr,
    # Bc, …). A free amp on top is an exact non-identifiability
    # ((amp, a, b) → (amp/c, c·a, c·b) leaves the likelihood invariant)
    # that destabilized logZ by ~80 nats run-to-run and let K inflate
    # (HD 18599 post-mortem 2026-06-12).
    names = ["gp_act_period$s",
             "gp_act_lambda_e$s", "gp_act_lambda_p$s"]
    # RV is always an active channel; coupling to G is always present,
    # coupling to dG/dt is gated globally by `use_derivative`. In
    # indicators_only mode the RV channel is not scored, so its
    # couplings are not sampled.
    if !nm.indicators_only
        push!(names, "Vc$s")
        if nm.use_derivative
            push!(names, "Vr$s")
        end
    end
    for ch in nm.channels
        ch === :rv && continue
        cg, cd = _ACTIVITY_GP_COEFFS[ch]
        push!(names, string(cg, s))
        if nm.use_derivative && cd !== nothing
            push!(names, string(cd, s))
        end
        # Per-channel indicator jitter: Σ_II's diagonal must carry a
        # FITTED floor — reported BIS/FWHM/index errors are routinely
        # underestimated, and a near-singular Σ_II lets the conditional
        # mean over-track the indicators (free-lunch amplifier).
        push!(names, "gp_act_jit_$(ch)$s")
    end
    return names
end

"""
    validate_noise_models(models; transdim=false)

Check composition constraints:
- Stage 1 (MeanModifier): any number, always allowed.
- Stage 2 (SequentialNoise): any number allowed (AR + MA = ARMA).
- Stage 3 (CovarianceNoise) per channel: either at most one *global* GP
  (`instruments=[]`, the back-compat case) OR any number of *restricted*
  GPs with **pairwise-disjoint** instrument sets — never both, never
  overlapping.
- Stage 2 + Stage 3: mutually exclusive within a channel **only when a
  global GP is present**. A restricted GP doesn't block AR/MA on the
  same channel because it covers only a subset of observations.

When `transdim=true`, the Stage 2 vs Stage 3 mutual exclusion is
relaxed at construction — both may exist in the layout but trans-dim
birth proposals enforce that only one category is active at runtime.
"""
function validate_noise_models(models::Vector{<:NoiseModel}; transdim::Bool=false)
    # Student-t is a white-branch likelihood: it destroys the Gaussian
    # marginalization every covariance/additive/sequential path relies on, so
    # it must never combine with one. Rejected at construction (non-trans-dim);
    # under trans-dim the eval-time −Inf guard handles inadmissible states and
    # the user is expected to put them in a noise_exclusion_groups entry.
    if !transdim && any(m -> m isa StudentT, models)
        for m in models
            (m isa CovarianceNoise || m isa AdditiveCovariance ||
             m isa SequentialNoise) && throw(ArgumentError(
                "StudentT is a white-branch likelihood and cannot combine with " *
                "$(typeof(m)) — it breaks the Gaussian marginalization the " *
                "covariance path relies on. Remove one, or (trans-dim) put them " *
                "in a mutually-exclusive noise_exclusion_groups entry."))
        end
    end

    # Per-channel: collect global vs restricted GPs.
    globals_per_channel = Dict{Symbol, Int}()
    restricted_per_channel = Dict{Symbol, Vector{Set{String}}}()
    for m in models
        m isa CovarianceNoise || continue
        ch = noise_channel(m)
        insts = noise_instruments(m)
        if isempty(insts)
            globals_per_channel[ch] = get(globals_per_channel, ch, 0) + 1
        else
            length(unique(insts)) == length(insts) || throw(ArgumentError(
                "CovarianceNoise on channel `:$ch` has duplicate instruments " *
                "in its `instruments` list: $insts"))
            push!(get!(restricted_per_channel, ch, Set{String}[]), Set(insts))
        end
    end

    for (ch, n) in globals_per_channel
        # Under trans-dim the user can declare multiple global
        # CovarianceNoise on the same channel (e.g. ActivityGP +
        # CeleriteRotation) as long as a `noise_exclusion_groups`
        # entry in TransDimConfig keeps them mutually exclusive at
        # runtime. We don't see exclusion_groups here, so we trust
        # the user-level contract.
        (transdim || n <= 1) || throw(ArgumentError(
            "At most one global CovarianceNoise allowed per channel " *
            "(non-trans-dim); got $n on channel `:$ch`. Either set " *
            "`instruments=[\"…\"]` on each, or keep one global GP, " *
            "or set `transdim_noise=true` + add a noise_exclusion_groups " *
            "entry in TransDimConfig."))
        # Global + any restricted on the same channel is not allowed —
        # the global already covers every instrument.
        haskey(restricted_per_channel, ch) && throw(ArgumentError(
            "Channel `:$ch` has both a global CovarianceNoise " *
            "(`instruments=[]`) and a restricted one. Pick one mode: " *
            "either a single global GP or several per-instrument GPs."))
    end

    # Restricted GPs on the same channel must have pairwise-disjoint
    # instrument sets — otherwise an observation is covered by two GPs
    # at once and the likelihood is double-counted.
    for (ch, sets) in restricted_per_channel
        for i in 1:length(sets), j in (i + 1):length(sets)
            inter = intersect(sets[i], sets[j])
            isempty(inter) || throw(ArgumentError(
                "Channel `:$ch` has CovarianceNoise models whose " *
                "instrument sets overlap on $(collect(inter)); " *
                "per-instrument GPs must cover disjoint instruments."))
        end
    end

    # SequentialNoise/CovarianceNoise mutex applies per channel and
    # only when a global GP is present (a restricted GP only covers
    # some observations and leaves room for AR/MA on the rest).
    if !transdim
        seq_channels = Set{Symbol}()
        for m in models
            m isa SequentialNoise && push!(seq_channels, noise_channel(m))
        end
        for (ch, _) in globals_per_channel
            ch in seq_channels && throw(ArgumentError(
                "SequentialNoise and a global CovarianceNoise are mutually " *
                "exclusive on channel `:$ch` (use transdim_noise=true to " *
                "allow both as toggleable components, or scope the GP to " *
                "specific instruments via `instruments=[\"…\"]`)."))
        end
    end
end
