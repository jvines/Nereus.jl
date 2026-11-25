# Noise-model birth and death: trans-dimensional selection over the noise menu.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# Noise birth/death
# =====================================================================

"""
    propose_noise_birth(theta, rng, toggleable;
                         exclusion_groups=Vector{NoiseModel}[]) -> (new_theta, log_q_ratio)

Activate a random inactive noise component from `toggleable`. Draws
its parameters from the prior. Enforces two layers of mutual exclusion:

1. **Hard typing rule** (always applied): SequentialNoise (AR/MA)
   cannot coexist with CovarianceNoise (GP); CovarianceNoise blocks
   other CovarianceNoise. AR + MA stay composable.
2. **User-declared exclusion groups** (optional, via `exclusion_groups`):
   at most one model in each group can be active. Use this when two
   toggleables capture the same physics with different mathematical
   structures (e.g. linear BIS regression vs GP rotation kernel both
   modelling stellar rotation).
"""
function propose_noise_birth(theta::Theta{T}, rng::AbstractRNG,
                              toggleable::Vector{NoiseModel};
                              scratch::Union{Theta{T}, Nothing}=nothing,
                              data::Union{Data, Nothing}=nothing,
                              exclusion_groups::Vector{Vector{NoiseModel}}=
                                  Vector{NoiseModel}[],
                              parts::Union{Nothing, Base.RefValue{Any}}=nothing) where {T}
    td = theta.td
    noise_models = theta.params.config.noise_models

    # Inactive toggleable components that may be switched on here.
    inactive = _noise_activatable(td, toggleable, noise_models, exclusion_groups)

    isempty(inactive) && return (theta, convert(T, -Inf))

    # Pick one
    nm_idx = inactive[rand(rng, 1:length(inactive))]
    nm = noise_models[nm_idx]

    # Draw new noise params from prior
    layout = theta.params.layout
    instruments = theta.params.config.instruments
    nm_names = noise_param_names(nm, instruments)

    # Use scratch if available, otherwise allocate
    if scratch !== nothing
        scratch.values .= theta.values
        copy_into!(scratch.td, td)
        new_values = scratch.values
        new_td = scratch.td
    else
        new_values = copy(theta.values)
        new_td = copy(td)
    end

    # Draw each param from its prior, accumulating the forward proposal
    # density. The −log_q_fwd term below cancels the +Σ logpdf(prior, β_new)
    # that Δlog_pi gains on activation (`_log_prior_transdim` skips inactive
    # noise slots). WITHOUT it the chain satisfies detailed balance for a
    # prior-density-TILTED pseudo-posterior π·exp(Σ_active logpdf) — the
    # noise-model occupancy is then not P(M|D) (the planet-side
    # PriorBirth carries the same cancellation for the same reason).
    log_q_fwd = zero(T)
    # OLS-INFORMED birth for ActivityDecorrelation: its coefficients are
    # a linear-Gaussian regression on the (AD-inactive) residual; blind
    # prior draws make AD uncompetitive in a trans-dim selection (the
    # climb can't reach the joint optimum for correlated regressors), so
    # its occupancy under-represents AD vs its evidence. Draw them jointly
    # from N(C_ols, inflate²·(XᵀWX)⁻¹) and carry that density (not the
    # prior) in the Hastings ratio. `theta` here has AD INACTIVE.
    ad_informed = false
    if AD_INFORMED_BIRTH[] && nm isa ActivityDecorrelation && data !== nothing
        cnames, C_ols, F = ad_ols_fit(theta, data, nm)
        if F !== nothing && length(C_ols) == length(cnames) && !isempty(cnames)
            inflate = 2.0
            z = randn(rng, length(C_ols))
            Cdraw = C_ols .+ inflate .* (F.U \ z)        # cov = inflate²(XᵀWX)⁻¹
            ad_informed = true
            @inbounds for (k, name) in enumerate(cnames)
                slot = layout.name_to_idx[name]
                uf_pos = findfirst(==(slot), layout.unfrozen_idx)
                uf_pos === nothing && continue
                lo, hi = bounds(layout.unfrozen_priors[uf_pos])
                Cdraw[k] = clamp(Cdraw[k], lo, hi)
                new_values[slot] = convert(T, Cdraw[k])
            end
            log_q_fwd = _ad_mvn_logq(Cdraw, C_ols, F, inflate)
        end
    end
    # Periodogram-informed PERIOD for period-bearing kernels; every other
    # parameter of the model still comes from its prior. `theta` has `nm`
    # inactive here, which is the residual the death side reconstructs.
    gp_hints = ad_informed ? NamedTuple[] :
               _gp_hints(theta, data, nm, layout, instruments)
    hint_of(sl) = (for h in gp_hints; h.slot == sl && return h; end; nothing)
    if !ad_informed
        for name in nm_names
            haskey(layout.name_to_idx, name) || continue
            slot = layout.name_to_idx[name]
            uf_pos = findfirst(==(slot), layout.unfrozen_idx)
            hh = hint_of(slot)
            if hh !== nothing
                v = _gp_informed_draw(rng, hh)
                lq = _gp_informed_logq(v, hh)
                isfinite(lq) || return (theta, convert(T, -Inf))
                new_values[slot] = convert(T, v)
                log_q_fwd += lq
            elseif uf_pos !== nothing
                prior = layout.unfrozen_priors[uf_pos]
                v = rand(rng, prior.dist)
                lo, hi = bounds(prior)
                v = clamp(v, lo, hi)
                new_values[slot] = convert(T, v)
                log_q_fwd += eval_packed_logpdf(v, layout.packed_priors, uf_pos)
            end
        end
    end

    activate_noise!(new_td, nm_idx)

    if scratch !== nothing
        new_theta = scratch
    else
        new_theta = Theta{T}(theta.params, new_values; td=new_td)
    end

    # Count active/inactive for proposal ratio. n_active_after is the
    # reverse-death's selection denominator, so it must count only active
    # TOGGLEABLE models — matching propose_noise_death's n_active =
    # length(active_toggle). Counting count(new_td.noise_active) instead folds
    # in always-on non-toggleable models (e.g. IndicatorFloor), which the death
    # never selects, breaking birth↔death symmetry and adding a spurious
    # +log((k+1)/k) "model-on" bias per always-on model (measured +0.21 in ln p
    # with one floor in toy_ad_ols_birth_gate.jl).
    n_inactive = length(inactive)
    n_active_after = count(toggleable) do nm
        j = findfirst(==(nm), noise_models)
        j !== nothing && is_noise_active(new_td, j)
    end
    # Selection: the forward birth picks this model among `n_inactive`, the
    # reverse death picks it among `n_active_after`, so the Hastings ratio
    # carries (1/n_active_after)/(1/n_inactive). (The opposite sign made every
    # one-model state 9× less likely than "none" at β → 0 with three
    # activatable models, i.e. (n_inactive/n_active_after)² too small.)
    log_q_ratio = log(n_inactive) - log(n_active_after) - log_q_fwd

    # The annealed move needs the pieces separately: which slot was born, the
    # forward proposal density on its parameters, and the combinatorial term.
    if parts !== nothing
        parts[] = (nm_idx = nm_idx, log_q_fwd = log_q_fwd,
                   log_comb = log(n_inactive) - log(n_active_after))
    end
    return (new_theta, convert(T, log_q_ratio))
end



"""
    _noise_activatable(td, toggleable, noise_models, exclusion_groups) -> Vector{Int}

Global indices of the inactive toggleable noise models that a birth may
switch on in `td`, under the two exclusion layers of `propose_noise_birth`:

1. typing: SequentialNoise (AR/MA) is blocked by an active CovarianceNoise;
   CovarianceNoise by any active Sequential or Covariance model;
2. user-declared exclusion groups: at most one member of each group.

A noise death counts the same set in its post-death state, so the reverse
birth's selection probability is the one the birth really uses.
"""
function _noise_activatable(td, toggleable::Vector{NoiseModel},
                            noise_models::Vector{NoiseModel},
                            exclusion_groups)
    inactive = Int[]
    for nm in toggleable
        nm_idx = findfirst(==(nm), noise_models)
        nm_idx === nothing && continue
        is_noise_active(td, nm_idx) && continue
        can_activate = true
        if nm isa SequentialNoise
            for (j, other) in enumerate(noise_models)
                if other isa CovarianceNoise && is_noise_active(td, j)
                    can_activate = false
                    break
                end
            end
        elseif nm isa CovarianceNoise
            for (j, other) in enumerate(noise_models)
                if (other isa SequentialNoise || other isa CovarianceNoise) &&
                   is_noise_active(td, j)
                    can_activate = false
                    break
                end
            end
        end
        if can_activate && !isempty(exclusion_groups)
            for group in exclusion_groups
                nm in group || continue
                for other in group
                    other === nm && continue
                    other_idx = findfirst(==(other), noise_models)
                    other_idx === nothing && continue
                    if is_noise_active(td, other_idx)
                        can_activate = false
                        break
                    end
                end
                can_activate || break
            end
        end
        can_activate && push!(inactive, nm_idx)
    end
    return inactive
end

"""
    _noise_birth_q_setup(theta_off, nm_idx, data)

What the birth proposal of noise model `nm_idx` depends on, from the state
`theta_off` (the model inactive): the OLS fit of an informed
ActivityDecorrelation birth, or the data-only GP hints. Evaluate with
`_noise_birth_q_eval`. It is the density `propose_noise_birth` draws from.
"""
function _noise_birth_q_setup(theta_off::Theta, nm_idx::Int, data)
    nm = theta_off.params.config.noise_models[nm_idx]
    layout = theta_off.params.layout
    instruments = theta_off.params.config.instruments
    if AD_INFORMED_BIRTH[] && nm isa ActivityDecorrelation && data !== nothing
        cnames, C_ols, F = ad_ols_fit(theta_off, data, nm)
        if F !== nothing && length(C_ols) == length(cnames) && !isempty(cnames)
            return (ad = true, cnames = cnames, C_ols = C_ols, F = F,
                    hints = NamedTuple[])
        end
    end
    return (ad = false, cnames = String[], C_ols = Float64[], F = nothing,
            hints = _gp_hints(theta_off, data, nm, layout, instruments))
end

"Log-density of noise model `nm_idx`'s birth proposal at its values in `values`."
function _noise_birth_q_eval(setup, theta::Theta, nm_idx::Int, values::AbstractVector)
    layout = theta.params.layout
    if setup.ad
        Ccur = Float64[values[layout.name_to_idx[nm]] for nm in setup.cnames]
        return Float64(_ad_mvn_logq(Ccur, setup.C_ols, setup.F, 2.0))
    end
    nm = theta.params.config.noise_models[nm_idx]
    lq = 0.0
    for name in noise_param_names(nm, theta.params.config.instruments)
        haskey(layout.name_to_idx, name) || continue
        slot = layout.name_to_idx[name]
        uf_pos = findfirst(==(slot), layout.unfrozen_idx)
        hh = nothing
        for h in setup.hints
            h.slot == slot && (hh = h; break)
        end
        if hh !== nothing
            l = _gp_informed_logq(values[slot], hh)
            isfinite(l) || return -Inf
            lq += l
        elseif uf_pos !== nothing
            lq += Float64(eval_packed_logpdf(values[slot], layout.packed_priors, uf_pos))
        end
    end
    return lq
end
