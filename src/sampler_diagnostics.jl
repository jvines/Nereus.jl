# Sampler diagnostics, persisted.
#
# Every engine computed run-health numbers and then dropped most of them on the
# floor: `_normalize` kept four fields off the result object and nothing else,
# and R-hat / ESS were computed by pt_emcee every `diag_every` steps purely to
# drive the progress bar and then discarded. The numbers you would quote to
# justify a fit were not in the output file.
#
# What is collectable differs by engine, and forcing one schema on all of them
# would be worse than collecting nothing:
#
#   * pt_emcee / pt_whitening keep walkers as the chain axis, so multi-chain
#     R-hat is meaningful. nuts runs 4 independent chains by default, likewise.
#   * pt, pt_hmc, transdim_pt_emcee, and moms/rjmcmc/ensemble/ess at their
#     default n_chains=1 return a single flat chain. ESS is still meaningful
#     within it; R-hat is NOT DEFINED and is omitted rather than faked.
#   * the nested family (nested, nested_ins, nested_dynamic, daedalus)
#     resamples to equal weight before returning. Those draws are not a Markov
#     trajectory, so neither autocorrelation-based ESS nor R-hat means what it
#     usually means. They get their evidence and iteration counts instead.
#   * pa / smc return an SMC population at beta=1, same argument.
#   * map has no chains at all; its diagnostics are trust flags.
#
# See SAMPLER_DIAGNOSTICS.md for what each sampler still discards internally.

"""Engines whose returned draws are not a Markov trajectory, so R-hat and
autocorrelation ESS do not carry their usual meaning."""
const _RESAMPLED_ENGINES = Set([
    "nested", "nested_ins", "nested_dynamic", "daedalus",  # resampled to equal weight
    "pa", "smc",                                            # SMC population at beta=1
    "ofti",                                                 # i.i.d. rejection draws
])

"""Result fields worth keeping, across every engine. Harvested by reflection so
an engine that grows a field gets it persisted without touching this list
twice."""
const _DIAG_FIELDS = (
    # evidence
    :log_evidence, :log_evidence_laplace, :log_evidence_bridge,
    :log_z, :log_z_ns, :log_z_ins, :log_z_baseline, :log_z_batch,
    :log_evidence_history,
    # effort
    :n_evals, :n_iters, :n_iters_baseline, :n_iters_batch,
    # mixing
    :acceptance, :acceptance_within, :acceptance_swap, :acceptance_transdim,
    :betas, :beta_history, :ess_history, :whitening_active_after,
    # trans-dim move bookkeeping
    :td_proposed, :td_accepted, :noise_td_proposed, :noise_td_accepted,
    :planet_birth_proposed, :planet_birth_accepted,
    :planet_death_proposed, :planet_death_accepted,
    # MAP trust flags
    :converged, :railed, :railed_params, :n_basins, :dominance, :log_posterior,
    # dynamic NS targets
    :target_L_lo, :target_L_hi,
)

"""Per-parameter ESS and, when there is more than one chain, R-hat.

R-hat needs at least two chains by construction. An engine that returns one
flat chain gets ESS alone rather than a fabricated 1.0.
"""
function chain_convergence(chains)
    chains === nothing && return nothing
    try
        n_chains = size(chains, 3)
        per = Dict{String, Any}()
        e_bulk = MCMCChains.ess(chains; kind = :bulk)
        e_tail = MCMCChains.ess(chains; kind = :tail)
        names_ = String.(e_bulk.nt.parameters)
        rh = n_chains >= 2 ? MCMCChains.rhat(chains).nt.rhat : nothing
        for (i, nm) in enumerate(names_)
            entry = Dict{String, Any}("ess_bulk" => e_bulk.nt.ess[i],
                                       "ess_tail" => e_tail.nt.ess[i])
            rh === nothing || (entry["rhat"] = rh[i])
            per[nm] = entry
        end
        out = Dict{String, Any}("n_chains" => n_chains,
                                 "n_iter" => size(chains, 1),
                                 "per_parameter" => per)
        finite(v) = filter(isfinite, v)
        eb, et = finite(e_bulk.nt.ess), finite(e_tail.nt.ess)
        isempty(eb) || (out["worst_ess_bulk"] = minimum(eb))
        isempty(et) || (out["worst_ess_tail"] = minimum(et))
        if rh !== nothing
            r = finite(rh)
            isempty(r) || (out["worst_rhat"] = maximum(r))
        else
            out["rhat_note"] = "one chain: R-hat is undefined, ESS is within-chain"
        end
        return out
    catch err
        return Dict{String, Any}("error" => sprint(showerror, err))
    end
end

"""Tuple-returning engines, by what their 2nd and 3rd slots actually hold.

They do not agree with each other. `pt`, `pt_hmc`, `nested` and `daedalus` put
log evidence in slot 2; `moms` and `rjmcmc` put an EVALUATION COUNT there. The
facade took slot 2 as log_z unconditionally, so moms and rjmcmc have been
reporting a count as their evidence.
"""
const _TUPLE_SLOTS = Dict{String, Tuple{Vararg{Symbol}}}(
    "pt"        => (:log_evidence, :n_evals),      # slot 3 only on the trans-dim path
    "pt_hmc"    => (:log_evidence, :evidence),
    "nested"    => (:log_evidence,),
    "daedalus"  => (:log_evidence, :_strategy),    # slot 3 is adaptation state
    "moms"      => (:n_evals, :_strategy),         # NOT evidence
    "rjmcmc"    => (:n_evals,),                    # NOT evidence
)

"""
    sampler_diagnostics(engine, raw, chains) -> Dict

Everything this engine actually reports about its own run.
"""
function sampler_diagnostics(engine::AbstractString, raw, chains)
    d = Dict{String, Any}("engine" => String(engine))

    # Tuple returns carry their diagnostics positionally, and `_normalize`
    # dropped every one of them.
    if raw isa Tuple
        slots = get(_TUPLE_SLOTS, String(engine), ())
        for (i, name) in enumerate(slots)
            idx = i + 1
            idx <= length(raw) || break
            name === :_strategy && continue          # proposal state, not a diagnostic
            v = raw[idx]
            if name === :evidence && !(v isa Real)
                hasproperty(v, :ti) && (d["evidence"] = Dict{String, Any}(
                    "ti" => Dict("log_z" => v.ti[1], "sigma" => v.ti[2]),
                    "ti_plus" => Dict("log_z" => v.ti_plus[1], "sigma" => v.ti_plus[2]),
                    "ss_plus" => Dict("log_z" => v.ss_plus[1], "sigma" => v.ss_plus[2]),
                    "hybrid" => Dict("log_z" => v.hybrid[1], "sigma" => v.hybrid[2]),
                    "hybrid_beta_star" => v.hybrid_beta_star))
            elseif v isa Real
                d[String(name)] = v
            end
        end
    end

    for f in _DIAG_FIELDS
        if raw !== nothing && hasproperty(raw, f)
            v = getproperty(raw, f)
            v === nothing || (d[String(f)] = v)
        end
    end

    # The four tempered evidence estimators, where the engine returns the
    # report rather than just its chosen headline.
    for rf in (:evidence, :evidence_report)
        if raw !== nothing && hasproperty(raw, rf)
            rep = getproperty(raw, rf)
            ev = Dict{String, Any}()
            for (k, f) in (("ti", :ti), ("ti_plus", :ti_plus),
                           ("ss_plus", :ss_plus), ("hybrid", :hybrid))
                if hasproperty(rep, f)
                    t = getproperty(rep, f)
                    ev[k] = t isa Tuple && length(t) >= 2 ?
                        Dict("log_z" => t[1], "sigma" => t[2]) : t
                end
            end
            hasproperty(rep, :hybrid_beta_star) &&
                (ev["hybrid_beta_star"] = rep.hybrid_beta_star)
            isempty(ev) || (d["evidence"] = ev)
            break
        end
    end

    # nuts hangs its diagnostics off chains.info rather than a struct field.
    if chains !== nothing
        try
            info = chains.info
            for k in (:n_divergent, :step_size, :mean_tree_depth,
                      :max_tree_depth, :mean_accept)
                haskey(info, k) && (d[String(k)] = getproperty(info, k))
            end
        catch
        end
    end

    # Swap acceptance is the number that decides whether a tempered run can be
    # trusted: the ladder fails SILENTLY, and a cold chain stuck in one mode is
    # beautifully converged to that mode. Surface the minimum so no caller has
    # to know to take it.
    if haskey(d, "acceptance_swap")
        a = collect(Float64, d["acceptance_swap"])
        isempty(a) || (d["min_swap"] = minimum(a))
    end
    if haskey(d, "acceptance_within")
        a = collect(Float64, d["acceptance_within"])
        isempty(a) || (d["cold_acceptance"] = a[1])   # the beta = 1 rung
    end

    if String(engine) in _RESAMPLED_ENGINES
        d["convergence_note"] = "draws are resampled/i.i.d., not a Markov " *
            "trajectory: R-hat and autocorrelation ESS are not reported"
    else
        c = chain_convergence(chains)
        c === nothing || (d["convergence"] = c)
    end
    return d
end
