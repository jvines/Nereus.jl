# The last reporting step of a run, shared by run_job and the fit_* route: the
# planet-count block of a trans-dim run, ONE convergence verdict that every
# convergence field of the summary is written from, and the reliability flag of
# the evidence. Reporting only: no chain, proposal or acceptance rule is touched.

# =====================================================================
# Trans-dim: the planet-count block
# =====================================================================

# (planets?, inclusion prior) of the run, from its config.
function _td_run_settings(cfg, engine::AbstractString)
    planets = true
    ip = 0.5
    if cfg isa AbstractDict
        td = _get(cfg, :transdim; default = nothing)
        td isa AbstractDict && (planets = _get(td, :planets; default = true) !== false)
        s = _get(cfg, :sampler; default = nothing)
        kw = s isa AbstractDict ? _get(s, :kwargs; default = nothing) : nothing
        v = kw isa AbstractDict ? _get(kw, :inclusion_prior; default = nothing) : nothing
        v isa Real && 0 < v < 1 && (ip = Float64(v))
    end
    # rjmcmc has no activation prior: the same as 0.5.
    engine == "rjmcmc" && (ip = 0.5)
    return (; planets, inclusion_prior = ip)
end

"""
    _transdim_k_summary!(summary, chains, params, result; cfg, engine, n_walkers)

For a chain with `n_planets`: `run_info["transdim"]` (the prior on k, occupancy
per k, ln B between adjacent k, the mixture evidence, k_max truncation and
occupancy health: `transdim_k_report`), the accepted k transitions in
`diagnostics["k_transitions"]` and `run_info["transdim"]["k_transitions"]`, the
adjacent ln B and the prior in `model_selection`, and the warnings.
"""
function _transdim_k_summary!(summary::AbstractDict, chains, params, result;
                              cfg = nothing, engine::AbstractString = "",
                              n_walkers = nothing)
    chains isa MCMCChains.Chains || return summary
    :n_planets in names(chains, :parameters) || return summary
    np = round.(Int, _draws(chains, :n_planets))
    isempty(np) && return summary
    K = count(n -> startswith(String(n), "planet_active_"), names(chains, :parameters))
    K = max(K, maximum(np), params === nothing ? 0 : params.config.max_kplanet)
    st = _td_run_settings(cfg, engine)
    # Ordering is per planet mode: one 1/k_g! per group of same-mode slots.
    modes = params === nothing ? Any[] : collect(params.config.planet_modes)
    gs = length(modes) == K ? [count(==(m), modes) for m in unique(modes)] : [K]
    lz = get(summary, "log_z", NaN)
    ev = get(summary, "evidence", nothing)
    method = ev isa AbstractDict ? String(get(ev, "reported", "")) : ""
    rep = transdim_k_report(np, K; nchains = size(chains, 3), n_walkers,
                            inclusion_prior = st.inclusion_prior, planets = st.planets,
                            group_sizes = gs,
                            log_evidence = lz isa Real ? Float64(lz) : NaN,
                            evidence_method = method)
    mv = transdim_move_counts(result)
    # an engine with no move counters: the k changes seen in the chain
    mv === nothing && (mv = rep["transitions_observed"])
    rep["k_transitions"] = mv
    ri = get!(() -> Dict{String, Any}(), summary, "run_info")
    ri["transdim"] = rep
    dg = get!(() -> Dict{String, Any}(), summary, "diagnostics")
    dg isa AbstractDict && (dg["k_transitions"] = mv)
    ms = get!(() -> Dict{String, Any}(), summary, "model_selection")
    if ms isa AbstractDict
        ms["n_planets_prior"] = rep["prior"]
        ms["n_planets_ln_bayes_factor_adjacent"] = rep["ln_bayes_factor_adjacent"]
        ms["n_planets_mixture_evidence"] = rep["mixture_evidence"]
    end
    _warn_transdim_k(rep; engine)
    return summary
end

# =====================================================================
# One convergence verdict
# =====================================================================

const _STOPPED_EARLY = ("interrupted", "requested", "max_seconds", "max_evals")

"""
    _convergence_verdict(summary) -> (; converged, failed, assessed)

The ONE rule every convergence field of a summary is written from. A run has
converged when none of the checks that looked at it failed:

- the active-conditional split-R̂ / ESS gate of `convergence_report`
  (`run_info.convergence`),
- the fit-health convergence check (`fit_health.checks.convergence`),
- the trans-dim occupancy health check (`run_info.transdim.occupancy_health`),
- a MAP fit's optimiser (`map.converged`), and
- the run having ended on its own (not `interrupted`, `requested`,
  `max_seconds` or `max_evals`).

`converged` is `nothing` when none of them assessed the run. `failed` says which
failed and why.
"""
function _convergence_verdict(summary::AbstractDict)
    failed = String[]; assessed = String[]
    ri = get(summary, "run_info", nothing)
    conv = ri isa AbstractDict ? get(ri, "convergence", nothing) : nothing
    if conv isa AbstractDict && get(conv, "assessed", false) === true
        push!(assessed, "rhat_ess")
        pass = get(conv, "rhat_ess_pass", get(conv, "pass", true))
        if pass === false
            w = get(conv, "worst_rhat", NaN); wp = get(conv, "worst_rhat_param", "?")
            e = get(conv, "min_ess", NaN); ep = get(conv, "min_ess_param", "?")
            push!(failed, "R-hat/ESS gate: $(get(conv, "n_fail", "?")) parameter(s) " *
                  "fail (worst R-hat $(round(w; digits = 3)) for $wp, min ESS " *
                  "$(round(e; digits = 0)) for $ep)")
        end
    end
    fh = get(summary, "fit_health", nothing)
    chk = fh isa AbstractDict && get(fh, "checks", nothing) isa AbstractDict ?
          get(fh["checks"], "convergence", nothing) : nothing
    if chk isa AbstractDict
        st = String(get(chk, "own_status", get(chk, "status", "")))
        if st in ("ok", "fail")
            push!(assessed, "fit_health")
            st == "fail" && push!(failed, "fit-health convergence check: " *
                  String(get(chk, "own_message", get(chk, "message", ""))))
        end
    end
    oh = ri isa AbstractDict && get(ri, "transdim", nothing) isa AbstractDict ?
         get(ri["transdim"], "occupancy_health", nothing) : nothing
    if oh isa AbstractDict
        push!(assessed, "occupancy")
        get(oh, "ok", true) === false && push!(failed,
            "occupancy health: " * join(get(oh, "problems", String[]), "; "))
    end
    mp = get(summary, "map", nothing)
    if mp isa AbstractDict && haskey(mp, "converged")
        push!(assessed, "map")
        mp["converged"] === false && push!(failed,
            "MAP optimiser did not converge (or railed): its Laplace evidence is " *
            "taken at a point that is not the posterior mode")
    end
    stopped = get(summary, "stopped", nothing)
    if stopped isa AbstractString
        push!(assessed, "stop")
        stopped in _STOPPED_EARLY && push!(failed,
            "the run was stopped early (\"$stopped\") before it finished on its own")
    end
    converged = isempty(assessed) ? nothing : isempty(failed)
    return (; converged, failed, assessed)
end

"""
    _apply_convergence_verdict!(summary) -> summary

Write the verdict (`_convergence_verdict`) into every convergence field:
`converged` and `convergence_verdict` at the top, `run_info.convergence.pass`
and `.converged`, `fit_health.converged` and the status of
`fit_health.checks.convergence`, so the fields can never disagree. What the
R-hat/ESS gate and the fit-health check each said on their own is kept as
`rhat_ess_pass` and `own_status`. Idempotent.
"""
function _apply_convergence_verdict!(summary::AbstractDict)
    v = _convergence_verdict(summary)
    summary["converged"] = v.converged
    summary["convergence_verdict"] = Dict{String, Any}(
        "converged" => v.converged, "failed" => v.failed, "assessed_by" => v.assessed,
        "rule" => "converged = no assessed check failed: R-hat/ESS gate " *
                  "(convergence_report), fit-health convergence, trans-dim occupancy " *
                  "health, MAP optimiser, run ended on its own")
    ri = get(summary, "run_info", nothing)
    conv = ri isa AbstractDict ? get(ri, "convergence", nothing) : nothing
    if conv isa AbstractDict
        haskey(conv, "rhat_ess_pass") || !haskey(conv, "pass") ||
            (conv["rhat_ess_pass"] = conv["pass"])
        v.converged === nothing || (conv["pass"] = v.converged)
        conv["converged"] = v.converged
        conv["failed_checks"] = v.failed
    end
    fh = get(summary, "fit_health", nothing)
    if fh isa AbstractDict
        fh["converged"] = v.converged
        chk = get(fh, "checks", nothing) isa AbstractDict ?
              get(fh["checks"], "convergence", nothing) : nothing
        if chk isa AbstractDict
            haskey(chk, "own_status") || (chk["own_status"] = chk["status"];
                                          chk["own_message"] = chk["message"])
            if v.converged === false && chk["own_status"] == "ok"
                chk["status"] = "fail"
                chk["message"] = "NOT converged by the run verdict: " *
                                 join(v.failed, " | ")
                old = Symbol(get(fh, "overall", "ok"))
                old in keys(_STATUS_RANK) || (old = :ok)
                fh["overall"] = String(_worst(old, :fail))
            end
        end
    end
    return summary
end

# =====================================================================
# Evidence reliability
# =====================================================================

"""
    _evidence_reliability!(summary) -> summary

When the run reports an evidence (`log_z`, or any estimator in `evidence`):
`evidence_reliable` beside it, in `summary`, `summary["evidence"]` and
`run_info`. It is `false` when the run has not converged (`_convergence_verdict`),
with `evidence_unreliable_reasons` naming the checks that failed and, if
`log_z` is the mode-Laplace fallback, saying so; and a warning that says the
same. `nothing` when nothing assessed the run. The evidence is never withheld.
Call after `_apply_convergence_verdict!`.
"""
function _evidence_reliability!(summary::AbstractDict)
    lz = get(summary, "log_z", NaN)
    ev = get(summary, "evidence", nothing)
    has_log_z = lz isa Real && isfinite(lz)
    has_ev = ev isa AbstractDict && any(v -> (v isa Real && isfinite(v)) ||
        (v isa AbstractDict && get(v, "log_z", NaN) isa Real && isfinite(get(v, "log_z", NaN))),
        values(ev))
    (has_log_z || has_ev) || return summary
    conv = get(summary, "converged", nothing)
    verdict = get(summary, "convergence_verdict", nothing)
    failed = verdict isa AbstractDict ? String.(get(verdict, "failed", String[])) : String[]
    reasons = String[]
    rep = ev isa AbstractDict ? String(get(ev, "reported", "")) : ""
    if conv === false
        append!(reasons, failed)
        rep == "mode_laplace" && push!(reasons,
            "log_z is the mode-Laplace fallback, a Gaussian approximation at the " *
            "posterior mode that was used because no better estimate was usable")
    end
    reliable = conv === nothing ? nothing : conv === true
    summary["evidence_reliable"] = reliable
    ev isa AbstractDict && (ev["evidence_reliable"] = reliable)
    ri = get!(() -> Dict{String, Any}(), summary, "run_info")
    if ri isa AbstractDict
        ri["evidence_reliable"] = reliable
        tdi = get(ri, "transdim", nothing)
        tdi isa AbstractDict && haskey(tdi, "mixture_evidence") &&
            (tdi["mixture_evidence"]["evidence_reliable"] = reliable)
    end
    ms = get(summary, "model_selection", nothing)
    ms isa AbstractDict && haskey(ms, "n_planets_mixture_evidence") &&
        (ms["n_planets_mixture_evidence"]["evidence_reliable"] = reliable)
    if conv === false
        summary["evidence_unreliable_reasons"] = reasons
        ev isa AbstractDict && (ev["unreliable_reasons"] = reasons)
        ri isa AbstractDict && (ri["evidence_unreliable_reasons"] = reasons)
        what = has_log_z ? "log_z = $(round(lz; digits = 2))" : "the evidence"
        rep == "" || (what *= " (reported estimator: $rep)")
        @warn "Nereus: the evidence is UNRELIABLE: the fit has not converged, so " *
              "$what is not a quotable evidence. It is reported, not withheld; do " *
              "not use it for model comparison until the fit converges.\n  " *
              join(["why: $r" for r in reasons], "\n  ") log_z = lz
    elseif conv === nothing
        summary["evidence_reliable_note"] =
            "convergence was not assessed for this engine, so the evidence is " *
            "neither flagged reliable nor unreliable"
    end
    return summary
end

"""
    _finalize_run_reporting!(summary, chains, params, result; cfg, engine, n_walkers)

The reporting that needs the whole summary: the trans-dim planet-count block,
the one convergence verdict, the evidence reliability. Fail-soft.
"""
function _finalize_run_reporting!(summary::AbstractDict, chains, params, result;
                                  cfg = nothing, engine::AbstractString = "",
                                  n_walkers = nothing)
    try
        _transdim_k_summary!(summary, chains, params, result; cfg, engine, n_walkers)
    catch err
        @warn "trans-dim planet-count report failed (continuing)" exception = (err, catch_backtrace())
        summary["transdim_report_error"] = sprint(showerror, err)
    end
    try
        _apply_convergence_verdict!(summary)
        _evidence_reliability!(summary)
    catch err
        @warn "convergence verdict / evidence reliability failed (continuing)" exception = (err, catch_backtrace())
        summary["convergence_verdict_error"] = sprint(showerror, err)
    end
    return summary
end
