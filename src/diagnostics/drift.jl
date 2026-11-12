# Collective drift of an ensemble sampler's walkers: a move of the whole
# ensemble through the recorded steps that R-hat and ESS cannot see.
#
# R-hat compares walkers with each other. When every walker drifts the same way
# -- a share of them draining out of a ridge that burn-in left them on, all at
# the same slow rate -- the walkers agree with each other at every step, R-hat
# stays at 1 and the ESS is large, while the ensemble as a whole is still
# moving. The test here looks at the ensemble itself: per parameter, the mean
# over walkers of the rank-normalised draws at each recorded step, and the
# least-squares slope of that series against the step. A stationary ensemble
# has no trend; the slope's standard error comes from the long-run variance of
# the detrended series (its variance times its integrated autocorrelation time,
# Geyer's initial positive sequence), so the walkers' own autocorrelation and
# the slow collective modes every ensemble has are allowed for.
#
# Calibration, read folded (default transit-only fit, six seeds; three RV +
# Hipparcos ridge fits on pt_emcee, the trans-dim sampler and four `ensemble`
# chains; a two-planet RV fit; joint RV + transit; a period alias; each with the
# stretch move alone and with the DE move at several shares and settings, 100+
# runs). Two runs, both stretch alone, were off the nested-sampling posterior
# and moving: the seed that kept 47 % of its draws on the constant-duration
# ridge (nested: 1.4 %) reaches |z| = 15.4, and a joint fit whose first 1,000
# recorded steps held the TESS q1 at 0.308 ± 0.005 (nested: 0.297), and whose
# next 3,000 sat stationary about 0.297, 12.3. Every other run reaches 9.4 at
# most over all parameters (7.2 with the stretch move alone, 5.5 at the default
# DE share). The threshold sits between, at 10. Read unfolded, a cyclic
# inclination whose walkers cross its fold reached 17.3 with the posterior
# unmoved (8 of 22 ridge runs over 10); folded, 2.4. A run flagged here is not
# stationary; one not flagged may still be wrong in ways no single run shows: a
# mode never found, a share of walkers held on a ridge that does not drain
# within the recorded steps (transit fits with 13-16 % of their draws above
# e = 0.5, against 1.4 %, reached 5.5), or walkers that do not mix with each
# other, which is R-hat's to see.

"""
    DRIFT_Z

The |z| of an ensemble-mean trend above which [`ensemble_drift`](@ref) flags a
parameter.
"""
const DRIFT_Z = 10.0

# Below this many recorded steps the long-run variance is too poorly estimated
# for the test to mean anything; no verdict.
const _DRIFT_MIN_STEPS = 100
# The longest lag the autocorrelation sum reaches, as a share of the steps.
const _DRIFT_MAXLAG_FRAC = 1 / 3

# Integrated autocorrelation time of `r` by Geyer's initial positive (and
# monotone) sequence, at least 1.
function _drift_tau(r::AbstractVector{Float64})
    n = length(r)
    μ = sum(r) / n
    v = 0.0
    @inbounds for x in r; v += (x - μ)^2; end
    v /= n
    v > 0 || return 1.0
    ac(k) = begin
        s = 0.0
        @inbounds for i in 1:(n - k); s += (r[i] - μ) * (r[i + k] - μ); end
        s / n / v
    end
    maxlag = max(2, floor(Int, n * _DRIFT_MAXLAG_FRAC))
    τ = -1.0; k = 0; prev = Inf
    while k + 1 <= maxlag
        g = ac(k) + ac(k + 1)
        g <= 0 && break
        g = min(g, prev); prev = g
        τ += 2g; k += 2
    end
    return max(τ, 1.0)
end

# The trend z of one parameter's draws X[step, walker], and the means of the
# early and the late half with the shift between them in posterior sd. `ws` is
# (x, z, p) scratch, resized to T·W.
function _drift_one(X::AbstractMatrix{Float64}, ws)
    T, W = size(X)
    n = T * W
    x, z, p = ws
    length(x) == n || (resize!(x, n); resize!(z, n); resize!(p, n))
    copyto!(x, X)
    lo, hi = extrema(x)
    lo == hi && return nothing                     # fixed, or parked throughout
    sortperm!(p, x)
    _rank_scores!(z, x, p)
    m = zeros(T)
    @inbounds for w in 1:W, s in 1:T
        m[s] += z[s + (w - 1) * T]
    end
    m ./= W
    c = (T + 1) / 2
    stt = 0.0; stm = 0.0
    @inbounds for s in 1:T
        stt += (s - c)^2; stm += (s - c) * m[s]
    end
    b = stm / stt
    μ = sum(m) / T
    r = [m[s] - μ - b * (s - c) for s in 1:T]
    v = sum(abs2, r) / (T - 1)
    lrv = v * _drift_tau(r)
    zt = lrv > 0 ? b / sqrt(lrv / stt) : 0.0
    h = T ÷ 2
    early = Statistics.mean(view(X, 1:h, :)); late = Statistics.mean(view(X, (h + 1):T, :))
    sd = Statistics.std(x)
    return (; z = zt, early, late, shift_sd = sd > 0 ? (late - early) / sd : 0.0)
end

"""
    ensemble_drift(cube, names; threshold = DRIFT_Z, groups = nothing) -> Dict or nothing

The collective-drift check of an ensemble sampler's β = 1 walkers, which
R-hat and ESS cannot see (src/diagnostics/drift.jl). `cube[step, param,
walker]` holds the recorded draws, `names` the parameter names. `groups`
splits the walkers into independent ensembles (`ensemble` chains), each tested
on its own; the largest |z| per parameter is reported.

Returns `nothing` with fewer than $(_DRIFT_MIN_STEPS) recorded steps, else a
`Dict` with `"flagged"` (any parameter over `threshold`), `"threshold"`,
`"max_z"`, `"max_parameter"`, `"z"` (every tested parameter's trend z),
`"drifted"` (per flagged parameter, strongest first: its z, the mean of the
early and the late half of the steps, and the shift between them in posterior
standard deviations), `"n_steps"` and `"n_walkers"`.
"""
function ensemble_drift(cube::AbstractArray{<:Real,3}, names;
                        threshold::Real = DRIFT_Z, groups = nothing)
    T, P, W = size(cube)
    T >= _DRIFT_MIN_STEPS || return nothing
    gs = groups === nothing ? [1:W] : collect(groups)
    nmax = T * maximum(length, gs)
    ws = (Vector{Float64}(undef, nmax), Vector{Float64}(undef, nmax),
          Vector{Int32}(undef, nmax))
    zs = Dict{String, Float64}()
    rows = Dict{String, Any}[]
    for j in 1:P
        best = nothing
        for g in gs
            Xg = Matrix{Float64}(undef, T, length(g))
            @inbounds for (i, w) in enumerate(g), s in 1:T
                Xg[s, i] = cube[s, j, w]
            end
            all(isfinite, Xg) || continue
            r = _drift_one(Xg, ws)
            r === nothing && continue
            (best === nothing || abs(r.z) > abs(best.z)) && (best = r)
        end
        best === nothing && continue
        nm = String(names[j])
        zs[nm] = best.z
        abs(best.z) > threshold && push!(rows, Dict{String, Any}(
            "parameter" => nm, "z" => best.z, "early_mean" => best.early,
            "late_mean" => best.late, "shift_sd" => best.shift_sd))
    end
    isempty(zs) && return nothing
    sort!(rows; by = r -> -abs(r["z"]))
    kmax = argmax(k -> abs(zs[k]), collect(keys(zs)))
    return Dict{String, Any}("flagged" => !isempty(rows), "threshold" => Float64(threshold),
        "max_z" => zs[kmax], "max_parameter" => kmax, "z" => zs, "drifted" => rows,
        "n_steps" => T, "n_walkers" => W)
end

# The warning for a flagged run: what drifted, by how much, and what to do --
# with the DE move named when the run had it off (`de_move` = 0).
function _drift_warn(d, sampler::AbstractString; de_move = nothing)
    (d === nothing || !d["flagged"]) && return d
    shown = first(d["drifted"], 4)
    what = join([@sprintf("%s (z = %.1f; mean %.4g → %.4g, %+.2f sd)", r["parameter"],
                          r["z"], r["early_mean"], r["late_mean"], r["shift_sd"])
                 for r in shown], ", ")
    more = length(d["drifted"]) > length(shown) ?
           " and $(length(d["drifted"]) - length(shown)) more" : ""
    de_off = de_move !== nothing && de_move == 0 ?
             " This run had de_move = 0, the stretch move alone, whose walkers relax " *
             "slowly along a thin or curved ridge; rerun with the DE move on " *
             "(de_move > 0, default $(DE_MOVE_DEFAULT))." : ""
    @warn "$sampler: the β = 1 walkers drifted together through the recorded " *
          "steps in $what$more, from the first half of the steps to the second. " *
          "R-hat and ESS cannot see a drift every walker shares, so these draws " *
          "are not yet the stationary posterior: run longer (more burn-in and " *
          "more steps), or rerun from a different seed or init and compare.$de_off " *
          "(Ensemble-mean trend test, |z| > $(d["threshold"]); " *
          "summary diagnostics \"drift\".)"
    return d
end

# Rows (s - 1) W + w of `samples[1:keep_idx, :]` as cube[step, column, walker],
# the first `n_par` columns -- the sampled parameters, in layout order -- read
# folded, as the convergence gate reads them: a cyclic inclination's walkers may
# sit on either side of its fold (src/circular.jl), and walkers crossing it
# move the unfolded mean without moving the posterior.
function _drift_cube(samples::AbstractMatrix{Float64}, keep_idx::Int, W::Int, ncol::Int,
                     params, n_par::Int)
    (W > 0 && keep_idx > 0 && keep_idx % W == 0) || return nothing
    T = keep_idx ÷ W
    P = fold_cyclic_inclinations!(samples[1:keep_idx, 1:n_par], params)
    cube = Array{Float64}(undef, T, ncol, W)
    @inbounds for w in 1:W, j in 1:ncol, s in 1:T
        r = (s - 1) * W + w
        cube[s, j, w] = j <= n_par ? P[r, j] : samples[r, j]
    end
    return cube
end

# pt_emcee / pt_whitening: the cold rung's walkers, the sampled parameters.
function _pt_drift(samples, keep_idx::Int, W::Int, params; sampler::AbstractString,
                   de_move = nothing)
    names = params.layout.unfrozen_names
    cube = _drift_cube(samples, keep_idx, W, length(names), params, length(names))
    cube === nothing && return nothing
    return _drift_warn(ensemble_drift(cube, names), sampler; de_move)
end

# The trans-dim sampler: every recorded column but `lp` -- the parameters, the
# planet count and the activity indicators (a slot parameter's draws include
# its parked value while the slot is off, which a stationary run mixes in at a
# steady share).
function _td_drift(samples, keep_idx::Int, W::Int, param_names, params;
                   sampler::AbstractString, de_move = nothing)
    names = String.(param_names[1:(end - 1)])          # drop :lp
    cube = _drift_cube(samples, keep_idx, W, length(names), params,
                       length(params.layout.unfrozen_names))
    cube === nothing && return nothing
    return _drift_warn(ensemble_drift(cube, names), sampler; de_move)
end

# The `ensemble` engine: each chain's walkers are one ensemble, cube[step,
# param, walker] in the bounded parameters; read folded, as above.
function _ensemble_drift_chains(cubes::AbstractVector, params; sampler::AbstractString,
                                de_move = nothing)
    isempty(cubes) && return nothing
    W = size(cubes[1], 3)
    cube = cat(cubes...; dims = 3)
    if !isempty(cyclic_inclination_indices(params))
        cube = permutedims(fold_cyclic_inclinations!(permutedims(cube, (1, 3, 2)), params),
                           (1, 3, 2))
    end
    groups = [((c - 1) * W + 1):(c * W) for c in 1:length(cubes)]
    return _drift_warn(ensemble_drift(cube, params.layout.unfrozen_names; groups),
                       sampler; de_move)
end
