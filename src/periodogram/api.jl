# Top-level RV-periodogram API.
#
#   find_rv_planets(t, y, err; indicators, method=:gls, …) -> PgramSet
#
# `indicators` is a `Dict{String, Tuple{Vector, Vector}}` mapping name →
# (values, errors) on the *same* `t` grid as the RV. Each indicator gets
# its own periodogram; the result also includes a window function
# computed via the user-configurable `window_def` (default `:gls_tt`
# matching Vines+ 2023's HD 18599 paper — see Mortier+ 2015 §3 for the
# canonical alternative).

"""
    find_rv_planets(t, y, err; indicators=Dict(), method=:gls,
                     period_min=1.0, period_max=nothing,
                     n_freqs=nothing, samples_per_peak=5,
                     window_def=:gls_tt,
                     fap_levels=[0.1, 0.05, 0.01, 0.001],
                     max_candidates=5, kwargs...) -> PgramSet

Compute a periodogram of the RV (and optional activity indicators) and
a window function on a shared frequency grid.

# Arguments
- `t`, `y`, `err` — RV time series (BJD, m/s, m/s).
- `indicators::Dict{String, Tuple}` — `"BIS" => (bis, σ_bis)` etc. The
  ordered name list is preserved in the output `PgramSet` so stacked
  plots reproduce the same order across calls.

# Keyword arguments
- `method ∈ (:gls, :bgls, :sbgls, :l1)` — periodogram estimator. Same
  estimator is applied to every channel + the window function.
- `period_min`, `period_max` — search range (days). `period_max`
  defaults to the data baseline.
- `n_freqs` — explicit grid size; otherwise auto from
  `samples_per_peak / baseline`.
- `window_def ∈ (:gls_tt, :unit_signal)` — `:gls_tt` is GLS(t, t),
  matching the HD 18599 paper (Vines+ 2023). `:unit_signal` is the
  canonical Roberts+ 1987 window function `GLS(t, ones)`. The two peak
  at the same daily aliases for typical RV cadences but differ in
  low-frequency power.
- Extra `kwargs...` forwarded to the chosen estimator (e.g. `alpha` for
  `:l1`, `n_min` for `:sbgls`).

Returns `PgramSet` with `.rv`, `.indicators` (ordered), `.window`.
"""
function find_rv_planets(t::AbstractVector{<:Real},
                          y::AbstractVector{<:Real},
                          err::AbstractVector{<:Real};
                          indicators::AbstractDict = Dict{String, Any}(),
                          method::Symbol = :gls,
                          period_min::Real = 1.0,
                          period_max::Union{Nothing, Real} = nothing,
                          n_freqs::Union{Nothing, Int} = nothing,
                          samples_per_peak::Int = 50,
                          window_def::Symbol = :gls_tt,
                          fap_levels::Vector{<:Real} = [0.1, 0.05, 0.01, 0.001],
                          max_candidates::Int = 5,
                          kwargs...)
    length(t) == length(y) == length(err) ||
        throw(ArgumentError("t, y, err length mismatch"))
    method in (:gls, :bgls, :sbgls, :l1) ||
        throw(ArgumentError("method must be :gls | :bgls | :sbgls | :l1"))
    window_def in (:gls_tt, :unit_signal) ||
        throw(ArgumentError("window_def must be :gls_tt | :unit_signal"))

    t_f   = collect(Float64, t)
    y_f   = collect(Float64, y)
    err_f = collect(Float64, err)
    T = maximum(t_f) - minimum(t_f)

    pgram_kwargs = (; period_min, period_max, n_freqs, samples_per_peak,
                      max_peaks = max_candidates, kwargs...)

    rv_pg = _run(method, t_f, y_f, err_f; pgram_kwargs..., fap_levels)

    # Indicators — same method, same kwargs, but FAPs only meaningful
    # for GLS so we pass them through unconditionally and the non-GLS
    # branches silently ignore.
    inds = Pair{String, Periodogram}[]
    for (name, val) in indicators
        if val isa Tuple || val isa NamedTuple
            v = collect(Float64, val[1])
            e = length(val) >= 2 ? collect(Float64, val[2]) :
                fill(median(abs.(v .- median(v))) * 1.4826, length(v))
        else
            v = collect(Float64, val)
            e = fill(median(abs.(v .- median(v))) * 1.4826, length(v))
        end
        length(v) == length(t_f) ||
            error("indicator `$name` length $(length(v)) != n_rv $(length(t_f))")
        pg_i = _run(method, t_f, v, e; pgram_kwargs..., fap_levels)
        push!(inds, name => pg_i)
    end

    # Window function — GLS regardless of `method`. By default GLS(t,t)
    # matching the HD 18599 paper convention; canonical Roberts+ 1987
    # via `window_def = :unit_signal`.
    win_y, win_e = if window_def === :gls_tt
        (copy(t_f), ones(length(t_f)))   # GLS(t, t) — t as the signal
    else
        (ones(length(t_f)), ones(length(t_f)))   # canonical unit signal
    end
    win = gls_periodogram(t_f, win_y, win_e;
                           period_min = period_min,
                           period_max = period_max,
                           n_freqs = n_freqs,
                           samples_per_peak = samples_per_peak,
                           fap_levels = fap_levels,
                           max_peaks = max_candidates)

    return PgramSet(method, rv_pg, inds, win, length(t_f), T)
end


# Method dispatch by Symbol — central place to add a new periodogram.
function _run(method::Symbol, t, y, err; kwargs...)
    if method === :gls
        return gls_periodogram(t, y, err; kwargs...)
    elseif method === :bgls
        return bgls_periodogram(t, y, err; kwargs...)
    elseif method === :sbgls
        return sbgls_periodogram(t, y, err; kwargs...)
    elseif method === :l1
        return l1_periodogram_compute(t, y, err; kwargs...)
    else
        error("unsupported method `$method`")
    end
end
