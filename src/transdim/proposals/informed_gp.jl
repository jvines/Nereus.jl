# Periodogram-informed birth for period-bearing noise kernels.
#
# OFF by default: measured WORSE than blind at every swap rate. Kept because the
# machinery is needed elsewhere and it may help where a planet is not dominating
# the periodogram, but it does not earn being on.
#
# Split out of the former single proposals.jl (2035 lines). Pure move: no logic
# changed, and the split was verified to reconstruct the original byte for byte.

# =====================================================================
# Periodogram-informed birth for period-bearing noise kernels
# =====================================================================
#
# The AD block above fixes ActivityDecorrelation. Every OTHER toggleable model
# — including the whole GP family — fell through to a blind prior draw, and a
# prior-drawn quasi-periodic kernel essentially never lands on the rotation
# mode. That breaks the occupancy in BOTH directions: a flexible model that is
# hard to ENTER under-represents, and one that is hard to LEAVE over-represents
# (measured: GP-Rot held 0.3% at a 14.8-nat gap, ~4 orders above its evidence;
# and AD<->GP-Rot occupancy missed the reference dlogZ by 6.79 nats).
#
# Only the PERIOD is informed. It is the parameter that decides whether the
# kernel is even in the right basin; amplitude and shape are low-dimensional
# and bounded, and a prior draw for them costs little. The hint is the
# Lomb-Scargle periodogram of the model-INACTIVE residual, which is the same
# quantity on both sides of the move (birth: `theta` has it inactive; death:
# `new_theta` has it deactivated) — that symmetry is what keeps it DB-correct.
#
# MaternGP is deliberately absent: `matern_rho` is a correlation length, not a
# period, so a periodogram peak is the wrong hint for it.

# (stub, coordinate) of the PERIOD parameter. Coordinate says how the stored
# parameter relates to the log-period the peaks live in.
_gp_period_param(::CeleriteRotation)     = ("gp_period",     :log)
_gp_period_param(::CeleriteRotationFM17) = ("gp_log_period", :ident)
_gp_period_param(::CeleriteSHO)          = ("gp_log_omega0", :omega)
_gp_period_param(::HarmonicBlock)        = ("harm_period",   :log)
_gp_period_param(::ActivityGP)           = ("gp_act_period", :log)
_gp_period_param(::NoiseModel)           = nothing

# (stub, coordinate) of the AMPLITUDE parameter, informed by the data scatter.
#
# Informing the period ALONE measurably changed nothing (AD<->GP-Rot discrepancy
# 3.49 -> 3.62 nats off, i.e. inside seed noise). The diagnosis: a death's
# Hastings ratio gains q_rev as a PRODUCT over every parameter of the killed
# model, so lifting one factor is irrelevant if another is the small one. For a
# GP the binding factor is the amplitude — a blind draw from a log-uniform
# amplitude prior lands in the right decade only rarely, whereas the data tell
# you almost exactly how much variance there is to explain.
#
# SHO is absent on purpose: its k(0) = S0*omega0*Q, so S0 is not an amplitude on
# its own and cannot be informed without the other two. ActivityGP likewise
# spreads amplitude over four couplings (Vc, Vr, Bc, Br).
_gp_amp_param(::CeleriteRotation)     = ("gp_sigma",      :log)
_gp_amp_param(::CeleriteRotationFM17) = ("gp_log_amp",    :ident)
_gp_amp_param(::MaternGP)             = ("matern_sigma",  :log)
_gp_amp_param(::HarmonicBlock)        = ("harm_amp_A",    :log)
_gp_amp_param(::NoiseModel)           = nothing

# Parameter value -> (transformed coordinate, log|d transformed / d value|).
# The Jacobian converts a density defined in the transformed coordinate into a
# density in the parameter's own coordinate; get it wrong and the Hastings
# ratio is silently biased.
_gp_logP_of(v, ::Val{:log})   = (log(v), -log(v))
_gp_logP_of(v, ::Val{:ident}) = (v, 0.0)
_gp_logP_of(v, ::Val{:omega}) = (log(2π) - v, 0.0)
_gp_val_of(t, ::Val{:log})   = exp(t)
_gp_val_of(t, ::Val{:ident}) = t
_gp_val_of(t, ::Val{:omega}) = log(2π) - t

# Width of the amplitude hint, in natural logs. Deliberately loose (~a factor
# of e either way): the scatter tells you the ORDER of the amplitude, not its
# value, since the GP shares the variance with jitter and the mean model.
const _GP_AMP_LOGSIGMA = 1.0

const _GP_HINT_CACHE = Dict{UInt64, Tuple{Vector{Float64}, Vector{Float64}}}()
const _GP_HINT_LOCK  = ReentrantLock()

"Robust scatter of the RV data — the amplitude scale a noise model can explain."
function _gp_data_scatter(data)
    y = Vector{Float64}(data.rv)
    m = median(y)
    return max(1.4826 * median(abs.(y .- m)), 1e-6)
end

"""
    _gp_hints(theta, data, nm, layout, instruments) -> Vector

Informed-proposal hints for `nm`'s period and amplitude parameters. Each entry
carries the peaks, weights and widths of a mixture in the transformed
coordinate, plus that coordinate's Jacobian tag. Empty when nothing can be
informed, in which case every parameter falls back to its prior.

Both hints are built from the DATA only, never from `theta`. That is what makes
them free (one cached periodogram per dataset) AND exactly reversible: a birth
and its reverse death evaluate an identical fixed density, with no staleness
argument required. The planet path's refresh-every-N-attempts cache cannot give
that guarantee — a birth and death straddling a refresh see different densities.
"""
# OFF by default. Measured on the AD<->GP-Rot occupancy gate it made things
# consistently WORSE at every swap rate (3.86/2.69/2.29 nats vs 3.49/2.18/1.56
# blind) — detailed balance is exact, so that is degraded mixing, not bias:
# a proposal concentrated near the data peak keeps re-offering values the chain
# already holds, and those get rejected. The annealed birth supersedes it by
# relaxing the newborn instead of trying to guess it right first time.
const GP_INFORMED_BIRTH = Ref(false)

# ON by default — the OLS-informed AD birth is the one informed proposal that
# demonstrably works. Switchable so a benchmark can force BLIND births and
# measure the blind-birth pathology in isolation, which is otherwise impossible
# to separate from everything else going on in a trans-dim run.
const AD_INFORMED_BIRTH = Ref(true)

function _gp_hints(theta::Theta, data::Union{Data,Nothing}, nm,
                   layout, instruments)
    hints = NamedTuple[]
    GP_INFORMED_BIRTH[] || return hints
    data === nothing && return hints
    (hasproperty(data, :t_rv) && data.t_rv !== nothing &&
     length(data.t_rv) > 3) || return hints
    names = noise_param_names(nm, instruments)

    function locate(stub)
        for n in names
            (n == stub || startswith(n, stub * "_")) || continue
            haskey(layout.name_to_idx, n) || return nothing
            slot = layout.name_to_idx[n]
            uf = findfirst(==(slot), layout.unfrozen_idx)
            uf === nothing && return nothing
            lo, hi = bounds(layout.unfrozen_priors[uf])
            return (slot = slot, lo = lo, hi = hi)
        end
        return nothing
    end

    # --- period: Lomb-Scargle peaks of the (mean-subtracted) data -----------
    spec = _gp_period_param(nm)
    if spec !== nothing
        stub, kind = spec
        loc = locate(stub)
        if loc !== nothing
            a, _ = _gp_logP_of(loc.lo, Val(kind))
            b, _ = _gp_logP_of(loc.hi, Val(kind))
            tmin, tmax = minmax(a, b)
            if isfinite(tmin) && isfinite(tmax) && tmax > tmin
                key = hash((objectid(data), tmin, tmax))
                pk = lock(_GP_HINT_LOCK) do
                    length(_GP_HINT_CACHE) > 256 && empty!(_GP_HINT_CACHE)
                    get!(_GP_HINT_CACHE, key) do
                        y = Vector{Float64}(data.rv); y .-= sum(y) / length(y)
                        P, w, _, _ = _find_peaks(Vector{Float64}(data.t_rv), y,
                                                 exp(tmin), exp(tmax);
                                                 t_ref = data.t_ref,
                                                 σ = data.rv_err,
                                                 max_nfreq = _SWAP_NFREQ)
                        (isempty(P) ? Float64[] : log.(P), w)
                    end
                end
                if !isempty(pk[1])
                    push!(hints, (slot = loc.slot, kind = kind,
                                  mu = pk[1], w = pk[2],
                                  sig = fill(INFORMED_SIGMA_P, length(pk[1])),
                                  tmin = tmin, tmax = tmax,
                                  lo = loc.lo, hi = loc.hi))
                end
            end
        end
    end

    # --- amplitude: one "peak" at the data scatter --------------------------
    aspec = _gp_amp_param(nm)
    if aspec !== nothing
        stub, kind = aspec
        loc = locate(stub)
        if loc !== nothing
            a, _ = _gp_logP_of(loc.lo, Val(kind))
            b, _ = _gp_logP_of(loc.hi, Val(kind))
            tmin, tmax = minmax(a, b)
            μ = log(_gp_data_scatter(data))
            if isfinite(tmin) && isfinite(tmax) && tmax > tmin
                push!(hints, (slot = loc.slot, kind = kind,
                              mu = [clamp(μ, tmin, tmax)], w = [1.0],
                              sig = [_GP_AMP_LOGSIGMA],
                              tmin = tmin, tmax = tmax,
                              lo = loc.lo, hi = loc.hi))
            end
        end
    end
    return hints
end

"Draw the period parameter from the informed mixture."
function _gp_informed_draw(rng::AbstractRNG, h)
    t = if rand(rng) < INFORMED_ALPHA
        r = rand(rng); cum = 0.0; ci = length(h.w)
        for (i, wt) in enumerate(h.w)
            cum += wt
            if r < cum; ci = i; break; end
        end
        h.mu[ci] + h.sig[ci] * randn(rng)
    else
        h.tmin + rand(rng) * (h.tmax - h.tmin)
    end
    t = clamp(t, h.tmin, h.tmax)
    return clamp(_gp_val_of(t, Val(h.kind)), h.lo, h.hi)
end

"Density of `_gp_informed_draw` at `v`, in the parameter's own coordinate."
function _gp_informed_logq(v, h)
    t, logJ = _gp_logP_of(v, Val(h.kind))
    isfinite(t) || return -Inf
    (h.tmin <= t <= h.tmax) || return -Inf
    return _informed_log_q(t, h.mu, h.w, h.tmin, h.tmax; sigmas = h.sig) + logJ
end

