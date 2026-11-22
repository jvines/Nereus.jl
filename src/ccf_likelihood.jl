# Likelihood of CCF profiles (`data.ccf`) modelled as a sum of Gaussian line
# components; the data container is in src/ccf_data.jl, the model and its
# conventions in docs/src/ccf.md.
#
# Pixel i of epoch n of a set:
#
#   m_ni = c_n − s_n Σ_j d_j exp(−(v_i − v_j(t_n))² / (2 σ_j²)),
#   σ_j  = FWHM_j / (2√(2 ln 2)),
#   v_j  = γ_j(t_n)                    (static component)
#        = γ_j(t_n) + K_j g_k(t_n)     (component tied to planet k),
#   γ_j(t) = γ_j + γ̇_j (t − t_ref) + ½ γ̈_j (t − t_ref)²  (trend terms optional),
#   g_k  = cos(f + ω) + e cos ω,
#
#   ln L = −½ Σ_n w_n Σ_i [ (y_ni − m_ni)² / (e_ni² + j_n²) + ln 2π(e_ni² + j_n²) ],
#
# w_n = 1 under pixel_correlation = :independent, min(1, N_eff/N_pix) under
# :neff (`ccf_epoch_weights`).
#
# g_k is the unit-amplitude Keplerian of planet k at the epoch, computed from
# the planet's own P, time anchor, e and ω exactly as the RV model computes it
# (`rv_keplerian` with K = 1 from the periastron time `anchor_to_tp` gives), so
# a component tied to a transiting planet follows the transit ephemeris: ω is
# the argument of periastron of the reflex the RVs measure, and the transit is
# at f + ω = π/2 (`tc_to_tp`). K_j > 0 moves the component in phase with the
# star's reflex, K_j < 0 in antiphase (the companion's own lines).
#
# Under :independent the pixels of one CCF are treated as independent. A CCF is
# oversampled, so neighbouring pixels are correlated and this likelihood is
# overconfident by roughly the oversampling factor: posterior widths from it
# are lower bounds. The point estimates are not biased by it. :neff divides the
# information of each epoch by that factor.

const _FWHM_PER_SIGMA = 2 * sqrt(2 * log(2))

# Planet k's orbit as the RV model decodes it: (P, e, ω, Tp), or `nothing` for
# an unphysical one.
@inline function _ccf_orbit(theta::Theta{T}, data::Data, k::Int) where {T}
    pc = theta.params.config.parametrization
    P = planet_P(theta, k)
    e, w = planet_e_w(theta, k)
    (e < 0 || e >= 1 || !(P > 0) || !isfinite(P)) && return nothing
    Tp = anchor_to_tp(pc.time, planet_time_anchor(theta, k), P, e, w, data.t_ref,
                      pc.t_ref_c)
    isfinite(Tp) || return nothing
    return (P, e, w, Tp)
end

# g = cos(f + ω) + e cos ω at time t.
@inline _ccf_geom(t, orb) = rv_keplerian(t, orb[1], one(orb[1]), orb[2], orb[3],
                                         zero(orb[1]), orb[4])

# γ_j(t) of component j: γ plus its trend terms, `dt = t − trend_t_ref`. γ itself
# (not γ + 0) without a trend, so a set without trends evaluates as it always did.
@inline function _ccf_gamma_at(γ, vals, ix::CCFIndices, j::Int, dt)
    g = γ
    ix.gdot[j] == 0 || (g += vals[ix.gdot[j]] * dt)
    ix.gddot[j] == 0 || (g += vals[ix.gddot[j]] * dt * dt / 2)
    return g
end

# Per component of set `s`: (orbit, γ, K, σ, d), the orbit unset for a static one, or `nothing` when
# the parameters are unphysical or a tied planet is not active.
function _ccf_components(theta::Theta{T}, data::Data, s::CCFSet, ix::CCFIndices,
                         p_idx) where {T}
    nc = length(s.components)
    orbs = Vector{NTuple{4, T}}(undef, nc)   # unused for a static component
    γ = Vector{T}(undef, nc); K = zeros(T, nc); σ = Vector{T}(undef, nc)
    d = Vector{T}(undef, nc)
    vals = theta.values
    for (j, c) in enumerate(s.components)
        if !is_static(c)
            c.planet in p_idx || return nothing
            o = _ccf_orbit(theta, data, c.planet)
            o === nothing && return nothing
            orbs[j] = (T(o[1]), T(o[2]), T(o[3]), T(o[4]))
            K[j] = vals[ix.K[j]]
        end
        γ[j] = vals[ix.gamma[j]]
        fw = vals[ix.fwhm[j]]
        fw > 0 || return nothing
        σ[j] = fw / _FWHM_PER_SIGMA
        d[j] = vals[ix.depth[j]]
    end
    return orbs, γ, K, σ, d
end

"""
    ccf_log_likelihood(theta, data) -> Real

Gaussian log-likelihood of the CCF profiles in `data.ccf` under their line
components (see src/ccf_likelihood.jl for the model). Exactly zero without CCF
sets; `-Inf` when a width, a total variance or a depth scale is not positive,
or when a component's planet is not active or has an unphysical orbit. Added
to the total by `rv_log_likelihood`, beside the astrometry and the transit
times, so every engine that evaluates the posterior includes it.
"""
function ccf_log_likelihood(theta::Theta{T}, data::Data) where {T}
    sets = data.ccf
    isempty(sets) && return zero(T)
    S = theta.params.layout.systemic
    p_idx = planet_indices(theta)
    vals = theta.values
    ll = zero(T)
    for (si, s) in enumerate(sets)
        ix = S.ccf[si]
        comps = _ccf_components(theta, data, s, ix, p_idx)
        comps === nothing && return convert(T, -Inf)
        orbs, γ, K, σ, d = comps
        nc = length(γ)
        ctr = Vector{T}(undef, nc)
        for n in eachindex(s.t)
            dt = s.t[n] - s.trend_t_ref
            for j in 1:nc
                gj = _ccf_gamma_at(γ[j], vals, ix, j, dt)
                ctr[j] = is_static(s.components[j]) ? gj : gj + K[j] * _ccf_geom(s.t[n], orbs[j])
            end
            c = vals[ix.cont[n]]
            sc = ix.scale[n] == 0 ? one(T) : vals[ix.scale[n]]
            sc > 0 || return convert(T, -Inf)
            jit2 = vals[ix.jit[n]]^2
            w = ix.weight[n]
            if w == 1
                # accumulated straight into ll: the sum of every set without
                # :neff, pixel by pixel in the order it always was
                ll = _ccf_epoch_ll(ll, s.v[n], s.y[n], s.err[n], ctr, σ, d, c, sc, jit2)
            else
                ll += w * _ccf_epoch_ll(zero(T), s.v[n], s.y[n], s.err[n], ctr, σ, d, c,
                                        sc, jit2)
            end
            isfinite(ll) || return convert(T, -Inf)
        end
        isfinite(ll) || return convert(T, -Inf)
    end
    return ll
end

# `acc` minus the −ln L terms of one epoch's pixels; −Inf for a total variance
# that is not positive.
function _ccf_epoch_ll(acc::T, v, y, e, ctr, σ, d, c, sc, jit2) where {T}
    two_pi = T(2π)
    nc = length(ctr)
    @inbounds for i in eachindex(v)
        prof = zero(T)
        for j in 1:nc
            z = (v[i] - ctr[j]) / σ[j]
            prof += d[j] * exp(-z * z / 2)
        end
        r = y[i] - (c - sc * prof)
        var = e[i]^2 + jit2
        var > 0 || return convert(T, -Inf)
        acc -= (r * r / var + log(two_pi * var)) / 2
    end
    return acc
end

"""
    ccf_component_velocities(theta, data) -> Vector{Matrix{Float64}}

Per CCF set, the centre of every component at every epoch (km/s), an
`n_epochs × n_components` matrix; NaN where the component's planet is not
active or its orbit is unphysical. A component's γ trend
(`ccf_gdot_*`, `ccf_gddot_*`) is included.
"""
function ccf_component_velocities(theta::Theta, data::Data)
    S = theta.params.layout.systemic
    p_idx = planet_indices(theta)
    out = Matrix{Float64}[]
    for (si, s) in enumerate(data.ccf)
        ix = S.ccf[si]
        M = fill(NaN, n_ccf_epochs(s), length(s.components))
        for (j, c) in enumerate(s.components)
            γ = Float64(theta.values[ix.gamma[j]])
            vals = theta.values
            if is_static(c)
                for n in eachindex(s.t)
                    M[n, j] = Float64(_ccf_gamma_at(γ, vals, ix, j, s.t[n] - s.trend_t_ref))
                end
            else
                c.planet in p_idx || continue
                o = _ccf_orbit(theta, data, c.planet)
                o === nothing && continue
                Kj = Float64(theta.values[ix.K[j]])
                for n in eachindex(s.t)
                    gj = Float64(_ccf_gamma_at(γ, vals, ix, j, s.t[n] - s.trend_t_ref))
                    M[n, j] = gj + Kj * _ccf_geom(s.t[n], o)
                end
            end
        end
        push!(out, M)
    end
    return out
end

"""
    ccf_model(theta, data) -> Vector{NamedTuple}

Per CCF set, the model at `theta`: `(tag, labels, t, v, y, model, residual,
sigma, velocities)`, each of `v`, `y`, `model`, `residual` and `sigma` one
vector per epoch (`y` as fitted, i.e. after edge normalisation), `sigma` the
per-pixel standard deviation the likelihood uses (errors and jitter in
quadrature), `velocities` as [`ccf_component_velocities`](@ref).
"""
function ccf_model(theta::Theta, data::Data)
    S = theta.params.layout.systemic
    vels = ccf_component_velocities(theta, data)
    out = NamedTuple[]
    for (si, s) in enumerate(data.ccf)
        ix = S.ccf[si]
        M = vels[si]
        models = Vector{Vector{Float64}}(); res = similar(models); sig = similar(models)
        for n in eachindex(s.t)
            c = Float64(theta.values[ix.cont[n]])
            sc = ix.scale[n] == 0 ? 1.0 : Float64(theta.values[ix.scale[n]])
            jit = Float64(theta.values[ix.jit[n]])
            m = fill(c, length(s.v[n]))
            for j in eachindex(s.components)
                σ = Float64(theta.values[ix.fwhm[j]]) / _FWHM_PER_SIGMA
                dj = Float64(theta.values[ix.depth[j]])
                m .-= sc * dj .* exp.(-((s.v[n] .- M[n, j]) ./ σ) .^ 2 ./ 2)
            end
            push!(models, m); push!(res, s.y[n] .- m)
            push!(sig, sqrt.(s.err[n] .^ 2 .+ jit^2))
        end
        push!(out, (tag = s.tag, labels = s.labels, t = s.t, v = s.v, y = s.y,
                    model = models, residual = res, sigma = sig, velocities = M))
    end
    return out
end
