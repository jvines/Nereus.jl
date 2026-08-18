# Derived physical parameters from posterior samples.
#
# All functions operate on scalars or arrays and are vectorized via
# broadcasting. They are post-processing — not part of the likelihood.
#
# Conventions:
#   - M_s, M_p in solar/Jupiter masses
#   - R_s, R_p in solar/Jupiter radii
#   - P in days, K in m/s, a in AU or R_s
#   - T_eff in K, rho in g/cm³ or solar units

# Aliases to constants.jl (avoids name collisions with exports)
const _MS  = M_SUN_G        # g
const _RS  = R_SUN_CM       # cm
const _MJ  = M_JUP_G        # g
const _RJ  = R_JUP_CM       # cm
const _ME  = M_EARTH_KG * 1e3  # g (constants.jl has kg)
const _RE  = R_EARTH_M * 1e2   # cm (constants.jl has m)
const _DS  = DAY_SECONDS     # s
const _SB  = SIGMA_SB_CGS   # erg cm⁻² s⁻¹ K⁻⁴

# =====================================================================
# Mass functions
# =====================================================================

"""
    msini(M_s, K, P, e) -> M_p sin(i) [M_Jup]

Minimum mass from RV semi-amplitude. M_s in solar masses, K in m/s,
P in days.
"""
function msini(M_s::Real, K::Real, P::Real, e::Real)
    # NaN (not throw) on unphysical inputs: trans-dim chains carry garbage
    # draws in INACTIVE planet slots (P<0 etc.), and compute_derived
    # broadcasts over the full chain — a negative P here turned into a
    # DomainError from (P_s/2piG)^(1/3) and killed the whole summary.
    (e < 0 || e >= 1 || !(P > 0) || !(M_s > 0)) && return NaN
    P_s = P * _DS
    M_s_g = M_s * _MS
    ee = sqrt(1 - e^2)
    # M_p sin(i) = K * sqrt(1-e²) * (P/(2πG))^(1/3) * M_s^(2/3)
    mp_sini = K * 100 * ee * (P_s / (2π * G_CGS))^(1/3) * M_s_g^(2/3)
    return mp_sini / _MJ
end

"""
    planet_mass(M_s, K, P, e, inc_deg) -> M_p [M_Jup]

True mass from RV + inclination. inc_deg in degrees.
"""
function planet_mass(M_s::Real, K::Real, P::Real, e::Real, inc_deg::Real)
    return msini(M_s, K, P, e) / sind(inc_deg)
end

"""
    stellar_mass_from_density(rho_s, R_s) -> M_s [M_sun]

Stellar mass from density (g/cm³) and radius (R_sun).
"""
function stellar_mass_from_density(rho_s::Real, R_s::Real)
    vol = 4π / 3 * (R_s * _RS)^3
    return rho_s * vol / _MS
end

# =====================================================================
# Radius and geometry
# =====================================================================

"""
    planet_radius(R_s, rr) -> R_p [R_Jup]

Planet radius from stellar radius (R_sun) and radius ratio.
"""
planet_radius(R_s::Real, rr::Real) = rr * R_s * _RS / _RJ

"""
    planet_radius_earth(R_s, rr) -> R_p [R_Earth]

Planet radius in Earth radii.
"""
planet_radius_earth(R_s::Real, rr::Real) = rr * R_s * _RS / _RE

"""
    inclination_deg(b, a_Rs) -> i [degrees]

Orbital inclination from impact parameter and a/R*.
"""
inclination_deg(b::Real, a_Rs::Real) = acosd(b / a_Rs)

"""
    semimajor_axis_au(P, M_s) -> a [AU]

Semi-major axis from Kepler's third law. P in days, M_s in solar masses.
"""
function semimajor_axis_au(P::Real, M_s::Real)
    # Same inactive-slot guard as msini — cbrt would silently return a
    # negative semi-major axis instead of throwing.
    (!(P > 0) || !(M_s > 0)) && return NaN
    P_s = P * _DS
    M_s_g = M_s * _MS
    a_cm = cbrt(G_CGS * M_s_g * P_s^2 / (4π^2))
    return a_cm / AU_CM
end

"""
    semimajor_axis_from_rho(rho_s, P, R_s) -> a [AU]

Semi-major axis from stellar density, period, and stellar radius.
rho_s in SOLAR units (ρ/ρ_sun, ρ_sun = 1.411 g/cm³) — same convention as
`rho_s_to_a_Rs`, which this delegates to. P in days, R_s in solar radii.

Note the asymmetry with the reported derived quantity `rho_star_transit`,
which is in g/cm³: feeding a g/cm³ value in here inflates a/R★ by
1.411^(1/3) ≈ 1.12.
"""
function semimajor_axis_from_rho(rho_s::Real, P::Real, R_s::Real)
    a_Rs = rho_s_to_a_Rs(rho_s, P)
    return a_Rs * R_s * _RS / AU_CM
end

# =====================================================================
# Density
# =====================================================================

"""
    planet_density(M_p_jup, R_p_jup) -> ρ_p [g/cm³]

Planet bulk density from mass (M_Jup) and radius (R_Jup).
"""
function planet_density(M_p_jup::Real, R_p_jup::Real)
    M_g = M_p_jup * _MJ
    R_cm = R_p_jup * _RJ
    return M_g / (4π / 3 * R_cm^3)
end

"""
    planet_density_earth(M_p_earth, R_p_earth) -> ρ_p [g/cm³]

Planet bulk density from mass (M_Earth) and radius (R_Earth).
"""
function planet_density_earth(M_p_earth::Real, R_p_earth::Real)
    M_g = M_p_earth * _ME
    R_cm = R_p_earth * _RE
    return M_g / (4π / 3 * R_cm^3)
end

# =====================================================================
# Temperature and insolation
# =====================================================================

"""
    equilibrium_temperature(T_eff, a_Rs; Ab=0.0) -> T_eq [K]

Planetary equilibrium temperature. a_Rs = a/R* (dimensionless).
Ab = Bond albedo (default 0, black body).
"""
function equilibrium_temperature(T_eff::Real, a_Rs::Real; Ab::Real=0.0)
    return T_eff * sqrt(1 / (2 * a_Rs)) * (1 - Ab)^(1/4)
end

"""
    incident_flux(T_eff, a_Rs) -> S [W/m²]

Incident stellar flux at the planet's orbit. a_Rs = a/R*.
Uses Stefan-Boltzmann: S = σ T⁴ / (a/R*)².
"""
function incident_flux(T_eff::Real, a_Rs::Real)
    return _SB * T_eff^4 / a_Rs^2 * 1e-3  # erg/s/cm² → W/m²
end

"""
    incident_flux_earth(L_s, a_au) -> S [S_Earth]

Insolation in Earth units. L_s in solar luminosities, a in AU.
"""
incident_flux_earth(L_s::Real, a_au::Real) = L_s / a_au^2

# =====================================================================
# Transmission/Emission Spectroscopy Metrics
# =====================================================================

"""
    tsm(R_p_earth, M_p_earth, R_s, T_eq, J_mag) -> TSM

Transmission Spectroscopy Metric (Kempton+ 2018, PASP, 130, 114401).
Scale factor depends on R_p range.
"""
function tsm(R_p_earth::Real, M_p_earth::Real, R_s::Real,
             T_eq::Real, J_mag::Real)
    # Scale factors from Kempton+ 2018 Table 1
    if R_p_earth < 1.5
        sf = 0.190
    elseif R_p_earth < 2.75
        sf = 1.26
    elseif R_p_earth < 4.0
        sf = 1.28
    else
        sf = 1.15
    end
    return sf * R_p_earth^3 * T_eq / (M_p_earth * R_s^2) * 10^(-J_mag / 5)
end

"""
    esm(T_day, T_eff, R_p_jup, R_s, K_mag; epsilon=1.0) -> ESM

Emission Spectroscopy Metric (Kempton+ 2018).
T_day = dayside temperature (K), default from T_eq * 1.1.
"""
function esm(T_day::Real, T_eff::Real, R_p_jup::Real, R_s::Real,
             K_mag::Real; epsilon::Real=1.0)
    # B_7.5(T) ∝ Planck at 7.5μm
    lambda = 7.5e-4  # 7.5 μm in cm
    h = 6.626e-27    # erg s
    c = 2.998e10      # cm/s
    k_B = 1.381e-16  # erg/K
    B(T) = 2h * c^2 / lambda^5 / (exp(h * c / (lambda * k_B * T)) - 1)

    R_p_cm = R_p_jup * _RJ
    R_s_cm = R_s * _RS
    return 4.29e6 * epsilon * B(T_day) / B(T_eff) * (R_p_cm / R_s_cm)^2 *
           10^(-K_mag / 5)
end

# =====================================================================
# AMD Stability (Laskar & Petit 2017)
# =====================================================================

"""
    is_amd_stable(periods, masses_jup, eccentricities, M_s) -> Bool

Check Hill stability under the AMD framework for a multi-planet system.
Planets must be sorted by period. Masses are Msini (RV) or true mass.
"""
function is_amd_stable(periods, masses_jup, eccentricities, M_s::Real)
    n = length(periods)
    n >= 2 || return true
    for k in 1:(n - 1)
        a1 = semimajor_axis_au(periods[k], M_s)
        a2 = semimajor_axis_au(periods[k + 1], M_s)
        alpha = a1 / a2
        m1 = masses_jup[k]
        m2 = masses_jup[k + 1]
        gamma = m1 / m2
        eps = (m1 + m2) * _MJ / (M_s * _MS)
        e1 = eccentricities[k]
        e2 = eccentricities[k + 1]

        # AMD criterion (Laskar & Petit 2017 eq. 29)
        amd_val = gamma * sqrt(alpha) * (1 - sqrt(1 - e1^2)) +
                  (1 - sqrt(1 - e2^2))

        # Hill criterion (Petit, Laskar & Boué 2018 eq. 30)
        gp1 = 1 + gamma
        hill = gamma * sqrt(alpha) + 1 -
               gp1^1.5 * sqrt(alpha / (gamma + alpha) *
                              (1 + 3^(4/3) * eps^(2/3) * gamma / gp1^2))

        if amd_val >= hill
            return false
        end
    end
    return true
end

# =====================================================================
# Gladman Hill Stability (Gladman 1993 / Marchal 1982)
# As implemented in EMPEROR I (Peña+ 2025, Eq. 13)
# =====================================================================

"""
    is_gladman_stable(periods, masses_jup, eccentricities, M_s) -> Bool

Check Hill stability using the Gladman 1993 / Marchal 1982 criterion
(as in EMPEROR I, Peña+ 2025, Eq. 13).

Planets must be sorted by period. Masses are Msini (RV) or true mass.

Key differences from AMD-Hill:
  - γⱼ = √(1 - eⱼ)      (NOT √(1 - eⱼ²))
  - δ = √(a₂/a₁)         (NOT a₂/a₁)
  - Angular momentum term is a SUM (not difference)
"""
function is_gladman_stable(periods, masses_jup, eccentricities, M_s::Real)
    n = length(periods)
    n >= 2 || return true

    # Masses and semi-major axes (already sorted by period)
    sma = [semimajor_axis_au(periods[k], M_s) for k in 1:n]
    for k in 1:n
        (sma[k] <= 0 || masses_jup[k] <= 0) && return false
    end

    # Total mass in Jupiter masses (star + all planets)
    M_s_mjup = M_s * _MS / _MJ
    M_total = M_s_mjup + sum(masses_jup)
    mu = masses_jup ./ M_total

    # gamma_j = sqrt(1 - e_j) — NOT sqrt(1 - e_j^2)
    gamma = sqrt.(1.0 .- eccentricities)

    for k in 1:(n - 1)
        alpha = mu[k] + mu[k + 1]
        delta = sqrt(sma[k + 1] / sma[k])

        lhs = alpha^(-3) *
              (mu[k] + mu[k + 1] / delta^2) *
              (mu[k] * gamma[k] + mu[k + 1] * gamma[k + 1] * delta)^2
        rhs = 1.0 + (3.0 / alpha)^(4/3) * mu[k] * mu[k + 1]

        if lhs <= rhs
            return false
        end
    end
    return true
end

# =====================================================================
# Unified stability dispatcher (for use in likelihood hot path)
# =====================================================================

# =====================================================================
# Near-MMR detection and non-crossing orbit constraint
# =====================================================================

# Low-order MMR period ratios to check (outer/inner).
# Tolerance in fractional period ratio — 5% covers libration widths
# for giant planets (Deck+ 2013).
const MMR_RATIOS = (2//1, 3//2, 4//3, 5//3, 5//4, 3//1)
const MMR_TOLERANCE = 0.05

"""
    _is_near_mmr(P_inner, P_outer) -> Bool

Check if a planet pair is near a low-order mean-motion resonance.
"""
function _is_near_mmr(P_inner::Real, P_outer::Real)
    ratio = P_outer / P_inner
    for r in MMR_RATIOS
        if abs(ratio - Float64(r)) / Float64(r) < MMR_TOLERANCE
            return true
        end
    end
    return false
end

"""
    _orbits_non_crossing(a1, e1, a2, e2) -> Bool

Check that two orbits don't cross: apoapsis of inner < periapsis of outer.
"""
function _orbits_non_crossing(a1::Real, e1::Real, a2::Real, e2::Real)
    return a1 * (1 + e1) < a2 * (1 - e2)
end

"""
    check_stability(Ps, Ks, es, M_s, mode) -> Bool

Check orbital stability from RV observables. Computes msini internally,
sorts by period, and dispatches to the selected criterion.

For planet pairs near low-order mean-motion resonances (2:1, 3:2, 4:3,
5:3, 5:4, 3:1), the AMD/Gladman criterion is skipped and replaced with
a non-crossing orbit constraint. This avoids rejecting resonance-
protected configurations like GJ 876 b+c (Deck+ 2013).

# Arguments
- `Ps`   : orbital periods (days)
- `Ks`   : RV semi-amplitudes (m/s)
- `es`   : eccentricities
- `M_s`  : stellar mass (M_sun)
- `mode` : `:amd` or `:gladman`

Returns `true` if stable (or if n < 2).
"""
function check_stability(Ps, Ks, es, M_s::Real, mode::Symbol)
    n = length(Ps)
    n >= 2 || return true

    # Sort by period
    order = sortperm(Ps)
    periods_sorted = Ps[order]
    masses_sorted  = [msini(M_s, Ks[i], Ps[i], es[i]) for i in order]
    ecc_sorted     = es[order]
    sma_sorted     = [semimajor_axis_au(periods_sorted[k], M_s) for k in 1:n]

    # Check each adjacent pair
    for k in 1:(n - 1)
        if _is_near_mmr(periods_sorted[k], periods_sorted[k + 1])
            # Near MMR: skip AMD/Gladman, apply non-crossing orbit constraint
            if !_orbits_non_crossing(sma_sorted[k], ecc_sorted[k],
                                      sma_sorted[k + 1], ecc_sorted[k + 1])
                return false
            end
        else
            # Non-resonant: apply full stability criterion to this pair
            pair_P = [periods_sorted[k], periods_sorted[k + 1]]
            pair_m = [masses_sorted[k], masses_sorted[k + 1]]
            pair_e = [ecc_sorted[k], ecc_sorted[k + 1]]
            if mode === :amd
                is_amd_stable(pair_P, pair_m, pair_e, M_s) || return false
            elseif mode === :gladman
                is_gladman_stable(pair_P, pair_m, pair_e, M_s) || return false
            end
        end
    end
    return true
end

"""
    check_stability(Ps, Ks, es, M_s, mode, order_buf, masses_buf, sma_buf) -> Bool

Workspace-aware stability check that reuses pre-allocated buffers
instead of allocating sortperm/masses/sma arrays. Functionally
identical to the allocating method.
"""
function check_stability(Ps, Ks, es, M_s::Real, mode::Symbol,
                         order_buf::AbstractVector{Int},
                         masses_buf::AbstractVector{Float64},
                         sma_buf::AbstractVector{Float64})
    n = length(Ps)
    n >= 2 || return true

    # Fill order buffer and sort in-place
    @inbounds for i in 1:n
        order_buf[i] = i
    end
    sort!(@view(order_buf[1:n]); by=i -> Ps[i])

    # Fill masses and sma using sorted order
    @inbounds for k in 1:n
        i = order_buf[k]
        masses_buf[k] = msini(M_s, Ks[i], Ps[i], es[i])
        sma_buf[k] = semimajor_axis_au(Ps[i], M_s)
    end

    # Check each adjacent pair
    for k in 1:(n - 1)
        P_k   = Ps[order_buf[k]]
        P_k1  = Ps[order_buf[k + 1]]
        ecc_k = es[order_buf[k]]
        ecc_k1 = es[order_buf[k + 1]]

        if _is_near_mmr(P_k, P_k1)
            if !_orbits_non_crossing(sma_buf[k], ecc_k,
                                      sma_buf[k + 1], ecc_k1)
                return false
            end
        else
            pair_P = [P_k, P_k1]
            pair_m = [masses_buf[k], masses_buf[k + 1]]
            pair_e = [ecc_k, ecc_k1]
            if mode === :amd
                is_amd_stable(pair_P, pair_m, pair_e, M_s) || return false
            elseif mode === :gladman
                is_gladman_stable(pair_P, pair_m, pair_e, M_s) || return false
            end
        end
    end
    return true
end

# =====================================================================
# Convenience: compute all derived params from chains
# =====================================================================

"""
    DerivedParams

Container for derived physical parameters of one planet.
"""
struct DerivedParams
    name::String
    values::Dict{String, Vector{Float64}}
end

"""
    compute_derived(chains, params; M_s=nothing, R_s=nothing,
                     T_eff=nothing, J_mag=nothing, K_mag=nothing, Ab=0.0)

Compute all derived parameters from posterior chains.
Returns a vector of `DerivedParams`, one per planet.

Stellar parameters (M_s, R_s, T_eff) can be scalars or distributions
(vectors of samples matching the chain length).
"""
function compute_derived(chains, params::Params;
                          M_s = nothing, R_s = nothing,
                          T_eff = nothing, J_mag = nothing,
                          K_mag = nothing, Ab = 0.0)
    config = params.config
    parametrization = config.parametrization
    # `chains` may be 2-D (n_iter, n_param) for legacy samplers or 3-D
    # (n_iter, n_param, n_chain) for ptemcee's per-walker reshape.
    # `vec(Array(chains[sym]))` always flattens across every chain
    # dimension, so n_samples must match that length — not size(_, 1).
    n_samples = length(vec(Array(chains[:, 1, :])))
    results = DerivedParams[]

    # Broadcast stellar params to sample arrays
    _to_vec(x::Nothing, n) = nothing
    _to_vec(x::Real, n) = fill(Float64(x), n)
    _to_vec(x::AbstractVector, n) = Float64.(x)

    ms = _to_vec(M_s, n_samples)
    rs = _to_vec(R_s, n_samples)
    teff = _to_vec(T_eff, n_samples)

    # Physical limb-darkening (u1, u2) per PHOTOMETRIC LD group, mapped from
    # the sampled Kipping (q1, q2) via the SAME kipping_q_to_u the likelihood
    # uses.  These are per-INSTRUMENT (LD group), NOT per-planet, but the only
    # structural carrier here is the per-planet `derived` dict, so the same
    # u1_<label>/u2_<label> vectors are injected into each transiting planet's
    # dict below (identical values; harmless, dimensionless).
    chain_param_syms = Set(names(chains, :parameters))
    ld_u = Dict{String, Vector{Float64}}()   # "u1_<label>"/"u2_<label>" → draws
    for sym in chain_param_syms
        ssym = String(sym)
        startswith(ssym, "q1_") || continue
        label = ssym[4:end]                  # strip "q1_"
        q2sym = Symbol("q2_" * label)
        q2sym in chain_param_syms || continue
        q1v = vec(Array(chains[sym]))
        q2v = vec(Array(chains[q2sym]))
        uu = kipping_q_to_u.(q1v, q2v)        # Vector of (u1, u2) tuples
        ld_u["u1_" * label] = first.(uu)
        ld_u["u2_" * label] = last.(uu)
    end

    kplanet = config.max_kplanet
    for k in 1:kplanet
        block = params.layout.planet_blocks[k]
        derived = Dict{String, Vector{Float64}}()

        chain_syms = Set(names(chains, :parameters))

        # --- SB2 binary: mass ratio, minimum masses, absolute masses ---
        # From the two amplitudes:  q = M_B/M_A = K_A/K_B;
        #   M_A sin³i = F·(K_A+K_B)²·K_B·P·(1−e²)^1.5   (M_sun)
        #   M_B sin³i = F·(K_A+K_B)²·K_A·P·(1−e²)^1.5
        # with F = _F_M_FACTOR (= 1/(2πG) in Nereus units). Absolute masses
        # follow when the binary has astrometry (inc sampled). Post-hoc from
        # the chains — never sampled.
        if block isa SB2Block
            draws(nm) = Symbol(nm) in chain_syms ?
                        vec(Array(chains[Symbol(nm)])) :
                        fill(get_param(Theta{Float64}(params), nm), n_samples)
            KA = draws("K_A_k$k"); KB = draws("K_B_k$k"); Pv = draws("P_k$k")
            ev = if parametrization.ew === :sesinw
                se = draws("sesinw_k$k"); sc = draws("secosw_k$k"); se.^2 .+ sc.^2
            elseif parametrization.ew === :esinw
                es = draws("esinw_k$k"); ec = draws("ecosw_k$k"); sqrt.(es.^2 .+ ec.^2)
            else
                draws("ecc_k$k")
            end
            omE = max.(1 .- ev.^2, 0.0).^(3/2)
            Ksum = KA .+ KB
            MA_sini3 = _F_M_FACTOR .* Ksum.^2 .* KB .* Pv .* omE
            MB_sini3 = _F_M_FACTOR .* Ksum.^2 .* KA .* Pv .* omE
            derived["mass_ratio_q"] = KA ./ KB           # q = M_B/M_A
            derived["M_A_sini3"]    = MA_sini3           # M_sun (minimum)
            derived["M_B_sini3"]    = MB_sini3
            derived["P_days"]       = Pv
            derived["ecc"]          = ev
            if Symbol("inc_k$k") in chain_syms           # astrometric SB2 → absolute
                inc = vec(Array(chains[Symbol("inc_k$k")]))
                s3  = max.(sin.(inc), 1e-6).^3
                derived["M_A"]     = MA_sini3 ./ s3
                derived["M_B"]     = MB_sini3 ./ s3
                derived["inc_deg"] = rad2deg.(inc)
            end
            push!(results, DerivedParams("binary_$k", derived))
            continue
        end
        # Period (days). Under :a_driven (orvara-style) the orbital slot is a_k
        # (AU) and there is no P_k — derive the period from Kepler's 3rd law,
        # P_yr² = a³ / (M_pri + M_sec).
        if Symbol("P_k$k") in chain_syms
            P = vec(Array(chains[Symbol("P_k$k")]))
        else
            a_au_s = vec(Array(chains[Symbol("a_k$k")]))
            msec_s = Symbol("M_sec_k$k") in chain_syms ?
                     vec(Array(chains[Symbol("M_sec_k$k")])) : zeros(length(a_au_s))
            mpri_s = :M_pri in chain_syms ? vec(Array(chains[:M_pri])) :
                     fill(isnan(config.M_s) ? 1.0 : Float64(config.M_s), length(a_au_s))
            P = sqrt.(a_au_s .^ 3 ./ max.(mpri_s .+ msec_s, 1e-6)) .* 365.25
        end

        # Eccentricity + omega (derived from parametrization)
        # Handle frozen eccentricity params (not in chains)
        if parametrization.ew === :sesinw
            se_sym = Symbol("sesinw_k$k")
            sc_sym = Symbol("secosw_k$k")
            se = se_sym in chain_syms ? vec(Array(chains[se_sym])) :
                 fill(get_param(Theta{Float64}(params), "sesinw_k$k"), n_samples)
            sc = sc_sym in chain_syms ? vec(Array(chains[sc_sym])) :
                 fill(get_param(Theta{Float64}(params), "secosw_k$k"), n_samples)
            e = se.^2 .+ sc.^2
            w = atan.(se, sc)
        elseif parametrization.ew === :esinw
            es_sym = Symbol("esinw_k$k")
            ec_sym = Symbol("ecosw_k$k")
            es = es_sym in chain_syms ? vec(Array(chains[es_sym])) :
                 fill(get_param(Theta{Float64}(params), "esinw_k$k"), n_samples)
            ec = ec_sym in chain_syms ? vec(Array(chains[ec_sym])) :
                 fill(get_param(Theta{Float64}(params), "ecosw_k$k"), n_samples)
            e = sqrt.(es.^2 .+ ec.^2)
            w = atan.(es, ec)
        else
            e_sym = Symbol("ecc_k$k")
            w_sym = Symbol("w_k$k")
            e = e_sym in chain_syms ? vec(Array(chains[e_sym])) :
                fill(get_param(Theta{Float64}(params), "ecc_k$k"), n_samples)
            w = w_sym in chain_syms ? vec(Array(chains[w_sym])) :
                fill(get_param(Theta{Float64}(params), "w_k$k"), n_samples)
        end
        derived["ecc"] = e
        derived["omega_deg"] = rad2deg.(w)

        # Semi-amplitude (RV planets). Under :a_driven / :M_sec_driven the RV
        # amplitude is DERIVED from the sampled mass, so there is no K_k slot —
        # guard on the chain actually carrying K_k (else msini-from-K is skipped;
        # the companion mass is the fitted M_sec_k directly).
        has_rv = has_K(block) && Symbol("K_k$k") in chain_syms
        if has_rv
            K = vec(Array(chains[Symbol("K_k$k")]))
        end

        # a/R* and a(AU)
        has_rho = haskey(params.layout.name_to_idx, "rho_s")
        if has_rho
            rho_samples = vec(Array(chains[:rho_s]))
            a_Rs_samples = rho_s_to_a_Rs.(rho_samples, P)
            derived["a_Rs"] = a_Rs_samples
            if rs !== nothing
                a_au = a_Rs_samples .* rs .* _RS ./ AU_CM
                derived["a_au"] = a_au
            end
        elseif Symbol("a_k$k") in chain_syms
            # :a_driven (orvara-style) — the semi-major axis is sampled directly
            # in AU; use it rather than recomputing from P (which uses M_s, not
            # the total mass, and would be biased by the mass ratio).
            derived["a_au"] = vec(Array(chains[Symbol("a_k$k")]))
        elseif ms !== nothing
            a_au = semimajor_axis_au.(P, ms)
            derived["a_au"] = a_au
            # When `use_rho_s=false`, compute a/R* from a_au and R_s so
            # downstream derived quantities (Teq, S_earth, TSM, ESM)
            # that require a/R* still get filled in.
            if rs !== nothing
                a_Rs_samples = a_au .* AU_CM ./ (rs .* _RS)
                derived["a_Rs"] = a_Rs_samples
            end
        end

        # Inclination (transit planets)
        has_geom = has_geometry(block)
        if has_geom
            if parametrization.geom === :b_rr
                b_samp = vec(Array(chains[Symbol("b_k$k")]))
                rr_samp = vec(Array(chains[Symbol("rr_k$k")]))
            else  # :r1r2
                r1 = vec(Array(chains[Symbol("r1_k$k")]))
                r2 = vec(Array(chains[Symbol("r2_k$k")]))
                rr_samp = r1 .* (1 .+ r2) ./ 2
                b_samp = r1 .* (1 .- r2) ./ 2
            end
            derived["rr"] = rr_samp
            derived["b"] = b_samp

            if haskey(derived, "a_Rs")
                inc = inclination_deg.(b_samp, derived["a_Rs"])
                derived["inc_deg"] = inc
            end
        end

        # Msini (RV planets + stellar mass)
        if has_rv && ms !== nothing
            msi = msini.(ms, K, P, e)
            derived["msini_jup"] = msi
            derived["msini_earth"] = msi .* (_MJ / _ME)
        end

        # True mass (RV + transit + stellar mass + inclination)
        if has_rv && has_geom && ms !== nothing && haskey(derived, "inc_deg")
            m_jup = planet_mass.(ms, K, P, e, derived["inc_deg"])
            derived["mass_jup"] = m_jup
            derived["mass_earth"] = m_jup .* (_MJ / _ME)
        end

        # Planet radius (transit + stellar radius)
        if has_geom && rs !== nothing
            rp_jup = planet_radius.(rs, rr_samp)
            rp_earth = planet_radius_earth.(rs, rr_samp)
            derived["radius_jup"] = rp_jup
            derived["radius_earth"] = rp_earth
        end

        # ----- Transit-geometry observables -----------------------------
        # Only for planets with transit geometry (rr/b present) and a known
        # a/R* — mirrors the gate that fills inc_deg above.  Computed per
        # draw from the SAME (e, ω, a/R*, b, i, rr, P) the loop already uses
        # so they inherit proper posterior CIs from the summarizer.
        if has_geom && haskey(derived, "a_Rs")
            a_Rs_d = derived["a_Rs"]
            inc_d  = derived["inc_deg"]            # degrees
            ns_k   = length(P)

            # Geometric transit depth (rr² in ppm) — the standard quoted depth.
            derived["depth_ppm"] = (rr_samp .^ 2) .* 1.0e6

            # ----- Spin-orbit angles ------------------------------------
            # lambda (sky-projected) is fitted whenever a planet has :RM* or
            # :GD; i_star only under :GD. When both exist the TRUE obliquity
            # follows per draw:
            #     cos psi = cos i* cos i_p + sin i* sin i_p cos lambda
            # Computed here rather than by hand afterwards so it inherits the
            # posterior CI from the same summariser as everything else, and so
            # the lambda-i_star-i_p correlations are carried rather than
            # propagated by quadrature.
            lam_sym = Symbol("lambda_k$k")
            if lam_sym in chain_param_syms
                lam = vec(Array(chains[lam_sym]))
                derived["lambda_deg"] = rad2deg.(lam)
                # |lambda| > 90 deg is a retrograde orbit; report the folded
                # angle too since that is what population studies compare.
                derived["lambda_abs_deg"] = abs.(rad2deg.(lam))
                if :i_star in chain_param_syms
                    ist = vec(Array(chains[:i_star]))
                    n_psi = min(length(lam), length(ist), length(inc_d))
                    psi = Vector{Float64}(undef, n_psi)
                    @inbounds for j in 1:n_psi
                        ip_r = deg2rad(inc_d[j])
                        c = cos(ist[j]) * cos(ip_r) +
                            sin(ist[j]) * sin(ip_r) * cos(lam[j])
                        psi[j] = rad2deg(acos(clamp(c, -1.0, 1.0)))
                    end
                    derived["i_star_deg"] = rad2deg.(ist[1:n_psi])
                    derived["psi_deg"] = psi
                end
            end

            # Grazing fraction: P(b + rr > 1 | planet) over this slot's draws.
            # ParamStats reports the MEDIAN, which would collapse a 0/1
            # indicator to 0 or 1 and lose the fraction — so emit a CONSTANT
            # per-draw vector equal to the mean indicator, making median = mean
            # = the grazing fraction (CI collapses to [frac, frac]).
            graz_frac = sum((b_samp .+ rr_samp) .> 1.0) / ns_k
            derived["grazing_frac"] = fill(graz_frac, ns_k)

            # Winn 2010 (arXiv:1001.2010) eqs 14–15 transit durations, in HOURS.
            #   ecc_factor = sqrt(1-e²)/(1 + e sin ω)
            #   T14 = (P/π) asin[ (1/aR*) sqrt((1+rr)² − b²)/sin i ] · ecc_factor
            #   T23 = (P/π) asin[ (1/aR*) sqrt((1−rr)² − b²)/sin i ] · ecc_factor
            # `w` here is in RADIANS (the loop stores rad2deg(w) into omega_deg).
            # sin i uses the ECCENTRIC inclination cos i = b(1+e sinω)/((a/R*)(1−e²))
            # — the same relation `sky_separation` uses (transit_likelihood.jl) —
            # NOT the circular cos i = b/(a/R*) of `inc_deg`. Mixing the circular
            # i with the eccentric ecc_factor is internally inconsistent and
            # biases T14/T23 by up to ~0.7% at high e·b; the two agree at e=0.
            T14 = Vector{Float64}(undef, ns_k)
            T23 = Vector{Float64}(undef, ns_k)
            @inbounds for j in 1:ns_k
                ecc_factor = sqrt(max(1.0 - e[j]^2, 0.0)) / (1.0 + e[j] * sin(w[j]))
                cos_i = b_samp[j] * (1.0 + e[j] * sin(w[j])) /
                        ((1.0 - e[j]^2) * a_Rs_d[j])
                sin_i = sqrt(max(1.0 - cos_i^2, 0.0))
                pref  = P[j] / π
                # T14: (1+rr)² − b² is > 0 for any transiting geometry; clamp
                # defensively. Guard the asin argument to [-1, 1].
                arg14 = sin_i == 0.0 ? 2.0 :
                        (1.0 / a_Rs_d[j]) * sqrt(max((1.0 + rr_samp[j])^2 - b_samp[j]^2, 0.0)) / sin_i
                T14[j] = pref * asin(clamp(arg14, -1.0, 1.0)) * ecc_factor * 24.0
                # T23: (1−rr)² − b² goes negative for grazing (b > 1−rr); the
                # full phase then does not exist → T23 = 0.
                rad23 = (1.0 - rr_samp[j])^2 - b_samp[j]^2
                if rad23 <= 0.0 || sin_i == 0.0
                    T23[j] = 0.0
                else
                    arg23 = (1.0 / a_Rs_d[j]) * sqrt(rad23) / sin_i
                    T23[j] = pref * asin(clamp(arg23, -1.0, 1.0)) * ecc_factor * 24.0
                end
            end
            derived["T14"] = T14
            derived["T23"] = T23

            # Transit-derived stellar density (photoeccentric cross-check):
            # invert rho_s_to_a_Rs.  That map is, with ρ in SOLAR units (ρ_sun
            # = 1.411 g/cm³), G_cgs = 6.674e-8, P_sec = P·86400:
            #   a/R* = cbrt( G_cgs · (ρ·1.411) · P_sec² / (3π) )
            # ⇒ ρ_solar = 3π · (a/R*)³ / (G_cgs · 1.411 · P_sec²),
            # and ρ_cgs[g/cm³] = ρ_solar · 1.411 = 3π · (a/R*)³ / (G_cgs · P_sec²).
            G_cgs = 6.674e-8
            rho_star_cgs = similar(a_Rs_d)
            @inbounds for j in eachindex(a_Rs_d)
                P_sec = P[j] * 86400.0
                rho_star_cgs[j] = 3π * a_Rs_d[j]^3 / (G_cgs * P_sec^2)
            end
            derived["rho_star_transit"] = rho_star_cgs

            # Per-instrument physical LD coefficients (same for every transiting
            # planet — emitted on each so they survive the per-planet table).
            for (uk, uv) in ld_u
                derived[uk] = uv
            end
        end

        # Planet density (mass + radius)
        if haskey(derived, "mass_jup") && haskey(derived, "radius_jup")
            rho_p = planet_density.(derived["mass_jup"], derived["radius_jup"])
            derived["density_gcc"] = rho_p
        end

        # Equilibrium temperature
        if teff !== nothing && haskey(derived, "a_Rs")
            teq = equilibrium_temperature.(teff, derived["a_Rs"]; Ab=Ab)
            derived["Teq"] = teq
        end

        # Incident flux
        if teff !== nothing && haskey(derived, "a_Rs")
            s_flux = incident_flux.(teff, derived["a_Rs"])
            derived["S_Wm2"] = s_flux
        end
        if haskey(derived, "a_au") && ms !== nothing && rs !== nothing
            # Rough L_s from Teff and Rs
            if teff !== nothing
                L_s = (rs ./ 1.0).^2 .* (teff ./ 5778.0).^4
                derived["S_earth"] = incident_flux_earth.(L_s, derived["a_au"])
            end
        end

        # TSM / ESM
        if haskey(derived, "radius_earth") && haskey(derived, "mass_earth") &&
           haskey(derived, "Teq") && rs !== nothing && J_mag !== nothing
            tsm_val = tsm.(derived["radius_earth"], derived["mass_earth"],
                           rs, derived["Teq"], J_mag)
            derived["TSM"] = tsm_val
        end
        if haskey(derived, "Teq") && haskey(derived, "radius_jup") &&
           teff !== nothing && rs !== nothing && K_mag !== nothing
            T_day = derived["Teq"] .* 1.1  # rough dayside
            esm_val = esm.(T_day, teff, derived["radius_jup"], rs, K_mag)
            derived["ESM"] = esm_val
        end

        push!(results, DerivedParams("planet_$k", derived))
    end

    return results
end

# =====================================================================
# Conditional γ point estimate (γ-marginalized RV likelihood)
# =====================================================================

"""
    conditional_gamma(theta, data) -> Dict{Int, Float64}

Per-instrument-group conditional (MAP) estimate of the RV systemic
offset γ, for the γ-marginalized likelihood path
(`parametrization.marginalize_gamma=true`). Returns a `Dict` mapping
each γ layout-slot index to its conditional point estimate

    γ̂_g = (Σ_i w_i (rv_i − μ_i)) / (Σ_i w_i),   w_i = 1/(σ_obs,i² + σ_g²)

where `μ_i` is the γ-free mean model (Keplerian + trend + RM) at the
current `theta`, evaluated via `rv_predictions` (the frozen γ slot
contributes 0, so the prediction already equals μ_i under
marginalization). This is the posterior-conditional value the analytic
marginalization integrates over — report it post-hoc so γ is not lost
to the user even though it is not sampled.

The keys are γ layout slots (`theta.params.layout.systemic.rv_gamma`);
map back to instrument names via that vector if a per-name report is
needed. Slots are shared when `:gamma` instrument-sharing is active.
"""
function conditional_gamma(theta::Theta, data::Data)
    predictions, variances = rv_predictions(theta, data)
    rv_gamma_slots = theta.params.layout.systemic.rv_gamma
    n_obs = length(data.t_rv)

    A = Dict{Int, Float64}()   # Σ w_i
    B = Dict{Int, Float64}()   # Σ w_i (rv_i − μ_i)
    @inbounds for i in 1:n_obs
        w = 1.0 / Float64(variances[i])
        d = Float64(data.rv[i] - predictions[i])
        g = rv_gamma_slots[data.rv_inst[i]]
        A[g] = get(A, g, 0.0) + w
        B[g] = get(B, g, 0.0) + w * d
    end

    γ̂ = Dict{Int, Float64}()
    for (g, Ag) in A
        γ̂[g] = B[g] / Ag
    end
    return γ̂
end
