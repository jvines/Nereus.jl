# A small synthetic obliquity dataset -- three RM nights and three
# line-profile stacks of one planet -- and the pieces the obliquity tests share:
# the geometry, the data written to disk the way a job config reads it, the
# bespoke log-posteriors the framework target must reproduce, and the map from
# a bespoke parameter vector to the framework's.
#
# Small on purpose (15 velocities and 14 x 25 map pixels per night): every
# test that touches it runs in seconds.

using Nereus
using Nereus: RMNight, TomoNight, Theta, set_param!
using Random, Statistics, DelimitedFiles

# P and Tc0 are binary fractions, so every night's Tc0 + E·P is exact in
# Float64. The bespoke velocities-only model measures time from Tc0, the others
# from each night's Tc0 + E·P; with real ephemerides those differ by the
# rounding of a BJD (~2e-10 d), which is not part of any model and would only
# blur the comparison. (parity_NGTS33_lambda.jl measures it on the real data.)
const OS_P    = 2.828125
const OS_TC0  = 2459986.40625
const OS_EPOCH = Dict("A" => 120, "B" => 248, "C" => 266)
const OS_TAGS = ["A", "B", "C"]
const OS_ARS  = (6.82, 0.09)
const OS_B    = (0.75, 0.01)
const OS_INC  = acos(OS_B[1] / OS_ARS[1])
const OS_RR   = 0.1158
const OS_VSINI_KMS = (25.9, 1.5)
const OS_VSINI_MS  = (25_900.0, 1_500.0)
const OS_K    = (368.0, 27.0)
const OS_U1, OS_U2 = 0.32, 0.30
const OS_SIG0 = Dict("A" => 15_702.0, "B" => 15_082.0, "C" => 15_912.0)
const OS_LAM  = deg2rad(-55.0)
const OS_T14H = 2.634 / 2
const OS_GRID = range(-42, 42; length = 25)
const OS_VSYS = 18.93

os_tc(tag) = OS_TC0 + OS_EPOCH[tag] * OS_P

"Write the synthetic nights to `dir` in the formats a job config reads."
function os_write_data(dir::AbstractString; seed = 11)
    mkpath(dir)
    rng = MersenneTwister(seed)
    for (j, tag) in enumerate(OS_TAGS)
        Tc = os_tc(tag)
        sim = Nereus.simulate_rm_night(; P = OS_P, Tc = Tc, a_Rs = OS_ARS[1],
            λ = OS_LAM, vsini = OS_VSINI_KMS[1], rr = OS_RR, T14 = 2 * OS_T14H,
            sigma_rv = 80.0, sigma0 = OS_SIG0[tag], K = OS_K[1],
            gamma = 18_900.0 + 300j, cadence = 1500.0, span = 4.4,
            u1 = OS_U1, u2 = OS_U2, b = OS_B[1], tag = tag, rng = rng)
        n = sim.night
        # a slow drift on top: something for the oscillator to do
        drift = 120 .* sin.(2π .* (n.t .- Tc) ./ 0.4)
        writedlm(joinpath(dir, "rm_$(tag).dat"),
                 hcat(n.t, n.rv .+ drift, n.err))
        s = Nereus.simulate_tomogram(; P = OS_P, Tc = Tc, a_Rs = OS_ARS[1],
            λ = OS_LAM, vsini = OS_VSINI_KMS[1], rr = OS_RR, T14 = 2 * OS_T14H,
            sigma_pixel = 2e-3, b = OS_B[1], cadence = 1100.0, span = 4.0,
            u1 = OS_U1, u2 = OS_U2, vsys = OS_VSYS, berv = 3.0j,
            vgrid = range(-70, 110; length = 73), tag = tag, rng = rng)
        writedlm(joinpath(dir, "tomo_$(tag)_prof.dat"), s.profiles)
        writedlm(joinpath(dir, "tomo_$(tag)_vgrid.dat"), s.vgrid)
        writedlm(joinpath(dir, "tomo_$(tag)_t.dat"), s.times)
        writedlm(joinpath(dir, "tomo_$(tag)_berv.dat"), s.bervs)
    end
    return dir
end

"Load the nights the way the bespoke drivers do."
function os_load(dir::AbstractString; tags = OS_TAGS)
    rm = RMNight[]; tomo = TomoNight[]
    for tag in tags
        d = readdlm(joinpath(dir, "rm_$(tag).dat"))
        push!(rm, RMNight(tag, d[:, 1], d[:, 2], d[:, 3], OS_SIG0[tag], os_tc(tag)))
    end
    for tag in OS_TAGS
        prof = readdlm(joinpath(dir, "tomo_$(tag)_prof.dat"))
        vg   = vec(readdlm(joinpath(dir, "tomo_$(tag)_vgrid.dat")))
        t    = vec(readdlm(joinpath(dir, "tomo_$(tag)_t.dat")))
        berv = vec(readdlm(joinpath(dir, "tomo_$(tag)_berv.dat")))
        Tc   = os_tc(tag)
        intr = abs.((t .- Tc) .* 24) .<= OS_T14H
        g, R = Nereus.tomogram_residuals(prof, vg, intr; vsys = OS_VSYS,
                                          bervs = berv, grid = OS_GRID)
        push!(tomo, TomoNight(tag, t, R, g, Tc))
    end
    return rm, tomo
end

# ---------------------------------------------------------------------------
# Bespoke log-posteriors (the references)
# ---------------------------------------------------------------------------

"The velocities-only log-posterior of fit_NGTS33_rm_bayes.jl, verbatim but for
 the data: θ = [λ, v sin i (m/s), K, (γ, log S0, log Q, log ω0, log jit) per night]."
function os_bespoke_vel(θ, rm::Vector{RMNight})
    t_all  = reduce(vcat, (n.t for n in rm))
    rv_all = reduce(vcat, (n.rv for n in rm))
    er_all = reduce(vcat, (n.err for n in rm))
    nig_all = reduce(vcat, (fill(j, length(n.t)) for (j, n) in enumerate(rm)))
    σ0_all = reduce(vcat, (fill(n.σ0, length(n.t)) for n in rm))
    AR, INC = OS_ARS[1], OS_INC
    βp_of(σ0, vsini) = sqrt(max(σ0^2 - (0.5503 * vsini)^2, 4.0e6))
    kep(t, K) = -K .* sin.(2π .* (t .- OS_TC0) ./ OS_P)
    function rm_rv(λ, vsini)
        out = zeros(length(t_all))
        for i in eachindex(t_all)
            x, y, z = Nereus.planet_sky_position(t_all[i] - OS_TC0, OS_P, 0.0, π / 2,
                                                  0.0, AR * cos(INC), AR)
            (hypot(x, y) < 1 && z > 0) || continue
            μ = sqrt(max(1 - x^2 - y^2, 0.0))
            Δf = OS_RR^2 * (1 - OS_U1 * (1 - μ) - OS_U2 * (1 - μ)^2) / (1 - OS_U1 / 3 - OS_U2 / 6)
            out[i] = Nereus.rm_signal_arome(x, y, Δf, vsini, λ, σ0_all[i],
                                             βp_of(σ0_all[i], vsini))
        end
        return out
    end
    λ = mod(θ[1] + π, 2π) - π
    vsini, K = θ[2], θ[3]
    (vsini > 5_000) || return (prior = -Inf, rv = -Inf)
    pr = -0.5 * ((vsini - OS_VSINI_MS[1]) / OS_VSINI_MS[2])^2 -
          0.5 * ((K - OS_K[1]) / OS_K[2])^2
    model = kep(t_all, K) .+ rm_rv(λ, vsini)
    ll = 0.0
    for j in 1:length(rm)
        o = 3 + (j - 1) * 5
        γ, lS, lQ, lw, lj = θ[o+1], θ[o+2], θ[o+3], θ[o+4], θ[o+5]
        (-2 <= lS <= 12) || return (prior = -Inf, rv = -Inf)
        (log10(0.2) <= lQ <= 2.0) || return (prior = -Inf, rv = -Inf)
        (log10(0.2) <= lw <= log10(60.0)) || return (prior = -Inf, rv = -Inf)
        (-1 <= lj <= 3.5) || return (prior = -Inf, rv = -Inf)
        m = nig_all .== j
        r = rv_all[m] .- model[m] .- γ
        v = er_all[m] .^ 2 .+ (10.0^lj)^2
        ar, cr, ac, bc, cc, dc = Nereus.sho_coefficients(10.0^lS, 10.0^lQ, 10.0^lw)
        ll += Nereus.celerite_loglike(t_all[m], r, v, ar, cr, ac, bc, cc, dc)
    end
    return (prior = pr, rv = ll)
end

os_hours(tomo) = [(nt.t .- nt.Tc) .* 24 for nt in tomo]

"Shadow fit (Nereus.tomogram_logpost), split into prior and map terms."
function os_bespoke_tomo(θ, tomo; shared = false)
    total = Nereus.tomogram_logpost(θ, tomo, os_hours(tomo), OS_P;
        vsini_mu = OS_VSINI_KMS[1], vsini_sd = OS_VSINI_KMS[2], b_mu = OS_B[1],
        b_sd = OS_B[2], a_mu = OS_ARS[1], a_sd = OS_ARS[2], rr = OS_RR,
        u1 = OS_U1, u2 = OS_U2, α_max = 20.0, shared_α = shared)
    pr = -0.5 * ((θ[2] - OS_VSINI_KMS[1]) / OS_VSINI_KMS[2])^2 -
          0.5 * ((θ[3] - OS_B[1]) / OS_B[2])^2 - 0.5 * ((θ[4] - OS_ARS[1]) / OS_ARS[2])^2
    return (prior = pr, tomo = total - pr, total = total)
end

"Joint fit (Nereus.joint_obliquity_logpost), split into prior, velocity and map terms."
function os_bespoke_joint(θ, tomo, rm; use_tomogram = true)
    kw = (vsini_mu = OS_VSINI_KMS[1], vsini_sd = OS_VSINI_KMS[2], b_mu = OS_B[1],
          b_sd = OS_B[2], a_mu = OS_ARS[1], a_sd = OS_ARS[2], rr = OS_RR,
          u1 = OS_U1, u2 = OS_U2, α_max = 20.0)
    total = Nereus.joint_obliquity_logpost(θ, tomo, os_hours(tomo), rm, OS_P;
                                           K_mu = OS_K[1], K_sd = OS_K[2],
                                           use_tomogram = use_tomogram, kw...)
    pr = -0.5 * ((θ[2] - OS_VSINI_KMS[1]) / OS_VSINI_KMS[2])^2 -
          0.5 * ((θ[3] - OS_B[1]) / OS_B[2])^2 - 0.5 * ((θ[4] - OS_ARS[1]) / OS_ARS[2])^2
    tm = 0.0
    if use_tomogram
        tm = Nereus.tomogram_logpost(θ[1:4+6length(tomo)], tomo, os_hours(tomo), OS_P;
                                     kw...) - pr
    end
    if !isempty(rm)
        off = 4 + 6 * length(tomo)
        pr += -0.5 * ((θ[off+1] - OS_K[1]) / OS_K[2])^2
        γb = Nereus._rm_gamma_bounds(rm)
        for q in eachindex(rm)
            lo, hi = γb[q]
            pr -= log(hi - lo)
        end
    end
    return (prior = pr, rv = total - pr - tm, tomo = tm, total = total)
end

# ---------------------------------------------------------------------------
# Bespoke vector -> framework Theta, and the constant prior offset
# ---------------------------------------------------------------------------

const OS_LN10 = log(10.0)

"""
Seat bespoke vector `θ` (kind `:vel`, `:shadow`, `:shadow_shared` or `:joint`)
into a framework `Theta` of `p`. `ntomo_theta` is the number of map blocks in
`θ` when it differs from the maps the framework has. Returns the Theta and `logJ`, the log
Jacobian |∂x_framework/∂θ_bespoke| over the shared coordinates.
"""
function os_seat(kind, θ, p; rm_tags, tomo_tags, ntomo_theta = length(tomo_tags))
    th = Theta{Float64}(p)
    has(k) = haskey(p.layout.name_to_idx, k)
    put!(k, v) = (has(k) || error("framework has no parameter $k"); set_param!(th, k, v))
    logJ = 0.0
    put!("lambda_k1", mod(θ[1] + π, 2π) - π)
    if kind === :vel
        put!("v_sin_i_star", θ[2]); put!("K_k1", θ[3]); base = 3
    else
        put!("v_sin_i_star", θ[2] * 1e3); logJ += log(1e3)
        put!("b_k1", θ[3]); put!("a_Rs_k1", θ[4])
        shared = kind === :shadow_shared
        shared && put!("tomo_alpha", θ[5])
        for (j, tag) in enumerate(tomo_tags)
            o, sh = shared ? (5 + (j - 1) * 5, 0) : (4 + (j - 1) * 6, 1)
            shared || put!("tomo_alpha_$tag", θ[o+1])
            put!("tomo_sigma_line_$tag", θ[o+sh+1])
            put!("matern_sigma_tomo_$tag", 10.0^θ[o+sh+2]); logJ += log(10.0^θ[o+sh+2] * OS_LN10)
            put!("tomo_ell_v_$tag", 10.0^θ[o+sh+3]);        logJ += log(10.0^θ[o+sh+3] * OS_LN10)
            put!("matern_rho_tomo_$tag", 10.0^θ[o+sh+4] / 24)
            logJ += log(10.0^θ[o+sh+4] / 24 * OS_LN10)
            put!("tomo_jit_$tag", 10.0^θ[o+sh+5]);          logJ += log(10.0^θ[o+sh+5] * OS_LN10)
        end
        # the bespoke joint vector carries the map block even when the maps are
        # off (`use_tomogram = false`)
        base = 4 + 6 * ntomo_theta + 1
        isempty(rm_tags) || put!("K_k1", θ[base])
    end
    for (q, tag) in enumerate(rm_tags)
        o = base + (q - 1) * 5
        put!("gamma_$tag", θ[o+1])
        put!("gp_log_S0_$tag", θ[o+2] * OS_LN10);     logJ += log(OS_LN10)
        put!("gp_log_Q_$tag", θ[o+3] * OS_LN10);      logJ += log(OS_LN10)
        put!("gp_log_omega0_$tag", θ[o+4] * OS_LN10); logJ += log(OS_LN10)
        put!("sigma_$tag", 10.0^θ[o+5]);              logJ += log(10.0^θ[o+5] * OS_LN10)
    end
    return th, logJ
end
