# Synthetic CCF blends for the CCF line-component tests (src/ccf_data.jl,
# src/ccf_likelihood.jl): a planet orbit, Gaussian components on it, and the
# CCFs they make, with white noise at a stated per-pixel level.
using Random
using Nereus

const CS_P = 4.2
const CS_TC = 2459500.3
const CS_SESINW = 0.3
const CS_SECOSW = 0.1
const CS_E = CS_SESINW^2 + CS_SECOSW^2
const CS_W = atan(CS_SESINW, CS_SECOSW)

# The unit Keplerian cos(f + ω) + e cos ω at t, written out independently of
# Nereus: transit at f = π/2 − ω, Kepler's equation by Newton's method.
function cs_kep(t; P = CS_P, Tc = CS_TC, e = CS_E, w = CS_W)
    f_tr = π / 2 - w
    E_tr = 2atan(sqrt((1 - e) / (1 + e)) * tan(f_tr / 2))
    M = E_tr - e * sin(E_tr) + 2π * (t - Tc) / P
    E = M
    for _ in 1:60
        E -= (E - e * sin(E) - M) / (1 - e * cos(E))
    end
    f = 2atan(sqrt(1 + e) * sin(E / 2), sqrt(1 - e) * cos(E / 2))
    return cos(f + w) + e * cos(w)
end

# Component truth: (name, planet, γ, K, FWHM, depth).
const CS_BLEND = [("A", 0, 10.0, 0.0, 12.0, 0.15), ("B", 1, 8.0, 20.0, 18.0, 0.06)]

"""
    cs_ccfs(; comps, n_epochs, noise, seed, unequal, nan, cont_sd, scale_sd, trend, t_ref)

Synthetic CCFs: `(t, v, y, truth)`. Epoch n has continuum `cont[n]` and depth
scale `scale[n]` (`scale[1] = 1`). `unequal`: every other epoch on a coarser,
shifted grid. `nan`: a few pixels of epoch 2 set to NaN. `trend`: name =>
(γ̇, γ̈) adds γ̇ (t − t_ref) + ½ γ̈ (t − t_ref)² to that component's centre,
`t_ref` (default the mean epoch) in `truth.t_ref`.
"""
function cs_ccfs(; comps = CS_BLEND, n_epochs = 14, noise = 0.004, seed = 1,
                 unequal = false, nan = false, cont_sd = 0.01, scale_sd = 0.15,
                 trend = Dict{String, NTuple{2, Float64}}(), t_ref = nothing)
    rng = MersenneTwister(seed)
    t = sort!(CS_TC .+ 40 .* rand(rng, n_epochs))
    cont = 1 .+ cont_sd .* randn(rng, n_epochs)
    scale = vcat(1.0, clamp.(1 .+ scale_sd .* randn(rng, n_epochs - 1), 0.5, 1.6))
    tr = t_ref === nothing ? sum(t) / n_epochs : t_ref
    vs = Vector{Vector{Float64}}(); ys = similar(vs)
    for n in 1:n_epochs
        v = (unequal && iseven(n)) ? collect(-39.3:0.8:60.0) : collect(-40.0:0.5:60.0)
        m = fill(cont[n], length(v))
        for (nm, pl, γ, K, fw, d) in comps
            c = pl == 0 ? γ : γ + K * cs_kep(t[n])
            if haskey(trend, nm)
                dt = t[n] - tr
                c += trend[nm][1] * dt + trend[nm][2] * dt^2 / 2
            end
            σ = fw / (2sqrt(2log(2)))
            m .-= scale[n] * d .* exp.(-((v .- c) ./ σ) .^ 2 ./ 2)
        end
        y = m .+ noise .* randn(rng, length(v))
        (nan && n == 2) && (y[[5, 50, 51]] .= NaN)
        push!(vs, v); push!(ys, y)
    end
    return t, vs, ys, (; cont, scale, comps, noise, trend, t_ref = tr)
end

cs_components(comps = CS_BLEND) = [CCFComponent(c[1]; planet = c[2]) for c in comps]

# Orbit priors fixed at the truth (time anchor Tc).
cs_orbit_priors() = Dict{String, Nereus.PriorSpec}(
    "P_k1" => FixedPrior(CS_P), "Tc_k1" => FixedPrior(CS_TC),
    "sesinw_k1" => FixedPrior(CS_SESINW), "secosw_k1" => FixedPrior(CS_SECOSW),
    "K_k1" => FixedPrior(100.0))

"""
A CCF-only Params/Data: CCFs synthesised from `comps`, fitted with the
components of `fit` (default the same); `kw` go to `CCFSet`.
"""
function cs_model(; comps = CS_BLEND, fit = comps,
                  priors = Dict{String, Nereus.PriorSpec}(), synth = (;), kw...)
    t, v, y, truth = cs_ccfs(; comps, synth...)
    s = CCFSet(; tag = "SIM", t, v, ccf = y, components = cs_components(fit),
               normalize_edges = 0, kw...)
    data = Data(; ccf = [s])
    pr = merge(cs_orbit_priors(), priors)
    p = Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY], data, priors = pr,
               parametrization = ParametrizationConfig(time = :Tc))
    return p, data, truth
end

"""A Theta of `p` at the truth (`jit` for every jitter)."""
function cs_truth_theta(p, truth; jit = truth.noise)
    th = Nereus.Theta{Float64}(p)
    L = p.layout.name_to_idx
    set!(nm, x) = haskey(L, nm) && (th.values[L[nm]] = x)
    for (nm, _, γ, K, fw, d) in truth.comps
        set!("ccf_gamma_SIM_$nm", γ); set!("ccf_K_SIM_$nm", K)
        set!("ccf_fwhm_SIM_$nm", fw); set!("ccf_depth_SIM_$nm", d)
    end
    for n in eachindex(truth.cont)
        set!("ccf_cont_SIM_$n", truth.cont[n]); set!("ccf_scale_SIM_$n", truth.scale[n])
        set!("ccf_jit_SIM_$n", jit)
    end
    set!("ccf_cont_SIM", 1.0); set!("ccf_jit_SIM", jit)
    return th
end

"""`data.ccf` block of a run_job config with the synthetic CCFs inline."""
function cs_ccf_block(; comps = CS_BLEND, synth = (;), extra = Dict{String, Any}())
    t, v, y, _ = cs_ccfs(; comps, synth...)
    eps = [Dict{String, Any}("bjd" => t[n], "velocities" => v[n], "ccf" => y[n])
           for n in eachindex(t)]
    cs = [Dict{String, Any}("name" => c[1], "planet" => c[2]) for c in comps]
    return merge(Dict{String, Any}("tag" => "SIM", "components" => cs, "epochs" => eps,
                                   "normalize_edges" => 0), extra)
end

cs_prior_cfg(; P = CS_P, Tc = CS_TC) = Dict{String, Any}(
    "P_k1" => Dict("type" => "fixed", "args" => [P]),
    "Tc_k1" => Dict("type" => "fixed", "args" => [Tc]),
    "sesinw_k1" => Dict("type" => "fixed", "args" => [CS_SESINW]),
    "secosw_k1" => Dict("type" => "fixed", "args" => [CS_SECOSW]),
    "K_k1" => Dict("type" => "fixed", "args" => [100.0]))
