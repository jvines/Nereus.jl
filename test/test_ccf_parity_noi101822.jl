# Parity of the CCF line-component target with the reference least-squares
# analysis of NOI-101822's FEROS CCFs (19 epochs): two Gaussian components,
# both on the transit ephemeris, per-epoch continuum and depth scale, P, Tc, e,
# ω fixed from the transit+RV fit. The reference procedure, reproduced here
# through the Nereus target and `sample_map`:
#
#   1. fit with unit weights;
#   2. per-epoch σ_n = std of that fit's residuals (population std);
#   3. refit with weights 1/σ_n (here: the per-epoch jitters fixed at σ_n, which
#      makes ln L the reference's χ² plus a constant), one and two components;
#   4. ΔBIC with N scaled to N_eff = N(1 − ρ)/(1 + ρ), ρ the lag-1
#      autocorrelation of the two-component residuals.
#
# The reference values are those of that script on these CCFs, to full
# precision (rounded in the comments). Its depths are not identified on their
# own -- a common factor moves between the depths and the 19 depth scales -- so
# depths are compared as d_j·s_n at the reference epoch, which are.
#
# The CCFs are unpublished and are not in this repository: the test runs when
# NEREUS_NOI101822_CCF_DIR names a directory holding the 19 dumps
# (`<id>.json`), and is skipped otherwise. NEREUS_CCF_PARITY_POSTERIOR=1 adds
# a pt_emcee posterior with free per-epoch jitters.
using Test
using Nereus
using Statistics

const NOI_DIR = get(ENV, "NEREUS_NOI101822_CCF_DIR", "")

# Argus id => BJD_TDB of each FEROS CCF.
const NOI_BJD = Dict(27930 => 2459267.59447, 27874 => 2459269.59447, 34078 => 2459645.64162,
    33999 => 2459647.57492, 34338 => 2459682.51380, 34340 => 2459682.54521,
    34939 => 2459683.51867, 42858 => 2459703.46864, 35109 => 2459704.49919,
    35249 => 2459835.83441, 33378 => 2459858.81837, 33454 => 2459859.77176,
    33482 => 2459861.82073, 33579 => 2459862.77395, 33436 => 2459870.83967,
    33796 => 2459927.77731, 33819 => 2459929.69523, 35875 => 2459953.70107,
    35716 => 2459966.71699)

# Orbit of nereus run 51 (transit + RV), held fixed.
const NOI_ORBIT = Dict{String, Nereus.PriorSpec}(
    "P_k1" => FixedPrior(9.483157326), "Tc_k1" => FixedPrior(2460097.5261792196),
    "sesinw_k1" => FixedPrior(0.5229), "secosw_k1" => FixedPrior(-0.1092),
    "K_k1" => FixedPrior(3021.0))

# The reference, full precision.
const NOI_REF = (
    gA = 61.96517006224131, KA = -0.3303990247413455, sA = 5.883539203858372,
    dA = 0.13414405056884365, gB = 60.988952443602024, KB = 17.296286369420322,
    sB = 7.776630951536299, dB = 0.06497066693459119, s_ref = 0.85901553,  # epoch 27874
    chi1 = 22164.58083689342, chi2 = 10047.300993338347, rho = 0.9922636510393266,
    Neff = 42.87047674703008, dBIC = 32.021031229037135)

# The reference's per-epoch σ_n (population std of its unit-weight two-component
# residuals), in sorted-id order.
const NOI_REF_SIG = [0.006045143823508335, 0.02494118077025889, 0.009599645355723578,
    0.011191619492788817, 0.010614337223002202, 0.009292435524545136, 0.010035177429715686,
    0.008945591622172622, 0.0076559364124589874, 0.01921842492400311, 0.009338548850027285,
    0.013942661222837605, 0.012038932319453053, 0.01267678085120557, 0.00752547883745103,
    0.010187077206737729, 0.007704560344444052, 0.005412995996040706, 0.02245865478673928]

const _FW = 2sqrt(2log(2))

function noi_load()
    ids = sort!(parse.(Int, first.(splitext.(filter(endswith(".json"), readdir(NOI_DIR))))))
    @assert Set(ids) == Set(keys(NOI_BJD)) "expected the 19 FEROS dumps in $NOI_DIR"
    vs = Vector{Vector{Float64}}(); ys = similar(vs)
    for id in ids
        _, v, y = load_ccf_json(joinpath(NOI_DIR, "$id.json")); push!(vs, v); push!(ys, y)
    end
    return ids, [NOI_BJD[i] for i in ids], vs, ys
end

# Reference bounds: γ ∈ [40, 80] (B: [0, 100]), K ∈ [−40, 40], σ ∈ [1.5, 25],
# depth ∈ [0, 1], continuum ∈ [0.9, 1.1], depth scale ∈ [0.2, 3].
function noi_model(ids, t, vs, ys, names; jit, planets = fill(1, length(names)),
                   jitter_free = false)
    comps = [CCFComponent(nm; planet = pl) for (nm, pl) in zip(names, planets)]
    s = CCFSet(; tag = "FEROS", t, v = vs, ccf = ys, labels = string.(ids), components = comps)
    data = Data(; ccf = [s])
    pr = copy(NOI_ORBIT)
    for (j, nm) in enumerate(names)
        pr["ccf_gamma_FEROS_$nm"] = j == 1 ? UniformPrior(40.0, 80.0) : UniformPrior(0.0, 100.0)
        planets[j] == 0 || (pr["ccf_K_FEROS_$nm"] = UniformPrior(-40.0, 40.0))
        pr["ccf_fwhm_FEROS_$nm"] = UniformPrior(1.5_FW, 25_FW)
        pr["ccf_depth_FEROS_$nm"] = UniformPrior(0.0, 1.0)
    end
    for (n, id) in enumerate(ids)
        pr["ccf_cont_FEROS_$id"] = UniformPrior(0.9, 1.1)
        n > 1 && (pr["ccf_scale_FEROS_$id"] = UniformPrior(0.2, 3.0))
        jitter_free || (pr["ccf_jit_FEROS_$id"] = FixedPrior(jit[n]))
    end
    p = Params(; max_kplanet = 1, planet_modes = [Nereus.RV_ONLY], data, priors = pr,
               parametrization = ParametrizationConfig(time = :Tc))
    return p, data
end

# A bounded start: components at `g` (γ, K, σ, d per component), continuum and
# scales 1, as the reference starts.
function noi_start(p, names, g)
    th = Nereus.Theta{Float64}(p); L = p.layout.name_to_idx
    for n in p.layout.names
        startswith(n, "ccf_cont_") && (th.values[L[n]] = 1.0)
        startswith(n, "ccf_scale_") && (th.values[L[n]] = 1.0)
    end
    for (j, nm) in enumerate(names)
        γ, K, σ, d = g[j]
        th.values[L["ccf_gamma_FEROS_$nm"]] = γ
        haskey(L, "ccf_K_FEROS_$nm") && (th.values[L["ccf_K_FEROS_$nm"]] = K)
        th.values[L["ccf_fwhm_FEROS_$nm"]] = σ * _FW
        th.values[L["ccf_depth_FEROS_$nm"]] = d
    end
    return th.values[p.layout.unfrozen_idx]
end

# Best MAP over the starts.
function noi_fit(p, data, names, starts)
    tg = NereusTarget(p, data)
    best = nothing
    for g in starts
        x0 = g isa AbstractVector{<:Real} ? g : noi_start(p, names, g)
        r = Nereus.sample_map(tg; init = Nereus.transform_forward(x0, tg.transform),
                              n_starts = 1)
        (best === nothing || r.log_posterior > best.log_posterior) && (best = r)
    end
    # Restart LBFGS from the best point until ln p stops moving: its stopping
    # rule can end a hair short of the optimum on this ridge (seen: χ² 0.025
    # above the reference's for one component).
    for _ in 1:5
        r = Nereus.sample_map(tg; init = Nereus.transform_forward(best.x_map, tg.transform),
                              n_starts = 1)
        gain = r.log_posterior - best.log_posterior
        gain > 0 && (best = r)
        gain < 1e-6 && break
    end
    th = Nereus.Theta{Float64}(p); Nereus.set_unfrozen!(th, best.x_map)
    return best, th
end

noi_chi2(th, data) = (m = ccf_model(th, data)[1];
                      sum(sum(abs2, m.residual[n] ./ m.sigma[n]) for n in eachindex(m.t)))

@testset "CCF parity: NOI-101822 FEROS" begin
    if isempty(NOI_DIR) || !isdir(NOI_DIR)
        @info "NEREUS_NOI101822_CCF_DIR not set: NOI-101822 CCF parity skipped"
        @test_skip false
    else
        ids, t, vs, ys = noi_load()
        ne = length(ids)
        two, one_ = ["A", "B"], ["A"]
        s2 = [[(61.0, 0.0, 5.0, 0.2), (gB, KB, 6.0, 0.08)] for KB in (-18.0, 18.0)
              for gB in (58.0, 66.0)]
        # One Gaussian on a blend has neighbouring local optima; the reference's
        # starts in K alone left LBFGS in one 0.025 above its χ², so the width is
        # gridded too (the model at the reference's point gives its χ² to 1e-12).
        s1 = [[(61.0, K, σ, 0.25)] for K in (-15.0, -3.0, 0.0, 3.0, 15.0) for σ in (5.0, 9.0)]

        # 1-2: unit weights, then per-epoch σ from the residuals
        p, d = noi_model(ids, t, vs, ys, two; jit = ones(ne))
        r0, th0 = noi_fit(p, d, two, s2)
        m0 = ccf_model(th0, d)[1]
        sig = [std(m0.residual[n]; corrected = false) for n in 1:ne]

        # 3: weighted refits, starting also from the unit-weight optimum
        p2, d2 = noi_model(ids, t, vs, ys, two; jit = sig)
        r2, th2 = noi_fit(p2, d2, two, vcat(s2, [r0.x_map]))
        # one component as the reference does it: unit weights first, and that
        # optimum among the starts of the weighted fit
        p1u, d1u = noi_model(ids, t, vs, ys, one_; jit = ones(ne))
        r1u, _ = noi_fit(p1u, d1u, one_, s1)
        p1, d1 = noi_model(ids, t, vs, ys, one_; jit = sig)
        r1, th1 = noi_fit(p1, d1, one_, vcat(s1, [r1u.x_map]))
        L = p2.layout.name_to_idx; g(nm) = th2.values[L[nm]]

        # the reference's numbers, to the precision it states them
        @test round(g("ccf_K_FEROS_A"); digits = 2) == -0.33
        @test round(g("ccf_gamma_FEROS_A"); digits = 2) == 61.97
        @test round(g("ccf_fwhm_FEROS_A"); digits = 1) == 13.9
        @test round(g("ccf_K_FEROS_B"); digits = 2) == 17.30
        @test round(g("ccf_gamma_FEROS_B"); digits = 2) == 60.99
        @test round(g("ccf_fwhm_FEROS_B"); digits = 1) == 18.3

        # and to the optimiser's tolerance against the full-precision values
        @test g("ccf_K_FEROS_A") ≈ NOI_REF.KA atol = 2e-3
        @test g("ccf_K_FEROS_B") ≈ NOI_REF.KB atol = 2e-3
        @test g("ccf_gamma_FEROS_A") ≈ NOI_REF.gA atol = 2e-3
        @test g("ccf_gamma_FEROS_B") ≈ NOI_REF.gB atol = 2e-3
        @test g("ccf_fwhm_FEROS_A") ≈ NOI_REF.sA * _FW atol = 5e-3
        @test g("ccf_fwhm_FEROS_B") ≈ NOI_REF.sB * _FW atol = 5e-3
        @test g("ccf_depth_FEROS_A") ≈ NOI_REF.dA * NOI_REF.s_ref rtol = 1e-3
        @test g("ccf_depth_FEROS_B") ≈ NOI_REF.dB * NOI_REF.s_ref rtol = 1e-3
        # σ_n from nereus' own unit-weight optimum against the reference's: they
        # differ only by how far each optimiser converged (nereus' LBFGS goes
        # further than the reference's trust-region stop; its χ² below comes out
        # lower), and χ² inherits that at 2δσ/σ.
        δσ = maximum(abs.(sig .- NOI_REF_SIG) ./ NOI_REF_SIG)
        @test δσ < 1e-4
        c1, c2 = noi_chi2(th1, d1), noi_chi2(th2, d2)
        @test c2 ≈ NOI_REF.chi2 rtol = 2δσ + 1e-7
        @test c1 ≈ NOI_REF.chi1 rtol = 2δσ + 1e-7
        # With the reference's own σ_n the weighted fits are the reference's
        # problem exactly, and χ² must agree to the optimiser's tolerance (nereus
        # may only do better).
        p2r, d2r = noi_model(ids, t, vs, ys, two; jit = NOI_REF_SIG)
        _, th2r = noi_fit(p2r, d2r, two, [r2.x_map])
        p1r, d1r = noi_model(ids, t, vs, ys, one_; jit = NOI_REF_SIG)
        _, th1r = noi_fit(p1r, d1r, one_, vcat(s1, [r1.x_map]))
        c1r, c2r = noi_chi2(th1r, d1r), noi_chi2(th2r, d2r)
        @test c2r <= NOI_REF.chi2 * (1 + 1e-9)
        @test c1r <= NOI_REF.chi1 * (1 + 1e-9)
        @test c2r ≈ NOI_REF.chi2 rtol = 1e-6
        @test c1r ≈ NOI_REF.chi1 rtol = 1e-6

        # 4: ΔBIC with the effective sample size
        m = ccf_model(th2, d2)[1]
        rr = reduce(vcat, [m.residual[n] ./ m.sigma[n] for n in 1:ne])
        ρ = cor(rr[1:end-1], rr[2:end]); N = length(rr); Neff = N * (1 - ρ) / (1 + ρ)
        dBIC = (c1 * Neff / N + (4 + 2ne) * log(Neff)) - (c2 * Neff / N + (8 + 2ne) * log(Neff))
        @test ρ ≈ NOI_REF.rho atol = 1e-4
        @test dBIC ≈ NOI_REF.dBIC atol = 0.1
        @test round(dBIC) == 32
        @info "NOI-101822 CCF parity" KA = g("ccf_K_FEROS_A") gA = g("ccf_gamma_FEROS_A") FWHM_A = g("ccf_fwhm_FEROS_A") depth_A_ref = g("ccf_depth_FEROS_A") / NOI_REF.s_ref KB = g("ccf_K_FEROS_B") gB = g("ccf_gamma_FEROS_B") FWHM_B = g("ccf_fwhm_FEROS_B") depth_B_ref = g("ccf_depth_FEROS_B") / NOI_REF.s_ref chi1 = c1 chi2 = c2 chi1_ref_sigma = c1r chi2_ref_sigma = c2r max_rel_dsigma = δσ rho = ρ dBIC

        if get(ENV, "NEREUS_CCF_PARITY_POSTERIOR", "0") == "1"
            # Posterior with free per-epoch jitters, A static and B on the orbit
            # (the physical reading of the reference fit, K_A ≈ 0).
            pp, dp = noi_model(ids, t, vs, ys, two; jit = sig, planets = [0, 1],
                               jitter_free = true)
            tg = NereusTarget(pp, dp; unconstrained = false)
            res = sample_pt_emcee(tg, dp; n_temps = 4, n_walkers = 160, n_steps = 6000,
                                  n_burnin = 3000, seed = 1, show_progress = false)
            kb = vec(Array(res.chains[:ccf_K_FEROS_B]))
            q = quantile(kb, [0.16, 0.5, 0.84])
            @info "NOI-101822 posterior K_B (white-noise likelihood)" q
            # the bootstrap's median and 68% interval: 17.0 [15.3, 18.1]
            @test 15.3 < q[2] < 18.1
        end
    end
end
