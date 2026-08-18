# Direct likelihood comparison: our found mode (P=372 yr) vs Brandt
# mode (P=2880 yr). Bypass priors, compute pure log-likelihood.

using Nereus, PlanetOrbits, Printf
using Statistics: median
using LinearAlgebra: I

DATADIR = joinpath(@__DIR__, "..", "..", "..", "data", "HD159062")
hgca   = load_hgca_row(joinpath(DATADIR, "HGCA_vEDR3.fits"), 85653)
rvdat  = load_orvara_rv(joinpath(DATADIR, "HD159062_RV.dat"))
relast = load_orvara_relast(joinpath(DATADIR, "HD159062_relAST.txt"))
data = Data(t_rv=rvdat.t, rv=rvdat.rv, rv_err=rvdat.rv_err,
            rv_inst=rvdat.rv_inst, relastrom=relast, hgca=hgca)

# Build a setup that lets us evaluate the likelihood at any (P, M_sec,
# e, ω, i, Ω, Tp) directly. We'll use the existing Nereus Theta but
# manually set all values.
params = Params(
    max_kplanet = 1, planet_modes = [RVAS],
    instruments = InstrumentConfig(rv = ["HIRES"]),
    data = data, stability = :none, M_s = 0.81,
    parametrization = ParametrizationConfig(mass = :M_sec_driven),
    priors = Dict{String, PriorSpec}(
        "n_p"          => FixedPrior(1.0),
        "P_k1"         => LogUniformPrior(50*365.25, 30000*365.25),
        "M_sec_k1"     => LogUniformPrior(0.001, 2.0),
        "sesinw_k1"    => UniformPrior(-1.0, 1.0),
        "secosw_k1"    => UniformPrior(-1.0, 1.0),
        "Mo_k1"        => UniformPrior(0.0, 2π),
        "inc_k1"       => SinePrior(),
        "Omega_k1"     => UniformPrior(0.0, 2π),
        "sigma_HIRES"  => LogUniformPrior(1e-3, 1e3),
        "M_pri"        => NormalPrior(0.81, 0.04, 0.5, 1.2),
    ),
)

function eval_at(P_yr, M_sec, e, ω_deg, i_deg, Ω_deg, Mo, M_pri, sigma_jit, gamma)
    theta = Theta(params)
    set_param!(theta, "n_p", 1.0)
    set_param!(theta, "P_k1", P_yr * 365.25)
    set_param!(theta, "M_sec_k1", M_sec)
    set_param!(theta, "sesinw_k1", sqrt(e) * sin(deg2rad(ω_deg)))
    set_param!(theta, "secosw_k1", sqrt(e) * cos(deg2rad(ω_deg)))
    set_param!(theta, "Mo_k1", Mo)
    set_param!(theta, "inc_k1", deg2rad(i_deg))
    set_param!(theta, "Omega_k1", deg2rad(Ω_deg))
    set_param!(theta, "plx", hgca.plx)
    set_param!(theta, "gamma_HIRES", gamma)
    set_param!(theta, "sigma_HIRES", sigma_jit)
    set_param!(theta, "M_pri", M_pri)
    return theta
end

function ll_breakdown(theta, data)
    ll_rel  = relastrom_log_likelihood(theta, data)
    ll_hg   = hgca_log_likelihood(theta, data)
    ll_full = rv_log_likelihood(theta, data)
    ll_rv   = ll_full - ll_rel - ll_hg
    return (rel=ll_rel, hgca=ll_hg, rv=ll_rv, total=ll_full)
end

println("=" ^ 70)
println("Direct likelihood comparison: HD 159062 B")
println("=" ^ 70)

# --- Our found mode (M_sec_driven Pigeons posterior) ---
println("\n[1] Our Pigeons mode: P=372yr, M_sec=0.542, e=0.72, ω~150°, i=43°, Ω=154°")
# Sweep Mo to find best phase
function _sweep_mo(P_yr, M_sec, e, ω_deg, i_deg, Ω_deg, M_pri, σ, γ)
    best_mo, best_ll = 0.0, -Inf
    for mo in range(0, 2π, length=30)
        th = eval_at(P_yr, M_sec, e, ω_deg, i_deg, Ω_deg, mo, M_pri, σ, γ)
        ll = rv_log_likelihood(th, data)
        if ll > best_ll
            best_ll = ll; best_mo = mo
        end
    end
    return best_mo, best_ll
end
best_mo_1, _ = _sweep_mo(372.4, 0.5419, 0.7228, 150.0, 42.94, 154.08, 0.7082, 5.0, median(rvdat.rv))
th_ours = eval_at(372.4, 0.5419, 0.7228, 150.0, 42.94, 154.08, best_mo_1, 0.7082, 5.0, median(rvdat.rv))
b = ll_breakdown(th_ours, data)
@printf("  Mo = %.2f rad (best of sweep)\n", best_mo_1)
@printf("  RV ll        = %.2f\n", b.rv)
@printf("  rel ll       = %.2f\n", b.rel)
@printf("  HGCA ll      = %.2f\n", b.hgca)
@printf("  TOTAL ll     = %.2f\n", b.total)

# --- Brandt's reported mode (orvara medians) ---
println("\n[2] Brandt+ 2021 mode: P=2880yr, M_sec=0.6083, e=0.51, ω=246°(comp), i=50.5°, Ω=90.4°, M_pri=0.81")
best_mo_2, _ = _sweep_mo(2880.0, 0.6083, 0.51, 246.0, 50.5, 90.4, 0.81, 5.0, median(rvdat.rv))
th_brandt = eval_at(2880.0, 0.6083, 0.51, 246.0, 50.5, 90.4, best_mo_2, 0.81, 5.0, median(rvdat.rv))
b = ll_breakdown(th_brandt, data)
@printf("  Mo = %.2f rad (best of sweep)\n", best_mo_2)
@printf("  RV ll        = %.2f\n", b.rv)
@printf("  rel ll       = %.2f\n", b.rel)
@printf("  HGCA ll      = %.2f\n", b.hgca)
@printf("  TOTAL ll     = %.2f\n", b.total)

# Sweep gamma + sigma at Brandt's mode
function _sweep_gs(P_yr, M_sec, e, ω, i, Ω, mo, M_pri)
    best_total, best_g, best_s = -Inf, 0.0, 0.0
    for γ in range(median(rvdat.rv) - 50, median(rvdat.rv) + 50, length=21)
        for σ in (1.0, 5.0, 20.0, 50.0, 100.0, 300.0, 1000.0)
            th = eval_at(P_yr, M_sec, e, ω, i, Ω, mo, M_pri, σ, γ)
            ll = rv_log_likelihood(th, data)
            if ll > best_total
                best_total = ll; best_g = γ; best_s = σ
            end
        end
    end
    return best_g, best_s, best_total
end
println("\n[2b] Brandt mode with gamma/sigma optimization:")
best_g, best_s, best_total = _sweep_gs(2880.0, 0.6083, 0.51, 246.0, 50.5, 90.4, best_mo_2, 0.81)
@printf("  best γ=%.2f, σ=%.2f → total ll = %.2f\n", best_g, best_s, best_total)

println("\n" * "=" ^ 70)
println("VERDICT:")
@printf("  Our mode (P=372)   total ll = %.2f\n", ll_breakdown(th_ours, data).total)
@printf("  Brandt mode (best) total ll = %.2f\n", best_total)
println("If Brandt's mode is much WORSE in our likelihood, our likelihood")
println("genuinely disagrees with orvara's. If they're comparable, the")
println("problem is sampler exploration only.")
