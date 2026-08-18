#!/usr/bin/env julia
# Add paper-standard plots from the saved Kepler-88 TTV-C chain:
#   1. M_c marginal posterior with literature reference lines
#   2. (M_c, e_c) joint corner — degeneracy + masses lit comparison

using Nereus
using Statistics: median, quantile
using Printf
using CairoMakie

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const OUT_DIR   = joinpath(REPO_ROOT, "Nereus.jl", "results", "Kepler88_TTV_C")
const M_SUN = 1.989e33
const M_E   = 5.972e27
const M_s_LIT  = 0.99
const P_c_LIT  = 22.3395

chains, _ = load_chains(joinpath(OUT_DIR, "kepler88_ttvc_chains.nc"))

K_c  = vec(Array(chains[:, :K_k2, :]))
se_c = vec(Array(chains[:, :sesinw_k2, :]))
sc_c = vec(Array(chains[:, :secosw_k2, :]))
e_c  = hypot.(se_c, sc_c).^2
m_c_E = [msini(M_s_LIT, K_c[i], P_c_LIT, e_c[i]) * 1.898e30 / M_E
         for i in 1:length(K_c)]

m_med = median(m_c_E)
m_lo  = quantile(m_c_E, 0.16)
m_hi  = quantile(m_c_E, 0.84)
@printf("M_c = %.0f [%.0f, %.0f] M_⊕\n", m_med, m_lo, m_hi)

e_med = median(e_c); e_lo = quantile(e_c, 0.16); e_hi = quantile(e_c, 0.84)
@printf("e_c = %.3f [%.3f, %.3f]\n", e_med, e_lo, e_hi)

set_theme!(nereus_theme())

# --- M_c marginal ---
fig1 = Figure(size = (700, 450))
ax1 = Axis(fig1[1, 1]; xlabel = "M_c [M_⊕]", ylabel = "density")
hist!(ax1, m_c_E; bins = 50, color = (:black, 0.55), strokewidth = 0,
       normalization = :pdf)
vlines!(ax1, [190.0]; color = :crimson, linewidth = 1.5, linestyle = :dash,
         label = "Nesvorný+ 2013 (190)")
vlines!(ax1, [214.0]; color = :royalblue, linewidth = 1.5, linestyle = :dash,
         label = "Yoffe+ 2021 (214)")
vlines!(ax1, [m_med]; color = :black, linewidth = 1.5,
         label = @sprintf("This work: %.0f [%.0f, %.0f]", m_med, m_lo, m_hi))
axislegend(ax1; position = :rt, framevisible = false)
save(joinpath(OUT_DIR, "mc_posterior.png"), fig1)
println("Wrote $(OUT_DIR)/mc_posterior.png")

# --- (M_c, e_c) corner ---
fig2 = Figure(size = (700, 700))
ax2 = Axis(fig2[1, 1]; xlabel = "M_c [M_⊕]", ylabel = "e_c")
scatter!(ax2, m_c_E, e_c; color = (:black, 0.10), markersize = 4,
          strokewidth = 0)
vlines!(ax2, [190.0]; color = :crimson, linewidth = 1.0, linestyle = :dash)
vlines!(ax2, [214.0]; color = :royalblue, linewidth = 1.0, linestyle = :dash)
save(joinpath(OUT_DIR, "mc_ec_joint.png"), fig2)
println("Wrote $(OUT_DIR)/mc_ec_joint.png")
