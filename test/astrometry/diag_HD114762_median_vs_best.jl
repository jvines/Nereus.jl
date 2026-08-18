#!/usr/bin/env julia
# Is HD 114762's "RV −2logL/N ≈ 80" a bad FIT, or a bad EVALUATION POINT?
#
# fit_HD114762_rv_dr4_epoch.jl reports per-channel logL at the MARGINAL MEDIAN of
# each parameter. For a curved or multimodal posterior that point need not lie in
# any mode, so the number can look catastrophic while the fit is fine. Meanwhile
# plot_rv_timeseries showed ±25 m/s Lick / ±3 m/s HIRES residuals — inconsistent
# with −2logL/N ≈ 80. Exactly one of those is telling the truth.
#
# This scores EVERY posterior draw, then compares the marginal-median theta with
# the best-likelihood draw, per channel, and plots the RV residuals at both.
#
# Usage:  julia -t auto --project=. test/astrometry/diag_HD114762_median_vs_best.jl
#   env HD114762_RV, NEREUS_GAIA_DR4_XML
#   env CHAINS=<chains.nc>   defaults to the trend_order=1 fit
#   env HD114762_TREND=<n>   must match the chains being loaded

using Nereus
using MCMCChains
using CairoMakie
using Statistics: median, quantile
using Printf
using Nereus: nereus_theme, NEREUS_COLORS, INST_COLORS, INST_MARKERS

const SID = 3937211745905473024
const M_PRI, PLX, PLX_ERR = 0.83, 25.36, 0.30

rvfile = get(ENV, "HD114762_RV", "")
isfile(rvfile) || error("set HD114762_RV")
tb = Dict{String, Vector{Float64}}()
for line in eachline(rvfile)
    s = strip(line); (isempty(s) || startswith(s, "#")) && continue
    t, rv, er, ins = split(s)
    append!(get!(() -> Float64[], tb, ins),
            (parse(Float64, t) - 2_400_000.5, parse(Float64, rv), parse(Float64, er)))
end
mkrv(k) = let v = reshape(tb[k], 3, :); (t = v[1, :], rv = v[2, :], rv_err = v[3, :]) end
hires, lick = mkrv("j"), mkrv("lick")

xml = get(ENV, "NEREUS_GAIA_DR4_XML", "")
(isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, SID)

target = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a = LogUniformPrior(0.30, 0.45), M_sec = LogUniformPrior(0.003, 0.5),
        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
        Mo = UniformPrior(0.0, 2π), inc = SinePrior(), Omega = UniformPrior(0.0, 2π),
    ),),
    rv = (HIRES = (data = hires, sigma = LogUniformPrior(0.5, 50.0)),
          Lick  = (data = lick,  sigma = LogUniformPrior(0.5, 50.0))),
    iad = src.iad, plx = NormalPrior(PLX, PLX_ERR), M_s = M_PRI,
    trend_order = parse(Int, get(ENV, "HD114762_TREND", "1")),
)
params, data = target.params, target.data
nms = params.layout.unfrozen_names

CHAINS = get(ENV, "CHAINS",
             joinpath(@__DIR__, "plots_HD114762_dr4", "chains.nc"))
chains, _ = load_chains(CHAINS)
cols = Dict(nm => vec(Array(chains[:, Symbol(nm), :])) for nm in nms)
ndraw = length(first(values(cols)))
@printf("scoring %d draws over %d params from %s\n", ndraw, length(nms), basename(CHAINS))

# --- score every draw ---------------------------------------------------------
th = Theta(params)
setdraw!(j) = for nm in nms; set_param!(th, string(nm), cols[nm][j]); end
ll_rv  = Vector{Float64}(undef, ndraw)
ll_ast = Vector{Float64}(undef, ndraw)
for j in 1:ndraw
    setdraw!(j)
    ll_rv[j]  = Nereus.rv_log_likelihood(th, data)
    ll_ast[j] = Nereus.astrom_log_likelihood(th, data)
end
tot = ll_rv .+ ll_ast
jbest = argmax(tot)

n_rv, n_ast = length(data.t_rv), length(src.iad.t)

# marginal-median theta (what the fit script reports)
for nm in nms; set_param!(th, string(nm), median(cols[nm])); end
med_rv  = Nereus.rv_log_likelihood(th, data)
med_ast = Nereus.astrom_log_likelihood(th, data)

@printf("\n%-22s %12s %12s %12s %12s\n", "", "RV logL", "RV −2logL/N", "AST logL", "AST −2logL/N")
@printf("%-22s %12.1f %12.1f %12.1f %12.1f\n", "marginal-median theta",
        med_rv, -2med_rv / n_rv, med_ast, -2med_ast / n_ast)
@printf("%-22s %12.1f %12.1f %12.1f %12.1f\n", "best-likelihood draw",
        ll_rv[jbest], -2ll_rv[jbest] / n_rv, ll_ast[jbest], -2ll_ast[jbest] / n_ast)
@printf("%-22s %12.1f %12.1f %12.1f %12.1f\n", "posterior-median draw",
        median(ll_rv), -2median(ll_rv) / n_rv, median(ll_ast), -2median(ll_ast) / n_ast)

# --- residuals at both points -------------------------------------------------
resid_at(j) = begin
    j === :median ? (for nm in nms; set_param!(th, string(nm), median(cols[nm])); end) : setdraw!(j)
    pred, _ = Nereus.rv_predictions(th, data)
    data.rv .- pred
end
r_med  = resid_at(:median)
r_best = resid_at(jbest)
inst   = data.rv_inst
labels = ["HIRES", "Lick"]

@printf("\nRV residual RMS:  median-theta = %.1f m/s   best draw = %.1f m/s\n",
        sqrt(sum(abs2, r_med) / n_rv), sqrt(sum(abs2, r_best) / n_rv))
for k in 1:2
    m = inst .== k
    @printf("  %-6s  median-theta %8.1f   best %8.1f   (median err %.1f) m/s\n",
            labels[k], sqrt(sum(abs2, r_med[m]) / count(m)),
            sqrt(sum(abs2, r_best[m]) / count(m)), median(data.rv_err[m]))
end

OUTDIR = joinpath(@__DIR__, "plots_HD114762_dr4")
mkpath(OUTDIR)
with_theme(nereus_theme()) do
    fig = Figure(; size = (1150, 800))
    for (row, (r, ttl)) in enumerate([(r_med,  "marginal-median θ  (what the fit script scores)"),
                                      (r_best, "best-likelihood draw")])
        ax = Axis(fig[row, 1];
                  xlabel = row == 2 ? "MJD" : "",
                  ylabel = "RV residual (m/s)",
                  xticklabelsvisible = row == 2)
        hlines!(ax, [0.0]; color = :black, linestyle = :dash)
        for k in 1:2
            m = inst .== k
            errorbars!(ax, data.t_rv[m], r[m], data.rv_err[m];
                       color = INST_COLORS[k], linewidth = 1.5)
            scatter!(ax, data.t_rv[m], r[m]; color = INST_COLORS[k],
                     marker = INST_MARKERS[k], markersize = 12,
                     strokewidth = 0.5, strokecolor = :black,
                     label = row == 1 ? labels[k] : nothing)
        end
        text!(ax, 0.015, 0.96;
              text = @sprintf("%s\nRMS = %.1f m/s", ttl, sqrt(sum(abs2, r) / n_rv)),
              space = :relative, align = (:left, :top), fontsize = 17)
        row == 1 && axislegend(ax; position = :rt, framevisible = true)
    end
    linkxaxes!(contents(fig[1, 1])[1], contents(fig[2, 1])[1])
    out = joinpath(OUTDIR, "median_vs_best_residuals.png")
    save(out, fig)
    println("\nsaved ", out)
end
