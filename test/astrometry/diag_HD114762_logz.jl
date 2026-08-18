# Diagnose the pathological joint-fit log Z + nested failure on HD 114762.
#
# The 4-sampler comparison gave log Z from −2.6e4 (ptemcee) to −1.1e8 (Pigeons)
# — a 4-order spread means the estimators are computing different things, not
# just "bad fit". This finds the mechanical cause:
#   1. MAP + Laplace evidence (sampler-independent). If Laplace logZ is also
#      hugely negative → real posterior pathology (model). If it's sane → the
#      sampler evidence estimators broke on a sharp posterior.
#   2. Railed params (does jitter hit its floor/ceiling?).
#   3. Per-channel logL + per-instrument RV residual RMS vs errors+jitter at the
#      MAP (not the median → no median-vector artifact).
#   4. RV residual structure vs time → is the wide companion HD 114762 B
#      unmodelled (quadratic trend insufficient → needs a 2nd Keplerian)?

using Nereus, Printf, Statistics, LinearAlgebra

const SID = 3937211745905473024
const M_PRI, PLX, PLX_ERR = 0.83, 25.36, 0.30

rvfile = get(ENV, "HD114762_RV", "")
tb = Dict{String, Vector{Float64}}()
for line in eachline(rvfile)
    s = strip(line); (isempty(s) || startswith(s, "#")) && continue
    t, rv, er, ins = split(s)
    append!(get!(() -> Float64[], tb, ins),
            (parse(Float64, t) - 2_400_000.5, parse(Float64, rv), parse(Float64, er)))
end
mkrv(k) = let v = reshape(tb[k], 3, :); (t = v[1, :], rv = v[2, :], rv_err = v[3, :]) end
hires, lick = mkrv("j"), mkrv("lick")
xml = get(ENV, "NEREUS_GAIA_DR4_XML", ""); (isempty(xml) || !isfile(xml)) && (xml = fetch_gaia_dr4_prerelease())
src = read_gaia_epoch_votable(xml, SID)

function build(; trend = 2, jit_lo = 0.5)
    build_target(
        M_pri = M_PRI,
        planets = (b = (
            a = LogUniformPrior(0.30, 0.45), M_sec = LogUniformPrior(0.003, 0.5),
            sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
            Mo = UniformPrior(0.0, 2π), inc = SinePrior(), Omega = UniformPrior(0.0, 2π),
        ),),
        rv = (HIRES = (data = hires, sigma = LogUniformPrior(jit_lo, 50.0)),
              Lick  = (data = lick,  sigma = LogUniformPrior(jit_lo, 50.0))),
        iad = src.iad, plx = NormalPrior(PLX, PLX_ERR), M_s = M_PRI, trend_order = trend,
    )
end

function diagnose(tag, target)
    println("\n", "#"^70, "\n# ", tag)
    data = target.data
    mp = sample_map(target; n_starts = 48, seed = 1)
    @printf("MAP: converged=%s  railed=%s  n_basins=%d  dominance=%.2f\n",
            mp.converged, mp.railed, mp.n_basins, mp.dominance)
    @printf("log_posterior(MAP) = %.1f     Laplace logZ = %.1f\n",
            mp.log_posterior, mp.log_evidence_laplace)
    mp.railed && println("  RAILED: ", join(mp.railed_params, ", "))

    th = Theta(target.params)
    for (nm, v) in zip(mp.param_names, mp.x_map); set_param!(th, nm, v); end
    ll_rv  = Nereus.rv_log_likelihood(th, data)
    ll_ast = Nereus.astrom_log_likelihood(th, data)
    n_rv, n_ast = length(data.t_rv), length(src.iad.t)
    @printf("MAP per-channel: RV logL=%.1f (−2logL/N=%.1f)   ASTROM logL=%.1f (−2logL/N=%.1f)\n",
            ll_rv, -2ll_rv / n_rv, ll_ast, -2ll_ast / n_ast)

    # RV residuals at MAP (rv_predictions returns (predictions, variances))
    pred, vars = Nereus.rv_predictions(th, data)
    resid = data.rv .- pred
    for nm in ("sigma_HIRES", "sigma_Lick", "dvdt", "d2vdt2")
        idx = findfirst(==(nm), mp.param_names)
        idx !== nothing && @printf("  %-12s(MAP) = %.3f\n", nm, mp.x_map[idx])
    end
    for ins in sort(unique(data.rv_inst))
        m = data.rv_inst .== ins
        @printf("  inst %d: N=%d  resid RMS=%.1f   median σ_eff(incl jitter)=%.1f   formal err=%.1f  m/s\n",
                ins, count(m), sqrt(mean(resid[m] .^ 2)),
                median(sqrt.(vars[m])), median(data.rv_err[m]))
    end
    return mp
end

# --- run the diagnostic on the current model, and on hypotheses --------------
diagnose("MODEL AS-IS: 1 planet + quadratic trend, jitter floor 0.5", build())
diagnose("HYPOTHESIS: jitter floor raised to 2 m/s (does railing explain it?)", build(jit_lo = 2.0))
diagnose("HYPOTHESIS: linear trend only (is the quadratic hiding a bad B model?)", build(trend = 1))

println("\n", "="^70)
println("READ: if Laplace logZ ≈ sampler logZ → geometry is genuinely pathological (model).")
println("      if a param rails → error model broken. if RV resid RMS ≫ err+jitter →")
println("      structured misfit (HD 114762 B needs a Keplerian, not a polynomial).")
