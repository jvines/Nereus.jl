# Test the fix for the joint-fit logZ pathology + nested hang on HD 114762.
#
# Diagnosis (diag_HD114762_logz.jl): the quadratic trend's d2vdt2 over a 29-yr
# baseline is a near-unconstrained degenerate direction → railed, mode-less
# posterior → NaN Laplace evidence, 4-order sampler-logZ spread, and nested that
# never satisfies dlogz. Secondary: overriding the default jitter prior with a
# LogUniform floor makes jitter rail. FIX = linear trend + Nereus's DEFAULT
# jitter prior (soft NormalPrior(5,5,0,50)). If the posterior now has a proper
# mode, Laplace / ptemcee-TI / nested should agree AND nested should converge.

using Nereus, MCMCChains, Printf, Statistics

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

# HYPOTHESIS TEST: linear trend + FIXED jitter (removes the railing nuisance).
# If the evidence pathology + nested hang were caused by jitter railing to ~0,
# fixing jitter at a sensible F-star value should give finite Laplace, a TI that
# matches it, and a nested run that CONVERGES.
const JIT = parse(Float64, get(ENV, "JIT", "3.0"))   # fixed jitter, m/s
build() = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a = LogUniformPrior(0.30, 0.45), M_sec = LogUniformPrior(0.003, 0.5),
        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
        Mo = UniformPrior(0.0, 2π), inc = SinePrior(), Omega = UniformPrior(0.0, 2π),
    ),),
    rv = (HIRES = (data = hires, sigma = FixedPrior(JIT)),
          Lick  = (data = lick,  sigma = FixedPrior(JIT))),
    iad = src.iad, plx = NormalPrior(PLX, PLX_ERR), M_s = M_PRI, trend_order = 1,
)

summ(c) = let a = vec(Array(c[:, :a_k1, :])), Ms = vec(Array(c[:, :M_sec_k1, :])),
              se = vec(Array(c[:, :sesinw_k1, :])), sc = vec(Array(c[:, :secosw_k1, :])),
              ic = vec(Array(c[:, :inc_k1, :]))
    (P = median([365.25 * sqrt(a[j]^3 / (M_PRI + Ms[j])) for j in eachindex(a)]),
     e = median(se .^ 2 .+ sc .^ 2), Mtrue = median(Ms),
     i = median([x > 90 ? 180 - x : x for x in rad2deg.(ic)]))
end

println("### 1) MAP + Laplace evidence (sampler-independent) ###"); flush(stdout)
mp = sample_map(build(); n_starts = 48, seed = 1)
@printf("converged=%s railed=%s  logpost=%.1f  LAPLACE logZ=%.2f\n",
        mp.converged, mp.railed, mp.log_posterior, mp.log_evidence_laplace)
mp.railed && println("  railed: ", join(mp.railed_params, ", "))
flush(stdout)

println("\n### 2) ptemcee (TI evidence) ###"); flush(stdout)
let tgt = build()
    t0 = time()
    r = sample_ptemcee(tgt, tgt.data; n_temps = 10, n_walkers = 60, n_steps = 2000,
                       n_burnin = 800, seed = 42, show_progress = false)
    s = summ(r.chains)
    @printf("%.1f min  TI logZ=%.2f | P=%.2f e=%.3f i=%.1f Mtrue=%.3f\n",
            (time() - t0) / 60, r.log_evidence, s.P, s.e, s.i, s.Mtrue)
end
flush(stdout)

if get(ENV, "RUN_NESTED", "0") == "1"
    println("\n### 3) nested — does it CONVERGE now (was hanging >45 min)? ###"); flush(stdout)
    let tgt = build()
        t0 = time()
        try
            c, lz = sample_nested(tgt, tgt.data; n_live = 300, bounds = :multi,
                                  proposal = :rslice, n_walks = 8, slices = 8,
                                  dlogz = 0.5, seed = 42)
            s = summ(c)
            @printf("CONVERGED in %.1f min  logZ=%.2f | P=%.2f e=%.3f i=%.1f Mtrue=%.3f\n",
                    (time() - t0) / 60, lz, s.P, s.e, s.i, s.Mtrue)
        catch err
            @printf("nested failed after %.1f min: %s\n", (time() - t0) / 60, sprint(showerror, err))
        end
        flush(stdout)
    end
end

println("\n=== VERDICT: do Laplace / TI / nested logZ agree, and did nested converge? ===")
