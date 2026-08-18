# Sampler head-to-head on the HARD problem: HD 114762 b joint RV + DR4 epoch
# astrometry (14 params, trend=2, the sin(i) break). Same four samplers as the
# Gaia-4 astrometry-only comparison — but this posterior has the sharp
# e–ω–Mo ridge + a0–sin i degeneracy where samplers differ, unlike the smooth
# astrometry-only case. Reports recovery (P, e, M sin i, i, M_true) + logZ +
# wall-clock.
#
#   env NEREUS_GAIA_DR4_XML=<path>  HD114762_RV=<4-col BJD RV eRV inst>

using Nereus, MCMCChains, Printf
using Statistics: median

const SID = 3937211745905473024
const M_PRI = 0.83
const PLX, PLX_ERR = 25.36, 0.30

# --- data ---
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

build() = build_target(
    M_pri = M_PRI,
    planets = (b = (
        a = LogUniformPrior(0.30, 0.45), M_sec = LogUniformPrior(0.003, 0.5),
        sesinw = UniformPrior(-1.0, 1.0), secosw = UniformPrior(-1.0, 1.0),
        Mo = UniformPrior(0.0, 2π), inc = SinePrior(), Omega = UniformPrior(0.0, 2π),
    ),),
    rv = (HIRES = (data = hires, sigma = LogUniformPrior(0.5, 50.0)),
          Lick  = (data = lick,  sigma = LogUniformPrior(0.5, 50.0))),
    iad = src.iad, plx = NormalPrior(PLX, PLX_ERR), M_s = M_PRI, trend_order = 2,
)

function summ(chains)
    a = vec(Array(chains[:, :a_k1, :])); Ms = vec(Array(chains[:, :M_sec_k1, :]))
    ses = vec(Array(chains[:, :sesinw_k1, :])); sec = vec(Array(chains[:, :secosw_k1, :]))
    inc = vec(Array(chains[:, :inc_k1, :]))
    P = [365.25 * sqrt(a[j]^3 / (M_PRI + Ms[j])) for j in eachindex(a)]
    ideg = rad2deg.(inc); ifold = [x > 90 ? 180 - x : x for x in ideg]
    (P = median(P), e = median(ses .^ 2 .+ sec .^ 2),
     msini = median(Ms .* 1047.57 .* sin.(inc)), i = median(ifold),
     Mtrue = median(Ms))
end

results = Tuple{String, Any, Float64, Float64}[]
function run!(name, f)
    @printf(">>> %-10s ... ", name); flush(stdout)
    t0 = time()
    try
        chains, logZ = f(); s = summ(chains); dt = time() - t0
        # print recovery INLINE so a later sampler hanging can't eat the results
        @printf("done %.1f min | P=%.2f e=%.3f Msini=%.1f i=%.1f Mtrue=%.3f logZ=%.1f\n",
                dt / 60, s.P, s.e, s.msini, s.i, s.Mtrue, logZ)
        push!(results, (name, s, logZ, dt))
    catch err
        dt = time() - t0; @printf("FAILED %.1f min (%s)\n", dt / 60, sprint(showerror, err))
        push!(results, (name, err, NaN, dt))
    end
end

# order: the robust ensemble/HMC samplers first (results captured inline);
# nested LAST + lean, because on this curved RV posterior its slice proposal
# under-mixes and it can run tens of minutes without finishing.
run!("Pigeons", () -> sample_pt(build(); n_rounds = 11, n_chains = 8, seed = 42, show_report = false))
run!("ptemcee", () -> let tgt = build()
        r = sample_ptemcee(tgt, tgt.data; n_temps = 10, n_walkers = 60,
                           n_steps = 2000, n_burnin = 800, seed = 42, show_progress = false)
        (r.chains, r.log_evidence)
    end)
run!("PT-HMC", () -> let (c, lz, _) = sample_pt_hmc(build(); n_temps = 12,
                                                    n_sweeps = 1000, n_warmup = 400, seed = 42)
        (c, lz)
    end)
run!("nested", () -> let tgt = build()
        sample_nested(tgt, tgt.data; n_live = 400, bounds = :multi, proposal = :rslice, seed = 42)
    end)

println("\n", "="^86)
@printf("%-9s %8s %7s %8s %7s %9s %9s %7s\n",
        "sampler", "P_d", "e", "Msini_J", "i_deg", "M_true⊙", "logZ", "min")
@printf("%-9s %8.2f %7.3f %8.1f %7.1f %9s %9s %7s\n",
        "KIEFER19", 83.92, 0.335, 11.0, 7.0, "~0.10", "—", "—")
println("-"^86)
for (name, s, logZ, dt) in results
    if s isa NamedTuple
        @printf("%-9s %8.2f %7.3f %8.1f %7.1f %9.3f %9.1f %7.1f\n",
                name, s.P, s.e, s.msini, s.i, s.Mtrue, logZ, dt / 60)
    else
        @printf("%-9s   FAILED: %s\n", name, sprint(showerror, s))
    end
end
println("="^86)
println("recovery: P≈83.9, M sin i≈11–12 M_J (RV planet), i≲7° ⇒ M_true≈0.1 M_⊙ (a STAR).")
println("logZ: watch whether nested reads LOW here (curved RV ridge) vs the smooth Gaia-4 case.")
