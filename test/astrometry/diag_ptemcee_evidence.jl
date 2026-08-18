# Why does ptemcee report logZ=-28059 when nested gets -2465 on the same
# (2-Keplerian) HD 114762 posterior? PTemceeResult.log_evidence = TI+ (the
# thermodynamic-integration estimator), which is fragile when ⟨logL⟩_β has a
# sharp phase transition / railed-jitter tail. The report ALSO carries SS+
# (stepping-stone) and H+ (hybrid), which are more robust. On PT-HMC this same
# target gave TI+=-4.2e6 but SS+=-3165 — a 1000× gap. Test: does ptemcee's SS+ /
# H+ land near nested's -2465, at increasing ladder resolution?

using Nereus, MCMCChains, Printf, Statistics

const SID = 3937211745905473024
const M_PRI, PLX, PLX_ERR = 0.83, 25.36, 0.30
const NESTED_LOGZ = -2465.4

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

t_rv = vcat(hires.t, lick.t); rv = vcat(hires.rv, lick.rv)
rv_err = vcat(hires.rv_err, lick.rv_err)
rv_inst = vcat(fill(1, length(hires.t)), fill(2, length(lick.t)))
data = Data(; t_rv = t_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst, iad = src.iad)
instruments = InstrumentConfig(rv = ["HIRES", "Lick"], pm = String[])
parametrization = ParametrizationConfig(mass = :M_sec_driven)
priors = Dict{String, PriorSpec}(
    "P_k1" => LogUniformPrior(60.0, 110.0), "M_sec_k1" => LogUniformPrior(0.003, 0.5),
    "inc_k1" => SinePrior(), "Omega_k1" => UniformPrior(0.0, 2π),
    "P_k2" => LogUniformPrior(1.1e4, 1.0e7), "M_sec_k2" => LogUniformPrior(1e-3, 0.5),
    "plx" => NormalPrior(PLX, PLX_ERR),
)
build() = NereusTarget(
    Params(; max_kplanet = 2, planet_modes = [RVAS, RV_ONLY], instruments = instruments,
           parametrization = parametrization, priors = priors, data = data,
           M_s = M_PRI, trend_order = 0), data; unconstrained = true)

e_of(c) = median(vec(Array(c[:, :sesinw_k1, :])) .^ 2 .+ vec(Array(c[:, :secosw_k1, :])) .^ 2)

@printf("nested (correct mode) logZ = %.1f\n\n", NESTED_LOGZ)
@printf("%-5s %-6s %-7s | %10s %10s %10s %10s | %s\n",
        "nT", "nW", "steps", "TI", "TI+", "SS+", "H+", "e_med(mode)")
println("-"^88)
for (nT, nW, nS, nB) in [(10, 60, 2000, 800), (16, 80, 2500, 1000), (24, 100, 3000, 1200)]
    tgt = build()
    r = sample_ptemcee(tgt, tgt.data; n_temps = nT, n_walkers = nW, n_steps = nS,
                       n_burnin = nB, seed = 42, show_progress = false)
    ev = r.evidence
    @printf("%-5d %-6d %-7d | %10.1f %10.1f %10.1f %10.1f | %.3f\n",
            nT, nW, nS, ev.ti[1], ev.ti_plus[1], ev.ss_plus[1], ev.hybrid[1], e_of(r.chains))
    flush(stdout)
end
println("\n=== Which estimator (SS+/H+) matches nested's −2465, and does more ladder help TI? ===")
