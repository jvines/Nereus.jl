#!/usr/bin/env julia
# Validate the mode-Laplace evidence + auto-switch rule wired into sample_ptemcee.
#   planet model (phase transition): TI is broken → log_evidence must SWITCH to
#     the Laplace value.
#   noise model (well-behaved): TI ≈ Laplace → NO switch, keep tempered.
#   Δ log_evidence (planet − noise) must be the correct, large-positive Bayes
#     factor (~+28500 for HD 114762 b vs noise-only).
using Nereus, MCMCChains, Printf, Statistics

const SID = 3937211745905473024
const M_PRI, PLX, PLX_ERR = 0.83, 25.36, 0.30
rvfile = get(ENV, "HD114762_RV", ""); isfile(rvfile) || error("set HD114762_RV")
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
t_rv = vcat(hires.t, lick.t); rv = vcat(hires.rv, lick.rv); rv_err = vcat(hires.rv_err, lick.rv_err)
rv_inst = vcat(fill(1, length(hires.t)), fill(2, length(lick.t)))
data = Data(; t_rv = t_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst, iad = src.iad)
instruments = InstrumentConfig(rv = ["HIRES", "Lick"], pm = String[])
pz = ParametrizationConfig(mass = :M_sec_driven)
planet_tgt() = NereusTarget(Params(; max_kplanet = 1, planet_modes = [RVAS],
    instruments = instruments, parametrization = pz, data = data, M_s = M_PRI, trend_order = 1,
    priors = Dict{String, PriorSpec}("P_k1" => LogUniformPrior(60.0, 110.0),
        "M_sec_k1" => LogUniformPrior(0.003, 0.5), "inc_k1" => SinePrior(),
        "Omega_k1" => UniformPrior(0.0, 2π), "plx" => NormalPrior(PLX, PLX_ERR))), data)
noise_tgt() = NereusTarget(Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
    instruments = instruments, parametrization = pz, data = data, M_s = M_PRI, trend_order = 1,
    priors = Dict{String, PriorSpec}()), data)

runit(mk) = sample_ptemcee(mk(), data; n_temps = 24, n_walkers = 100, n_steps = 3000,
                           n_burnin = 1000, seed = 42, show_progress = false)

ok = true; chk(nm, c) = (global ok; ok &= c; @printf("  [%s] %s\n", c ? "PASS" : "FAIL", nm))

println("=== planet model (phase transition) ===")
rp = runit(planet_tgt)
@printf("TI+=%.1f  SS+=%.1f  Laplace=%.1f  log_evidence(used)=%.1f\n",
        rp.evidence.ti_plus[1], rp.evidence.ss_plus[1], rp.log_evidence_laplace, rp.log_evidence)
chk("Laplace ≈ -2350 (stable)", abs(rp.log_evidence_laplace - (-2350)) < 120)
chk("TI+ is far from Laplace (broken)", abs(rp.evidence.ti_plus[1] - rp.log_evidence_laplace) > 1000)
chk("log_evidence SWITCHED to Laplace", rp.log_evidence == rp.log_evidence_laplace)

println("\n=== noise-only model (well-behaved) ===")
rn = runit(noise_tgt)
@printf("TI+=%.1f  SS+=%.1f  Laplace=%.1f  log_evidence(used)=%.1f\n",
        rn.evidence.ti_plus[1], rn.evidence.ss_plus[1], rn.log_evidence_laplace, rn.log_evidence)
chk("TI+ ≈ Laplace (agree)", abs(rn.evidence.ti_plus[1] - rn.log_evidence_laplace) < 50)
chk("log_evidence did NOT switch (kept tempered)", rn.log_evidence != rn.log_evidence_laplace ||
                                                    isfinite(rn.evidence.hybrid[1]) == false)

println("\n=== Bayes factor from log_evidence (auto-selected) ===")
Δ = rp.log_evidence - rn.log_evidence
@printf("Δ log_evidence = %.1f   (true ≈ +28500; strong detection of planet b)\n", Δ)
chk("Δ is large & positive (25000–32000)", 25000 < Δ < 32000)

println(ok ? "\n✅ PTEMCEE MODE-LAPLACE EVIDENCE + SWITCH RULE VALIDATED" :
             "\n❌ FAILURES ABOVE")
