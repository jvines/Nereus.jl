# HD 114762 joint RV + DR4 astrometry with the wide outer M-dwarf B modelled as
# a REAL 2nd Keplerian (RV-only, long period) instead of a polynomial trend.
#
# Question (from the evidence investigation): does replacing the pathological
# quadratic trend with a proper long-period companion collapse the multimodal /
# railed posterior into a clean, evidence-trustworthy one — and does nested then
# converge to the CORRECT mode?
#
#   b: RVAS (P≈84 d, astrometric → sin i break). k1.
#   B: RV_ONLY, P ≫ 29-yr baseline (only a partial arc constrained). k2.
#   Single M_sec_driven parametrization (RV-only M_sec = msini; RVAS = true mass).
#
#   env NEREUS_GAIA_DR4_XML, HD114762_RV ; RUN_NESTED=1 to include nested.

using Nereus, MCMCChains, Printf, Statistics

const SID = 3937211745905473024
const M_PRI, PLX, PLX_ERR = 0.83, 25.36, 0.30

# --- data ---
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

# assemble Data: 2 RV instruments (1=HIRES, 2=Lick) + DR4 epoch astrometry
t_rv    = vcat(hires.t, lick.t)
rv      = vcat(hires.rv, lick.rv)
rv_err  = vcat(hires.rv_err, lick.rv_err)
rv_inst = vcat(fill(1, length(hires.t)), fill(2, length(lick.t)))
data = Data(; t_rv = t_rv, rv = rv, rv_err = rv_err, rv_inst = rv_inst, iad = src.iad)

instruments = InstrumentConfig(rv = ["HIRES", "Lick"], pm = String[])
parametrization = ParametrizationConfig(mass = :M_sec_driven)

priors = Dict{String, PriorSpec}(
    # b — RVAS, period bracketed to the 35-yr-established 83.9 d
    "P_k1"     => LogUniformPrior(60.0, 110.0),
    "M_sec_k1" => LogUniformPrior(0.003, 0.5),      # allow the stellar regime
    "inc_k1"   => SinePrior(),
    "Omega_k1" => UniformPrior(0.0, 2π),
    # B — RV_ONLY, P ≫ baseline (partial arc)
    "P_k2"     => LogUniformPrior(1.1e4, 1.0e7),     # ~30 – 27000 yr
    "M_sec_k2" => LogUniformPrior(1e-3, 0.5),        # msini
    # parallax
    "plx"      => NormalPrior(PLX, PLX_ERR),
)

build() = NereusTarget(
    Params(; max_kplanet = 2, planet_modes = [RVAS, RV_ONLY],
           instruments = instruments, parametrization = parametrization,
           priors = priors, data = data, M_s = M_PRI, trend_order = 0),
    data; unconstrained = true)

function summ(c)
    P  = vec(Array(c[:, :P_k1, :]))
    Ms = vec(Array(c[:, :M_sec_k1, :]))
    se = vec(Array(c[:, :sesinw_k1, :])); sc = vec(Array(c[:, :secosw_k1, :]))
    ic = vec(Array(c[:, :inc_k1, :]))
    PB = vec(Array(c[:, :P_k2, :]))
    (P = median(P), e = median(se .^ 2 .+ sc .^ 2), Mtrue = median(Ms),
     i = median([x > 90 ? 180 - x : x for x in rad2deg.(ic)]),
     PB_yr = median(PB) / 365.25)
end

println("### 1) MAP + Laplace evidence ###"); flush(stdout)
mp = sample_map(build(); n_starts = 48, seed = 1)
@printf("converged=%s railed=%s n_basins=%d  logpost=%.1f  LAPLACE logZ=%.2f\n",
        mp.converged, mp.railed, mp.n_basins, mp.log_posterior, mp.log_evidence_laplace)
mp.railed && println("  railed: ", join(mp.railed_params, ", "))
flush(stdout)

println("\n### 2) ptemcee (TI evidence) ###"); flush(stdout)
let tgt = build()
    t0 = time()
    r = sample_ptemcee(tgt, tgt.data; n_temps = 10, n_walkers = 60, n_steps = 2000,
                       n_burnin = 800, seed = 42, show_progress = false)
    s = summ(r.chains)
    @printf("%.1f min TI logZ=%.2f | P_b=%.2f e=%.3f i=%.1f Mtrue=%.3f | P_B≈%.0f yr\n",
            (time() - t0) / 60, r.log_evidence, s.P, s.e, s.i, s.Mtrue, s.PB_yr)
end
flush(stdout)

if get(ENV, "RUN_NESTED", "0") == "1"
    println("\n### 3) nested (multi-ellipsoid) — correct mode + convergence? ###"); flush(stdout)
    let tgt = build()
        t0 = time()
        try
            c, lz = sample_nested(tgt, tgt.data; n_live = 500, bounds = :multi,
                                  proposal = :rslice, n_walks = 10, slices = 10,
                                  dlogz = 0.3, seed = 42)
            s = summ(c)
            @printf("%.1f min logZ=%.2f | P_b=%.2f e=%.3f i=%.1f Mtrue=%.3f (P_b=83.9 = correct?)\n",
                    (time() - t0) / 60, lz, s.P, s.e, s.i, s.Mtrue)
        catch err
            @printf("nested failed after %.1f min: %s\n", (time() - t0) / 60, sprint(showerror, err))
        end
        flush(stdout)
    end
end
println("\n=== Does B-as-Keplerian give: non-railed MAP + finite Laplace + TI≈Laplace + nested@P_b=83.9? ===")
