# Missing activity-indicator values (a night a pipeline did not report
# H-alpha, a sentinel turned into NaN) are masked per channel. Every likelihood
# that scores the indicators -- IndicatorFloor (:qp and :white) and ActivityGP
# (joint, conditional, indicators-only, semiseparable, instrument-scoped), on
# the generic and the workspace path -- must equal the same likelihood with
# those indicator entries removed, while the RV entries of those epochs still
# count. Before, one NaN made IndicatorFloor and ActivityGP return NaN, and the
# floor is always on in `default_noise_menu`.
using Nereus, Test, Random, LinearAlgebra, Statistics
import ForwardDiff
using JSON3
using Nereus: Theta, Params, Data, InstrumentConfig, PlanetDataSources, NoiseModel,
              IndicatorFloor, ActivityGP, PTWorkspace, NereusTarget,
              indicator_floor_log_likelihood, rv_log_likelihood, set_param!,
              activity_gp_covariance, bounds

# Epochs without H-alpha and without BIS (none at either end of the baseline,
# so the workspace kernel's reference time is the same with or without them).
const _NM_MISS = Dict("halpha" => [4, 11, 12, 23], "bis" => [11, 17], "fwhm" => Int[])
const _NM_N = 30

function _nm_raw(; seed = 20261124)
    rng = MersenneTwister(seed)
    n = _NM_N
    t = sort(2000.0 .+ 200 .* rand(rng, n))
    inst = [isodd(i) ? 1 : 2 for i in 1:n]
    rv = 3 .* randn(rng, n); rv_err = 1 .+ rand(rng, n)
    chans = ("bis", "halpha", "fwhm")
    ind = Dict{String,Vector{Float64}}(c => randn(rng, n) for c in chans)
    err = Dict{String,Vector{Float64}}(c => 0.2 .+ 0.2 .* rand(rng, n) for c in chans)
    return (; t, inst, rv, rv_err, ind, err)
end

# The dataset with the missing entries as NaN (values and errors).
function _nm_nan(r; chans = ("bis", "halpha", "fwhm"), normalize = false)
    ind = Dict{String,Vector{Float64}}(); err = Dict{String,Vector{Float64}}()
    for c in chans
        v = copy(r.ind[c]); e = copy(r.err[c])
        v[_NM_MISS[c]] .= NaN; e[_NM_MISS[c]] .= NaN
        ind[c] = v; err[c] = e
    end
    Data(; t_rv = r.t, rv = r.rv, rv_err = r.rv_err, rv_inst = r.inst,
           indicators = ind, indicator_errs = err, normalize_indicators = normalize)
end

# The complete dataset (every indicator present).
_nm_full(r; chans = ("bis", "halpha", "fwhm")) =
    Data(; t_rv = r.t, rv = r.rv, rv_err = r.rv_err, rv_inst = r.inst,
           indicators = Dict{String,Vector{Float64}}(c => copy(r.ind[c]) for c in chans),
           indicator_errs = Dict{String,Vector{Float64}}(c => copy(r.err[c]) for c in chans),
           normalize_indicators = false)

# One channel with its missing epochs removed altogether (RV and indicator).
function _nm_reduced(r, c)
    keep = setdiff(1:_NM_N, _NM_MISS[c])
    Data(; t_rv = r.t[keep], rv = r.rv[keep], rv_err = r.rv_err[keep],
           rv_inst = r.inst[keep],
           indicators = Dict{String,Vector{Float64}}(c => r.ind[c][keep]),
           indicator_errs = Dict{String,Vector{Float64}}(c => r.err[c][keep]),
           normalize_indicators = false)
end

_nm_params(data, noise) =
    Params(; max_kplanet = 0, planet_modes = PlanetDataSources[],
             instruments = InstrumentConfig(rv = ["A", "B"]), data = data, M_s = 1.0,
             noise_models = NoiseModel[noise...])

_nm_ws(p, d) = PTWorkspace(p, p.config.max_kplanet, length(p.config.noise_models);
                           n_obs = length(d.t_rv))

# A point inside every prior: geometric centre of positive boxes, centre
# otherwise, then the named overrides.
function _nm_theta(p; over = Dict{String,Float64}())
    th = Theta{Float64}(p)
    L = p.layout
    for (nm, ps) in zip(L.unfrozen_names, L.unfrozen_priors)
        lo, hi = bounds(ps)
        set_param!(th, nm, lo > 0 ? sqrt(lo * hi) : (lo + hi) / 2)
    end
    for (k, v) in over
        haskey(L.name_to_idx, k) && set_param!(th, k, v)
    end
    th
end

# Copy every parameter `dst` shares with `src` by name.
function _nm_copy!(dst::Theta, src::Theta)
    for (nm, i) in dst.params.layout.name_to_idx
        j = get(src.params.layout.name_to_idx, nm, 0)
        j == 0 || (dst.values[i] = src.values[j])
    end
    dst
end

_nm_close(a, b; rtol = 1e-10) = isfinite(a) && isfinite(b) &&
    abs(a - b) <= rtol * max(1.0, abs(a))

const _NM_FLOOR_OVER = Dict("ind_floor_period" => 17.3, "ind_floor_lambda_e" => 60.0,
    "ind_floor_lambda_p" => 0.8, "ind_floor_bis_amp" => 1.1, "ind_floor_bis_jit" => 0.3,
    "ind_floor_halpha_amp" => 0.7, "ind_floor_halpha_jit" => 0.2,
    "ind_floor_bis" => 0.5, "ind_floor_halpha" => 0.8,
    "sigma_A" => 0.7, "sigma_B" => 1.3, "gamma_A" => 0.4, "gamma_B" => -0.6)

_nm_floor_alloc(th, d, ws) = @allocated indicator_floor_log_likelihood(th, d, ws)

@testset "IndicatorFloor masks missing indicator entries" begin
    r = _nm_raw()
    chans = ("bis", "halpha")
    d = _nm_nan(r; chans)
    dfull = _nm_full(r; chans)
    @testset "kernel :$kernel" for kernel in (:qp, :white)
        floor = IndicatorFloor(channels = [:bis, :halpha], kernel = kernel)
        p = _nm_params(d, [floor])
        th = _nm_theta(p; over = _NM_FLOOR_OVER)
        ws = _nm_ws(p, d)
        a = indicator_floor_log_likelihood(th, d)
        b = indicator_floor_log_likelihood(th, d, ws)
        # Reference: each channel scored alone on its epochs that have it.
        ref = 0.0
        for c in chans
            dr = _nm_reduced(r, c)
            pr = _nm_params(dr, [IndicatorFloor(channels = [Symbol(c)], kernel = kernel)])
            thr = _nm_copy!(Theta{Float64}(pr), th)
            ref += indicator_floor_log_likelihood(thr, dr)
        end
        @test isfinite(a) && isfinite(b)
        @test _nm_close(a, ref; rtol = 1e-12)
        @test _nm_close(b, ref; rtol = 1e-10)
        # The RV part keeps every epoch: total − floor is the RV likelihood of
        # the complete dataset (no indicator enters it here).
        pf = _nm_params(dfull, [floor])
        thf = _nm_copy!(Theta{Float64}(pf), th)
        wsf = _nm_ws(pf, dfull)
        rv_ref = rv_log_likelihood(thf, dfull) - indicator_floor_log_likelihood(thf, dfull)
        @test _nm_close(rv_log_likelihood(th, d) - a, rv_ref; rtol = 1e-12)
        @test _nm_close(rv_log_likelihood(th, d, ws) - b, rv_ref; rtol = 1e-10)
        # Over prior draws the two paths agree and stay finite.
        rng = MersenneTwister(7)
        L = p.layout
        nfin = 0
        for _ in 1:60
            for (j, idx) in enumerate(L.unfrozen_idx)
                th.values[idx] = Nereus.quantile(L.unfrozen_priors[j], rand(rng))
            end
            x = indicator_floor_log_likelihood(th, d)
            y = indicator_floor_log_likelihood(th, d, ws)
            @test !isnan(x) && !isnan(y)
            @test x === y || _nm_close(x, y; rtol = 1e-6)
            isfinite(x) && (nfin += 1)
        end
        @test nfin > 30
        if kernel === :qp
            # The masked workspace path allocates nothing, as the full one.
            _nm_floor_alloc(th, d, ws)
            @test _nm_floor_alloc(th, d, ws) == 0
            _nm_floor_alloc(thf, dfull, wsf)
            @test _nm_floor_alloc(thf, dfull, wsf) == 0
        end
    end
end

# Dense reference for an ActivityGP: Rajpaul covariance over the points that
# exist (RV rows of the AGP's instruments unless indicators_only, then each
# channel's finite entries), built point by point with activity_gp_covariance.
# Returns (log N(joint), log N(indicator block)).
function _nm_agp_ref(th::Theta, d::Data, agp::ActivityGP)
    p = th.params
    s = Nereus._gp_suffix(agp)
    v(nm) = th.values[p.layout.name_to_idx[nm]]
    P = v("gp_act_period$s"); λe = v("gp_act_lambda_e$s"); λp = v("gp_act_lambda_p$s")
    inv_sdG = 1 / sqrt(1 / λe^2 + π^2 / (P^2 * λp^2))
    preds, vars = Nereus.rv_predictions(th, d)
    inst_ids = isempty(agp.instruments) ? collect(1:length(p.config.instruments.rv_names)) :
               [findfirst(==(n), p.config.instruments.rv_names) for n in agp.instruments]
    rows = findall(i -> d.rv_inst[i] in inst_ids, eachindex(d.rv))
    tt = Float64[]; ch = Symbol[]; a = Float64[]; b = Float64[]; y = Float64[]; s2 = Float64[]
    if !agp.indicators_only
        Vc = v("Vc$s"); Vr = agp.use_derivative ? v("Vr$s") * inv_sdG : 0.0
        for i in rows
            push!(tt, d.t_rv[i]); push!(ch, :rv); push!(a, Vc); push!(b, Vr)
            push!(y, d.rv[i] - preds[i]); push!(s2, vars[i])
        end
    end
    n_R = length(y)
    for c in agp.channels
        c === :rv && continue
        cg, cd = Nereus._ACTIVITY_GP_COEFFS[c]
        ac = v(string(cg, s))
        bc = (cd === nothing || !agp.use_derivative) ? 0.0 : v(string(cd, s)) * inv_sdG
        ji = get(p.layout.name_to_idx, "gp_act_jit_$(c)$s", 0)
        jit2 = ji == 0 ? 0.0 : th.values[ji]^2
        vals = d.indicators[String(c)]; errs = d.indicator_errs[String(c)]
        for i in rows
            isfinite(vals[i]) || continue
            push!(tt, d.t_rv[i]); push!(ch, c); push!(a, ac); push!(b, bc)
            push!(y, vals[i]); push!(s2, errs[i]^2 + jit2)
        end
    end
    Σ = activity_gp_covariance(tt, ch, a, b, 1.0, P, λe, λp) + Diagonal(s2)
    lN(S, z) = (F = cholesky(Symmetric(S)); -(dot(z, F \ z) + logdet(F) + length(z) * log(2π)) / 2)
    I = (n_R + 1):length(y)
    return lN(Σ, y), lN(Σ[I, I], y[I])
end

function _nm_white(th, d, rows)
    preds, vars = Nereus.rv_predictions(th, d)
    sum(-0.5 * ((d.rv[i] - preds[i])^2 / vars[i] + log(2π * vars[i])) for i in rows; init = 0.0)
end

const _NM_AGP_OVER = Dict("gp_act_period" => 13.1, "gp_act_lambda_e" => 45.0,
    "gp_act_lambda_p" => 0.7, "Vc" => 2.1, "Vr" => 0.6, "Bc" => 0.8, "Br" => -0.3,
    "Hc" => -0.5, "Hr" => 0.2, "Fc" => 0.4, "Fr" => 0.1,
    "gp_act_jit_bis" => 0.15, "gp_act_jit_halpha" => 0.25, "gp_act_jit_fwhm" => 0.2,
    "sigma_A" => 0.7, "sigma_B" => 1.3, "gamma_A" => 0.4, "gamma_B" => -0.6)

@testset "ActivityGP masks missing indicator entries" begin
    r = _nm_raw()
    d = _nm_nan(r)
    chans = [:bis, :halpha, :fwhm]
    variants = (
        ("joint, low-rank", ActivityGP(channels = chans)),
        ("joint, two channels", ActivityGP(channels = [:bis, :halpha])),
        ("joint, no derivative", ActivityGP(channels = chans, use_derivative = false)),
        ("conditional (marginalize_indicators)",
            ActivityGP(channels = chans, marginalize_indicators = true)),
        ("indicators_only", ActivityGP(channels = chans, indicators_only = true)),
        ("semiseparable latent (:sho)", ActivityGP(channels = chans, latent_kernel = :sho)),
        ("instrument-scoped", ActivityGP(channels = chans, instruments = ["A"])))
    @testset "$label" for (label, agp) in variants
        p = _nm_params(d, [agp])
        s = Nereus._gp_suffix(agp)
        over = isempty(s) ? _NM_AGP_OVER :
               Dict((startswith(k, "sigma_") || startswith(k, "gamma_") ? k : k * s) => v
                    for (k, v) in _NM_AGP_OVER)
        th = _nm_theta(p; over)
        ws = _nm_ws(p, d)
        preds, vars = Nereus.rv_predictions(th, d)
        joint_ref, ind_ref = _nm_agp_ref(th, d, agp)
        all_rows = collect(1:_NM_N)
        if agp.latent_kernel !== :qp_dense
            # Same semiseparable solver on the points that exist.
            kern = Nereus._ss_latent_kernel(agp.latent_kernel, 1.0, 13.1, 45.0, 0.7)
            P, λe, λp = 13.1, 45.0, 0.7
            inv_sdG = 1 / sqrt(1 / λe^2 + π^2 / (P^2 * λp^2))
            tf = Float64[]; yf = Float64[]; vf = Float64[]; sid = Int[]
            for i in all_rows
                push!(tf, d.t_rv[i]); push!(yf, d.rv[i] - preds[i]); push!(vf, vars[i]); push!(sid, 1)
            end
            for (k, c) in enumerate(chans), i in all_rows
                x = d.indicators[String(c)][i]
                isfinite(x) || continue
                push!(tf, d.t_rv[i]); push!(yf, x); push!(sid, k + 1)
                push!(vf, d.indicator_errs[String(c)][i]^2 +
                          _NM_AGP_OVER["gp_act_jit_$(c)"]^2)
            end
            α = [2.1, 0.8, -0.5, 0.4]; β = [0.6, -0.3, 0.2, 0.1] .* inv_sdG
            expected = Nereus.multiseries_loglike(tf, yf, vf, sid, α, β, kern)
        elseif agp.marginalize_indicators
            expected = joint_ref - ind_ref
        elseif agp.indicators_only
            expected = ind_ref
        else
            expected = joint_ref
        end
        if isempty(agp.instruments)
            g = Nereus._activity_gp_joint_ll(th, d, preds, vars, agp)
            gw = Nereus._activity_gp_joint_ll(th, d, preds, vars, agp, ws)
            @test isfinite(g) && isfinite(gw)
            @test _nm_close(g, expected; rtol = 1e-9)
            @test _nm_close(gw, expected; rtol = 1e-9)
            total = agp.indicators_only ? expected + _nm_white(th, d, all_rows) : expected
        else
            rowsB = findall(==(2), d.rv_inst)
            total = expected + _nm_white(th, d, rowsB)
        end
        @test _nm_close(rv_log_likelihood(th, d), total; rtol = 1e-9)
        @test _nm_close(rv_log_likelihood(th, d, ws), total; rtol = 1e-9)
    end

    @testset "with the always-on :qp floor" begin
        # The default menu's pairing: a joint AGP covers the floor's channels
        # (floor skipped); an indicators_only AGP does not (both score).
        floor = IndicatorFloor(channels = [:bis, :halpha], kernel = :qp)
        for agp in (ActivityGP(channels = [:bis, :halpha]),
                    ActivityGP(channels = [:bis, :halpha], indicators_only = true))
            p = _nm_params(d, [agp, floor])
            th = _nm_theta(p; over = merge(_NM_AGP_OVER, _NM_FLOOR_OVER))
            ws = _nm_ws(p, d)
            a = rv_log_likelihood(th, d); b = rv_log_likelihood(th, d, ws)
            @test isfinite(a) && isfinite(b)
            @test _nm_close(a, b; rtol = 1e-9)
        end
    end

    @testset "derivatives (ForwardDiff) through the masked likelihood" begin
        # Non-singular couplings take the whitened low-rank route; couplings 0
        # take the smooth fallbacks (dense with ≤ 3 channels, sequential with 4).
        for (chs, label) in (([:bis, :halpha], "3 channels"), (chans, "4 channels"))
            agp = ActivityGP(channels = chs)
            p = _nm_params(d, [agp])
            tg = NereusTarget(p, d; unconstrained = false)
            th = _nm_theta(p; over = _NM_AGP_OVER)
            L = p.layout
            x0 = [th.values[i] for i in L.unfrozen_idx]
            cpl = findall(nm -> nm in ("Vc", "Vr", "Bc", "Br", "Hc", "Hr", "Fc", "Fr"),
                          L.unfrozen_names)
            f(z) = Nereus.LogDensityProblems.logdensity(tg, z)
            for scale in (1.0, 0.0)
                x = copy(x0); x[cpl] .*= scale
                fx = f(x)
                @test isfinite(fx)
                gr = ForwardDiff.gradient(f, x)
                @test all(isfinite, gr)
                h = 1e-6
                for j in (first(cpl), findfirst(==("gp_act_period"), L.unfrozen_names))
                    e = zeros(length(x)); e[j] = h
                    fd = (f(x .+ e) - f(x .- e)) / (2h)
                    @test abs(gr[j] - fd) <= 1e-4 * max(1.0, abs(fd))
                end
            end
        end
    end
end

@testset "Data accepts missing indicator errors only where the value is missing" begin
    r = _nm_raw()
    v = copy(r.ind["halpha"]); e = copy(r.err["halpha"])
    v[[3, 8]] .= NaN; e[[3, 8]] .= NaN
    mk(v, e; normalize = true) = Data(; t_rv = r.t, rv = r.rv, rv_err = r.rv_err,
        rv_inst = r.inst, indicators = Dict("halpha" => v),
        indicator_errs = Dict("halpha" => e), normalize_indicators = normalize)
    d = mk(v, e)
    hv = d.indicators["halpha"]; he = d.indicator_errs["halpha"]
    @test isnan(hv[3]) && isnan(hv[8])
    @test all(isfinite, hv[setdiff(1:_NM_N, [3, 8])])
    @test all(>(0), he[setdiff(1:_NM_N, [3, 8])])
    # Per-instrument normalisation over the finite entries only: median 0 and
    # unit RMS per instrument.
    for ins in (1, 2)
        k = filter(i -> r.inst[i] == ins && isfinite(hv[i]), 1:_NM_N)
        @test abs(median(hv[k])) < 1e-12
        @test sqrt(sum(abs2, hv[k]) / length(k)) ≈ 1.0 rtol = 1e-12
    end
    # Any error is accepted at a missing value; a finite value still needs a
    # strictly positive error.
    e2 = copy(e); e2[3] = 0.0; e2[8] = -1.0
    @test mk(v, e2) isa Data
    e3 = copy(e); e3[5] = NaN
    @test_throws ArgumentError mk(v, e3)
    e4 = copy(e); e4[5] = 0.0
    @test_throws ArgumentError mk(v, e4)
end

# ---- run_job: inline JSON `values` with nulls -------------------------
function _nm_rv_values(r; null_in = nothing)
    vals = Dict{String,Any}("bjd" => Any[r.t...], "rv" => Any[r.rv...],
                            "rv_err" => Any[r.rv_err...],
                            "instrument" => Any[(i == 1 ? "A" : "B") for i in r.inst])
    for c in ("bis", "halpha")
        col = Any[r.ind[c]...]; ecol = Any[r.err[c]...]
        for i in _NM_MISS[c]
            col[i] = nothing; ecol[i] = nothing
        end
        vals[c] = col; vals[c * "_err"] = ecol
    end
    null_in === nothing || (vals[null_in][5] = nothing)
    return vals
end

# Through a JSON string, as a job file is read: JSON nulls arrive as `nothing`.
_nm_json(x) = JSON3.read(JSON3.write(x), Dict{String,Any})

@testset "run_job inline values: null indicator entries are missing values" begin
    r = _nm_raw()
    vals = _nm_json(_nm_rv_values(r))
    t, v, σ, inst, inames, ind, ierr, comp = Nereus._parse_rv_values(vals)
    @test inames == ["A", "B"]
    @test all(isfinite, t) && all(isfinite, v) && all(isfinite, σ)
    @test sort(collect(keys(ind))) == ["bis", "halpha"]
    for c in ("bis", "halpha")
        @test findall(isnan, ind[c]) == _NM_MISS[c]
        @test findall(isnan, ierr[c]) == _NM_MISS[c]
        @test ind[c][1] == r.ind[c][1]
    end

    @testset "null in `$col` is refused" for col in ("bjd", "rv", "rv_err")
        bad = _nm_json(_nm_rv_values(r; null_in = col))
        err = try
            Nereus._parse_rv_values(bad); nothing
        catch e
            e
        end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("`$col`", msg) && occursin("null", msg) && occursin("5", msg)
    end

    cfg = _nm_json(Dict{String,Any}(
        "seed" => 4, "output_dir" => mktempdir(),
        "star" => Dict{String,Any}("M_s" => 1.0, "R_s" => 1.0),
        "model" => Dict{String,Any}("max_kplanet" => 0, "planet_modes" => String[]),
        "priors" => Dict{String,Any}(),
        "data" => Dict{String,Any}("rv" => Dict{String,Any}("values" => _nm_rv_values(r))),
        "noise_menu" => Dict{String,Any}("indicators" => ["bis", "halpha"]),
        "sampler" => Dict{String,Any}("name" => "transdim_pt_emcee",
            "kwargs" => Dict{String,Any}("n_temps" => 2, "n_walkers" => 16,
                "n_steps" => 30, "n_burnin" => 10, "show_progress" => false)),
        "output" => Dict{String,Any}("plots" => String[], "ppc" => false,
                                     "detection_limits" => false, "loo" => false,
                                     "show_progress" => false)))

    @testset "the menu's first likelihood is finite" begin
        data, inst_rv, _ = Nereus._build_data(cfg["data"])
        @test any(isnan, data.indicators["halpha"])
        star = (; M_s = 1.0, R_s = nothing)
        params, target, menu = Nereus._build_model(cfg, data, star, inst_rv, String[])
        @test menu !== nothing
        @test any(m -> m isa IndicatorFloor, params.config.noise_models)
        # The data-scaled default priors read the finite entries only.
        for ps in params.layout.unfrozen_priors
            lo, hi = bounds(ps)
            @test !isnan(lo) && !isnan(hi) && lo < hi
        end
        # Every noise model active (floor always on, AGP among the toggles).
        th = _nm_theta(params)
        @test isfinite(rv_log_likelihood(th, data))
        ws = _nm_ws(params, data)
        @test isfinite(rv_log_likelihood(th, data, ws))
        x = [th.values[i] for i in params.layout.unfrozen_idx]
        @test isfinite(Nereus.LogDensityProblems.logdensity(target, x))
    end

    @testset "run_job completes" begin
        summary = Nereus.run_job(cfg)
        @test summary["status"] == "ok"
        ch, _ = Nereus.load_chains(joinpath(cfg["output_dir"], "chains.nc"))
        lpn = filter(in((:lp, :logl, :loglike, :log_likelihood)), names(ch))
        @test !isempty(lpn)
        for nm in lpn
            @test all(isfinite, vec(Array(ch[nm])))
        end
    end
end

@testset "ActivityJitter: a missing indicator adds no activity jitter" begin
    # σ_j = jit_base + jit_act·indicator; where the indicator is missing only
    # the base jitter applies -- the same as an indicator value of 0.
    r = _nm_raw()
    d = _nm_nan(r; chans = ("halpha",))
    v0 = copy(d.indicators["halpha"]); v0[_NM_MISS["halpha"]] .= 0.0
    e0 = copy(d.indicator_errs["halpha"]); e0[_NM_MISS["halpha"]] .= 1.0
    d0 = Data(; t_rv = r.t, rv = r.rv, rv_err = r.rv_err, rv_inst = r.inst,
                indicators = Dict("halpha" => v0), indicator_errs = Dict("halpha" => e0),
                normalize_indicators = false)
    aj = Nereus.ActivityJitter(indicator = "halpha")
    p = _nm_params(d, [aj]); p0 = _nm_params(d0, [aj])
    th = _nm_theta(p; over = Dict("sigma_A" => 0.7, "sigma_B" => 1.3))
    for nm in p.layout.unfrozen_names
        startswith(nm, "jit_act") && set_param!(th, nm, 0.4)
        startswith(nm, "jit_base") && set_param!(th, nm, 0.9)
    end
    th0 = _nm_copy!(Theta{Float64}(p0), th)
    a = rv_log_likelihood(th, d)
    @test isfinite(a)
    @test a == rv_log_likelihood(th0, d0)
    @test rv_log_likelihood(th, d, _nm_ws(p, d)) == rv_log_likelihood(th0, d0, _nm_ws(p0, d0))
end

@testset "AGP prediction, RV overlay and LOO conditional drop missing entries" begin
    # Reference: the missing entries replaced by a value with an enormous
    # error (no information) -- the same answer to ~1/err².
    r = _nm_raw()
    d = _nm_nan(r)
    vi = Dict{String,Vector{Float64}}(); ei = Dict{String,Vector{Float64}}()
    for (c, v) in d.indicators
        vi[c] = map(x -> isfinite(x) ? x : 0.0, v)
        ei[c] = [isfinite(v[k]) ? d.indicator_errs[c][k] : 1e7 for k in eachindex(v)]
    end
    dinf = Data(; t_rv = r.t, rv = r.rv, rv_err = r.rv_err, rv_inst = r.inst,
                  indicators = vi, indicator_errs = ei, normalize_indicators = false)
    chans = [:bis, :halpha, :fwhm]
    agp = ActivityGP(channels = chans, marginalize_indicators = true)
    p = _nm_params(d, [agp]); pinf = _nm_params(dinf, [agp])
    th = _nm_theta(p; over = _NM_AGP_OVER)
    thinf = _nm_copy!(Theta{Float64}(pinf), th)
    tp = collect(range(2000.0, 2200.0; length = 40))

    a = Nereus._activity_gp_rv_mean_at_theta(th, d, tp, agp)
    b = Nereus._activity_gp_rv_mean_at_theta(thinf, dinf, tp, agp)
    @test all(isfinite, a)
    @test maximum(abs.(a .- b)) <= 1e-6 * maximum(abs, b)

    names_ = Symbol.(p.layout.unfrozen_names)
    x = [th.values[i] for i in p.layout.unfrozen_idx]
    arr = repeat(reshape(x, 1, :, 1), 3, 1, 1)
    ch = Nereus.MCMCChains.Chains(arr, names_)
    pr = Nereus.activity_gp_predict(ch, p, d; t_pred = tp, n_draws = 3,
                                    rng = MersenneTwister(1))
    prinf = Nereus.activity_gp_predict(ch, pinf, dinf; t_pred = tp, n_draws = 3,
                                       rng = MersenneTwister(1))
    @test all(isfinite, pr.G_mean)
    @test maximum(abs.(pr.G_mean .- prinf.G_mean)) <= 1e-6 * maximum(abs, prinf.G_mean)

    preds, vars = Nereus.rv_predictions(th, d)
    rc, Cc = Nereus._agp_conditional_gaussian(th, d, preds, vars, agp)
    F = cholesky(Symmetric(Cc))
    lc = -(dot(rc, F \ rc) + logdet(F) + length(rc) * log(2π)) / 2
    joint_ref, ind_ref = _nm_agp_ref(th, d, agp)
    @test _nm_close(lc, joint_ref - ind_ref; rtol = 1e-9)
end
