# Provenance of a fit, for summary.json: the engine that produced it, the Nereus
# version (and the commit when the package is a git checkout), and every prior
# the model was given -- family and bounds/parameters -- as the user specified
# them. The photo-eccentric reparametrization's surrogate slots (`rho_circ_k*`,
# `eta_k*`) are not the user's priors and are left out; the prior on ρ★ and
# (e, ω) they carry is listed under its own names.

# JSON-friendly number: ±Inf and NaN become `nothing` (JSON null).
_prov_num(x::Real) = isfinite(x) ? Float64(x) : nothing

# A prior's family and hyperparameters, by the distribution behind it.
# `ps` supplies the bounds; the entry always has "family", "lo", "hi".
_prior_family(d::Fixed) = ("fixed", Dict{String, Any}("value" => _prov_num(d.value)))
_prior_family(::Uniform) = ("uniform", Dict{String, Any}())
_prior_family(::LogUniform) = ("loguniform", Dict{String, Any}())
_prior_family(::Sine) = ("sine", Dict{String, Any}())
_prior_family(::WrappedUniform) = ("wrapped_uniform", Dict{String, Any}())
_prior_family(::WrappedSine) = ("wrapped_sine", Dict{String, Any}())
_prior_family(d::Normal) =
    ("normal", Dict{String, Any}("mu" => _prov_num(d.μ), "sigma" => _prov_num(d.σ)))
_prior_family(d::Truncated{<:Normal}) =
    ("truncated_normal", Dict{String, Any}("mu" => _prov_num(d.untruncated.μ),
                                           "sigma" => _prov_num(d.untruncated.σ)))
_prior_family(d::Beta) =
    ("beta", Dict{String, Any}("a" => _prov_num(d.α), "b" => _prov_num(d.β)))
_prior_family(d::Distributions.AffineDistribution{<:Any, <:Any, <:Beta}) =
    ("beta", Dict{String, Any}("a" => _prov_num(d.ρ.α), "b" => _prov_num(d.ρ.β)))
_prior_family(d::ModJeffreys) =
    ("modified_jeffreys", Dict{String, Any}("knee" => _prov_num(d.knee),
                                            "upper" => _prov_num(d.upper)))
function _prior_family(d::PolarFold)
    fam, par = _prior_family(d.base.dist)
    return ("polar_fold", merge(par, Dict{String, Any}("base" => fam)))
end
function _prior_family(d::WrappedPushforward)
    if d.kind == _PUSH_NORMAL
        return ("wrapped_normal", Dict{String, Any}("mu" => _prov_num(d.μ),
                                                    "sigma" => _prov_num(d.σ)))
    end
    return ("wrapped_uniform", Dict{String, Any}())
end
_prior_family(d) = (lowercase(string(nameof(typeof(d)))), Dict{String, Any}())

"""
    prior_provenance(ps::PriorSpec) -> Dict{String,Any}

The prior's family (`"uniform"`, `"loguniform"`, `"normal"`, `"truncated_normal"`,
`"beta"`, `"fixed"`, ...), its hyperparameters and its bounds (`lo`, `hi`; null
where unbounded). A fixed parameter is `{"family": "fixed", "value": v}`.
"""
function prior_provenance(ps::PriorSpec)
    fam, par = _prior_family(ps.dist)
    out = Dict{String, Any}("family" => fam)
    if fam == "fixed"
        out["fixed"] = true
        out["value"] = par["value"]
        return out
    end
    out["fixed"] = false
    out["lo"] = _prov_num(ps.lo)
    out["hi"] = _prov_num(ps.hi)
    merge!(out, par)
    return out
end

# The commit of a git checkout, read from .git without running git (a bundle
# relocated away from its checkout has no .git, and nothing here may throw).
function _git_commit(dir::AbstractString)
    try
        g = joinpath(dir, ".git")
        if isfile(g)                              # a worktree: "gitdir: <path>"
            m = match(r"^gitdir:\s*(.+?)\s*$", readchomp(g))
            m === nothing && return nothing
            g = isabspath(m[1]) ? String(m[1]) : normpath(joinpath(dir, m[1]))
        end
        isdir(g) || return nothing
        head = strip(read(joinpath(g, "HEAD"), String))
        if startswith(head, "ref:")
            ref = strip(head[5:end])
            for base in (g, let c = joinpath(g, "commondir")
                             isfile(c) ? normpath(joinpath(g, strip(read(c, String)))) : g
                         end)
                f = joinpath(base, ref)
                isfile(f) && return String(strip(read(f, String)))
                p = joinpath(base, "packed-refs")
                if isfile(p)
                    for l in eachline(p)
                        endswith(l, " " * ref) && return String(split(l)[1])
                    end
                end
            end
            return nothing
        end
        return occursin(r"^[0-9a-f]{40}$", head) ? String(head) : nothing
    catch
        return nothing
    end
end

function _nereus_version_info()
    ver = try
        v = pkgversion(@__MODULE__)
        v === nothing ? nothing : string(v)
    catch
        nothing
    end
    dir = try pkgdir(@__MODULE__) catch; nothing end
    commit = dir === nothing ? nothing : _git_commit(dir)
    return Dict{String, Any}("nereus_version" => ver, "nereus_commit" => commit)
end

"""
    provenance_block(engine, params) -> Dict{String,Any}

The `provenance` block of summary.json: `engine` (the sampler as selected),
`nereus_version`, `nereus_commit` (null unless Nereus is a git checkout) and
`priors` (parameter name => `prior_provenance`), plus `external_priors` (a list
of `{quantity, per_planet, ...prior}`) when the model has any.
"""
function provenance_block(engine::AbstractString, params)
    cfg = params.config
    priors = Dict{String, Any}()
    for name in sort!(collect(keys(cfg.priors)))
        is_density_reparam_name(name) && continue
        priors[name] = prior_provenance(cfg.priors[name])
    end
    out = Dict{String, Any}("engine" => String(engine))
    merge!(out, _nereus_version_info())
    out["priors"] = priors
    if !isempty(cfg.external_priors)
        out["external_priors"] = Any[
            merge(Dict{String, Any}("quantity" => String(e.quantity),
                                    "per_planet" => e.per_planet),
                  prior_provenance(e.prior)) for e in cfg.external_priors]
    end
    return out
end
