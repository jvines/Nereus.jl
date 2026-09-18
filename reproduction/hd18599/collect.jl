#!/usr/bin/env julia
# Assemble the artifact table: trans-dim occupancy alongside the fixed-config
# evidences, with the estimator each log Z came from and its error.
using JSON3, Printf

const HERE = @__DIR__
# Outputs live in the ignored results/ tree (chains.nc are ~400 MB each);
# this package tracks only inputs, configs, scripts and reference outputs.
const OUT  = joinpath(HERE, "..", "..", "results", "HD18599_artifact")
rd(p) = isfile(p) ? JSON3.read(read(p, String)) : nothing

td = rd(joinpath(OUT, "transdim", "summary.json"))
println("="^74)
println("HD 18599 (TOI-179) -- trans-dimensional noise-model selection")
println("="^74)
if td === nothing
    println("  trans-dim summary.json missing")
else
    ms = get(td, :model_selection, nothing)
    if ms !== nothing && haskey(ms, :noise_models)
        @printf("\n%-12s %10s %12s\n", "model", "occupancy", "BF vs best")
        rows = [(String(k), v) for (k, v) in pairs(ms[:noise_models])]
        sort!(rows; by = r -> -r[2][:occupancy])
        for (k, v) in rows
            @printf("%-12s %10.4f %12.4f\n", k, v[:occupancy], v[:bayes_factor_vs_best])
        end
    end
    tdh = get(td, :fit_health, nothing)
    tdh === nothing || @printf("\ntrans-dim fit health: %s\n", get(tdh, :overall, "?"))
end

println("\n" * "="^74)
println("Fixed-configuration evidences (independent runs, same data)")
println("="^74)
@printf("\n%-10s %14s %10s %-14s %s\n", "config", "log Z", "se", "estimator", "health")
rows = Tuple{String,Float64,Float64,String,String}[]
for tag in ("white", "AD", "AGP", "GPRot", "ErrScale")
    s = rd(joinpath(OUT, "fixed_$(tag)", "summary.json"))
    s === nothing && (@printf("%-10s %14s\n", tag, "(missing)"); continue)
    lz = Float64(get(s, :log_z, NaN))
    ev = get(s, :evidence, nothing)
    est = ev === nothing ? "?" : String(get(ev, :reported, "?"))
    se  = NaN
    if ev !== nothing && haskey(ev, Symbol(est))
        e = ev[Symbol(est)]
        se = e isa Number ? 0.0 : Float64(get(e, :se, NaN))
    end
    health = get(s, :fit_health, nothing)
    h = health === nothing ? "?" : String(get(health, :overall, "?"))
    push!(rows, (tag, lz, se, est, h))
end
sort!(rows; by = r -> -r[2])
best = isempty(rows) ? NaN : rows[1][2]
for (tag, lz, se, est, h) in rows
    @printf("%-10s %14.3f %10.3f %-14s %s\n", tag, lz, se, est, h)
end
if !isempty(rows) && isfinite(best)
    println("\nDelta log Z vs best (", rows[1][1], "):")
    for (tag, lz, _, _, _) in rows[2:end]
        @printf("  %-10s %+10.2f\n", tag, lz - best)
    end
end
println("\nNOTE: compare Delta log Z only between rows whose `estimator` matches.")
