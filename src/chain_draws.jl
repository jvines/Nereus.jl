# One parameter's draws out of an MCMCChains.Chains, as a flat vector.

using MCMCChains

"""
    _draws(chains, name) -> Vector

`vec(Array(chains[name]))`: one parameter's draws, iterations fastest, then
chains. `chains[name]` copies the column and `Array` copies it again; this
reads the chain array directly, in one copy, with the same values in the same
order. Anything else (not a `Chains`, a name it lacks, a non-Float64 array)
goes the long way, errors included.
"""
function _draws(chains::MCMCChains.Chains, name::Union{Symbol, AbstractString})
    d = chains.value.data
    if eltype(d) === Float64 && ndims(d) == 3
        j = findfirst(==(Symbol(name)), names(chains))
        j === nothing || return vec(d[:, j, :])
    end
    return vec(Array(chains[name]))
end
_draws(chains, name) = vec(Array(chains[name]))

"""
    _draws_view(chains, name) -> AbstractVector

The same draws as [`_draws`](@ref), in the same order, WITHOUT copying them: a
read-only view into the chain's own array. For a caller that reads a few
draws, a subset (`v[idx]` copies only those), or reduces over the column.

`_draws` copies the whole column, which is what most callers want when they
go on to sort or broadcast over it. A figure that only needs the max-lp row,
or the draws of a pool, paid for every column of a 5.8M-draw chain to read a
fraction of it -- 46 MB per column, per call. Mutating the view writes into
the chain, so a caller that sorts or edits in place takes `_draws` instead.
Anything `_draws` would take the long way gets its copy.
"""
_draws_view(chains, name) = vec(_iter_chain_view(chains, name))

"""
    _iter_chain_view(chains, name) -> AbstractMatrix

One parameter's draws as (iteration, chain), the shape of
`Array(chains[name])`, as a view into the chain (see `_draws_view`). Anything
else gets that copy.
"""
function _iter_chain_view(chains::MCMCChains.Chains, name::Union{Symbol, AbstractString})
    d = chains.value.data
    if eltype(d) === Float64 && ndims(d) == 3
        j = findfirst(==(Symbol(name)), names(chains))
        j === nothing || return view(d, :, j, :)
    end
    return Array(chains[name])
end
_iter_chain_view(chains, name) = Array(chains[name])
