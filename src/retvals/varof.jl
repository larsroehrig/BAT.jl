# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    BAT.varof(r)

*Experimental feature, not part of stable public API.*

Extracts the variate from a value `r` returned by some function.

By default, `varof` supports `NamedTuple`s like `(v = ..., ...), and returns
`r.v`.

The behavior of `varof` should be specialized for custom return types.
"""
function varof end
export varof

varof(r::NamedTuple{(:v)}) = r.v
varof(r::NamedTuple{(:v, :ladj)}) = r.v
