# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    gradlogof(r)
    gradlogof(f::Function, v::Any)
    gradlogof(r::Any, f::Function, v::Any)

*Experimental feature, not part of stable public API.*

Get the log of the gradient of `r = f(v)`.

Default behavior:

```julia
gradlogd((..., gradlogd = x, ...)) = x
gradlogd((..., gradlogd = x, ...), f, v) = x
````
"""
function gradlogof end
export gradlogof

gradlogof(r::NamedTuple) = r.gradlogd
gradlogof(r::NamedTuple, f::Function, v::Any) = r.gradlogd



"""
    logvalgradof(f::Function, v::Any)

*Experimental feature, not part of stable public API.*

ToDo: Document behavior
````
"""
function logvalgradof end
export logvalgradof
