# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type VariateTransform

*BAT-internal, not part of stable public API.*

Abstract super-type for change-of-variables transformations.

Subtypes (e.g. `SomeTrafo <: VariateTransform`) must implement:

* `(trafo::SomeTrafo)(x)`
* `log_psvf(trafo::SomeTrafo, x)`

For real values and/or real-valued vectors `x`.
"""
abstract type VariateTransform <: Function end


struct InvVT{T<:VariateTransform} <: VariateTransform
    orig::T
end


Base.inv(trafo::VariateTransform) = InvVT(trafo)
Base.inv(trafo::InvVT) = trafo.orig


"""
    log_psvf(trafo::VariateTransform, x::Any)

*BAT-internal, not part of stable public API.*

Logarithm of the phase space volume factor of change-of-variables transform
`trafo` at `x`.

Equivalent to `log(abs(det(jacobian(trafo, x))))`.
"""
function log_psvf end

log_psvf(trafo::VariateTransform, x::Real) =
    _stable_logabsdet(ForwardDiff.derivative(trafo, x))

log_psvf(trafo::VariateTransform, x::AbstractMatrix{<:Real}) =
    _stable_logabsdet(ForwardDiff.jacobian(trafo, x))

_stable_logabsdet(x::T) where {T<:Real} = log(abs(x + eps(T)))
_stable_logabsdet(A::AbstractMatrix{T}) where {T<:Real} = logabsdet(A .+ eps(T))



struct VariateTransformChain{N,T<:NTuple{N,VariateTransform}} <: VariateTransform
    transforms::T
end


import Base.∘
∘(a::VariateTransform, b::VariateTransform) = VariateTransformChain((b, a))
∘(a::VariateTransform, b::VariateTransformChain) = VariateTransformChain((b, a.transforms...))
∘(a::VariateTransformChain, b::VariateTransform) = VariateTransformChain((b.transforms..., a))
∘(a::VariateTransformChain, b::VariateTransformChain) = VariateTransformChain((b.transforms..., a.transforms...))


Base.inv(trafo::VariateTransformChain) = VariateTransformChain(reverse(map(inv, trafo.transforms)))


_eval_var_trafos(x::Any) = x

_eval_var_trafos(x::Any, t::VariateTransform) = t(x)

_eval_var_trafos(x::Any, t::VariateTransform, t2::VariateTransform..., ts::VariateTransform...) =
    _eval_var_trafos(t(x), t2, ts...)

function (trafo::VariateTransformChain)(x::Any)
    _eval_var_trafos(x, trafo.transforms...)
end


"""
    BAT.to_unitvol(d::Distribution)::VariateTransform
    BAT.to_unitvol(d::AbstractDensity)::VariateTransform

*BAT-internal, not part of stable public API.*

Returns a change-of-variables transformation 
"""
function to_unitvol end


struct ToUnitVol{D<:Union{Distribution,AbstractDensity}} <: VariateTransform
    d::D
end


to_unitvol(d::Union{Distribution,AbstractDensity}) = ToUnitVol(d)


function to_unitvol(d::Truncated)
    d = trafo.d
    utrafo = to_unit(d)
    lb, ub = minimum(d), maximum(d)
    lb_unit, ub_unit = utrafo(lb), utrafo(ub)
    utrafo(lb), utrafo(ub)
    trunctrafo = to_unit(Uniform(lb, ub))

    trunctrafo ∘ utrafo(x)
end


double_pmass_beyond_cdf_value(u::Real) = 1 - abs(2*u - 1)

function (trafo::ToUnitVol{<:Distribution{Univariate,Continuous}})(x::Real)
    d = trafo.d
    u = cdf(d, x)
    
    is_relevant = double_pmass_beyond_cdf_value(u) > 10^-10
    x_rec = invlogcdf(d, log(u))
    if is_relevant && !approx(x, x_rec, rtol = 10^-6)
        throw(ArgumentError("Conversion to unit interval not precise enough"))
    end

    u
end

function (trafo::InvVT{<:ToUnitVol{<:Distribution{Univariate,Continuous}}})(u::Real)
    d = trafo.d
    x = invlogcdf(d, log(u))

    is_relevant = double_pmass_beyond_cdf_value(u) > 10^-10
    u_rec = cdf(d, x)
    if is_relevant && !approx(u, u_rec, rtol = 10^-6)
        throw(ArgumentError("Conversion from unit interval not precise enough"))
    end

    x
end






(trafo::ToUnitVol{<:Logistic})(x::Real) = logistic_to_unit(x, trafo.d.μ, trafo.d.θ)
(trafo::InvVT{<:ToUnitVol{<:Logistic}})(u::Real) = unit_to_logistic(u, trafo.d.μ, trafo.d.θ)

logistic_to_unit(x::Real, mu::Real, theta::Real) = inv(exp(-(x - mu) / theta) + one(x))  # Exact CDF
unit_to_logistic(u::Real, mu::Real, theta::Real) = log(u / (one(u) - u)) * theta + mu  # Exact inverse CDF

logistic_to_unit(x::Real) = logistic_to_unit(x, 0, 1)
unit_to_logistic(u::Real) = unit_to_logistic(u, 0, 1)


(trafo::ToUnitVol{<:Uniform})(x::Real) = uniform_to_unit(x, trafo.d.a, trafo.d.b)
(trafo::InvVT{<:ToUnitVol{<:Uniform}})(x::Real) = unit_to_uniform(u, trafo.d.a, trafo.d.b)

uniform_to_unit(x::Real, a::Real, b::Real) = (x - a) / (b - a)  # Exact CDF
unit_to_uniform(u::Real, a::Real, b::Real) = (b - a) * u + a  # Exact inverse CDF


(trafo::ToUnitVol{<:Normal})(x::Real) = approxnormal_to_unit(x, trafo.d.μ, trafo.d.σ)
(trafo::InvVT{<:ToUnitVol{<:Normal}})(x::Real) = normal_to_unit(u, trafo.d.μ, trafo.d.σ)

approxnormal_to_unit(x::Real, mu::Real, sigma::Real) = infinite_to_unit((x - mu) / sigma * pi / 2)
unit_to_approxnormal(u::Real, mu::Real, sigma::Real) = unit_to_infinite(u) * sigma * 2 / π + mu

normal_to_unit(x::Real, mu::Real, sigma::Real) = erfc((mu - x) / sigma * invsqrt2) / 2
unit_to_normal(u::Real, mu::Real, sigma::Real) = mu - erfcinv(2 * u) * sigma / invsqrt2


(trafo::ToUnitVol{<:Exponential})(x::Real) = exp_to_unit(x, 0, trafo.d.θ)
(trafo::InvVT{<:ToUnitVol{<:Exponential}})(x::Real) = unit_to_exp(u, 0, trafo.d.θ)

exp_to_unit(x::Real, theta::Real) = exp(x / -theta)  # Exact CDF
unit_to_exp(u::Real, theta::Real) = -theta * log(u)  # Exact inverse CDF

# lefttruncexp_to_unit(x::Real, x0::Real, theta::Real) = exp((x - x0) / -theta)  # Exact CDF
# unit_to_lefttruncexp(u::Real, x0::Real, theta::Real) = -theta * log(u) + x0  # Exact inverse CDF


(trafo::ToUnitVol{<:Weibull})(x::Real) = weibull_to_unit(x, trafo.d.α, trafo.d.θ)
(trafo::InvVT{<:ToUnitVol{<:Exponential}})(x::Real) = unit_to_weibull(u, trafo.d.α, trafo.d.θ)

weibull_to_unit(x::Real, alpha::Real, theta::Real) = 1 - exp(- (x / theta)^alpha)  # Exact CDF
unit_to_weibull(x::Real, alpha::Real, theta::Real) = (-log(1 - x))^(1/alpha) * theta  # Exact inverse CDF
