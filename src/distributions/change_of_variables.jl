# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type VariateTransform <: Function end


struct InvVT{T<:VariateTransform} <: VariateTransform
    orig::T
end


Base.inv(trafo::VariateTransform) = InvVT(trafo)
Base.inv(trafo::InvVT) = trafo.orig



struct ToUnitVol{D<:Union{Distribution,AbstractDensity}} <: VariateTransform
    d::D
end


to_unitvol(d::Union{Distribution,AbstractDensity}) = ToUnitVol(d)
from_unitvol(d::Union{Distribution,AbstractDensity}) = inv(ToUnitVol(d))



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



function(trafo::ToUnitVol{<:Truncated})(x::Real)
    d = trafo.d
    orig_to_unit = to_unit(d)
    lb, ub = minimum(d), maximum(d)
    lb_unit, ub_unit = orig_to_unit(lb), orig_to_unit(ub)
    ifelse(isfinite(lb) && ub > lb && isfinite(ub),
        lefttruncexp_to_unit(x, lb, trafo.d.θ),
        anybounds_to_unit(x, lb, ub)
    end
end





(trafo::ToUnitVol{<:Weibull})(x::Real) = weibull_to_unit(x, trafo.d.α, trafo.d.θ)
(trafo::InvVT{<:ToUnitVol{<:Exponential}})(x::Real) = unit_to_weibull(u, trafo.d.α, trafo.d.θ)

weibull_to_unit(x::Real, alpha::Real, theta::Real) = 1 - exp(- (x / theta)^alpha)  # Exact CDF
unit_to_weibull(x::Real, alpha::Real, theta::Real) = (-log(1 - x))^(1/alpha) * theta  # Exact inverse CDF



double_pmass_beyond_cdf_value(u::Real) = 1 - abs(2*u - 1)

# Fallback implementation:
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


# Fallback implementation:
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
