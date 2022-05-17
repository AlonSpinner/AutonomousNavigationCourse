using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters

includet("./00_misc.jl")

function PropagateBelief(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64})::FullNormal
    μ, Σ = b.μ, b.Σ
    F  = 𝒫.F
    Σw = 𝒫.Σw
    
    # predict
    μ⁻ = F * μ  + a
    Σ⁻ = F * Σ * F' + Σw
    return MvNormal(μ⁻, Σ⁻)
end 

function UpdateBelief(𝒫::POMDPscenario, b⁻::FullNormal, z::Float64, r::Float64)::FullNormal
    #bp - belief + predict
    μ⁻, Σ⁻ = b⁻.μ, b⁻.Σ
    H  = 𝒫.H
    Σv = 𝒫.Σv₀ * max(r,𝒫.rmin)^2 #update covariance noise to fit measurement

    # update
    K = Σ⁻ * H' * inv(H*Σ⁻*H'+Σv)
    μ⁺ = μ⁻ + K*(z-H*μ⁻) 
    Σ⁺ = (I - K*H)*Σ⁻
    return MvNormal(μ⁺, Σ⁺)
end

function TranistBeliefMDP(𝒫::POMDPscenario, b::FullNormal, a :: Vector{Float64}, obs)
    b⁻ = PropagateBelief(𝒫, b,a)
    if obs !== nothing
        b⁺ = UpdateBelief(𝒫, b⁻, obs.z, obs.r)
        return b⁺
    else
        return b⁻
    end
end

function ObservationModel(𝒫::POMDPscenario, b ::FullNormal)::FullNormal
    r = minimum([norm(b.μ-beacon) for beacon in eachrow(𝒫.beacons)])
    if r <= 𝒫.d
        z = 𝒫.H * b.μ
        Σv = (max(r,𝒫.rmin))^2 * 𝒫.Σv₀
        return MvNormal(z,𝒫.H*b.Σ*𝒫.H' + Σv), r
    end    
    return nothing
end