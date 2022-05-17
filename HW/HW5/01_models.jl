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

function UpdateBelief(𝒫::POMDPscenario, b⁻::FullNormal, z::Vector{Float64}, r::Float64)::FullNormal
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

function TranistBeliefMDP(𝒫::POMDPscenario, b::FullNormal, a :: Vector{Float64}, z::Vector{Float64}, r::Float64)
    b⁻ = PropagateBelief(𝒫, b, a)
    b⁺ = UpdateBelief(𝒫, b⁻, z, r)
    return b⁺
end

function ObservationModel(𝒫::POMDPscenario, b ::FullNormal)
    r = minimum([norm(b.μ-beacon) for beacon in eachrow(𝒫.beacons)])
    if r <= 𝒫.d
        Σv = (max(r,𝒫.rmin))^2 * 𝒫.Σv₀
        
        μ = 𝒫.H * b.μ
        Σ = 𝒫.H*b.Σ*𝒫.H' + Σv
        
        Z = MvNormal(μ,Σ)
        return (Z = Z, r = r)
    end    
    return nothing
end

function SampleMotionModel(𝒫::POMDPscenario, a::Vector{Float64}, x::Vector{Float64})::Vector{Float64}
    noise = rand(𝒫.rng,MvNormal([0;0],𝒫.Σw))
    x′ = 𝒫.F * x + a + noise
    return x′
end 


function GenerateObservation(𝒫::POMDPscenario, x::Vector{Float64})
    r = minimum([norm(x-b) for b in eachrow(𝒫.beacons)])
    if r <= 𝒫.d
        Σv = (max(r,𝒫.rmin))^2 * 𝒫.Σv₀
        noise = rand(𝒫.rng,MvNormal([0,0],Σv))
        z = 𝒫.H * x + noise
        return (z = z, r = r) #assumes only 1 beacon is in range
    end    
    return nothing
end