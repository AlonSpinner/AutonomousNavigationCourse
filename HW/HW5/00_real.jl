using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters

function SampleMotionModel(𝒫::POMDPscenario, a::Vector{Float64}, x::Vector{Float64})::Vector{Float64}
    noise = rand(𝒫.rng,MvNormal([0;0],𝒫.Σw))
    x′ = 𝒫.F * x + a + noise
    return x′
end 


function GenerateObservation(𝒫::POMDPscenario, x::Vector{Float64})
    r = minimum([norm(x-b) for b in eachrow(𝒫.beacons)])
    if r <= 𝒫.d
        𝒫.Σv = (max(distance,𝒫.rmin))^2 * 𝒫.Σv₀
        noise = rand(𝒫.rng,MvNormal([0,0],𝒫.Σv))
        z = 𝒫.H * x + noise
        return (z = z, r = r) #assumes only 1 beacon is in range
    end    
    return nothing
end