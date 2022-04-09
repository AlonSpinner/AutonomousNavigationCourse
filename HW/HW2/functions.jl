using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters

const STATE_SIZE = 2
const I₂ = Matrix{Float64}(I(STATE_SIZE))

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    H::Array{Float64, 2}
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    rng::MersenneTwister
    #optional: only used in beams case
    Σv₀::Array{Float64, 2} = I₂
    beacons::Array{Float64, 2} = I₂
    d::Float64 = 0.0
    rmin::Float64 = 0.0
end

function PropagateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1})::FullNormal
    
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    
    # predict
    μp = F * μb  + a
    Σp = F * Σb * F' + Σw
    return MvNormal(μp, Σp)
end 

function PropagateUpdateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1}, z::Array{Float64, 1})::FullNormal
    # kalman filter litrature from probobalistic robotics
    μb, Σb = b.μ, b.Σ
    F  = 𝒫.F
    H  = 𝒫.H
    Σw, Σv = 𝒫.Σw, 𝒫.Σv
    
    # kalman predict
    μp = F * μb  + a
    Σp = F * Σb * F' + Σw
    # update
    K = Σp * H' * inv(H*Σp*H'+Σv)
    μb′ = μp + K*(z-H*μp) 
    Σb′ = (I - K*H)*Σp
    return MvNormal(μb′, Σb′)
end    

function SampleMotionModel(𝒫::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    noise = rand(𝒫.rng,MvNormal([0;0],𝒫.Σw))
    x′ = 𝒫.F * x + a + noise
    return x′
end 

function GenerateObservation(𝒫::POMDPscenario, x::Array{Float64, 1})
    noise = rand(𝒫.rng,MvNormal([0,0],𝒫.Σv))
    x′ = 𝒫.H * x + noise
    return x′
end   


function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Array{Float64, 1}; rangeDependentCov::Bool = false)
    distances = [norm(x-b) for b in eachrow(𝒫.beacons)]
    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            if rangeDependentCov
                𝒫.Σv = (max(distance,𝒫.rmin))^2 * 𝒫.Σv₀
            end
            noise = rand(𝒫.rng,MvNormal([0,0],𝒫.Σv))
            z = 𝒫.H * x + noise
            return z #assumes only 1 beacon is in range
        end    
    end 
    return nothing    
end

function OrderBeacons(x,y)::Array{Float64, 2}
    X = x' .* ones(3)
    Y  = ones(3)' .* y
    beacons = hcat(X[:],Y[:])
    return beacons
end

