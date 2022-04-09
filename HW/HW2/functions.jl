module ModelsAndScenario
using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters

export POMDPscenario, PropagateBelief, 
        PropagateUpdateBelief, 
        SampleMotionModel, 
        GenerateObservation,
        GenerateObservationFromBeacons

const STATE_SIZE = 2
const I2 = Matrix{Float64}(I(STATE_SIZE))

@with_kw mutable struct POMDPscenario
    F::Array{Float64, 2}   
    H::Array{Float64, 2}
    Σw::Array{Float64, 2}
    Σv::Array{Float64, 2}
    rng::MersenneTwister
    beacons::Array{Float64, 2}
    d::Float64
    rmin::Float64
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


function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Array{Float64, 1})::Union{NamedTuple, Nothing}
    distances = # calculate distances from x to all beacons
    for (index, distance) in enumerate(distances)
        if distance <= 𝒫.d
            obs = # add your code for creating observation here 
            return (obs=obs, index=index) 
        end    
    end 
    return nothing    
end    

end #module