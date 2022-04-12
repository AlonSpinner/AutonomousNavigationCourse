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
    Σw = 𝒫.Σw
    
    # predict
    μp = F * μb  + a
    Σp = F * Σb * F' + Σw
    return MvNormal(μp, Σp)
end 

function UpdateBelief(bp::FullNormal,𝒫::POMDPscenario, z::Array{Float64, 1})::FullNormal
    #bp - belief + predict
    μp, Σp = bp.μ, bp.Σ
    H  = 𝒫.H
    Σv = 𝒫.Σv

    # update
    K = Σp * H' * inv(H*Σp*H'+Σv)
    μb′ = μp + K*(z-H*μp) 
    Σb′ = (I - K*H)*Σp
    return MvNormal(μb′, Σb′)
end

function PropagateUpdateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1}, z::Array{Float64, 1})::FullNormal
    # kalman filter litrature from probobalistic robotics
    bp = PropagateBelief(b,𝒫,a)
    b′ = UpdateBelief(bp,𝒫,z)
    return b′
end    

function SampleMotionModel(𝒫::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    noise = rand(𝒫.rng,MvNormal([0;0],𝒫.Σw))
    x′ = 𝒫.F * x + a + noise
    return x′
end 

function GenerateObservation(𝒫::POMDPscenario, x::Array{Float64, 1}) #GPS
    noise = rand(𝒫.rng,MvNormal([0,0],𝒫.Σv))
    x′ = 𝒫.H * x + noise
    return x′
end   

function GenerateObservationFromBeacons(𝒫::POMDPscenario, x::Array{Float64, 1}; rangeDependentCov::Bool = false)
    distance = minimum([norm(x-b) for b in eachrow(𝒫.beacons)])
    if distance <= 𝒫.d
        if rangeDependentCov
            𝒫.Σv = (max(distance,𝒫.rmin))^2 * 𝒫.Σv₀
        end
        noise = rand(𝒫.rng,MvNormal([0,0],𝒫.Σv))
        z = 𝒫.H * x + noise
        return z #assumes only 1 beacon is in range
    end    
    return nothing    
end

function OrderBeacons(x,y)::Array{Float64, 2}
    X = x' .* ones(3)
    Y  = ones(3)' .* y
    beacons = hcat(X[:],Y[:])
    return beacons
end

function generateSigmaPoints(p::FullNormal; β = 2, α = 1, n = 2)
    #https://en.wikipedia.org/wiki/Unscented_transform
    #https://drive.google.com/file/d/0By_SW19c1BfhSVFzNHc0SjduNzg/view?resourcekey=0-41olC9ht9xE3wQe2zHZ45A
    κ = 3 - n
    λ = α^2 * (n+κ) - n
    M = sqrt(n+λ)*p.Σ

    points = []
    for i=1:n
        push!(points,p.μ + M[:,i])
    end
    for i=n+1:2*n
        push!(points,p.μ - M[:,i-n])
    end
    push!(points,p.μ)

    weights = 0.5/(n+λ) * ones(2*n+1)
    weights[2*n+1] = λ/(n+λ) #overwrite

    return points, weights
end

function GenerateSigmaPointsFromBeacons(𝒫::POMDPscenario, x::MvNormal)
    distance = minimum([norm(x.μ-b) for b in eachrow(𝒫.beacons)])
    if distance <= 𝒫.d
        z = MvNormal(𝒫.H * x.μ ,𝒫.H * x.Σ * 𝒫.H' + 𝒫.Σv)
        zi, wi = generateSigmaPoints(z)
        return (points = zi, weights = wi) #assumes only 1 beacon is in range
    end 
    return nothing    
end

function J_beacons(𝒫::POMDPscenario,bk::FullNormal,A,r, rₜ)
    #bk - belief in step k 
    #A - sequence of actions to be taken [ak,akp1,akp2...]
    #T - timer step
    #r - reward/cost(bk,a)

    if isempty(A)
        return rₜ(bk) #terminal cost is same as regular
    end
    
    J = r(bk,A[1])

    bkp1⁻ = PropagateBelief(bk,𝒫,A[1]) #predict step
    z = GenerateSigmaPointsFromBeacons(𝒫,bkp1⁻)
    if ~isnothing(z) && (size(A)[1] % 10 == 0)
        for (point,weight) in zip(z.points,z.weights)
            #weights ~ probabilities, already normalized
            bkp1 = UpdateBelief(bkp1⁻,𝒫, point)
            J += weight * J_beacons(𝒫,bkp1,A[2:end],r,rₜ)
        end
    else
        bkp1 = bkp1⁻
        J += J_beacons(𝒫,bkp1,A[2:end],r,rₜ)
    end

    return J
end
