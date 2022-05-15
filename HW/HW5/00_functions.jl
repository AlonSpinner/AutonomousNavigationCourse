using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters

const STATE_SIZE = 2
const I₂ = Matrix{Float64}(I(STATE_SIZE))

@with_kw mutable struct POMDPscenario
    rng::MersenneTwister
    F::Array{Float64, 2}   
    H::Array{Float64, 2}
    Σw::Array{Float64, 2}
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

function UpdateBelief(bp::FullNormal,𝒫::POMDPscenario, z::Float64, r::float64)::FullNormal
    #bp - belief + predict
    μp, Σp = bp.μ, bp.Σ
    H  = 𝒫.H
    Σv = 𝒫.Σv₀ * max(r,𝒫.rmin)^2 #update covariance noise to fit measurement

    # update
    K = Σp * H' * inv(H*Σp*H'+Σv)
    μb′ = μp + K*(z-H*μp) 
    Σb′ = (I - K*H)*Σp
    return MvNormal(μb′, Σb′)
end

function TranistBeliefMDP(b::FullNormal, 𝒫::POMDPscenario, a, z::float64)
    b⁻ = PropagateBelief(b,𝒫,a)
    #if possible: generate observation and update with it 
    r = minimum([norm(b⁻.μ-x) for x in eachrow(𝒫.beacons)])
    if r <= 𝒫.d
        z =  b⁻.μ
        x′ = UpdateBelief(b⁻, 𝒫, z, distance)
    else
        x′ = x⁻ #update == predict if no measurement
    end
end

function SampleMotionModel(𝒫::POMDPscenario, a::Array{Float64, 1}, x::Array{Float64, 1})
    noise = rand(𝒫.rng,MvNormal([0;0],𝒫.Σw))
    x′ = 𝒫.F * x + a + noise
    return x′
end 


function GenerateObservation(𝒫::POMDPscenario, x::Array{Float64, 1})::Float64
    r = minimum([norm(x-b) for b in eachrow(𝒫.beacons)])
    if r <= 𝒫.d
        𝒫.Σv = (max(distance,𝒫.rmin))^2 * 𝒫.Σv₀
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
        z = MvNormal(𝒫.H * x.μ ,𝒫.Σv)
        zi, wi = generateSigmaPoints(z)
        return (points = zi, weights = wi) #assumes only 1 beacon is in range
    end 
    return nothing    
end

function J(𝒫::POMDPscenario,bk::FullNormal,A,r, rₜ; mod = 4)
    #bk - belief in step k 
    #A - sequence of actions to be taken [ak,akp1,akp2...]
    #T - timer step
    #r - reward/cost(bk,a)

    if isempty(A)
        return rₜ(bk) #terminal cost is same as regular
    end
    
    cost = r(bk,A[1])

    bkp1⁻ = PropagateBelief(bk,𝒫,A[1]) #motion model
    z = GenerateSigmaPointsFromBeacons(𝒫,bkp1⁻)
    if ~isnothing(z) && (size(A)[1] % mod == 0)
        for (point,weight) in zip(z.points,z.weights)
            #weights ~ probabilities, already normalized
            bkp1 = UpdateBelief(bkp1⁻,𝒫, point)
            cost += weight * J(𝒫,bkp1,A[2:end],r,rₜ)
        end
    else
        bkp1 = bkp1⁻
        cost += J(𝒫,bkp1,A[2:end],r,rₜ)
    end

    return cost
end
