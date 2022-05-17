using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters

const STATE_SIZE = 2
const I₂ = Matrix{Float64}(I(STATE_SIZE))

@with_kw mutable struct POMDPscenario
    rng :: MersenneTwister
    F :: Matrix{Float}  
    H :: Matrix{Float}
    Σw :: Matrix{Float}
    Σv₀ :: Matrix{Float} = I₂
    beacons :: Matrix{Float} = I₂
    d :: Float = 0.0
    rmin :: Float = 0.0
    cost :: Function
    costₜ :: Function
    𝒜 :: Vector{Vector{Float}}
end

### FILTERING AND SENSING

function PropagateBelief(𝒫::POMDPscenario, b::FullNormal, a::Vector{Float64})::FullNormal
    μ, Σ = b.μ, b.Σ
    F  = 𝒫.F
    Σw = 𝒫.Σw
    
    # predict
    μ⁻ = F * μ  + a
    Σ⁻ = F * Σ * F' + Σw
    return MvNormal(μ⁻, Σ⁻)
end 

function UpdateBelief(𝒫::POMDPscenario, b⁻::FullNormal, z::Float64, r::float64)::FullNormal
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

function TranistBeliefMDP(𝒫::POMDPscenario, b::FullNormal, a :: Vector{Float64}, obs::(float,float))
    b⁻ = PropagateBelief(𝒫, b,a)
    if obs !== nothing
        b⁺ = UpdateBelief(𝒫, b⁻, obs.z, obs.r)
        return b⁺
    else
        return b⁻
end

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


function ObservationModel(𝒫::POMDPscenario, b ::FullNormal)::FullNormal
    r = minimum([norm(b.μ-beacon) for beacon in eachrow(𝒫.beacons)])
    if r <= 𝒫.d
        z = 𝒫.H * x
        Σv = (max(distance,𝒫.rmin))^2 * 𝒫.Σv₀
        return MvNormal(z,H*b.Σ*H' + Σv), r
    end    
    return nothing
end

### GENERATE PLAN

function Plan(𝒫 :: POMDPscenario, b :: FullNormal, L :: Int)
    #returns action and cost
    if L <= 0
        return (a = nothing, J = 𝒫.costₜ(b))
    end
    
    best  = (a = nothing, J = Inf)
    for a in 𝒫.𝒜
        J = 𝒫.cost(b,a)
        b⁻ = PropagateBelief(b, 𝒫, a)
        
        z, r = ObservationModel(𝒫, b⁻) #z::FullNormal
        if z !== nothing
            zᵢ, wᵢ = generateSigmaPoints(z) 
            for i in len(zᵢ)
                b⁺ = TranistBeliefMDP(b, 𝒫, a, (zᵢ[i], r))
                a⁺, J⁺ = Plan(𝒫, b⁺, L-1)
                J += J⁺*wᵢ[i]
            end
        else
            a⁺, J⁺ = Plan(𝒫, b⁻, L-1)
            J += J⁺
        end
        
        if J < best.J
            best = (a, J)
        end
    end
    return best
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

###--- MISC

function OrderBeacons(x,y)::Array{Float64, 2}
    X = x' .* ones(3)
    Y  = ones(3)' .* y
    beacons = hcat(X[:],Y[:])
    return beacons
end