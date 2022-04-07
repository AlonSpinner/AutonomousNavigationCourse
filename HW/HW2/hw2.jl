using Revise
using Distributions
using Random
using LinearAlgebra
using Plots
using StatsPlots
using Parameters

const STATE_SIZE = 2

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



function PropagateUpdateBelief(b::FullNormal, 𝒫::POMDPscenario, a::Array{Float64, 1}, o::Array{Float64, 1})::FullNormal
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
    μb′ = μp + K*(o-H*μp) 
    Σb′ = (I(STATE_SIZE) - K*H)*Σp
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


function main()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = [1.0 0.0; 0.0 1.0]
    b0 = MvNormal(μ0, Σ0)
    d =1.0 
    rmin = 0.1
    # set beacons locations 
    beacons = [0.0 1.0;
                0.0 2.0]
    𝒫 = POMDPscenario(F= [1.0 0.0; 0.0 1.0],
                      H = [1.0 0.0; 0.0 1.0],
                      Σw = 0.1^2*[1.0 0.0; 0.0 1.0],
                      Σv = 0.01^2*[1.0 0.0; 0.0 1.0], 
                      rng = rng , beacons=beacons, d=d, rmin=rmin) 
    ak = [1.0, 0.0]
    xgt0 = [0.5, 0.2]
    T = 10
    
    #Initalization
    x_gt, x_kalman, x_deadreckoning  = xgt0, b0, b0
    Hist_gt, Hist_obs = [x_gt], []
    Hist_deadreckoning, Hist_kalman  = [b0], [b0]
    for _ in 1:T-1
        #move robot
        x_gt = SampleMotionModel(𝒫, ak, x_gt)

        #generate GPS observation
        z = GenerateObservation(𝒫, x_gt)

        #generate beliefs
        x_deadreckoning = PropagateBelief(x_deadreckoning, 𝒫, ak)
        x_kalman = PropagateUpdateBelief(x_kalman, 𝒫, ak, z)

        #record to history
        push!(Hist_gt,x_gt)
        push!(Hist_obs,z)
        push!(Hist_deadreckoning,x_deadreckoning)
        push!(Hist_kalman,x_kalman)
    end
    
    # # plots 
    fig=scatter([x[1] for x in Hist_gt], [x[2] for x in Hist_gt], label="gt")
    for i in 1:T
        covellipse!(Hist_deadreckoning[i].μ, Hist_deadreckoning[i].Σ, showaxes=true, n_std=1, label="step $i")
    end
    savefig(fig,"dead_reckoning.pdf")

    # tr=scatter([x[1] for x in τ], [x[2] for x in τ], label="gt")
    # for i in 1:T
    #     covellipse!(τb[i].μ, τb[i].Σ, showaxes=true, n_std=1, label="step $i")
    # end
    # savefig(tr,"tr.pdf")
           
    # xgt0 = [-0.5, -0.2]           
    # ak = [0.1, 0.1]           

    # # generate motion trajectory
    # for i in 1:T-1
    #     push!(τ, SampleMotionModel(𝒫, ak, τ[end]))
    # end 

    # # generate observation trajectory
    # τobsbeacons = []
    # for i in 1:T
    #     push!(τobsbeacons, GenerateObservationFromBeacons(𝒫, τ[i]))
    # end  

    # println(τobsbeacons)
    # bplot =  scatter(beacons[:, 1], beacons[:, 2], label="beacons", markershape=:utriangle)
    # savefig(bplot,"beacons.pdf")

    
    # use function det(b.Σ) to calculate determinant of the matrix
end 

main()