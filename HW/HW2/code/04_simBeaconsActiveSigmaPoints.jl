using Revise
using Distributions
using Random
using Plots
using StatsPlots
using Colors, ColorSchemes
includet("./00_functions.jl") #include and track changes

function main()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = I₂
    b0 = MvNormal(μ0, Σ0)
    generateSigmaPoints(b0)
    d =1.0 
    rmin = 0.1

    beacons = OrderBeacons(LinRange(0,9,3), LinRange(0,9,3))
    𝒫 = POMDPscenario(F= I₂,
                        H = I₂,
                        Σw = 0.1^2*I₂,
                        Σv = 0.01^2*I₂, 
                        rng = rng , 
                        beacons=beacons, 
                        d=d, rmin=rmin) 

    T = 100
    N = 10 #amount of trajectories
    𝒜 = [repeat([0.1,0.1*j/5]',T-1,1) for j in 1:N] #action sequences
   
    #cost functions
    cost(a,b) = det(b)
    costₜ = cost #terminal

    𝒥 = zeros(10)
    for (i, 𝒜ᵢ) in enumerate(𝒜)
        𝒥[i] = J_beacons(𝒫,b0,𝒜ᵢ,100,cost,costₜ)
    end

    ##----- plot J 
    p = bar(1:N,𝒥, fillcolor = colors, label = "", xlabel="τ", ylabel="cost")
    savefig(p,"./out/04_simBeaconsActiveSigmaPoints_cost.pdf")
end

main()

