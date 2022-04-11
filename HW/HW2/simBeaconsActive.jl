using Revise
using Distributions
using Random
using Plots
using StatsPlots
using Colors, ColorSchemes
includet("./functions.jl") #include and track changes

function main()
    # definition of the random number generator with seed 
    rng = MersenneTwister(1)
    μ0 = [0.0,0.0]
    Σ0 = I₂
    b0 = MvNormal(μ0, Σ0)
    d =1.0 
    rmin = 0.1

    beacons = OrderBeacons(LinRange(0,9,3), LinRange(0,9,3))
    𝒫 = POMDPscenario(F= I₂,
                        H = I₂,
                        Σw = 0.1^2*I₂,
                        Σv = 0.01^2*I₂, 
                        Σv₀ = 0.01^2*I₂, 
                        rng = rng , 
                        beacons=beacons, 
                        d=d, rmin=rmin) 

    T = 100
    N= 10 #amount of trajectories
    𝒜 = [repeat([0.1,0.1*j/5]',T-1,1) for j in 1:N] #action sequences
   
    #Initalization
    ℬ = [[b0] for _ in 1:N]

    for (i, a) in enumerate(𝒜)
        for t in 1:T-1
            ak = a[t,:]
            
            #generate beliefs
            x_meas = PropagateBelief(ℬ[i][end], 𝒫, ak)
            #generate observation
            z = GenerateObservationFromBeacons(𝒫, x_meas.μ; rangeDependentCov = false)

            if isnothing(z)
                x = PropagateBelief(ℬ[i][end], 𝒫, ak)
            else
                x = PropagateUpdateBelief(ℬ[i][end], 𝒫, ak, z)
            end

            #add to belief
            push!(ℬ[i],x)
        end
    end

    ##----- plot trajectories 
    colors = range(HSL(colorant"red"), stop=HSL(colorant"green"), length=N)
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    for i = 1:N
        covellipse!(ℬ[i][1].μ, ℬ[i][1].Σ, n_std=1, label = "", color = colors[i])
        for t in 2:T
            covellipse!(ℬ[i][t].μ, ℬ[i][t].Σ, n_std=1, label = "", color = colors[i])
        end
    end
    scatter!(beacons[:,1], beacons[:,2], label="beacons", markershape = :hexagon)
    savefig(p,"simBeaconsActive_planning.pdf")

end

main()

