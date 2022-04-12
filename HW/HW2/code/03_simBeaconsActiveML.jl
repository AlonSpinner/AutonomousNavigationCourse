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
    N= 10 #amount of trajectories
    𝒜 = [repeat([0.1,0.1*j/5]',T-1,1) for j in 1:N] #action sequences
   
    #Initalization
    ℬ = [[b0] for _ in 1:N]
    𝒥 = zeros(10)
    cost(a,b) = det(b)
    for (i, 𝒜ᵢ) in enumerate(𝒜)
        for t in 1:T-1
            ak = 𝒜ᵢ[t,:]
            𝒥[i] += cost(ak,ℬ[i][end].Σ)
            
            #motion model
            x⁻ = PropagateBelief(ℬ[i][end], 𝒫, ak)
            
            #if possible: generate observation and update with it 
            distance = minimum([norm(x⁻.μ-b) for b in eachrow(𝒫.beacons)])
            if distance <= 𝒫.d
                z =  x⁻.μ
                x′ = UpdateBelief(x⁻, 𝒫, z)
            else
                x′ = x⁻ #update == predict if no measurement
            end

            #add to belief
            push!(ℬ[i],x′)           
        end
        𝒥[i] += cost(0,ℬ[i][end].Σ) #terminal cost
    end

    ##----- plot trajectories 
    colors = range(HSL(colorant"red"), stop=HSL(colorant"green"), length=N)
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    for i = 1:N
        covellipse!(ℬ[i][1].μ, ℬ[i][1].Σ, n_std=1, label = "τ " * string(i), color = colors[i])
        for t in 2:T
            covellipse!(ℬ[i][t].μ, ℬ[i][t].Σ, n_std=1, label = "", color = colors[i])
        end
    end
    scatter!(beacons[:,1], beacons[:,2], label="beacons", markershape = :hexagon)
    savefig(p,"./out/03_simBeaconsActiveML_planning.pdf")

    ##----- plot J 
    p = bar(1:N,𝒥, fillcolor = colors, label = "", xlabel="τ", ylabel="cost")
    savefig(p,"./out/03_simBeaconsActiveML_cost.pdf")
    
    print(𝒥)
    print("finished\n")
end

main()

