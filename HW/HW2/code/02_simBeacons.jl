using Revise
using Distributions
using Random
using Plots
using StatsPlots
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
                        Σv₀ = 0.01^2*I₂, 
                        rng = rng , 
                        beacons=beacons, 
                        d=d, rmin=rmin) 
    ak = [0.1, 0.1]
    xgt0 = [-0.5, -0.2]
    T = 100

    #Initalization
    x_gt, x_deadreckoning  = xgt0, b0
    x_kalman_fixedObsCov, x_kalman_nonFixedObsCov = b0, b0
    Hist_gt, Hist_obs_fixedObsCov, Hist_obs_nonFixedObsCov = [x_gt], [], []
    Hist_deadreckoning, Hist_kalman_fixedObsCov, Hist_kalman_nonFixedObsCov  = [b0], [b0], [b0]
    for _ in 1:T-1
        #move robot
        x_gt = SampleMotionModel(𝒫, ak, x_gt)

        #generate observation
        #if rangeDependentCov is true, than also updated 𝒫.Σv
        z_nonFixedObsCov = GenerateObservationFromBeacons(𝒫, x_gt; rangeDependentCov = true)
        z_fixedObsCov = GenerateObservationFromBeacons(𝒫, x_gt; rangeDependentCov = false)

        #generate beliefs
        x_deadreckoning = PropagateBelief(x_deadreckoning, 𝒫, ak)

        if isnothing(z_nonFixedObsCov)
            x_kalman_nonFixedObsCov = PropagateBelief(x_kalman_nonFixedObsCov, 𝒫, ak)
        else
            x_kalman_nonFixedObsCov = PropagateUpdateBelief(x_kalman_nonFixedObsCov, 𝒫, ak, z_nonFixedObsCov)       
        end

        if isnothing(z_fixedObsCov)
            x_kalman_fixedObsCov = PropagateBelief(x_kalman_fixedObsCov, 𝒫, ak)
        else
            x_kalman_fixedObsCov = PropagateUpdateBelief(x_kalman_fixedObsCov, 𝒫, ak, z_fixedObsCov)
        end

        #record to history
        push!(Hist_gt,x_gt)
        if ~isnothing(z_nonFixedObsCov)
            push!(Hist_obs_nonFixedObsCov,z_nonFixedObsCov)
        end
        if ~isnothing(z_fixedObsCov)
            push!(Hist_obs_fixedObsCov,z_fixedObsCov)
        end
        push!(Hist_deadreckoning,x_deadreckoning)
        push!(Hist_kalman_fixedObsCov,x_kalman_fixedObsCov)
        push!(Hist_kalman_nonFixedObsCov,x_kalman_nonFixedObsCov)
    end

    ##----- plot dead_reckoning
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    covellipse!(Hist_deadreckoning[1].μ, Hist_deadreckoning[1].Σ, n_std=3, label = "est", color = "blue")
    for i in 2:T
        covellipse!(Hist_deadreckoning[i].μ, Hist_deadreckoning[i].Σ, n_std=3, label = "", color = "blue")
    end
    scatter!([x[1] for x in Hist_gt], [x[2] for x in Hist_gt], label="gt")
    scatter!(beacons[:,1], beacons[:,2], label="beacons", markershape = :hexagon)
    savefig(p,"./out/02_simBeacons_dead_reckoning.pdf")

    ##----- plot kalman_filter_nonFixedObsCov
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    covellipse!(Hist_kalman_nonFixedObsCov[1].μ, Hist_kalman_nonFixedObsCov[1].Σ, n_std=3, label = "est", color = "blue")
    for i in 2:T
        covellipse!(Hist_kalman_nonFixedObsCov[i].μ, Hist_kalman_nonFixedObsCov[i].Σ, n_std=3, label = "", color = "blue")
    end
    scatter!([x[1] for x in Hist_gt], [x[2] for x in Hist_gt], label="gt")
    scatter!([x[1] for x in Hist_obs_nonFixedObsCov], [x[2] for x in Hist_obs_nonFixedObsCov], label="measurements", markersize=3)
    scatter!(beacons[:,1], beacons[:,2], label="beacons", markershape = :hexagon)
    savefig(p,"./out/02_simBeacons_kalman_nonFixedCov.pdf")

    ##----- plot kalman_filter_fixedObsCov
    p = plot(; xlabel="x", ylabel="y", aspect_ratio = 1.0,  grid=:true, legend=:outertopright, legendfont=font(5))
    covellipse!(Hist_kalman_fixedObsCov[1].μ, Hist_kalman_fixedObsCov[1].Σ, n_std=3, label = "est", color = "blue")
    for i in 2:T
        covellipse!(Hist_kalman_fixedObsCov[i].μ, Hist_kalman_fixedObsCov[i].Σ, n_std=3, label = "", color = "blue")
    end
    scatter!([x[1] for x in Hist_gt], [x[2] for x in Hist_gt], label="gt")
    scatter!([x[1] for x in Hist_obs_fixedObsCov], [x[2] for x in Hist_obs_fixedObsCov], label="measurements", markersize=3)
    scatter!(beacons[:,1], beacons[:,2], label="beacons", markershape = :hexagon)
    savefig(p,"./out/02_simBeacons_kalman_fixedCov.pdf")

    ##---- plot comparison between methods: forbenius norm
    p = plot(; xlabel="time step", ylabel="forbenius error",  grid=:true, legend=:outertopright, legendfont=font(5))
    error_fixedCov = []
    error_nonFixedCov = []
    for (x_gt, x_fixedCov, x_nonFixedCov) in zip(Hist_gt,Hist_kalman_fixedObsCov,Hist_kalman_nonFixedObsCov)
        push!(error_fixedCov,norm(x_gt-x_fixedCov.μ))
        push!(error_nonFixedCov,norm(x_gt-x_nonFixedCov.μ))
    end
    plot!(error_fixedCov, label="fixed covariance")
    plot!(error_nonFixedCov, label="range dependant covariance")
    savefig(p,"./out/02_simBeacons_errorComparison.pdf")

    ##---- plot comparison between methods: covariance trace
    p = plot(; xlabel="time step", ylabel="posterior covariance trace",  grid=:true, legend=:outertopright, legendfont=font(5))
    trace_fixedCov = []
    trace_nonFixedCov = []
    for (x_fixedCov, x_nonFixedCov) in zip(Hist_kalman_fixedObsCov,Hist_kalman_nonFixedObsCov)
        push!(trace_fixedCov,tr(x_fixedCov.Σ))
        push!(trace_nonFixedCov,tr(x_nonFixedCov.Σ))
    end
    plot!(trace_fixedCov, label="fixed covariance")
    plot!(trace_nonFixedCov, label="range dependant covariance")
    savefig(p,"./out/02_simBeacons_posteriorCovTraceComparison.pdf")

    print("finished\n")
end

main()

