using Revise
using Graphs #note: Graphs.jl is a reboot of the LightGraphs package which was archived
using SimpleWeightedGraphs
using LinearAlgebra
includet("./graphForwardSearch.jl") #include and track changes

#create graph
#--------------------------------------------------------
#note: Vertices locations dont matter when solving dijkstra
V =  zeros(Float64,5,2)
V[1,:] = [1,1]
V[2,:] = [2,1]
V[3,:] = [3,3]
V[4,:] = [4,5]
V[5,:] = [5,5]

E = zeros(Int,7,2)
E[1,:] = [1,2]
E[2,:] = [2,4]
E[3,:] = [4,1]
E[4,:] = [4,5]
E[5,:] = [2,5]
E[6,:] = [3,5]
E[7,:] = [3,4]

W = zeros(Float64,size(E,1))
for i in 1:length(W)
    W[i] = norm(V[E[i,1],:] - V[E[i,2],:])
end

graph = SimpleWeightedGraph(size(V,1))
for i in 1:length(W)
    add_edge!(graph, E[i,1], E[i,2], W[i])
end

LG = LocatedGraph(graph,V)
#----------------------------------------------------
s = 1
τ = 5

h(LG::LocatedGraph,i::Int) = norm(LG.locations[i,:]-LG.locations[τ,:])

println("my solution:")
println(Astar(LG,s,τ,h))