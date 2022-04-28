using Revise
using Graphs #note: Graphs.jl is a reboot of the LightGraphs package which was archived
using SimpleWeightedGraphs
includet("./graphForwardSearch.jl") #include and track changes

#create graph
#--------------------------------------------------------
#note: Vertices locations dont matter when solving dijkstra
V =  zeros(5,2)
V[1,:] = [1,1]
V[2,:] = [2,1]
V[3,:] = [3,3]
V[4,:] = [4,5]
V[5,:] = [5,5]

E = zeros(7,2)
E[1,:] = [1,2]
E[2,:] = [2,4]
E[3,:] = [4,1]
E[4,:] = [4,5]
E[5,:] = [2,5]
E[6,:] = [3,5]
E[7,:] = [3,4]

W = [1.5,2.5,2.6,2.3,2.4,3.0,4.0]

graph = SimpleWeightedGraph(size(V,1))
for i in 1:length(W)
    add_edge!(graph, E[i,1], E[i,2], W[i])
end
#----------------------------------------------------
s = 1
τ = 5

println("my solution:")
println(Dijkstra(graph,s,τ))

println("package solution:")
println(enumerate_paths(dijkstra_shortest_paths(graph, s), τ))