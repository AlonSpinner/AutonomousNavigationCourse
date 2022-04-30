#https://juliagraphs.org/Graphs.jl/dev/
using Graphs #note: Graphs.jl is a reboot of the LightGraphs package which was archived
using SimpleWeightedGraphs

mutable struct LocatedGraph
    graph::SimpleWeightedGraph
    locations::Matrix{Float64}
end

function Dijkstra(G :: SimpleWeightedGraph, s::Int, τ::Int)::Vector{Int}
    #openSet - the set of vertices that are at the frontier of the tree, and we think we should expand
    #a vertex belongs to a set iff set[j]=true
    openSet = zeros(Bool,ne(G))
    openSet[s]  = true
    #set all labels to inf except where we started
    g = Inf*ones(Float64,ne(G)) 
    g[s] = 0
    #initalize parent array. parent[i] = j
    parent = zeros(Int,ne(G)) 

    while any(openSet)
        #i = vertex in openSet with lowest g[i]
        i = argmin(g .+ Inf.*.!(openSet))  #add inf to every label not in openSet and then argmin

        openSet[i] = false  #remove i from openSet
        #for children j of vertex i:
            #if there is a better way to reach j, update g[j], and add to openSet as it might be worth traveling through
        for j in all_neighbors(G,i) 
            c_ij = G.weights[i,j]
            if g[i]+c_ij < g[j] && g[i]+c_ij < g[τ] #note: this will disqualify backtracking in the tree, so its ok we use all_neighbors
                g[j] = g[i] + c_ij
                parent[j] = i
                if j != τ
                    openSet[j] = true
                end
            end
        end
    end

    #find shortestpath by backtracking from τ to s
    shortestpath = [τ]
    while shortestpath[end] != s
        append!(shortestpath,parent[shortestpath[end]])
    end
    return shortestpath[end:-1:1] #flip
end

function Astar(LG::LocatedGraph, s::Int, τ::Int, h::Function)::Vector{Int}
    #h(LG,i) -> Float64
    G = LG.graph

    #openSet - the set of vertices that are at the frontier of the tree, and we think we should expand
    #a vertex belongs to a set iff set[j]=true
    openSet = zeros(Bool,ne(G))
    openSet[s]  = true
    closedSet = zeros(Bool,ne(G))
    #set all labels to inf except where we started
    g = Inf*ones(Float64,ne(G)) 
    g[s] = 0

    f = Inf*ones(Float64,ne(G)) 
    f[s] = h(LG,s)
    #initalize parent array. parent[i] = j
    parent = zeros(Int,ne(G)) 

    while closedSet[τ] == false
        i = argmin(f .+ Inf.*.!(openSet))  #add inf to every label not in openSet and then argmin
            
        openSet[i] = false 
        closedSet[i] = true
        
        for j in all_neighbors(G,i)
            if closedSet[j] == true
                continue
            end
            c_ij = G.weights[i,j]
            if g[i]+c_ij < g[j]
                g[j] = g[i] + c_ij
                f[j] = g[j] + h(LG,j)
                parent[j] = i
                openSet[j] = true
            end
        end
    end

    #find shortestpath by backtracking from τ to s
    shortestpath = [τ]
    while shortestpath[end] != s
        append!(shortestpath,parent[shortestpath[end]])
    end
    return shortestpath[end:-1:1] #flip
end
