using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters
using Graphs #note: Graphs.jl is a reboot of the LightGraphs package which was archived
using Pkg
using Plots
using StatsPlots
using Colors, ColorSchemes
using SimpleWeightedGraphs
using GraphPlot
using Compose #drawing to file
import Cairo, Fontconfig #for PNG from Compose to work
using Statistics
includet("./graphForwardSearch.jl") #include and track changes


struct Point{T}
    x::T
    y::T
end
 
struct Line{T}
    s::Point{T}
    e::Point{T}
end
 
function intersection(l1::Line{T}, l2::Line{T}) where T<:Real
    a1 = l1.e.y - l1.s.y
    b1 = l1.s.x - l1.e.x
    c1 = a1 * l1.s.x + b1 * l1.s.y
 
    a2 = l2.e.y - l2.s.y
    b2 = l2.s.x - l2.e.x
    c2 = a2 * l2.s.x + b2 * l2.s.y
 
    Δ = a1 * b2 - a2 * b1
    # If lines are parallel, intersection point will contain infinite values
    return Point((b2 * c1 - b1 * c2) / Δ, (a1 * c2 - a2 * c1) / Δ)
end


function croos(edge::Line,node_to_node::Line)
    intersection_point = intersection(edge,node_to_node) 
    println(intersection_point)
    if (edge.s.x<=intersection_point.x<=edge.e.x || edge.e.x<=intersection_point.x<=edge.s.x)
        if (edge.s.y<=intersection_point.y<=edge.e.y || edge.e.y<=intersection_point.y<=edge.s.y)
            if (node_to_node.s.x<=intersection_point.x<=node_to_node.e.x || node_to_node.e.x<=intersection_point.x<=node_to_node.s.x)
                if (node_to_node.s.y<=intersection_point.y<=node_to_node.e.y || node_to_node.e.y<=intersection_point.y<=node_to_node.s.y)
                    println("cross")
                    return true
                end
            end
        end
    end
    println("don't cross")
    return false
end


print("GeneratePRM")
function GeneratePRM(threshold::Float64, nodes_number::Integer, obstacles = Array{Float64, 2}(undef, x_size::Integer, y_size::Integer))

    N::Integer=100 
    num_of_obstacles::Integer=15

    Map=zeros(N, N) # occupied index matrix for the NxN 2D vertex.
    nObsX,nObsY = size(obstacles);
    n = 0
    Obs_X = zeros(num_of_obstacles,4);
    Obs_Y = zeros(num_of_obstacles,4);

    while(n<num_of_obstacles) # create the obstacle
        ObsVertx = [rand(1:N-nObsX,1) ; rand(1:N-nObsY,1)]; # generte and x&y cordinate for the left down corner of the obstacle
        if count(!iszero, Map[ObsVertx[1]:(ObsVertx[1]-1+nObsX),ObsVertx[2]:(ObsVertx[2]-1+nObsY)])==0 # check if we can create an obstacle there, using the map.
            Map[ObsVertx[1]:(ObsVertx[1]-1+nObsX),ObsVertx[2]:(ObsVertx[2]-1+nObsY)] = obstacles; # if the cells aren't occupied, occupied them with ones.
            n = n+1;
            Obs_X[n,:] = [ObsVertx[1],ObsVertx[1]-1+nObsX,ObsVertx[1],ObsVertx[1]-1+nObsX]; # save the 4 corner of the x obstacle
            Obs_Y[n,:] = [ObsVertx[2],ObsVertx[2],ObsVertx[2]-1+nObsY,ObsVertx[2]-1+nObsY]; # save the 4 corner of the y obstacle
        end
    end

    n = 0;
    Node=zeros(nodes_number,2)
    while(n<nodes_number) # create the nodes
        NodesVertx = [rand(1:N,1) ; rand(1:N,1)]; # generte and x&y cordinate for the nodes
        if Map[NodesVertx[1],NodesVertx[2]] == 0 # check if the cell is occupied with an obstacle.
            Map[NodesVertx[1],NodesVertx[2]] = 2; # mark the cell   
            n = n + 1;
            Node[n,:] = transpose(NodesVertx);
        end
    end
    

    DistMat = -ones(nodes_number,nodes_number);
    for jj = 1:nodes_number
        refNode = Node[jj,:];
        for zz = (jj+1):nodes_number
            dist = norm(refNode .- Node[zz,:]); # matrix of distance
            if dist<=threshold
                DistMat[jj,zz] = dist; 
                node_to_node = Line(Point{Float64}(refNode[1],refNode[2]), Point{Float64}(Node[zz,1],Node[zz,2]))
                for qq = 1:num_of_obstacles
                    c1 = Line(Point{Float64}(Obs_X[qq,1],Obs_Y[qq,1]), Point{Float64}(Obs_X[qq,2],Obs_Y[qq,2])) # create the 4 egdes of the obstacle qq
                    c1_cross = croos(c1,node_to_node)

                    c2 = Line(Point{Float64}(Obs_X[qq,1],Obs_Y[qq,3]), Point{Float64}(Obs_X[qq,2],Obs_Y[qq,4]))
                    c2_cross = croos(c2,node_to_node)

                    c3 = Line(Point{Float64}(Obs_X[qq,1],Obs_Y[qq,1]), Point{Float64}(Obs_X[qq,1],Obs_Y[qq,3]))
                    c3_cross = croos(c3,node_to_node)

                    c4 = Line(Point{Float64}(Obs_X[qq,2],Obs_Y[qq,1]), Point{Float64}(Obs_X[qq,2],Obs_Y[qq,3]))
                    c4_cross = croos(c4,node_to_node)

                    if (c1_cross || c2_cross || c3_cross || c4_cross)
                         DistMat[jj,zz] = -1;
                    end
                end
            end
        end
    end
    println(DistMat)

    E = []; W= [] 
    for jj=1:nodes_number 
        for zz=jj+1:nodes_number
            if DistMat[jj,zz] != -1
                push!(E,[jj,zz])
                push!(W,DistMat[jj,zz])
            end
        end
    end
    E = reduce(vcat,transpose.(E))
    return Node, E, W, Obs_X, Obs_Y, DistMat
end

V, E, W, Obs_X, Obs_Y, DistMat = GeneratePRM(50.0, 100, ones(15,10)) # threshold [20,50]  number_of_nodes = [100,500] 

nodes_number = length(DistMat[1,:])
println(nodes_number)

# calculate the number of edges - because of that we run over jj=1:nodes_number and zz=jj+1:nodes_number:
number_of_edges = zeros(nodes_number);
for jj=1:nodes_number 
    for zz=jj+1:nodes_number
        if DistMat[jj,zz] != -1
            number_of_edges[jj] = number_of_edges[jj]+1
        end
    end
end
total_of_edges = sum(number_of_edges)
########################################################################

# calculate the average node degree - because of that we run over jj=1:nodes_number and zz=1:nodes_number:
edge_degree = zeros(nodes_number);
for jj=1:nodes_number 
    for zz=1:nodes_number
        if jj==zz
            edge_degree[jj] = edge_degree[jj]
        elseif DistMat[jj,zz] != -1
            edge_degree[jj] = edge_degree[jj]+1
        end
    end
end
average_node_degree = mean(edge_degree)
println(edge_degree)
###########################################################

graph = SimpleWeightedGraph(size(V,1))
for i in 1:length(W)
    add_edge!(graph, E[i,1], E[i,2], W[i])
end
LG = LocatedGraph(graph,V)
#----------------------------------------------------
norma = zeros(length(V[:,1]))
for i in 1:length(V[:,1])
    norma[i] = norm(V[i,:])
end
s = argmin(norma)
τ = argmax(norma)
#do the thing
h(LG::LocatedGraph,i::Int) = norm(LG.locations[i,:]-LG.locations[τ,:])

AS = Astar(LG,s,τ,h)

for pp = 1:15
    x=[Obs_X[pp,1],Obs_X[pp,1],Obs_X[pp,2],Obs_X[pp,2],Obs_X[pp,1]]
    y=[Obs_Y[pp,1],Obs_Y[pp,3],Obs_Y[pp,3],Obs_Y[pp,1],Obs_Y[pp,1]]
    plt = plot!(Shape(x, y),label="", xlabel="X", ylabel="Y",c="red",xlims=[0,100],ylims=[0,100])
end


for jj=1:length(V[:,1])
    for zz=jj+1:length(V[:,1])
        if DistMat[jj,zz] != -1
            x = [V[jj,1],V[zz,1]]
            y = [V[jj,2],V[zz,2]]
            plt = plot!(x,y,label="")
        end
    end
end

plt = plot!(V[:,1], V[:,2], seriestype = :scatter, label="")

Vas = V[AS,:]
println("The number of edges is")
println(total_of_edges)
println("The average node degree is")
println(average_node_degree)
plt = plot!(Vas[:,1],Vas[:,2],label="", linewidth=7, c="black",title = "Treshhold=50 Number of nodes=100\nThe number of edges is:  $total_of_edges\nThe average node degree is:  $average_node_degree", titlefontsize=10)
savefig(plt,"C:/Users/danhazzan/OneDrive - Technion/Desktop/DAN/treshhold=50 number of nodes=100.png")

println("finished")