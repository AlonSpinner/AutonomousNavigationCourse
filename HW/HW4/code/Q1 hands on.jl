using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters
using Graphs
#using PlotlyJS
using Pkg
using Plots
#using PyPlot
using StatsPlots
using Colors, ColorSchemes


struct Point{T}
    x::T
    y::T
end
 
struct Line{T}
    s::Point{T}
    e::Point{T}
end
 
# function intersection(l1::Line{T}, l2::Line{T}) where T<:Real
#     a1 = l1.e.y - l1.s.y
#     b1 = l1.s.x - l1.e.x
#     c1 = a1 * l1.s.x + b1 * l1.s.y
 
#     a2 = l2.e.y - l2.s.y
#     b2 = l2.s.x - l2.e.x
#     c2 = a2 * l2.s.x + b2 * l2.s.y
 
#     Δ = a1 * b2 - a2 * b1
#     # If lines are parallel, intersection point will contain infinite values
#     return Point((b2 * c1 - b1 * c2) / Δ, (a1 * c2 - a2 * c1) / Δ)
# end

function intersection(l1::Line{T}, l2::Line{T}) where T<:Real

function croos(edge::Line,node_to_node::Line)
    intersection_point = intersection(edge,node_to_node) 
    println(intersection_point)
    if (edge.s.x<intersection_point.x<edge.e.x || edge.e.x<intersection_point.x<edge.s.x)
        if (edge.s.y<intersection_point.y<edge.e.y || edge.e.y<intersection_point.y<edge.s.y)
            if (node_to_node.s.x<intersection_point.x<node_to_node.e.x || node_to_node.e.x<intersection_point.x<node_to_node.s.x)
                if (node_to_node.s.y<intersection_point.y<node_to_node.e.y || node_to_node.e.y<intersection_point.y<node_to_node.s.y)
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

    plt =plot([Obs_X[1,1],Obs_X[1,2]],[Obs_Y[1,1],Obs_Y[1,2]])
    savefig(plt,"./out/1.pdf")

    plt = plot([Obs_X[1,1],Obs_X[1,2]], [Obs_Y[1,3],Obs_Y[1,4]])
    savefig(plt,"./out/2.pdf")

    plt = plot([Obs_X[1,1],Obs_X[1,1]],[Obs_Y[1,1],Obs_Y[1,3]])
    savefig(plt,"./out/3.pdf")

    plt = plot([Obs_X[1,2],Obs_X[1,2]], [Obs_Y[1,1],Obs_Y[1,3]])
    savefig(plt,"./out/4.pdf")

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

                    # if (croos(c1,node_to_node) || croos(c2,node_to_node) || croos(c3,node_to_node) || croos(c4,node_to_node))
                    #     DistMat[jj,zz] = -1;
                    # end
                end
            end
        end
    end
    println(DistMat)

    for pp = 1:num_of_obstacles
        x=[Obs_X[pp,1],Obs_X[pp,1],Obs_X[pp,2],Obs_X[pp,2],Obs_X[pp,1]]
        y=[Obs_Y[pp,1],Obs_Y[pp,3],Obs_Y[pp,3],Obs_Y[pp,1],Obs_Y[pp,1]]
        plt = plot!(Shape(x, y),label="", xlabel="X", ylabel="Y",c="red",xlims=[0,100],ylims=[0,100])
        savefig(plt,"./out/aaa.pdf")
    end



    for jj=1:nodes_number 
        for zz=jj+1:nodes_number
            if DistMat[jj,zz] != -1
                plt = plot!(([Node[jj,1],Node[zz,1]],[Node[jj,2],Node[zz,2]]),label="")
                savefig(plt,"./out/aaa.pdf")
            end
        end
    end

    plt = plot!(Node[:,1], Node[:,2], seriestype = :scatter, label="")
    savefig(plt,"./out/aaa.pdf")

    # for ii = 1:nodes_number
    #     DistCurr(ii) = norm([Node(1,1,ii) Node(1,2,ii)]);
    # end

    # scatter(Node(1,1,:),Node(1,2,:))#,'bo','LineWidth',2);
    # savefig(a,"./out/aaa.pdf")
    # (~,Xs) = min(DistCurr);
    # (~,Xg) = max(DistCurr);
    # figure(1)
    # plot(Node(1,1,Xs),Node(1,2,Xs))#,'x','LineWidth',5,'MarkerSize',10)
    # plot(Node(1,1,Xg),Node(1,2,Xg))#,'x','LineWidth',5,'MarkerSize',10)
    
    # G = struct
    # G.Map     = Map;
    # G.DistMat = DistMat;
    # G.Node    = Node;
    # G.ObsX    = ObsX;
    # G.ObsY    = ObsY;
    # G.Xs      = Xs;
    # G.Xg      = Xg;
    # return G
end

GeneratePRM(50.0, 10, ones(15,10)) # threshold [20,50]  number_of_nodes = [100,500] 
