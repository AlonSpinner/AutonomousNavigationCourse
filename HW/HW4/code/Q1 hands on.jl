using Revise
using Distributions
using Random
using LinearAlgebra
using Parameters
using Graphs

print("GeneratePRM")
function GeneratePRM(threshold::Float64, nodes_number::Integer, obstacles = Array{Float64, 2}(undef, x_size::Integer, y_size::Integer))# obstacles::ones(x_size::Integer, y_size::Integer))# nObsX::int64, nObsY::int64))

    N::Integer=100 
    num_of_obstacles::Integer=15 

    Map::zeros(Int64, N, N) # occupied index matrix for the NxN 2D vertex.
    nObsX,nObsY = size(obstacles);
    n = 0
    Obs_X = zeros(num_of_obstacles,4);
    Obs_Y = zeros(num_of_obstacles,4);

    while(n<num_of_obstacles) # create the obstacle
        ObsVertx = [rand(1:N-nObsX,1) ; rand(1:N-nObsY,1)]; # generte and x&y cordinate for the left down corner of the obstacle
        if nnz(Map(ObsVertx(1):(ObsVertx(1)-1+nObsX),ObsVertx(2):(ObsVertx(2)-1+nObsY))) == 0 # check if we can create an obstacle there, using the map.
            Map[ObsVertx[1]:(ObsVertx[1]-1+nObsX),ObsVertx[2]:(ObsVertx[2]-1+nObsY)] = obstacles; # if the cells aren't occupied, occupied them with ones.
            n = n+1;
            Obs_X[n,:] = [ObsVertx[1]-1+nObsX,ObsVertx[1],ObsVertx[1],ObsVertx[1]-1+nObsX]; # save the 4 corner of the x obstacle
            Obs_Y[n,:] = [ObsVertx[2],ObsVertx[2],ObsVertx[2]-1+nObsY,ObsVertx[2]-1+nObsY]; # save the 4 corner of the y obstacle
        end
    end

    n = 0;
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
        refNode = Node[jj,jj];
        for zz = 1:NumNodes
            dist = norm(refNode - Node(:,:,zz)); # matrix of distance
            if dist<=threshold
                for qq = 1:num_of_obstacles
                    (xi,yi) = polyxpoly([refNode(1),Node(1,1,zz)],[refNode(2),Node(1,2,zz)],[Obs_X(qq,:),Obs_X(qq,1)],[ObsY(qq,:),ObsY(qq,1)]);
                    if isempty(xi) 
                        break
                    end
                end
                if isempty(xi) && isempty(yi)
                    DistMat[jj,zz] = dist;
                end
            end
        end
    end
    
    for pp = 1:NumObs
        p = patch(ObsX(pp,:),ObsY(pp,:))#,'red');
    end
    for ii=1:NumNodes
        for jj=1:NumNodes
            if DistMat(ii,jj)==-1
                plot([Node(1,1,ii),Node(1,1,jj)],[Node(1,2,ii),Node(1,2,jj)],'k');
            end
        end
    end
    for ii = 1:NumNodes
        DistCurr(ii) = norm([Node(1,1,ii) Node(1,2,ii)]);
    end
    # scatter(Node(1,1,:),Node(1,2,:),'bo','LineWidth',2);
    # [~,Xs] = min(DistCurr);
    # [~,Xg] = max(DistCurr);
    # figure(1)
    # plot(Node(1,1,Xs),Node(1,2,Xs),'x','LineWidth',5,'MarkerSize',10)
    # plot(Node(1,1,Xg),Node(1,2,Xg),'x','LineWidth',5,'MarkerSize',10)
    
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
