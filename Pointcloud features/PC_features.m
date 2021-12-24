clear
close all


FLAG_DEBUG = 1;
FLAG_H_SEP = 1;
FLAG_SHOW_3D = 0;
%% geo-fence
pc = pcread('combined_10.pcd');

proj_pln = [0;0;1;30];
Lidar_pos_P = [0;0;0];
if FLAG_H_SEP
    roi = [-inf inf -inf inf -inf 20];
else
    roi = [-inf inf -inf inf -inf inf];
end

indices = findPointsInROI(pc,roi);
pc = select(pc,indices);
tmp_pnts = pc.Location;
% tmp_pnts(:,3) = -tmp_pnts(:,3); 
pc = pointCloud(tmp_pnts);

%% Voxel Downsample
% figure(1)
% subplot(1,2,1)
% pcshow(pc)
% colormap(gca,'turbo')
% 
% subplot(1,2,2)
% gridStep = 0.5;
% ptCloudOut = pcdownsample(pc,'gridAverage',gridStep);
% pcshow(ptCloudOut)
% colormap(gca,'turbo')
% colorbar

%% normals
if FLAG_DEBUG
%     normals = pcnormals(pc);
%     x = pc.Location(1:10:end,1);
%     y = pc.Location(1:10:end,2);
%     z = pc.Location(1:10:end,3);
%     u = normals(1:10:end,1);
%     v = normals(1:10:end,2);
%     w = normals(1:10:end,3);
%     
%     figure(2)
%     pcshow(pc)
%     hold on
%     quiver3(x,y,z,u,v,w);
%     hold off
end
%% K-R or KNN local features
method = 0; %% 0:K-R 1:KNN
num_nn = 10; % in case of KNN, number of neighbors
radius = 0.5; %in case of KR, radius in meters 

tree = KDTreeSearcher(pc.Location);
if method == 0
    kr_indices = rangesearch(tree,pc.Location,radius);% not confirmed working
elseif method == 1
    knn_indices = knnsearch(tree,pc.Location,'K',num_nn);
end

pc_feature=struct;
idx_keep = zeros(pc.Count,1);
cnt = 0;
tic
for i = 1:pc.Count
    
    if(method == 0)
%         [indices,dists] = findNeighborsInRadius(pc,pc.Location(i,:),radius);
%         pc_local = pc.Location(indices,:);
        pc_local = pc.Location(kr_indices{i},:);
    elseif method == 1
        pc_local = pc.Location(knn_indices(i,:)',:);
    end
    
    if(size(pc_local,1))>3
        cnt = cnt+1;
        idx_keep(cnt) = i;
        centroid = mean(pc_local);
        
        diff = (pc_local-centroid);
        pc_feature(cnt).pnts_neighbor = pc_local;
        cov = diff' * diff;
        [V,D] = eig(cov);
        pc_feature(cnt).eig_vars = diag(D);
        pc_feature(cnt).eig_vars = pc_feature(cnt).eig_vars/sum(pc_feature(cnt).eig_vars);
        pc_feature(cnt).eig_vecs = V;
        
        lambda_1 = pc_feature(cnt).eig_vars(3);
        lambda_2 = pc_feature(cnt).eig_vars(2);
        lambda_3 = pc_feature(cnt).eig_vars(1);
        e1 = pc_feature(cnt).eig_vecs(:,3);
        e2 = pc_feature(cnt).eig_vecs(:,2);
        e3 = pc_feature(cnt).eig_vecs(:,1);
        
        
        % Calculate feature descriptors
        pc_feature(cnt).sum = lambda_1 + lambda_2 + lambda_3;
        pc_feature(cnt).omnivarriance = (lambda_1*lambda_2*lambda_3)^(1/3);
        pc_feature(cnt).eigenentropy = -(lambda_1*log(lambda_1)+lambda_2*log(lambda_2)+lambda_3*log(lambda_3));
        pc_feature(cnt).anisotropy = (lambda_1-lambda_3)/lambda_1;
        pc_feature(cnt).planarity = (lambda_2-lambda_3)/lambda_1;
        pc_feature(cnt).linearity = (lambda_1-lambda_2)/lambda_1;
        pc_feature(cnt).curvature = lambda_3/(lambda_1+lambda_2+lambda_3);
        pc_feature(cnt).sphericity = lambda_3/lambda_1;
        pc_feature(cnt).verticality = 1- abs(dot([0,0,1],e3));
        
        pc_feature(cnt).o1a1 = sum(diff*e1);
        pc_feature(cnt).o1a2 = sum(diff*e2);
        pc_feature(cnt).o2a1 = (diff*e1)'*(diff*e1);
        pc_feature(cnt).o2a2 = (diff*e2)'*(diff*e2);
        
        h_max = max(pc_local(:,3));
        h_min = min(pc_local(:,3));
        pc_feature(cnt).h_above = h_max-pc.Location(i,3);
        pc_feature(cnt).h_range = h_max-h_min;
        pc_feature(cnt).h_below = pc.Location(i,3)-h_min;
    end
    
end
toc
%% Remove points with few nearby points
idx_keep = idx_keep(find(idx_keep)~=0);
pc = select(pc,idx_keep);


% figure()
% histogram([pc_feature.verticality])

%%  Try projection of the PC onto one plane
pnts_proj = PC_3dTo2d_traceback(pc.Location,Lidar_pos_P,proj_pln);

%% Visualize local features
if FLAG_DEBUG
%     plot_one_feature_ESS(100,FLAG_SHOW_3D, pc, pnts_proj,"sum", [pc_feature.sum] );
%     plot_one_feature_ESS(101,FLAG_SHOW_3D, pc, pnts_proj,"omnivarriance", [pc_feature.omnivarriance] );
%     plot_one_feature_ESS(102,FLAG_SHOW_3D, pc, pnts_proj,"eigenentropy", [pc_feature.eigenentropy] );
%     plot_one_feature_ESS(103,FLAG_SHOW_3D, pc, pnts_proj,"anisotropy", [pc_feature.anisotropy] );
%     
%     plot_one_feature_ESS(104,FLAG_SHOW_3D, pc, pnts_proj,"planarity", [pc_feature.planarity] );
%     plot_one_feature_ESS(105,FLAG_SHOW_3D, pc, pnts_proj,"linearity", [pc_feature.linearity] );
%     plot_one_feature_ESS(106,FLAG_SHOW_3D, pc, pnts_proj,"curvature", [pc_feature.curvature] );
%     plot_one_feature_ESS(107,FLAG_SHOW_3D, pc, pnts_proj,"sphericity", [pc_feature.sphericity] );
%     plot_one_feature_ESS(108,FLAG_SHOW_3D, pc, pnts_proj,"verticality", [pc_feature.verticality] );
%     
%     plot_one_feature_ESS(109,FLAG_SHOW_3D, pc, pnts_proj,"o1a1", [pc_feature.o1a1] );
%     plot_one_feature_ESS(110,FLAG_SHOW_3D, pc, pnts_proj,"o1a2", [pc_feature.o1a2] );
%     plot_one_feature_ESS(111,FLAG_SHOW_3D, pc, pnts_proj,"o2a1", [pc_feature.o2a1] );
%     plot_one_feature_ESS(112,FLAG_SHOW_3D, pc, pnts_proj,"o2a2", [pc_feature.o2a2] );
% 
%     plot_one_feature_ESS(113,FLAG_SHOW_3D, pc, pnts_proj,"h_{above}", [pc_feature.h_above] );
%     plot_one_feature_ESS(114,FLAG_SHOW_3D, pc, pnts_proj,"h_{range}", [pc_feature.h_range] );
%     plot_one_feature_ESS(115,FLAG_SHOW_3D, pc, pnts_proj,"h_{below}", [pc_feature.h_below] );
%     

end

%% Refine the map 
D2R = pi/180;
thres_v_ang_d = 10; % degrees
thres_v_ang_r = thres_v_ang_d * D2R;
thres_verticality = cos(thres_v_ang_r);

linearity_pc = [pc_feature.linearity];
planarity_pc = [pc_feature.planarity];
verticality_pc = [pc_feature.verticality];
curvature_pc = [pc_feature.curvature];

% find points with large verticality
idx_v_pc = find(verticality_pc>thres_verticality); 
pc_v = select(pc,idx_v_pc);

% find points with small verticality
idx_h_pc = find(verticality_pc< 1 - thres_verticality);
pc_h = select(pc,idx_h_pc);

% Find a proper height for separating deck points and others
h_med = median(pc_h.Location(:,3));
deck_margin = 2;% set a 2 meters margin
h_med_low_bound = h_med - deck_margin; 

% Define a bounding box to remove partial pile points 
box_rm = [-inf inf -inf inf -inf inf]; % initialize the box to remove points inside of the walls
box_rm_margin = 5; % meters
idx_rm_somePile = find( ...
    pc.Location(:,1) > min(pc_v.Location(:,1)) + box_rm_margin & ...
    pc.Location(:,1) < max(pc_v.Location(:,1)) - box_rm_margin & ...
    pc.Location(:,2) > min(pc_v.Location(:,2)) + box_rm_margin & ...
    pc.Location(:,2) < max(pc_v.Location(:,2)) - box_rm_margin & ...
    pc.Location(:,3) < h_med_low_bound);
idx_rm = idx_rm_somePile;

% Get cleaned point cloud and features
idx_kept = setdiff(1:pc.Count, idx_rm);
pc_cln = select(pc,idx_kept);
linearity = linearity_pc(idx_kept);
planarity = planarity_pc(idx_kept);
verticality = verticality_pc(idx_kept);
curvature = curvature_pc(idx_kept);

idx_v = find(verticality>thres_verticality); 
idx_h = find(verticality< 1 - thres_verticality);
idx_pln = find(planarity > 0.86);
idx_edge = find(linearity > 0.5);
idx_curv = find(curvature > 0.1);

pc_v_cln = select(pc_cln,idx_v);
pc_h_high_cln = select(pc_cln, idx_h);
pc_edge = select(pc_cln, intersect(idx_edge,idx_curv));


pc_v_cln_denoise = pcdenoise(pc_v_cln);
pc_h_high_cln_denoise = pcdenoise(pc_h_high_cln);


figure(200)
scatter3(pc_v_cln.Location(:,1),pc_v_cln.Location(:,2),pc_v_cln.Location(:,3),'.r')
hold on 
scatter3(pc_h_high_cln.Location(:,1),pc_h_high_cln.Location(:,2),pc_h_high_cln.Location(:,3),'.b')
% scatter3(pc_edge.Location(:,1),pc_edge.Location(:,2),pc_edge.Location(:,3),'.g')
hold off
axis equal
xlabel('x, m')
ylabel('y, m')
zlabel('z, m')
title('refine map with verticality')
legend({'vertical','hort & planar & high','edge'})

figure(201)
scatter3(pc_v_cln_denoise.Location(:,1),pc_v_cln_denoise.Location(:,2),pc_v_cln_denoise.Location(:,3),'.r')
hold on 
scatter3(pc_h_high_cln_denoise.Location(:,1),pc_h_high_cln_denoise.Location(:,2),pc_h_high_cln_denoise.Location(:,3),'.b')
% scatter3(pc_edge.Location(:,1),pc_edge.Location(:,2),pc_edge.Location(:,3),'.g')
hold off
axis equal
xlabel('x, m')
ylabel('y, m')
zlabel('z, m')
title('refine map with verticality')
legend({'vertical','hort & planar & high','edge'})

% 
% figure(201)
% idx = find(linearity>0.5);
% pc_sel = select(pc,idx);
% pcshow(pc_sel)

