clear

FLAG_DEBUG = 1;
FLAG_H_SEP = 0;
FLAG_SHOW_3D = 0;
%% geo-fence
pc = pcread('dig_to_process(SHU).pcd');

proj_pln = [0;0;1;30];
Lidar_pos_P = [0;0;0];
if FLAG_H_SEP
    roi = [-15 20 -10 30 -inf 27];
else
    roi = [-15 20 -10 30 -inf inf];
end

indices = findPointsInROI(pc,roi);
pc = select(pc,indices);
tmp_pnts = pc.Location;
tmp_pnts(:,3) = -tmp_pnts(:,3);
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
%% K-R local features
radius = 1; %meter
pc_feature=struct;
idx_keep = zeros(pc.Count,1);
cnt = 0;
for i = 1:pc.Count
    
    [indices,dists] = findNeighborsInRadius(pc,pc.Location(i,:),radius);
    pc_local = pc.Location(indices,:);
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
    end
    
end

%% Remove points with few nearby points
idx_keep = idx_keep(find(idx_keep)~=0);
pc = select(pc,idx_keep);


% figure()
% histogram([pc_feature.verticality])

%%  Try projection of the PC onto one plane
pnts_proj = PC_3dTo2d_traceback(pc.Location,Lidar_pos_P,proj_pln);

%% Visualize local features
if FLAG_DEBUG
%     plot_one_feature(100,FLAG_SHOW_3D, pc, pnts_proj,"sum", [pc_feature.sum] );
%     plot_one_feature(101,FLAG_SHOW_3D, pc, pnts_proj,"omnivarriance", [pc_feature.omnivarriance] );
%     plot_one_feature(102,FLAG_SHOW_3D, pc, pnts_proj,"eigenentropy", [pc_feature.eigenentropy] );
%     plot_one_feature(103,FLAG_SHOW_3D, pc, pnts_proj,"anisotropy", [pc_feature.anisotropy] );
    
    plot_one_feature(104,FLAG_SHOW_3D, pc, pnts_proj,"planarity", [pc_feature.planarity] );
    plot_one_feature(105,FLAG_SHOW_3D, pc, pnts_proj,"linearity", [pc_feature.linearity] );
    plot_one_feature(106,FLAG_SHOW_3D, pc, pnts_proj,"curvature", [pc_feature.curvature] );
    plot_one_feature(107,FLAG_SHOW_3D, pc, pnts_proj,"sphericity", [pc_feature.sphericity] );
    plot_one_feature(108,FLAG_SHOW_3D, pc, pnts_proj,"verticality", [pc_feature.verticality] );
%     
%     plot_one_feature(109,FLAG_SHOW_3D, pc, pnts_proj,"o1a1", [pc_feature.o1a1] );
%     plot_one_feature(110,FLAG_SHOW_3D, pc, pnts_proj,"o1a2", [pc_feature.o1a2] );
%     plot_one_feature(111,FLAG_SHOW_3D, pc, pnts_proj,"o2a1", [pc_feature.o2a1] );
%     plot_one_feature(112,FLAG_SHOW_3D, pc, pnts_proj,"o2a2", [pc_feature.o2a2] );
end

%% FURther process on features
D2R = pi/180;
thres_v_ang_d = 7.5; % degrees
thres_v_ang_r = thres_v_ang_d * D2R;

thres_verticality = cos(thres_v_ang_r);
linearity = [pc_feature.linearity];
planarity = [pc_feature.planarity];
verticality = [pc_feature.verticality];
idx = find(verticality>thres_verticality);
pc_sel = select(pc,idx);

% figure(200)
% subplot(1,2,1)
% pcshow(pc)
% subplot(1,2,2)
% pcshow(pc_sel)
% 
% figure(201)
% idx = find(linearity>0.5);
% pc_sel = select(pc,idx);
% pcshow(pc_sel)