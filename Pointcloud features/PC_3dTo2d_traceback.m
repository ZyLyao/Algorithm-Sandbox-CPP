function [PC_out] = PC_3dTo2d_traceback(PC_in, Lidar_pos,pln_paras)
%PROJECT_PNT2PLN projecting a pointcloud onto a plane along it's laser
%   Notice that the Lidar_pos and PC coordiantes should be in one same
%   coordinate system.
% PC: Nx3 coordiantes of points in the PC

pln_N = pln_paras(1:3);
pln_d = pln_paras(4);
d_L_pln = dot(pln_N,Lidar_pos) + pln_d;

LP = PC_in-Lidar_pos';
ds_P_L = abs(LP*pln_N);

LP_prime = LP * d_L_pln ./ds_P_L;

PC_out = LP_prime + Lidar_pos';
end

