function plot_one_feature(fig_num,FLAG_SHOW_3D, pc, pnts_proj,feature_name, feature )
disp("Draw for "+feature_name)
figure(fig_num)
subplot(1,2,1)
    scatter3(pc.Location(:,1),pc.Location(:,2),pc.Location(:,3),0.1,feature);
    xlabel('x, m')
    ylabel('y, m')
    zlabel('z, m')
    title("original "+feature_name)
    colormap(gca,'turbo')
    colorbar;
    axis equal

    subplot(1,2,2)
    histogram(feature)
end

