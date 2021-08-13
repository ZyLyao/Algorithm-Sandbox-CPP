function plot_one_feature(fig_num,FLAG_SHOW_3D, pc, pnts_proj,feature_name, feature )
disp("Draw for "+feature_name)
figure(fig_num)
subplot(2,2,1)
scatter(pnts_proj(:,2),pnts_proj(:,1),0.1,feature);
xlabel('y, m')
ylabel('x, m')
colormap(gca,'turbo')
c = colorbar;
c.Label.String = feature_name;
grid on
title("projected "+feature_name)
subplot(2,2,2)
scatter(feature,pnts_proj(:,1),0.1,pc.Location(:,3));
xlabel(feature_name)
ylabel('x, m')
title("x v.s. "+feature_name)
colormap(gca,'turbo')
c = colorbar;
c.Label.String = 'height. m';
grid on
subplot(2,2,3)
scatter(pnts_proj(:,2),feature,0.1,pc.Location(:,3));
xlabel('y, m')
ylabel(feature_name)
title("y v.s. "+feature_name)
colormap(gca,'turbo')
colorbar
c = colorbar;
c.Label.String = 'height. m';
grid on
if FLAG_SHOW_3D
    subplot(2,2,4)
    scatter3(pc.Location(:,1),pc.Location(:,2),pc.Location(:,3),0.1,feature);
    xlabel('x, m')
    ylabel('y, m')
    zlabel('z, m')
    title("original "+feature_name)
    colormap(gca,'turbo')
    colorbar;
end


end

