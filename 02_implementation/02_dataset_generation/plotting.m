%% PARAMETERS

% saving path 
% foldername = 'U:/Dokumente/Projekte/DED_FOR/07_APs/AP1/01_Doku/figures/annular_beam/L_WOW/'; % L_WW  L_WOW   NL_WW   NL_WOW
foldername = 'data/figures/tests/'; % L_WW  L_WOW   NL_WW   NL_WOW
title_fig = 'multispot distribution'; % 'annular beam distribution'
% plotting definitions
fontAxisTicks = 12;
fontAxisLabel = 12;
fontClrb = 12; 
fontTitle = 12;
fontLegend = 10;

bool_save = 0;

bool_save_source =                  bool_save;
bool_save_wire =                    bool_save;
bool_save_2d_Tfield_top_view =      bool_save;
bool_save_2d_meltpool_top_view =    bool_save;
bool_save_2d_side_view =            bool_save;
bool_save_2d_meltpool_side_view =   bool_save;
bool_save_2d_meltpool_depth_view =  bool_save;
bool_save_2d_depth_view =           bool_save;
bool_save_3d_melt_pool_volume =     bool_save;

% paraview specs
bool_paraview = 0;
ini_t = nt;
dnt = 1;
title_vtk_file = 'FD_MF_multispot_beam';
filename_vtk = [foldername 'Paraview/result'];
%% PREPARATIONS
set(0,'defaultTextInterpreter','latex');

%% PLOTTING SOURCE
figure
if size(q,3) == 1
    contourf(X(:,:,end),Y(:,:,end),q*1e-6,'LineColor','none')
else
    contourf(X(:,:,end),Y(:,:,end),q(:,:,end)*1e-6,'LineColor','none')
end 
title('laser beam source','Interpreter','Latex','FontSize',fontTitle)
colormap(jet(256))
% colormap('hot')

clrbr = colorbar;
clrbr.Label.String = 'I $\left[\mathrm{W} / \mathrm{mm}^2\right]$';
clrbr.Label.Interpreter = 'latex';
clrbr.TickLabelInterpreter = 'latex';
clrbr.FontSize = fontClrb;

axis equal
set(gcf,'Color','w')
ax = gca;

xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;

xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
set(gca, 'XTick',xt, 'XTickLabel',xt*1000)
set(gca, 'YTick',yt, 'YTickLabel',yt*1000)
ax.TickLabelInterpreter = 'latex';

grid on 

if bool_save_source == 1
    set(gcf,'Units','centimeters');
    screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
    print(gcf,[foldername 'laser_beam_intensity'],'-dpdf','-r0')
end

%% PLOTTING WIRE INTENSITY 
if exist('q_wire','var')
    figure
    contourf(X(:,:,end),Y(:,:,end),-q_wire*1e-6,'LineColor','none')
    title('wire heat sink','Interpreter','Latex','FontSize',fontTitle)
    colormap(jet(256))
    % colormap('hot')
    
    clrbr = colorbar;
    clrbr.Label.String = 'I $\left[\mathrm{W}/\mathrm{mm}^2\right]$';
    clrbr.Label.Interpreter = 'latex';
    clrbr.TickLabelInterpreter = 'latex';
    clrbr.FontSize = fontClrb;
    
    axis equal
    set(gcf,'Color','w')
    ax = gca;
    
    xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
    ylabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
    
    ax.XAxis.FontSize = fontAxisTicks;
    ax.XLabel.FontSize = fontAxisLabel;
    ax.YAxis.FontSize = fontAxisTicks;
    ax.YLabel.FontSize = fontAxisLabel;
    
    xt = get(gca, 'XTick');
    yt = get(gca, 'YTick');
    set(gca, 'XTick',xt, 'XTickLabel',xt*1000)
    set(gca, 'YTick',yt, 'YTickLabel',yt*1000)
    ax.TickLabelInterpreter = 'latex';
    
    grid on 

    if bool_save_wire == 1
        set(gcf,'Units','centimeters');
        screenposition = get(gcf,'Position');
        set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
        print(gcf,[foldername 'wire_intensity'],'-dpdf','-r0')
    end
end


%% PLOTTING TEMPERATURE FIELD FROM TOP
% T_test = T_plot(:,:,end);
% T_test(T_test > Tm) = Tm;
color_text = 'w';
figure
% contourf(X(:,:,end),Y(:,:,end),T_test(:,:,end))
imagesc(x,y,T_plot(:,:,end))
hold on 
xline(0,'--','Color','b','LineWidth',1)
yline(0,'--','Color','b','LineWidth',1)
text(-0.25e-3, 3.5e-3, 'A-A', 'Color', color_text, 'FontSize', 14, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'Rotation', 90);
text(3.5e-3, 0.25e-3, 'B-B', 'Color', color_text, 'FontSize', 14, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center');
axis equal
set(gcf,'Color','w')
colormap("hot")

clrbr = colorbar;
caxis([T0 Tm])
clrbr.Label.String = '$T$ [K]';
clrbr.Label.Interpreter = 'latex';
clrbr.TickLabelInterpreter = 'latex';
clrbr.FontSize = fontClrb;

ticks = clrbr.Ticks;  % aktuelle Tick-Werte
labels = string(clrbr.Ticks);  % aktuelle Ticks als Strings
labels(end) = "$>1600$";      % letzte Beschriftung ersetzen
clrbr.TickLabels = labels;     % neue Labels setzen

ax = gca;
ax.TickLabelInterpreter = 'latex';
xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
ax.GridColor = [1, 1, 1];  % [R, G, B]
title('temperature distribution (camera view)','Interpreter','latex')

if bool_save_2d_Tfield_top_view == 1
    set(gcf,'Units','centimeters');
    screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
    print(gcf,[foldername 'Tfield_top_view'],'-dpdf','-r0')
end

%% PLOTTING MELT POOL AREA FROM TOP

figure
CM = contourf(X(:,:,end),Y(:,:,end),T_plot(:,:,end),[Tm Tm]);
set(gcf,'Color','w')
colormap("hot")
clim([T0 Tm])
ax = gca;
ax.TickLabelInterpreter = 'latex';
xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
ax.GridColor = [1, 1, 1];  % [R, G, B]
title('melted area (top view)','Interpreter','latex')
axis equal

CM(:,CM(2,:)>=1) = [];
[min_CM_y_val,ind_min_CM] = min(CM(2,:));
[max_CM_y_val,ind_max_CM] = max(CM(2,:));

min_CM_x_val = CM(1,ind_min_CM);
max_CM_x_val = CM(1,ind_max_CM);

width = max_CM_y_val - min_CM_y_val;
hold on 
OBJ = Annotate(ax, 'doublearrow', [min_CM_x_val max_CM_x_val], [min_CM_y_val max_CM_y_val],'Color', 'red');
hold off
hline = line(NaN,NaN,'LineWidth',1,'Color','red');
legend(hline,['width $w = $' num2str(round(width*1e+3,2)) ' mm'],'Interpreter','latex','FontSize',fontLegend)

if bool_save_2d_meltpool_top_view == 1
    set(gcf,'Units','centimeters');
    screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
    print(gcf,[foldername 'meltpool_top_view'],'-dpdf','-r0')
end

% %% PLOTTING MELTPOOL AREA DEPTH VIEW  
% figure('Units','pixels','Position',[100 100 560 280]); % kompakte Figuregröße
% ax = gca;
% contourf(reshape(Y(:,round(nx/2),:),[size(Y,1) size(X,3)]),...
%     reshape(Z(:,round(nx/2),:),[size(Z,1) size(Z,3)]),...
%     reshape(T_plot(:,round(nx/2),:),[size(T_plot,1) size(T_plot,3)]),[Tm Tm])
% hold on 
% text(ax, 0.94, 0.93, 'A-A', 'Units', 'normalized', 'Color', 'w', 'FontSize', 14, ...
%     'FontWeight', 'bold', 'HorizontalAlignment', 'center');
% axis equal
% set(gcf,'Color','w')
% colormap("hot")
% clim([T0 Tm])
% clrbr = colorbar;
% clrbr.Label.String = '$T$ [K]';
% clrbr.Label.Interpreter = 'latex';
% clrbr.TickLabelInterpreter = 'latex';
% clrbr.FontSize = fontClrb;
% 
% ax.TickLabelInterpreter = 'latex';
% xlabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
% ylabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
% ax.XAxis.FontSize = fontAxisTicks;
% ax.XLabel.FontSize = fontAxisLabel;
% ax.YAxis.FontSize = fontAxisTicks;
% ax.YLabel.FontSize = fontAxisLabel;
% xt = get(gca, 'XTick');
% yt = get(gca, 'YTick');
% set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
% set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
% ax.GridColor = [1, 1, 1];  % [R, G, B]
% title(title_fig,'Interpreter','latex')

%% PLOTTING 2D SIDE VIEW 
figure('Units','pixels','Position',[100 100 560 280]); % kompakte Figuregröße
ax = gca;
contourf(reshape(X(round(ny/2),:,:),[size(X,2) size(X,3)])...
    ,reshape(Z(round(ny/2),:,:),[size(X,2) size(X,3)])...
    ,reshape(T_plot(round(ny/2),:,:),[size(X,2) size(X,3)]))
hold on 
text(ax, 0.94, 0.93, 'B-B', 'Units', 'normalized', 'Color', 'w', 'FontSize', 14, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center');
axis equal
set(gcf,'Color','w')
colormap("hot")
clim([T0 Tm])
clrbr = colorbar;
clrbr.Label.String = '$T$ [K]';
clrbr.Label.Interpreter = 'latex';
clrbr.TickLabelInterpreter = 'latex';
clrbr.FontSize = fontClrb;

ax.TickLabelInterpreter = 'latex';
xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
ax.GridColor = [1, 1, 1];  % [R, G, B]
title(title_fig,'Interpreter','latex')


if bool_save_2d_side_view == 1
    set(gcf,'Units','centimeters');
    screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
    print(gcf,[foldername 'Tfield_side_view'],'-dpdf','-r0')
end 

%% PLOTTING 2D SIDE VIEW MELTED AREA
figure('Units','pixels','Position',[100 100 560 280]); % kompakte Figuregröße
ax = gca;
CM = contourf(reshape(X(round(ny/2),:,:),[size(X,2) size(X,3)])...
    ,reshape(Z(round(ny/2),:,:),[size(X,2) size(X,3)])...
    ,reshape(T_plot(round(ny/2),:,:),[size(X,2) size(X,3)]),[Tm Tm]);
hold on 
text(ax, 0.94, 0.93, 'B-B', 'Units', 'normalized', 'Color', 'k', 'FontSize', 14, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center');
axis equal
set(gcf,'Color','w')
colormap("hot")
clim([T0 Tm])
clrbr = colorbar;
clrbr.Label.String = '$T$ [K]';
clrbr.Label.Interpreter = 'latex';
clrbr.TickLabelInterpreter = 'latex';
clrbr.FontSize = fontClrb;

ax.TickLabelInterpreter = 'latex';
xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
ax.GridColor = [1, 1, 1];  % [R, G, B]
title(title_fig,'Interpreter','latex')

CM(:,CM(2,:)>=1) = [];
[min_CM_z_val,ind_min_CM] = min(CM(2,:));
[max_CM_z_val,ind_max_CM] = max(CM(2,:));

min_CM_x_val = CM(1,ind_min_CM);
max_CM_x_val = CM(1,ind_max_CM);

depth = max_CM_z_val - min_CM_z_val;
hold on 
% OBJ = Annotate(ax, 'doublearrow', [min_CM_x_val min_CM_x_val], [min_CM_z_val max_CM_z_val],'Color', 'red');
hold off
hline = line(NaN,NaN,'LineWidth',1,'Color','red');
legend(hline,['max depth $d = $' num2str(round(depth*1e+3,2)) ' mm'],'Interpreter','latex','FontSize',fontLegend,'Location','se')

if bool_save_2d_meltpool_side_view == 1
    set(gcf,'Units','centimeters');
    screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
    print(gcf,[foldername 'meltpool_side_view'],'-dpdf','-r0')
end

%% PLOTTING 2D DEPTH FRONT VIEW MELTED AREA
np = abs(min(X(:)) - min_CM_x_val)/dx;
figure('Units','pixels','Position',[100 100 560 280]); % kompakte Figuregröße
ax = gca;
CM = contourf(reshape(Y(:,round(np),:),[size(Y,1) size(X,3)])...
    ,reshape(Z(:,round(np),:),[size(Z,1) size(Z,3)])...
    ,reshape(T_plot(:,round(np),:),[size(T_plot,1) size(T_plot,3)]),[Tm Tm]);
hold on 
text(ax, 0.94, 0.93, 'A-A', 'Units', 'normalized', 'Color', 'k', 'FontSize', 14, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center');
axis equal
set(gcf,'Color','w')
colormap("hot")
clim([T0 Tm])
clrbr = colorbar;
clrbr.Label.String = '$T$ [K]';
clrbr.Label.Interpreter = 'latex';
clrbr.TickLabelInterpreter = 'latex';
clrbr.FontSize = fontClrb;

ax.TickLabelInterpreter = 'latex';
xlabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
ax.GridColor = [1, 1, 1];  % [R, G, B]
title(title_fig,'Interpreter','latex')
grid on 

CM(:,CM(2,:)>=1) = [];
[min_CM_z_val,ind_min_CM] = min(CM(2,:));
[max_CM_z_val,ind_max_CM] = max(CM(2,:));

min_CM_y_val = CM(1,ind_min_CM);
max_CM_y_val = CM(1,ind_max_CM);

depth = max_CM_z_val - min_CM_z_val;
hold on 
% OBJ = Annotate(ax, 'doublearrow', [min_CM_x_val min_CM_x_val], [min_CM_z_val max_CM_z_val],'Color', 'red');
hold off
hline = line(NaN,NaN,'LineWidth',1,'Color','red');
legend(hline,['max depth $d = $' num2str(round(depth*1e+3,2)) ' mm'],'Interpreter','latex','FontSize',fontLegend,'Location','se')

if bool_save_2d_meltpool_depth_view == 1
    set(gcf,'Units','centimeters');
    screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
    print(gcf,[foldername 'meltpool_depth_view'],'-dpdf','-r0')
end 

%% PLOTTING 2D DEPTH FRONT VIEW 
figure('Units','pixels','Position',[100 100 560 280]); % kompakte Figuregröße
ax = gca;
contourf(reshape(Y(:,round(nx/2),:),[size(Y,1) size(X,3)])...
    ,reshape(Z(:,round(nx/2),:),[size(Z,1) size(Z,3)])...
    ,reshape(T_plot(:,round(nx/2),:),[size(T_plot,1) size(T_plot,3)]))
hold on 
text(ax, 0.94, 0.93, 'A-A', 'Units', 'normalized', 'Color', 'w', 'FontSize', 14, ...
    'FontWeight', 'bold', 'HorizontalAlignment', 'center');
axis equal
set(gcf,'Color','w')
colormap("hot")
clim([T0 Tm])
clrbr = colorbar;
clrbr.Label.String = '$T$ [K]';
clrbr.Label.Interpreter = 'latex';
clrbr.TickLabelInterpreter = 'latex';
clrbr.FontSize = fontClrb;

ax.TickLabelInterpreter = 'latex';
xlabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
ax.GridColor = [1, 1, 1];  % [R, G, B]
title(title_fig,'Interpreter','latex')

if bool_save_2d_depth_view == 1
    set(gcf,'Units','centimeters');
    screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
    print(gcf,[foldername 'Tfield_depth_view'],'-dpdf','-r0')
end 

%% PLOTTING 3D MELT POOL VOLUME
fac = 0.7; % factor to scale the axis label 
figure
subplot(2,2,1)
p = patch(isosurface(X,Y,Z,T_plot, Tm));
set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
caps = patch(isocaps(X,Y,Z,T_plot, Tm));
set(caps, 'FaceColor', 'red', 'EdgeColor', 'none');

ax = gca;
ax.TickLabelInterpreter = 'latex';
xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
zlabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize =     fac*fontAxisTicks;
ax.XLabel.FontSize =    fac*fontAxisLabel;
ax.YAxis.FontSize =     fac*fontAxisTicks;
ax.YLabel.FontSize =    fac*fontAxisLabel;
ax.ZAxis.FontSize =     fac*fontAxisTicks;
ax.ZLabel.FontSize =    fac*fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
zt = get(gca, 'ZTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
set(gca,'ZTick',zt,'ZTickLabel',zt*1e+3)

% Ansicht anpassen
view(3); % 3D-Ansicht
% axis tight;
axis equal;
camlight;
lighting gouraud;
light('Position', [0 -1 0], 'Style', 'infinite', 'Color', [1 0.8 0.8]);
light('Position', [1 -1 0], 'Style', 'infinite', 'Color', [0.5 0.5 0.5]);

subplot(2,2,2)
p = patch(isosurface(X,Y,Z,T_plot, Tm));
set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
caps = patch(isocaps(X,Y,Z,T_plot, Tm));
set(caps, 'FaceColor', 'red', 'EdgeColor', 'none');

ax = gca;
ax.TickLabelInterpreter = 'latex';
xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
zlabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize =     fac*fontAxisTicks;
ax.XLabel.FontSize =    fac*fontAxisLabel;
ax.YAxis.FontSize =     fac*fontAxisTicks;
ax.YLabel.FontSize =    fac*fontAxisLabel;
ax.ZAxis.FontSize =     fac*fontAxisTicks;
ax.ZLabel.FontSize =    fac*fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
zt = get(gca, 'ZTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
set(gca,'ZTick',zt,'ZTickLabel',zt*1e+3)


% Ansicht anpassen
view(3); % 3D-Ansicht
% axis tight;
axis equal;
camlight;
lighting gouraud;
light('Position', [0 -1 0], 'Style', 'infinite', 'Color', [1 0.8 0.8]);
light('Position', [1 -1 0], 'Style', 'infinite', 'Color', [0.5 0.5 0.5]);

subplot(2,2,3)
p = patch(isosurface(X,Y,Z,T_plot, Tm));
set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
caps = patch(isocaps(X,Y,Z,T_plot, Tm));
set(caps, 'FaceColor', 'red', 'EdgeColor', 'none');

ax = gca;
ax.TickLabelInterpreter = 'latex';
xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
zlabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize =     fac*fontAxisTicks;
ax.XLabel.FontSize =    fac*fontAxisLabel;
ax.YAxis.FontSize =     fac*fontAxisTicks;
ax.YLabel.FontSize =    fac*fontAxisLabel;
ax.ZAxis.FontSize =     fac*fontAxisTicks;
ax.ZLabel.FontSize =    fac*fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
zt = get(gca, 'ZTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
set(gca,'ZTick',zt,'ZTickLabel',zt*1e+3)


% Ansicht anpassen
view(3); % 3D-Ansicht
% axis tight;
axis equal;
camlight;
lighting gouraud;
light('Position', [0 -1 0], 'Style', 'infinite', 'Color', [1 0.8 0.8]);
light('Position', [1 -1 0], 'Style', 'infinite', 'Color', [0.5 0.5 0.5]);

subplot(2,2,4)
p = patch(isosurface(X,Y,Z,T_plot, Tm));
set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
caps = patch(isocaps(X,Y,Z,T_plot, Tm));
set(caps, 'FaceColor', 'red', 'EdgeColor', 'none');

ax = gca;
ax.TickLabelInterpreter = 'latex';
xlabel('$x\: [\mathrm{mm}] $','Interpreter','Latex')
ylabel('$y\: [\mathrm{mm}] $','Interpreter','Latex')
zlabel('$z\: [\mathrm{mm}] $','Interpreter','Latex')
ax.XAxis.FontSize =     fac*fontAxisTicks;
ax.XLabel.FontSize =    fac*fontAxisLabel;
ax.YAxis.FontSize =     fac*fontAxisTicks;
ax.YLabel.FontSize =    fac*fontAxisLabel;
ax.ZAxis.FontSize =     fac*fontAxisTicks;
ax.ZLabel.FontSize =    fac*fontAxisLabel;
xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
zt = get(gca, 'ZTick');
set(gca,'XTick',xt,'XTickLabel',xt*1e+3)
set(gca,'YTick',yt,'YTickLabel',yt*1e+3)
set(gca,'ZTick',zt,'ZTickLabel',zt*1e+3)


% Ansicht anpassen
view(3); % 3D-Ansicht
% axis tight;
axis equal;
camlight;
lighting gouraud;
light('Position', [0 -1 0], 'Style', 'infinite', 'Color', [1 0.8 0.8]);
light('Position', [1 -1 0], 'Style', 'infinite', 'Color', [0.5 0.5 0.5]);
sgtitle('melt pool volume','Interpreter','latex')
if bool_save_3d_melt_pool_volume == 1
    rotate3d on;
    disp('Drücke Enter, um den Plot zu speichern...');
    pause;
    set(gcf,'Units','centimeters');
    screenposition = get(gcf,'Position');
    set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',[screenposition(3:4)]);
    print(gcf,[foldername 'meltpool_volume'],'-dpdf','-vector')
end 

