% INIT
fclose("all")
close all 
clear 
clc

% plotting definitions
fontAxisTicks = 12;
fontAxisLabel = 12;
fontClrb = 12; 
fontTitle = 12;
fontLegend = 10;
linewidth = 2;

% paths
addpath('functions/')

%% PARAMETERS

% nonlinear material properties: SS316
Tm_s = 1360 + 273.15; % [K] melting point (solidus)
Tm_l = 1410 + 273.15; % [K] melting point (liquidus)

Tm = 1410 + 273.15; % [K] melting point (liquidus)

dh = 270e+3;    % [J/kg] latent heat of fusion  

Tvec_lamvec = [25 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1450 1500 1600]+273.15;              % [K] vector of temperatures 
lamvec = [13.4 15.5 17.6 19.4 21.8 23.4 24.5 25.1 27.2 27.9 29.1 29.3 30.9 31.1 28.5 29.5 30.5];               % [W/(m*K)] thermal conductivity
T_lam_thres = 1600+273.15;
lam_thres = lamvec(end);
lam_lin = interp1(Tvec_lamvec,lamvec,Tm);

Tvec_cpvec = [25 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1385 1450]+273.15;        % [K] vector of temperatures
cpvec = [0.47 0.49 0.52 0.54 0.56 0.57 0.59 0.60 0.63 0.64 0.66 0.67 0.70 0.71 0.72 0.83]*1e+3;    % [J/(kg*K)] specific heat
T_cp_thres = 1450+273.15;
cp_thres = cpvec(end);
cp_lin = interp1(Tvec_cpvec,cpvec,Tm);

Tvec_rhovec = [25 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1385 1450 1500 1600]+273.15;       % [K] vector of temperatures
rhovec = [7950 7921 7880 7833 7785 7735 7681 7628 7575 7520 7462 7411 7361 7311 7269 6881 6842 6765];        % [kg/m^3] density
T_rho_thres = 1600+273.15;
rho_thres = rhovec(end);
rho_lin = interp1(Tvec_rhovec,rhovec,Tm);

%% FUNCTION FIT
% thermal conductivity
poly_degree_lam = 10;
p_lam = polyfit(Tvec_lamvec, lamvec, poly_degree_lam);

Tvec_lamvec_plot = linspace(min(Tvec_lamvec), max(Tvec_lamvec), 200);
lamvec_fit = polyval(p_lam, Tvec_lamvec_plot);

% specific heat
poly_degree_cp = 8;
p_cp = polyfit(Tvec_cpvec, cpvec, poly_degree_cp);

Tvec_cpvec_plot = linspace(min(Tvec_cpvec), max(Tvec_cpvec), 200);
cpvec_fit = polyval(p_cp, Tvec_cpvec_plot);

% density
poly_degree_rho = 7;
p_rho = polyfit(Tvec_rhovec, rhovec, poly_degree_rho);

Tvec_rhovec_plot = linspace(min(Tvec_rhovec), max(Tvec_rhovec), 200);
rhovec_fit = polyval(p_rho, Tvec_rhovec_plot);





%% PLOTTING

% thermal conductivity
figure
plot(Tvec_lamvec,lamvec,'d','LineWidth',linewidth)
hold on 
plot(Tvec_lamvec_plot,lamvec_fit,'LineWidth',linewidth)
legend('data points','function fit','FontSize',fontLegend,'Location','nw')
set(gcf,'Color','w')
ax = gca;
xlabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
ylabel('$\lambda\: [\mathrm{Wm^{-1}K^{-1}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('thermal conductivity','FontSize',fontTitle)

% specific heat 
figure
plot(Tvec_cpvec,cpvec,'d','LineWidth',linewidth)
hold on 
plot(Tvec_cpvec_plot,cpvec_fit,'LineWidth',linewidth)
legend('data points','function fit','FontSize',fontLegend,'Location','nw')
set(gcf,'Color','w')
ax = gca;
xlabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
ylabel('$c_{\mathrm{p}}\: [\mathrm{Jkg^{-1}K^{-1}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('specific heat','FontSize',fontTitle)

% density
figure
plot(Tvec_rhovec,rhovec,'d','LineWidth',linewidth)
hold on 
plot(Tvec_rhovec_plot,rhovec_fit,'LineWidth',linewidth)
legend('data points','function fit','FontSize',fontLegend)
set(gcf,'Color','w')
ax = gca;
xlabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
ylabel('$\rho \: [\mathrm{kgm^{-3}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('density','FontSize',fontTitle)




