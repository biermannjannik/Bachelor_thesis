% from "THERMOPHYSICAL PROPERTIES OF STAINLESS STEELS by Choong S. Kim 1975"
% COMMENTS
% - values of 304L/316L can be used for 304/316
% - 1 calorie := 4184 Joule

%% PARAMETERS 
% plotting definitions
fontAxisTicks = 12;
fontAxisLabel = 12;
fontClrb = 12; 
fontTitle = 12;
fontLegend = 10;
linewidth = 2;

%% THERMOPHYSICAL PROPERTIES
Tm = 1700; % [K] melting temperature 
dh = 270e+3;    % [J/kg] latent heat of fusion  NOT FROM THE SAME REFERENCE 
% temperature vector 
T = [ ...
  300,  400,  500,  600,  700,  800,  900, 1000, 1100, 1200, ...
 1300, 1400, 1500, 1600, 1700, 1700, 1800, 1900, 2000, 2100, ...
 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000];

% conductivity in [W/(cm·K)]
lam_304  = [0.1297, 0.1459, 0.1620, 0.1782, 0.1944, 0.2106, 0.2267, 0.2429, 0.2591, 0.2753, ...
           0.2914, 0.3076, 0.3238, 0.3400, 0.3561, 0.1781, 0.1811, 0.1846, 0.1879, 0.1911, ...
           0.1944, 0.1976, 0.2009, 0.2041, 0.2073, 0.2106, 0.2138, 0.2171, 0.2203];

lam_316  = [0.1396, 0.1553, 0.1710, 0.1868, 0.2025, 0.2182, 0.2339, 0.2496, 0.2653, 0.2810, ...
           0.2967, 0.3125, 0.3282, 0.3439, 0.3596, 0.1798, 0.1831, 0.1864, 0.1897, 0.1930, ...
           0.1962, 0.1995, 0.2028, 0.2061, 0.2094, 0.2126, 0.2159, 0.2192, 0.2225];

% diffusivity in 1e-2 cm²/s
alpha_304 = [3.237, 3.546, 3.849, 4.148, 4.440, 4.727, 5.009, 5.285, 5.555, 5.820, ...
              6.080, 6.334, 6.582, 6.825, 7.062, 3.238, 3.323, 3.413, 3.508, 3.608, ...
              3.713, 3.822, 3.936, 4.055, 4.179, 4.307, 4.441, 4.579, 4.721];

alpha_316 = [3.529, 3.834, 4.132, 4.424, 4.710, 4.989, 5.262, 5.529, 5.190, 6.044, ...
              6.292, 6.534, 6.769, 6.999, 7.222, 3.350, 3.435, 3.526, 3.622, 3.723, ...
              3.829, 3.940, 4.057, 4.178, 4.305, 4.437, 4.574, 4.716, 4.864];


% density [g/cm^3] 
rho_304 = [ ...
 7.894; 7.860; 7.823; 7.783; 7.742; 7.698; 7.652; 7.603; 7.552; 7.499; 7.444; ...
 7.386; 7.326; 7.264; 7.199; 6.926; 6.862; 6.795; 6.725; 6.652; 6.577; 6.498; ...
 6.416; 6.331; 6.243; 6.152; 6.058; 5.961; 5.861];

rho_316 = [ ...
 7.954; 7.910; 7.864; 7.818; 7.771; 7.723; 7.674; 7.624; 7.574; 7.523; 7.471; ...
 7.419; 7.365; 7.311; 7.256; 6.979; 6.920; 6.857; 6.791; 6.721; 6.648; 6.571; ...
 6.490; 6.406; 6.318; 6.226; 6.131; 6.032; 5.930];


% enthalpie [cal/g]
H_304 = [ ...
 0.23; 12.57; 25.24; 38.24; 51.55; 65.19; 79.14; 93.43; 108.03; 122.95; 138.20; 153.77; ...
 169.66; 185.88; 202.41; 226.41; 285.41; 304.41; 323.41; 342.41; 361.41; 380.41; 399.41; ...
 418.41; 437.41; 456.41; 475.41; 494.31; 513.41];

H_316 = [ ...
 0.21; 14.29; 24.69; 37.41; 50.44; 63.79; 77.46; 91.44; 105.75; 120.37; 135.30; 150.56; ...
 166.13; 182.02; 198.23; 262.23; 285.67; 299.03; 317.43; 335.83; 354.23; 372.63; 391.03; ...
 409.43; 427.83; 446.23; 464.63; 483.03; 501.43];

% specific heat [cal/(g*K)]
cp_304 = [ ...
 0.1219; 0.1251; 0.1283; 0.1315; 0.1348; 0.1380; 0.1412; 0.1444; 0.1476; 0.1509; 0.1541; ...
 0.1573; 0.1605; 0.1638; 0.1670; 0.1900; 0.1900; 0.1900; 0.1900; 0.1900; 0.1900; 0.1900; ...
 0.1900; 0.1900; 0.1900; 0.1900; 0.1900; 0.1900; 0.1900];

cp_316 = [ ...
 0.1192; 0.1224; 0.1256; 0.1287; 0.1319; 0.1351; 0.1383; 0.1416; 0.1446; 0.1478; 0.1510; ...
 0.1541; 0.1573; 0.1605; 0.1637; 0.1840; 0.1840; 0.1840; 0.1840; 0.1840; 0.1840; 0.1840; ...
 0.1840; 0.1840; 0.1840; 0.1840; 0.1840; 0.1840; 0.1840];

%% CONVERSION TO SI-UNITS 
lam_304 = lam_304*1e2;
lam_304_lin = mean(lam_304);

lam_316 = lam_316*1e2;
lam_316_lin = mean(lam_316);

alpha_304 = alpha_304*(1e-2)^2;
alpha_316 = alpha_316*(1e-2)^2;

rho_304 = rho_304*1e3;
rho_304_lin = mean(rho_304);

rho_316 = rho_316*1e3; 
rho_316_lin = mean(rho_316);

H_304 = H_304*(4.184/1e-3);
H_316 = H_316*(4.184/1e-3);

cp_304 = cp_304*(4.184/1e-3);
cp_304_lin = mean(rho_304);

cp_316 = cp_316*(4.184/1e-3);
cp_316_lin = mean(cp_316);

%% FUNCTION FIT 
% thermal conductivity
T_c = 1700;
T1 = T(T < T_c);
lam_304_1 = lam_304(T < T_c);
lam_316_1 = lam_316(T < T_c);

T2 = unique(T(T >= T_c));

bool_vec = (T >= T_c);

for i = 1:length(bool_vec)
    if bool_vec(i) == 1
        bool_vec(i) = false;
        break; 
    end 
end 

lam_304_2 = lam_304(bool_vec);
lam_316_2 = lam_316(bool_vec);

% Lineare Fits
p_lam_304_1 = polyfit(T1, lam_304_1, 1);  % unterer Bereich
p_lam_304_2 = polyfit(T2, lam_304_2, 1);  % oberer Bereich

p_lam_316_1 = polyfit(T1, lam_316_1, 1);  % unterer Bereich
p_lam_316_2 = polyfit(T2, lam_316_2, 1);  % oberer Bereich

% Fit-Funktionen:
lam_304_fit = @(T) (T < T_c) .* (p_lam_304_1(1)*T + p_lam_304_1(2)) + ...
                  (T >= T_c) .* (p_lam_304_2(1)*T + p_lam_304_2(2));

lam_316_fit = @(T) (T < T_c) .* (p_lam_316_1(1)*T + p_lam_316_1(2)) + ...
                  (T >= T_c) .* (p_lam_316_2(1)*T + p_lam_316_2(2));

% specific heat 
T_c = 1700;
T1 = T(T < T_c);
cp_304_1 = cp_304(T < T_c);
cp_316_1 = cp_316(T < T_c);

T2 = unique(T(T >= T_c));

bool_vec = (T >= T_c);

for i = 1:length(bool_vec)
    if bool_vec(i) == 1
        bool_vec(i) = false;
        break; 
    end 
end 

cp_304_2 = cp_304(bool_vec);
cp_316_2 = cp_316(bool_vec);

% Lineare Fits
p_cp_304_1 = polyfit(T1, cp_304_1, 1);  % unterer Bereich
p_cp_304_2 = polyfit(T2, cp_304_2, 1);  % oberer Bereich

p_cp_316_1 = polyfit(T1, cp_316_1, 1);  % unterer Bereich
p_cp_316_2 = polyfit(T2, cp_316_2, 1);  % oberer Bereich

% Fit-Funktionen:
cp_304_fit = @(T) (T < T_c) .* (p_cp_304_1(1)*T + p_cp_304_1(2)) + ...
                  (T >= T_c) .* (p_cp_304_2(1)*T + p_cp_304_2(2));

cp_316_fit = @(T) (T < T_c) .* (p_cp_316_1(1)*T + p_cp_316_1(2)) + ...
                  (T >= T_c) .* (p_cp_316_2(1)*T + p_cp_316_2(2));


% density
T_c = 1700;
T1 = T(T < T_c);
rho_304_1 = rho_304(T < T_c);
rho_316_1 = rho_316(T < T_c);

T2 = unique(T(T >= T_c));

bool_vec = (T >= T_c);

for i = 1:length(bool_vec)
    if bool_vec(i) == 1
        bool_vec(i) = false;
        break; 
    end 
end 

rho_304_2 = rho_304(bool_vec);
rho_316_2 = rho_316(bool_vec);

% Lineare Fits
p_rho_304_1 = polyfit(T1, rho_304_1, 1);  % unterer Bereich
p_rho_304_2 = polyfit(T2, rho_304_2, 1);  % oberer Bereich

p_rho_316_1 = polyfit(T1, rho_316_1, 1);  % unterer Bereich
p_rho_316_2 = polyfit(T2, rho_316_2, 1);  % oberer Bereich

% Fit-Funktionen:
rho_304_fit = @(T) (T < T_c) .* (p_rho_304_1(1)*T + p_rho_304_1(2)) + ...
                  (T >= T_c) .* (p_rho_304_2(1)*T + p_rho_304_2(2));

rho_316_fit = @(T) (T < T_c) .* (p_rho_316_1(1)*T + p_rho_316_1(2)) + ...
                  (T >= T_c) .* (p_rho_316_2(1)*T + p_rho_316_2(2));


%% PLOTTING
dT = 20;
T_plot_fit = T(1):dT:T(end);

% thermal conductivity
figure
l1 = plot(T,lam_304,'d','LineWidth',linewidth);
hold on  
plot(T_plot_fit,lam_304_fit(T_plot_fit),'LineWidth',linewidth,'Color',l1.Color)

l2 = plot(T,lam_316,'o','LineWidth',linewidth);
plot(T_plot_fit,lam_316_fit(T_plot_fit),'LineWidth',linewidth,'Color',l2.Color)

legend('','S304','','S316','FontSize',fontLegend,'Location','nw')
set(gcf,'Color','w')
ax = gca;
xlabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
ylabel('$\lambda\: [\mathrm{Wm^{-1}K^{-1}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('\textbf{thermal conductivity}','Interpreter','latex')


% specific heat 
figure
l1 = plot(T,cp_304,'d','LineWidth',linewidth);
hold on  
plot(T_plot_fit,cp_304_fit(T_plot_fit),'LineWidth',linewidth,'Color',l1.Color)

l2 = plot(T,cp_316,'o','LineWidth',linewidth);
plot(T_plot_fit,cp_316_fit(T_plot_fit),'LineWidth',linewidth,'Color',l2.Color)

legend('','S304','','S316','FontSize',fontLegend,'Location','nw')
set(gcf,'Color','w')
ax = gca;
xlabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
ylabel('$c_{\mathrm{p}}\: [\mathrm{Jkg^{-1}K^{-1}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('\textbf{specific heat}','FontSize','Interpreter','latex')


% density
figure
l1 = plot(T,rho_304,'d','LineWidth',linewidth);
hold on  
plot(T_plot_fit,rho_304_fit(T_plot_fit),'LineWidth',linewidth,'Color',l1.Color)

l2 = plot(T,rho_316,'o','LineWidth',linewidth);
plot(T_plot_fit,rho_316_fit(T_plot_fit),'LineWidth',linewidth,'Color',l2.Color)

legend('','S304','','S316','FontSize',fontLegend,'Location','sw')
set(gcf,'Color','w')
ax = gca;
xlabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
ylabel('$\rho \: [\mathrm{kgm^{-3}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('\textbf{density}','FontSize',fontTitle,'Interpreter','latex')


% enthalpie
figure
plot(T,H_304,'d','LineWidth',linewidth)
hold on  
plot(T,H_316,'o','LineWidth',linewidth)
legend('S304','S316','FontSize',fontLegend,'Location','nw')
set(gcf,'Color','w')
ax = gca;
xlabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
ylabel('$H \: [\mathrm{Jkg^{-1}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('\textbf{enthalpie}','FontSize',fontTitle,'Interpreter','latex')

% enthalpie
figure
plot(H_304,T,'d','LineWidth',linewidth)
set(gcf,'Color','w')
ax = gca;
ylabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
xlabel('$H \: [\mathrm{Jkg^{-1}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('\textbf{enthalpie} S304','FontSize',fontTitle,'Interpreter','latex')


figure
plot(H_316,T,'o','LineWidth',linewidth)
set(gcf,'Color','w')
ax = gca;
ylabel('$T\: [\mathrm{K}] $','Interpreter','Latex')
xlabel('$H \: [\mathrm{Jkg^{-1}}] $','Interpreter','Latex')

ax.XAxis.FontSize = fontAxisTicks;
ax.XLabel.FontSize = fontAxisLabel;
ax.YAxis.FontSize = fontAxisTicks;
ax.YLabel.FontSize = fontAxisLabel;
ax.TickLabelInterpreter = 'latex';
title('\textbf{enthalpie} S316','FontSize',fontTitle,'Interpreter','latex')







