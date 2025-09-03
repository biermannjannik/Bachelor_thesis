%% INIT
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

% % paths
% addpath('../functions/')

%% Parameters
% nonlinear material properties: SS316
Tm = 1410 + 273.15; % [K] melting point (liquidus)
% Tm_SS316 = 1360 + 273.15; % [K] melting point (solidus)
dh = 270e+3;    % [J/kg] latent heat of fusion  

% function fit of material properties from script: nl_matProps_fct_fit
run("nl_matProps_fct_fit.m");

% linear material properties 
lam = lam_lin;     % [W/(m*K)] thermal conductivity
rho = rho_lin;     % [kg/m^3] density
cp = cp_lin;        % [J/(kg*K)] specific heat
a = lam/(rho*cp);                         % [m2/s] thermal diffusivity
A_b = 0.4;

t0 = 0;             % [s] begin time  
te = 1;             % [s] end time 

x1 = -4e-3;         % [m] begin of domain x and y
x2 = 4e-3;          % [m] end of domain x and y
z1 = -4e-3;             % [m] begin of domain z
z2 = 0;          % [m] end of domain z

T0 = 300;           % [K] initial temperature

Nx = 20; Ny = 20; Nz = 10;  % 3D grid size
Nt = 1001;                  % Time steps

N_samples = 100;

% Discretization
dx = (x2-x1) / (Nx - 1);
dy = dx;
dz = (z2-z1) / (Nz - 1);
dt = (te-t0) / (Nt - 1);
x = linspace(x1,x2, Nx);
y = x;
z = linspace(z1,z2, Nz);
t = linspace(0, (te-t0), Nt);

Fo = (lam*dt)/(rho*cp*dx^2);

% condition for spatial and time discretization
if (1-6*Fo) < 0
    error("Adjust spatial and time discretization.")
end 

% meshgrid
[X,Y] = meshgrid(x,y);


% Heat source Q: [N_samples x Nx x Ny]
Q_rel = zeros(Nx, Ny, 'single');

w = 0.5e-3;
r_max = 1e-3;

for iy = 1:Ny
    for ix = 1:Nx
        r = sqrt(X(iy,ix)^2+Y(iy,ix)^2);
        Q_rel(iy,ix) = annular_beam_V2(1,A_b,w,r_max,r);
    end 
end 

 disp(sum(Q_rel.*dx^2,'all')/A_b);
 figure;
 imagesc(Q_rel)

%%

Nt_save = 21;
dt_save = (te-t0)/(Nt_save-1);
t_save = linspace(0,te,Nt_save);
each_t = (Nt-1)/(Nt_save-1);

t_save_2_t = @(t_save) 1+each_t.*(t_save-1);

P = zeros(N_samples,Nt_save-1,'single');

for i = 1:N_samples
    %P(i,:) = gen_random_signal(500,2000,dt_save*5,dt_save*10,Nt_save,t_power,te);
    P(i,:) = generate_random_signal(Nt_save-1,600,6000,2,8);
end

v = zeros(N_samples,Nt_save-1,'single');

for i = 1:N_samples
    %v(i:i+19,:) = ones(20,1)*generate_random_signal(Nt_save,35e-4,35e-3,3,22);
    v(i,:) = generate_random_signal(Nt_save-1,0,35e-3,2,8);
end

for i = 1:N_samples
    % One random constant value in [35e-4, 35e-3]
    constV = 35e-4 + (35e-3 - 35e-4) * rand();
    v(i,:) = constV * ones(1, Nt_save-1);
end


%%
figure
plot(P(2,:))
title('Beispielhafter Leistungsverlauf')
%%
% Temperature field: U [N_samples x Nt x Nx x Ny x Nz]
T = zeros(Nx, Ny, Nz, Nt, 'single');

%U = zeros(N_samples,Nx,Ny,Nz,Nt_save,'single');
IC = zeros(N_samples,Nx,Ny,Nz,'single');
IC(1,:,:,:) = T0*ones(Nx,Ny,Nz,'single');
IC_candidates = zeros(N_samples,Nx,Ny,Nz,'single');

N_samples_train = N_samples*4/5;
N_samples_test = N_samples*1/5;

N_points_bias = 4000;
N_points_v = 1000;
N_points_rd = 1000;
N_points_ic = 500;
N_points_bc = 500;

xyz_t_bias_train = biased_sampling_center_box_annular(N_points_bias,[Nx,Ny,Nz],Nt_save,[(Nx+1)/2,(Ny+1)/2,Nz],0.15*Nx,-1);
U_bias_train = zeros(N_samples_train,N_points_bias,'single');

xyz_t_bias_test = biased_sampling_center_box_annular(N_points_bias,[Nx,Ny,Nz],Nt_save,[(Nx+1)/2,(Ny+1)/2,Nz],0.15*Nx,-1);
U_bias_test = zeros(N_samples_test,N_points_bias,'single');

xyz_t_rd_train = zeros(N_points_rd,4,'single');
U_rd_train = zeros(N_samples_train,N_points_rd,'single');

xyz_t_rd_test = zeros(N_points_rd,4,'single');
U_rd_test = zeros(N_samples_test,N_points_rd,'single');

xyz_t_v_train = biased_sampling(N_points_v,[Nx,Ny,Nz],Nt_save,[(Nx+1)/2,(Ny+1)/2,Nz],0.15*Nx);
rd_x_v_train = randi([1,ceil(Nx/2)],N_points_v,1);
xyz_t_v_train(:,1) = rd_x_v_train;
U_v_train = zeros(N_samples_train,N_points_v,'single');

xyz_t_v_test = biased_sampling(N_points_v,[Nx,Ny,Nz],Nt_save,[(Nx+1)/2,(Ny+1)/2,Nz],0.15*Nx);
rd_x_v_test = randi([1,ceil(Nx/2)],N_points_v,1);
xyz_t_v_test(:,1) = rd_x_v_test;
U_v_test = zeros(N_samples_test,N_points_v,'single');

rd_x_rd_test = randi([1,Nx],N_points_rd,1,'uint8');
rd_y_rd_test = randi([1,Ny],N_points_rd,1,'uint8');
rd_z_rd_test = randi([1,Nz],N_points_rd,1,'uint8');
rd_t_rd_test = randi([1,Nt_save],N_points_rd,1,'uint8');

rd_x_rd_train = randi([1,Nx],N_points_rd,1,'uint8');
rd_y_rd_train = randi([1,Ny],N_points_rd,1,'uint8');
rd_z_rd_train = randi([1,Nz],N_points_rd,1,'uint8');
rd_t_rd_train = randi([1,Nt_save],N_points_rd,1,'uint8');

xyz_t_ic_train = zeros(N_points_ic,4,'single');
U_ic_train = zeros(N_samples_train,N_points_ic,'single');

xyz_t_ic_test = zeros(N_points_ic,4,'single');
U_ic_test = zeros(N_samples_test,N_points_ic,'single');

rd_x_ic_train = randi([1,Nx],N_points_ic,1,'uint8');
rd_y_ic_train = randi([1,Ny],N_points_ic,1,'uint8');
ic_bias = biased_sampling(N_points_bias,[Nx,Ny,Nz],Nt_save,[(Nx+1)/2,(Ny+1)/2,Nz]);
rd_z_ic_train = ic_bias(:,3);

rd_x_ic_test = randi([1,Nx],N_points_ic,1,'uint8');
rd_y_ic_test = randi([1,Ny],N_points_ic,1,'uint8');
ic_bias = biased_sampling(N_points_bias,[Nx,Ny,Nz],Nt_save,[(Nx+1)/2,(Ny+1)/2,Nz]);
rd_z_ic_test = ic_bias(:,3);

N_bc1 = idivide(N_points_bc,int16(3));
N_bc2 = N_bc1;
N_bc3 = N_points_bc-N_bc1*2;

xyz_t_bc_train = zeros(N_points_bc,4,'single');
U_bc_train = zeros(N_samples_train,N_points_bc,'single');

xyz_t_bc_test = zeros(N_points_bc,4,'single');
U_bc_test = zeros(N_samples_test,N_points_bc,'single');

rd_x_bc_train = datasample([1,2,3,Nx-2,Nx-1,Nx],N_points_bc);
rd_y_bc_train = randi([1,Ny],N_points_bc,1,'uint8');
rd_z_bc_train = randi([1,Nz],N_points_bc,1,'uint8');
rd_t_bc_train = randi([1,Nt_save],N_points_bc,1,'uint8');


rd_x_bc_train(N_bc1+1:N_bc1+N_bc2) = randi([1,Nx],N_bc2,1,'uint8');
rd_y_bc_train(N_bc1+1:N_bc1+N_bc2) = datasample([1,2,3,Ny-2,Ny-1,Ny],N_bc2);

rd_x_bc_train(N_bc1+N_bc2+1:N_points_bc) = randi([1,Nx],N_bc3,1,'uint8');
rd_y_bc_train(N_bc1+N_bc2+1:N_points_bc) = randi([1,Ny],N_bc3,1,'uint8');
rd_z_bc_train(N_bc1+N_bc2+1:N_points_bc) = datasample([1,2,3],N_bc3);
rd_t_bc_train(N_bc1+N_bc2+1:N_points_bc) = randi([1,Nt_save],N_bc3,1,'uint8');


rd_x_bc_test = datasample([1,2,3,Nx-2,Nx-1,Nx],N_points_bc);
rd_y_bc_test = randi([1,Ny],N_points_bc,1,'uint8');
rd_z_bc_test = randi([1,Nz],N_points_bc,1,'uint8');
rd_t_bc_test = randi([1,Nt_save],N_points_bc,1,'uint8');

rd_x_bc_test(N_bc1+1:N_bc1+N_bc2) = randi([1,Nx],N_bc2,1,'uint8');
rd_y_bc_test(N_bc1+1:N_bc1+N_bc2) = datasample([1,2,3,Ny-2,Ny-1,Ny],N_bc2);

rd_x_bc_test(N_bc1+N_bc2+1:N_points_bc) = randi([1,Nx],N_bc3,1,'uint8');
rd_y_bc_test(N_bc1+N_bc2+1:N_points_bc) = randi([1,Ny],N_bc3,1,'uint8');
rd_z_bc_test(N_bc1+N_bc2+1:N_points_bc) = datasample([1,2,3],N_bc3);
rd_t_bc_test(N_bc1+N_bc2+1:N_points_bc) = randi([1,Nt_save],N_bc3,1,'uint8');

for i = 1:N_points_rd
    xyz_t_rd_train(i,:) = [x(rd_x_rd_train(i)),y(rd_y_rd_train(i)),z(rd_z_rd_train(i)),t_save(rd_t_rd_train(i))];
    xyz_t_rd_test(i,:) = [x(rd_x_rd_test(i)),y(rd_y_rd_test(i)),z(rd_z_rd_test(i)),t_save(rd_t_rd_test(i))];
end

for i = 1:N_points_ic
    xyz_t_ic_train(i,:) = [x(rd_x_ic_train(i)),y(rd_y_ic_train(i)),z(rd_z_ic_train(i)),0];
    xyz_t_ic_test(i,:) = [x(rd_x_ic_test(i)),y(rd_y_ic_test(i)),z(rd_z_ic_test(i)),0];
end

for i = 1:N_points_bc
    xyz_t_bc_train(i,:) = [x(rd_x_bc_train(i)),y(rd_y_bc_train(i)),z(rd_z_bc_train(i)),t_save(rd_t_bc_train(i))];
    xyz_t_bc_test(i,:) = [x(rd_x_bc_train(i)),y(rd_y_bc_train(i)),z(rd_z_bc_train(i)),t_save(rd_t_bc_train(i))];
end

n_val = 100;
xyz_t_val = zeros(Nt_save*Nx*Ny*Nz,4,'single');
U_val = zeros(n_val,Nt_save*Nx*Ny*Nz,'single');
U_val_1 = zeros(n_val,Nx,Ny,Nz,Nt_save,'single');
P_val = zeros(n_val,Nt_save-1);
IC_resh_val = zeros(n_val,Nx*Ny);
IC_val = zeros(n_val,Nx*Ny);
v_val = zeros(n_val,Nt_save-1);

i=1;
for ix =1:Nx
    for iy =1:Ny
        for iz=1:Nz
            for it = 1:Nt_save
                xyz_t_val(i,:) = [x(ix),y(iy),z(iz),t_save(it)];
                i = i+1;
            end
        end
    end
end

i_val  = 1;

%%
% Time stepping
for is = 1:N_samples
    disp(['Sample ', num2str(is)]);

    T(:,:,:,:) = 0;
    T(:,:,:,1) = IC(is,:,:,:);

for it = 1:Nt-1
    %disp(it/Nt*100)
    for ix = 2:Nx-1
        for iy = 2:Ny-1
            for iz = 2:Nz

                if iz == Nz % top boundary
                    d2Tdx2 = T(ix+1, iy, Nz, it) - 2*T(ix, iy, Nz, it) + T(ix-1, iy, Nz, it);
                    d2Tdy2 = T(ix, iy+1, Nz, it) - 2*T(ix, iy, Nz, it) + T(ix, iy-1, Nz, it);
                    d2Tdz2 = T(ix, iy, Nz-1, it) - T(ix, iy, Nz, it) + P(is,1+floor((it-1)/each_t)) * Q_rel(ix, iy) * dx/lam;

                    T(ix, iy, Nz, it+1) = T(ix, iy, Nz, it) + ...
                        (lam*dt)/(dx^2*rho*cp) * (d2Tdx2 + d2Tdy2 + d2Tdz2) + ...
                        dt/(2*dx)*v(is,1+floor((it-1)/each_t))*(T(ix+1, iy, Nz, it) - T(ix-1, iy, Nz, it));
                else
                    T(ix, iy, iz, it+1) = T(ix, iy, iz, it) + ...
                        (lam*dt)/(rho*cp*dx^2) * ( ...
                            T(ix+1, iy, iz, it) + T(ix-1, iy, iz, it) + ...
                            T(ix, iy+1, iz, it) + T(ix, iy-1, iz, it) + ...
                            T(ix, iy, iz+1, it) + T(ix, iy, iz-1, it) - ...
                            6 * T(ix, iy, iz, it)) + ...
                        dt/(2*dx) * (T(ix+1, iy, iz, it) - T(ix-1, iy, iz, it)) * v(is,1+floor((it-1)/each_t));
                end

            end 
        end 
    end

    % front boundary
    T(Nx, :, :, it+1) = 300;
    % back boundary 
    T(1, :, :, it+1) = T(2, :, :, it+1);    
    % right boundary 
    T(:, 1, :, it+1) = 300;
    % T(:,1,:,it+1) = T(:,2,:,it+1);
    % left boundary
    T(:, Ny, :, it+1) = 300;
    % T(:,Ny,:,it+1) = T(:,ny-1,:,it+1)
    % bottom boundary
    T(:, :, 1, it+1) = 300;
    %T(:, :, 1, it+1) = T(:, :, 2, it+1);
end
IC_candidates(is,:,:,:) = T(:,:,:,end);
if is ~= N_samples
    if mod(is,10) == 1
        IC(is+1,:,:,:) = T0.*ones(Nx,Ny,Nz);
    else
        IC(is+1,:,:,:) = IC_candidates(randi(is),:,:,:);
    end
end

n_same = 1;

if is ~= N_samples
    if N_samples-is >= n_same+1
        if mod(is,n_same*10) == 1
            IC(is+1:is+n_same,:,:,:) = T0.*ones(n_same,Nx,Ny,Nz);
        elseif mod(is,n_same) == 1
            IC(is+1:is+n_same,:,:,:) = ones(n_same,1) .* IC_candidates(randi(is),:,:,:);
        end
    else
        if mod(is,n_same*10) == 1
            IC(is+1:end,:,:,:) = T0.*ones(N_samples-is,Nx,Ny,Nz);
        elseif mod(is,n_same) == 1
            IC(is+1:end,:,:,:) = ones(N_samples-is,1) .* IC_candidates(randi(is),:,:,:);
        end
    end
end

if is <= N_samples_train
    for i = 1:N_points_bias
        U_bias_train(is,i) = T(xyz_t_bias_train(i,1),xyz_t_bias_train(i,2),xyz_t_bias_train(i,3), t_save_2_t(xyz_t_bias_train(i,4)));
    end

    for i = 1:N_points_v
        U_v_train(is,i) = T(xyz_t_v_train(i,1),xyz_t_v_train(i,2),xyz_t_v_train(i,3), t_save_2_t(xyz_t_v_train(i,4)));
    end
    
    for i = 1:N_points_rd
        U_rd_train(is,i) = T(rd_x_rd_train(i),rd_y_rd_train(i),rd_z_rd_train(i), t_save_2_t(rd_t_rd_train(i)));
    end
    
    for i = 1:N_points_ic
        U_ic_train(is,i) = T(rd_x_ic_train(i),rd_y_ic_train(i),rd_z_ic_train(i),1);
    end
    
    for i = 1:N_points_bc
        U_bc_train(is,i) = T(rd_x_bc_train(i),rd_y_bc_train(i),1, t_save_2_t(rd_t_bc_train(i)));
    end
else
    for i = 1:N_points_bias
        U_bias_test(is-N_samples_train,i) = T(xyz_t_bias_test(i,1),xyz_t_bias_test(i,2),xyz_t_bias_test(i,3),t_save_2_t(xyz_t_bias_test(i,4)));
    end

    for i = 1:N_points_v
        U_v_test(is-N_samples_train,i) = T(xyz_t_v_test(i,1),xyz_t_v_test(i,2),xyz_t_v_test(i,3),t_save_2_t(xyz_t_v_test(i,4)));
    end
    
    for i = 1:N_points_rd
        U_rd_test(is-N_samples_train,i) = T(rd_x_rd_test(i),rd_y_rd_test(i),rd_z_rd_test(i),t_save_2_t(rd_t_rd_test(i)));
    end
    
    for i = 1:N_points_ic
        U_ic_test(is-N_samples_train,i) = T(rd_x_ic_test(i),rd_y_ic_test(i), rd_z_ic_test(i),1);
    end
    
    for i = 1:N_points_bc
        U_bc_test(is-N_samples_train,i) = T(rd_x_bc_train(i),rd_y_bc_train(i),rd_z_bc_train(i),t_save_2_t(rd_t_bc_train(i)));
    end
end
if N_samples-is < n_val
    U_val_1(i_val,:,:,:,:) = T(:,:,:,1:each_t:Nt);
    i_val = i_val+1;
end
end

%% Plot

% Plot the matrix using imagesc
figure; % Open a new figure window
imagesc(reshape(T(:,:,Nz,Nt), Nx, Ny)); % Display the matrix as an image
colorbar; % Add a colorbar to the plot


%% Generate random points

xyz_t_bias_train = [((xyz_t_bias_train(:,1)-1).*dx)+x1, ((xyz_t_bias_train(:,2)-1).*dy)+x1, ((xyz_t_bias_train(:,3)-1).*dz)+z1, (xyz_t_bias_train(:,4)-1).*dt_save];
xyz_t_bias_test = [((xyz_t_bias_test(:,1)-1).*dx)+x1, ((xyz_t_bias_test(:,2)-1).*dy)+x1, ((xyz_t_bias_test(:,3)-1).*dz)+z1, (xyz_t_bias_test(:,4)-1).*dt_save];

xyz_t_v_train = [((xyz_t_v_train(:,1)-1).*dx)+x1, ((xyz_t_v_train(:,2)-1).*dy)+x1, ((xyz_t_v_train(:,3)-1).*dz)+z1, (xyz_t_v_train(:,4)-1).*dt_save];
xyz_t_v_test = [((xyz_t_v_test(:,1)-1).*dx)+x1, ((xyz_t_v_test(:,2)-1).*dy)+x1, ((xyz_t_v_test(:,3)-1).*dz)+z1, (xyz_t_v_test(:,4)-1).*dt_save];

P_train = P(1:N_samples_train,:);
P_test = P(end-N_samples_test+1:end,:);

v_train = v(1:N_samples_train,:);
v_test = v(end-N_samples_test+1:end,:);

IC_resh = reshape(IC(:,:,:,Nz),[N_samples,Nx*Ny]);

IC_train = IC_resh(1:N_samples_train,:);
IC_test = IC_resh(end-N_samples_test+1:end,:);



i=1;
for ix =1:Nx
    for iy =1:Ny
        for iz=1:Nz
            for it = 1:Nt_save
                xyz_t_val(i,:) = [x(ix),y(iy),z(iz),t(1+(it-1)*each_t)];
                U_val(:,i) = U_val_1(:,ix,iy,iz,it);
                i = i+1;
            end
        end
    end
end

i_val = 1;
for is = N_samples-n_val+1:N_samples
     P_val(i_val,:) = P(is,:);
     IC_val(i_val,:) = IC_resh(is,:);
     v_val(i_val,:) = v(is,:);
     i_val = i_val +1;
end


%% Validierung der generierten Punkte
varNames = {'x', 'y', 'z', 't'}; % Namen der Variablen in Reihenfolge

A = xyz_t_val;

figure;
for i = 1:4
    subplot(2,2,i);
    % Alle eindeutigen Werte und ihre Häufigkeit
    [uniqueVals, ~, idx] = unique(A(:,i));
    counts = accumarray(idx, 1);
    % Balkendiagramm
    bar(uniqueVals, counts, 'FaceColor', [0.2 0.6 0.8]);
    % Titel mit Variablennamen
    title(varNames{i});
    xlabel('Wert');
    ylabel('Anzahl');
    grid on;
end
sgtitle('Häufigkeitsverteilung Validierungsdaten');

A = xyz_t_bias_train;

figure;
for i = 1:4
    subplot(2,2,i);
    % Alle eindeutigen Werte und ihre Häufigkeit
    [uniqueVals, ~, idx] = unique(A(:,i));
    counts = accumarray(idx, 1);
    % Balkendiagramm
    bar(uniqueVals, counts, 'FaceColor', [0.2 0.6 0.8]);
    % Titel mit Variablennamen
    title(varNames{i});
    xlabel('Wert');
    ylabel('Anzahl');
    grid on;
end
sgtitle('Häufigkeitsverteilung Trainingsdaten "bias"');

A = xyz_t_bc_train;

figure;
for i = 1:4
    subplot(2,2,i);
    % Alle eindeutigen Werte und ihre Häufigkeit
    [uniqueVals, ~, idx] = unique(A(:,i));
    counts = accumarray(idx, 1);
    % Balkendiagramm
    bar(uniqueVals, counts, 'FaceColor', [0.2 0.6 0.8]);
    % Titel mit Variablennamen
    title(varNames{i});
    xlabel('Wert');
    ylabel('Anzahl');
    grid on;
end
sgtitle('Häufigkeitsverteilung Trainingsdaten "bc"');

A = xyz_t_ic_train;

figure;
for i = 1:4
    subplot(2,2,i);
    % Alle eindeutigen Werte und ihre Häufigkeit
    [uniqueVals, ~, idx] = unique(A(:,i));
    counts = accumarray(idx, 1);
    % Balkendiagramm
    bar(uniqueVals, counts, 'FaceColor', [0.2 0.6 0.8]);
    % Titel mit Variablennamen
    title(varNames{i});
    xlabel('Wert');
    ylabel('Anzahl');
    grid on;
end
sgtitle('Häufigkeitsverteilung Trainingsdaten "ic"');




%%
figure;
scatter3(xyz_t_bias_test(:,1), xyz_t_bias_test(:,2), xyz_t_bias_test(:,3), 36, U_bias_test(1,:), "filled")
colorbar;                     % Farbleiste anzeigen
colormap jet;                 % Heatmap-Farben (oder 'parula', 'hot', etc.)

figure;
scatter3(xyz_t_rd_test(:,1), xyz_t_rd_test(:,2), xyz_t_rd_test(:,3), 36, U_rd_test(1,:), "filled")
colorbar;                     % Farbleiste anzeigen
colormap jet;                 % Heatmap-Farben (oder 'parula', 'hot', etc.)

figure;
scatter3(xyz_t_v_test(:,1), xyz_t_v_test(:,2), xyz_t_v_test(:,3), 36, U_v_test(1,:), "filled")
colorbar;                     % Farbleiste anzeigen
colormap jet;                 % Heatmap-Farben (oder 'parula', 'hot', etc.)

figure;
scatter3(xyz_t_bc_test(:,1), xyz_t_bc_test(:,2), xyz_t_bc_test(:,3), 36, U_bc_test(1,:), "filled")
colorbar;                     % Farbleiste anzeigen
colormap jet;                 % Heatmap-Farben (oder 'parula', 'hot', etc.)

figure;
scatter3(xyz_t_ic_test(:,1), xyz_t_ic_test(:,2), xyz_t_ic_test(:,3), 36, U_ic_test(1,:), "filled")
colorbar;                     % Farbleiste anzeigen
colormap jet;                 % Heatmap-Farben (oder 'parula', 'hot', etc.)

%%
plot_Umax_vs_P(U_bias_train, P_train, xyz_t_bias_train)
plot_Umax_vs_P(U_bias_test, P_test, xyz_t_bias_test)
plot_Umax_vs_P(U_val, P_val, xyz_t_val)

plot_Umax_vs_PV_heatmap(U_bias_train, P_train, v_train, xyz_t_bias_train)
plot_Umax_vs_PV_heatmap(U_bias_test, P_test, v_test, xyz_t_bias_test)
plot_Umax_vs_PV_heatmap(U_val, P_val, v_val, xyz_t_val)


%%
disp('Start saving ...')

save('annular_lin_val_20%_vcost.mat', ...
    'P_test','P_train','P_val',"IC_train","IC_test","IC_val","v_test","v_train","v_val", ...
    "U_val","U_bc_test","U_bc_train","U_ic_test","U_ic_train", ...
    "U_rd_test","U_rd_train","U_bias_test","U_bias_train","U_v_test","U_v_train", ...
    "xyz_t_val","xyz_t_bc_test","xyz_t_bc_train","xyz_t_ic_test","xyz_t_ic_train", ...
    "xyz_t_rd_test","xyz_t_rd_train","xyz_t_bias_test","xyz_t_bias_train","xyz_t_v_test","xyz_t_v_train",'-v7.3');

disp('saving complete')

%%

function plot_Umax_vs_P(U, P, xyz_t)
    t_vals = xyz_t(:,4);
    valid_time_mask = t_vals > 0;
    idx_valid_points = find(valid_time_mask);
    t_vals_filtered = t_vals(idx_valid_points);
    t_indices = round(t_vals_filtered / 0.05);

    U_bias_filtered = U(:, idx_valid_points);

    U_max_cell = cell(1, 20);
    P_cell = cell(1, 20);

    for t_idx = 1:20
        t_mask = (t_indices == t_idx);
        U_t = U_bias_filtered(:, t_mask);
        U_max = max(U_t, [], 2);
        P_t = P(:, t_idx);

        U_max_cell{t_idx} = U_max;
        P_cell{t_idx} = P_t;
    end

    U_max_all = vertcat(U_max_cell{:});
    P_all = vertcat(P_cell{:});

    figure;
    scatter(P_all, U_max_all, 'filled');
    xlabel('P\_test (pro Zeitstufe)');
    ylabel('max(U) (t > 0)');
    title('U\_max über P für alle t > 0');
    grid on;
end

function plot_Umax_vs_PV_heatmap(U, P, V, xyz_t)
    % For all time steps, plot P vs V, color is mean(max(U)), per unique (P,V)
    % U: [n_samples x n_points], P/V: [n_samples x n_time], xyz_t: [n_points x 4]

    t_vals = xyz_t(:,4);
    valid_time_mask = t_vals > 0;
    idx_valid_points = find(valid_time_mask);
    t_vals_filtered = t_vals(idx_valid_points);
    t_indices = round(t_vals_filtered / 0.05);

    U_filtered = U(:, idx_valid_points);
    n_t = size(P,2); % Number of time steps

    all_P = [];
    all_V = [];
    all_U = [];

    for t_idx = 1:n_t
        t_mask = (t_indices == t_idx);
        U_t = U_filtered(:, t_mask);
        U_max = max(U_t, [], 2);  % Max over space for each sample at this t
        P_t = P(:, t_idx);
        V_t = V(:, t_idx);

        all_P = [all_P; P_t(:)];
        all_V = [all_V; V_t(:)];
        all_U = [all_U; U_max(:)];
    end

    % Find unique (P,V) pairs and their mean U_max
    data = [all_P, all_V, all_U];
    [uniq_pairs, ~, ic] = unique(data(:,1:2), 'rows');
    mean_U = accumarray(ic, data(:,3), [], @mean);

    % Plot
    figure;
    scatter(uniq_pairs(:,1), uniq_pairs(:,2), 70, mean_U, 'filled');
    colormap jet; colorbar;
    xlabel('P');
    ylabel('V');
    title('Mean max(U) over all timesteps for unique (P,V)');
    grid on;
end


