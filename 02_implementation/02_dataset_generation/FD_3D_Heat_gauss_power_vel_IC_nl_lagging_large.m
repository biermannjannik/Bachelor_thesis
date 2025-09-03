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

%% PARAMETER/INPUT

% function fit of material properties from script: 
run("thermal_properties_SS304L_SS316L.m");

% mat props 
cp_fit = cp_304_fit; 
lam_fit = lam_304_fit; 
rho_fit = rho_304_fit;

T_lam_thres = T(end); 
lam_thres = lam_304(end);
T_lam_thres_ll = T(1);
lam_thres_ll = lam_304(1);
T_cp_thres = T(end);
cp_thres = cp_304(end);
T_cp_thres_ll = T(1);
cp_thres_ll = cp_304(1);
T_rho_thres = T(end); 
rho_thres = rho_304(end);
T_rho_thres_ll = T(1); 
rho_thres_ll = rho_304(1);

% % linear material properties 
% lam = interp1(Tvec_lamvec,lamvec,Tm);           % [W/(m*K)] thermal conductivity
% rho = interp1(Tvec_rhovec,rhovec,Tm);     % [kg/m^3] density
% cp = interp1(Tvec_cpvec,cpvec,Tm);        % [J/(kg*K)] specific heat
% a = lam/(rho*cp);                                       % [m2/s] thermal diffusivity

A_b = 0.4;

t0 = 0;             % [s] begin time  
te = 1;             % [s] end time 

x1 = -4e-3;         % [m] begin of domain x and y
x2 = 4e-3;          % [m] end of domain x and y
z1 = -4e-3;             % [m] begin of domain z
z2 = 0;          % [m] end of domain z

T0 = 300;           % [K] initial temperature

Nx = 40; Ny = 40; Nz = 20;  % 3D grid size
Nt = 1001;                  % Time steps

N_samples = 10000;

% Discretization
dx = (x2-x1) / (Nx - 1);
dy = dx;
dz = (z2-z1) / (Nz - 1);
dt = (te-t0) / (Nt - 1);
x = linspace(x1,x2, Nx);
y = x;
z = linspace(z1,z2, Nz);
t = linspace(0, (te-t0), Nt);

d_T_Fo = 10;
n_T_Fo = (T(end)-T0)/d_T_Fo+1;
T_Fo = T0:d_T_Fo:T(end);
Fo = zeros(n_T_Fo);

for i = 1:n_T_Fo
    % EVALUATION WITH FITTED FUNCTION
    lam = mat_prop_eval2(lam_fit,T_Fo(i),T_lam_thres,lam_thres);
    cp = mat_prop_eval2(cp_fit,T_Fo(i),T_cp_thres,cp_thres);
    rho = mat_prop_eval2(rho_fit,T_Fo(i),T_rho_thres,rho_thres);

    Fo(i) = (lam*dt)/(rho*cp*dx^2);
end
figure;
plot(Fo);

% condition for spatial and time discretization
if (1-6*max(Fo)) < 0
    error("Adjust spatial and time discretization.")
end 

% meshgrid
[X,Y] = meshgrid(x,y);


% Heat source Q: [N_samples x Nx x Ny x Nz]
Q_rel = zeros(Nx, Ny, 'single');

mu = [x1+(x2-x1)/2,x1+(x2-x1)/2];
sigma =  0.0005;
A_eff = sigma^2*10;
I_max_rel = (2*A_b)/A_eff;

% 3D Gaussian
Q_rel(:,:) = I_max_rel * exp(-((X - mu(1)).^2 + (Y - mu(2)).^2 )/ (2 * sigma^2));


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

N_points_bias = 16000;
N_points_v = 4000;
N_points_rd = 4000;
N_points_ic = 2000;
N_points_bc = 2000;

xyz_t_bias_train = biased_sampling_center_box(N_points_bias,[Nx,Ny,Nz],Nt_save,[(Nx+1)/2,(Ny+1)/2,Nz],0.15*Nx,-1);
U_bias_train = zeros(N_samples_train,N_points_bias,'single');

xyz_t_bias_test = biased_sampling_center_box(N_points_bias,[Nx,Ny,Nz],Nt_save,[(Nx+1)/2,(Ny+1)/2,Nz],0.15*Nx,-1);
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

    % Evaluate material properties
    T_current = T(:,:,:,it);
    LAM = mat_prop_eval_mat2(lam_fit, T_current, T_lam_thres, lam_thres, T_lam_thres_ll, lam_thres_ll);
    CP = mat_prop_eval_mat2(cp_fit, T_current, T_cp_thres, cp_thres, T_cp_thres_ll, cp_thres_ll);
    RHO = mat_prop_eval_mat2(rho_fit, T_current, T_rho_thres, rho_thres, T_rho_thres_ll, rho_thres_ll);

    % Precompute coefficient
    coeff = (LAM .* dt) ./ (RHO .* CP * dx^2);

    for ix = 2:Nx-1
        for iy = 2:Ny-1
            for iz = 2:Nz

                if iz == Nz % top boundary
                    d2Tdx2 = T(ix+1, iy, Nz, it) - 2*T(ix, iy, Nz, it) + T(ix-1, iy, Nz, it);
                    d2Tdy2 = T(ix, iy+1, Nz, it) - 2*T(ix, iy, Nz, it) + T(ix, iy-1, Nz, it);
                    d2Tdz2 = T(ix, iy, Nz-1, it) - T(ix, iy, Nz, it) + P(is,1+floor((it-1)/each_t)) * Q_rel(ix, iy) * dx/LAM(ix,iy,iz);

                    T(ix, iy, Nz, it+1) = T(ix, iy, Nz, it) + ...
                        coeff(ix,iy,iz) * (d2Tdx2 + d2Tdy2 + d2Tdz2) + ...
                        dt/(2*dx)*v(is,1+floor((it-1)/each_t))*(T(ix+1, iy, Nz, it) - T(ix-1, iy, Nz, it));
                else
                    T(ix, iy, iz, it+1) = T(ix, iy, iz, it) + ...
                        coeff(ix,iy,iz) * ( ...
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

% %%
% % Time stepping
% for is = 1:N_samples
%     disp(['Sample ', num2str(is)]);
% 
%     T(:,:,:,:) = 0;
%     T(:,:,:,1) = IC(is,:,:,:);
% 
% for it = 1:Nt-1
%     % Evaluate material properties
%     T_current = T(:,:,:,it);
%     LAM = mat_prop_eval_mat2(lam_fit, T_current, T_lam_thres, lam_thres, T_lam_thres_ll, lam_thres_ll);
%     CP = mat_prop_eval_mat2(cp_fit, T_current, T_cp_thres, cp_thres, T_cp_thres_ll, cp_thres_ll);
%     RHO = mat_prop_eval_mat2(rho_fit, T_current, T_rho_thres, rho_thres, T_rho_thres_ll, rho_thres_ll);
% 
%     % Precompute coefficient
%     coeff = (LAM .* dt) ./ (RHO .* CP * dx^2);
% 
%     % Compute 3D Laplacian using finite difference (excluding boundaries)
%     % Symmetric padding of T_current by 1 layer in all directions
%     T_pad = padarray(T_current, [1 1 1], 'replicate');
% 
%     % Compute 3D Laplacian (central differences)
%     Laplacian = ...
%         T_pad(3:end, 2:end-1, 2:end-1) + ...  % T(i+1,j,k)
%         T_pad(1:end-2, 2:end-1, 2:end-1) + ... % T(i-1,j,k)
%         T_pad(2:end-1, 3:end, 2:end-1) + ...  % T(i,j+1,k)
%         T_pad(2:end-1, 1:end-2, 2:end-1) + ... % T(i,j-1,k)
%         T_pad(2:end-1, 2:end-1, 3:end) + ...  % T(i,j,k+1)
%         T_pad(2:end-1, 2:end-1, 1:end-2) - ... % T(i,j,k-1)
%         6 * T_current;
% 
% 
%     % Upwind convection term (x-direction velocity component assumed)
%     vx = v(is, 1+floor((it-1)/each_t));
%     conv_term = dt/(2*dx) * vx .* ( ...
%         circshift(T_current, [-1 0 0]) - circshift(T_current, [1 0 0]) );
% 
%     % Top boundary needs special handling
%     T_top = T_current(:,:,Nz);
%     Q_term = P(is, 1+floor((it-1)/each_t)) .* Q_rel * dx ./ LAM(:,:,Nz);
%     d2Tdz2_top = T_current(:,:,Nz-1) - T_top + Q_term;
% 
%     % Update interior values (excluding top boundary)
%     T_new = T_current + coeff .* Laplacian + conv_term;
% 
%     % Update top boundary (iz = Nz)
%     d2Tdx2_top = circshift(T_top, [-1 0]) - 2*T_top + circshift(T_top, [1 0]);
%     d2Tdy2_top = circshift(T_top, [0 -1]) - 2*T_top + circshift(T_top, [0 1]);
%     T_new(:,:,Nz) = T_top + ...
%         (LAM(:,:,Nz)*dt)./(dx^2 * RHO(:,:,Nz) .* CP(:,:,Nz)) .* ...
%         (d2Tdx2_top + d2Tdy2_top + d2Tdz2_top) + ...
%         dt/(2*dx) * vx .* (circshift(T_top, [-1 0]) - circshift(T_top, [1 0]));
% 
%     % Assign result
%     T(:,:,:,it+1) = T_new;
% 
%     % Boundary Conditions
%     T(Nx,:,:,it+1) = 300;             % Front
%     T(1,:,:,it+1) = T(2,:,:,it+1);    % Back
%     T(:,1,:,it+1) = 300;              % Right
%     % T(:,1,:,it+1) = T(:,2,:,it+1);
%     T(:,Ny,:,it+1) = 300;             % Left
%     % T(:,Ny,:,it+1) = T(:,ny-1,:,it+1)
%     T(:,:,1,it+1) = 300;              % Bottom
%     %T(:, :, 1, it+1) = T(:, :, 2, it+1);
% end
% 
% IC_candidates(is,:,:,:) = T(:,:,:,end);
% if is ~= N_samples
%     if mod(is,10) == 1
%         IC(is+1,:,:,:) = T0.*ones(Nx,Ny,Nz);
%     else
%         IC(is+1,:,:,:) = IC_candidates(randi(is),:,:,:);
%     end
% end
% 
% n_same = 1;
% 
% if is ~= N_samples
%     if N_samples-is >= n_same+1
%         if mod(is,n_same*10) == 1
%             IC(is+1:is+n_same,:,:,:) = T0.*ones(n_same,Nx,Ny,Nz);
%         elseif mod(is,n_same) == 1
%             IC(is+1:is+n_same,:,:,:) = ones(n_same,1) .* IC_candidates(randi(is),:,:,:);
%         end
%     else
%         if mod(is,n_same*10) == 1
%             IC(is+1:end,:,:,:) = T0.*ones(N_samples-is,Nx,Ny,Nz);
%         elseif mod(is,n_same) == 1
%             IC(is+1:end,:,:,:) = ones(N_samples-is,1) .* IC_candidates(randi(is),:,:,:);
%         end
%     end
% end
% 
% if is <= N_samples_train
%     for i = 1:N_points_bias
%         U_bias_train(is,i) = T(xyz_t_bias_train(i,1),xyz_t_bias_train(i,2),xyz_t_bias_train(i,3), t_save_2_t(xyz_t_bias_train(i,4)));
%     end
% 
%     for i = 1:N_points_v
%         U_v_train(is,i) = T(xyz_t_v_train(i,1),xyz_t_v_train(i,2),xyz_t_v_train(i,3), t_save_2_t(xyz_t_v_train(i,4)));
%     end
% 
%     for i = 1:N_points_rd
%         U_rd_train(is,i) = T(rd_x_rd_train(i),rd_y_rd_train(i),rd_z_rd_train(i), t_save_2_t(rd_t_rd_train(i)));
%     end
% 
%     for i = 1:N_points_ic
%         U_ic_train(is,i) = T(rd_x_ic_train(i),rd_y_ic_train(i),rd_z_ic_train(i),1);
%     end
% 
%     for i = 1:N_points_bc
%         U_bc_train(is,i) = T(rd_x_bc_train(i),rd_y_bc_train(i),1, t_save_2_t(rd_t_bc_train(i)));
%     end
% else
%     for i = 1:N_points_bias
%         U_bias_test(is-N_samples_train,i) = T(xyz_t_bias_test(i,1),xyz_t_bias_test(i,2),xyz_t_bias_test(i,3),t_save_2_t(xyz_t_bias_test(i,4)));
%     end
% 
%     for i = 1:N_points_v
%         U_v_test(is-N_samples_train,i) = T(xyz_t_v_test(i,1),xyz_t_v_test(i,2),xyz_t_v_test(i,3),t_save_2_t(xyz_t_v_test(i,4)));
%     end
% 
%     for i = 1:N_points_rd
%         U_rd_test(is-N_samples_train,i) = T(rd_x_rd_test(i),rd_y_rd_test(i),rd_z_rd_test(i),t_save_2_t(rd_t_rd_test(i)));
%     end
% 
%     for i = 1:N_points_ic
%         U_ic_test(is-N_samples_train,i) = T(rd_x_ic_test(i),rd_y_ic_test(i), rd_z_ic_test(i),1);
%     end
% 
%     for i = 1:N_points_bc
%         U_bc_test(is-N_samples_train,i) = T(rd_x_bc_train(i),rd_y_bc_train(i),rd_z_bc_train(i),t_save_2_t(rd_t_bc_train(i)));
%     end
% end
% if N_samples-is < n_val
%     U_val_1(i_val,:,:,:,:) = T(:,:,:,1:each_t:Nt);
%     i_val = i_val+1;
% end
% 
% end

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
for is = N_samples-99:N_samples
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

%figure;
%scatter3(xyz_t_v_test(:,1), xyz_t_v_test(:,2), xyz_t_v_test(:,3), 36, U_v_test(1,:), "filled")
%colorbar;                     % Farbleiste anzeigen
%colormap jet;                 % Heatmap-Farben (oder 'parula', 'hot', etc.)

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

save('gauss_P_v_IC_vvar_nl_10k_16k_40x.mat', ...
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
