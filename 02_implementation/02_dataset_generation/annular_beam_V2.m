% Calculates the intensity distrubution of an annular laser beam on a plane.

% from Paper:
% "Bessel and annular beams for materials processing" Duocastella 2012

% INPUTS ------------------------------------------------------------------
% P         := [W] laser power [1 x 1]
% A_b       := [-] absorption coefficient [1 x 1]
% r_max     := [m] position of I_max [1 x 1]
% w         := [m] beam waist [1 x 1], or beam radius/effective radius
% r         := [m] radius coordinate 

% OUTPUTS -----------------------------------------------------------------
% I     := [W/m^2] intensity distribution [N_x x N_y]

% Explanation -------------------------------------------------------------
% ...

function I = annular_beam_V2(P,A_b,w,r_max,r)
    
    A_eff = pi*((r_max+w/2)^2 - (r_max-w/2)^2);
    I_max = (A_b*P)/A_eff;
    I = I_max*exp(-2*(r-r_max)^2/w^2);

end 