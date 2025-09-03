function mat_prop = mat_prop_eval_mat2(mat_prop_fit, T_ACT, T_thres, mat_prop_thres, T_thres_ll, mat_prop_thres_ll)
    % Preallocate the output matrix with the same size as T_ACT
    mat_prop = zeros(size(T_ACT));
    
    % Logical index for where T_ACT <= T_thres
    idx = T_ACT < T_thres & T_ACT > T_thres_ll;

    idx_high = T_ACT >= T_thres;
    idx_low = T_ACT <= T_thres_ll;

    
    % Apply the fit function where condition is true
    mat_prop(idx) = mat_prop_fit(T_ACT(idx));
    
    % Assign threshold value where condition is false
    mat_prop(idx_high) = mat_prop_thres;
    mat_prop(idx_low) = mat_prop_thres_ll;

end
