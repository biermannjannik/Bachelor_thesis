function mat_prop = mat_prop_eval_mat(mat_prop_fit, T_ACT, T_thres, mat_prop_thres)
    % Preallocate the output matrix with the same size as T_ACT
    mat_prop = zeros(size(T_ACT));
    
    % Logical index for where T_ACT <= T_thres
    idx = T_ACT <= T_thres;
    
    % Apply the fit function where condition is true
    mat_prop(idx) = mat_prop_fit(T_ACT(idx));
    
    % Assign threshold value where condition is false
    mat_prop(~idx) = mat_prop_thres;
end
