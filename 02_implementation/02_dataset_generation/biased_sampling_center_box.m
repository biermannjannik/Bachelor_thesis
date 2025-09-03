function samples = biased_sampling_center_box(n_samples, grid_shape, n_timesteps, bias_center, sigma, x_offset)
    % biased_sampling_center_box: Gaussian spatial bias, always including all [4,4,2] box
    % around round(bias_center) (z only in negative direction), with x_offset.
    %
    % Inputs:
    %   n_samples   - Total number of samples (>= must-have set)
    %   grid_shape  - [Nx, Ny, Nz]
    %   n_timesteps - Number of time steps
    %   bias_center - [cx, cy, cz], spatial bias center (can be non-integer)
    %   sigma       - Stddev for spatial Gaussian
    %   x_offset    - Offset for x center
    %
    % Output:
    %   samples     - n_samples x 4 matrix [x_idx, y_idx, z_idx, t_idx]
    if nargin < 2, grid_shape = [20, 20, 10]; end

    if nargin < 6, x_offset = 0; end
    if nargin < 5, sigma = 3.0; end
    if nargin < 4, bias_center = [ceil(grid_shape(1)/2), ceil(grid_shape(2)/2), grid_shape(3)]; end
    if nargin < 3, n_timesteps = 21; end
    if nargin < 2, grid_shape = [20, 20, 10]; end
    if nargin < 1, n_samples = 1000; end

    Nx = grid_shape(1);
    Ny = grid_shape(2);
    Nz = grid_shape(3);

    % For the box: round the bias center (indices)
    cx = round(bias_center(1)) + x_offset;
    cy = round(bias_center(2));
    cz = round(bias_center(3));

    % x/y box: [4,4] (centered), z: [2] (downwards only)
    x_box = max(1, cx-2) : min(Nx, cx+1);
    y_box = max(1, cy-2) : min(Ny, cy+1);
    z_box = max(1, cz-1) : cz; % two layers: center and one below

    % All combinations in the box for all t
    [xg, yg, zg, tg] = ndgrid(x_box, y_box, z_box, 1:n_timesteps);
    must_have = [xg(:), yg(:), zg(:), tg(:)];
    n_must = size(must_have, 1);

    if n_samples < n_must
        error('Number of samples requested (%d) is less than required must-have samples (%d).', n_samples, n_must);
    end

    % Create full grid for Gaussian bias
    [X, Y, Z, T] = ndgrid(1:Nx, 1:Ny, 1:Nz, 1:n_timesteps);
    X = X(:); Y = Y(:); Z = Z(:); T = T(:);

    % Gaussian bias: can use non-integer bias_center
    dists = sqrt((X - bias_center(1)).^2 + (Y - bias_center(2)).^2 + (Z - bias_center(3)).^2);
    probs = exp(-0.5 * (dists / sigma).^2);

    % Exclude must-have indices from random selection
    all_idx = sub2ind([Nx,Ny,Nz,n_timesteps], must_have(:,1), must_have(:,2), must_have(:,3), must_have(:,4));
    mask = true(length(probs),1);
    mask(all_idx) = false;

    probs(~mask) = 0;
    probs = probs / sum(probs);

    n_left = n_samples - n_must;
    if n_left > 0
        idx_rest = weighted_sample_no_replacement(probs, n_left);
        [x_r, y_r, z_r, t_r] = ind2sub([Nx,Ny,Nz,n_timesteps], idx_rest);
        samples = [must_have; [x_r, y_r, z_r, t_r]];
    else
        samples = must_have;
    end
end

function idx = weighted_sample_no_replacement(weights, n)
    weights = weights / sum(weights);
    cdf = cumsum(weights);
    idx = zeros(n, 1);
    taken = false(length(weights), 1);
    i = 1;
    while i <= n
        r = rand();
        k = find(cdf >= r, 1, 'first');
        if ~taken(k)
            idx(i) = k;
            taken(k) = true;
            i = i + 1;
        end
    end
end
