function samples = biased_sampling_center_box_annular(n_samples, grid_shape, n_timesteps, bias_center, sigma, x_offset)
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

    Nx = grid_shape(1); Ny = grid_shape(2); Nz = grid_shape(3);

    % ---- must-have box (uses rounded indices) ----
    cx = round(bias_center(1)) + x_offset;
    cy = round(bias_center(2));
    cz = round(bias_center(3));
    cz = min(max(cz,1), Nz);

    % top: 6x6 , bottom: 4x4
    x_top = max(1, cx-3) : min(Nx, cx+2);
    y_top = max(1, cy-3) : min(Ny, cy+2);

    x_bot = max(1, cx-2) : min(Nx, cx+1);
    y_bot = max(1, cy-2) : min(Ny, cy+1);

    [x1,y1,z1,t1] = ndgrid(x_top, y_top, cz,   1:n_timesteps);
    if cz > 1
        [x2,y2,z2,t2] = ndgrid(x_bot, y_bot, cz-1, 1:n_timesteps);
        must_have = [x1(:), y1(:), z1(:), t1(:);
                     x2(:), y2(:), z2(:), t2(:)];
    else
        must_have = [x1(:), y1(:), z1(:), t1(:)];
    end

    n_must = size(must_have, 1);
    if n_samples < n_must
        error('Number of samples requested (%d) is less than required must-have samples (%d).', n_samples, n_must);
    end

    % ---- Gaussian bias on full 4D lattice (replicated over t) ----
    [X, Y, Z, T] = ndgrid(1:Nx, 1:Ny, 1:Nz, 1:n_timesteps); %#ok<ASGLU>
    center_cont = [bias_center(1) + x_offset, bias_center(2), bias_center(3)]; % non-integer allowed
    dists = sqrt( (X - center_cont(1)).^2 + ...
                  (Y - center_cont(2)).^2 + ...
                  (Z - center_cont(3)).^2 );

    probs = exp(-0.5 * (dists / sigma).^2);
    probs = probs(:); % vectorize for masking/sampling

    % Exclude must-have indices from the random pool
    all_idx_must = sub2ind([Nx,Ny,Nz,n_timesteps], must_have(:,1), must_have(:,2), must_have(:,3), must_have(:,4));
    mask = true(numel(probs),1);
    mask(all_idx_must) = false;

    probs(~mask) = 0;
    s = sum(probs);
    if s == 0
        % fallback: uniform over the remaining cells if Gaussian mass is entirely masked (edge case)
        probs(mask) = 1 / nnz(mask);
    else
        probs = probs / s;
    end

    % ---- sample the rest without replacement ----
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
    % Simple roulette-wheel w/o replacement; assumes weights sum to 1 and >=0
    weights = weights(:);
    idx = zeros(n, 1);
    taken = false(length(weights), 1);
    for i = 1:n
        % re-normalize remaining mass each draw
        w = weights;
        w(taken) = 0;
        s = sum(w);
        if s == 0
            error('Not enough remaining probability mass to sample %d items.', n);
        end
        w = w / s;
        cdf = cumsum(w);
        r = rand();
        k = find(cdf >= r, 1, 'first');
        idx(i) = k;
        taken(k) = true;
    end
end
