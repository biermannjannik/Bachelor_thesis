function samples = biased_sampling(n_samples, grid_shape, n_timesteps, bias_center, sigma)
    % biased_sampling: Samples spatiotemporal points with Gaussian bias toward bias_center
    % 
    % Inputs:
    %   n_samples   - Number of samples to draw
    %   grid_shape  - [Nx, Ny, Nz] spatial grid dimensions
    %   n_timesteps - Number of time steps
    %   bias_center - [cx, cy, cz] center of spatial bias
    %   sigma       - standard deviation of Gaussian bias
    %
    % Output:
    %   samples     - n_samples x 4 matrix [x, y, z, t]

    if nargin < 1, n_samples = 1000; end
    if nargin < 2, grid_shape = [20, 20, 10]; end
    if nargin < 3, n_timesteps = 20; end
    if nargin < 4, bias_center = [10.5, 10.5, 10]; end
    if nargin < 5, sigma = 3.0; end

    % Create grid
    [X, Y, Z, T] = ndgrid(1:grid_shape(1), 1:grid_shape(2), 1:grid_shape(3), 1:n_timesteps);
    X = X(:); Y = Y(:); Z = Z(:); T = T(:);
    
    % Compute distances to bias center (spatial only)
    dists = sqrt((X - bias_center(1)).^2 + ...
                 (Y - bias_center(2)).^2 + ...
                 (Z - bias_center(3)).^2);

    % Gaussian weights
    probs = exp(-0.5 * (dists / sigma).^2);
    probs = probs / sum(probs);  % normalize

    % Sample indices with bias
    idx = weighted_sample_no_replacement(probs, n_samples);

    % Assemble output
    samples = [X(idx), Y(idx), Z(idx), T(idx)];
end

function idx = weighted_sample_no_replacement(weights, n)
    weights = weights / sum(weights);  % Normalisieren
    cdf = cumsum(weights);
    r = rand(n, 1);
    idx = zeros(n, 1);

    % Ziehen ohne Duplikate
    taken = false(length(weights), 1);
    for i = 1:n
        k = find(cdf >= r(i), 1, 'first');
        while taken(k)  % falls schon gewählt, ziehe neu
            r(i) = rand();
            k = find(cdf >= r(i), 1, 'first');
        end
        idx(i) = k;
        taken(k) = true;
    end
end
