function signal = generate_random_signal_20ramp(n_samples, y_min, y_max, c_min, c_max)
%GENERATE_RANDOM_SIGNALV3  Piecewise-constant random signal with bounded step size.
% Each subsequent segment deviates by at most 20% from the previous segment value.
%
% Parameters are the same as V2:
%   n_samples ... total number of samples
%   y_min, y_max ... allowed signal range
%   c_min, c_max ... segment length range (in samples)

    % Preallocate
    signal = zeros(1, n_samples);

    % Helper: clamp x to [a,b]
    clamp = @(x,a,b) min(max(x,a),b);

    % First segment value is free within [y_min, y_max]
    i = 1;
    value_prev = y_min + (y_max - y_min) * rand();

    while i <= n_samples
        % Random duration in [c_min, c_max]
        duration = randi([c_min, c_max]);

        % End index without exceeding n_samples
        end_idx = min(i + duration - 1, n_samples);

        % Fill section with the current value
        signal(i:end_idx) = value_prev;

        % Prepare next start index
        i = end_idx + 1;
        if i > n_samples
            break;
        end

        % ---- Choose the next segment's value with <=20% deviation ----
        % Allowed deviation (relative to previous segment's magnitude)
        eps_rel = 1e-12; % prevents getting stuck when value_prev == 0
        band = 0.20 * max(abs(value_prev), eps_rel);

        lo = value_prev - band;
        hi = value_prev + band;

        % Respect global bounds
        lo = max(lo, y_min);
        hi = min(hi, y_max);

        % If numerical edge cases collapse the interval, fall back to the nearest bound
        if hi <= lo
            next_value = lo; % (== hi)
        else
            % Uniform in the intersected interval
            next_value = lo + (hi - lo) * rand();
        end

        value_prev = next_value;
    end
end
