function signal = generate_random_signal(n_samples, y_min, y_max, c_min, c_max)
    signal = zeros(1, n_samples);
    i = 1;
    while i <= n_samples
        % Zufälliger Wert im Bereich [y_min, y_max]
        value = y_min + (y_max - y_min) * rand();
        
        % Zufällige Dauer zwischen c_min und c_max
        duration = randi([c_min, c_max]);
        
        % Endindex berechnen, ohne n_samples zu überschreiten
        end_idx = min(i + duration - 1, n_samples);
        
        % Abschnitt mit konstantem Wert füllen
        signal(i:end_idx) = value;
        
        % Nächster Startindex
        i = end_idx + 1;
    end
end
