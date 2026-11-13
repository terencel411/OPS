phi = @(r) ...
    (abs(r) <= 0.5) .* (1/3*(1 + sqrt(1 - 3*r.^2))) + ...
    (abs(r) > 0.5 & abs(r) <= 1.5) .* ...
    (1/6*(5 - 3*abs(r) - sqrt(-2 + 6*abs(r) - 3*r.^2)));

M2 = integral(@(r) r.^2 .* phi(r), -1.5, 1.5);

phi1 = @(r) ...
    (abs(r) <= 0.5) .* (1/6 * (1 + sqrt(1 - 3*r.^2))) + ...
    (abs(r) > 0.5 & abs(r) <= 1.5) .* ...
    (1/6 * (5 - 3*abs(r) - sqrt(-2 + 6*abs(r) - 3*(abs(r)).^2)));

% --- check normalization
M0 = integral(phi1Are, -1.5, 1.5);

% --- check second moment
M2 = integral(@(r) r.^2 .* phi(r), -1.5, 1.5);

disp([M0 M2])

M21 = integral(@(r) r.^2 .* phi1(r), -1.5, 1.5);
