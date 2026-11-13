function val = phi4(r)

    r = abs(r);
    val = zeros(size(r));

    % support over 4 grid points
    for k = -1:2
        val = val + w_k(r - k);
    end

end

function w = w_k(s)
    s = abs(s);

    if s < 1
        w = (1/6)*(4 - 6*s^2 + 3*s^3);
    elseif s < 2
        w = (1/6)*(2 - s)^3;
    else
        w = 0;
    end
end