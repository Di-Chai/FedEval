function y = rmse(x_pred, x_real)
    x_pred = (x_pred - min(x_pred(:))) / (max(x_pred(:)) - min(x_pred(:)));
    x_real = (x_real - min(x_real(:))) / (max(x_real(:)) - min(x_real(:)));
    [m, ~] = size(x_pred);
    result = 0;
    for i = 1:m
        result = result + min(mean((x_pred(i, :) - x_real).^2, 2));
    end
    y = sqrt(result / m);
end