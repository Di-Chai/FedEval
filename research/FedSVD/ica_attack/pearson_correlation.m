function [y1, y2]=pearson_correlation(x_pred, x_real)
    [~, n] = size(x_pred);
    y = corrcoef([x_pred, x_real]);
    y1 = mean(max(y(1:n,n+1:2*n)));
    y2 = max(max(y(1:n,n+1:2*n)));
end