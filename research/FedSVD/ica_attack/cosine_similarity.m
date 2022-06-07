function y=cosine_similarity(a, b)
    c = a * b';
    d = sqrt(sum(a.^2, 2)) * sqrt(sum(b.^2, 2))';
    y = c ./ d;
end