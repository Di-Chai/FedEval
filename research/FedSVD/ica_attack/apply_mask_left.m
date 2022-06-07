function y=apply_mask_left(x, b)

[m, ~] = size(x);
random_mask = generate_ortho(m, b);

y = zeros(size(x));

size_of_p = size(random_mask);
counter = 1;
for c = 1:size_of_p(1)
    tmp_size = size(random_mask{c});
    y(counter:counter+tmp_size(1)-1, :) = random_mask{c} * x(counter:counter+tmp_size(1)-1, :);
    counter = counter + tmp_size(1);
end

end