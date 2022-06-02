function y=one_side_attack(x, b)

[m, n] = size(x);
y = zeros(m, n);
counter = 1;
while counter < n
    tmp_block = min(b, n-counter+1);
    disp(tmp_block);
    ica_model = rica(x(:, counter:counter+tmp_block-1), tmp_block, 'VerbosityLevel', 1, 'Lambda', 0);
    y(:, counter:counter+tmp_block-1) = transform(ica_model, x(:, counter:counter+tmp_block-1));
    counter = counter + tmp_block;
end

end