function y = generate_ortho(m, b)
    num_blocks = floor(m / b);
    if mod(m, b) > 0
        num_blocks = num_blocks + 1;
    end
    y = cell(num_blocks, 1);
    counter = 0;
    index = 1;
    while counter < m
        tmp_block = min(b, m-counter);
        tmp_random = normrnd(0, 1, [tmp_block, tmp_block]);
        [q, ~] = qr(tmp_random);
        y{index} = q;
        counter = counter + tmp_block;
        index = index + 1;
    end
end