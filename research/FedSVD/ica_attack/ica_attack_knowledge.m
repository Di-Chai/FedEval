function ica_attack_knowledge(X, block_size, log)
    
    [m, n] = size(X);
    
    PXQ = apply_mask_left(X, block_size);
    PXQ = apply_mask_left(PXQ', block_size)';

    file_hd = fopen(log, 'a+');
    fprintf(file_hd, '########################################\n');
    fprintf(file_hd, '########################################\n');
    fprintf(file_hd, 'm %d n %d block_size %d\n', m, n, block_size);
    
    % Random
    random = rand([m, n]);
    [~, random_left_max] = pearson_correlation(random, X);
    [~, random_right_max] = pearson_correlation(random', X');
    random_max = (random_left_max + random_right_max)/2;
    fprintf(file_hd, '########################################\n');
    fprintf(file_hd, 'Random Mean: PMax %f \n', random_max);
    
    guess_blocks = [max(m, n), block_size];

    for guess_block = guess_blocks
        if guess_block == 1
            pxq_recover = PXQ;
        else
            pxq_recover = one_side_attack(PXQ, guess_block);
            pxq_recover = one_side_attack(pxq_recover', guess_block)';
        end
        [~, pxq_left_max] = pearson_correlation(pxq_recover, X);
        [~, pxq_right_max] = pearson_correlation(pxq_recover', X');
        pxq_max = (pxq_left_max + pxq_right_max)/2;
        fprintf(file_hd, '########################################\n');
        fprintf(file_hd, 'GuessBlock %d \n', guess_block);
        fprintf(file_hd, 'PXQ Mean: PMax %f \n', pxq_max);
    end
    
    fclose(file_hd);
    
end
