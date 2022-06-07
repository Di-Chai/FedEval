clc;
clear all;

% All the attachs are repeated 10 times and take the average vale.

log = 'attack_ml100k.log';
for i = 1:10
    rng(i);
    X = cell2mat(struct2cell(load('ml100k')));
    for b = [10, 100, 1000]
        ica_attack_knowledge(X, b, log);
    end
end

log = 'attack_wine.log';
for i = 1:10
    rng(i);
    X = cell2mat(struct2cell(load('wine')));
    for b = [10, 100, 1000]
        ica_attack_knowledge(X, b, log);
    end
end

log = 'attack_mnist.log';
for i = 1:10
    rng(i);
    X = cell2mat(struct2cell(load('mnist')));
    X = X(1:10000, :);
    for b = [10, 100, 1000]
        ica_attack_knowledge(X, b, log);
    end
end
