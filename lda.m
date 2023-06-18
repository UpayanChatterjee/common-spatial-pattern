function coeff = lda(X, Y)
    % Calculate class means
    classes = unique(Y);
    num_classes = numel(classes);
    means = zeros(num_classes, size(X, 2));
    for i = 1:num_classes
        means(i, :) = mean(X(Y == classes(i), :));
    end

    % Calculate between-class scatter matrix
    overall_mean = mean(X);
    Sb = zeros(size(X, 2));
    for i = 1:num_classes
        N = sum(Y == classes(i));
        Sb = Sb + N * (means(i, :) - overall_mean).' * (means(i, :) - overall_mean);
    end

    % Calculate within-class scatter matrix
    Sw = zeros(size(X, 2));
    for i = 1:num_classes
        indices = find(Y == classes(i));
        class_data = X(indices, :);
        class_mean = means(i, :);
        class_cov = cov(class_data);
        Sw = Sw + (length(indices) - 1) * class_cov;
    end

    % Calculate eigenvectors and eigenvalues
    [V, D] = eig(Sw \ Sb);

    % Sort eigenvectors based on eigenvalues
    [~, indices] = sort(diag(D), 'descend');
    coeff = V(:, indices);
end
