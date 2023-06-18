% X = load('left_data.csv');
% Y = load('right_data.csv');

%%%%%%%%%%%%%%%%%%%%%%%
% Preprocess the data %
%%%%%%%%%%%%%%%%%%%%%%%

% eeglab;
ns = 9; % number of subjects
data_train = cell(1, ns);
header_train = cell(1, ns);
data_test = cell(1, ns);
header_test = cell(1, ns);

% load all the training data for all ns subjects
for i = 1:ns
    [data_train_i, header_train_i] = sload(sprintf('../BBCI dataset 2a/A0%dT.gdf', i), 0, 'OVERFLOWDETECTION:OFF');
    data_train{i} = data_train_i;
    header_train{i} = header_train_i;
end

% % load all the test data for all ns subjects
% for i = 1:ns
%     [data_test_i, header_test_i] = sload(sprintf('../BBCI dataset 2a/A0%dE.gdf', i));
%     data_test{i} = data_test_i;
%     header_test{i} = header_test_i;
% end

% preprocess the data by removing the last three channels of each data_train
for i = 1:ns
    data_train{i} = data_train{i}(:, 1:22);
end

fs = 250; % sampling rate, given
% creating band pass filter
b = fir_bandpass(51, 8, 30, fs);

% apply band pass filter to all the training data
for i = 1:ns
    data_train{i} = apply_bandpass(data_train{i}, b);
end

% take the positions of left and right hand classes from the header
% and store them in a matrix
left_hand_pos = cell(1, ns);
right_hand_pos = cell(1, ns);

for i = 1:ns
    % Find the indices of all entries of 769 in header{i}.EVENT.TYP
    idx_769 = find(header_train{i}.EVENT.TYP == 769);

    % Find the corresponding entries in header{i}.EVENT.POS and store them in left_hand_pos{i}
    left_hand_pos{i} = header_train{i}.EVENT.POS(idx_769);

    % Find the indices of all entries of 770 in header{i}.EVENT.TYP
    idx_770 = find(header_train{i}.EVENT.TYP == 770);

    % Find the corresponding entries in header{i}.EVENT.POS and store them in right_hand_pos{i}
    right_hand_pos{i} = header_train{i}.EVENT.POS(idx_770);
end

% Now take EEG data within [0.5 3.5] seconds after cue onset position of each class
start = 0.5;
stop = 3.5;
EEG_left = cell(1, ns);
EEG_right = cell(1, ns);

for i = 1:ns

    % Temporary variable of left and right pos
    temp_pos_left = left_hand_pos{i};
    temp_pos_right = right_hand_pos{i};

    % temp_EEG_left = zeros(length(temp_pos_left), floor((stop - start) * fs))';
    % temp_EEG_right = zeros(length(temp_pos_right), floor((stop - start) * fs))';
    % temp_EEG_left = [];
    % temp_EEG_right = [];
    % temp_EEG_left = cell(1, length(temp_pos_left));
    % temp_EEG_right = cell(1, length(temp_pos_right));
    % LEFT
    for j = 1:length(temp_pos_left)
        temp_EEG_left{j} = data_train{i}(temp_pos_left(j) + floor(start * fs):temp_pos_left(j) + floor(stop * fs) - 1, :)';
    end

    EEG_left{i} = temp_EEG_left;

    % RIGHT
    for j = 1:length(temp_pos_right)
        temp_EEG_right{j} = data_train{i}(temp_pos_right(j) + floor(start * fs):temp_pos_right(j) + floor(stop * fs) - 1, :)';
    end

    EEG_right{i} = temp_EEG_right;
end

for i = 1:ns
    EEG_left{i} = EEG_left{i}';
    EEG_right{i} = EEG_right{i}';
end

% disp(size(EEG_left{1}));
% Now split the data into training and testing data
percent_train = 0.8; % 80 % training data, 20 % testing data
EEG_left_train = cell(1, ns);
EEG_left_test = cell(1, ns);
EEG_right_train = cell(1, ns);
EEG_right_test = cell(1, ns);

for i = 1:ns
    % LEFT
    [EEG_left_train{i}, EEG_left_test{i}] = split_EEG_one_class(EEG_left{i}, percent_train);

    % RIGHT
    [EEG_right_train{i}, EEG_right_test{i}] = split_EEG_one_class(EEG_right{i}, percent_train);
end

X = cell(1, length(EEG_left_train));

for i = 1:length(EEG_left_train)
    X{i} = horzcat(EEG_left_train{i}{:});
end

Y = cell(1, length(EEG_right_train));

for i = 1:length(EEG_right_train)
    Y{i} = horzcat(EEG_right_train{i}{:});
end

W_orig = cell(1, ns);
W_orig_inv = cell(1, ns);
% W_new = cell(1, ns);

c = size(X{1}, 1); % Number of channels
B = eye(c);

trials_x = size(X{1}, 2);
trials_y = size(Y{1}, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% hyperparameters for CSP-L1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
norm_p = 1;
norm_q = 2;
delta = -1e-6;
epsilon = 1e-5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1:ns
    W = [];

    for k = 1:c
        w = randn(c, 1);
        w = compute_w_k(w, X{j}, Y{j}, trials_x, trials_y, norm_p, norm_q, delta, epsilon);
        w = B * w / compute_p_norm(B * w, 2);
        W = [W w];

        for i = 1:trials_x
            X{j}(:, i) = (eye(c) - w * w') * X{j}(:, i);
        end

        for i = 1:trials_y
            Y{j}(:, i) = (eye(c) - w * w') * Y{j}(:, i);
        end

        B = (eye(c) - w * w') * B;
    end

    W_orig{j} = W;
end

B = eye(c);

for j = 1:ns
    W = [];

    for k = 1:c
        w = randn(c, 1);
        w = compute_w_k_inv(w, X{j}, Y{j}, trials_x, trials_y, norm_p, norm_q, delta, epsilon);
        w = B * w / compute_p_norm(B * w, 2);
        W = [W w];

        for i = 1:trials_x
            X{j}(:, i) = (eye(c) - w * w') * X{j}(:, i);
        end

        for i = 1:trials_y
            Y{j}(:, i) = (eye(c) - w * w') * Y{j}(:, i);
        end

        B = (eye(c) - w * w') * B;
    end

    W_orig_inv{j} = W;
end

m = 5; % number of components to be used

W = cell(1, ns);

for i = 1:ns
    W{i} = (horzcat(W_orig{i}(:, 1:m), W_orig_inv{i}(:, 1:m)));

    for j = 1:size(EEG_left_train{i}, 1)
        feat_left_train{i}{j} = [];

        for k = 1:m
            feat_left_train{i}{j} = [feat_left_train{i}{j}, compute_p_norm_p(W{i}(:, k)' * EEG_left_train{i}{j}, norm_p)];
        end

        for k = m + 1:2 * m
            feat_left_train{i}{j} = [feat_left_train{i}{j}, compute_p_norm_p(W{i}(:, k)' * EEG_left_train{i}{j}, norm_q)];
        end

    end

    for j = 1:size(EEG_right_train{i}, 1)
        feat_right_train{i}{j} = [];

        for k = 1:m
            feat_right_train{i}{j} = [feat_right_train{i}{j}, compute_p_norm_p(W{i}(:, k)' * EEG_right_train{i}{j}, norm_p)];
        end

        for k = m + 1:2 * m
            feat_right_train{i}{j} = [feat_right_train{i}{j}, compute_p_norm_p(W{i}(:, k)' * EEG_right_train{i}{j}, norm_q)];
        end

    end

    feat_left_train{i} = feat_left_train{i}';
    feat_right_train{i} = feat_right_train{i}';

end

for i = 1:ns
    left_label = ones(size(feat_left_train{i}{1}, 1), 1) * -1;
    right_label = ones(size(feat_right_train{i}{1}, 1), 1);

    for j = 1:size(feat_left_train{i}, 1)
        left{i}{j} = [feat_left_train{i}{j}, left_label];
    end

    for j = 1:size(feat_right_train{i}, 1)
        right{i}{j} = [feat_right_train{i}{j}, right_label];
    end

    left{i} = left{i}';
    right{i} = right{i}';

    feat_train{i} = shuffle([left{i}; right{i}]);
end

for i = 1:ns

    for j = 1:size(EEG_left_test{i}, 1)
        feat_left_test{i}{j} = [];

        for k = 1:m
            feat_left_test{i}{j} = [feat_left_test{i}{j}, compute_p_norm_p(W{i}(:, k)' * EEG_left_test{i}{j}, norm_p)];
        end

        for k = m + 1:2 * m
            feat_left_test{i}{j} = [feat_left_test{i}{j}, compute_p_norm_p(W{i}(:, k)' * EEG_left_test{i}{j}, norm_q)];
        end

    end

    for j = 1:size(EEG_right_test{i}, 1)
        feat_right_test{i}{j} = [];

        for k = 1:m
            feat_right_test{i}{j} = [feat_right_test{i}{j}, compute_p_norm_p(W{i}(:, k)' * EEG_right_test{i}{j}, norm_p)];
        end

        for k = m + 1:2 * m
            feat_right_test{i}{j} = [feat_right_test{i}{j}, compute_p_norm_p(W{i}(:, k)' * EEG_right_test{i}{j}, norm_q)];
        end

    end

    feat_left_test{i} = feat_left_test{i}';
    feat_right_test{i} = feat_right_test{i}';
end

for i = 1:ns
    left_label_test = ones(size(feat_left_test{i}{1}, 1), 1) * -1;
    right_label_test = ones(size(feat_right_test{i}{1}, 1), 1);

    for j = 1:size(feat_left_test{i}, 1)
        left_test{i}{j} = [feat_left_test{i}{j}, left_label_test];
    end

    for j = 1:size(feat_right_test{i}, 1)
        right_test{i}{j} = [feat_right_test{i}{j}, right_label_test];
    end

    left_test{i} = left_test{i}';
    right_test{i} = right_test{i}';

    feat_test{i} = shuffle([left_test{i}; right_test{i}]);
end

% Concatenate all training data into a single matrix
feat_train_all = cell(1, ns);
label_train_all = cell(1, ns);
lda = cell(1, ns);
accuracy = cell(1, ns);

for i = 1:ns

    for j = 1:size(feat_train{i}, 1)
        feat_train_all{i} = [feat_train_all{i}; feat_train{i}{j}(:, 1:end - 1)];
        label_train_all{i} = [label_train_all{i}; feat_train{i}{j}(:, end)];
    end

    lda{i} = fitcdiscr(feat_train_all{i}, label_train_all{i}, 'DiscrimType', 'linear');

end

% Train an LDA classifier on the training data
% lda = fitcdiscr(feat_train_all, label_train_all, 'DiscrimType', 'linear');

% Concatenate all test data into a single matrix
feat_test_all = cell(1, ns);
label_test_all = cell(1, ns);
label_predict_all = cell(1, ns);

for i = 1:ns

    for j = 1:size(feat_test{i}, 1)
        feat_test_all{i} = [feat_test_all{i}; feat_test{i}{j}(:, 1:end - 1)];
        label_test_all{i} = [label_test_all{i}; feat_test{i}{j}(:, end)];
    end

    label_predict_all{i} = predict(lda{i}, feat_test_all{i});
    accuracy{i} = sum(label_predict_all{i} == label_test_all{i}) / length(label_test_all{i});
end

% Test the accuracy of the classifier on the test data
% label_pred_all = predict(lda, feat_test_all);
% accuracy = sum(label_pred_all == label_test_all) / length(label_test_all);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function to band-pass filter the data to [8 30] Hz to remove muscle artifacts, powerline noise, and DC drift
function b = fir_bandpass(numtaps, low, high, fs)
    fnyq = fs / 2;
    b = fir1(numtaps - 1, [low, high] / fnyq, 'bandpass');
end

% function to apply band pass filtering to the raw EEG data
function EEG_filtered = apply_bandpass(raw_EEG, b)
    % Apply bandpass filter to raw EEG data
    % INPUT:
    % raw_EEG : EEG data in the shape of S x N
    % b : coefficient of band-pass filter
    %
    % OUTPUT:
    % EEG_filtered : filtered EEG data shape S x N
    %
    % N : number of channel
    % S : number of sample

    EEG_filtered = filter(b, 1, raw_EEG, [], 1);
end

% function to split eeg data into training and testing data
function [EEG_train, EEG_test] = split_EEG_one_class(EEG_one_class, percent_train)
    % split_EEG_one_class will receive EEG data of one class, with size of T x N x M, where
    % T = number of trial
    % N = number of electrodes
    % M = sample number
    %
    % INPUT:
    % EEG_data_one_class: the data of one class of EEG data
    %
    % percent_train: allocation percentage of training data, default is 0.8
    %
    % OUTPUT:
    % EEG_train: EEG data for training
    %
    % EEG_test: EEG data for test
    %
    % Both have type of np.arrray dimension of T x M x N

    % Number of all trials
    n = size(EEG_one_class, 1);
    % disp("size of EEG_one_class: " + n);

    n_tr = round(n * percent_train);
    n_te = n - n_tr;

    EEG_train = EEG_one_class(1:n_tr, :, :);
    EEG_test = EEG_one_class(n_tr:n_tr + n_te - 1, :, :);
end

function w_k = compute_w_k(w, X, Y, trials_x, trials_y, norm_p, norm_q, delta, epsilon)
    % COMPUTE_W_K Computes the updated weight vector using CSP-L1 algorithm
    %
    %   w_k = COMPUTE_W_K(w, X, Y, trials_x, trials_y) computes the updated weight vector
    %   using the CSP-L1 algorithm with the input arguments.
    %
    %   w: a column vector
    %   X: a matrix with num_cols_x columns
    %   Y: a matrix with num_cols_y columns
    %   trials_x: a scalar value
    %   trials_y: a scalar value
    %
    %   The output w_k is a column vector.

    w_k = w; % initialize the weight vector

    while 1 % start an infinite loop
        J_curr = compute_objective(w_k, X, Y, trials_x, trials_y, norm_p, norm_q); % compute the current objective value
        grad = compute_gradient(w_k, X, Y, norm_p, norm_q); % compute the gradient of the objective function
        % disp(grad);
        w_k = w_k + delta * grad; % update the weight vector
        J_next = compute_objective(w_k, X, Y, trials_x, trials_y, norm_p, norm_q); % compute the next objective value
        % disp(['J_curr = ', num2str(J_curr)]);
        % disp(['grad = ', mat2str(grad)]);
        % disp((J_next - J_curr) / J_curr);

        if (((J_next - J_curr) / J_curr) < epsilon) % check if the objective value has converged
            % disp("meow");
            break % exit the loop if the objective value has converged
        end

    end % end the loop

end

function w_k_inv = compute_w_k_inv(w, X, Y, trials_x, trials_y, norm_p, norm_q, delta, epsilon)
    % COMPUTE_W_K_INV Computes the updated weight vector using CSP-L1 algorithm
    %
    %   w_k_inv = COMPUTE_W_K_INV(w, X, Y, trials_x, trials_y) computes the updated weight vector
    %   using the CSP-L1 algorithm with the input arguments.
    %
    %   w: a column vector
    %   X: a matrix with num_cols_x columns
    %   Y: a matrix with num_cols_y columns
    %   trials_x: a scalar value
    %   trials_y: a scalar value
    %
    %   The output w_k_inv is a column vector.

    w_k_inv = w; % initialize the weight vector

    while 1 % start an infinite loop
        J_curr = compute_objective_inv(w_k_inv, X, Y, trials_x, trials_y); % compute the current objective value
        grad = -compute_gradient(w_k_inv, X, Y, norm_p, norm_q); % compute the gradient of the objective function
        w_k_inv = w_k_inv + delta * grad; % update the weight vector
        J_next = compute_objective_inv(w_k_inv, X, Y, trials_x, trials_y); % compute the next objective value

        if (((J_next - J_curr) / J_curr) < epsilon) % check if the objective value has converged
            % disp("kitty");
            break % exit the loop if the objective value has converged
        end

    end % end the loop

end

function gradient = compute_gradient(weights, input_x, input_y, norm_p, norm_q)
    % COMPUTE_GRADIENT Computes the gradient of a function
    %
    %   gradient = COMPUTE_GRADIENT(weights, input_x, input_y, norm_p, norm_q, num_cols_x, num_cols_y)
    %   computes the gradient of the given function using the input arguments.
    %
    %   weights: a column vector
    %   input_x: a matrix with num_cols_x columns
    %   input_y: a matrix with num_cols_y columns
    %   norm_p: a scalar value
    %   norm_q: a scalar value
    %   num_cols_x: a scalar value
    %   num_cols_y: a scalar value
    %
    %   The output gradient is a column vector.

    num_cols_x = size(input_x, 2);
    num_cols_y = size(input_y, 2);
    % Compute the numerator of the first term
    numerator_1 = zeros(size(weights));

    for i = 1:num_cols_x
        a_i_t = compute_a_i_t(weights, input_x(:, i));
        numerator_1 = numerator_1 + a_i_t * (abs(weights' * input_x(:, i)) ^ (norm_p - 1)) * input_x(:, i);
    end

    numerator_1 = norm_p * numerator_1;

    % Compute the denominator of the first term
    denominator_1 = 0;

    for i = 1:num_cols_x
        denominator_1 = denominator_1 + (abs(weights' * input_x(:, i)) ^ norm_p);
    end

    % Compute the numerator of the second term
    numerator_2 = zeros(size(weights));

    for j = 1:num_cols_y
        b_j_t = compute_b_j_t(weights, input_y(:, j));
        numerator_2 = numerator_2 + b_j_t * (abs(weights' * input_y(:, j)) ^ (norm_q - 1)) * input_y(:, j);
    end

    numerator_2 = norm_q * numerator_2;

    % Compute the denominator of the second term
    denominator_2 = 0;

    for j = 1:num_cols_y
        denominator_2 = denominator_2 + (abs(weights' * input_y(:, j)) ^ norm_q);
    end

    % Compute the gradient
    gradient = numerator_1 / denominator_1 - numerator_2 / denominator_2;
end

function objective = compute_objective(weights, input_x, input_y, trials_x, trials_y, norm_p, norm_q)
    % COMPUTE_OBJECTIVE Computes the objective function
    %
    %   objective = COMPUTE_OBJECTIVE(weights, input_x, input_y, num_cols_x, num_cols_y, trials_x, trials_y)
    %   computes the objective function using the input arguments.
    %
    %   weights: a column vector
    %   input_x: a matrix with num_cols_x columns
    %   input_y: a matrix with num_cols_y columns
    %   num_cols_x: a scalar value
    %   num_cols_y: a scalar value
    %   trials_x: a scalar value
    %   trials_y: a scalar value
    %
    %   The output objective is a scalar value.

    % Compute the numerator of the objective function
    numerator = 0;
    num_cols_x = size(input_x, 2);
    num_cols_y = size(input_y, 2);

    for i = 1:num_cols_x
        numerator = numerator + (abs(weights' * input_x(:, i)) ^ norm_p);
    end

    numerator = trials_y * numerator;

    % Compute the denominator of the objective function
    denominator = 0;

    for j = 1:num_cols_y
        denominator = denominator + abs(weights' * input_y(:, j)) ^ norm_q;
    end

    denominator = trials_x * denominator;

    % Compute the objective function
    objective = numerator / denominator;
end

function objective_inv = compute_objective_inv(weights, input_x, input_y, trials_x, trials_y)
    % COMPUTE_OBJECTIVE_INV Computes the objective function
    %
    %   objective_inv = COMPUTE_OBJECTIVE_INV(weights, input_x, input_y, num_cols_x, num_cols_y, trials_x, trials_y)
    %   computes the objective function using the input arguments.
    %
    %   weights: a column vector
    %   input_x: a matrix with num_cols_x columns
    %   input_y: a matrix with num_cols_y columns
    %   num_cols_x: a scalar value
    %   num_cols_y: a scalar value
    %   trials_x: a scalar value
    %   trials_y: a scalar value
    %
    %   The output objective_inv is a scalar value.

    % Compute the numerator of the objective function
    numerator = 0;
    num_cols_x = size(input_x, 2);
    num_cols_y = size(input_y, 2);

    for i = 1:num_cols_x
        numerator = numerator + abs(weights' * input_x(:, i));
    end

    numerator = trials_y * numerator;

    % Compute the denominator of the objective function
    denominator = 0;

    for j = 1:num_cols_y
        denominator = denominator + abs(weights' * input_y(:, j));
    end

    denominator = trials_x * denominator;

    % Compute the objective function
    objective_inv = denominator / numerator;
end

function a_i_t = compute_a_i_t(weights, input_x)
    % COMPUTE_A_I_T Computes the signum function of the scalar product of two vectors
    %
    %   a_i_t = COMPUTE_A_I_T(weights, input_x) computes the signum function of the scalar
    %   product of the input vectors weights and input_x. The output a_i_t is a scalar value
    %   that is either 1 or -1.

    % Compute the scalar product of the vectors
    if isrow(weights)
        weights = weights';
    end

    w_dot_x = weights' * input_x;

    % Compute the signum function of the scalar product
    if w_dot_x >= 0
        a_i_t = 1;
    else
        a_i_t = -1;
    end

end

function b_j_t = compute_b_j_t(weights, input_y)
    % COMPUTE_B_J_T Computes the signum function of the scalar product of two vectors
    %
    %   b_j_t = COMPUTE_B_J_T(weights, input_y) computes the signum function of the scalar
    %   product of the input vectors weights and input_y. The output b_j_t is a scalar value
    %   that is either 1 or -1.

    if isrow(weights)
        weights = weights';
    end

    % Compute the scalar product of the vectors
    w_dot_y = weights' * input_y;

    % Compute the signum function of the scalar product
    if w_dot_y >= 0
        b_j_t = 1;
    else
        b_j_t = -1;
    end

end

function norm_p_p = compute_p_norm_p(vector, p)
    % COMPUTE_P_NORM Computes the p-norm of a vector
    %
    %   norm_p = COMPUTE_P_NORM(vector, p) computes the p-norm of the input
    %   vector using the specified value of p. The input vector should be a
    %   column vector or a row vector. The output norm_p is a scalar value.

    % Check if the input vector is a row vector or a column vector
    if isrow(vector)
        vector = vector';
    end

    % Compute the p-norm of the vector
    norm_p_p = sum(abs(vector) .^ p);
end

function norm_p = compute_p_norm(vector, p)
    % COMPUTE_P_NORM Computes the p-norm of a vector
    %
    %   norm_p = COMPUTE_P_NORM(vector, p) computes the p-norm of the input
    %   vector using the specified value of p. The input vector should be a
    %   column vector or a row vector. The output norm_p is a scalar value.

    % Check if the input vector is a row vector or a column vector
    if isrow(vector)
        vector = vector';
    end

    % Compute the p-norm of the vector
    norm_p = sum(abs(vector) .^ p) ^ (1 / p);
end
