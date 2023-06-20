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
% b = fir_bandpass(51, 8, 30, fs);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% Now remove the mean of each channel from the data
for i = 1:ns

    for j = 1:length(EEG_left{i})

        for k = 1:size(EEG_left{i}{j}, 1)
            EEG_left{i}{j}(k, :) = EEG_left{i}{j}(k, :) - mean(EEG_left{i}{j}(k, :));
        end

    end

    for j = 1:length(EEG_right{i})

        for k = 1:size(EEG_right{i}{j}, 1)
            EEG_right{i}{j}(k, :) = EEG_right{i}{j}(k, :) - mean(EEG_right{i}{j}(k, :));
        end

    end

end

disp(size(EEG_left{1}));
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

cov_left = cell(1, ns);
cov_right = cell(1, ns);
cov_comp = cell(1, ns);
whitening = cell(1, ns);
S_l = cell(1, ns);
S_r = cell(1, ns);
B = cell(1, ns);
eigen_value = cell(1, ns);
W_orig = cell(1, ns);
W_new = cell(1, ns);
W = cell(1, ns);

for i = 1:ns
    cov_left{i} = compute_avg_cov(EEG_left_train{i});
    cov_right{i} = compute_avg_cov(EEG_right_train{i});
    cov_comp{i} = cov_left{i} + cov_right{i};
    [eig_vec, eig_val] = eig_decompose_descend(cov_comp{i});
    whitening{i} = compute_whitening(eig_vec, eig_val);
    S_l{i} = whitening{i} * cov_left{i} * whitening{i}';
    S_r{i} = whitening{i} * cov_right{i} * whitening{i}';
    % [~, temp_eigenvalue_l] = eig_decompose_descend(S_l{i});
    % [~, temp_eigenvalue_r] = eig_decompose_ascend(S_r{i});
    [B{i}, eigen_value{i}] = eig_decompose_descend(S_l{i});
    W_orig{i} = (B{i}' * whitening{i});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % now we apply csp again on the Z_left and Z_right
% for i = 1:ns

%     for j = 1:size(EEG_left_train{i}, 1)
%         Z_left{i}{j} = W_orig{i} * EEG_left_train{i}{j};
%         % feat_left_train{i}{j} = (log(var(Z_left{i}{j}, 0, 2) / sum(var(Z_left{i}{j}, 0, 2))))';
%     end

%     for j = 1:size(EEG_right_train{i}, 1)
%         Z_right{i}{j} = W_orig{i} * EEG_right_train{i}{j};
%         % feat_right_train{i}{j} = (log(var(Z_right{i}{j}, 0, 2) / sum(var(Z_right{i}{j}, 0, 2))))';
%     end

%     % feat_left_train{i} = feat_left_train{i}';
%     % feat_right_train{i} = feat_right_train{i}';
% end

% for i = 1:ns
%     cov_left_{i} = compute_avg_cov(Z_left{i});
%     cov_right_{i} = compute_avg_cov(Z_right{i});
%     cov_comp_{i} = cov_left_{i} + cov_right_{i};
%     [eig_vec, eig_val] = eig_decompose_descend(cov_comp_{i});
%     whitening_{i} = compute_whitening(eig_vec, eig_val);
%     S_l_{i} = whitening_{i} * cov_left_{i} * whitening_{i}';
%     S_r_{i} = whitening_{i} * cov_right_{i} * whitening_{i}';
%     % [~, temp_eigenvalue_l] = eig_decompose_descend(S_l_{i});
%     % [~, temp_eigenvalue_r] = eig_decompose_ascend(S_r_{i});
%     [B_{i}, ~] = eig_decompose_descend(S_l_{i});
%     W_new{i} = B_{i}' * whitening_{i};
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = 3; % number of components to be used

for i = 1:ns
    % W{i} = W_new{i}([1:m, end - m + 1:end], :);
    W{i} = W_orig{i}([1:m, end - m + 1:end], :);

    for j = 1:size(EEG_left_train{i}, 1)
        Z_left{i}{j} = W{i} * EEG_left_train{i}{j};
        feat_left_train{i}{j} = (log(var(Z_left{i}{j}, 0, 2) / sum(var(Z_left{i}{j}, 0, 2))))';
    end

    for j = 1:size(EEG_right_train{i}, 1)
        Z_right{i}{j} = W{i} * EEG_right_train{i}{j};
        feat_right_train{i}{j} = (log(var(Z_right{i}{j}, 0, 2) / sum(var(Z_right{i}{j}, 0, 2))))';
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
        Z_left_test{i}{j} = W{i} * EEG_left_test{i}{j};
        feat_left_test{i}{j} = (log(var(Z_left_test{i}{j}, 0, 2) / sum(var(Z_left_test{i}{j}, 0, 2))))';
    end

    for j = 1:size(EEG_right_test{i}, 1)
        Z_right_test{i}{j} = W{i} * EEG_right_test{i}{j};
        feat_right_test{i}{j} = (log(var(Z_right_test{i}{j}, 0, 2) / sum(var(Z_right_test{i}{j}, 0, 2))))';
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

    lda{i} = fitcdiscr(feat_train_all{i}, label_train_all{i});

end

% Train an LDA classifier on the training data
% lda = fitcdiscr(feat_train_all, label_train_all);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%   functions   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% function to calculate covariance matrix of EEG data
function cov_matrix = compute_avg_cov(EEG_data)
    % compute_cov will receive EEG data of one class, with size of T x N x S, where
    % T = number of trial
    % N = number of electrodes
    % S = sample number
    %
    % INPUT:
    % EEG_data: the data of one class of EEG data
    %
    % OUTPUT:
    % cov_matrix: covariance matrix of the EEG data, with size of N x N

    % Number of all trials
    T = size(EEG_data, 1);

    % Number of electrodes
    N = size(EEG_data{1}, 1);

    % Number of samples
    S = size(EEG_data{1}, 2);

    % disp("T: " + T);
    % disp("N: " + N);
    % disp("S: " + S);

    % Initialize covariance matrix
    cov_matrix = zeros(N, N);

    % Calculate covariance matrix
    for i = 1:T
        cov_matrix = cov_matrix + (EEG_data{i} * EEG_data{i}') / trace(EEG_data{i} * EEG_data{i}');
    end

    cov_matrix = cov_matrix / T;
end

% function to calculate eigen decomposition of average covariance matrix
function [eig_vec, eig_val] = eig_decompose_descend(cov_matrix)
    % compute_eig will receive covariance matrix of EEG data, with size of N x N, where
    % N = number of electrodes
    %
    % INPUT:
    % cov_matrix: covariance matrix of the EEG data
    %
    % OUTPUT:
    % eig_vec: eigenvector of the covariance matrix, with size of N x N
    %
    % eig_val: eigenvalue of the covariance matrix, with size of N x 1

    % Calculate eigen decomposition of covariance matrix
    [eig_vec, eig_val] = eig(cov_matrix);

    % Sort eigenvalue and eigenvector in descending order
    [eig_val, ind] = sort(diag(eig_val), 'descend');
    eig_vec = eig_vec(:, ind);
end

function [eig_vec, eig_val] = eig_decompose_ascend(cov_matrix)
    % compute_eig will receive covariance matrix of EEG data, with size of N x N, where
    % N = number of electrodes
    %
    % INPUT:
    % cov_matrix: covariance matrix of the EEG data
    %
    % OUTPUT:
    % eig_vec: eigenvector of the covariance matrix, with size of N x N
    %
    % eig_val: eigenvalue of the covariance matrix, with size of N x 1

    % Calculate eigen decomposition of covariance matrix
    [eig_vec, eig_val] = eig(cov_matrix);

    % Sort eigenvalue and eigenvector in ascending order
    [eig_val, ind] = sort(diag(eig_val), 'ascend');
    eig_vec = eig_vec(:, ind);
end

% function to calculate whitening transformation matrix
function W = compute_whitening(eig_vec, eig_val)
    % compute_whitening will receive eigenvector and eigenvalue of covariance matrix, with size of N x N and N x 1, where
    % N = number of electrodes
    %
    % INPUT:
    % eig_vec: eigenvector of the covariance matrix
    %
    % eig_val: eigenvalue of the covariance matrix
    %
    % OUTPUT:
    % W: whitening transformation matrix, with size of N x N

    % Calculate whitening transformation matrix
    W = sqrtm(inv(diag(eig_val))) * eig_vec';
end
