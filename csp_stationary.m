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

