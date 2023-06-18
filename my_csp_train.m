% load the dataset
[s, h] = sload('~/Roorkee internship/BBCI dataset 2a/A01T.gdf', 0, 'OVERFLOWDETECTION:OFF');

% Remove last three columns of matrix 's' and store in 't'
t = s(:, 1:end-3);

% Find indices of occurrences of '769' in 'h.EVENT.TYP'
idx_769 = find(h.EVENT.TYP == 769);

% Initialize 'm1' as an empty matrix
m1 = [];

% Iterate over each occurrence of '769'
for i = 1:length(idx_769)
    % Get position and duration corresponding to current occurrence
    pos = h.EVENT.POS(idx_769(i));
    dur = h.EVENT.DUR(idx_769(i));
    
    % Extract data from 't' and append to 'm1'
    m1 = [m1; t(pos:pos+dur-1, :)];
end



% Find indices of occurrences of '770' in 'h.EVENT.TYP'
idx_770 = find(h.EVENT.TYP == 770);

% Initialize 'm2' as an empty matrix
m2 = [];

% Iterate over each occurrence of '770'
for i = 1:length(idx_770)
    % Get position and duration corresponding to current occurrence
    pos = h.EVENT.POS(idx_770(i));
    dur = h.EVENT.DUR(idx_770(i));
    
    % Extract data from 't' and append to 'm2'
    m2 = [m2; t(pos:pos+dur-1, :)];
end



% Find indices of occurrences of '276' in 'h.EVENT.TYP'
idx_276 = find(h.EVENT.TYP == 276);

% Initialize 'eeg_idle_openeye' as an empty matrix
eeg_idle_openeye = [];

% Iterate over each occurrence of '276'
for i = 1:length(idx_276)
    % Get position and duration corresponding to current occurrence
    pos = h.EVENT.POS(idx_276(i));
    dur = h.EVENT.DUR(idx_276(i));
    
    % Extract data from 't' and append to 'eeg_idle_openeye'
    eeg_idle_openeye = [eeg_idle_openeye; t(pos:pos+dur-1, :)];
end
%number of rows in m1
m1_rows = size(m1, 1);
eeg_idle_openeye = eeg_idle_openeye(1:m1_rows, :);

m1 = m1 - eeg_idle_openeye;
m2 = m2 - eeg_idle_openeye;


% standardize the data for m1 and m2 such that each column has unit variance
% m1 = m1 ./ repmat(std(m1), m1_rows, 1);
% m2 = m2 ./ repmat(std(m2), m1_rows, 1);

m1 = m1';
m2 = m2';

% center the data for m1 and m2 such that each column has zero mean
m1 = m1 - repmat(mean(m1, 2), 1, m1_rows);
m2 = m2 - repmat(mean(m2, 2), 1, m1_rows);

[W] = f_CSP(m1, m2);

%[W1] = f_CSP(W'*m1, W'*m2);

%[W2] = f_CSP(W1'*m1, W1'*m2);

%[W3] = f_CSP(W2'*m1, W2'*m2);
% W_master = [];
% for i = 1:100
%     W = f_CSP(W'*m1, W'*m2);
%     W_master{i} = W;
% end

% %select the first 3 and last 3 rows of W and store it in W itself
% % W = [W(1:3, :); W(end-2:end, :)];
% % for i = 1:10
% %     W_master{i} = [W_master{i}(1:3, :); W_master{i}(end-2:end, :)];
% % end

% for i = 1:100
%     Z{i}{1} = W_master{i}' * m1;
%     Z{i}{2} = W_master{i}' * m2;
% end

%loop over m1 and m2 and project the data onto the CSP space
Z01 = W' * m1;
Z02 = W' * m2;

% %Z21 = W1 * m1;
% %Z22 = W1 * m2;

% %Z31 = W2 * m1;
% %Z32 = W2 * m2;
% select the first 3 and last 3 rows of Z1 and Z2
% Z01 = [Z01(1:1, :); Z01(end:end, :)];
% Z02 = [Z02(1:1, :); Z02(end:end, :)];


% for i = 1:100
%     f{i}{1} = log(var(Z{i}{1}, 0, 2) ./ sum(var(Z{i}{1}, 0, 2)));
%     f{i}{2} = log(var(Z{i}{2}, 0, 2) ./ sum(var(Z{i}{2}, 0, 2)));
% end
% make a new vector whose element in ith row would be equal to the log of (variance of the ith row of Z1 divided by the sum of the variances of all the rows of Z1)
f_01 = log(var(Z01, 0, 2) ./ sum(var(Z01, 0, 2)));
f_02 = log(var(Z02, 0, 2) ./ sum(var(Z02, 0, 2)));

% f_21 = log(var(Z21, 0, 2) ./ sum(var(Z21, 0, 2)));
% f_22 = log(var(Z22, 0, 2) ./ sum(var(Z22, 0, 2)));

% f_31 = log(var(Z31, 0, 2) ./ sum(var(Z31, 0, 2)));
% f_32 = log(var(Z32, 0, 2) ./ sum(var(Z32, 0, 2)));
% feature = [f_1 f_2];
% label = [ones(1, size(f_1, 2)); zeros(1, size(f_2, 2))];

% for i = 1:100
%     feature{i} = [f{i}{1};f{i}{2}];
%     label{i} = [zeros(1, size(f{i}{1}, 2)) ones(1, size(f{i}{2}, 2))];
% end
% f_01 = randn(2, 1);
% f_02 = randn(2, 1);
feature0 = [f_01; f_02];
% feature2 = [f_21; f_22];
% feature3 = [f_31; f_32];
label0 = [zeros(size(f_01, 1), 1); ones(size(f_02, 1), 1)];

% train a linear discriminant classifier
% declare a cell array to store the classifier
% ldaClassifier = cell(1, 10);
% for i = 1:100
%     feature_ = feature{i};
%     label_ = label{i};
%     ldaClassifier{i} = fitcdiscr(feature_, label0, 'DiscrimType', 'linear');
% end
ldaClassifier0 = fitcdiscr(feature0, label0, 'DiscrimType', 'linear');
% ldaClassifier1 = fitcdiscr(feature{1}, label{1}, 'DiscrimType', 'linear');
% ldaClassifier2 = fitcdiscr(feature{2}, label{2}, 'DiscrimType', 'linear');
% ldaClassifier3 = fitcdiscr(feature{3}, label{3}, 'DiscrimType', 'linear');
% ldaClassifier4 = fitcdiscr(feature{4}, label{4}, 'DiscrimType', 'linear');
% ldaClassifier5 = fitcdiscr(feature{5}, label{5}, 'DiscrimType', 'linear');
% ldaClassifier6 = fitcdiscr(feature{6}, label{6}, 'DiscrimType', 'linear');
% ldaClassifier7 = fitcdiscr(feature{7}, label{7}, 'DiscrimType', 'linear');
% ldaClassifier8 = fitcdiscr(feature{8}, label{8}, 'DiscrimType', 'linear');
% ldaClassifier9 = fitcdiscr(feature{9}, label{9}, 'DiscrimType', 'linear');
% ldaClassifier10 = fitcdiscr(feature{10}, label{10}, 'DiscrimType', 'linear');

% ldaClassifier2 = fitcdiscr(feature2, label, 'DiscrimType', 'linear');
% ldaClassifier3 = fitcdiscr(feature3, label, 'DiscrimType', 'linear');



% load the test data
[s_tst, h_tst] = sload('~/Roorkee internship/BBCI dataset 2a/A01T.gdf', 0, 'OVERFLOWDETECTION:OFF');

% Remove last three columns of matrix 's_tst' and store in 't_tst'
t_tst = s_tst(:, 1:end-3);

% Find indices of occurrences of '769' in 'h_tst.EVENT.TYP'
idx_769_tst = find(h_tst.EVENT.TYP == 769);

% Initialize 'm1_tst' as an empty matrix
m1_tst = [];

% Iterate over each occurrence of '769'
for i = 1:length(idx_769_tst)
    % Get position and duration corresponding to current occurrence
    pos = h_tst.EVENT.POS(idx_769_tst(i));
    dur = h_tst.EVENT.DUR(idx_769_tst(i));
    
    % Extract data from 't_tst' and append to 'm1_tst'
    m1_tst = [m1_tst; t_tst(pos:pos+dur-1, :)];
end



% Find indices of occurrences of '770' in 'h_tst.EVENT.TYP'
idx_770_tst = find(h_tst.EVENT.TYP == 770);

% Initialize 'm2_tst' as an empty matrix
m2_tst = [];

% Iterate over each occurrence of '770'
for i = 1:length(idx_770_tst)
    % Get position and duration corresponding to current occurrence
    pos = h_tst.EVENT.POS(idx_770_tst(i));
    dur = h_tst.EVENT.DUR(idx_770_tst(i));
    
    % Extract data from 't_tst' and append to 'm2_tst'
    m2_tst = [m2_tst; t_tst(pos:pos+dur-1, :)];
end



% Find indices of occurrences of '276' in 'h_tst.EVENT.TYP'
idx_276_tst = find(h_tst.EVENT.TYP == 276);

% Initialize 'eeg_idle_openeye' as an empty matrix
eeg_idle_openeye_tst = [];

% Iterate over each occurrence of '276'
for i = 1:length(idx_276_tst)
    % Get position and duration corresponding to current occurrence
    pos = h_tst.EVENT.POS(idx_276_tst(i));
    dur = h_tst.EVENT.DUR(idx_276_tst(i));
    
    % Extract data from 't_tst' and append to 'eeg_idle_openeye_tst'
    eeg_idle_openeye_tst = [eeg_idle_openeye_tst; t_tst(pos:pos+dur-1, :)];
end
%number of rows in m1_tst
m1_tst_rows = size(m1_tst, 1);
eeg_idle_openeye_tst = eeg_idle_openeye_tst(1:m1_tst_rows, :);

m1_tst = m1_tst - eeg_idle_openeye_tst;
m2_tst = m2_tst - eeg_idle_openeye_tst;


% standardize the data for m1_tst and m2_tst such that each column has unit variance
% m1_tst = m1_tst ./ repmat(std(m1_tst), m1_tst_rows, 1);
% m2_tst = m2_tst ./ repmat(std(m2_tst), m1_tst_rows, 1);

m1_tst = m1_tst';
m2_tst = m2_tst';

% center the data for m1_tst and m2_tst such that each column has zero mean
m1_tst = m1_tst - repmat(mean(m1_tst, 2), 1, m1_rows);
m2_tst = m2_tst - repmat(mean(m2_tst, 2), 1, m1_rows);
%loop over m1_tst and m2_tst and project the data onto the CSP space
Z01_tst = W' * m1_tst;
Z02_tst = W' * m2_tst;

% for i = 1:100
%     Z1_tst{i} = W_master{i} * m1_tst;
%     Z2_tst{i} = W_master{i} * m2_tst;
% end
% Z21_tst = W1 * m1_tst;
% Z22_tst = W1 * m2_tst;

% Z31_tst = W2 * m1_tst;
% Z32_tst = W2 * m2_tst;

% select the first 3 and last 3 rows of Z1_tst and Z2_tst
% Z01_tst = [Z01_tst(1:1, :), Z01_tst(end:end, :)];
% Z02_tst = [Z02_tst(1:1, :), Z02_tst(end:end, :)];

% make a new vector whose element in ith row would be equal to the log of (variance of the ith row of Z1_tst divided by the sum of the variances of all the rows of Z1_tst)
f_01_tst = log(var(Z01_tst, 0, 2) ./ sum(var(Z01_tst, 0, 2)));
f_02_tst = log(var(Z02_tst, 0, 2) ./ sum(var(Z02_tst, 0, 2)));

% for i = 1:100
%     f_1_tst{i} = log(var(Z1_tst{i}, 0, 2) ./ sum(var(Z1_tst{i}, 0, 2)));
%     f_2_tst{i} = log(var(Z2_tst{i}, 0, 2) ./ sum(var(Z2_tst{i}, 0, 2)));
% end

% f_21_tst = log(var(Z21_tst, 0, 2) ./ sum(var(Z21_tst, 0, 2)));
% f_22_tst = log(var(Z22_tst, 0, 2) ./ sum(var(Z22_tst, 0, 2)));

% f_31_tst = log(var(Z31_tst, 0, 2) ./ sum(var(Z31_tst, 0, 2)));
% f_32_tst = log(var(Z32_tst, 0, 2) ./ sum(var(Z32_tst, 0, 2)));

feature0_tst = [f_01_tst; f_02_tst];

% for i = 1:100
%     feature1_tst{i} = [f_1_tst{i}; f_2_tst{i}];
% end
% feature2_tst = [f_21; f_22];
% feature3_tst = [f_31; f_32];
trueLabels = [zeros(size(f_01_tst, 1), 1); ones(size(f_02_tst, 1), 1)];

predictedLabels0 = predict(ldaClassifier0, feature0_tst);

% for i = 1:100
%     predictedLabels{i} = predict(ldaClassifier{i}, feature1_tst{i});
% end
% predictedLabels2 = predict(ldaClassifier2, feature2_tst);
% predictedLabels3 = predict(ldaClassifier3, feature3_tst);

accuracy0 = sum(predictedLabels0 == trueLabels) / numel(trueLabels);

% for i = 1:100
%     accuracy(i) = sum(predictedLabels{i} == trueLabels) / numel(trueLabels);
% end
% accuracy2 = sum(predictedLabels2 == trueLabels) / numel(trueLabels);
% accuracy3 = sum(predictedLabels3 == trueLabels) / numel(trueLabels);