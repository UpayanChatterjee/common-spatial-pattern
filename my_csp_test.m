% load the dataset
[s, h] = sload('~/Roorkee internship/BBCI dataset 2a/A01E.gdf', 0, 'OVERFLOWDETECTION:OFF');

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

% center the data for m1 and m2 such that each column has zero mean
m1 = m1 - repmat(mean(m1), m1_rows, 1);
m2 = m2 - repmat(mean(m2), m1_rows, 1);

% standardize the data for m1 and m2 such that each column has unit variance
m1 = m1 ./ repmat(std(m1), m1_rows, 1);
m2 = m2 ./ repmat(std(m2), m1_rows, 1);

% apply csp filter on 