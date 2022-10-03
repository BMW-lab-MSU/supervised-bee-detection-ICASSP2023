% SPDX-License-Identifier: BSD-3-Clause
%% Setup
clear

if isempty(gcp('nocreate'))
    parpool();
end

% Set random number generator properties for reproducibility
rng(0, 'twister');

baseDataDir ='../data';
rawDataDir = [baseDataDir filesep 'raw-combined'];
datafile = 'scans.mat';

%% Load data
load([rawDataDir filesep datafile])


% TODO: extracting features and splitting into training/testing sets should be separate scripts
%% Extract features
scanFeatures = cell(numel(scans), 1);

for i = progress(1:numel(scans))
    scanFeatures{i} = cellfun(@(X) extractFeatures(X, 'UseParallel', true), ...
        scans(i).Data, 'UniformOutput', false);
end

%%
labels = {scans.Labels}';
scanLabels = vertcat(scans.ScanLabel);

%% Partition into training and test sets
TEST_PCT = 0.2;

holdoutPartition = cvpartition(scanLabels, 'Holdout', TEST_PCT, 'Stratify', true);


trainingData = {scans(training(holdoutPartition)).Data}';
testingData = {scans(test(holdoutPartition)).Data}';
trainingFeatures = scanFeatures(training(holdoutPartition));
testingFeatures = scanFeatures(test(holdoutPartition));
trainingLabels = labels(training(holdoutPartition));
testingLabels = labels(test(holdoutPartition));

%% Partition the data for k-fold cross validation
N_FOLDS = 4;

crossvalPartition = cvpartition(scanLabels(training(holdoutPartition)), ...
    'KFold', N_FOLDS, 'Stratify', true);


%% Save training and testing data
mkdir(baseDataDir, 'testing');
save([baseDataDir filesep 'testing' filesep 'testingData.mat'], ...
    'testingData', 'testingFeatures', 'testingLabels', ...
    'holdoutPartition', 'scanLabels', '-v7.3');

mkdir(baseDataDir, 'training');
save([baseDataDir filesep 'training' filesep 'trainingData.mat'], ...
    'trainingData', 'trainingFeatures', 'trainingLabels', ...
    'crossvalPartition', 'holdoutPartition', 'scanLabels', '-v7.3');
