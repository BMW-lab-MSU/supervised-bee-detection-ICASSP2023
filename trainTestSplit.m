% SPDX-License-Identifier: BSD-3-Clause
%% Setup
clear

if isempty(gcp('nocreate'))
    parpool();
end

% Set random number generator properties for reproducibility
rng(0, 'twister');

baseDataDir ='../data';
preprocessedDataDir = [baseDataDir filesep 'preprocessed'];
datafile = 'preprocessedScans.mat';

%% Load data
load([preprocessedDataDir filesep datafile])

data = vertcat(scans.Data);
features = vertcat(scans.Features);

%% Format labels
labels = vertcat(scans.Labels);
% TODO: I should save testing and training set image labels
imageLabels = vertcat(scans.ImageLabels);

%% Partition into training and test sets
TEST_PCT = 0.2;

holdoutPartition = cvpartition(imageLabels, 'Holdout', TEST_PCT, 'Stratify', true);

trainingData = data(training(holdoutPartition));
testingData = data(test(holdoutPartition));
trainingFeatures = features(training(holdoutPartition));
testingFeatures = features(test(holdoutPartition));
trainingLabels = labels(training(holdoutPartition));
testingLabels = labels(test(holdoutPartition));

%% Partition the training data for k-fold cross validation
N_FOLDS = 5;

crossvalPartition = cvpartition(imageLabels(training(holdoutPartition)), ...
    'KFold', N_FOLDS, 'Stratify', true);


%% Save training and testing data
mkdir(baseDataDir, 'testing');
save([baseDataDir filesep 'testing' filesep 'testingData.mat'], ...
    'testingData', 'testingFeatures', 'testingLabels', ...
    'holdoutPartition', 'imageLabels', '-v7.3');

mkdir(baseDataDir, 'training');
save([baseDataDir filesep 'training' filesep 'trainingData.mat'], ...
    'trainingData', 'trainingFeatures', 'trainingLabels', ...
    'crossvalPartition', 'holdoutPartition', 'imageLabels', '-v7.3');
