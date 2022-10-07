% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data/insect-lidar';

if isempty(gcp('nocreate'))
	parpool();
end
statset('UseParallel', true);

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

load([datadir filesep 'training' filesep 'samplingTuningNet'])
load([datadir filesep 'training' filesep 'hyperparameterTuningNet'], 'bestParams')
undersamplingRatio = result.undersamplingRatio
nAugment = result.nAugment
params = bestParams
clear result

%% Undersample the majority class
idxRemove = randomUndersample(...
    imageLabels(training(holdoutPartition)), 0, ...
    'UndersamplingRatio', undersamplingRatio, ...
    'Reproducible', true);

trainingFeatures(idxRemove) = [];
trainingData(idxRemove) = [];
trainingLabels(idxRemove) = [];

%% Un-nest data/labels
data = nestedcell2mat(trainingData);
features = nestedcell2mat(trainingFeatures);
labels = nestedcell2mat(trainingLabels);

%% Create synthetic features
[synthFeatures, synthLabels] = dataAugmentation(data, ...
    labels, nAugment, 'UseParallel', true);

features = vertcat(features, synthFeatures);
labels = vertcat(labels, synthLabels);
clear('synthFeatures', 'synthLabels');

%% Train the model
model = nnet(features, labels, params);

mkdir([datadir filesep 'training' filesep 'models']);
save([datadir filesep 'training' filesep 'models' filesep 'nnet'], 'model');

%% Model fitting function
function model = nnet(data, labels, params)
    model = compact(fitcnet(data, labels, 'Standardize', true, ...
        'LayerSizes', params.LayerSizes, ...
        'Activations', char(params.activations), ...
        'Lambda', params.Lambda));
end
