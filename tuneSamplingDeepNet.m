% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data/insect-lidar';

if isempty(gcp('nocreate'))
    parpool(gpuDeviceCount("available"));
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

%% Tune sampling ratios
result = tuneSamplingBase(@nnet, trainingFeatures, trainingData, ...
    trainingLabels, imageLabels, crossvalPartition, ...
    'Progress', true, 'UseParallel', true);

min(result.objective)
result.undersamplingRatio
result.nAugment

save([datadir filesep 'training' filesep 'samplingTuningNet.mat'], 'result')

%% Model fitting function
function model = nnet(data, labels, ~)
    inputSize = width(data);
    layers = [
        featureInputLayer(inputSize, 'Normalization', 'zscore'),
	fullyConnectedLayer(25),
	reluLayer,
	fullyConnectedLayer(15),
	reluLayer,
	fullyConnectedLayer(5),
	reluLayer,
	fullyConnectedLayer(15),
	reluLayer,
	fullyConnectedLayer(25),
	reluLayer,
	fullyConnectedLayer(2),
	softmaxLayer,
	focalLossLayer('Alpha', 1, 'Gamma', 0.75)
    ];
    options = trainingOptions('adam', ...
        'MaxEpochs', 100,...
        'MiniBatchSize', 512,...
	'InitialLearnRate', 0.001,...
	'Verbose', true, ...
	'VerboseFrequency', 500, ...
	'ExecutionEnvironment', 'gpu');
    
    data.Labels = categorical(labels);

    model = trainNetwork(data, layers, options);
end
