% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';

if isempty(gcp('nocreate'))
    parpool();
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
    model = compact(fitcnet(data, labels, 'LayerSizes', [100], ...
        'Standardize', true));
end
