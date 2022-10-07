% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data/insect-lidar';

if isempty(gcp('nocreate'))
    parpool();
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

%% Tune sampling ratios
result = tuneSamplingBase(@rusboost, trainingFeatures, trainingData, ...
    trainingLabels, imageLabels, crossvalPartition, ...
    'Progress', true, 'UseParallel', true);

min(result.objective)
result.undersamplingRatio
result.nAugment

save([datadir filesep 'training' filesep 'samplingTuningRUSBoost.mat'], 'result')

%% Model fitting function
function model = rusboost(data, labels, ~)
    t = templateTree('Reproducible',true);
    model = compact(fitcensemble(data, labels, 'Method', 'RUSBoost', 'Learners', t, 'ScoreTransform', 'doublelogit'));
end
