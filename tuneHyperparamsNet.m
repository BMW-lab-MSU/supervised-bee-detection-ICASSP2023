% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';

if isempty(gcp('nocreate'))
    parpool();
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

load([datadir filesep 'training' filesep 'samplingTuningNet'])
undersamplingRatio = result.undersamplingRatio
nAugment = result.nAugment
min(result.objective)
clear result

%% Tune nnet hyperparameters
nObservations = height(nestedcell2mat(trainingFeatures));

optimizeVars = [
   optimizableVariable('LayerSizes',[10,300], 'Type', 'integer', 'Transform', 'log'),...
   optimizableVariable('Lambda',1/nObservations * [1e-5,1e5], 'Transform', 'log'),...
   optimizableVariable('activations', {'relu', 'tanh', 'sigmoid'}),...
];

minfun = @(hyperparams)cvobjfun(@nnet, hyperparams, ...
    undersamplingRatio, nAugment, crossvalPartition, trainingFeatures, ...
    trainingData, trainingLabels, imageLabels, 'UseParallel', true, ...
    'Progress', true);

results = bayesopt(minfun, optimizeVars, ...
    'IsObjectiveDeterministic', true, 'UseParallel', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 25);

bestParams = bestPoint(results);

sum(results.UserDataTrace{results.IndexOfMinimumTrace(end)}.confusion,3)

save([datadir filesep 'training' filesep 'hyperparameterTuningNet.mat'],...
    'results', 'bestParams', '-v7.3');

%% Model fitting function
function model = nnet(data, labels, params)
    model = compact(fitcnet(data, labels, 'Standardize', true, ...
        'LayerSizes', params.LayerSizes, ...
        'Activations', char(params.activations), ...
        'Lambda', params.Lambda));
end
