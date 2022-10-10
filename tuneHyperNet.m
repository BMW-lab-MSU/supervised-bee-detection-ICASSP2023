% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';

if isempty(gcp('nocreate'))
    parpool('IdleTimeout', Inf);
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

load([datadir filesep 'training' filesep 'augCostTuningNet.mat'])

%% Tune nnet hyperparameters

undersamplingRatio = result.undersamplingRatio;
nAugment = round(result.nAugment);
fixedParams=struct();
fixedParams.CostRatio=result.CostRatio;    %used to calculate weight vector in cvobjfun.m
fixedParams.Verbose=0;
fixedParams.Standardize=true;

nObservations = height(vertcat(trainingFeatures{:}));

optimizeVars = [
   optimizableVariable('LayerSizes',[10,300], 'Type', 'integer', 'Transform', 'log'),...
   optimizableVariable('Lambda',1/nObservations * [1e-5,1e5], 'Transform', 'log'),...
   optimizableVariable('activations', {'relu', 'tanh', 'sigmoid'}),...
];

minfun = @(optParams)hyperTuneObjFun(@NNet, optParams, fixedParams, ...
    undersamplingRatio, nAugment, crossvalPartition, trainingFeatures, ...
    trainingData, trainingLabels, imageLabels, 'UseParallel', true, ...
    'Progress', true);
%hyperTuneObjFun() is identical to cvobjfun() except that it takes two
%hyperparam arguments and concatenates them inside. This allows us to keep
%some hyperparams fixed while varying others

results = bayesopt(minfun, optimizeVars, ...
    'IsObjectiveDeterministic', true, 'UseParallel', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 25, 'Verbose', 1);

bestParams = bestPoint(results);

sum(results.UserDataTrace{results.IndexOfMinimumTrace(end)}.confusion,3)

save([datadir filesep 'training' filesep 'hyperparameterTuningNet.mat'],...
    'bestParams', '-v7.3');
