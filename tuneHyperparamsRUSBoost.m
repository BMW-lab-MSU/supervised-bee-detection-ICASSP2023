% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(2, 'twister');

datadir = '../data/insect-lidar';

if isempty(gcp('nocreate'))
    parpool();
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

load([datadir filesep 'training' filesep 'samplingTuningRUSBoost'])
undersamplingRatio = result.undersamplingRatio
nAugment = result.nAugment
[m,i]=min(result.objective)
sum(result.userdata{i}.confusion,3)
clear result

%% Tune rusboost hyperparameters
nObservations = height(nestedcell2mat(trainingFeatures));

optimizeVars = [
   optimizableVariable('NumLearningCycles',[10, 500], 'Type', 'integer', 'Transform','log'),...
   optimizableVariable('LearnRate',[1e-3, 1], 'Transform','log'),...
   optimizableVariable('fncost', [1 20], 'Type', 'integer')
   optimizableVariable('MaxNumSplits',[1, nObservations - 1],'Transform','log', 'Type', 'integer'),...
   optimizableVariable('MinLeafSize',[1 floor(nObservations/2)],'Transform','log', 'Type', 'integer'),...
   optimizableVariable('SplitCriterion', {'gdi', 'deviance'}),...
];

minfun = @(hyperparams)cvobjfun(@rusboost, hyperparams, ...
    undersamplingRatio, nAugment, crossvalPartition, trainingFeatures, ...
    trainingData, trainingLabels, imageLabels, 'UseParallel', true, ...
    'Progress', true);

results = bayesopt(minfun, optimizeVars, ...
    'IsObjectiveDeterministic', true, 'UseParallel', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 25);

bestParams = bestPoint(results);

sum(results.UserDataTrace{results.IndexOfMinimumTrace(end)}.confusion,3)

save([datadir filesep 'training' filesep 'hyperparameterTuningRUSBoost.mat'],...
    'results', 'bestParams', '-v7.3');

%% Model fitting function
function model = rusboost(data, labels, params)
    t = templateTree('Reproducible',true, ...
       'MaxNumSplits', params.MaxNumSplits, ...
       'MinLeafSize', params.MinLeafSize, ...
       'SplitCriterion', char(params.SplitCriterion));

    model = compact(fitcensemble(data, labels, 'Method', 'RUSBoost', ...
       'Learners', t, 'Cost', [0 1; params.fncost 0], ...
       'NumLearningCycles', params.NumLearningCycles, ...
       'LearnRate', params.LearnRate));
end
