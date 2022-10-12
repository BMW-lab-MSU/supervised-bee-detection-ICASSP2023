 % SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';

if isempty(gcp('nocreate'))
    parpool('IdleTimeout', Inf);
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

load([datadir filesep 'training' filesep 'augCostTuningADA.mat'])

%% Tune adaboost hyperparameters

undersamplingRatio = result.undersamplingRatio;
nAugment = round(result.nAugment);
costRatio = result.CostRatio;
fixedParams=struct();
fixedParams.CostRatio=costRatio;    %used to calculate weight vector in cvobjfun.m
fixedParams.ScoreTransform='doublelogit';
fixedParams.Cost=[0,1;costRatio,0];
fixedParams.ClassNames=logical([0,1]);
fixedParams.SplitCriterion='gdi';
fixedParams.LearnRate=0.1;

nObservations = height(vertcat(trainingFeatures{:}));

optimizeVars = [
   optimizableVariable('NumLearningCycles',[10, 200], 'Type', 'integer', 'Transform','log'),...
   optimizableVariable('MaxNumSplits',[1, nObservations - 1],'Transform','log', 'Type', 'integer'),...
   optimizableVariable('MinLeafSize',[1 3],'Transform','log', 'Type', 'integer')
];

minfun = @(optParams)hyperTuneObjFun(@ADAboost, optParams, fixedParams, ...
    undersamplingRatio, nAugment, crossvalPartition, trainingFeatures, ...
    trainingData, trainingLabels, imageLabels, 'UseParallel', true, ...
    'Progress', true);
%hyperTuneObjFun() is identical to cvobjfun() except that it takes two
%hyperparam arguments and concatenates them inside. This allows us to keep
%some hyperparams fixed while varying others

results = bayesopt(minfun, optimizeVars, ...
    'IsObjectiveDeterministic', true, 'UseParallel', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 30, 'Verbose', 1);

bestParams = bestPoint(results);

sum(results.UserDataTrace{results.IndexOfMinimumTrace(end)}.confusion,3)

save([datadir filesep 'training' filesep 'hyperparameterTuningADA.mat'],...
    'bestParams', '-v7.3');
