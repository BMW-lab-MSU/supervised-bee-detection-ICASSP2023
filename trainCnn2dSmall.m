basedir = '../data';

if isempty(gcp('nocreate'))
    parpool('IdleTimeout', Inf);
end

%% Load relevant files
load([basedir filesep 'training' filesep 'tuningCNN2dSmall']);            %"result"
load([basedir filesep 'training' filesep 'trainingData']);

%% extract the optimal hyperparameter values
undersamplingRatio=result.UndersamplingRatio;
costRatio=result.CostRatio;

%% create hyperparameter structure

hyperparams.UndersamplingRatio = undersamplingRatio;
hyperparams.CostRatio=costRatio;    %used to calculate weight vector in cvobjfun.m
hyperparams.Cost=[1,costRatio];
hyperparams.Cost = hyperparams.Cost / mean(hyperparams.Cost);

%% Undersample the majority class
idxRemove = randomUndersample(...
    imageLabels(training(holdoutPartition)), 0, ...
    'UndersamplingRatio', undersamplingRatio, ...
    'Reproducible', true);

trainingFeatures(idxRemove) = [];
trainingData(idxRemove) = [];
trainingLabels(idxRemove) = [];


%% train the model
model = fitCnn2dSmall(trainingData, trainingLabels, hyperparams);

mkdir([basedir filesep 'training' filesep 'models']);
save([basedir filesep 'training' filesep 'models' filesep 'Cnn2dSmall.mat'] ,"model")
