basedir = '../data';
%% Load relevant files
load([basedir filesep 'training' filesep 'augCostTuningNet.mat']);
load([basedir filesep 'training' filesep 'trainingData']);

%% extract the optimal hyperparameter values
undersamplingRatio=result.UndersamplingRatio;
costRatio=result.CostRatio;
nAugment=round(result.nAugment);

%%  create hyperparameter structure
hyperparams.CostRatio=costRatio;
hyperparams.Standardize=true;
hyperparams.Verbose=0;
hyperparams.LayerSizes=[100];
hyperparams.Standardize=true;

%% Undersample the majority class
idxRemove = randomUndersample(...
    imageLabels(training(holdoutPartition)), 0, ...
    'UndersamplingRatio', undersamplingRatio, ...
    'Reproducible', true);

trainingFeatures(idxRemove) = [];
trainingData(idxRemove) = [];
trainingLabels(idxRemove) = [];

%% Un-nest data/labels
data = vertcat(trainingData{:});
features = vertcat(trainingFeatures{:});
labels = vertcat(trainingLabels{:});

%% Create synthetic features
[synthFeatures, synthLabels] = dataAugmentation(data, ...
    labels, nAugment, 'UseParallel', true);

features = vertcat(features, synthFeatures);
labels = vertcat(labels, synthLabels);
clear('synthFeatures', 'synthLabels');

%% train the model
model = NNet(features, labels, hyperparams);

mkdir([basedir filesep 'training' filesep 'models']);
save([basedir filesep 'training' filesep 'models' filesep 'NNet.mat'] ,"model")

