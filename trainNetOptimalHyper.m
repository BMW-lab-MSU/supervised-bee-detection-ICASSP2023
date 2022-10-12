basedir = '../data';
%% Load relevant files
load([basedir filesep 'training' filesep 'augCostTuningNet.mat']);
load([basedir filesep 'training' filesep 'hyperparameterTuningNet.mat']); 
load([basedir filesep 'training' filesep 'trainingData']);

%% extract the optimal hyperparameter values
undersamplingRatio=result.undersamplingRatio;
costRatio=result.CostRatio;
nAugment=round(result.nAugment);
LayerSize=bestParams.LayerSizes;
Lambda=bestParams.Lambda;
Activations=string(bestParams.activations);

%%  create hyperparameter structure
hyperparams.CostRatio=costRatio;
hyperparams.Verbose=1;
hyperparams.LayerSizes=LayerSize;
hyperparams.Lambda=Lambda;
hyperparams.Activations=Activations;
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

