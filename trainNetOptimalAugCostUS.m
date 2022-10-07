basedir = '../data/projects/optec2022';
%% Load relevant files
load([basedir filesep 'training' filesep 'augCostTuningNet.mat']);            %"result"
load([basedir filesep 'training' filesep 'trainingData']);

%%
undersamplingRatio=result.undersamplingRatio;
costRatio=result.CostRatio;
nAugment=round(result.nAugment);

%% 
hyperparams.CostRatio=costRatio;    %used to calculate weight vector in cvobjfun.m
hyperparams.Verbose=1;
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
data = nestedcell2mat(trainingData);
features = nestedcell2mat(trainingFeatures);
labels = nestedcell2mat(trainingLabels);

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

