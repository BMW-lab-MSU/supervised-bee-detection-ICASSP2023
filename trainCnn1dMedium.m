basedir = '../data';

if isempty(gcp('nocreate'))
    parpool('IdleTimeout', Inf);
end

%% Load relevant files
load([basedir filesep 'training' filesep 'tuningCNN1dMedium']);            %"result"
load([basedir filesep 'training' filesep 'trainingData']);

%% extract the optimal hyperparameter values
undersamplingRatio=result.UndersamplingRatio;
costRatio=result.CostRatio;
nAugment=round(result.nAugment);

%% create hyperparameter structure

hyperparams.UndersamplingRatio = undersamplingRatio;
hyperparams.nAugment = nAugment;
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


% Create synthetic data
[synthData, synthLabels] = createSyntheticData(vertcat(trainingData{:}), vertcat(trainingLabels{:}), hyperparams.nAugment);
synthData = mat2cell(synthData, ones(height(synthData),1), width(synthData));

% Format training data and labels
trainingData2 = [mat2cell(vertcat(trainingData{:}), ones(178*numel(trainingData),1),1024); synthData];
trainingLabels2 = categorical([vertcat(trainingLabels{:}); synthLabels]);


%% train the model
model = fitCnn1dMedium(trainingData2, trainingLabels2, hyperparams);

mkdir([basedir filesep 'training' filesep 'models']);
save([basedir filesep 'training' filesep 'models' filesep 'Cnn1dMedium.mat'] ,"model")
