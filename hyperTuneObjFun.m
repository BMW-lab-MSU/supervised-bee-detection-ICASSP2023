function [objective, constraints, userdata] = hyperTuneObjFun(fitcfun, optParams, fixedParams, undersamplingRatio, nAugment, crossvalPartition, features, data, labels, imageLabel, opts)
% cvobjfun Optimize hyperparameters via cross-validation

% SPDX-License-Identifier: BSD-3-Clause
arguments
    fitcfun (1,1) function_handle
    optParams (1,:) table
    fixedParams
    undersamplingRatio (1,1) double
    nAugment (1,1) double
    crossvalPartition (1,1) cvpartition
    features (:,1) cell
    data (:,1) cell
    labels (:,1) cell
    imageLabel (:,1) logical
    opts.Progress (1,1) logical = false
    opts.UseParallel (1,1) logical = false
end

hyperparams=structFieldAppend(table2struct(optParams), fixedParams);

MAJORITY_LABEL = 0;

if opts.UseParallel
    statset('UseParallel', true);
end

crossvalConfusion = zeros(2, 2, crossvalPartition.NumTestSets);
% losses = nan(1, crossvalPartition.NumTestSets);
models = cell(1, crossvalPartition.NumTestSets);
predLabels = cell(1, crossvalPartition.NumTestSets);

if opts.Progress
    progressbar = ProgressBar(crossvalPartition.NumTestSets, ...
        'UpdateRate', inf, 'Title', 'Cross validation');
    progressbar.setup([], [], []);
end

for i = 1:crossvalPartition.NumTestSets
    % Get validation and training partitions
    validationSet = test(crossvalPartition, i); 
    trainingSet = training(crossvalPartition, i);
    
    trainingFeatureImages = features(trainingSet);
    trainingDataImages = data(trainingSet);
    trainingLabelImages = labels(trainingSet);

    % Undersample the majority class
    idxRemove = randomUndersample(...
        imageLabel(trainingSet), MAJORITY_LABEL, ...
        'UndersamplingRatio', undersamplingRatio, ...
        'Reproducible', true, 'Seed', i);
    
    trainingFeatureImages(idxRemove) = [];
    trainingDataImages(idxRemove) = [];
    trainingLabelImages(idxRemove) = [];
    
    % Un-nest data out of cell arrays
    trainingFeatures = vertcat(trainingFeatureImages{:});
    trainingData = vertcat(trainingDataImages{:});
    trainingLabels = vertcat(trainingLabelImages{:});
    testingFeatures = vertcat(features{validationSet});
    testingLabels = vertcat(labels{validationSet});

    clear('trainingDataImages', 'trainingLabelImages', 'trainingFeatureImages');

    % Create synthetic features
    [synthFeatures, synthLabels] = dataAugmentation(trainingData, ...
        trainingLabels, nAugment, 'UseParallel', opts.UseParallel);
    trainingFeatures = vertcat(trainingFeatures, synthFeatures);
    trainingLabels = vertcat(trainingLabels, synthLabels);
    clear('synthFeatures', 'synthLabels');

    % Create Weights hyperparameter vector

    if strcmp(func2str(fitcfun),'NNet') && isfield(hyperparams,'CostRatio')
        Weights=ones(length(trainingLabels),1);
        Weights(trainingLabels)=hyperparams.CostRatio;
        hyperparams.Weights=Weights;
    end
    
    % Train the model
    models{i} = fitcfun(trainingFeatures, trainingLabels, hyperparams);

    % Predict labels on the validation set
    predLabels{i} = predict(models{i}, testingFeatures);

    % Compute performance metrics
    crossvalConfusion(:, :, i) = confusionmat(testingLabels, predLabels{i});

    % losses(i) = loss(models{i}, testingData, testingLabels, 'loss', @focalLoss);
    
    if opts.Progress
        progressbar([], [], []);
    end
end

if opts.Progress
    progressbar.release();
end

[accuracy, precision, recall, f2, f3, mcc] = analyzeConfusion(sum(crossvalConfusion, 3));
objective = -mcc;

constraints = [];

userdata.confusion = crossvalConfusion;
userdata.model = models;
userdata.accuracy=accuracy;
userdata.precision=precision;
userdata.recall=recall;
userdata.f2=f2;
userdata.f3=f3;
userdata.mcc = mcc;
userdata.predLabels = predLabels;
end
