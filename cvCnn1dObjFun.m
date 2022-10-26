function [objective, constraints, userdata] = cvCnn1dObjFun(fitcfun, hyperparams, crossvalPartition, data, labels, opts)
% cvobjfun Optimize hyperparameters via cross-validation

% SPDX-License-Identifier: BSD-3-Clause
arguments
    fitcfun (1,1) function_handle
    hyperparams
    crossvalPartition (1,1) cvpartition
    data (:,1) cell
    labels (:,1) cell
    opts.Progress (1,1) logical = false
    opts.UseParallel (1,1) logical = false
end

MAJORITY_LABEL = 0;

statset('UseParallel', opts.UseParallel);

crossvalConfusion = zeros(2, 2, crossvalPartition.NumTestSets);
% losses = nan(1, crossvalPartition.NumTestSets);
models = cell(1, crossvalPartition.NumTestSets);
predLabels = cell(1, crossvalPartition.NumTestSets);


parfor i = 1:crossvalPartition.NumTestSets
    % Get validation and training partitions
    validationSet = test(crossvalPartition, i); 
    trainingSet = training(crossvalPartition, i);
    
    trainingData = data(trainingSet);
    trainingLabels = labels(trainingSet);
    trainingImageLabels = cellfun(@(c) any(c), trainingLabels);

    % Undersample the majority class
    idxRemove = randomUndersample(...
        trainingImageLabels, MAJORITY_LABEL, ...
        'UndersamplingRatio', hyperparams.UndersamplingRatio, ...
        'Reproducible', true, 'Seed', i);
    
    trainingData(idxRemove) = [];
    trainingLabels(idxRemove) = [];

    % Create synthetic data
    [synthData, synthLabels] = createSyntheticData(vertcat(trainingData{:}), vertcat(trainingLabels{:}), hyperparams.nAugment);
    synthData = mat2cell(synthData, ones(height(synthData),1), width(synthData));

    % Format training data and labels
    trainingData2 = [mat2cell(vertcat(trainingData{:}), ones(178*numel(trainingData),1),1024); synthData];
    trainingLabels2 = categorical([vertcat(trainingLabels{:}); synthLabels]);
    

    
    % format testing data and labels
    % The data needs to be a cell array where each cell is a row from an image
    testingData = mat2cell(vertcat(data{validationSet}), ones(178*numel(data(validationSet)),1), 1024);
    % The deep learning toolbox needs categorical labels
    testingLabels = categorical(vertcat(labels{validationSet}));

    % Train the model
    models{i} = fitcfun(trainingData2, trainingLabels2, hyperparams);

    % Predict labels on the validation set
    predLabels{i} = classify(models{i}, testingData);

    % Compute performance metrics
    crossvalConfusion(:, :, i) = confusionmat(testingLabels, predLabels{i});

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
