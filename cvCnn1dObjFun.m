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

if opts.Progress
    progressbar = ProgressBar(crossvalPartition.NumTestSets, ...
        'UpdateRate', inf, 'Title', 'Cross validation');
    progressbar.setup([], [], []);
end

for i = 1:crossvalPartition.NumTestSets
    % Get validation and training partitions
    validationSet = test(crossvalPartition, i); 
    trainingSet = training(crossvalPartition, i);
    
    trainingData = data(trainingSet);
    trainingLabels = labels(trainingSet);

    % % Undersample the majority class
    % idxRemove = randomUndersample(...
    %     imageLabel(trainingSet), MAJORITY_LABEL, ...
    %     'UndersamplingRatio', undersamplingRatio, ...
    %     'Reproducible', true, 'Seed', i);
    
    % trainingDataImages(idxRemove) = [];
    % trainingLabelImages(idxRemove) = [];
    
    % format testing data and labels
    % The data needs to be a cell array where each cell is a row from an image
    testingData = mat2cell(vertcat(data{validationSet}), ones(178*numel(data(validationSet)),1), 1024);
    % The deep learning toolbox needs categorical labels
    testingLabels = categorical(vertcat(labels{validationSet}));

    % Train the model
    models{i} = fitcfun(trainingData, trainingLabels, hyperparams);

    % Predict labels on the validation set
    predLabels{i} = classify(models{i}, testingData);

    % Compute performance metrics
    crossvalConfusion(:, :, i) = confusionmat(testingLabels, predLabels{i});

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