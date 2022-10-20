function model = fitCnn1d(data, labels, hyperparams)
arguments
    data (:,1) cell 
    labels (:,1) cell 
    hyperparams (1,1) struct
end

% SPDX-License-Identifier: BSD-3-Clause

%% Format data
% The data needs to be a cell array where each cell is a row from an image
trainingData = mat2cell(vertcat(data{:}), ones(178*numel(data),1), 1024);

% The deep learning toolbox needs categorical labels
trainingLabels = categorical(vertcat(labels{:}));

%%
classes = categories(trainingLabels);
classWeights = 1./countcats(trainingLabels);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(classes);

%%
filterSize = 20;
numFilters = 20;

layers = [ ...
    sequenceInputLayer(1, MinLength=1024)
    convolution1dLayer(filterSize/4,numFilters)
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    convolution1dLayer(filterSize,2*numFilters)
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    globalMaxPooling1dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer(Classes=classes,ClassWeights=classWeights)];

options = trainingOptions("adam", ...
    MaxEpochs=10, ...
    InitialLearnRate=0.01, ...
    SequenceLength=1024, ...
    Verbose=false, ...
    MiniBatchSize=2048, ...
    Shuffle="every-epoch", ...
    Plots="training-progress");

%% Train

model = trainNetwork(trainingData, trainingLabels, layers, options);
