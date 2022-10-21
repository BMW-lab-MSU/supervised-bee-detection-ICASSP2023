function model = fitLstm(data, labels, hyperparams)
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
layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100, OutputMode="last")
    dropoutLayer(0.2)
%     bilstmLayer(20, OutputMode="last")
%     dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer
%     focalLossLayer
    classificationLayer(Classes=classes,ClassWeights=classWeights)];


options = trainingOptions("adam", ...
    ExecutionEnvironment="auto", ...
    GradientThreshold=1, ...
    MaxEpochs=10, ...
    MiniBatchSize=256, ...
    SequenceLength=1024, ...
    Verbose=1, ...
    Shuffle="every-epoch",...
    Plots="training-progress");

%% Train

model = trainNetwork(trainingData, trainingLabels, layers, options);
