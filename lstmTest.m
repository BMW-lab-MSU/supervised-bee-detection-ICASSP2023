% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';

if isempty(gcp('nocreate'))
    parpool('IdleTimeout', Inf);
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

%% Format data
trainingData2 = mat2cell(vertcat(trainingData{:}), ones(178*numel(trainingData),1), 1024);
trainingLabels2 = categorical(vertcat(trainingLabels{:}));

%% oversample
insectData = trainingData2(find(trainingLabels2(1:25000)=='true'));
[synthData,synthLabels] = createSyntheticData(cell2mat(insectData), 400, 'UseParallel', true);
% 
synthData = mat2cell(synthData, ones(height(synthData),1), 1024);
synthLabels = categorical(synthLabels);

%% 
bunchOfInsects = trainingData2(find(trainingLabels2(1:300000)=='true'));
bunchOfNotInsects = trainingData2(find(trainingLabels2(1:300000)=='false'));
bunchOfNotInsectsRand = bunchOfNotInsects(randperm(numel(bunchOfNotInsects),numel(bunchOfInsects)));
% [synthData,synthLabels] = createSyntheticData(cell2mat(insectData), 1, 'UseParallel', true);

%% Define LSTM

classWeights = [1/numel(find(trainingLabels2=='false')),1/numel(find(trainingLabels2=='true'))];
classWeights = classWeights/min(classWeights);

layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(100, OutputMode="last")
    dropoutLayer(0.2)
%     bilstmLayer(20, OutputMode="last")
%     dropoutLayer(0.2)
    fullyConnectedLayer(2)
    softmaxLayer
%    classificationLayer(Classes=unique(trainingLabels2),ClassWeights=classWeights)
%     focalLossLayer
    classificationLayer
   ];


options = trainingOptions("adam", ...
    ExecutionEnvironment="auto", ...
    GradientThreshold=1, ...
    MaxEpochs=200, ...
    MiniBatchSize=512, ...
    SequenceLength=1024, ...
    Verbose=1, ...
    Shuffle="every-epoch",...
    Plots="training-progress");
%     LearnRateSchedule="piecewise", ...
%     InitialLearnRate=0.01, ...

%% Train

% lstmNet = trainNetwork([trainingData2(1:25000); synthData], [trainingLabels2(1:25000); synthLabels], layers, options);
lstmNet = trainNetwork([bunchOfNotInsectsRand; bunchOfInsects], categorical([false(size(bunchOfNotInsectsRand));true(size(bunchOfInsects))]), layers, options);

%% Test

predLabels = classify(lstmNet, trainingData2(25001:50000));