% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';
% 
% if isempty(gcp('nocreate'))
%     parpool('IdleTimeout', Inf);
% end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

%% Format data
trainingData2 = mat2cell(vertcat(trainingData{:}), ones(178*numel(trainingData),1), 1024);
trainingLabels2 = categorical(vertcat(trainingLabels{:}));

%% oversample
insectData = trainingData2(find(trainingLabels2(1:25000)=='true'));
[synthData,synthLabels] = createSyntheticData(cell2mat(insectData), 50, 'UseParallel', true);
% 
synthData = mat2cell(synthData, ones(height(synthData),1), 1024);
synthLabels = categorical(synthLabels);

%% 
% bunchOfInsects = trainingData2(find(trainingLabels2(1:300000)=='true'));
% bunchOfNotInsects = trainingData2(find(trainingLabels2(1:300000)=='false'));
% bunchOfNotInsectsRand = bunchOfNotInsects(randperm(numel(bunchOfNotInsects),numel(bunchOfInsects)));
% [synthData,synthLabels] = createSyntheticData(cell2mat(insectData), 1, 'UseParallel', true);

%%
classes = unique(trainingLabels2)';


numClasses = numel(classes);
for i=1:numClasses
    classFrequency(i) = sum(trainingLabels2(:) == classes(i));
    classWeights(i) = numel(trainingData2)/(numClasses*classFrequency(i));
end


%%
filterSize = 20;
numFilters = 20;

layers = [ ...
    sequenceInputLayer(1, MinLength=1024)
    convolution1dLayer(filterSize/4,numFilters)
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    convolution1dLayer(filterSize,2snumFilters)
    batchNormalizationLayer
    reluLayer
    dropoutLayer
    globalMaxPooling1dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer(Classes=classes,ClassWeights=classWeights)];

options = trainingOptions("adam", ...
    MaxEpochs=400, ...
    InitialLearnRate=0.01, ...
    SequenceLength=1024, ...
    Verbose=false, ...
    MiniBatchSize=2048, ...
    Shuffle="every-epoch", ...
    Plots="training-progress");

%% Train

cnn = trainNetwork([trainingData2(1:25000); synthData], [trainingLabels2(1:25000); synthLabels], layers, options);
% cnn = trainNetwork(trainingData2(1:25000), trainingLabels2(1:25000), layers, options);
% lstmNet = trainNetwork([bunchOfNotInsectsRand; bunchOfInsects], categorical([false(size(bunchOfNotInsectsRand));true(size(bunchOfInsects))]), layers, options);

%% Test

predLabelsTrain = classify(cnn, trainingData2(1:25000));
confusionmat(trainingLabels2(1:25000), predLabelsTrain)

predLabelsVal = classify(cnn, trainingData2(25001:50000));
confusionmat(trainingLabels2(25001:50000), predLabelsVal)