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
% trainingData2 = mat2cell(vertcat(trainingData{:}), ones(178*numel(trainingData),1), 1024);
% trainingLabels2 = categorical(vertcat(trainingLabels{:}));
trainingData2 = cat(4, trainingData{:});
trainingImageLabels = categorical(cellfun(@(c) any(c), trainingLabels, 'UniformOutput',true));

%%
classes = categories(trainingImageLabels);
classWeights = 1./countcats(trainingImageLabels);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(classes);


dropoutProb = 0.2;
numF = 24;
layers = [
    imageInputLayer([size(trainingData{1}), 1])
    
    convolution2dLayer(3,numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer([3 10],2*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,5*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,6*numF,Padding="same")
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([23, 128])
    dropoutLayer(dropoutProb)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer(Classes=classes,ClassWeights=classWeights)];

miniBatchSize = 64;
options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=30, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false);

%%
a = trainingData2(:,:,:,1:2000);
net = trainNetwork(a, trainingImageLabels(1:2000), layers, options);

%%
predLabelsTrain = classify(net, trainingData2(:,:,:,1:2000));
confusionmat(trainingImageLabels(1:2000), predLabelsTrain)

predLabelsVal = classify(net, trainingData2(:,:,:,2001:end));
confusionmat(trainingImageLabels(2001:end), predLabelsVal)