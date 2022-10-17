% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';
waveletdir = [datadir filesep 'training' filesep 'wavelets'];


if isempty(gcp('nocreate'))
    parpool('IdleTimeout', Inf);
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat'], 'trainingLabels', 'trainingData');

trainingData2 = mat2cell(vertcat(trainingData{:}), ones(178*numel(trainingData),1), 1024);
trainingLabels2 = categorical(vertcat(trainingLabels{:}));

%%
wavelets = imageDatastore(waveletdir, 'FileExtensions','.tiff');
wavelets.Labels = trainingLabels2;

%%
trainingWavelets = wavelets.partition(25,1);
validationWavelets = wavelets.partition(25,2);


%% Format data

% trainingData2 = cat(4, trainingData{:});
% trainingImageLabels = categorical(cellfun(@(c) any(c), trainingLabels, 'UniformOutput',true));

%% Extract wavelets

% nRows = numel(trainingLabels2);
% 
% waveletFilterbank = cwtfilterbank;
% 
% % waveletdir = [datadir filesep 'training' filesep 'wavelets'];
% % mkdir(waveletdir);
% 
% parfor(i = 1:nRows)
%     cwavelet = abs(wt(waveletFilterbank,trainingData2{i,:}).^2);
%     cwavelet = uint16(rescale(cwavelet, 0, 2^16 - 1));
%     imwrite(cwavelet, [waveletdir filesep 'trainingWaveletRow' num2str(i,'%06d') '.tiff'], 'tiff');
% end

%%

%%
classes = categories(trainingLabels2);
classWeights = 1./countcats(trainingLabels2);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(classes);


dropoutProb = 0.2;
numF = 24;
layers = [
    imageInputLayer([71, 1024, 1])
    
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
    globalMaxPooling2dLayer
    dropoutLayer(dropoutProb)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer(Classes=classes,ClassWeights=classWeights)];

miniBatchSize = 128;
options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=30, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false);

%%
net = trainNetwork(trainingWavelets, layers, options);

%%
predLabelsTrain = classify(net, trainingWavelets);
confusionmat(trainingWavelets.Labels, predLabelsTrain)

predLabelsVal = classify(net, validationWavelets);
confusionmat(validationWavelets.Labels, predLabelsVal)