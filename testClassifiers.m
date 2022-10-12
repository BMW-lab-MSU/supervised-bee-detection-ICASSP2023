% SPDX-License-Identifier: BSD-3-Clause
%% Setup
clear

datadir = '../data';

%% Load data
load([datadir filesep 'testing' filesep 'testingData.mat']);

features = vertcat(testingFeatures{:});
labels = vertcat(testingLabels{:});


%% Test AdaBoost
disp('Testing AdaBoost....')
disp('---------------')
disp('')
load([datadir filesep 'training' filesep 'models' filesep 'ADABoost.mat']);

%%%%%%%%%%%%%%%%%%
% row results
%%%%%%%%%%%%%%%%%%
adaBoost.Row.PredLabels = predict(model, features);

adaBoost.Row.Confusion = confusionmat(labels, adaBoost.Row.PredLabels);

[a, p, r, f2, mcc] = analyzeConfusion(adaBoost.Row.Confusion);
adaBoost.Row.Accuracy = a;
adaBoost.Row.Precision = p;
adaBoost.Row.Recall = r;
adaBoost.Row.F2 = f2;
adaBoost.Row.MCC = mcc;

%%%%%%%%%%%%%%%%%%
% image results
%%%%%%%%%%%%%%%%%%
adaBoost.Image.Confusion = imageConfusion(adaBoost.Row.PredLabels, testingLabels, holdoutPartition);

[a, p, r, f2, mcc] = analyzeConfusion(adaBoost.Image.Confusion);
adaBoost.Image.Accuracy = a;
adaBoost.Image.Precision = p;
adaBoost.Image.Recall = r;
adaBoost.Image.F2 = f2;
adaBoost.Image.MCC = mcc;

%%%%%%%%%%%%%%%%%%
% Display results
%%%%%%%%%%%%%%%%%%
disp('Row results')
disp(adaBoost.Row.Confusion)
disp(adaBoost.Row)
disp('Image results')
disp(adaBoost.Image.Confusion)
disp(adaBoost.Image)

%% Test RUSBoost
disp('Testing RUSBoost....')
disp('---------------')
disp('')
load([datadir filesep 'training' filesep 'models' filesep 'RUSBoost.mat']);

%%%%%%%%%%%%%%%%%%
% row results
%%%%%%%%%%%%%%%%%%
rusBoost.Row.PredLabels = predict(model, features);

rusBoost.Row.Confusion = confusionmat(labels, rusBoost.Row.PredLabels);

[a, p, r, f2, mcc] = analyzeConfusion(rusBoost.Row.Confusion);
rusBoost.Row.Accuracy = a;
rusBoost.Row.Precision = p;
rusBoost.Row.Recall = r;
rusBoost.Row.F2 = f2;
rusBoost.Row.MCC = mcc;

%%%%%%%%%%%%%%%%%%
% image results
%%%%%%%%%%%%%%%%%%
rusBoost.Image.Confusion = imageConfusion(rusBoost.Row.PredLabels, testingLabels, holdoutPartition);

[a, p, r, f2, mcc] = analyzeConfusion(rusBoost.Image.Confusion);
rusBoost.Image.Accuracy = a;
rusBoost.Image.Precision = p;
rusBoost.Image.Recall = r;
rusBoost.Image.F2 = f2;
rusBoost.Image.MCC = mcc;

%%%%%%%%%%%%%%%%%%
% Display results
%%%%%%%%%%%%%%%%%%
disp('Row results')
disp(rusBoost.Row.Confusion)
disp(rusBoost.Row)
disp('Image results')
disp(rusBoost.Image.Confusion)
disp(rusBoost.Image)

%% Test neural net
disp('Testing neural net....')
disp('---------------')
disp('')
load([datadir filesep 'training' filesep 'models' filesep 'NNet.mat']);

%%%%%%%%%%%%%%%%%%
% row results
%%%%%%%%%%%%%%%%%%
nnet.Row.PredLabels = predict(model, features);

nnet.Row.Confusion = confusionmat(labels, nnet.Row.PredLabels);

[a, p, r, f2, mcc] = analyzeConfusion(nnet.Row.Confusion);
nnet.Row.Accuracy = a;
nnet.Row.Precision = p;
nnet.Row.Recall = r;
nnet.Row.F2 = f2;
nnet.Row.MCC = mcc;

%%%%%%%%%%%%%%%%%%
% image results
%%%%%%%%%%%%%%%%%%
nnet.Image.Confusion = imageConfusion(nnet.Row.PredLabels, testingLabels, holdoutPartition);

[a, p, r, f2, mcc] = analyzeConfusion(nnet.Image.Confusion);
nnet.Image.Accuracy = a;
nnet.Image.Precision = p;
nnet.Image.Recall = r;
nnet.Image.F2 = f2;
nnet.Image.MCC = mcc;

%%%%%%%%%%%%%%%%%%
% Display results
%%%%%%%%%%%%%%%%%%
disp('Row results')
disp(nnet.Row.Confusion)
disp(nnet.Row)
disp('Image results')
disp(nnet.Image.Confusion)
disp(nnet.Image)

%% Save results
save([datadir filesep 'testing' filesep 'results'], 'adaBoost', 'rusBoost', 'nnet');
