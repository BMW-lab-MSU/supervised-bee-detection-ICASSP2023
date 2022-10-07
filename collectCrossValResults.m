% SPDX-License-Identifier: BSD-3-Clause
%% Setup

datadir = '../data/insect-lidar';

modelNames = {'Adaboost', 'RUSBoost', 'Net'};

load([datadir filesep 'training' filesep 'trainingData'])

%% Collect cross validation results for all models
for model = modelNames
    tmp = load([datadir filesep 'training' filesep ...
        'hyperparameterTuning' model{:}]);
    tmp2 = load([datadir filesep 'training' filesep ...
        'samplingTuning' model{:}]);
    
    results.(model{:}).Params = tmp.bestParams;
    results.(model{:}).Undersampling = tmp2.result.undersamplingRatio;
    results.(model{:}).NAugment = tmp2.result.nAugment;

    
    results.(model{:}).Row.CVResults = ...
        tmp.results.UserDataTrace{tmp.results.IndexOfMinimumTrace(end)};
    results.(model{:}).Row.Confusion = ...
        sum(results.(model{:}).Row.CVResults.confusion, 3);
    
    [a,p,r,f2,mcc] = analyzeConfusion(results.(model{:}).Row.Confusion);
    results.(model{:}).Row.Accuracy = a;
    results.(model{:}).Row.Precision = p;
    results.(model{:}).Row.Recall = r;
    results.(model{:}).Row.F2 = f2;
    results.(model{:}).Row.MCC = mcc;

    results.(model{:}).Image.Confusion = ...
        imageConfusion(results.(model{:}).Row.CVResults.predLabels, ...
        trainingLabels, crossvalPartition);

    [a,p,r,f2,mcc] = analyzeConfusion(results.(model{:}).Image.Confusion);
    results.(model{:}).Image.Accuracy = a;
    results.(model{:}).Image.Precision = p;
    results.(model{:}).Image.Recall = r;
    results.(model{:}).Image.F2 = f2;
    results.(model{:}).Image.MCC = mcc;
end

save([datadir filesep 'training' filesep 'cvResults'], '-struct', 'results');
