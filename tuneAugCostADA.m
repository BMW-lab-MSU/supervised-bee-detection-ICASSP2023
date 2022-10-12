% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';

if isempty(gcp('nocreate'))
    parpool('IdleTimeout', Inf);
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

%% Tune cost and aug ratios

% Create the grid
costRatios = logspace(-1,7,9);
augs = round([0,logspace(0,log10(250),5)]);
[cR, nA] = ndgrid(costRatios, augs);
cR = reshape(cR, 1, numel(cR));
nA = reshape(nA, 1, numel(nA));

GRID_SIZE = numel(cR);

progressbar = ProgressBar(GRID_SIZE, 'Title', 'Grid search');

disp(['AdaBoost: cost/aug search'])

progressbar.setup([],[],[]);

for i = 1:GRID_SIZE
    tic
    costRatio=cR(i);
    nAugment=nA(i);
    undersamplingRatio=0.3;

    hyperparams.CostRatio=costRatio;    %used to calculate weight vector in cvobjfun.m
    hyperparams.ScoreTransform='doublelogit';
    hyperparams.Cost=[0,1;costRatio,0];
    hyperparams.ClassNames=logical([0,1]);
    hyperparams.LearnRate=0.1;

    [objective(i), ~, userdata{i}] = cvobjfun(@ADAboost, hyperparams, undersamplingRatio, ...
        nAugment, crossvalPartition, trainingFeatures, trainingData, trainingLabels, ...
        imageLabels, 'Progress', true, 'UseParallel', true);
    toc
    disp(append('Iteration number:', int2str(i)))
    disp(objective(i))

    progressbar([], [], []);
end

progressbar.release();

result.objective = objective;
result.userdata = userdata;
[~, minIdx] = min(result.objective);
result.CostRatio = cR(minIdx);
result.nAugment = nA(minIdx);
result.undersamplingRatio=undersamplingRatio;

save([datadir filesep 'training' filesep 'augCostTuningADA.mat'],...
    'result', 'result', '-v7.3');