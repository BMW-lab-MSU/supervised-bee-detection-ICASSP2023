% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = 'C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data';

% if isempty(gcp('nocreate'))
%     parpool();
% end

%% Load data
load([datadir filesep 'training3' filesep 'trainingData.mat']);

%% Tune cost and aug ratios

% Create the grid
costRatios = logspace(-1,7,9);
augs = round([0,logspace(0,log10(250),5)]);
[cR, nA] = ndgrid(costRatios, augs);
cR = reshape(cR, 1, numel(cR));
nA = reshape(nA, 1, numel(nA));

GRID_SIZE = numel(cR);

%progressbar = ProgressBar(GRID_SIZE, 'Title', 'Grid search');

disp(['two class SVM: cost/aug search'])

%progressbar.setup([],[],[]);

for i = 1:GRID_SIZE
    tic
    costRatio=cR(i);
    nAugment=nA(i);
    undersamplingRatio=0.3;

    hyperparams=struct();
    hyperparams.Cost=[0,1;costRatio,0];
    hyperparams.ClassNames=logical([0,1]);
    hyperparams.ShrinkagePeriod=1000;
    hyperparams.Solver='ISDA';
    hyperparams.IterationLimit=150E3;
    hyperparams.Verbose=0;

    [objective(i), ~, userdata{i}] = cvobjfun(@twoClassSVM, hyperparams, undersamplingRatio, ...
        nAugment, crossvalPartition, trainingFeatures, trainingData, trainingLabels, ...
        imageLabels, 'Progress', false, 'UseParallel', false);
    toc
    disp(append('Iteration number:', int2str(i)))
    disp(objective(i))

    %progressbar([], [], []);
end

%progressbar.release();

result.objective = objective;
result.userdata = userdata;
[~, minIdx] = min(result.objective);
result.CostRatio = cR(minIdx);
result.nAugment = nA(minIdx);
result.undersamplingRatio=undersamplingRatio;

save([datadir filesep 'training3' filesep 'augCostTuning2ClassSVM.mat'],...
    'result', 'result', '-v7.3');