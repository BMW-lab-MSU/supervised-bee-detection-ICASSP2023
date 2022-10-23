% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';

if isempty(gcp('nocreate'))
    numGPUs = gpuDeviceCount("available");
    parpool(numGPUs, 'IdleTimeout', Inf);
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);


%% "tune" parameters; for now, just checking if the framework works...

costRatios = logspace(-1,7,9);

GRID_SIZE = numel(costRatios);

progressbar = ProgressBar(GRID_SIZE, 'Title', 'Grid search');

disp(['CNN: class weight search'])

progressbar.setup([],[],[]);

for i = 1:GRID_SIZE
    costRatio=costRatios(i);
    undersamplingRatio=0.3;

    hyperparams.UndersamplingRatio = undersamplingRatio;
    hyperparams.CostRatio=costRatio;    %used to calculate weight vector in cvobjfun.m
    hyperparams.Cost=[0,1;costRatio,0];


    [objective, ~, userdata] = cvCnn2dObjFun(@fitCnn2d, hyperparams, ...
        crossvalPartition, trainingData, trainingLabels, ...
        'Progress', true, 'UseParallel', true);

end

result.objective = objective;
result.userdata = userdata;
[~, minIdx] = min(result.objective);

% save([datadir filesep 'training' filesep 'augCostTuningADA.mat'],...
    % 'result', 'result', '-v7.3');
