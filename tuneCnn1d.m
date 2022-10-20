% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data';

% if isempty(gcp('nocreate'))
%     parpool('IdleTimeout', Inf);
% end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);


%% "tune" parameters; for now, just checking if the framework works...

% empty struct for now since we aren't tuning anything
hyperparams = struct();

[objective, ~, userdata] = cvCnn1dObjFun(@fitCnn1d, hyperparams, ...
    crossvalPartition, trainingData, trainingLabels, ...
    'Progress', true, 'UseParallel', true);


result.objective = objective;
result.userdata = userdata;
[~, minIdx] = min(result.objective);

% save([datadir filesep 'training' filesep 'augCostTuningADA.mat'],...
    % 'result', 'result', '-v7.3');
