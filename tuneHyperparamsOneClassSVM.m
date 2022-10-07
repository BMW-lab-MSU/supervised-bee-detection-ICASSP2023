% SPDX-License-Identifier: BSD-3-Clause
%% Setup
rng(0, 'twister');

datadir = '../data/insect-lidar';

if isempty(gcp('nocreate'))
    parpool();
end

%% Load data
load([datadir filesep 'training' filesep 'trainingData.mat']);

%% Tune one class SVM hyperparameters

nAugmented = linspace(0, 1500, 7);
nuGrid = linspace(0.1, 1, 7);
combinations = combvec(nAugmented, nuGrid);
augment = combinations(1,:);
nu = combinations(2,:);
GRID_SIZE = width(combinations);

progressbar = ProgressBar(GRID_SIZE, 'Title', 'Grid search');

disp(['one class SVM: grid search'])

progressbar.setup([],[],[]);

for i = 1:GRID_SIZE
    [objective(i), ~, userdata{i}] = cvobjfun(@oneClassSVM, nu(i), 0, ...
        augment(i), crossvalPartition, trainingFeatures, trainingData, trainingLabels, ...
        imageLabels, 'Progress', true, 'UseParallel', true);
    
    disp(objective(i))

    progressbar([], [], []);
end

progressbar.release();

result.objective = objective;
result.userdata = userdata;
[~, minIdx] = min(result.objective);
result.undersamplingRatio = under(minIdx);
result.nAugment = augment(minIdx);

save([datadir filesep 'training' filesep 'hyperparameterTuningOneClassSVM.mat'],...
    'results', 'result', '-v7.3');

%% Model fitting function
function model = oneClassSVM(data, labels, nu)
    alpha = zeros(size(data(labels == 1, :), 1), 1);
    alpha(randperm(size(alpha, 1), round(0.2 * size(alpha, 1)))) = 0.5;
    model = compact(fitcsvm(data(labels == 1, :),...
        labels(labels == 1), 'Nu', nu));
        
end
