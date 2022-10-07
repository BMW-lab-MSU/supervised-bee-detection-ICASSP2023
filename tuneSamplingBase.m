function result = tuneSamplingBase(fitcfun, features, data, labels, imageLabel, crossvalPartition, opts)

% SPDX-License-Identifier: BSD-3-Clause
arguments
    fitcfun (1,1) function_handle
    features (:,1) cell
    data (:,1) cell
    labels (:,1) cell
    imageLabel (:,1) logical
    crossvalPartition (1,1) cvpartition
    opts.Progress (1,1) logical = false
    opts.UseParallel (1,1) logical = false
    opts.NumThreads (1,1) int8 = 1
end

name = functions(fitcfun).function;

% Create the grid
undersampling = linspace(0, 0.9, 7);
nAugmented = linspace(0, 1500, 7);
[under, augment] = ndgrid(undersampling, nAugmented);
under = reshape(under, 1, numel(under));
augment = reshape(augment, 1, numel(augment));

GRID_SIZE = numel(under);

% Preallocate data structures for grid search results
objective = zeros(1, GRID_SIZE);
userdata = cell(1, GRID_SIZE);

if opts.Progress
    progressbar = ProgressBar(GRID_SIZE, 'Title', 'Grid search');
end

% Training
disp([name, ': grid search'])

if opts.Progress
    progressbar.setup([],[],[]);
end

for i = 1:GRID_SIZE
    [objective(i), ~, userdata{i}] = cvobjfun(fitcfun, [], under(i), ...
        augment(i), crossvalPartition, features, data, labels, ...
        imageLabel, 'Progress', opts.Progress, 'UseParallel', opts.UseParallel);

    if opts.Progress
        progressbar([], [], []);
    end
end

if opts.Progress
    progressbar.release();
end

% Find the undersampling ratio that resulted in the minimum objective
result.objective = objective;
result.userdata = userdata;
[~, minIdx] = min(result.objective);
result.undersamplingRatio = under(minIdx);
result.nAugment = augment(minIdx);
