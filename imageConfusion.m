function confmat = imageConfusion(pred, target, partition)

% SPDX-License-Identifier: BSD-3-Clause

% split predicted labels for each test set into 178x1 vectors for each image
if iscell(pred)
    tmp = cellfun(@(c) mat2cell(c, 178*ones(1,numel(c)/178), 1), pred, 'UniformOutput', false);
    % group the image labels into scans; ensure that the images end up in the appropriate scan position using the cv split indices; we need this primarily so the predicted labels are in the same order as the ground truth labels
    for i = 1:partition.NumTestSets
        predimageLabels(test(partition, i)) = mat2cell(tmp{i}, cellfun('length', target(test(partition, i))), 1);
    end
else
    tmp = mat2cell(pred, 178*ones(1,numel(pred)/178), 1);
    % group the image labels into scans; ensure that the images end up in the appropriate scan position using the cv split indices; we need this primarily so the predicted labels are in the same order as the ground truth labels
        predimageLabels = mat2cell(tmp, cellfun('length', target), 1);
end

predimageLabels = predimageLabels';

% get vectors of image labels
trueImageLabels = cellfun(@(c) any(c), vertcat(target{:}));
predImageLabels = cellfun(@(c) any(c), vertcat(predimageLabels{:}));

confmat = confusionmat(trueImageLabels, predImageLabels);
