function [newFeatures, labels] = dataAugmentation(data, labels, nAugmented, opts)

% SPDX-License-Identifier: BSD-3-Clause
arguments
    data (:,:) {mustBeNumeric}
    labels (:, 1) logical
    nAugmented (1,1) {mustBeNumeric, mustBeNonnegative}
    opts.UseParallel (1,1) logical = false
end

if opts.UseParallel
    nWorkers = gcp('nocreate').NumWorkers;
else
    nWorkers = 0;
end

insectIdx = find(labels == 1);

synthData = createSyntheticData(data(insectIdx, :), nAugmented, nWorkers);

newFeatures = extractFeatures(synthData, 'UseParallel', opts.UseParallel);

labels = true(height(newFeatures), 1);
end


function synthData = createSyntheticData(data, nAugmented, nWorkers)
N_VARIATIONS = 6;
nInsects = height(data);

synthDataCell = cell(nInsects, 1);

parfor (i = 1:nInsects, nWorkers)

    synthDataTmp = zeros(nAugmented, width(data), 'like', data);

    insect = data(i,:);

    stop = 0;

    for j = 1:ceil(nAugmented/N_VARIATIONS)
        variations = zeros(N_VARIATIONS, width(data), 'like', data);

        variations(1, :) = circshift(insect, randi(width(data), 1));
        variations(2, :) = fliplr(insect);
        variations(3, :) = interpolate(variations(1, :));
        variations(4, :) = decimate_(variations(1, :));
        variations(5, :) = interpolate(variations(2, :));
        variations(6, :) = decimate_(variations(2, :));

        % add noise to the variations
        variations = variations + ...
            randn(N_VARIATIONS, width(data), 'like', data) * ...
            max(variations, [], 'all')/2;

        % the original lidar data is normalized between 0 and 1, so we do the
        % same to the new data;
        variations = normalize(variations, 2, 'range');

        % put the variations into the correct location in synthData
        if nAugmented < j * N_VARIATIONS
            nAugmentedLeft = mod(nAugmented, (j - 1) * N_VARIATIONS);
            
            start = stop + 1;
            stop = start + nAugmentedLeft - 1;
            
            if stop > height(synthDataTmp)
                stop = height(synthDataTmp)
            end
            
            synthDataTmp(start:stop, :) = variations(1:nAugmentedLeft, :);
        else
            start = stop + 1;
            stop = start + N_VARIATIONS - 1;
            
            if stop > height(synthDataTmp)
                stop = height(synthDataTmp)
            end
            
            synthDataTmp(start:stop, :) = variations;
        end
    end
    synthDataCell{i} = synthDataTmp;
end

synthData = cell2mat(synthDataCell);
end


function y = interpolate(x)
    % interpolate by a factor of 3
    y = interp(x, 3);

    % remove random values until y is the same size as x
    y(randperm(width(y), width(y) - width(x))) = [];
end

function y = decimate_(x)
    % decimate by 2
    if ~isa(class(x), 'double')
        y = decimate(double(x), 2, 'FIR');
        y = cast(y, 'like', x);
    else
        y = decimate(x, 2, 'FIR');
    end


    % pad y with it's mean until it is the same size as x
    y = [y, repelem(mean(y), width(x) - width(y))];
end