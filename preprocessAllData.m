% SPDX-License-Identifier: BSD-3-Clause
%% Setup
clear

if isempty(gcp('nocreate'))
    parpool();
end

baseDataDir ='../data';
rawDataDir = [baseDataDir filesep 'raw'];
preprocessedDataDir = [baseDataDir filesep 'preprocessed'];
rawDataFile = 'scans.mat';

%% Load data
load([rawDataDir filesep rawDataFile])


%% Extract features
for scanNum = progress(1:numel(scans))
    scanFeatures = cellfun(@(X) extractFeatures(X, 'UseParallel', true), ...
        scans(scanNum).Data, 'UniformOutput', false);
    scans(scanNum).Features = scanFeatures;
end


%% Save preprocessed data
mkdir(preprocessedDataDir);
save([preprocessedDataDir filesep 'preprocessedScans.mat'], ...
    'scans', '-v7.3');
