%% Load relevant files
load('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\PSONetaugcost.mat')            %"result"
load('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\hyperparameterTuningNet.mat')  %"results"
load('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\trainingData.mat')
%% extract the optimal hyperparameter values
undersamplingRatio=result.US;
costRatio=result.costRatio;
nAugment=round(result.nAugment);
LayerSize=bestParams.LayerSizes;
Lambda=bestParams.Lambda;
Activations=string(bestParams.activations);
%%  create hyperparameter structure
hyperparams.CostRatio=costRatio;    %used to calculate weight vector in cvobjfun.m
hyperparams.Verbose=1;
hyperparams.LayerSizes=LayerSize;
hyperparams.Lambda=Lambda;
hyperparams.Activations=Activations;
hyperparams.Standardize=true;

%% train the model
[objective, ~, userdata] = cvobjfun(@NNet, hyperparams, undersamplingRatio, ...
        nAugment, crossvalPartition, trainingFeatures, trainingData, trainingLabels, ...
        imageLabels, 'Progress', false, 'UseParallel', false);

save('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\finalModelNet',"userdata")