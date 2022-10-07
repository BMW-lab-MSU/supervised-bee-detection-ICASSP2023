%%Load relevant files
load('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\PSOADAaugcost.mat')
load('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\trainingData.mat')
%%
undersamplingRatio=result.US;
costRatio=result.costRatio;
nAugment=round(result.nAugment);
%%
%trainingFeatures=nestedcell2mat(trainingFeatures);
%trainingLabels=nestedcell2mat(trainingLabels);
%% 
hyperparams.Verbose=1;
hyperparams.ScoreTransform='doublelogit';
hyperparams.Cost=[0,1;costRatio,0];
hyperparams.ClassNames=logical([0,1]);
hyperparams.LearnRate=0.1;

[objective, ~, userdata] = cvobjfun(@ADAboost, hyperparams, undersamplingRatio, ...
        nAugment, crossvalPartition, trainingFeatures, trainingData, trainingLabels, ...
        imageLabels, 'Progress', false, 'UseParallel', false);

save('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\ADA_w_optimal_imbal_params.mat',"userdata")