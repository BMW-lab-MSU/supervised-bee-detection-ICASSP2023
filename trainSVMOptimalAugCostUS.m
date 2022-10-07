%%Load relevant files
load('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\PSOSVMaugcost.mat')
load('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\trainingData.mat')
%%
undersamplingRatio=result.US;
costRatio=result.costRatio;
nAugment=result.nAugment;
%%
%trainingFeatures=nestedcell2mat(trainingFeatures);
%trainingLabels=nestedcell2mat(trainingLabels);
%% 
hyperparams=struct();
hyperparams.ShrinkagePeriod=1000;
hyperparams.Solver='ISDA';
hyperparams.IterationLimit=250E3;
hyperparams.Verbose=1;
hyperparams.Cost=[0,1;costRatio,0];
hyperparams.ClassNames=logical([0,1]);

[objective, ~, userdata] = cvobjfun(@twoClassSVM, hyperparams, undersamplingRatio, ...
        nAugment, crossvalPartition, trainingFeatures, trainingData, trainingLabels, ...
        imageLabels, 'Progress', false, 'UseParallel', false);

save('C:\Users\v59g786\Desktop\REU_project\code_by_walden\revisedPipeline\data\training3\SVM_w_optimal_imbal_params.mat',"userdata")