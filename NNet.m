    %% Model fitting function
    function model = NNet(data, labels, hyperparams)

        hyperparams=rmfield(hyperparams,"CostRatio");%not used here,
            %but needed in cvobjfun.m and hyperTuneObjFun.m and its more
            %convenient to just delete it inside this function
        hyperparams=namedargs2cell(hyperparams);
        model = compact(fitcnet(data, labels,hyperparams{:}));
    end