function features = extractTFStats(cwavelet, opts)

arguments
    cwavelet
    opts.UseParallel = false
end

[range,~] = size(cwavelet);
if range < 1
    max_mean = [];
    max_std = [];
    avg_mean = [];
    avg_skewness = [];
    max_peak = [];
    max_diff = [];
    brk_up = [];
end


for(i = 1:range)
    max_mean(i) = max(mean(cwavelet{i},2));
    max_std(i) = max(std(cwavelet{i},0,2));
    avg_mean(i) = mean(mean(cwavelet{i},2));
    avg_skewness(i) = mean(skewness(cwavelet{i},1,2));
    max_peak(i) = max(max(cwavelet{i}));
    max_diff(i) = max(max(diff(cwavelet{i})));
    for j = 0:15
        rc(j+1) = sum(sum(cwavelet{i}(:,(64*j + 1 : 64*(j+1)))))/(71*64);
    end
    brk_up(i) = sum(rc)/16;
end
features = table;
features.max_mean = max_mean';
features.max_std = max_std';
features.avg_mean = avg_mean';
features.avg_skewness = avg_skewness';
features.max_peak = max_peak';
features.max_diff = max_diff';
features.brk_up = brk_up';
end
