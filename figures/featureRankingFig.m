% SPDX-License-Identifier: BSD-3-Clause
%%
clear

addpath('..');

datadir = '../../data';
load([datadir filesep 'training' filesep 'trainingData'], ...
    'trainingFeatures', 'trainingLabels');

features = vertcat(trainingFeatures{:});
labels = vertcat(trainingLabels{:});

%%
close all;
[idx, scores] = fscmrmr(features, labels);

%%
colors = colororder(brewermap([],'dark2'));

fig = figure('Units', 'inches', 'Position', [2 2 8.25 2.5]);

bar(scores(idx),'FaceColor',colors(1,:));
ylabel('Importance')
xticks(1:numel(idx))
xticklabels(features.Properties.VariableNames(idx))
set(gca, 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')

% fig.Visible = 'off';

%%
exportgraphics(fig, 'featureRanking.pdf', 'ContentType', 'vector')


