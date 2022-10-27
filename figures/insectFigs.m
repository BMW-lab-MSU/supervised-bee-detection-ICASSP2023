% SPDX-License-Identifier: BSD-3-Clause
%% Insect example figures for IEEE RAPID 2020 paper

%%
RANGE_INCREMENT = 0.75;

%% load data and labels
addpath('..');

datadir = '../../data';
load([datadir filesep 'training' filesep 'trainingData']);
load([datadir filesep 'preprocessed' filesep 'preprocessedScans'])

%%
trainingImageLabels = imageLabels(training(holdoutPartition));

%% Get timestamps/resolution for the images
timestamps = vertcat(scans.Time);
trainingTimestamps = timestamps(training(holdoutPartition));

%%
% for i = find(trainingImageLabels)'
%     imagesc(trainingData{i}*-1)
%     disp(i)
%     pause
% end

%%
% this is a good example image: 1 body hit, 1 wing hit, 1 hard target
imageNum = 653;

t = trainingTimestamps{imageNum} * 1000; %ms

%% 
mainFig = figure('Units','inches','Position',[3,3,8,4])
mainLayout = tiledlayout(6,2);

%% Create the 2D image
% imageFig = figure('Units','inches','Position',[3,3,3.5,2]);

nexttile(mainLayout,2,[3,1])
imagesc(t, 1:height(trainingData{imageNum}), trainingData{imageNum});
colormap(flipud(brewermap([],'greys')))
% colorbar
% xlabel('ms')
% ylabel('range bin')
set(gca, 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')

%%
insect1RangeBin = 28;
insect2RangeBin = 32;
hardTargetRangeBin = 132;

%% Time domain plots
% timeDomainFig = figure('Units','inches','Position',[3,3,3.5,2]);
colors = colororder(brewermap([],'dark2'));

% timeDomainFigLayout = tiledlayout(3,1);

nexttile(7)
plot(t, trainingData{imageNum}(hardTargetRangeBin,:),'LineWidth',2,'Color',colors(1,:))
set(gca, 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')

nexttile(9)
plot(t, trainingData{imageNum}(insect1RangeBin,:),'LineWidth',2,'Color',colors(2,:))
set(gca, 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')

nexttile(11)
plot(t, trainingData{imageNum}(insect2RangeBin,:),'LineWidth',2,'Color',colors(3,:))
set(gca, 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')

% timeDomainFigLayout.TileSpacing = 'tight';
% timeDomainFigLayout.Padding = 'tight';
% timeDomainFigLayout.Children(3).XTickLabels = [];
% timeDomainFigLayout.Children(2).XTickLabel = [];
% ylabel(timeDomainFigLayout,'Volts','FontSize',9,'FontName','Times New Roman')
% xlabel(timeDomainFigLayout,'Time','FontSize',9,'FontName','Times New Roman')


%% frequency domain plots
Ts = mean(diff(t))/1000;
Fs = 1/Ts;
f = linspace(0,Fs/2,numel(t));

insect1Spectrum = abs(fft(trainingData{imageNum}(insect1RangeBin,:))).^2;
insect2Spectrum = abs(fft(trainingData{imageNum}(insect2RangeBin,:))).^2;
hardTargetSpectrum = abs(fft(trainingData{imageNum}(hardTargetRangeBin,:))).^2;

% freqDomainFig = figure('Units','inches','Position',[3,3,3.5,2]);
% freqDomainFigLayout = tiledlayout(3,1);

nexttile(8)
plot(f(1:500), hardTargetSpectrum(1:500),'LineWidth',2, 'Color',colors(1,:))
ylim([0 20])
xlim([0 1000])
set(gca, 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')

nexttile(10)
plot(f(1:500), insect1Spectrum(1:500),'LineWidth',2,'Color',colors(2,:))
ylim([0 130])
xlim([0 1000])
set(gca, 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')


nexttile(12)
plot(f(1:500), insect2Spectrum(1:500),'LineWidth',2,'Color',colors(3,:))
ylim([0 30])
xlim([0 1000])
set(gca, 'FontSize', 9)
set(gca, 'FontName', 'Times New Roman')

% 
% freqDomainFigLayout.TileSpacing = 'tight';
% freqDomainFigLayout.Padding = 'tight';
% freqDomainFigLayout.Children(3).XTickLabels = [];
% freqDomainFigLayout.Children(2).XTickLabel = [];

% xlabel(freqDomainFigLayout,'Frequency','FontSize',9,'FontName','Times New Roman')


%%
mainLayout.TileSpacing = 'compact';
mainLayout.Padding = 'compact';
mainLayout.Children(2).XTickLabel = [];
mainLayout.Children(3).XTickLabel = [];
mainLayout.Children(5).XTickLabel = [];
mainLayout.Children(6).XTickLabel = [];





%%
% exportgraphics(imageFig, 'insectImage.pdf','ContentType','vector')
% exportgraphics(timeDomainFig, 'timeDomainExamples.pdf','ContentType','vector')
% exportgraphics(freqDomainFig, 'freqDomainExamples.pdf','ContentType','vector')
exportgraphics(mainFig, 'insectExampleIntuition.pdf', 'ContentType','vector')