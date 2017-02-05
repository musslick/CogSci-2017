%% Plots for Fig. 3a

% author: Sebastian Musslick

%% data settings
clear all;
clc;

datafilePrefix ='Simulation1_3P5F_v7_h100_r';       

performanceMeasure = 'PCorrect';   

repetitionIdx = 1;

% load data
logFolder = 'logfiles/';
files = dir(logFolder);

% get valid file names
validFileNames = {};
for i =1:length(files)
    % check if this is a desired data file
    if(~isempty(strfind(files(i).name, datafilePrefix)))
        validFileNames{end+1} = files(i).name;
    end
end
if(~isempty(validFileNames))
    numRepetitions = length(validFileNames);
    % load every valid file
    for fileIdx = 1:length(validFileNames);
        disp(['loading ' validFileNames{fileIdx} '...']);
        load(strcat(logFolder, validFileNames{fileIdx}));
        % initial setup of batch_log_tmp
        if(fileIdx == 1)
            batch_log_tmp = repmat(batch_log(1,1), numRepetitions, 1);
        end
        batch_log_tmp(fileIdx) = batch_log(1,1);
    end
else
    error('No valid file names found');
end
batch_log = batch_log_tmp;

% plot settings

fontSize_title = 14;
fontSize_gca = 14;
fontSize_xlabel = 14;
fontSize_ylabel = 14;
fontSize_legend = 14;

fontName = 'Helvetica';
markerSize = 50;
sem_width = 2;
sem_marker = '-';

lineWidth = 3;

colors = [253 120 21; ... % orange
              31 104 172; ... % blue
              44 155 37; ... % green
              0     0   0  ; ... % black
            142 142 142; ... % grey 
            255 255 255] / 255; % white 
        
cContrast1 = 1;
cContrast2 = 2;
cContrast3 = 3;
cSingle = 4;
cWeak = 5;
cWhite = 6;

% plot graph analysis results

close all;

%% prepare data

multiPerformance  = nan(numRepetitions, length(taskSimilarities), NPathways);
learningPerformance = nan(numRepetitions, length(taskSimilarities));
weightCorr_final = nan(numRepetitions, length(taskSimilarities));

basis_template = eye(nTasks,nTasks);
for row = 1:size(basis_template,1)
    basis_template(row, (ceil(row/NPathways)-1)*NPathways+(1:NPathways)) = 1;
    basis_template(row, row:end) = 0;
end

for i = 1:length(taskSimilarities)
    
    for rep = 1:numRepetitions
        
        % learning speed
        learnIdx = find(batch_log(rep).MSE_log(i,:) > 0);
        learnIdx = learnIdx(end);
        learningPerformance(rep, i) = learnIdx;
        
        % learned weight correlations
        hiddenTaskRepWeights = squeeze(batch_log(rep).hiddenTaskRepWeights(i, learnIdx, :,:));
        weightCorr_final(rep, i) = mean(hiddenTaskRepWeights(basis_template == 1));
        
        
        % merge data
        switch performanceMeasure
            case 'MSE'
                % multitasking performance
                for cap = 2:NPathways
                    multiPerformance(rep, i, cap) = batch_log(rep).MSE_multi_inc{i, learnIdx, cap};
                end

            case 'PCorrect'
                % multitasking performance
                for cap = 2:NPathways
                    multiPerformance(rep, i, cap) = batch_log(rep).PCorrect_multi_inc{i, learnIdx, cap};
                end
        end
       
    end
    
end

%% Fig 3b: Learned Weight Similarity vs. Multitasking Performance vs. Feature Overlap

cap = 2;
plotSEM = 1;

% plot data
figure(1)

x = nanmean(weightCorr_final,1);
x_sem = nanstd(weightCorr_final,[],1)/sqrt(size(weightCorr_final,1));
y = nanmean(multiPerformance(:,:,cap),1);
y_sem = nanstd(multiPerformance(:,:,cap), [], 1)/sqrt(size(multiPerformance(:,:,cap),1));
z = taskSimilarities;

% create scatter plot
if(plotSEM)
    e_v = errorbar(x, y, y_sem, '.', 'Color', 'k'); hold on;
    e_v.LineStyle = 'none';
    e_v.LineWidth = lineWidth-2;
    e_h = herrorbar(x, y, x_sem, x_sem, '.'); hold on;
    e_h(1).LineWidth = lineWidth-2;
    e_h(1).Color = [0 0 0];
end

% compute axis parameters
switch performanceMeasure
    case 'MSE'
        measurementLabel = 'MSE';
        ygranularity = 1000;
        ymargin = 0.01;
    case 'PCorrect'   
        %measurementLabel = 'P(Correct)';
        y = y*100;
        y_sem = y_sem*100;
        measurementLabel = 'Accuracy (%)';
        ygranularity = 1000;
        ymargin = 0.04;
end
ylimit = [floor(min(y-y_sem)*ygranularity)/ygranularity ceil(max(y+y_sem)*ygranularity)/ygranularity] .* [1-ymargin 1+ymargin];
xgranularity = 10;
xmargin = 0.02;
xlimit = [floor(min(x)*xgranularity)/xgranularity ceil(max(x)*xgranularity)/xgranularity] .* [1-xmargin 1+xmargin];

scatter(x, y, markerSize, z, 'filled', 'MarkerEdgeColor',[0 0 0]);
hold off;


set(gca, 'FontSize', fontSize_gca-2, 'FontName', fontName);
xlim(xlimit);
ylim(ylimit);

ylabel(['Multitasking ' measurementLabel],'FontSize', fontSize_gca, 'FontName', fontName, 'Color', 'k');
xlabel(['Learned Task Correlation'],'FontSize', fontSize_gca, 'Color', 'k');

hcb = colorbar;
caxis([0 1]);
set(hcb,'Color','k');
ylabel(hcb, 'Feature Overlap','FontSize', fontSize_gca-2, 'Color', 'k');
set(gcf, 'Color', 'k');
set(gcf, 'Position', [500 500 280 200]);

hold off;

% added for white figures

set(gcf, 'Color', 'w');
set(gca, 'Color', 'w');
set(gca, 'xColor', 'k');
set(gca, 'yColor', 'k');

% preserve background color when saving figure
fig = gcf;
fig.InvertHardcopy = 'off';

