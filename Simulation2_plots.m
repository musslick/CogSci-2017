%% data settings
clear all;
clc;

testedFeatureOverlaps = [0 0.8 1];
for i = 1:length(testedFeatureOverlaps)
    datafilePrefix{i} = ['Simulation2_3P5F_v7_h100_FO' num2str(testedFeatureOverlaps(i)*100) '_r'];   % 100 % feature overlap
end

performanceMeasure = 'PCorrect';   % alternatives: 'MSE', 'Accuracy', 'PCorrect'

repetitionIdx = 1;

% load data
logFolder = 'logfiles/';
files = dir(logFolder);

for featureOverlapIdx = 1:length(testedFeatureOverlaps)

    disp(['Loading files for ' num2str(testedFeatureOverlaps(featureOverlapIdx)*100) '% feature overlap...']);
    % get valid file names
    validFileNames = {};
    for i =1:length(files)
        % check if this is a desired data file
        if(~isempty(strfind(files(i).name, datafilePrefix{featureOverlapIdx})))
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
                batch_log_tmp{featureOverlapIdx} = repmat(batch_log(1,1), numRepetitions, 1);
            end
            batch_log_tmp{featureOverlapIdx}(fileIdx) = batch_log(1,1);
        end
    else
        error('No valid file names found');
    end

end

batch_log = batch_log_tmp;

% plot settings

fontSize_title = 14;
fontSize_gca = 14;
fontSize_xlabel = 14;
fontSize_ylabel = 14;
fontSize_legend = 14;

fontName = 'Helvetica';
markerSize = 38;
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

basis_template = eye(nTasks,nTasks);
for row = 1:size(basis_template,1)
    basis_template(row, (ceil(row/NPathways)-1)*NPathways+(1:NPathways)) = 1;
    basis_template(row, row:end) = 0;
end


for featureOverlapIdx = 1:length(testedFeatureOverlaps)

    genPerformance_full{featureOverlapIdx} = nan(numRepetitions, length(init_taskCorrelations), iterations_Train);
    trainPerformance_full{featureOverlapIdx} = nan(numRepetitions, length(init_taskCorrelations), iterations_Train);
    multiPerformance{featureOverlapIdx}  = nan(numRepetitions, length(init_taskCorrelations), NPathways);
    learningPerformance{featureOverlapIdx} = nan(numRepetitions, length(init_taskCorrelations));
    weightCorr_final{featureOverlapIdx} = nan(numRepetitions, length(init_taskCorrelations));


    for i = 1:length(init_taskCorrelations)

        for rep = 1:numRepetitions

            % learning speed
            learnIdx = find(batch_log{featureOverlapIdx}(rep).MSE_log(i,:) > 0);
            learnIdx = learnIdx(end);
            learningPerformance{featureOverlapIdx}(rep, i) = learnIdx;
            
            % learned weight correlations
            hiddenTaskRep = squeeze(batch_log{featureOverlapIdx}(rep).hiddenTaskRepWeights(i, learnIdx, :,:));
            weightCorr_final{featureOverlapIdx}(rep, i) = mean(hiddenTaskRep(basis_template == 1));
            
            % testing performance
            trainPerformance_full{featureOverlapIdx}(rep, i, 1:size(batch_log{featureOverlapIdx}(rep).train_MSE,2)) = batch_log{featureOverlapIdx}(rep).train_MSE(i, :);
            % generalization performance
            genPerformance_full{featureOverlapIdx}(rep, i, 1:size(batch_log{featureOverlapIdx}(rep).test_MSE,2)) = batch_log{featureOverlapIdx}(rep).test_MSE(i, :);

            % merge data
            switch performanceMeasure
                case 'MSE'
                    % multitasking performance
                    for cap = 2:NPathways
                        multiPerformance{featureOverlapIdx}(rep, i, cap) = batch_log{featureOverlapIdx}(rep).MSE_multi_inc{i, learnIdx, cap};
                    end
                    
                case 'PCorrect'
                    % multitasking performance
                    for cap = 2:NPathways
                        multiPerformance{featureOverlapIdx}(rep, i, cap) = batch_log{featureOverlapIdx}(rep).PCorrect_multi_inc{i, learnIdx, cap};
                    end
            end

        end

    end

    taskSimilarities{featureOverlapIdx} = nanmean(batch_log{featureOverlapIdx}(1).init_taskCorr,2);

end

%% Fig 5a: Multitasking Performance vs. Learning Speed vs. Initial Weight Condition (for environment with full feature overlap)

cap = 2;
plotSEM = 1;

% plot data
figure(1)
featureOverlapIdx = 3;

x = nanmean(learningPerformance{featureOverlapIdx}(:,:),1);
x_sem = nanstd(learningPerformance{featureOverlapIdx}, [], 1)/sqrt(size(learningPerformance{featureOverlapIdx},1));

switch performanceMeasure
    case 'MSE'
        scale = 1;
    case 'PCorrect'
        scale = 100;
end
y = nanmean(multiPerformance{featureOverlapIdx}(:,:,cap),1)*scale; 
y_sem = nanstd(multiPerformance{featureOverlapIdx}(:,:,cap), [], 1)/sqrt(size(multiPerformance{featureOverlapIdx}(:,:,cap),1))*scale;
z = taskSimilarities{featureOverlapIdx};

% create scatter plot
if(plotSEM)
    e_v = errorbar(x, y, y_sem, '.', 'Color', 'k'); hold on;
    e_v.LineStyle = 'none';
    e_v.LineWidth = lineWidth-2;
    e_h = herrorbar(x, y, x_sem, x_sem, '.'); hold on;
    e_h(1).LineWidth = lineWidth-2;
    e_h(1).Color = [0 0 0];
end
scatter(x, y, markerSize, z, 'filled', 'MarkerEdgeColor',[0 0 0]);

hold off

% compute axis parameters
switch performanceMeasure
    case 'MSE'
        measurementLabel = 'MSE';
        ygranularity = 1000;
        ymargin = 0.01;
        ylimit = [floor(min(y)*ygranularity)/ygranularity ceil(max(y)*ygranularity)/ygranularity] .* [1-ymargin 1+ymargin];
    case 'PCorrect'   
        measurementLabel = 'Accuracy (%)';
        ygranularity = 100;
        ymargin = 0.05;
        ylimit = [floor(min(y)*ygranularity)/ygranularity ceil(max(y)*ygranularity)/ygranularity] .* [1-ymargin 1+ymargin];
end
xgranularity = 1;
xmargin = 0.05;
xlimit = [floor(min(x)*xgranularity)/xgranularity ceil(max(x)*xgranularity)/xgranularity] .* [1-xmargin 1+xmargin];

set(gca, 'FontSize', fontSize_gca-2, 'FontName', fontName);
xlim(xlimit);
ylim(ylimit);

ylabel(['Multitasking ' measurementLabel],'FontSize', fontSize_gca, 'FontName', fontName, 'Color', 'k');
xlabel(['Iterations Required To Train'],'FontSize', fontSize_gca, 'Color', 'k');
% title(['Feature Overlap = ' num2str(taskSimilarity*100) '%'],'FontSize', fontSize_title);

hcb = colorbar;
set(hcb,'Color','k');
if(isfield(batch_log{featureOverlapIdx}(rep), 'hiddenTaskRepWeights'))
    ylabel(hcb, 'Initial Task Correlation','FontSize', fontSize_gca-2, 'Color', 'k');
    caxis([0 1]);
else
    ylabel(hcb, 'Max. Initial Cosine Similairty','FontSize', fontSize_gca-2, 'Color', 'k');
    caxis([-1 1]);
end
set(gcf, 'Color', 'k');
set(gcf, 'Position', [500 500 220 200]); % [500 500 540 200]

hold off;

% added for black figures

set(gcf, 'Color', 'w');
set(gca, 'Color', 'w');
set(gca, 'xColor', 'k');
set(gca, 'yColor', 'k');

% preserve background color when saving figure
fig = gcf;
fig.InvertHardcopy = 'off';


%% Fig 5b: Multitasking Performance vs. Learning Speed vs. Initial Weight Condition (across environments)

cap = 2;
plotSEM = 1;

% plot data
figure(1)

switch performanceMeasure
    case 'MSE'
        scale = 1;
    case 'PCorrect'
        scale = 100;
end

x_all = [];
x_all_sem = [];
y_all = [];
y_all_sem = [];

for featureOverlapIdx = 1:length(testedFeatureOverlaps)

    x = nanmean(learningPerformance{featureOverlapIdx}(:,:),1);
    x_sem = nanstd(learningPerformance{featureOverlapIdx}, [], 1)/sqrt(size(learningPerformance{featureOverlapIdx},1));
    y = nanmean(multiPerformance{featureOverlapIdx}(:,:,cap),1)*scale;
    y_sem = nanstd(multiPerformance{featureOverlapIdx}(:,:,cap), [], 1)/sqrt(size(multiPerformance{featureOverlapIdx}(:,:,cap),1))*scale;
    z = taskSimilarities{featureOverlapIdx};
    
    x_all = [x_all x];
    x_all_sem = [x_all_sem x_sem];
    y_all = [y_all y];
    y_all_sem = [y_all_sem y_sem];

    % create scatter plot
    if(plotSEM)
        e_v = errorbar(x, y, y_sem, '.', 'Color', 'k'); hold on;
        e_v.LineStyle = 'none';
        e_v.LineWidth = lineWidth-2;
        e_h = herrorbar(x, y, x_sem, x_sem, '.'); hold on;
        e_h(1).LineWidth = lineWidth-2;
        e_h(1).Color = [0 0 0];
    end
    scatter(x, y, markerSize, z, 'filled', 'MarkerEdgeColor',[0 0 0]);
    hold on;

end
hold off;

% compute axis parameters
switch performanceMeasure
    case 'MSE'
        measurementLabel = 'MSE';
        ygranularity = 1000;
        ymargin = 0.01;
        ylimit = [floor(min(y_all)*ygranularity)/ygranularity ceil(max(y_all)*ygranularity)/ygranularity] .* [1-ymargin 1+ymargin];
    case 'PCorrect'   
        measurementLabel = 'Accuracy (%)';
        ygranularity = 100;
        ymargin = 0.1;
        ylimit = [floor(min(y_all)*ygranularity)/ygranularity ceil(max(y_all)*ygranularity)/ygranularity] .* [1-ymargin 1+ymargin];
end
xgranularity = 1;
xmargin = 0.05;
xlimit = [floor(min(x_all)*xgranularity)/xgranularity ceil(max(x_all)*xgranularity)/xgranularity] .* [1-xmargin 1+xmargin];

set(gca, 'FontSize', fontSize_gca-2, 'FontName', fontName);
xlim(xlimit);
ylim(ylimit);

ylabel(['Multitasking ' measurementLabel],'FontSize', fontSize_gca, 'FontName', fontName, 'Color', 'k');
xlabel(['Iterations Required To Train'],'FontSize', fontSize_gca, 'Color', 'k');
title(['Feature Overlap = ' num2str(taskSimilarity*100) '%'],'FontSize', fontSize_title);

hcb = colorbar;
set(hcb,'Color','k');
ylabel(hcb, 'Initial Task Similarity','FontSize', fontSize_gca-2, 'Color', 'k');
caxis([0 1]);
set(gcf, 'Color', 'k');
set(gcf, 'Position', [500 500 400 200]); % [500 500 540 200]

hold off;

% added for white figures

set(gcf, 'Color', 'w');
set(gca, 'Color', 'w');
set(gca, 'xColor', 'k');
set(gca, 'yColor', 'k');

% preserve background color when saving figure
fig = gcf;
fig.InvertHardcopy = 'off';


