function Simulation2(taskSimilarity, rep)
%% description
%
% Simulation2(taskSimilarity, rep)
%
% INPUT PARAMETERS
% rep                  ...Replication (instance) of simulation
% taskSimilarity    ...Amount of shared stimulus features between tasks of differnet response dimensions ("feature overlap"), can be between 0 and 1 in steps of 0.2
% 
% DESCRIPTION
% The simulation involves training a neural networks on a set of tasks with
% different initial weight conditions. Each initial weight condition
% determines the initial similarity between tasks that rely on the same
% input dimension. At each training iteration the learning performance, as
% well as multitasking performance are assessed.
%
% author: Sebastian Musslick

addpath('main code');

% meta simulation parameters
log_version = 7;

% set up network parameters
nHidden = 100;              % number of hidden units
hiddenPathSize = 1;         % group size of hidden units that receive the same weights from the task layer)
outputPathSize = 1;         % group size of output units that receive the same weights from the task layer)
learningRate = 0.3;         % learning rate
thresh = 0.01;            % mean-squared error stopping criterion
decay = 0.0000;             % weight penalization parameter
bias = -2;                  % weight from bias units to hidden & output units
init_scale = 0.1;           % scales for initialized random weights
iterations_Train = 1000;     % number of training iterations

% task environment parameters 
NPathways = 3;                     % number of pathways (i.e. number of feature dimensions % output dimensions)
NFeatures = 5;                     % the number of features per feature dimension
nTasks = NPathways^2;

sameClassifierAcrossTasks = 1;
sameStimuliAcrossTasks = 1;
init_taskCorrelations = linspace(0, 0.9750, 40);  % magnitude of initial weights
init_task_scale = 5;

samplesPerTask = 50;
samplesPerTask_gen = 100;

batch_log = repmat(struct('MSE_log',nan(length(init_taskCorrelations), iterations_Train), ...
                          'train_MSE',nan(length(init_taskCorrelations), samplesPerTask*nTasks), ...  
                          'train_Pcorrect',nan(length(init_taskCorrelations), samplesPerTask*nTasks), ... 
                          'test_MSE',nan(length(init_taskCorrelations), samplesPerTask_gen*nTasks), ...  
                          'test_Pcorrect',nan(length(init_taskCorrelations), samplesPerTask*nTasks), ... 
                          'hiddenTaskRepWeights',nan(length(init_taskCorrelations), iterations_Train, nTasks, nTasks), ...
                          'outputTaskRepWeights',nan(length(init_taskCorrelations), iterations_Train, nTasks, nTasks), ...
                          'init_taskCorr',nan(length(init_taskCorrelations), 1), ...
                          'weightCorr',nan(length(init_taskCorrelations), 1)),1, 1); 

tic
% create trainign environment
[inputSgl_cut tasksSgl_cut trainSgl_cut tasksIdxSgl_cut multiCap classLog classificationFunction] = createTaskPatterns_GenN(NPathways, NFeatures, samplesPerTask, taskSimilarity, sameClassifierAcrossTasks, sameStimuliAcrossTasks, [], length(nHidden));
[inputSgl_gen tasksSgl_gen trainSgl_gen tasksIdxSgl_gen classLog_gen] = createTaskPatterns_GenN(NPathways, NFeatures, samplesPerTask_gen, taskSimilarity, sameClassifierAcrossTasks, sameStimuliAcrossTasks, classificationFunction);
[multiCap_con, multiCap_inc] = splitTrainingPatternsByCongruency(multiCap, NFeatures, NPathways);

basis_template = eye(nTasks,nTasks);
for row = 1:size(basis_template,1)
    basis_template(row, (ceil(row/NPathways)-1)*NPathways+(1:NPathways)) = 1;
    basis_template(row, row:end) = 0;
end

for init_taskCorr_idx = 1:length(init_taskCorrelations)

    % initialize network
    taskNet = NNmodel(nHidden, learningRate, bias, init_scale, thresh, decay, hiddenPathSize, outputPathSize);
    taskNet.setData(inputSgl_cut, tasksSgl_cut, trainSgl_cut);
    taskNet.configure(); 
    taskNet.NPathways = NPathways;

    %% set up task weights according to max. cosine similarity

    rand('seed', rep);

    % get task weights as a function of initial weight vector correlation
    init_taskCorr = init_taskCorrelations(init_taskCorr_idx);
    taskWeights = generateWeightBasisSet(nHidden, nTasks, init_taskCorr);

    % scale task weights
    taskWeights = taskWeights .*init_task_scale(end);

    % set up network weights
    taskNet.weights.W_TH = taskWeights;

    % compute final weight correlation
    basis_template = eye(nTasks,nTasks);
    for row = 1:size(basis_template,1)
        basis_template(row, (ceil(row/NPathways)-1)*NPathways+(1:NPathways)) = 1;
        basis_template(row, row:end) = 0;
    end

    R = corr(taskWeights);
    weightCorr = R(basis_template == 1);

    % log initial weight correlation
    batch_log.weightCorr(init_taskCorr_idx, rep) = mean(weightCorr);
    batch_log.init_taskCorr(init_taskCorr_idx, rep) = init_taskCorr;

    %% training loop
    for iter = 1:iterations_Train

        % train network for one iteration
        taskNet.trainOnline(1);
        
        if(taskNet.MSE_log < thresh)
            break
        end

        % record error
        batch_log.MSE_log(init_taskCorr_idx,iter) = taskNet.MSE_log;
        
        % run network on traning and test set
        [outData, ~, MSE_train]  = taskNet.runSet(inputSgl_cut, tasksSgl_cut, trainSgl_cut);
        [~, PCorrect_mean_train] = taskNet.calculateOutcomeProbabilitiesTasks(outData,  trainSgl_cut, tasksSgl_cut);

        [outData, ~, MSE_test] = taskNet.runSet(inputSgl_gen, tasksSgl_gen, trainSgl_gen);
        [~, PCorrect_mean_test] = taskNet.calculateOutcomeProbabilitiesTasks(outData, trainSgl_gen, tasksSgl_gen);

        % store training performance
        batch_log.train_MSE(init_taskCorr_idx, iter) = mean(MSE_train);
        batch_log.train_PCorrect(init_taskCorr_idx, iter) = mean(PCorrect_mean_train);

        % store testing performance
        batch_log.test_MSE(init_taskCorr_idx, iter) = mean(MSE_test);
        batch_log.test_PCorrect(init_taskCorr_idx, iter) = mean(PCorrect_mean_test);

        % store task representations
        batch_log.hiddenTaskRepWeights(init_taskCorr_idx, iter, :,:) = corr(taskNet.weights.W_TH);
        batch_log.outputTaskRepWeights(init_taskCorr_idx, iter, :,:) = corr(taskNet.weights.W_TO);

        %% test multitasking performance

        for cap = 2:length(multiCap) 

            % incongruent

            [outData, ~, MSE_multi] = taskNet.runSet(multiCap_inc{cap}.input, multiCap_inc{cap}.tasks, multiCap_inc{cap}.train);
            [~, PCorrect_mean_train] = taskNet.calculateOutcomeProbabilitiesTasks(outData,  multiCap_inc{cap}.train, multiCap_inc{cap}.tasks);

            batch_log.MSE_multi_inc{init_taskCorr_idx, iter, cap} = mean(MSE_multi);
            batch_log.PCorrect_multi_inc{init_taskCorr_idx, iter, cap} = mean(PCorrect_mean_train);

            % congruent

            [outData, ~, MSE_multi] = taskNet.runSet(multiCap_con{cap}.input, multiCap_con{cap}.tasks, multiCap_con{cap}.train);
            [~, PCorrect_mean_train] = taskNet.calculateOutcomeProbabilitiesTasks(outData,  multiCap_con{cap}.train, multiCap_con{cap}.tasks);

            batch_log.MSE_multi_con{init_taskCorr_idx, iter, cap} = mean(MSE_multi);
            batch_log.PCorrect_multi_con{init_taskCorr_idx, iter, cap} = mean(PCorrect_mean_train);

        end

        % print current training iteration
        disp([iter taskNet.MSE_log]);
    end

    %% print progress
    progress = init_taskCorr_idx/length(init_taskCorrelations);
    disp([ 'init_scale ' num2str(init_taskCorr_idx) '/' num2str(length(init_taskCorrelations))]);
    disp(['progress: ' num2str(progress*100) '%']);
    disp('---');
end
toc




save(['logfiles/Simulation2_' num2str(NPathways) 'P' num2str(NFeatures) 'F_v' num2str(log_version) '_h' num2str(nHidden(1)) '_FO' num2str(taskSimilarity*100) '_r' num2str(rep)], '-v7.3');

end