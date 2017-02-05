function Simulation1(rep)
%% description
% 
% Simulation1(rep)
%
% INPUT PARAMETERS
% rep                  ...Replication (instance) of simulation
% 
% DESCRIPTION
% The simulation involves training a neural networks on a set of tasks in
% environments with different task similarities. Each task requires to map
% a particular set of 5 stimulus features to a particular set of 5 response 
% units. Task similarity between a pair of two tasks is operationalized in 
% terms of the number of shared relevant stimulus features in the
% environment. At each training iteration the learning performance, as
% well as multitasking performance are assessed.
%
% author: Sebastian Musslick

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
iterations_Train = 1000;     % max. number of training iterations

% task environment parameters 
NPathways = 3;                     % number of pathways (i.e. number of feature dimensions % output dimensions)
NFeatures = 5;                     % the number of features per feature dimension
nTasks = NPathways^2;

sameClassifierAcrossTasks = 1;
sameStimuliAcrossTasks = 1;
taskSimilarities = [0 0.2 0.4 0.6 0.8 1];

samplesPerTask = 50; 
samplesPerTask_gen = 100;

batch_log = repmat(struct('MSE_log',nan(length(taskSimilarities), iterations_Train), ...
                          'train_MSE',nan(length(taskSimilarities), samplesPerTask*nTasks), ...  
                          'train_Accuracy',nan(length(taskSimilarities), samplesPerTask*nTasks), ... 
                          'train_Pcorrect',nan(length(taskSimilarities), samplesPerTask*nTasks), ... 
                          'test_MSE',nan(length(taskSimilarities), samplesPerTask_gen*nTasks), ...  
                          'test_Pcorrect',nan(length(taskSimilarities), samplesPerTask*nTasks), ... 
                          'hiddenTaskRepWeights',nan(length(taskSimilarities), iterations_Train, nTasks, nTasks), ...
                          'outputTaskRepWeights',nan(length(taskSimilarities), iterations_Train, nTasks, nTasks)),1, 1); 

tic

for taskSimilarity_idx = 1:length(taskSimilarities)

    taskSimilarity = taskSimilarities(taskSimilarity_idx);
    batch_log(1).featureOverlap(taskSimilarity_idx) = taskSimilarity;
    
    % create trainign environment
    [inputSgl_cut, tasksSgl_cut, trainSgl_cut, ~, multiCap, ~, classificationFunction] = createTaskPatterns_GenN(NPathways, NFeatures, samplesPerTask, taskSimilarity, sameClassifierAcrossTasks, sameStimuliAcrossTasks, [], length(nHidden));
    [inputSgl_gen, tasksSgl_gen, trainSgl_gen] = createTaskPatterns_GenN(NPathways, NFeatures, samplesPerTask_gen, taskSimilarity, sameClassifierAcrossTasks, sameStimuliAcrossTasks, classificationFunction);
    [multiCap_con, multiCap_inc] = splitTrainingPatternsByCongruency(multiCap, NFeatures, NPathways);

    % initialize network
    taskNet = NNmodel(nHidden, learningRate, bias, init_scale, thresh, decay, hiddenPathSize, outputPathSize);
    taskNet.setData(inputSgl_cut, tasksSgl_cut, trainSgl_cut);
    taskNet.configure(); 
    taskNet.NPathways = NPathways;

    %% training loop
    for iter = 1:iterations_Train

        % train network for one iteration
        taskNet.trainOnline(1);
        
        if(taskNet.MSE_log < thresh)
            break
        end

        % record error
        batch_log.MSE_log(taskSimilarity_idx,iter) = taskNet.MSE_log;
        batch_log.CE_log(taskSimilarity_idx,iter) = taskNet.CE_log;
        batch_log.CF_log(taskSimilarity_idx,iter) = taskNet.CF_log;
        batch_log.DimCF_log(taskSimilarity_idx,iter) = taskNet.DimCF_log;

        % run network on traning and test set
        [outData, ~, MSE_train]  = taskNet.runSet(inputSgl_cut, tasksSgl_cut, trainSgl_cut);
         [~, PCorrect_mean_train] = taskNet.calculateOutcomeProbabilitiesTasks(outData,  trainSgl_cut, tasksSgl_cut);

         [outData, ~, MSE_test] = taskNet.runSet(inputSgl_gen, tasksSgl_gen, trainSgl_gen);
         [~, PCorrect_mean_test] = taskNet.calculateOutcomeProbabilitiesTasks(outData, trainSgl_gen, tasksSgl_gen);

        % store training performance
        batch_log.train_MSE(taskSimilarity_idx, iter) = mean(MSE_train);
        batch_log.train_PCorrect(taskSimilarity_idx, iter) = mean(PCorrect_mean_train);

        % store testing performance
        batch_log.test_MSE(taskSimilarity_idx, iter) = mean(MSE_test);
        batch_log.test_PCorrect(taskSimilarity_idx, iter) = mean(PCorrect_mean_test);
        
        % get task representations
        batch_log.hiddenTaskRepWeights(taskSimilarity_idx, iter, :,:) = corr(taskNet.weights.W_TH);
        batch_log.outputTaskRepWeights(taskSimilarity_idx, iter, :,:) = corr(taskNet.weights.W_TO);

        %% test multitasking performance

        for cap = 2:length(multiCap) 

            % incongruent

            [outData, ~, MSE_multi] = taskNet.runSet(multiCap_inc{cap}.input, multiCap_inc{cap}.tasks, multiCap_inc{cap}.train);
            [~, PCorrect_mean_train] = taskNet.calculateOutcomeProbabilitiesTasks(outData,  multiCap_inc{cap}.train, multiCap_inc{cap}.tasks);

            batch_log.MSE_multi_inc{taskSimilarity_idx, iter, cap} = mean(MSE_multi);
            batch_log.PCorrect_multi_inc{taskSimilarity_idx, iter, cap} = mean(PCorrect_mean_train);

            % congruent

            [outData, ~, MSE_multi] = taskNet.runSet(multiCap_con{cap}.input, multiCap_con{cap}.tasks, multiCap_con{cap}.train);
            [~, PCorrect_mean_train] = taskNet.calculateOutcomeProbabilitiesTasks(outData,  multiCap_con{cap}.train, multiCap_con{cap}.tasks);

            batch_log.MSE_multi_con{taskSimilarity_idx, iter, cap} = mean(MSE_multi);
            batch_log.PCorrect_multi_con{taskSimilarity_idx, iter, cap} = mean(PCorrect_mean_train);

        end

        % print current training iteration
        disp([iter taskNet.MSE_log]);
    end

    %% print progress
    progress = taskSimilarity_idx/length(taskSimilarities);
    disp([ 'feature overlap ' num2str(taskSimilarity_idx) '/' num2str(length(taskSimilarities))]);
    disp(['progress: ' num2str(progress*100) '%']);
    disp('---');
end
toc




save(['logfiles/Simulation1_' num2str(NPathways) 'P' num2str(NFeatures) 'F_v' num2str(log_version) '_h' num2str(nHidden(1))  '_r' num2str(rep)], '-v7.3');

end