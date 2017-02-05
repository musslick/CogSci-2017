function [optimalTaskVector_meanAccuracy taskCardinality_meanAccuracy optimalPerformance_meanAccuracy optimalTaskVector_responseProbability taskCardinality_responseProbability optimalPerformance_responseProbability] = findOptimalControlPolicy(taskNet, NPathways, multiCap, goodPerformanceThresh)

    optimalPerformance_meanAccuracy = [];   % stores optimal performance (using mean accuracy as performance criterion)
    optimalTaskVector_meanAccuracy = [];     % stores optimal contrl policy (using mean accuracy as performance criterion)
    taskCardinality_meanAccuracy = [];           % stores optimal number of parallel tasks (using mean accuracy as performance criterion)

    optimalPerformance_responseProbability = [];   % stores optimal performance (using response probability as performance criterion)
    optimalTaskVector_responseProbability = [];     % stores optimal contrl policy (using response probability as performance criterion)
    taskCardinality_responseProbability = [];           % stores optimal number of parallel tasks (using response probability as performance criterion)

    for cap = 1:length(multiCap)
        
        % find all available tasks
        allTaskCombs = unique(multiCap{cap}.tasks, 'rows');
        
        for combIdx = 1:size(allTaskCombs, 1);
        
            currentTaskComb = allTaskCombs(combIdx,:);
            
            % check task cardinality
            if(cap ~= sum(currentTaskComb))
                warning(['Number of tasks does not match assigned cardinality. Cardinality is ' num2str(cap) ' and number of tasks is ' num2str(sum(currenTasks))]);
            end

            % find corresponding test patterns
            patternIdx = find(ismember(multiCap{cap}.tasks, currentTaskComb, 'rows'));
            input = multiCap{cap}.input(patternIdx, :);
            tasks = multiCap{cap}.tasks(patternIdx, :);
            train = multiCap{cap}.train(patternIdx, :);

            % compute output for all task patterns of a task
            [outputPatterns] = taskNet.runSet(input, tasks, train);

            %% COMPUTE OPTIMIZATION CRITERION

            % identify relevant output dimension
            taskM = reshape(currentTaskComb, NPathways, NPathways);
            [relevantOutputDims relevantInputDim] = find(taskM == 1);

            % test if two tasks afford a response at the same output dimension
            if(length(unique(relevantOutputDims)) ~= length(relevantOutputDims))
                warning('Tested multitasking pair affords response at the same output dimension.');
            end
            
            %% mean task accuracy as performance criterion
            
            % number of features per output dimension
            NFeatures = size(train,2)/NPathways;

            % generate mask to extract relevant output dimension
            relevantOutputMask = zeros(NFeatures, NPathways);
            relevantOutputMask(:, relevantOutputDims) = repmat(relevantOutputDims', NFeatures, 1);
            relevantOutputMask = relevantOutputMask(:)';

            summedTaskAccuracy_meanAccuracy = 0;
            
            meanTaskError = mean(abs(outputPatterns -  train), 1);
             
            for i = 1:length(relevantOutputDims)
                
                dim = relevantOutputDims(i);
                taskAccuracy = 1 - mean(meanTaskError(relevantOutputMask == dim));
                if(taskAccuracy < goodPerformanceThresh)
                    taskAccuracy = - 1;
                end
                summedTaskAccuracy_meanAccuracy = summedTaskAccuracy_meanAccuracy + taskAccuracy;

            end
            
            % define optimization criterion
            % If capacity is higher than previous capcity and performance
            % criterion is met, then select new control policy. If no optimal 
            % performance assigned, then use current control policy as starting point
            if(isempty(optimalPerformance_meanAccuracy) || (summedTaskAccuracy_meanAccuracy > optimalPerformance_meanAccuracy) )

                optimalPerformance_meanAccuracy = summedTaskAccuracy_meanAccuracy;
                optimalTaskVector_meanAccuracy = currentTaskComb;
                taskCardinality_meanAccuracy = cap;

            end
             
             %% response probability as performance criterion
            
           summedTaskAccuracy_responseProbability = 0;
                         
            [~, ~, PCorrect_tasks] = taskNet.calculateOutcomeProbabilitiesTasks(outputPatterns, train, tasks);
            PCorrect_tasks = mean(PCorrect_tasks,1);
            
            taskIDs = find(currentTaskComb == 1);
            
            for i = 1:length(taskIDs);
                
                taskAccuracy = PCorrect_tasks(taskIDs(i));
                
                if(taskAccuracy < goodPerformanceThresh)
                    taskAccuracy = - 1;
                end
                summedTaskAccuracy_responseProbability = summedTaskAccuracy_responseProbability + taskAccuracy;

            end

            % define optimization criterion
            % If capacity is higher than previous capcity and performance
            % criterion is met, then select new control policy. If no optimal 
            % performance assigned, then use current control policy as starting point
            if(isempty(optimalPerformance_responseProbability) || (summedTaskAccuracy_responseProbability > optimalPerformance_responseProbability) )

                optimalPerformance_responseProbability = summedTaskAccuracy_responseProbability;
                optimalTaskVector_responseProbability = currentTaskComb;
                taskCardinality_responseProbability = cap;

            end

            %%
        
        end
        
    end

end