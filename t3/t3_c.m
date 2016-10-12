clear all
clc
clf

% Import data
points = dlmread('task3.txt',' ');

nPoints = length(points);
nTrainingPoints = ceil(0.7*nPoints);
nValidationPoints = nPoints - nTrainingPoints;

% Set various parameters
nBoundaryPoints = 5e2;  % Gives resolution for decision boundary plot

nIterations = 1e5;      % Iterations to move the weights around
nTrainingIterations = 3e3;  % Iterations to train the classification
nMainIterations = 20;   % How many tries to get the best classification error
nodeNumberList= 1:20;          % List of all numbers of nodes to check

finalClassificationErrors = zeros(size(nodeNumberList));

activationBeta = 0.5;
learnRate = 0.02;

% Loop starts here
for nNodes = nodeNumberList
    nodes = rand(nNodes,2)*2-1;
    outputs = zeros(nNodes,1);

    for iIteration = 1:nIterations
        % Select a random point
        iSelectedPoint = ceil(rand()*nPoints);
        selectedPos = points(iSelectedPoint,2:3);

        % Calculate activation of nodes by this point
        denominator = 0;
        for iNode = 1:nNodes
            outputs(iNode) = exp(-norm(selectedPos - nodes(iNode,:))^2/2);
            denominator = denominator + outputs(iNode);
        end
        outputs = outputs / denominator;

        % Update the winning node
        iWinningNode = find(outputs == max(outputs));
        nodes(iWinningNode,:) = nodes(iWinningNode,:) + learnRate*(selectedPos-nodes(iWinningNode,:));
    end

    bestClassificationError = 2;    % Just to make sure the first calculated value gets stored

    for iMainIteration = 1:nMainIterations
        % Initalize to random values
        weights = rand(nNodes,1)*2 - 1;
        threshold = rand()*2 - 1;

        % Get a random division of data to training set and validation set
        trainingSelection = sort(randsample(nPoints,nTrainingPoints));
        trainingSet = points(trainingSelection,:);
        validationSet = points(~ismember(1:nPoints, trainingSelection),:);

        % Train the network
        for iTrainIt = 1:nTrainingIterations
            % Get a random point
            iRandomPoint = ceil(rand()*nTrainingPoints);
            selectedPos = points(iRandomPoint,2:3);

            % Find activation
            denominator = 0;
            for iNode = 1:nNodes
                outputs(iNode) = exp(-norm(selectedPos - nodes(iNode,:))^2/2);
                denominator = denominator + outputs(iNode);
            end
            outputs = outputs / denominator;

            activation = tanh(activationBeta*weights'*outputs + threshold);

            % Calculate updates
            thresholdUpdate = activationBeta*(points(iRandomPoint,1) - activation)*(1-tanh(activationBeta*weights'*outputs)^2);
            weightUpdate = thresholdUpdate * outputs;

            threshold = threshold + thresholdUpdate;
            weights = weights + weightUpdate;
        end

        % Run through validation set and see how we're doing
        classificationError = 0;
        for iValidIt = 1:nValidationPoints
            iRandomPoint = ceil(rand()*nValidationPoints);
            selectedPos = validationSet(iRandomPoint,2:3);

            denominator = 0;
            for iNode = 1:nNodes
                outputs(iNode) = exp(-norm(selectedPos - nodes(iNode,:))^2/2);
                denominator = denominator + outputs(iNode);
            end
            outputs = outputs / denominator;

            activation = tanh(activationBeta*weights'*outputs + threshold);

            % Accumulate correctly classified points...
            classificationError = classificationError + sign(activation)*validationSet(iRandomPoint,1);
        end
        % ... and when we're done, normalize and count the misclassified ones instead
        classificationError = (nValidationPoints - classificationError) / nValidationPoints;

        if classificationError < bestClassificationError
            bestClassificationError = classificationError;
            bestWeights = weights;
            bestThreshold = threshold;
        end
    end
    finalClassificationErrors(nNodes) = bestClassificationError;
end

% Plot and save
plot(finalClassificationErrors)
xlabel('Number of RBFs')
ylabel('Classification error')

saveas(gcf, 't3_c.png','png')