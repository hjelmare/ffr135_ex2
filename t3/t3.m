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

activationBeta = 0.5;
learnRate = 0.02;

nNodes = 5;    % Number of "weights", change between 5 and 20 for a) and b)

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

% Start preparing plotting of decision boundary

xMin = min(points(:,2));
xMax = max(points(:,2));
yMin = min(points(:,3));
yMax = max(points(:,3));

% Generate ticks that span the input data
xTics = linspace(xMin, xMax, nBoundaryPoints);
yTics = linspace(yMin, yMax, nBoundaryPoints);

boundaryActivation = zeros(nBoundaryPoints);

threshold = bestThreshold;
weights = bestWeights;

% For all ticks, find activation
for iX = 1:nBoundaryPoints
    for iY = 1:nBoundaryPoints
        selectedPos = [xTics(iX), yTics(iY)];

        denominator = 0;
        for iNode = 1:nNodes
            outputs(iNode) = exp(-norm(selectedPos - nodes(iNode,:))^2/2);
            denominator = denominator + outputs(iNode);
        end
        outputs = outputs / denominator;

        activation = tanh(activationBeta*weights'*outputs + threshold);
        boundaryActivation(iX, iY) = sign(activation);
    end
end

% Use contour plot to get the decision boundary line
contour(xTics, yTics, boundaryActivation', 'rx')
hold on
plot(points(points(:,1)==1,2), points(points(:,1)==1,3), '.b')
plot(points(points(:,1)==-1,2), points(points(:,1)==-1,3), '.k')
plot(nodes(:,1), nodes(:,2), 'r.', 'MarkerSize', 20)
hold off

filename = ['t3_' num2str(nNodes) '.png'];
saveas(gcf, filename,'png')