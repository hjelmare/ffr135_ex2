clear all
clc
clf

points = dlmread('task3.txt',' ');

nNodes = 5;
nPoints = length(points);
nTrainingPoints = ceil(0.7*nPoints);
nValidationPoints = nPoints - nTrainingPoints;

nBoundaryPoints = 1e5;

nMainIterations = 20;
nIterations = 1e5;
nTrainingIterations = 3e3;

activationBeta = 0.5;

learnRate = 0.02;

nodes = rand(nNodes,2)*2-1;

outputs = zeros(nNodes,1);

for iIteration = 1:nIterations
    iSelectedPoint = ceil(rand()*nPoints);
    selectedPos = points(iSelectedPoint,2:3);
    
    denominator = 0;
    for iNode = 1:nNodes
        outputs(iNode) = exp(-norm(selectedPos - nodes(iNode,:))^2/2);
        denominator = denominator + outputs(iNode);
    end
    outputs = outputs / denominator;
    
    iWinningNode = find(outputs == max(outputs));
    nodes(iWinningNode,:) = nodes(iWinningNode,:) + learnRate*(selectedPos-nodes(iWinningNode,:));
    
%     plot(points(:,2),points(:,3),'ob')
%     hold on
%     
%     plot(points(iSelectedPoint,2),points(iSelectedPoint,3),'ok')
%     
%     plot(nodes(:,1),nodes(:,2),'or')
%     
%     text(-12,10,num2str(iIteration))
% 
%     hold off
% 	pause(0.000001)
%     
end

bestClassificationError = 2;

for iMainIteration = 1:nMainIterations

    weights = rand(nNodes,1)*2 - 1;
    threshold = rand()*2 - 1;

    trainingSelection = sort(randsample(nPoints,nTrainingPoints));
    trainingSet = points(trainingSelection,:);
    validationSet = points(~ismember(1:nPoints, trainingSelection),:);

    for iTrainIt = 1:nTrainingIterations
        iRandomPoint = ceil(rand()*nTrainingPoints);
        selectedPos = points(iRandomPoint,2:3);

        denominator = 0;
        for iNode = 1:nNodes
            outputs(iNode) = exp(-norm(selectedPos - nodes(iNode,:))^2/2);
            denominator = denominator + outputs(iNode);
        end
        outputs = outputs / denominator;

        activation = tanh(activationBeta*weights'*outputs + threshold);

        thresholdUpdate = activationBeta*(points(iRandomPoint,1) - activation)*(1-tanh(activationBeta*weights'*outputs)^2);
        weightUpdate = thresholdUpdate * outputs;

        threshold = threshold + thresholdUpdate;
        weights = weights + weightUpdate;

    %     plot(points(:,2),points(:,3),'.b')
    %     hold on
    %     plot(nodes(:,1),nodes(:,2),'ob')
    %     plot(selectedPos(1),selectedPos(2),'or')
    %     text(-12, 12, num2str(iTrainIt))
    %     text(-12,10,[num2str(activation) '  ' num2str(points(iRandomPoint,1))])
    %     hold off
    %     pause

    end

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

        classificationError = classificationError + sign(activation)*validationSet(iRandomPoint,1);

    end

    classificationError = (nValidationPoints - classificationError) / nValidationPoints;
    
    if classificationError < bestClassificationError
        bestClassificationError = classificationError;
        bestWeights = weights;
        bestThreshold = threshold;
    end
end

xMin = min(points(:,2));
xMax = max(points(:,2));
yMin = min(points(:,3));
yMax = max(points(:,3));

randomPoints = rand(nBoundaryPoints,2);
randomPoints(:,1) = xMin + randomPoints(:,1)*(xMax-xMin);
randomPoints(:,2) = yMin + randomPoints(:,2)*(yMax-yMin);

boundaryActivation = zeros(nBoundaryPoints,1);

threshold = bestThreshold;
weights = bestWeights;

for iBoundaryPoint = 1:nBoundaryPoints
    selectedPos = randomPoints(iBoundaryPoint,:);

    denominator = 0;
    for iNode = 1:nNodes
        outputs(iNode) = exp(-norm(selectedPos - nodes(iNode,:))^2/2);
        denominator = denominator + outputs(iNode);
    end
    outputs = outputs / denominator;

    activation = tanh(activationBeta*weights'*outputs + threshold);
    boundaryActivation(iBoundaryPoint) = sign(activation);
end

plot(randomPoints(boundaryActivation>0,1),randomPoints(boundaryActivation>0,2),'or')
hold on
plot(randomPoints(boundaryActivation<0,1),randomPoints(boundaryActivation<0,2),'ob')
plot(points(:,2),points(:,3),'og')
hold off