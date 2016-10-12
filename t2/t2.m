clear all
close all
clc

% Set some basic parameters
nOrderIterations = 1e3;
nConvergenceIterations = 2e4;

initNbhWidth = 30;
initLearnRate = 0.1;
tau = 300;

convNbhWidth = 0.9;
convLearnRate = 0.01;

% Import data
tmpData = dlmread('wine.data.txt',',');

for iColumn = 2:size(tmpData,2)
    tmpData(:,iColumn) = tmpData(:,iColumn) - mean(tmpData(:,iColumn));
    tmpData(:,iColumn) = tmpData(:,iColumn) / std(tmpData(:,iColumn));
end

inputData = tmpData(:,2:14);

nInputPoints = length(inputData);

% Set functions for neighbourhood width and learning rate
nbhWidthFunc = @(t) initNbhWidth * exp(-t/tau);
learnRateFunc = @(t) initLearnRate * exp(-t/tau);

% Initialise kohonen nodes
nKohonenPointsX = 20;
nKohonenPointsY = 20;
nKohonenPoints = nKohonenPointsX*nKohonenPointsY;
inputCoords = rand(nKohonenPoints,13);  % Positions in 13D input space
outputCoords = zeros(nKohonenPoints,2);  % Positions in 2D output space

for iX = 1:nKohonenPointsX
    for iY = 1:nKohonenPointsY
        linearIndex = (iX-1)*nKohonenPointsX + iY;
        outputCoords(linearIndex,:) = [iX, iY];
    end
end

% Start ordering phase
for iOrderIt = 1:nOrderIterations
    iRandom = ceil(rand()*nInputPoints);
    selectedPoint = inputData(iRandom,:);
    
    smallestNorm = inf;
    for iKohonenPoint = 1:nKohonenPoints
        currentNorm = norm(selectedPoint - inputCoords(iKohonenPoint,:));
        if currentNorm < smallestNorm
            smallestNorm = currentNorm;
            iWinningPoint = iKohonenPoint;
        end
    end
    winningPoint = inputCoords(iWinningPoint,:);
    
    for iKohonenPoint = 1:nKohonenPoints
        currentPos = inputCoords(iKohonenPoint,:);
        
        nbhWidth = nbhWidthFunc(iOrderIt);
        nbhFunction = exp(-(norm(outputCoords(iWinningPoint,:)-outputCoords(iKohonenPoint,:)))^2 / (2*nbhWidth^2));
        
        learnRate = learnRateFunc(iOrderIt);
        
        newPos = currentPos + learnRate*nbhFunction*(selectedPoint-currentPos);
        
        inputCoords(iKohonenPoint,:) = newPos;
    end
end

% Start convergence phase
for iConvIt = 1:nConvergenceIterations
    iRandom = ceil(rand()*nInputPoints);
    selectedPoint = inputData(iRandom,:);
    
    smallestNorm = inf;
    for iKohonenPoint = 1:nKohonenPoints
        currentNorm = norm(selectedPoint - inputCoords(iKohonenPoint,:));
        if currentNorm < smallestNorm
            smallestNorm = currentNorm;
            iWinningPoint = iKohonenPoint;
        end
    end
    winningPoint = inputCoords(iWinningPoint,:);
    
    for iKohonenPoint = 1:nKohonenPoints
        currentPos = inputCoords(iKohonenPoint,:);
        
        nbhFunction = exp(-(norm(outputCoords(iWinningPoint,:)-outputCoords(iKohonenPoint,:)))^2 / (2*convNbhWidth^2));
        
        newPos = currentPos + convLearnRate*nbhFunction*(selectedPoint-currentPos);
        
        inputCoords(iKohonenPoint,:) = newPos;
    end
end


% % % Second phase the more intuitive way
% % 
% % outputColors = zeros(nKohonenPoints,1);
% % 
% % for iKohonenPoint = 1:nKohonenPoints
% %     smallestNorm = inf;
% %     for iWine = 1:length(inputData)
% %         currentNorm = norm(inputData(iWine,:) - inputCoords(iKohonenPoint,:));
% %         if currentNorm < smallestNorm
% %             smallestNorm = currentNorm;
% %             iWinningWine = iWine;
% %         end
% %     end
% %         
% %     outputColors(iKohonenPoint) = tmpData(iWinningWine,1);    
% % end

% Second phase - the way it is described
    
outputColors = zeros(nKohonenPoints,1);

for iWine = 1:length(inputData)
    selectedPoint = inputData(iWine,:);
    
    smallestNorm = inf;
    for iKohonenPoint = 1:nKohonenPoints
        currentNorm = norm(selectedPoint - inputCoords(iKohonenPoint,:));
        if currentNorm < smallestNorm
            smallestNorm = currentNorm;
            iWinningPoint = iKohonenPoint;
        end
    end
        
    outputColors(iWinningPoint) = tmpData(iWine,1);
end

clf
clc

for iPlot = 1:400;
    switch outputColors(iPlot)
        case 1
            plot(outputCoords(iPlot,1),outputCoords(iPlot,2),'ro');
        case 2
            plot(outputCoords(iPlot,1),outputCoords(iPlot,2),'go');
        case 3
            plot(outputCoords(iPlot,1),outputCoords(iPlot,2),'bo');
    end    
    hold on
end
axis([0 21 0 21])
hold off

saveas(gcf,'t2.png','png')