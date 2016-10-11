clear all
clc
clf

nRandomPoints = 1000;
nKohonenPoints = 100;

nOrderIts = 1e3;
nConvIts = 5e4;

initNbhWidth = 100; %100 but should be closer to 1 according to lecture notes
initLearnRate = 0.1;
tau = 200; %200

convNbhWidth = 0.9;
convLearnRate = 0.01;

% Generate point cloud in triangle shape

leftLine = @(x) sqrt(3)*x;
rightLine = @(x) sqrt(3)*(1-x);

randomPoints = zeros(nRandomPoints,2);

for i = 1:nRandomPoints
    while randomPoints(i,2) > leftLine(randomPoints(i,1)) || randomPoints(i,2) > rightLine(randomPoints(i,1)) || isequal(randomPoints(i,:),[0,0])
        randomPoints(i,:) = rand(1,2);    % Generate new random coords
    end                             % until we get a pt below the two lines
end

% % Plot points to see that they are what we want
% scatter(randomPoints(:,1), randomPoints(:,2))
% axis equal

% Initialize Kohonen network to vertical line at x = 0.5
%kohonenPoints = [linspace(0.5,0.5,nKohonenPoints)',linspace(0,sqrt(3/4),nKohonenPoints)'];
% Let's try just random positions instead - seems to work well
kohonenPoints = rand(nKohonenPoints,2);
kohonenPoints(:,2) = kohonenPoints(:,2) * sqrt(3/4);
kohonenPoints = randomPoints(1:nKohonenPoints,:);
kohonenPointWeights = zeros(nKohonenPoints,1); % For debugging purposes

% % Check initalization
% hold on
% kohonenColors = zeros(nKohonenPoints,3);
% kohonenColors(:,1) = linspace(0,1,nKohonenPoints)';
% kohonenColors(:,3) = linspace(1,0,nKohonenPoints)';
% scatter(kohonenPoints(:,1),kohonenPoints(:,2),10,kohonenColors)

% Set functions for neighbourhood width and learning rate
nbhWidthFunc = @(t) initNbhWidth * exp(-t/tau);
learnRateFunc = @(t) initLearnRate * exp(-t/tau);

% Start ordering iterations
for iOrderIt = 1:nOrderIts
    % Select a random point from our sample
    iRandomPoint = floor(rand()*nRandomPoints) + 1;
    selectedPoint = randomPoints(iRandomPoint,:);
    
    % Find the closest Kohonen point
    smallestNorm = Inf;
    for iKohonenPoint = 1:nKohonenPoints
        currentNorm = norm(kohonenPoints(iKohonenPoint,:) - randomPoints(iRandomPoint,:));
        if currentNorm < smallestNorm
            smallestNorm = currentNorm;
            iSmallestNorm = iKohonenPoint;
        end
    end
    winningPos = kohonenPoints(iSmallestNorm,:);
        
    % Move the Kohonen points
    for iKohonenPoint = 1:nKohonenPoints
        currentPos = kohonenPoints(iKohonenPoint,:);
        
        nbhWidth = nbhWidthFunc(iOrderIt);
        nbhFunction = exp(-(norm(winningPos - currentPos))^2 / (2*nbhWidth^2)); % the way it should be according to the paper
        nbhFunction = exp(-(500*norm(winningPos - currentPos))^2 / (2*nbhWidth^2));
        %nbhFunction = exp(-(iSmallestNorm-iKohonenPoint)^2 / (2*nbhWidth^2));
        
        kohonenPointWeights(iKohonenPoint) = nbhFunction;
        
        learnRate = learnRateFunc(iOrderIt);
        
        newPos = currentPos + learnRate*nbhFunction*(selectedPoint-currentPos);
        
        kohonenPoints(iKohonenPoint,:) = newPos;
    end
    

    % Plotting to see progress, disable after debugging
    scatter(randomPoints(:,1), randomPoints(:,2),1)
    axis equal
    hold on
    plot(randomPoints(iRandomPoint,1),randomPoints(iRandomPoint,2),'or')
    kohonenColors = zeros(nKohonenPoints,3);
%     kohonenColors(:,1) = linspace(0,1,nKohonenPoints)';
%     kohonenColors(:,3) = linspace(1,0,nKohonenPoints)';
    kohonenColors(:,1) = kohonenPointWeights / max(kohonenPointWeights);    % debugging colors
    kohonenColors(:,3) = 1 - kohonenPointWeights / max(kohonenPointWeights);
    scatter(kohonenPoints(:,1),kohonenPoints(:,2),10,kohonenColors)
    text(0,0.2,num2str(iOrderIt));
    hold off
    pause(0.0000001)
    %pause

    
end     % End of ordering phase


% Start convergence iterations
for iConvIt = 1:nConvIts
    % Select a random point from our sample
    iRandomPoint = floor(rand()*nRandomPoints) + 1;
    selectedPoint = randomPoints(iRandomPoint,:);
    
    % Find the closest Kohonen point
    smallestNorm = Inf;
    for iKohonenPoint = 1:nKohonenPoints
        currentNorm = norm(kohonenPoints(iKohonenPoint,:) - randomPoints(iRandomPoint,:));
        if currentNorm < smallestNorm
            smallestNorm = currentNorm;
            iSmallestNorm = iKohonenPoint;
        end
    end
    winningPos = kohonenPoints(iSmallestNorm,:);
        
    % Move the Kohonen points
    for iKohonenPoint = 1:nKohonenPoints
        currentPos = kohonenPoints(iKohonenPoint,:);
        
        nbhFunction = exp(-(500*norm(winningPos - currentPos))^2 / (2*convNbhWidth^2));
        %nbhFunction = exp(-(iSmallestNorm-iKohonenPoint)^2 / (2*convNbhWidth^2));
        
        %disp([nbhFunction, -(norm(selectedPoint - currentPos))^2])
        
        newPos = currentPos + convLearnRate*nbhFunction*(selectedPoint-currentPos);
        
        kohonenPoints(iKohonenPoint,:) = newPos;
    end
    

    % Plotting to see progress, disable after debugging
    scatter(randomPoints(:,1), randomPoints(:,2),1)
    axis equal
    hold on
    plot(randomPoints(iRandomPoint,1),randomPoints(iRandomPoint,2),'or')
    kohonenColors = zeros(nKohonenPoints,3);
    kohonenColors(:,1) = linspace(0,1,nKohonenPoints)';
    kohonenColors(:,3) = linspace(1,0,nKohonenPoints)';
    scatter(kohonenPoints(:,1),kohonenPoints(:,2),10,kohonenColors)
    text(0,0.2,num2str(iConvIt));
    hold off
    pause(0.0000001)
    %pause

    
end     % End of convergence phase



% Plot after final iteration
scatter(randomPoints(:,1), randomPoints(:,2))
axis equal

hold on
kohonenColors = zeros(nKohonenPoints,3);
kohonenColors(:,1) = linspace(0,1,nKohonenPoints)';
kohonenColors(:,3) = linspace(1,0,nKohonenPoints)';
scatter(kohonenPoints(:,1),kohonenPoints(:,2),10,kohonenColors)
